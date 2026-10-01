"""The manual route: QGIS Georeferencer ``.points`` files.

When automatic georeferencing fails (or is not wanted), open the map image in QGIS
(Layer > Georeferencer), place points, save them with "Save GCP Points", and hand the file
to ``gossan ingest-points``. This module reads that file and can also write one, so the
candidate points from a failed automatic attempt can be corrected in QGIS instead of
starting over.

The QGIS format is a CSV with optional ``#`` comment lines, one of which (``#CRS: ...``)
names the CRS of the map coordinates. QGIS has changed its header names and CRS notation
between versions, so the parser does not insist on them:

* columns are found by what their names say (``mapX``, ``sourceX``, ``pixelY``,
  ``Easting``, ``source_col``, ... in any case or punctuation), not by exact spelling; if
  the header cannot be understood, by position in QGIS's order: map X, map Y, source X,
  source Y, [enable] (and the result says so);
* delimiters other than a comma are accepted;
* an ``enable`` column of 0 switches a point off (it is skipped);
* QGIS stores the image row as a *negative* source Y; non-negative values are read as plain
  rows, and a mix of signs is rejected as ambiguous;
* a missing ``#CRS:`` line is allowed; the caller supplies the CRS.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

_ENABLE_NAMES = {"enable", "enabled", "use", "active"}
_POSITIONS = {"map_x": 0, "map_y": 1, "src_x": 2, "src_y": 3, "enable": 4}


class PointsError(ValueError):
    """The ``.points`` file cannot be used; the message says why."""


@dataclass(frozen=True)
class Gcp:
    """A ground control point: image position (``col``, ``row``, pixels from the top-left
    corner of the image, pixel edges as integers) and map position (``x``, ``y``)."""

    col: float
    row: float
    x: float
    y: float
    label: str = ""


@dataclass
class PointsFile:
    gcps: list[Gcp]
    crs: str | None  # as written in the file's #CRS line, if any
    n_disabled: int = 0
    n_unreadable: int = 0
    notes: list[str] = field(default_factory=list)


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _classify(name: str) -> str | None:
    """Which of map_x / map_y / src_x / src_y / enable a normalised header name means, if any.

    Judged by what the name says (source/pixel/image/col/row vs map/easting/northing, and an
    X or Y), not by exact spelling, since QGIS versions and other tools differ.
    """
    if name in _ENABLE_NAMES:
        return "enable"
    is_src = any(k in name for k in ("source", "pixel", "image", "col", "row", "line"))
    is_map = not is_src and any(k in name for k in ("map", "east", "north", "world", "ground"))
    if not (is_src or is_map):
        return None
    if any(k in name for k in ("north", "row", "line")) or name.endswith("y") or name.startswith("y"):
        axis = "y"
    elif any(k in name for k in ("east", "col")) or name.endswith("x") or name.startswith("x"):
        axis = "x"
    else:
        return None
    return f"{'src' if is_src else 'map'}_{axis}"


def _to_float(value: str) -> float | None:
    try:
        return float(value.strip())
    except ValueError:
        return None


def _delimiter(sample: str) -> str:
    counts = {d: sample.count(d) for d in (",", ";", "\t")}
    best = max(counts, key=counts.get)
    return best if counts[best] else ","


def parse_points(source: Path | str) -> PointsFile:
    """Read QGIS ``.points`` content from a file path or a string.

    Raises ``PointsError`` if fewer than three usable points remain.
    """
    text = source.read_text(encoding="utf-8-sig") if isinstance(source, Path) else source
    crs = None
    body = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            m = re.match(r"#\s*crs\s*:\s*(.+)", stripped, re.IGNORECASE)
            if m:
                crs = m[1].strip()
            continue
        body.append(stripped)
    if not body:
        raise PointsError("the file has no points")

    delimiter = _delimiter(body[0])
    rows = list(csv.reader(io.StringIO("\n".join(body)), delimiter=delimiter))
    notes: list[str] = []

    # Header: a first row with any non-numeric cell.
    index = dict(_POSITIONS)
    first = rows[0]
    if any(_to_float(c) is None for c in first if c.strip()):
        rows = rows[1:]
        found: dict[str, int] = {}
        for i, cell in enumerate(first):
            key = _classify(_norm(cell))
            if key and key not in found:
                found[key] = i
        if all(k in found for k in ("map_x", "map_y", "src_x", "src_y")):
            index.update(found)
            if "enable" not in found:
                index.pop("enable")
        else:
            notes.append(
                "header names not recognised; columns read by position "
                "(mapX, mapY, sourceX, sourceY, enable)"
            )

    parsed: list[tuple[float, float, float, float, bool]] = []
    n_unreadable = 0
    for row in rows:
        try:
            vals = [_to_float(row[index[k]]) for k in ("map_x", "map_y", "src_x", "src_y")]
        except IndexError:
            vals = [None]
        if any(v is None for v in vals):
            n_unreadable += 1
            continue
        enabled = True
        if "enable" in index and index["enable"] < len(row):
            flag = _to_float(row[index["enable"]])
            enabled = flag is None or flag != 0
        parsed.append((*vals, enabled))  # type: ignore[arg-type]

    n_disabled = sum(1 for p in parsed if not p[4])
    active = [p for p in parsed if p[4]]
    if len(active) < 3:
        raise PointsError(
            f"{len(active)} usable point(s) found; at least 3 enabled points are needed"
        )

    ys = [p[3] for p in active]
    if all(y <= 0 for y in ys) and any(y < 0 for y in ys):
        negate = True  # QGIS convention: row stored as -row
    elif all(y >= 0 for y in ys):
        negate = False
    else:
        raise PointsError("source Y values have mixed signs; cannot tell rows from QGIS's -rows")

    gcps = [Gcp(col=sx, row=-sy if negate else sy, x=mx, y=my) for mx, my, sx, sy, _ in active]
    return PointsFile(gcps, crs, n_disabled, n_unreadable, notes)


def write_points(path: Path, gcps: list[Gcp], crs: str) -> None:
    """Write ``gcps`` in QGIS Georeferencer format (so they can be opened and corrected)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"#CRS: {crs}", "mapX,mapY,sourceX,sourceY,enable,dX,dY,residual"]
    lines += [f"{g.x:.3f},{g.y:.3f},{g.col:.2f},{-g.row:.2f},1,0,0,0" for g in gcps]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
