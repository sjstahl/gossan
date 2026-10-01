"""Drillhole tables in ``data.gpkg``: ``collar``, ``survey``, ``assay`` and ``lithology``.

GeoPackage is SQLite, so these are ordinary attribute tables that QGIS also lists as layers.
All four share one *long* layout, one row per hole, interval and attribute, because the
columns of an assay sheet differ from report to report and a wide table could not hold them:

==============  ===============================================================
``hole_id``     hole (or sample-series) identifier, as printed
``from_m``      interval start in metres, NULL for a point record
``to_m``        interval end in metres, NULL for a point record
``attribute``   column heading without its unit, e.g. ``Au``
``unit``        unit from the heading's brackets, e.g. ``ppb``; empty if none
``value``       numeric value; NULL for text or non-numeric entries
``flag``        ``<`` or ``>`` when the printed value is a detection-limit bound
``raw_value``   the cell exactly as approved (always kept)
``aris_number``, ``page``, ``review_item``, ``source_file``, ``approved_at``,
``approved_by``  where the record came from and who signed it off
==============  ===============================================================

A censored value such as ``<5`` is stored as value 5 with flag ``<``: never as 5. Rows only
enter these tables through ``load_approved_table``, which is called when a person approves a
review item, so everything here has been through a human check.

A report may hold surface samples (soil, rock chips) rather than drill intervals. Those rows
have no depths; they are loaded with NULL ``from_m`` / ``to_m`` and counted in the result so
the reviewer knows the table is not a set of drill intervals.
"""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyogrio

from gossan.core.project import Project
from gossan.digitize.review import ReviewItem

TABLES = ("collar", "survey", "assay", "lithology")
REQUIRED_KEYS = ("hole_id", "from_m", "to_m")
#: Table type as recorded by the scanned-table extraction -> target table.
TYPE_TO_TABLE = {"assay": "assay", "lithology": "lithology", "collar": "collar", "survey": "survey"}
COLUMNS: dict[str, str] = {
    "hole_id": "object",
    "from_m": "float64",
    "to_m": "float64",
    "attribute": "object",
    "unit": "object",
    "value": "float64",
    "flag": "object",
    "raw_value": "object",
    "aris_number": "Int64",
    "page": "Int64",
    "review_item": "object",
    "source_file": "object",
    "approved_at": "object",
    "approved_by": "object",
}
DERIVED_LICENSE = "Derived from ARIS reports (Access Only): internal use only"
_ITEM_ID = re.compile(r"^R\d{4,}$")
_CENSORED = re.compile(r"^([<>])=?\s*(-?\d+(?:\.\d+)?)$")
_THOUSANDS = re.compile(r"^-?\d{1,3}(?:,\d{3})+(?:\.\d+)?$")
_HEADING = re.compile(r"^\s*(.*?)\s*(?:\(([^()]*)\))?\s*$")


class DrillholeDbError(Exception):
    """The table cannot be loaded; the message says why (nothing was written)."""


@dataclass
class LoadResult:
    table: str
    n_rows: int  # CSV rows
    n_records: int  # long-format records appended
    n_no_interval: int  # rows without depths (surface samples)
    n_censored: int  # values stored with a < or > flag
    n_text: int  # non-numeric entries (value NULL)
    csv_sha256: str


# --- tables -------------------------------------------------------------------------------


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in COLUMNS.items()})


def existing_tables(project: Project) -> set[str]:
    if not project.data_gpkg.exists():
        return set()
    return {str(n) for n in pyogrio.list_layers(project.data_gpkg)[:, 0]}


def init_tables(project: Project) -> list[str]:
    """Create any of the four tables that do not exist yet; returns the ones created."""
    have = existing_tables(project)
    created = []
    for table in TABLES:
        if table not in have:
            pyogrio.write_dataframe(_empty(), project.data_gpkg, layer=table, driver="GPKG")
            created.append(table)
    if created:
        project.provenance.log(
            dataset="drillhole_db",
            source="gossan",
            identifier=",".join(created),
            url=None,
            license="N/A (empty schema)",
            operation="init_drillhole_tables",
            parameters={"gpkg": project.data_gpkg.name, "tables": created, "columns": list(COLUMNS)},
        )
    return created


def table_type_for(project: Project, source: str) -> str | None:
    """Table type (assay, lithology, ...) recorded when ``source`` (a CSV) was extracted."""
    found = None
    for rec in project.provenance.read():
        if rec.operation == "extract_scanned_table" and rec.parameters.get("output") == source:
            found = rec.parameters.get("table_type")
    return found


# --- reading and validating a CSV --------------------------------------------------------


def split_heading(heading: str) -> tuple[str, str]:
    """'Au (ppb)' -> ('Au', 'ppb'); 'Lithology' -> ('Lithology', '')."""
    m = _HEADING.match(heading)
    name, unit = (m[1], m[2] or "") if m else (heading.strip(), "")
    return name.strip() or heading.strip(), unit.strip()


def parse_value(raw: str) -> tuple[float | None, str]:
    """(numeric value or None, flag) for a printed cell: '<5' -> (5.0, '<'), '12' -> (12.0, ''),
    'nil' -> (None, '')."""
    text = raw.strip()
    if m := _CENSORED.match(text):
        return float(m[2]), m[1]
    if _THOUSANDS.match(text):
        return float(text.replace(",", "")), ""
    try:
        return float(text), ""
    except ValueError:
        return None, ""


def _depth(raw: str, what: str, row: int, problems: list[str]) -> float | None:
    if not raw.strip():
        return None
    try:
        value = float(raw.strip())
    except ValueError:
        problems.append(f"row {row}: {what} {raw!r} is not a number")
        return None
    if value < 0:
        problems.append(f"row {row}: {what} {raw!r} is negative")
    return value


def build_records(path: Path, item: ReviewItem, approved_by: str, approved_at: str) -> tuple[pd.DataFrame, LoadResult]:
    """Validate the CSV at ``path`` and turn it into long-format records.

    Raises ``DrillholeDbError`` listing every problem found (up to ten), writing nothing.
    """
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8-sig")
    except (OSError, UnicodeDecodeError) as e:
        raise DrillholeDbError(f"cannot read {path.name}: {e}") from e
    reader = csv.reader(text.splitlines())
    header = next(reader, None)
    if not header:
        raise DrillholeDbError(f"{path.name} is empty")
    keys = {h.strip().lower(): i for i, h in enumerate(header)}
    missing = [k for k in REQUIRED_KEYS if k not in keys]
    if missing:
        raise DrillholeDbError(f"{path.name} is missing required column(s): {', '.join(missing)}")
    key_idx = {keys[k] for k in REQUIRED_KEYS}
    attributes = [(i, h) for i, h in enumerate(header) if i not in key_idx and h.strip()]
    if not attributes:
        raise DrillholeDbError(f"{path.name} has no attribute columns besides {', '.join(REQUIRED_KEYS)}")

    problems: list[str] = []
    records: list[dict] = []
    n_rows = n_no_interval = n_censored = n_text = 0
    for row_no, row in enumerate(reader, start=2):  # row 1 is the header
        if not any(c.strip() for c in row):
            continue
        row = [*row, *[""] * (len(header) - len(row))]
        n_rows += 1
        hole = row[keys["hole_id"]].strip()
        if not hole:
            problems.append(f"row {row_no}: hole_id is empty")
        from_m = _depth(row[keys["from_m"]], "from_m", row_no, problems)
        to_m = _depth(row[keys["to_m"]], "to_m", row_no, problems)
        if (from_m is None) != (to_m is None):
            problems.append(f"row {row_no}: from_m and to_m must both be given or both empty")
        elif from_m is not None and to_m is not None and to_m <= from_m:
            problems.append(f"row {row_no}: to_m ({to_m:g}) is not greater than from_m ({from_m:g})")
        if from_m is None and to_m is None:
            n_no_interval += 1
        for i, heading in attributes:
            cell = row[i].strip()
            if not cell:
                continue
            if cell == "?":
                problems.append(f"row {row_no}: {heading!r} is still marked illegible ('?'); correct the CSV")
                continue
            value, flag = parse_value(cell)
            n_censored += bool(flag)
            n_text += value is None
            name, unit = split_heading(heading)
            records.append(
                {
                    "hole_id": hole,
                    "from_m": from_m,
                    "to_m": to_m,
                    "attribute": name,
                    "unit": unit,
                    "value": value,
                    "flag": flag,
                    "raw_value": cell,
                }
            )
    if n_rows == 0:
        problems.append("the CSV has no data rows")
    if problems:
        shown = "; ".join(problems[:10]) + (f"; ... and {len(problems) - 10} more" if len(problems) > 10 else "")
        raise DrillholeDbError(f"{path.name} failed validation, nothing loaded: {shown}")

    df = pd.DataFrame(records)
    page = int(item.page) if item.page.isdigit() else None
    df["aris_number"] = int(item.report)
    df["page"] = page
    df["review_item"] = item.item_id
    df["source_file"] = item.source
    df["approved_at"] = approved_at
    df["approved_by"] = approved_by
    df = df[list(COLUMNS)].astype({"aris_number": "Int64", "page": "Int64"})
    result = LoadResult("", n_rows, len(df), n_no_interval, n_censored, n_text, hashlib.sha256(raw).hexdigest())
    return df, result


# --- loading ------------------------------------------------------------------------------


def load_approved_table(project: Project, item: ReviewItem, approved_by: str) -> LoadResult:
    """Validate an approved table item's CSV and append its records to the right table.

    Creates the four tables first if they are missing. The table is chosen from the type the
    extraction recorded (assay -> ``assay`` and so on); a table of type ``other``, or one with
    no recorded type, is refused. Raises ``DrillholeDbError`` before writing anything if the CSV
    fails validation or this item was already loaded.
    """
    if not _ITEM_ID.match(item.item_id):
        raise DrillholeDbError(f"unexpected item id {item.item_id!r}")
    csv_path = project.root / item.source
    if not item.source or not csv_path.is_file():
        raise DrillholeDbError(f"the table file {item.source or '(none)'} does not exist")
    kind = table_type_for(project, item.source)
    table = TYPE_TO_TABLE.get(kind or "")
    if table is None:
        raise DrillholeDbError(
            f"{item.source} has table type {kind!r} (from extraction provenance); only "
            f"{', '.join(TYPE_TO_TABLE)} tables can be loaded into the drillhole tables"
        )

    approved_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    df, result = build_records(csv_path, item, approved_by, approved_at)  # validates; writes nothing
    result.table = table

    created = init_tables(project)
    already = pyogrio.read_dataframe(
        project.data_gpkg, layer=table, columns=["review_item"], where=f"review_item = '{item.item_id}'"
    )
    if len(already):
        raise DrillholeDbError(f"{item.item_id} is already loaded into {table} ({len(already)} records)")
    pyogrio.write_dataframe(df, project.data_gpkg, layer=table, driver="GPKG", append=True)

    project.provenance.log(
        dataset=table,
        source="gossan (human-approved table)",
        identifier=item.item_id,
        url=None,
        license=DERIVED_LICENSE,
        operation="load_drillhole_table",
        parameters={
            "gpkg": project.data_gpkg.name,
            "table": table,
            "tables_created": created,
            "csv": item.source,
            "csv_sha256": result.csv_sha256,
            "aris_number": int(item.report),
            "page": item.page,
            "rows": result.n_rows,
            "records": result.n_records,
            "rows_without_depth_interval": result.n_no_interval,
            "censored_values": result.n_censored,
            "text_values": result.n_text,
            "approved_by": approved_by,
        },
    )
    return result
