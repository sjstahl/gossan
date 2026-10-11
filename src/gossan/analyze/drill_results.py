"""What drilling has already found (or not found): digitized drill holes placed, graded and classified.

This closes the loop from ``gossan digitize`` to the rest of Gossan. The tables a reviewer approved
(``collar`` and ``assay`` in ``data.gpkg``; see ``gossan.digitize.db``) become one row per hole:

* **Placement.** Collar easting and northing are read from the collar table and assumed to be UTM in
  the project's zone and the datum set in ``georef:`` (NAD83 by default; old ARIS maps are often
  NAD27), or in ``georef.gcp_crs`` if given. A collar whose coordinates do not land within
  ``placement_buffer_m`` of the AOI is probably on a local mine grid or in the wrong zone or datum:
  it is not placed (``unplaced``), though easting and northing swapped round are recognized and
  corrected. A hole that cannot be placed, but whose ARIS report is in the ``aris_index`` layer, is
  put at the report's location and marked ``report_location``: good for "this property was drilled",
  not for "this cell was drilled".
* **Grades.** Assay attributes are mapped to elements, converted to one unit each (Au and Ag in g/t,
  the rest in %; ppm, ppb, oz/t and % are understood) and censored values (``<5``) count as zero. An
  attribute with no unit is *not used*: a bare "Au" column could be ppb or g/t, and guessing wrong
  turns barren ground into a discovery.
* **Class.** ``mineralized``: some interval reaches a screening threshold (``DrillConfig``);
  ``barren``: sampled over at least ``min_metres`` in at least ``min_samples`` intervals and nothing
  got within ``barren_fraction`` of any threshold; otherwise ``inconclusive``. Holes with no usable
  assay are ``no_assays``.

A barren hole says only that *that hole* found nothing at that spot: ore bodies are narrow and holes
miss them. It is evidence (a real negative, which the rest of the data lack), not proof that the
ground is empty. Everything here is as reliable as the extraction and the reviewer's approval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from gossan.analyze.common import (
    log_outputs,
    read_layer,
    write_json,
    write_layer,
    write_table,
)
from gossan.core.config import DrillConfig
from gossan.core.project import Project
from gossan.digitize.db import TABLES, existing_tables

#: Element symbol -> (attribute-name pattern, canonical unit).
ELEMENTS: dict[str, tuple[re.Pattern[str], str]] = {
    "Au": (re.compile(r"^(au|gold)$", re.IGNORECASE), "g/t"),
    "Ag": (re.compile(r"^(ag|silver)$", re.IGNORECASE), "g/t"),
    "Cu": (re.compile(r"^(cu|copper)$", re.IGNORECASE), "%"),
    "Pb": (re.compile(r"^(pb|lead)$", re.IGNORECASE), "%"),
    "Zn": (re.compile(r"^(zn|zinc)$", re.IGNORECASE), "%"),
    "Mo": (re.compile(r"^(mo|moly(bdenum)?)$", re.IGNORECASE), "%"),
    "Ni": (re.compile(r"^(ni|nickel)$", re.IGNORECASE), "%"),
    "Co": (re.compile(r"^(co|cobalt)$", re.IGNORECASE), "%"),
    "W": (re.compile(r"^(w|wo3|tungsten)$", re.IGNORECASE), "%"),
    "Sb": (re.compile(r"^(sb|antimony)$", re.IGNORECASE), "%"),
}
OZ_PER_TONNE_TO_G = 34.2857  # troy ounces per short ton to g/t
#: Factors to the canonical unit, by canonical unit and the unit as printed (lower case, no spaces).
UNIT_FACTORS: dict[str, dict[str, float]] = {
    "g/t": {"g/t": 1.0, "gpt": 1.0, "g/tonne": 1.0, "ppm": 1.0, "ppb": 1e-3, "oz/t": OZ_PER_TONNE_TO_G, "opt": OZ_PER_TONNE_TO_G, "oz/ton": OZ_PER_TONNE_TO_G, "oz/st": OZ_PER_TONNE_TO_G},
    "%": {"%": 1.0, "pct": 1.0, "wt%": 1.0, "ppm": 1e-4, "g/t": 1e-4, "ppb": 1e-7},
}  # fmt: skip
COORDINATES = {
    "east": re.compile(r"^(east(ing)?|x|utm[ _-]*e(ast(ing)?)?|e)$", re.IGNORECASE),
    "north": re.compile(r"^(north(ing)?|y|utm[ _-]*n(orth(ing)?)?|n)$", re.IGNORECASE),
    "elevation": re.compile(r"^(elev(ation)?|z|rl)$", re.IGNORECASE),
    "azimuth": re.compile(r"^(azi(muth)?|bearing)$", re.IGNORECASE),
    "dip": re.compile(r"^dip$", re.IGNORECASE),
    "length": re.compile(r"^(length|depth|total[ _-]*depth|td|eoh)$", re.IGNORECASE),
}
CLASSES = ("mineralized", "barren", "inconclusive", "no_assays")
PLACED = ("collar_coordinates", "report_location")


@dataclass
class DrillResult:
    holes: gpd.GeoDataFrame
    summary: dict
    outputs: list[str] = field(default_factory=list)


# --- reading the tables ----------------------------------------------------------------------------


def _table(project: Project, name: str) -> pd.DataFrame | None:
    if name not in existing_tables(project):
        return None
    df = gpd.read_file(project.data_gpkg, layer=name, engine="pyogrio", ignore_geometry=True)
    return df if len(df) else None


def hole_key(df: pd.DataFrame) -> pd.Series:
    """``<ARIS number>:<hole id>``: hole ids repeat from report to report."""
    return df["aris_number"].astype("Int64").astype(str) + ":" + df["hole_id"].astype(str)


def collar_crs(project: Project) -> str:
    """The CRS collar coordinates are assumed to be in (see the module docstring)."""
    cfg = project.config
    if cfg.georef.gcp_crs:
        return cfg.georef.gcp_crs
    zone = cfg.utm_zone
    if zone:
        base = {"NAD83": 26900, "NAD27": 26700, "WGS84": 32600}[cfg.georef.datum]
        if cfg.georef.datum == "WGS84" or zone <= 23:  # the NAD83 and NAD27 zone codes stop at 23
            return f"EPSG:{base + zone}"
    return cfg.crs


def collar_fields(collars: pd.DataFrame) -> pd.DataFrame:
    """One row per hole with easting, north, elevation, azimuth, dip and length where the table gave them."""
    rows: dict[str, dict] = {}
    for rec in collars.itertuples():
        row = rows.setdefault(rec.key, {"key": rec.key, "hole_id": rec.hole_id, "aris_number": rec.aris_number})
        for field_name, pattern in COORDINATES.items():
            if pattern.match(str(rec.attribute).strip()) and pd.notna(rec.value):
                value = float(rec.value)
                if "ft" in str(rec.unit).lower() or "feet" in str(rec.unit).lower():
                    value *= 0.3048
                row.setdefault(field_name, value)
    return pd.DataFrame(list(rows.values()))


def place(collars: pd.DataFrame, project: Project, cfg: DrillConfig) -> pd.DataFrame:
    """Add ``x``, ``y`` (project CRS), ``location_quality`` and a note to each collar row."""
    aoi = project.read_aoi().union_all()
    reach = aoi.buffer(cfg.placement_buffer_m)
    crs_in = collar_crs(project)
    out = collars.copy()
    for col in ("east", "north", "elevation", "azimuth", "dip", "length"):
        if col not in out:
            out[col] = np.nan
    out["x"], out["y"] = np.nan, np.nan
    out["location_quality"], out["location_note"] = "unplaced", "no coordinates in the collar table"
    has = out["east"].notna() & out["north"].notna()
    if has.any():
        east, north = out.loc[has, "east"].to_numpy(), out.loc[has, "north"].to_numpy()
        swapped = (east > 1e6) & (north < 1e6)  # northing in the easting column
        east, north = np.where(swapped, north, east), np.where(swapped, out.loc[has, "east"].to_numpy(), north)
        pts = gpd.GeoSeries([Point(e, n) for e, n in zip(east, north, strict=True)], crs=crs_in).to_crs(project.config.crs)
        inside = pts.within(reach).to_numpy()
        idx = out.index[has]
        out.loc[idx, "x"], out.loc[idx, "y"] = pts.x.to_numpy(), pts.y.to_numpy()
        out.loc[idx[inside], "location_quality"] = "collar_coordinates"
        out.loc[idx[inside], "location_note"] = np.where(swapped[inside], "easting and northing were swapped; corrected", "")
        out.loc[idx[~inside], ["x", "y"]] = np.nan
        out.loc[idx[~inside], "location_note"] = (
            f"coordinates fall outside the AOI (+{cfg.placement_buffer_m:g} m) as {crs_in}: a local grid, or the wrong zone or datum?"
        )
    return out


def _factor(canonical: str, unit: str) -> float | None:
    return UNIT_FACTORS[canonical].get(re.sub(r"\s+", "", str(unit).lower()))


def grade_intervals(assays: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Assay records as one row per interval and element, in canonical units.

    Returns (rows with key, element, from_m, to_m, grade), and counts of what was left out:
    ``no_unit`` (attribute has no unit), ``unknown_unit`` (unit not understood) and ``not_interval``
    (no depths: a surface sample).
    """
    left = {"no_unit": 0, "unknown_unit": 0, "not_interval": 0}
    rows = []
    for rec in assays.itertuples():
        element = next((sym for sym, (pat, _) in ELEMENTS.items() if pat.match(str(rec.attribute).strip())), None)
        if element is None:
            continue
        if pd.isna(rec.from_m) or pd.isna(rec.to_m):
            left["not_interval"] += 1
            continue
        if not str(rec.unit).strip():
            left["no_unit"] += 1
            continue
        factor = _factor(ELEMENTS[element][1], rec.unit)
        if factor is None:
            left["unknown_unit"] += 1
            continue
        if pd.isna(rec.value):
            continue
        grade = 0.0 if rec.flag == "<" else float(rec.value) * factor  # below detection: nothing found
        rows.append({"key": rec.key, "element": element, "from_m": float(rec.from_m), "to_m": float(rec.to_m), "grade": grade})
    return pd.DataFrame(rows, columns=["key", "element", "from_m", "to_m", "grade"]), left


def classify(intervals: pd.DataFrame, cfg: DrillConfig) -> pd.DataFrame:
    """One row per hole with samples: class, best element and interval, and what was sampled."""
    out = []
    for key, g in intervals.groupby("key"):
        metres = float(g.groupby("element").apply(lambda e: (e["to_m"] - e["from_m"]).sum(), include_groups=False).max())
        n = int(g.groupby("element").size().max())
        hits, ratios, best = [], {}, None
        for element, e in g.groupby("element"):
            threshold = cfg.thresholds.get(element)
            if threshold is None:
                continue
            top = e.loc[e["grade"].idxmax()]
            ratios[element] = top["grade"] / threshold
            if top["grade"] >= threshold:
                hits.append(element)
            if best is None or ratios[element] > ratios[best[0]]:
                best = (element, top)
        if best is None:
            continue  # only elements without a threshold
        element, top = best
        if hits:
            cls = "mineralized"
        elif metres >= cfg.min_metres and n >= cfg.min_samples and max(ratios.values()) < cfg.barren_fraction:
            cls = "barren"
        else:
            cls = "inconclusive"
        out.append(
            {
                "key": key, "class": cls, "n_samples": n, "metres_sampled": round(metres, 1),
                "best_element": element, "best_grade": round(float(top["grade"]), 4),
                "best_unit": ELEMENTS[element][1], "best_interval": f"{top['from_m']:g}-{top['to_m']:g} m",
                "exceeded": ", ".join(sorted(hits)), "grades": "; ".join(f"{el} {ratios[el] * cfg.thresholds[el]:g}" for el in sorted(ratios)),
            }
    )  # fmt: skip
    return pd.DataFrame(out)


# --- the step -----------------------------------------------------------------------------------------


@dataclass
class Classified:
    holes: gpd.GeoDataFrame
    #: Assay values left out of the grades: ``no_unit``, ``unknown_unit`` and ``not_interval``.
    left_out: dict[str, int]
    #: Distinct ids in the assay table that had no depths: surface samples, not holes.
    n_surface_ids: int


def classify_holes(project: Project) -> Classified | None:
    """Every approved drill hole placed, graded and classified, or None if there are no approved
    ``collar`` or ``assay`` rows. Pure: writes nothing, so the target model can call it too."""
    cfg = project.config.drill
    assays, collars = _table(project, "assay"), _table(project, "collar")
    if assays is None and collars is None:
        return None
    placed = pd.DataFrame(columns=["key", "hole_id", "aris_number"])
    if collars is not None:
        collars = collars.assign(key=hole_key(collars))
        placed = place(collar_fields(collars), project, cfg)
    grades = pd.DataFrame()
    left = {"no_unit": 0, "unknown_unit": 0, "not_interval": 0}
    n_surface_ids = 0
    if assays is not None:
        assays = assays.assign(key=hole_key(assays))
        intervals, left = grade_intervals(assays)
        grades = classify(intervals, cfg)
        n_surface_ids = int(assays.loc[assays["from_m"].isna() & assays["to_m"].isna(), "key"].nunique())

    # Every hole that is in either table, once.
    keys = pd.Index(sorted(set(placed.get("key", [])) | set(grades.get("key", []))), name="key")
    holes = pd.DataFrame(index=keys).reset_index()
    if len(placed):
        holes = holes.merge(placed, on="key", how="left")
    else:
        for col in ("hole_id", "aris_number", "x", "y", "length", "dip", "azimuth", "location_quality", "location_note"):
            holes[col] = np.nan
    if len(grades):
        holes = holes.merge(grades, on="key", how="left")
    else:
        for col in ("class", "n_samples", "metres_sampled", "best_element", "best_grade", "best_unit", "best_interval", "exceeded", "grades"):
            holes[col] = np.nan
    ident = holes["key"].str.split(":", n=1, expand=True)
    holes["aris_number"] = holes["aris_number"].fillna(pd.to_numeric(ident[0], errors="coerce")).astype("Int64")
    holes["hole_id"] = holes["hole_id"].fillna(ident[1])
    holes["class"] = holes["class"].fillna("no_assays")
    holes["location_quality"] = holes["location_quality"].fillna("unplaced")
    holes["location_note"] = holes["location_note"].fillna("no collar table row")

    # An unplaced hole whose report has a location in the ARIS index: place it there, flagged.
    aris = read_layer(project, "aris_index")
    if aris is not None and "ARIS_NUMBER" in aris:
        report_pts = aris.assign(n=pd.to_numeric(aris["ARIS_NUMBER"], errors="coerce")).dropna(subset=["n"]).drop_duplicates("n").set_index("n")
        for i in holes.index[(holes["location_quality"] == "unplaced") & holes["aris_number"].notna()]:
            n = float(holes.at[i, "aris_number"])
            if n in report_pts.index:
                pt = report_pts.loc[n, "geometry"]
                holes.loc[i, ["x", "y"]] = pt.x, pt.y
                holes.at[i, "location_quality"] = "report_location"
                holes.at[i, "location_note"] = f"{holes.at[i, 'location_note']}; placed at the ARIS report's location".lstrip("; ")

    ok = holes["x"].notna() & holes["y"].notna()
    geometry = [Point(x, y) if good else None for x, y, good in zip(holes["x"], holes["y"], ok, strict=True)]
    gdf = gpd.GeoDataFrame(holes.drop(columns=["x", "y"]), geometry=geometry, crs=project.config.crs)
    return Classified(gdf, left, n_surface_ids)


def build_drill_results(project: Project) -> DrillResult:
    """Place, grade and classify the approved drill holes; writes to ``analysis/``.

    Without any approved ``collar`` or ``assay`` rows there is nothing to do: the result says so and
    only the short summary file recording that is written.
    """
    cfg = project.config.drill
    classified = classify_holes(project)
    empty = gpd.GeoDataFrame({"key": []}, geometry=[], crs=project.config.crs)
    if classified is None:
        reason = "no approved drillhole tables (run 'gossan digitize' and approve the extracted tables with 'gossan review')"
        summary = {"available": False, "reason": reason, "tables": sorted(existing_tables(project) & set(TABLES))}
        outputs = [write_json(project, summary, "drill_summary")]
        log_outputs(project, "drilling", outputs, [], {"available": False})
        return DrillResult(empty, summary, outputs)
    gdf, left, n_surface_ids = classified.holes, classified.left_out, classified.n_surface_ids
    holes = gdf.drop(columns="geometry").assign(x=gdf.geometry.x, y=gdf.geometry.y)
    summary = {
        "available": True,
        "n_holes": len(gdf),
        "by_class": {c: int((gdf["class"] == c).sum()) for c in CLASSES},
        "by_location": {q: int((gdf["location_quality"] == q).sum()) for q in ("collar_coordinates", "report_location", "unplaced")},
        "n_reports": int(gdf["aris_number"].nunique()),
        "left_out_of_grades": left,
        "n_surface_sample_ids": n_surface_ids,
        "thresholds": cfg.thresholds,
        "mineralized_holes": [
            {"hole": f"{r.hole_id} (ARIS {r.aris_number})", "element": r.best_element, "grade": r.best_grade, "unit": r.best_unit, "interval": r.best_interval}
            for r in gdf[gdf["class"] == "mineralized"].sort_values("best_grade", ascending=False).head(10).itertuples()
        ],
    }
    placed_holes = gdf[gdf.geometry.notna()]
    outputs = [write_table(project, gdf.drop(columns="geometry").assign(x=holes["x"], y=holes["y"]), "drill_holes"), write_json(project, summary, "drill_summary")]
    if len(placed_holes):
        outputs.append(write_layer(project, placed_holes, "drill_holes"))
    # The tables come from ARIS reports (Access Only), so the outputs inherit that whatever the ledger says.
    log_outputs(
        project, "drilling", outputs, ["aris_pdfs"],
        {"holes": len(gdf), "by_class": summary["by_class"], "by_location": summary["by_location"]},
    )  # fmt: skip
    return DrillResult(gdf, summary, outputs)
