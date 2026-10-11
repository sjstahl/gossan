"""Which assessment reports to digitize next, so drilling can start informing the targets.

The ARIS index tags each report with the work it describes. Reports with drilling are the ones that
can supply collars and assays (see ``drill_results``); but digitizing is human work, so it pays to
start with the reports that bear on the targets. This ranks the drilling reports that have not been
digitized yet:

1. by how close the report's location is to a target (``tier``: within ``drill.report_radius_m`` of
   one, within 5 km, or further; without a target model, by distance to a known occurrence);
2. within a tier, by reported exploration spend (more spend, more holes, usually), then newest first.

A report's location in ARIS is a point on the property, not the holes, so distance is a guide to
relevance, not a measurement. ``gossan harvest --limit-pdfs N --pdf-order priority`` then downloads
the reports in this order instead of newest-first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import pandas as pd

from gossan.analyze.common import (
    log_outputs,
    read_layer,
    write_json,
    write_layer,
    write_table,
)
from gossan.analyze.drill_results import classify_holes
from gossan.core.project import Project

DRILL_WORK = re.compile(r"drill|diamond|percussion|reverse circ|rotary", re.IGNORECASE)
NEAR_TIERS_M = (None, 5000.0)  # tier 1 uses ``drill.report_radius_m``; tier 2 this; tier 3 the rest
FILE = "digitize_priorities"


@dataclass
class PriorityResult:
    table: pd.DataFrame
    summary: dict
    outputs: list[str] = field(default_factory=list)


def _is_drilling(row: pd.Series) -> bool:
    text = f"{row.get('GENERAL_WORK_TYPES') or ''};{row.get('SPECIFIC_WORK_TYPES') or ''}"
    return bool(DRILL_WORK.search(text))


def build_digitize_priorities(project: Project) -> PriorityResult:
    """Rank the drilling reports of the ARIS index that are not digitized yet; writes to ``analysis/``."""
    aris = read_layer(project, "aris_index")
    if aris is None:
        summary = {"available": False, "reason": "no aris_index layer (run 'gossan harvest --source aris_index')"}
        outputs = [write_json(project, summary, "digitize_priorities_summary")]
        log_outputs(project, "priorities", outputs, [], {"available": False})
        return PriorityResult(pd.DataFrame(), summary, outputs)

    cfg = project.config.drill
    drilling = aris[aris.apply(_is_drilling, axis=1)].copy()
    drilling["number"] = pd.to_numeric(drilling["ARIS_NUMBER"], errors="coerce")
    drilling = drilling.dropna(subset=["number"]).drop_duplicates("number")

    targets = None
    path = project.root / "target" / "targets.gpkg"
    if path.exists():
        targets = gpd.read_file(path, layer="targets_top", engine="pyogrio").to_crs(project.config.crs)
        targets = targets if len(targets) else None
    occurrences = read_layer(project, "minfile")
    reference, reference_name = (targets, "target") if targets is not None else (occurrences, "known occurrence")

    classified = classify_holes(project)
    done = {} if classified is None else classified.holes.groupby("aris_number").size().to_dict()
    pdf_dir = project.raw_dir / "aris_pdfs"

    rows = []
    for rec in drilling.itertuples():
        number = int(rec.number)
        dist = nearest = None
        if reference is not None and len(reference):
            d = reference.distance(rec.geometry)
            dist = float(d.min())
            nearest = str(reference.iloc[int(np.argmin(d.to_numpy()))].get("target_id", "")) if targets is not None else None
        tier = 3 if dist is None else 1 if dist <= cfg.report_radius_m else 2 if dist <= NEAR_TIERS_M[1] else 3
        rows.append(
            {
                "aris_number": number,
                "year": None if pd.isna(rec.REPORT_YEAR) else int(rec.REPORT_YEAR),
                "title": getattr(rec, "ASSESSMENT_REPORT_TITLE", None),
                "work_types": getattr(rec, "SPECIFIC_WORK_TYPES", None),
                "operator": getattr(rec, "OPERATOR_NAMES", None),
                "spend_2002_dollars": None if pd.isna(getattr(rec, "MERGE_SPENT_2002", np.nan)) else round(float(rec.MERGE_SPENT_2002)),
                f"distance_to_nearest_{reference_name.replace(' ', '_')}_m": None if dist is None else round(dist),
                "nearest_target": nearest,
                "tier": tier,
                "status": f"digitized ({done[number]} hole(s))" if number in done else "not digitized",
                "pdf_downloaded": (pdf_dir / f"ARIS_{number:05d}.pdf").exists(),
                "report_url": getattr(rec, "ARIS_REPORT_URL", None),
                "geometry": rec.geometry,
            }
        )
    table = gpd.GeoDataFrame(rows, geometry="geometry", crs=project.config.crs)
    if len(table):
        todo = table["status"] == "not digitized"
        ranked = table[todo].sort_values(["tier", "spend_2002_dollars", "year"], ascending=[True, False, False], na_position="last")
        table["priority"] = pd.Series(np.arange(1, len(ranked) + 1), index=ranked.index).reindex(table.index).astype("Int64")
        table = table.sort_values(["priority", "aris_number"], na_position="last").reset_index(drop=True)
    else:
        table["priority"] = pd.Series(dtype="Int64")
    summary = {
        "available": True,
        "reference": reference_name,
        "n_reports": len(aris),
        "n_drilling_reports": len(table),
        "n_digitized": int((table["status"] != "not digitized").sum()) if len(table) else 0,
        "n_tier1": int(((table["tier"] == 1) & (table["status"] == "not digitized")).sum()) if len(table) else 0,
        "n_tier2": int(((table["tier"] == 2) & (table["status"] == "not digitized")).sum()) if len(table) else 0,
        "radius_m": cfg.report_radius_m,
        "top": table[table["priority"].notna()].head(8).drop(columns="geometry").to_dict("records") if len(table) else [],
    }
    flat = table.drop(columns="geometry")
    outputs = [write_table(project, flat, FILE), write_json(project, summary, "digitize_priorities_summary")]
    if len(table):
        outputs.append(write_layer(project, table, FILE))
    log_outputs(project, "priorities", outputs, ["aris_index", "aris_pdfs"], {"drilling_reports": len(table), "reference": reference_name})
    return PriorityResult(flat, summary, outputs)
