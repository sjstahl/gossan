"""Apply the scoring engine and write the lead outputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import geopandas as gpd

from gossan.core.licensing import derived_license
from gossan.core.project import Project
from gossan.leads.scoring import (
    ARIS_LAYER,
    TENURE_LAYER,
    LeadsConfig,
    MissingLayerError,
    config_summary,
    read_layer,
    score_leads,
)
from gossan.leads.scoring_us import (
    OCCURRENCE_LAYER,
    TARGETS_GPKG,
    TARGETS_LAYER,
    UsLeadsConfig,
    score_us_leads,
)
from gossan.report.licensing import project_includes_aris

LEADS_LAYER = "leads"
LEADS_CSV = "leads_ranked.csv"
CSV_COLUMNS = [
    "rank", "lead_id", "score", "claim_name", "owner", "tenure_type", "issue_date",
    "good_to_date", "area_ha", "recency_points", "count_points", "legacy_points",
    "aris_count", "aris_legacy_count", "aris_earliest_year", "aris_latest_year",
    "aris_per_km2", "aris_numbers",
]  # fmt: skip
# Leads carry ARIS-derived values, and ARIS is Access Only (see harvest/bc_provincial/aris.py).
DERIVED_LICENSE = (
    "Derived product from mto_tenure (OGL-BC) and aris_index (Access Only): internal use only"
)


@dataclass
class LeadsResult:
    leads: gpd.GeoDataFrame
    csv_path: Path


def run_leads(project: Project, as_of: date | None = None) -> LeadsResult:
    """Score every tenure in ``data.gpkg`` and write ``outputs/leads_ranked.csv`` and the
    ``leads`` layer (project CRS, for QGIS). ``as_of`` defaults to today (UTC); pass it to
    reproduce an earlier run.
    """
    as_of = as_of or datetime.now(UTC).date()
    config = LeadsConfig.from_project(project)
    crs = project.config.crs
    if project.config.jurisdiction == "us":
        return _run_us_leads(project, as_of)
    tenure = read_layer(project, TENURE_LAYER).to_crs(crs)
    aris = read_layer(project, ARIS_LAYER).to_crs(crs)
    leads = score_leads(tenure, aris, config, as_of)

    project.outputs_dir.mkdir(exist_ok=True)
    csv_path = project.outputs_dir / LEADS_CSV
    leads[CSV_COLUMNS].to_csv(csv_path, index=False)
    common = {
        "dataset": LEADS_LAYER,
        "source": "gossan",
        "url": None,
        "license": DERIVED_LICENSE,
    }
    params = {
        "as_of": as_of.isoformat(),
        "config": config_summary(config),
        "n_tenure_in": len(tenure),
        "n_aris_in": len(aris),
        "n_leads": len(leads),
    }
    project.provenance.log(
        identifier=f"outputs/{LEADS_CSV}", operation="write_leads_csv", parameters=params, **common
    )
    if not leads.empty:  # an empty layer cannot be written to GeoPackage
        leads.to_file(project.data_gpkg, layer=LEADS_LAYER, driver="GPKG", engine="pyogrio")
    project.provenance.log(
        identifier=LEADS_LAYER, operation="write_leads_layer", parameters=params, **common
    )
    return LeadsResult(leads, csv_path)


US_CSV_COLUMNS = [
    "rank", "lead_id", "score", "claim_name", "owner", "tenure_type", "area_ha",
    "overlap_points", "proximity_points", "target_overlap_pct", "n_targets",
    "nearest_occurrence_m", "occurrences_inside",
]  # fmt: skip


def is_us_leads(leads: gpd.GeoDataFrame) -> bool:
    """US leads are scored on spatial evidence only: no recency or ARIS columns."""
    return "overlap_points" in leads.columns


def _run_us_leads(project: Project, as_of: date) -> LeadsResult:
    """Score BLM claims by overlap with the Phase 6 targets and proximity to MRDS occurrences."""
    config = UsLeadsConfig.from_project(project)
    crs = project.config.crs
    targets_path = project.root.joinpath(*TARGETS_GPKG)
    if not targets_path.is_file():
        raise MissingLayerError("US lead scoring needs the target polygons; run 'gossan target' first")
    try:
        targets = gpd.read_file(targets_path, layer=TARGETS_LAYER, engine="pyogrio").to_crs(crs)
    except Exception as e:
        raise MissingLayerError(f"cannot read layer {TARGETS_LAYER!r} from {targets_path}; run 'gossan target' again") from e
    claims = read_layer(project, TENURE_LAYER).to_crs(crs)
    occurrences = read_layer(project, OCCURRENCE_LAYER).to_crs(crs)
    leads = score_us_leads(claims, targets, occurrences, config)

    project.outputs_dir.mkdir(exist_ok=True)
    csv_path = project.outputs_dir / LEADS_CSV
    leads[US_CSV_COLUMNS].to_csv(csv_path, index=False)
    common = {
        "dataset": LEADS_LAYER,
        "source": "gossan",
        "url": None,
        "license": derived_license(project.provenance.read(), includes_aris=project_includes_aris(project)),
    }
    params = {
        "as_of": as_of.isoformat(),
        "scoring": "us_spatial",
        "temporal_metrics": "bypassed (BLM MLRS has no issue or good-to dates)",
        "config": config.model_dump(mode="json"),
        "n_claims_in": len(claims),
        "n_targets_in": len(targets),
        "n_occurrences_in": len(occurrences),
        "n_leads": len(leads),
    }
    project.provenance.log(identifier=f"outputs/{LEADS_CSV}", operation="write_leads_csv", parameters=params, **common)
    if not leads.empty:
        leads.to_file(project.data_gpkg, layer=LEADS_LAYER, driver="GPKG", engine="pyogrio")
    project.provenance.log(identifier=LEADS_LAYER, operation="write_leads_layer", parameters=params, **common)
    return LeadsResult(leads, csv_path)
