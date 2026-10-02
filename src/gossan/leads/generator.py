"""Apply the scoring engine and write the lead outputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import geopandas as gpd

from gossan.core.project import Project
from gossan.leads.scoring import (
    ARIS_LAYER,
    TENURE_LAYER,
    LeadsConfig,
    config_summary,
    read_layer,
    score_leads,
)

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
