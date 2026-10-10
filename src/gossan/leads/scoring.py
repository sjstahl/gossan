"""Lead scoring: which tenures look ripe for data compilation or digitization.

The score is additive and out of 100 (all weights are configurable in ``project.yaml``
under ``parameters: leads:``; see ``LeadsConfig``):

``score = min(100, recency_points + count_points + legacy_points + drilling_points)``

* ``recency_points``: points for how recently the tenure was issued, by age tier. By
  default 50 within 1 year, 30 within 3 years, 15 within 5 years, else 0. A new
  holder is the most likely to want their legacy data pulled together.
* ``count_points``: ARIS history on the property, linear in the number of reports whose
  location falls inside the tenure, reaching its maximum (25) at 10 reports.
* ``legacy_points``: historical depth, linear in the number of those reports dated
  before 1990 (paper-era, the likeliest to need digitizing), reaching its maximum (25)
  at 5 such reports.
* ``drilling_points``: a small bonus when a Notice of Work with surface drilling is on the
  tenure (10 if active, 5 if pending, 0 if lapsed by default; see ``gossan.leads.drilling``).
  The flag itself (``drilling_flag``) is reported separately from the score. The total is
  capped at 100.

Caveats, by design of the source data:

* The tenure layer is a snapshot of *current* tenure with its original ``ISSUE_DATE``;
  it carries no change history, so "recent" means recently issued, not recently
  transferred or re-staked. Snapshot comparison is a later refinement.
* ARIS features are one point per report (a property location), not a footprint, so a
  report counts for a tenure when its point falls inside the tenure polygon.
* Commodity and deposit-type fit (PLAN.md section 7.3) is not scored yet.
* ARIS is Access Only: lead outputs derived from it are for internal use.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import geopandas as gpd
import pandas as pd
import pyogrio
from pydantic import BaseModel, ConfigDict, Field

from gossan.core.project import Project
from gossan.leads.drilling import COLUMNS, NONE_FOUND, NOT_ASSESSED, drilling_summary

TENURE_LAYER = "mto_tenure"
ARIS_LAYER = "aris_index"
DRILLING_LAYER = "drilling_notices"


class LeadsConfig(BaseModel):
    """Scoring parameters; defaults apply unless ``parameters.leads`` overrides them."""

    model_config = ConfigDict(extra="forbid")

    #: (max tenure age in years, points), checked in order; older than all tiers scores 0.
    recency_tiers: list[tuple[float, float]] = [(1, 50), (3, 30), (5, 15)]
    aris_count_points: float = Field(default=25, ge=0)
    aris_count_cap: int = Field(default=10, ge=1)
    legacy_points: float = Field(default=25, ge=0)
    legacy_cap: int = Field(default=5, ge=1)
    #: ARIS reports dated before this year count as legacy (paper-era).
    legacy_before_year: int = 1990
    #: Tenure types to score (matched case-insensitively against TENURE_TYPE_DESCRIPTION).
    tenure_types: list[str] = ["Mineral"]
    #: Skip tenures whose good-to date is before the as-of date.
    active_only: bool = True
    #: Bonus for a drilling Notice of Work on the tenure, by its status (0 turns the bonus off).
    drilling_active_points: float = Field(default=10, ge=0)
    drilling_pending_points: float = Field(default=5, ge=0)
    drilling_lapsed_points: float = Field(default=0, ge=0)
    #: A notice point within this distance (m) of a tenure counts as on it, as well as by tenure number.
    drilling_match_buffer_m: float = Field(default=500, ge=0)

    @classmethod
    def from_project(cls, project: Project) -> LeadsConfig:
        return cls.model_validate(project.config.parameters.get("leads", {}))


class MissingLayerError(Exception):
    """A required input layer is not in ``data.gpkg``."""


def read_layer(project: Project, layer: str) -> gpd.GeoDataFrame:
    gpkg = project.data_gpkg
    if not gpkg.exists() or layer not in pyogrio.list_layers(gpkg)[:, 0]:
        if layer == ARIS_LAYER and project.config.jurisdiction != "bc":
            raise MissingLayerError(
                "lead scoring needs the ARIS assessment-report index, a British Columbia dataset; "
                "there is no equivalent US source yet, so 'gossan leads run' is not available for this project"
            )
        raise MissingLayerError(
            f"layer {layer!r} is not in data.gpkg; run 'gossan harvest --source {layer}' first"
        )
    return gpd.read_file(gpkg, layer=layer, engine="pyogrio")


def optional_layer(project: Project, layer: str) -> gpd.GeoDataFrame | None:
    """The layer, or None when ``data.gpkg`` does not have it."""
    gpkg = project.data_gpkg
    if not gpkg.exists() or layer not in pyogrio.list_layers(gpkg)[:, 0]:
        return None
    return gpd.read_file(gpkg, layer=layer, engine="pyogrio")


def recency_points(age_years: float, tiers: list[tuple[float, float]]) -> float:
    """Points for a tenure of the given age; NaN age (no issue date) scores 0."""
    if pd.isna(age_years):
        return 0.0
    for max_age, points in tiers:
        if age_years <= max_age:
            return float(points)
    return 0.0


def score_leads(
    tenure: gpd.GeoDataFrame,
    aris: gpd.GeoDataFrame,
    config: LeadsConfig,
    as_of: date,
    notices: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    """Score tenure polygons against ARIS points (both in the same projected CRS).

    ``notices`` are the ``drilling_notices`` points; None means the layer was not harvested, and
    the drilling columns say "not assessed".

    Returns one row per tenure, ranked best first, with the score components.
    """
    tenure = _one_row_per_tenure(tenure)
    wanted = {t.lower() for t in config.tenure_types}
    tenure = tenure[tenure["TENURE_TYPE_DESCRIPTION"].str.lower().isin(wanted)]

    as_of_ts = pd.Timestamp(as_of, tz="UTC")
    issue = pd.to_datetime(tenure["ISSUE_DATE"], utc=True, errors="coerce")
    good_to = pd.to_datetime(tenure["GOOD_TO_DATE"], utc=True, errors="coerce")
    if config.active_only:
        tenure = tenure[~(good_to < as_of_ts)]
        issue, good_to = issue[tenure.index], good_to[tenure.index]

    hits = gpd.sjoin(
        tenure[["TENURE_NUMBER_ID", "geometry"]],
        aris[["ARIS_NUMBER", "REPORT_YEAR", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    per_tenure = hits.groupby("TENURE_NUMBER_ID")
    stats = pd.DataFrame(
        {
            "aris_count": per_tenure["ARIS_NUMBER"].nunique(),
            "aris_legacy_count": per_tenure["REPORT_YEAR"].apply(
                lambda y: int((y < config.legacy_before_year).sum())
            ),
            "aris_earliest_year": per_tenure["REPORT_YEAR"].min(),
            "aris_latest_year": per_tenure["REPORT_YEAR"].max(),
            "aris_numbers": per_tenure["ARIS_NUMBER"].apply(
                lambda n: "; ".join(str(int(v)) for v in sorted(set(n)))
            ),
        }
    )

    out = gpd.GeoDataFrame(
        {
            "lead_id": tenure["TENURE_NUMBER_ID"].astype(int).astype(str),
            "claim_name": tenure["CLAIM_NAME"],
            "tenure_type": tenure["TENURE_TYPE_DESCRIPTION"],
            "owner": tenure["OWNER_NAME"],
            "issue_date": issue.dt.strftime("%Y-%m-%d"),
            "good_to_date": good_to.dt.strftime("%Y-%m-%d"),
            "area_ha": tenure["AREA_IN_HECTARES"],
            "tenure_age_years": ((as_of_ts - issue).dt.days / 365.25).round(2),
        },
        geometry=tenure.geometry,
        crs=tenure.crs,
    )
    keyed = tenure["TENURE_NUMBER_ID"]
    for col in stats.columns:
        out[col] = keyed.map(stats[col])
    for col in ("aris_count", "aris_legacy_count"):
        out[col] = out[col].fillna(0).astype(int)
    for col in ("aris_earliest_year", "aris_latest_year"):
        out[col] = out[col].astype("Int64")  # whole years; <NA> when the tenure has no reports
    out["aris_numbers"] = out["aris_numbers"].fillna("")
    area_km2 = out.geometry.area / 1e6
    out["aris_per_km2"] = (out["aris_count"] / area_km2).round(2)

    out["recency_points"] = out["tenure_age_years"].map(
        lambda a: recency_points(a, config.recency_tiers)
    )
    out["count_points"] = (
        config.aris_count_points * out["aris_count"].clip(upper=config.aris_count_cap)
        / config.aris_count_cap
    )
    out["legacy_points"] = (
        config.legacy_points * out["aris_legacy_count"].clip(upper=config.legacy_cap)
        / config.legacy_cap
    )  # fmt: skip
    if notices is None:
        drilling = pd.DataFrame(columns=COLUMNS)
    else:
        drilling = drilling_summary(tenure, notices, config.drilling_match_buffer_m, as_of)
    for col in COLUMNS:
        out[col] = keyed.map(drilling[col]) if len(drilling) else None
    unflagged = NOT_ASSESSED if notices is None else NONE_FOUND
    out["drilling_flag"] = out["drilling_flag"].fillna(unflagged)
    for col in ("drilling_active_count", "drilling_pending_count", "drilling_lapsed_count"):
        out[col] = out[col].fillna(0).astype(int)
    for col in ("drilling_latest_filed", "drilling_operator", "drilling_notice_numbers"):
        out[col] = out[col].fillna("")
    bonus = {
        "active": config.drilling_active_points,
        "pending": config.drilling_pending_points,
        "lapsed": config.drilling_lapsed_points,
    }
    out["drilling_points"] = out["drilling_flag"].map(bonus).fillna(0.0)
    out["score"] = (
        (out["recency_points"] + out["count_points"] + out["legacy_points"] + out["drilling_points"])
        .clip(upper=100)
        .round(2)
    )

    out = out.sort_values(
        ["score", "aris_count", "lead_id"], ascending=[False, False, True]
    ).reset_index(drop=True)
    out.insert(1, "rank", range(1, len(out) + 1))
    return out


def _one_row_per_tenure(tenure: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """The layer has one row per owner of a tenure; merge them, joining owner names."""
    owners = tenure.groupby("TENURE_NUMBER_ID")["OWNER_NAME"].agg(
        lambda s: "; ".join(sorted({o for o in s.dropna()}))
    )
    unique = tenure.drop_duplicates("TENURE_NUMBER_ID").copy()
    unique["OWNER_NAME"] = unique["TENURE_NUMBER_ID"].map(owners)
    return unique


def config_summary(config: LeadsConfig) -> dict[str, Any]:
    return config.model_dump(mode="json")
