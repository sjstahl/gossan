"""Lead scoring for US projects: spatial evidence only.

BLM MLRS claims carry no issue date, good-to date or claimant, so the temporal metrics of the BC
scoring (recency, expiry) are bypassed. A claim is scored out of 100 on two spatial signals:

``score = overlap_points + proximity_points``

* ``overlap_points``: the share of the claim's area that falls inside the Phase 6 target polygons
  (``target/targets.gpkg``, layer ``targets_top``), scaled to ``overlap_points`` (default 60).
* ``proximity_points``: closeness to MRDS occurrences (the ``minfile`` layer). Full marks when an
  occurrence lies on the claim, falling linearly to zero at ``proximity_m`` (default 2000 m)
  from the claim boundary.

Caveats: BLM claims are resolved to quarter-section geocodes, not surveyed boundaries, so overlap
and distance are approximate to about a quarter section. MRDS is a historical inventory (not
systematically updated since 2011): proximity says "near a documented occurrence", not "near
current activity". Lead scores prioritise compilation work; they are not a mineral-potential rating.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from gossan.core.project import Project
from gossan.leads.scoring import _one_row_per_tenure

TARGETS_GPKG = ("target", "targets.gpkg")
TARGETS_LAYER = "targets_top"
OCCURRENCE_LAYER = "minfile"


class UsLeadsConfig(BaseModel):
    """Weights for US scoring; override under ``parameters: leads:`` in ``project.yaml``."""

    model_config = ConfigDict(extra="forbid")

    overlap_points: float = Field(default=60, ge=0)
    proximity_points: float = Field(default=40, ge=0)
    #: Distance (m) from a claim to the nearest occurrence beyond which proximity scores 0.
    proximity_m: float = Field(default=2000, gt=0)

    @classmethod
    def from_project(cls, project: Project) -> UsLeadsConfig:
        return cls.model_validate(project.config.parameters.get("leads", {}))


def score_us_leads(
    claims: gpd.GeoDataFrame, targets: gpd.GeoDataFrame, occurrences: gpd.GeoDataFrame, config: UsLeadsConfig
) -> gpd.GeoDataFrame:
    """Score claims against target polygons and MRDS points (all in one projected CRS)."""
    claims = _one_row_per_tenure(claims).reset_index(drop=True)
    area = claims.geometry.area

    overlap = pd.Series(0.0, index=claims.index)
    n_targets = pd.Series(0, index=claims.index, dtype=int)
    if len(targets):
        union = targets.geometry.union_all()
        inter = claims.geometry.intersection(union).area
        overlap = (inter / area.where(area > 0)).fillna(0.0).clip(0, 1)
        hits = gpd.sjoin(claims[["geometry"]], targets[["geometry"]], how="inner", predicate="intersects")
        n_targets = hits.groupby(level=0).size().reindex(claims.index, fill_value=0).astype(int)

    nearest = pd.Series(np.nan, index=claims.index)
    inside = pd.Series(0, index=claims.index, dtype=int)
    if len(occurrences):
        joined = gpd.sjoin_nearest(claims[["geometry"]], occurrences[["geometry"]], how="left", distance_col="d")
        nearest = joined.groupby(level=0)["d"].min().reindex(claims.index)
        within = gpd.sjoin(claims[["geometry"]], occurrences[["geometry"]], how="inner", predicate="intersects")
        inside = within.groupby(level=0).size().reindex(claims.index, fill_value=0).astype(int)
    closeness = (1 - nearest / config.proximity_m).clip(0, 1).fillna(0.0)

    out = gpd.GeoDataFrame(
        {
            "lead_id": claims["TENURE_NUMBER_ID"].astype(int).astype(str),
            "claim_name": claims["CLAIM_NAME"],
            "tenure_type": claims["TENURE_TYPE_DESCRIPTION"],
            "owner": claims["OWNER_NAME"],
            "area_ha": claims["AREA_IN_HECTARES"],
            "target_overlap_pct": (overlap * 100).round(1),
            "n_targets": n_targets,
            "nearest_occurrence_m": nearest.round(0),
            "occurrences_inside": inside,
        },
        geometry=claims.geometry,
        crs=claims.crs,
    )
    out["overlap_points"] = (config.overlap_points * overlap).round(2)
    out["proximity_points"] = (config.proximity_points * closeness).round(2)
    out["score"] = (out["overlap_points"] + out["proximity_points"]).round(2)
    out = out.sort_values(["score", "target_overlap_pct", "lead_id"], ascending=[False, False, True]).reset_index(drop=True)
    out.insert(1, "rank", range(1, len(out) + 1))
    return out
