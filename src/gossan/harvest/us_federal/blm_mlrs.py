"""BLM Mineral and Land Record System (MLRS) mining claims, mapped to the ``mto_tenure`` layer.

Access method: the BLM National Land Status Database ArcGIS REST service
``https://gis.blm.gov/nlsdb/rest/services/HUB/BLM_Natl_MLRS_Mining_Claims_Not_Closed/MapServer/0``
("Mining Claims - Not Closed": claim cases whose disposition is anything but closed; no
credentials; ``maxRecordCount`` 2000, paged 1000 at a time). See ``gossan.harvest.arcgis_rest``
for the request details. Claims closed or abandoned long ago are not in this layer.

Source fields and the canonical ``mto_tenure`` columns they become:

* TENURE_NUMBER_ID: the digits of the case serial ``CSE_NR`` (``NV105223401`` becomes
  105223401), an integer like BC's tenure id. If those digits would repeat across different
  cases, or are missing, the service's own numeric ``ID`` is used instead. The serial itself is
  kept in BLM_CASE_SERIAL.
* CLAIM_NAME = CSE_NAME; TENURE_TYPE_DESCRIPTION from ``BLM_PROD``: Lode Claim -> "Mineral",
  Placer Claim -> "Placer", Mill Site -> "Mill site", Tunnel Site -> "Tunnel site".
* AREA_IN_HECTARES = recorded acres (``RCRD_ACRS``) x 0.4047.
* ISSUE_DATE, GOOD_TO_DATE and OWNER_NAME are **empty**: the public layer carries no location
  date, no expiry (unpatented claims stay valid by paying annual maintenance fees, not by a
  good-to date) and no claimant. ``Created`` is the date the record was created in the database
  (many old claims were loaded in bulk), not when the claim was staked, so it is kept as
  RECORD_CREATED and deliberately not used as ISSUE_DATE. Consequence for ``leads``: no US claim
  earns recency points.
* Kept as extra columns: DISPOSITION (``CSE_DISP``), LEGAL_DESCRIPTION (``CSE_META``),
  POSITION_QUALITY (first line of ``QLTY``), STATE (letters of the serial).

One row per case: when the service returns several polygons for a case they are merged.

Accuracy: BLM only requires claims to be identified to the quarter section, and the polygons are
geocoded from legal land descriptions on the PLSS grid. They are neither surveys nor legal
documents; the staked, monumented boundary is in the case file at the BLM State Office. Some cases
have attributes but no geometry; they cannot be mapped and are dropped.

Terms of use: BLM publishes the layer "as is" (disclaimer on the item page: no warranty, not a
legal document or land survey, may change). It states no licence; U.S. Government works are
generally in the public domain (17 U.S.C. 105), but no explicit statement was found.
"""

from __future__ import annotations

import re

import geopandas as gpd
import pandas as pd

from gossan.core.schemas import TENURE_LAYER, conform
from gossan.harvest.arcgis_rest import ArcGisRestConnector

LAYER_URL = "https://gis.blm.gov/nlsdb/rest/services/HUB/BLM_Natl_MLRS_Mining_Claims_Not_Closed/MapServer/0"
LICENSE = "Public domain (U.S. Government work, BLM); no explicit licence stated; provided as is, not a legal document or survey"
ACRES_TO_HECTARES = 0.40468564224
CLAIM_TYPES = {
    "lode claim": "Mineral",
    "placer claim": "Placer",
    "mill site": "Mill site",
    "tunnel site": "Tunnel site",
}


def claim_type(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        return "Unknown"
    return CLAIM_TYPES.get(value.strip().lower(), value.strip())


def serial_digits(serial: object) -> int | None:
    """105223401 from 'NV105223401'; None if there are no digits."""
    digits = re.sub(r"\D", "", serial) if isinstance(serial, str) else ""
    return int(digits) if digits else None


class BlmClaimsConnector(ArcGisRestConnector):
    """BLM MLRS mining claims intersecting the AOI, written to the ``mto_tenure`` layer."""

    name = "blm_claims"
    canonical_layer = TENURE_LAYER
    jurisdiction = "us"
    source = "BLM Mineral and Land Record System, mining claims not closed (NLSDB ArcGIS REST)"
    license = LICENSE
    layer_url = LAYER_URL
    page_size = 1000
    order_field = "OBJECTID"

    def normalize(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
        gdf["CSE_NR"] = gdf["CSE_NR"].where(gdf["CSE_NR"].astype(str).str.strip() != "", None)
        has_serial = gdf["CSE_NR"].notna()
        merged = pd.concat(
            [
                gdf[has_serial].dissolve(by="CSE_NR", aggfunc="first", as_index=False),
                gdf[~has_serial],  # a case without a serial cannot be merged: keep each row
            ],
            ignore_index=True,
        )
        out = gpd.GeoDataFrame(merged, geometry="geometry", crs=gdf.crs)

        # Python ints, not floats: the service's own ids have 18 digits, more than a float holds exactly.
        service_id = pd.array([int(v) if str(v).strip().isdigit() else None for v in out.get("ID", [None] * len(out))], dtype="Int64")
        number = pd.array([serial_digits(v) for v in out["CSE_NR"]], dtype="Int64")
        repeated = pd.Series(number).duplicated(keep=False).to_numpy() | pd.isna(number)
        number = pd.array([sid if rep else n for n, sid, rep in zip(number, service_id, repeated, strict=True)], dtype="Int64")  # fall back to the service's own id
        out["TENURE_NUMBER_ID"] = number
        out["CLAIM_NAME"] = out["CSE_NAME"]
        out["TENURE_TYPE_DESCRIPTION"] = out["BLM_PROD"].map(claim_type)
        out["AREA_IN_HECTARES"] = pd.to_numeric(out["RCRD_ACRS"], errors="coerce") * ACRES_TO_HECTARES
        area_ha = out.geometry.area / 1e4  # project CRS is metric
        out["AREA_IN_HECTARES"] = out["AREA_IN_HECTARES"].fillna(area_ha)
        out["ISSUE_DATE"] = pd.NaT
        out["GOOD_TO_DATE"] = pd.NaT
        out["OWNER_NAME"] = None
        out["BLM_CASE_SERIAL"] = out["CSE_NR"]
        out["DISPOSITION"] = out["CSE_DISP"]
        out["LEGAL_DESCRIPTION"] = out["CSE_META"]
        out["POSITION_QUALITY"] = out["QLTY"].map(lambda v: v.splitlines()[0].strip() if isinstance(v, str) and v.strip() else None)
        out["STATE"] = out["CSE_NR"].map(lambda v: re.match(r"[A-Za-z]+", v).group(0).upper() if isinstance(v, str) and re.match(r"[A-Za-z]+", v) else None)
        created = pd.to_numeric(out.get("Created"), errors="coerce")
        out["RECORD_CREATED"] = pd.to_datetime(created, unit="ms", utc=True, errors="coerce")
        out = out.dropna(subset=["TENURE_NUMBER_ID"])
        return conform(out, TENURE_LAYER)
