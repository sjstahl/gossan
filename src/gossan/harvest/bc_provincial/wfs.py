"""Shared logic for layers served by the DataBC public WFS.

Endpoint: ``https://openmaps.gov.bc.ca/geo/pub/wfs`` (no authentication). Features are
requested as GeoJSON in EPSG:4326 with a bounding-box CQL filter around the AOI, paged
by ``OBJECTID``, and then selected against the AOI polygon locally (whole features that
intersect the AOI are kept, not cut at its edge).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlencode

import geopandas as gpd

from gossan.harvest.base import SearchItem
from gossan.harvest.vector_service import Download, VectorServiceConnector

WFS_URL = "https://openmaps.gov.bc.ca/geo/pub/wfs"
PAGE_SIZE = 5000
#: GeoServer invents a feature id like ``LAYER.fid-509cbb71_1a127e60453_-343b`` for views without a
#: primary key, and the part after ``fid-`` changes on every request. It is not data.
SESSION_FID = re.compile(r"\.fid-[0-9a-fA-F_-]+$")
OGL_BC = "Open Government Licence - British Columbia"
OGL_CANADA = "Open Government Licence - Canada"
#: Licence text for layers the BC Data Catalogue lists as "Access Only". Any record carrying it
#: makes deliverables internal (see ``gossan.core.licensing``).
ACCESS_ONLY_BC = (
    "Access Only (BC Data Catalogue): no reproduction or redistribution "
    "without written permission of the Province of BC"
)


def normalize_session_ids(features: list[dict]) -> int:
    """Replace per-request ``fid-...`` feature ids by the stable ``<layer>.<OBJECTID>``, in place;
    returns how many were replaced. Without this two downloads of unchanged data differ byte for
    byte, which defeats ``gossan.core.rawstore``'s duplicate detection. A feature without an
    OBJECTID keeps its id."""
    count = 0
    for feature in features:
        fid = feature.get("id")
        object_id = (feature.get("properties") or {}).get("OBJECTID")
        if isinstance(fid, str) and SESSION_FID.search(fid) and object_id is not None:
            feature["id"] = f"{SESSION_FID.sub('', fid)}.{object_id}"
            count += 1
    return count


class WfsConnector(VectorServiceConnector):
    """A vector layer from the DataBC WFS. Subclasses set the class attributes below.

    ``name`` is also the ``data.gpkg`` layer and the folder under ``raw/``.
    """

    jurisdiction = "bc"
    #: WFS feature type, e.g. ``pub:WHSE_MINERAL_TENURE.MINFIL_MINERAL_FILE``.
    type_name: str
    #: Name of the layer's geometry property (differs between layers).
    geometry_column: str
    #: Optional extra CQL condition, ANDed with the bounding-box filter (for example a field > 0).
    extra_cql: str | None = None

    def search(self, aoi: gpd.GeoDataFrame) -> list[SearchItem]:
        minx, miny, maxx, maxy = aoi.to_crs("EPSG:4326").total_bounds
        bbox = f"{minx:.6f},{miny:.6f},{maxx:.6f},{maxy:.6f}"
        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeNames": self.type_name,
            "outputFormat": "application/json",
            "srsName": "EPSG:4326",
            "sortBy": "OBJECTID",
            "CQL_FILTER": f"BBOX({self.geometry_column},{bbox},'EPSG:4326')"
            + (f" AND {self.extra_cql}" if self.extra_cql else ""),
        }
        url = f"{WFS_URL}?{urlencode(params)}"
        return [SearchItem(identifier=self.type_name, url=url, parameters=params)]

    def download(self, item: SearchItem) -> Download:
        features: list[dict] = []
        start = 0
        while True:
            page = json.loads(self._get(f"{item.url}&count={PAGE_SIZE}&startIndex={start}"))
            features.extend(page["features"])
            if len(page["features"]) < PAGE_SIZE:
                break
            start += PAGE_SIZE
        n_normalized = normalize_session_ids(features)
        content = json.dumps({"type": "FeatureCollection", "features": features}).encode("utf-8")
        parameters = {"cql_filter": item.parameters["CQL_FILTER"]}
        if n_normalized:
            parameters["normalized_session_ids"] = n_normalized
        return Download(content, "geojson", len(features), parameters)

    def read_raw(self, path: Path) -> gpd.GeoDataFrame | None:
        if not json.loads(path.read_text(encoding="utf-8"))["features"]:
            return None  # an empty collection has no schema for the reader to infer
        return gpd.read_file(path, engine="pyogrio")
