"""A vector layer from an ArcGIS REST ``MapServer`` / ``FeatureServer`` endpoint.

Many US federal and state agencies publish through ArcGIS Server. The ``query`` operation takes
an envelope in WGS84 and returns GeoJSON, a page at a time (``resultOffset`` and
``resultRecordCount``; servers cap a page at their ``maxRecordCount``, commonly 1000 to 2000).
Pages are ordered by a stable field so they cannot overlap or skip. ArcGIS reports problems as an
``error`` object in a normal 200 response, so every page is checked for one.

Subclasses set ``layer_url`` (the endpoint ending in the layer number) and normalise the fields.
"""

from __future__ import annotations

import json
from urllib.parse import urlencode

import geopandas as gpd

from gossan.harvest.base import SearchItem
from gossan.harvest.vector_service import Download, VectorServiceConnector

MAX_PAGES = 500


class ArcGisRestConnector(VectorServiceConnector):
    """Base class for ArcGIS REST layers."""

    #: Layer endpoint, e.g. ``https://host/arcgis/rest/services/Folder/Name/MapServer/0``.
    layer_url: str
    #: Records per request; keep at or below the service's ``maxRecordCount``.
    page_size = 1000
    #: Field the pages are ordered by.
    order_field = "OBJECTID"
    #: Attribute filter applied on the server.
    where = "1=1"

    def search(self, aoi: gpd.GeoDataFrame) -> list[SearchItem]:
        minx, miny, maxx, maxy = aoi.to_crs("EPSG:4326").total_bounds
        params = {
            "where": self.where,
            "geometry": f"{minx:.6f},{miny:.6f},{maxx:.6f},{maxy:.6f}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*",
            "outSR": "4326",
            "orderByFields": self.order_field,
            "f": "geojson",
        }
        return [SearchItem(identifier=self.layer_url, url=f"{self.layer_url}/query?{urlencode(params)}", parameters=params)]

    def download(self, item: SearchItem) -> Download:
        features: list[dict] = []
        pages = 0
        while pages < MAX_PAGES:
            url = f"{item.url}&resultOffset={len(features)}&resultRecordCount={self.page_size}"
            page = json.loads(self._get(url))
            if "error" in page:
                err = page["error"]
                raise OSError(f"ArcGIS service error {err.get('code')}: {err.get('message')} {err.get('details', '')}".strip())
            got = page.get("features", [])
            features.extend(got)
            pages += 1
            more = page.get("exceededTransferLimit") or len(got) >= self.page_size
            if not got or not more:
                break
        else:
            raise OSError(f"stopped after {MAX_PAGES} pages ({len(features)} features): the area is too large for one harvest")
        content = json.dumps({"type": "FeatureCollection", "features": features}).encode("utf-8")
        p = item.parameters
        return Download(content, "geojson", len(features), {"where": p["where"], "envelope": p["geometry"], "pages": pages})

    def read_raw(self, path):
        if not json.loads(path.read_text(encoding="utf-8"))["features"]:
            return None
        return gpd.read_file(path, engine="pyogrio")
