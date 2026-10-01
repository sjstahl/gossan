"""BC MINFILE mineral occurrences.

Access method: the BC Geographic Warehouse (DataBC) public WFS at
``https://openmaps.gov.bc.ca/geo/pub/wfs``, layer
``pub:WHSE_MINERAL_TENURE.MINFIL_MINERAL_FILE`` ("MINFILE Mineral Occurrence
Database"), requested as GeoJSON in EPSG:4326 with a bounding-box CQL filter and
paged by ``OBJECTID``. Results are then clipped to the AOI polygon locally.
No authentication is required.

Catalogue record: https://catalogue.data.gov.bc.ca/dataset/92206d94-bc64-4111-a295-cd14eb5a501c

Terms of use: Open Government Licence - British Columbia
(https://www2.gov.bc.ca/gov/content/data/policy-standards/open-data/open-government-licence-bc).
Reuse, including commercially, is permitted with attribution: "Contains
information licensed under the Open Government Licence - British Columbia."
The source is the BC Geological Survey, Ministry of Mining and Critical Minerals.
"""

from __future__ import annotations

import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode

import geopandas as gpd
import pandas as pd

from gossan.harvest.base import Connector, SearchItem

WFS_URL = "https://openmaps.gov.bc.ca/geo/pub/wfs"
TYPE_NAME = "pub:WHSE_MINERAL_TENURE.MINFIL_MINERAL_FILE"
PAGE_SIZE = 5000
OGL_BC = "Open Government Licence - British Columbia"


class MinfileConnector(Connector):
    """MINFILE occurrences intersecting the AOI, written to the ``minfile`` layer."""

    name = "minfile"
    layer = "minfile"
    source = "BC Geological Survey (DataBC WFS)"
    license = OGL_BC

    def search(self, aoi: gpd.GeoDataFrame) -> list[SearchItem]:
        minx, miny, maxx, maxy = aoi.to_crs("EPSG:4326").total_bounds
        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeNames": TYPE_NAME,
            "outputFormat": "application/json",
            "srsName": "EPSG:4326",
            "sortBy": "OBJECTID",
            "CQL_FILTER": f"BBOX(SHAPE,{minx:.6f},{miny:.6f},{maxx:.6f},{maxy:.6f},'EPSG:4326')",
        }
        url = f"{WFS_URL}?{urlencode(params)}"
        return [SearchItem(identifier=TYPE_NAME, url=url, parameters=params)]

    def fetch(self, items: list[SearchItem]) -> list[Path]:
        raw = self.project.raw_dir / self.name
        raw.mkdir(parents=True, exist_ok=True)
        files = []
        for item in items:
            retrieved_at = datetime.now(UTC)
            features: list[dict] = []
            start = 0
            while True:
                page = json.loads(self._get(f"{item.url}&count={PAGE_SIZE}&startIndex={start}"))
                features.extend(page["features"])
                if len(page["features"]) < PAGE_SIZE:
                    break
                start += PAGE_SIZE
            # Timestamped name: originals in raw/ are never overwritten.
            path = raw / f"minfile_{retrieved_at:%Y%m%dT%H%M%SZ}.geojson"
            path.write_text(
                json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8"
            )
            self.project.provenance.log(
                dataset=self.layer,
                source=self.source,
                identifier=item.identifier,
                url=item.url,
                retrieved_at=retrieved_at,
                license=self.license,
                operation="fetch",
                parameters={
                    "cql_filter": item.parameters["CQL_FILTER"],
                    "n_features": len(features),
                    "file": f"raw/{self.name}/{path.name}",
                },
            )
            files.append(path)
        return files

    def load(self, files: list[Path]) -> gpd.GeoDataFrame:
        gdf = gpd.GeoDataFrame(
            pd.concat([gpd.read_file(f, engine="pyogrio") for f in files], ignore_index=True)
        )
        if gdf.empty:
            return gdf.set_crs(self.project.config.crs, allow_override=True)
        # The database pads text fields to fixed width.
        for col in gdf.columns.drop(gdf.geometry.name):
            gdf[col] = gdf[col].map(lambda v: v.strip() if isinstance(v, str) else v)
        gdf = gdf.set_crs("EPSG:4326", allow_override=True).to_crs(self.project.config.crs)
        aoi = self.project.read_aoi().union_all()
        return gdf[gdf.intersects(aoi)].reset_index(drop=True)

    def _get(self, url: str) -> bytes:
        """HTTP GET; the single network call, replaced in tests."""
        request = urllib.request.Request(url, headers={"User-Agent": "gossan"})
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()
