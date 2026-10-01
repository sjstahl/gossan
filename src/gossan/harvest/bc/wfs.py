"""Shared logic for layers served by the DataBC public WFS.

Endpoint: ``https://openmaps.gov.bc.ca/geo/pub/wfs`` (no authentication). Features are
requested as GeoJSON in EPSG:4326 with a bounding-box CQL filter around the AOI, paged
by ``OBJECTID``, and then selected against the AOI polygon locally (whole features that
intersect the AOI are kept, not cut at its edge).
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
PAGE_SIZE = 5000
OGL_BC = "Open Government Licence - British Columbia"


class WfsConnector(Connector):
    """A vector layer from the DataBC WFS. Subclasses set the class attributes below.

    ``name`` is also the ``data.gpkg`` layer and the folder under ``raw/``.
    """

    #: WFS feature type, e.g. ``pub:WHSE_MINERAL_TENURE.MINFIL_MINERAL_FILE``.
    type_name: str
    #: Name of the layer's geometry property (differs between layers).
    geometry_column: str

    @property
    def layer(self) -> str:  # type: ignore[override]
        return self.name

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
            "CQL_FILTER": f"BBOX({self.geometry_column},{bbox},'EPSG:4326')",
        }
        url = f"{WFS_URL}?{urlencode(params)}"
        return [SearchItem(identifier=self.type_name, url=url, parameters=params)]

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
            path = raw / f"{self.name}_{retrieved_at:%Y%m%dT%H%M%S%fZ}.geojson"
            with path.open("x", encoding="utf-8") as f:  # "x": fail rather than overwrite
                json.dump({"type": "FeatureCollection", "features": features}, f)
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
        crs = self.project.config.crs
        frames = [gpd.read_file(f, engine="pyogrio") for f in files if _has_features(f)]
        if not frames:
            return gpd.GeoDataFrame(geometry=[], crs=crs)
        gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True))
        # The databases pad some text fields to fixed width.
        for col in gdf.columns.drop(gdf.geometry.name):
            gdf[col] = gdf[col].map(lambda v: v.strip() if isinstance(v, str) else v)
        gdf = gdf.set_crs("EPSG:4326", allow_override=True).to_crs(crs)
        aoi = self.project.read_aoi().union_all()
        return gdf[gdf.intersects(aoi)].reset_index(drop=True)

    def _get(self, url: str) -> bytes:
        """HTTP GET; the single network call, replaced in tests."""
        request = urllib.request.Request(url, headers={"User-Agent": "gossan"})
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()


def _has_features(path: Path) -> bool:
    return bool(json.loads(path.read_text(encoding="utf-8"))["features"])
