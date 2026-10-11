"""Shared engine for connectors that pull a vector layer from a web service.

A jurisdiction's connectors (BC's DataBC WFS, the USGS MRDS WFS, the BLM MLRS ArcGIS REST
service, ...) differ in how they *ask* and in what the records are called, and agree on
everything else. This base class holds the part that agrees:

``search``  (subclass)  builds the request(s) for the AOI.
``download`` (subclass) performs them and returns the raw bytes, however many pages it took.
``fetch``   (here)      saves those bytes under ``raw/<name>/`` with a timestamped name that is
                        never overwritten, and logs the retrieval to provenance.
``load``    (here)      reads the raw files, strips fixed-width padding, reprojects to the project
                        CRS, keeps what intersects the AOI, then calls ``normalize``.
``normalize`` (subclass) maps the source's field names onto the canonical layer schema
                        (``gossan.core.schemas``), which is what ``leads``, ``target`` and
                        ``report`` read. BC's layers already use that schema, so for them it
                        does nothing.

A connector's ``name`` is what ``gossan harvest --source`` takes and the folder under ``raw/``;
its ``layer`` is what it writes to ``data.gpkg`` (the same unless ``canonical_layer`` is set).
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd

from gossan.core.rawstore import store_raw
from gossan.harvest.base import Connector, SearchItem

USER_AGENT = "gossan"


@dataclass
class Download:
    """What one request (or one paged series of requests) returned."""

    content: bytes
    extension: str  # file extension for the raw copy, e.g. "geojson" or "gml"
    n_features: int
    parameters: dict = field(default_factory=dict)  # extra provenance parameters


class VectorServiceConnector(Connector):
    """Base class for web-service vector connectors; see the module docstring."""

    #: "bc", "us" or "any": which projects run this connector by default.
    jurisdiction: str = "any"
    #: Layer written to ``data.gpkg`` when it differs from ``name`` (US sources write the same
    #: ``minfile`` and ``mto_tenure`` layers the BC ones do).
    canonical_layer: str | None = None

    @property
    def layer(self) -> str:  # type: ignore[override]
        return self.canonical_layer or self.name

    # --- to be defined by subclasses ----------------------------------------------------

    def download(self, item: SearchItem) -> Download:
        raise NotImplementedError

    def normalize(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Map the source's fields onto the canonical schema (identity by default)."""
        return gdf

    def read_raw(self, path: Path) -> gpd.GeoDataFrame | None:
        """Read one raw file; None if it holds no features."""
        gdf = gpd.read_file(path, engine="pyogrio")
        return gdf if len(gdf) else None

    # --- shared -------------------------------------------------------------------------

    def fetch(self, items: list[SearchItem]) -> list[Path]:
        raw = self.project.raw_dir / self.name
        raw.mkdir(parents=True, exist_ok=True)
        files = []
        for item in items:
            retrieved_at = datetime.now(UTC)
            result = self.download(item)
            # Timestamped name; originals are never overwritten, and an identical download reuses
            # the file already there (see gossan.core.rawstore).
            stored = store_raw(raw, self.name, result.extension, result.content, retrieved_at)
            path = stored.path
            self.project.provenance.log(
                dataset=self.layer,
                source=self.source,
                identifier=item.identifier,
                url=item.url,
                retrieved_at=retrieved_at,
                license=self.license,
                operation="fetch",
                parameters={
                    **result.parameters,
                    "n_features": result.n_features,
                    "file": f"raw/{self.name}/{path.name}",
                    "sha256": stored.sha256,
                    "unchanged": stored.unchanged,
                },
            )
            files.append(path)
        return files

    def load(self, files: list[Path]) -> gpd.GeoDataFrame:
        crs = self.project.config.crs
        frames = [f for f in (self.read_raw(p) for p in files) if f is not None]
        if not frames:
            return gpd.GeoDataFrame(geometry=[], crs=crs)
        gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True))
        # Some databases pad text fields to a fixed width.
        for col in gdf.columns.drop(gdf.geometry.name):
            gdf[col] = gdf[col].map(lambda v: v.strip() if isinstance(v, str) else v)
        gdf = gdf.set_crs("EPSG:4326", allow_override=True).to_crs(crs)
        aoi = self.project.read_aoi().union_all()
        gdf = gdf[gdf.intersects(aoi)].reset_index(drop=True)
        if gdf.empty:
            return gdf
        return self.normalize(gdf)

    def _get(self, url: str) -> bytes:
        """HTTP GET; the single network call, replaced in tests."""
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()
