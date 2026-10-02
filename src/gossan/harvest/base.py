"""Connector interface shared by all data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd

from gossan.core.project import Project


@dataclass(frozen=True)
class SearchItem:
    """One retrievable unit found by ``Connector.search``."""

    identifier: str
    url: str
    parameters: dict[str, Any] = field(default_factory=dict)


class Connector(ABC):
    """A data source.

    Before writing a connector, confirm the access method and terms of use and
    record them in the subclass docstring. Prefer official APIs and bulk
    downloads over scraping.

    ``fetch`` and ``run`` write provenance records; ``raw/`` is only ever added
    to, never modified.
    """

    #: Short name used by ``gossan harvest --source``.
    name: str
    #: Layer written to ``data.gpkg`` (or the raster product name).
    layer: str
    #: Who publishes the data, and under which licence (recorded in provenance).
    source: str
    license: str
    #: Whether a plain ``gossan harvest`` runs this connector.
    default: bool = True
    #: Which projects run it by default: "bc", "us", or "any" (works anywhere).
    jurisdiction: str = "any"
    #: How the CLI describes ``run()``'s count.
    output_description: str = "feature(s) written to data.gpkg"

    def __init__(self, project: Project) -> None:
        self.project = project

    @abstractmethod
    def search(self, aoi: gpd.GeoDataFrame) -> list[SearchItem]:
        """Find what the source holds for the AOI (project CRS)."""

    @abstractmethod
    def fetch(self, items: list[SearchItem]) -> list[Path]:
        """Download items into ``raw/`` and log each retrieval to provenance."""

    @abstractmethod
    def load(self, files: list[Path]) -> gpd.GeoDataFrame | list[Path]:
        """Turn fetched files into project data.

        Vector sources return a GeoDataFrame in the project CRS, selected to the AOI.
        Raster sources write COGs under ``rasters/`` and return their paths.
        """

    def run(self) -> int:
        """Search, fetch, load and write the layer to ``data.gpkg``; return the feature count.

        An empty result is logged but no layer is written.
        """
        aoi = self.project.read_aoi()
        files = self.fetch(self.search(aoi))
        gdf = self.load(files)
        if not gdf.empty:
            gdf.to_file(self.project.data_gpkg, layer=self.layer, driver="GPKG", engine="pyogrio")
        self.project.provenance.log(
            dataset=self.layer,
            source=self.source,
            identifier=self.layer,
            url=None,
            license=self.license,
            operation="write_layer",
            parameters={
                "gpkg": self.project.data_gpkg.name,
                "crs": self.project.config.crs,
                "n_features": len(gdf),
                "inputs": [f.name for f in files],
            },
        )
        return len(gdf)
