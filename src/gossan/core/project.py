"""A project folder on disk (layout in PLAN.md section 5)."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from gossan.core.aoi import read_aoi
from gossan.core.config import PROJECT_FILE, ProjectConfig
from gossan.core.provenance import ProvenanceLog


class Project:
    """Paths and config for one client job."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        if not (self.root / PROJECT_FILE).is_file():
            raise FileNotFoundError(f"{self.root} is not a Gossan project (no {PROJECT_FILE})")
        self.config = ProjectConfig.load(self.root)
        self.provenance = ProvenanceLog(self.root)

    @property
    def aoi_path(self) -> Path:
        return self.root / self.config.aoi_path

    @property
    def data_gpkg(self) -> Path:
        return self.root / "data.gpkg"

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

    def read_aoi(self) -> gpd.GeoDataFrame:
        """AOI polygons in the project CRS."""
        return read_aoi(self.aoi_path).to_crs(self.config.crs)
