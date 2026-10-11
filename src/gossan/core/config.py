"""Pydantic models for ``project.yaml``."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

PROJECT_FILE = "project.yaml"
AOI_FILE = "aoi.gpkg"


class Module(StrEnum):
    """Gossan modules that can be enabled for a project."""

    HARVEST = "harvest"
    DIGITIZE = "digitize"
    LEADS = "leads"
    SPECTRAL = "spectral"
    TARGET = "target"
    REPORT = "report"


class GeorefConfig(BaseModel):
    """Settings for georeferencing scanned maps (``gossan digitize``, ``gossan ingest-points``)."""

    model_config = ConfigDict(extra="forbid")

    #: Reject a fit whose root-mean-square GCP error (metres) is above this.
    rms_threshold_m: float = Field(default=15.0, gt=0)
    #: Fewest GCPs the automatic step accepts (4 leaves one redundant, so the RMS means something).
    min_gcps: int = Field(default=4, ge=3)
    #: Datum printed on the maps' grid labels. Old ARIS maps are often NAD27; the labels are
    #: read in this datum's UTM zone (the project's zone) and the result is reprojected.
    datum: Literal["NAD83", "NAD27", "WGS84"] = "NAD83"
    #: Explicit CRS of the grid labels (e.g. "EPSG:26910"); overrides datum + UTM zone.
    gcp_crs: str | None = None
    #: Resolution maps are rendered at for georeferencing.
    dpi: int = Field(default=200, ge=72, le=600)


class SatelliteConfig(BaseModel):
    """Settings for the satellite scene connectors (``gossan harvest --source emit|enmap|aster``)."""

    model_config = ConfigDict(extra="forbid")

    #: STAC date or interval, e.g. "2023-01-01/2024-12-31" or "2022-06-01/..". Unset: all dates
    #: (ASTER is then capped at April 2008, when its SWIR detectors failed).
    datetime: str | None = None
    #: Drop scenes with more cloud than this (percent); scenes with no cloud value are kept.
    max_cloud_percent: float = Field(default=100.0, ge=0, le=100)
    emit_collection: str = "EMITL2ARFL_001"
    enmap_collection: str = "ENMAP_HSI_L2A"
    aster_collection: str = "AST_L1T_004"


class TargetConfig(BaseModel):
    """Settings for prospectivity modelling (``gossan target``). See ``gossan.target``."""

    model_config = ConfigDict(extra="forbid")

    #: Grid cell size in metres (``--resolution`` overrides it).
    resolution_m: float = Field(default=100.0, gt=0)
    #: Refuse a grid with more cells than this (bounding box); raise the resolution instead.
    max_cells: int = Field(default=2_000_000, ge=1)
    #: PU-bagging: number of bags, trees per bag, minimum samples per leaf.
    n_bags: int = Field(default=25, ge=2)
    n_trees: int = Field(default=100, ge=5)
    min_samples_leaf: int = Field(default=3, ge=1)
    #: Unlabeled cells drawn per positive in each bag.
    unlabeled_ratio: float = Field(default=1.0, gt=0)
    #: Spatial cross-validation folds, block edge (metres; default: a quarter of the AOI's
    #: shorter side) and the gap left between test blocks and training cells.
    folds: int = Field(default=5, ge=2)
    block_size_m: float | None = Field(default=None, gt=0)
    cv_buffer_m: float = Field(default=0.0, ge=0)
    #: Fewest known occurrences inside the AOI that a model may be trained on.
    min_positives: int = Field(default=10, ge=3)
    seed: int = 42
    #: Cells at or above this percentile of the probability surface become target polygons.
    top_percentile: float = Field(default=95.0, gt=0, lt=100)
    #: Neighbourhood for tenure density and the ARIS report-density kernel (metres).
    density_radius_m: float = Field(default=1000.0, gt=0)
    #: RGS geochemistry interpolation: search radius (metres) and neighbours.
    idw_radius_m: float = Field(default=10_000.0, gt=0)
    idw_neighbours: int = Field(default=8, ge=1)
    geochem_elements: list[str] = [
        "COPPER", "GOLD", "SILVER", "ZINC", "LEAD", "ARSENIC", "MOLYBDENUM", "ANTIMONY",
    ]  # fmt: skip
    #: A feature with more than this fraction of the AOI missing is dropped; the rest are
    #: filled with the feature's median.
    max_missing_fraction: float = Field(default=0.6, ge=0, le=1)
    #: Which known occurrences are the positives, by BC deposit-profile class: "all", "metallic",
    #: "industrial" or "specialty" (see ``gossan.analyze.occurrences``). Needs DEPOSIT_TYPE_CODE1.
    occurrence_class: Literal["all", "metallic", "industrial", "specialty"] = "all"
    #: Exploration effort (ARIS report density, tenure density). "ignore" (default): not used, so
    #: the map is built from geology, geochemistry and terrain alone. "feature": the two layers are
    #: model inputs (the pre-2026-10 behaviour; high plain AUC because occurrences sit where people
    #: explored, so the map largely redraws known camps). "bias": not features, but unlabeled cells
    #: are drawn in proportion to effort (target-group background); on the East Kootenay test
    #: area it scored worst of the three on effort-matched AUC, so it is opt-in.
    effort: Literal["bias", "feature", "ignore"] = "ignore"
    #: Add the label-derived distance to the nearest known occurrence as a feature. It mostly
    #: reproduces known camps, so it is off by default.
    occurrence_distance: bool = False
    #: Add ``unit_favourability``: how enriched each bedrock unit is in (training) occurrences,
    #: rebuilt from the training occurrences only in every cross-validation fold.
    unit_favourability: bool = True
    #: RGS geochemistry features: "indices" (default; deposit-model pathfinder indices and leading
    #: compositional principal components from ``gossan analyze``'s scoring) or "elements" (the
    #: raw ``geochem_elements``).
    geochem_features: Literal["indices", "elements"] = "indices"
    #: How many leading geochemical principal components become features ("indices" mode).
    geochem_components: int = Field(default=3, ge=0, le=6)


class ProjectConfig(BaseModel):
    """Contents of a project's ``project.yaml``."""

    name: str = Field(min_length=1)
    client: str | None = None
    crs: str = Field(description="Project CRS, e.g. 'EPSG:32610'.")
    utm_zone: int | None = Field(
        default=None, ge=1, le=60, description="UTM zone if the CRS is UTM, else null."
    )
    aoi_path: str = Field(default=AOI_FILE, description="AOI path relative to the project folder.")
    modules: list[Module] = Field(default_factory=lambda: [Module.HARVEST, Module.REPORT])
    #: Which jurisdiction's data sources a plain ``gossan harvest`` uses ("bc" or "us").
    jurisdiction: Literal["bc", "us"] = "bc"
    georef: GeorefConfig = Field(default_factory=GeorefConfig)
    satellite: SatelliteConfig = Field(default_factory=SatelliteConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _name_is_a_folder_name(cls, v: str) -> str:
        if v in {".", ".."} or any(c in v for c in '/\\:*?"<>|'):
            raise ValueError(f"{v!r} is not usable as a folder name")
        return v

    def save(self, project_dir: Path) -> Path:
        """Write ``project.yaml`` into ``project_dir``."""
        path = project_dir / PROJECT_FILE
        data = self.model_dump(mode="json")
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, project_dir: Path) -> ProjectConfig:
        """Read and validate ``project.yaml`` from ``project_dir``."""
        data = yaml.safe_load((project_dir / PROJECT_FILE).read_text(encoding="utf-8"))
        return cls.model_validate(data)
