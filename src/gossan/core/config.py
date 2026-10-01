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
    georef: GeorefConfig = Field(default_factory=GeorefConfig)
    satellite: SatelliteConfig = Field(default_factory=SatelliteConfig)
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
