"""Pydantic models for ``project.yaml``."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

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
