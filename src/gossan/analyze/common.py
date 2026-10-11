"""Shared plumbing for the analysis steps: reading layers, the output folder, provenance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import pyogrio

from gossan.core.licensing import derived_license_from
from gossan.core.project import Project

ANALYSIS_DIR = "analysis"
ANALYSIS_GPKG = "analysis.gpkg"


class AnalysisError(Exception):
    """An analysis step cannot run; the message says why (usually a missing layer)."""


def analysis_dir(project: Project) -> Path:
    out = project.root / ANALYSIS_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out


def layer_names(project: Project) -> list[str]:
    gpkg = project.data_gpkg
    if not gpkg.exists():
        return []
    return [str(n) for n in pyogrio.list_layers(gpkg)[:, 0]]


def read_layer(project: Project, name: str, *, required: bool = False) -> gpd.GeoDataFrame | None:
    """A ``data.gpkg`` layer in the project CRS; None (or ``AnalysisError`` if ``required``) when
    the layer is absent or empty."""
    if name in layer_names(project):
        gdf = gpd.read_file(project.data_gpkg, layer=name, engine="pyogrio")
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
        if len(gdf):
            return gdf.to_crs(project.config.crs).reset_index(drop=True)
    if required:
        raise AnalysisError(f"no {name} layer in data.gpkg (run 'gossan harvest --source {name}')")
    return None


def write_layer(project: Project, gdf: gpd.GeoDataFrame, layer: str) -> str:
    """Write ``gdf`` as ``layer`` of ``analysis/analysis.gpkg`` (replacing an earlier run's)."""
    path = analysis_dir(project) / ANALYSIS_GPKG
    gdf.to_file(path, layer=layer, driver="GPKG", engine="pyogrio")
    return f"{ANALYSIS_DIR}/{ANALYSIS_GPKG}:{layer}"


def write_table(project: Project, df: pd.DataFrame, name: str) -> str:
    path = analysis_dir(project) / f"{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    return f"{ANALYSIS_DIR}/{path.name}"


def write_json(project: Project, data: dict[str, Any], name: str) -> str:
    path = analysis_dir(project) / f"{name}.json"
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return f"{ANALYSIS_DIR}/{path.name}"


def log_outputs(
    project: Project, step: str, outputs: list[str], inputs: list[str], parameters: dict[str, Any]
) -> None:
    """One provenance record per output of an analysis step, licensed from its inputs."""
    licence = derived_license_from(project.provenance.read(), inputs)
    for output in outputs:
        project.provenance.log(
            dataset=f"analysis_{step}",
            source="gossan",
            identifier=output,
            url=None,
            license=licence,
            operation=f"analyze_{step}",
            parameters={"inputs": inputs, **parameters},
        )

