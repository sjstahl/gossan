"""Manual ingest of a local vector or raster file (fallback for failed harvests)."""

from __future__ import annotations

import hashlib
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import pyogrio

from gossan.core.project import Project
from gossan.core.raster import RASTER_SUFFIXES, RasterError, raster_path, write_cog

LAYER_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
MULTI_FILE_SUFFIXES = {".shp", ".ers", ".asc", ".gxf"}


class IngestError(Exception):
    """The file cannot be ingested; the message says why."""


def ingest(
    project: Project,
    path: Path,
    layer: str,
    *,
    source: str = "user-supplied",
    license: str = "unspecified (user-supplied)",
    overwrite: bool = False,
    crs: str | None = None,
    buffer_m: float = 0.0,
) -> str:
    """Ingest a vector file into ``data.gpkg`` or a raster grid into ``rasters/``.

    Either way the data are clipped to the AOI and put in the project CRS, a copy of
    the original (with its companion files) goes to ``raw/ingest/<layer>/`` and a provenance record
    is written. ``crs`` is used when the file has none; ``buffer_m`` keeps a margin of that many
    metres around the AOI in a raster (derivatives such as magnetic tilt need one). Returns a
    one-line description of the result.
    """
    if not LAYER_NAME.match(layer):
        raise IngestError(f"invalid name {layer!r}: use letters, digits and underscores")
    if not path.is_file():
        raise IngestError(f"{path} is not a file")
    if path.suffix.lower() in RASTER_SUFFIXES:
        return _ingest_raster(project, path, layer, source, license, overwrite, crs, buffer_m)
    if buffer_m:
        raise IngestError("--buffer-m applies to raster grids only")
    return _ingest_vector(project, path, layer, source, license, overwrite, crs)


def _ingest_vector(
    project: Project, path: Path, layer: str, source: str, license: str, overwrite: bool, crs: str | None = None
) -> str:
    if (
        project.data_gpkg.exists()
        and not overwrite
        and layer in pyogrio.list_layers(project.data_gpkg)[:, 0]
    ):
        raise IngestError(f"layer {layer!r} already exists in data.gpkg (use --overwrite)")

    gdf = gpd.read_file(path, engine="pyogrio")
    if gdf.crs is None:
        if not crs:
            raise IngestError(f"{path.name} has no CRS: give it with --crs (for example EPSG:4326), or assign one in QGIS")
        gdf = gdf.set_crs(crs)
    n_read = len(gdf)
    source_crs = gdf.crs.to_string()
    gdf = gdf[~gdf.geometry.isna() & ~gdf.geometry.is_empty].to_crs(project.config.crs)
    clipped = gdf.clip(project.read_aoi().union_all())
    clipped = clipped[~clipped.geometry.is_empty].reset_index(drop=True)
    if clipped.empty:
        raise IngestError(f"no features in {path.name} intersect the AOI; nothing written")

    now, raw = _copy_original(project, path, layer)
    clipped.to_file(project.data_gpkg, layer=layer, driver="GPKG", engine="pyogrio")
    facts = {
        "kind": "vector",
        "source_crs": source_crs,
        "n_read": n_read,
        "n_written": len(clipped),
        "clipped_to_aoi": True,
    }
    _log(project, path, layer, source, license, now, raw, facts)
    return f"{layer}: {len(clipped)} feature(s) written to data.gpkg"


def _ingest_raster(
    project: Project, path: Path, layer: str, source: str, license: str, overwrite: bool,
    crs: str | None = None, buffer_m: float = 0.0,
) -> str:  # fmt: skip
    dst = raster_path(project, layer)
    if dst.exists() and not overwrite:
        raise IngestError(f"raster {dst.name!r} already exists in rasters/ (use --overwrite)")
    try:
        facts = write_cog(project, path, dst, buffer_m=buffer_m, assign_crs=crs)
    except RasterError as e:
        raise IngestError(str(e)) from e

    now, raw = _copy_original(project, path, layer)
    output = dst.relative_to(project.root).as_posix()
    facts = {"kind": "raster", "output": output, "clipped_to_aoi": True, **facts}
    _log(project, path, layer, source, license, now, raw, facts)
    return f"{layer}: COG written to {output}"


def _copy_original(project: Project, path: Path, layer: str) -> tuple[datetime, Path]:
    """Keep an untouched copy under ``raw/ingest/<layer>/<timestamp>/``."""
    now = datetime.now(UTC)
    raw = project.raw_dir / "ingest" / layer / f"{now:%Y%m%dT%H%M%S%fZ}"
    raw.mkdir(parents=True)
    if path.suffix.lower() in MULTI_FILE_SUFFIXES:
        # Companion files share the stem: .dbf/.prj for a shapefile, .prj for an ASCII grid, and for
        # an ER Mapper header the data file with no extension at all.
        copies = [f for f in path.parent.iterdir() if f.is_file() and (f.stem == path.stem or f.name == path.stem)]
    else:
        copies = [path]
    for f in copies:
        shutil.copy2(f, raw / f.name)
    return now, raw


def _log(
    project: Project,
    path: Path,
    layer: str,
    source: str,
    license: str,
    now: datetime,
    raw: Path,
    parameters: dict[str, Any],
) -> None:
    project.provenance.log(
        dataset=layer,
        source=source,
        identifier=path.name,
        url=None,
        retrieved_at=now,
        license=license,
        operation="manual_ingest",
        parameters={
            "original_path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "raw_copy": raw.relative_to(project.root).as_posix(),
            "project_crs": project.config.crs,
            **parameters,
        },
    )
