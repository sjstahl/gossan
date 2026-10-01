"""Manual ingest of a local vector file into ``data.gpkg`` (fallback for failed harvests)."""

from __future__ import annotations

import hashlib
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import pyogrio

from gossan.core.project import Project

LAYER_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class IngestError(Exception):
    """The file cannot be ingested; the message says why."""


def ingest_vector(
    project: Project,
    path: Path,
    layer: str,
    *,
    source: str = "user-supplied",
    license: str = "unspecified (user-supplied)",
    overwrite: bool = False,
) -> int:
    """Clip ``path`` to the AOI, reproject to the project CRS and write it as ``layer``.

    A copy of the original goes to ``raw/ingest/<layer>/`` (with shapefile sidecars)
    and a provenance record is written. Returns the number of features written.
    """
    if not LAYER_NAME.match(layer):
        raise IngestError(f"invalid layer name {layer!r}: use letters, digits and underscores")
    if not path.is_file():
        raise IngestError(f"{path} is not a file")
    if (
        project.data_gpkg.exists()
        and not overwrite
        and layer in pyogrio.list_layers(project.data_gpkg)[:, 0]
    ):
        raise IngestError(f"layer {layer!r} already exists in data.gpkg (use --overwrite)")

    gdf = gpd.read_file(path, engine="pyogrio")
    if gdf.crs is None:
        raise IngestError(f"{path.name} has no CRS; assign one (e.g. in QGIS) and retry")
    n_read = len(gdf)
    source_crs = gdf.crs.to_string()
    gdf = gdf[~gdf.geometry.isna() & ~gdf.geometry.is_empty].to_crs(project.config.crs)
    clipped = gdf.clip(project.read_aoi().union_all())
    clipped = clipped[~clipped.geometry.is_empty].reset_index(drop=True)
    if clipped.empty:
        raise IngestError(f"no features in {path.name} intersect the AOI; nothing written")

    now = datetime.now(UTC)
    raw = project.raw_dir / "ingest" / layer / f"{now:%Y%m%dT%H%M%S%fZ}"
    raw.mkdir(parents=True)
    copies = list(path.parent.glob(f"{path.stem}.*")) if path.suffix.lower() == ".shp" else [path]
    for f in copies:  # a shapefile is several files sharing a stem
        shutil.copy2(f, raw / f.name)

    clipped.to_file(project.data_gpkg, layer=layer, driver="GPKG", engine="pyogrio")
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
            "source_crs": source_crs,
            "project_crs": project.config.crs,
            "n_read": n_read,
            "n_written": len(clipped),
            "clipped_to_aoi": True,
        },
    )
    return len(clipped)
