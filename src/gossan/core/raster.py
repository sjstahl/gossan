"""Raster handling: clip to the AOI, reproject to the project CRS, write a COG."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rioxarray
from rioxarray.exceptions import NoDataInBounds, OneDimensionalRaster

from gossan.core.project import Project

#: Gridded formats ``gossan ingest`` accepts (all read through GDAL). Geophysical grids arrive as
#: GeoTIFF, ER Mapper (.ers, with its data file), GXF or ASCII grids; Geosoft binary .grd is not
#: readable by GDAL and must be exported to GeoTIFF first (a Surfer .grd is read).
RASTER_SUFFIXES = {".tif", ".tiff", ".ers", ".gxf", ".asc", ".grd", ".img"}
RASTERS_DIR = "rasters"


class RasterError(Exception):
    """The raster cannot be used; the message says why."""


def raster_path(project: Project, name: str) -> Path:
    """Where the COG named ``name`` lives."""
    return project.root / RASTERS_DIR / f"{name}.tif"


def write_cog(
    project: Project, src: Path, dst: Path, buffer_m: float = 0.0, assign_crs: str | None = None
) -> dict[str, object]:
    """Clip ``src`` to the project AOI (grown by ``buffer_m`` metres), reproject to the project CRS
    and write a COG at ``dst``.

    ``assign_crs`` (for example ``EPSG:3978``) is used when the file has no CRS of its own: many
    geophysical grids do not. Returns facts about the result for provenance. Raises ``RasterError`` if the raster
    has no CRS or does not overlap the AOI.
    """
    try:
        da = rioxarray.open_rasterio(src, masked=False)
    except Exception as e:
        hint = " A Geosoft .grd cannot be read: export it to GeoTIFF in QGIS or Geosoft." if src.suffix.lower() == ".grd" else ""
        raise RasterError(f"cannot read {src.name} as a grid: {e}.{hint}") from e
    if da.rio.crs is None:
        if not assign_crs:
            raise RasterError(f"{src.name} has no CRS: give it with --crs (for example EPSG:3978 or EPSG:4269), or assign one in QGIS")
        da = da.rio.write_crs(assign_crs)
    source_crs = da.rio.crs.to_string()
    if da.rio.nodata is None:  # the data standard requires nodata to be defined
        da = da.rio.write_nodata(_default_nodata(da.dtype))

    aoi = project.read_aoi()
    if buffer_m > 0:
        aoi = aoi.set_geometry(aoi.buffer(buffer_m))
    try:
        # Clip in the source CRS first (cheap), then reproject the small result.
        da = da.rio.clip(aoi.to_crs(da.rio.crs).geometry, all_touched=True, drop=True)
    except (NoDataInBounds, OneDimensionalRaster) as e:
        raise RasterError(f"{src.name} does not overlap the AOI; nothing written") from e

    resampling = 1 if np.issubdtype(da.dtype, np.floating) else 0  # bilinear / nearest
    da = da.rio.reproject(project.config.crs, resampling=resampling)

    dst.parent.mkdir(parents=True, exist_ok=True)
    da.rio.to_raster(dst, driver="COG", compress="DEFLATE")
    return {
        "source_crs": source_crs,
        "project_crs": project.config.crs,
        "bands": int(da.sizes["band"]),
        "shape": [int(da.sizes["y"]), int(da.sizes["x"])],
        "dtype": str(da.dtype),
        "nodata": float(da.rio.nodata),
        "resampling": "bilinear" if resampling == 1 else "nearest",
        "aoi_buffer_m": buffer_m,
        "crs_assigned": assign_crs if assign_crs and source_crs == assign_crs else None,
        "format": "COG",
    }


def _default_nodata(dtype: np.dtype) -> float:
    if np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.signedinteger):
        return -9999
    return float(np.iinfo(dtype).max)
