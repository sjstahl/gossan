"""AOI reading, CRS selection and writing."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd


def read_aoi(path: Path) -> gpd.GeoDataFrame:
    """Read an AOI vector file; require a CRS and at least one non-empty polygon."""
    gdf = gpd.read_file(path, engine="pyogrio")
    if gdf.crs is None:
        raise ValueError(f"{path} has no CRS; assign one before using it as an AOI")
    gdf = gdf[~gdf.geometry.isna() & ~gdf.geometry.is_empty]
    if gdf.empty or not gdf.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError(f"{path} must contain only polygon geometries (and at least one)")
    return gdf.reset_index(drop=True)


def utm_zone_for(gdf: gpd.GeoDataFrame) -> tuple[int, str]:
    """Return ``(zone, epsg)`` of the WGS84 UTM zone containing the AOI centroid."""
    centroid = gdf.to_crs("EPSG:4326").union_all().centroid
    zone = min(int((centroid.x + 180) // 6) + 1, 60)
    base = 32600 if centroid.y >= 0 else 32700
    return zone, f"EPSG:{base + zone}"


def write_aoi(gdf: gpd.GeoDataFrame, path: Path) -> None:
    """Write the AOI to a GeoPackage layer named ``aoi``."""
    gdf.to_file(path, layer="aoi", driver="GPKG", engine="pyogrio")
