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


def utm_zones_spanned(gdf: gpd.GeoDataFrame) -> list[int]:
    """UTM zone numbers (1-60) of the AOI's western and eastern edges, ascending and distinct.

    More than one means the AOI straddles a zone boundary; the project then uses the zone of the
    centroid and distances far from that zone's central meridian are slightly distorted.
    """
    minx, _, maxx, _ = gdf.to_crs("EPSG:4326").total_bounds
    zones = {min(int((lon + 180) // 6) + 1, 60) for lon in (minx, maxx)}
    return sorted(zones)


def detect_jurisdiction(gdf: gpd.GeoDataFrame) -> str | None:
    """Jurisdiction ("bc" or "us") from the AOI centroid, or None where it is not clear.

    A coarse rule of thumb, not a boundary lookup. BC: 49 to 60 degrees north between its
    longitudes. US: the western United States (31 to 48.3 degrees north, 125 to 102 degrees west),
    where BLM claims exist and the two countries cannot be confused. Between 48.3 and 49 degrees the
    Gulf Islands and Vancouver Island (west of -122.9) count as BC and the rest as US. Everything
    else (Alaska, Yukon, the eastern US and Canada, other countries) returns None, and ``gossan new``
    then asks for ``--jurisdiction``.
    """
    c = gdf.to_crs("EPSG:4326").union_all().centroid
    lon, lat = c.x, c.y
    if 49.0 <= lat < 60.0 and -139.1 <= lon <= -114.0:
        return "bc"
    if 31.0 <= lat < 48.3 and -125.0 <= lon <= -102.0:
        return "us"
    if 48.3 <= lat < 49.0 and -125.5 <= lon <= -117.0:
        return "bc" if lon < -122.9 else "us"
    return None
