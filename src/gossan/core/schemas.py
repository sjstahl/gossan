"""The canonical layer schemas that ``leads``, ``target`` and ``report`` read.

Gossan grew up on BC data, so the canonical names are BC's: the ``minfile`` layer (mineral
occurrence points) and the ``mto_tenure`` layer (mining tenure polygons), with the column names
those datasets use. A connector for another jurisdiction does not teach the rest of Gossan a
new vocabulary; it maps its source onto these columns and writes the same layers. Each source's
own extra fields may be kept as additional columns, but the columns below must exist.

``minfile`` (points)
    MINFILE_NUMBER (text id), MINFILE_NAME1, STATUS_DESCRIPTION, COMMODITY_DESCRIPTION1 to 3,
    DEPOSIT_TYPE_DESCRIPTION1, LATITUDE_DECIMAL, LONGITUDE_DECIMAL, MINFILE_SUMMARY_URL.
    ``target`` uses only the geometry; ``report`` also reads COMMODITY_DESCRIPTION1.

``mto_tenure`` (polygons)
    TENURE_NUMBER_ID (integer, one per tenure), CLAIM_NAME, TENURE_TYPE_DESCRIPTION ("Mineral",
    "Placer", ...), ISSUE_DATE and GOOD_TO_DATE (datetimes; missing where the source has none),
    AREA_IN_HECTARES, OWNER_NAME. ``leads`` and ``report`` read all of these; a source that lacks
    one leaves it empty, and the consumers treat empty as unknown.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd

MINFILE_LAYER = "minfile"
TENURE_LAYER = "mto_tenure"

MINFILE_COLUMNS: dict[str, str] = {
    "MINFILE_NUMBER": "str",
    "MINFILE_NAME1": "str",
    "STATUS_DESCRIPTION": "str",
    "COMMODITY_DESCRIPTION1": "str",
    "COMMODITY_DESCRIPTION2": "str",
    "COMMODITY_DESCRIPTION3": "str",
    "DEPOSIT_TYPE_DESCRIPTION1": "str",
    "LATITUDE_DECIMAL": "float",
    "LONGITUDE_DECIMAL": "float",
    "MINFILE_SUMMARY_URL": "str",
}
TENURE_COLUMNS: dict[str, str] = {
    "TENURE_NUMBER_ID": "int",
    "CLAIM_NAME": "str",
    "TENURE_TYPE_DESCRIPTION": "str",
    "ISSUE_DATE": "datetime",
    "GOOD_TO_DATE": "datetime",
    "AREA_IN_HECTARES": "float",
    "OWNER_NAME": "str",
}
SCHEMAS = {MINFILE_LAYER: MINFILE_COLUMNS, TENURE_LAYER: TENURE_COLUMNS}
GEOMETRY_TYPES = {MINFILE_LAYER: {"Point"}, TENURE_LAYER: {"Polygon", "MultiPolygon"}}


def conform(gdf: gpd.GeoDataFrame, layer: str) -> gpd.GeoDataFrame:
    """``gdf`` with the canonical columns of ``layer`` first (added empty if absent) and typed,
    followed by whatever other columns it already had."""
    schema = SCHEMAS[layer]
    out = gdf.copy()
    for col, kind in schema.items():
        if col not in out:
            out[col] = pd.NA
        if kind == "str":
            out[col] = out[col].astype("object").where(out[col].notna(), None)
            out[col] = out[col].map(lambda v: None if v is None else str(v))
        elif kind == "float":
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
        elif kind == "int":
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
        elif kind == "datetime":
            out[col] = pd.to_datetime(out[col], errors="coerce", utc=True)
    extras = [c for c in out.columns if c not in schema and c != out.geometry.name]
    return out[[*schema, *extras, out.geometry.name]]


def schema_problems(gdf: gpd.GeoDataFrame, layer: str) -> list[str]:
    """What is wrong with ``gdf`` as a ``layer`` (empty list if it satisfies the schema)."""
    problems = []
    missing = [c for c in SCHEMAS[layer] if c not in gdf]
    if missing:
        problems.append(f"missing column(s): {', '.join(missing)}")
    kinds = set(gdf.geometry.geom_type.dropna())
    if not kinds <= GEOMETRY_TYPES[layer]:
        problems.append(f"geometry types {sorted(kinds)} (expected {sorted(GEOMETRY_TYPES[layer])})")
    if layer == TENURE_LAYER and "TENURE_NUMBER_ID" in gdf:
        ids = gdf["TENURE_NUMBER_ID"]
        if ids.isna().any():
            problems.append("TENURE_NUMBER_ID has empty values")
        if not str(ids.dtype).lower().startswith("int"):
            problems.append(f"TENURE_NUMBER_ID is {ids.dtype}, not integer")
    return problems
