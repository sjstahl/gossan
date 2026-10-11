"""Data inventory: what the project holds, how much of the AOI it covers, and under which terms.

One row per ``data.gpkg`` layer and per raster in ``rasters/``. Coverage is measured inside the AOI:
the share of its area for polygon layers, features per 100 km2 for points and km per 100 km2 for
lines, and the share of non-nodata cells for rasters. Source, licence and retrieval date come from
the latest provenance record for the dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask

from gossan.analyze.common import (
    layer_names,
    log_outputs,
    read_layer,
    write_table,
)
from gossan.core.project import Project
from gossan.core.raster import RASTERS_DIR


@dataclass
class InventoryResult:
    table: pd.DataFrame
    outputs: list[str]


def _provenance_by_dataset(project: Project) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for r in project.provenance.read():
        if r.operation in {"fetch", "manual_ingest", "write_raster"}:
            out[r.dataset] = {
                "source": r.source,
                "license": r.license,
                "retrieved_at": r.retrieved_at.date().isoformat(),
            }
    return out


def build_inventory(project: Project) -> InventoryResult:
    aoi = project.read_aoi().union_all()
    aoi_km2 = aoi.area / 1e6
    prov = _provenance_by_dataset(project)
    rows = []
    for name in layer_names(project):
        gdf = read_layer(project, name)
        row: dict[str, object] = {"dataset": name, "kind": "vector", "n_features": 0}
        if gdf is not None:
            kinds = set(gdf.geom_type)
            inside = gdf.geometry.intersection(aoi)
            row["n_features"] = len(gdf)
            row["geometry"] = "/".join(sorted(kinds))
            if kinds <= {"Polygon", "MultiPolygon"}:
                row["coverage"] = f"{min(inside.union_all().area / 1e6 / aoi_km2, 1.0):.1%} of AOI"
            elif kinds <= {"LineString", "MultiLineString"}:
                row["coverage"] = f"{inside.length.sum() / 1e3 / aoi_km2 * 100:.1f} km per 100 km2"
            else:
                row["coverage"] = f"{(gdf.within(aoi)).sum() / aoi_km2 * 100:.2f} per 100 km2"
        rows.append({**row, **prov.get(name, {})})

    raster_dir = project.root / RASTERS_DIR
    for path in sorted(raster_dir.glob("*.tif")) if raster_dir.is_dir() else []:
        with rasterio.open(path) as ds:
            data = ds.read(1, masked=True)
            in_aoi = ~geometry_mask([aoi], out_shape=ds.shape, transform=ds.transform)
            valid = (~np.ma.getmaskarray(data)) & in_aoi
            share = valid.sum() / max(in_aoi.sum(), 1)
            rows.append(
                {
                    "dataset": path.stem,
                    "kind": "raster",
                    "n_features": ds.count,
                    "geometry": f"{ds.res[0]:.0f} m grid",
                    "coverage": f"{share:.0%} of AOI",
                    **prov.get(path.stem, {}),
                }
            )
    columns = ["dataset", "kind", "n_features", "geometry", "coverage", "source", "license", "retrieved_at"]
    table = pd.DataFrame(rows).reindex(columns=columns)
    outputs = [write_table(project, table, "inventory")]
    log_outputs(project, "inventory", outputs, list(table["dataset"]), {"n_datasets": len(table)})
    return InventoryResult(table, outputs)
