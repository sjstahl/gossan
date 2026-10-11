"""Writing the target outputs: rasters, target polygons, importances, grid description."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.shutil import copy as rio_copy
from scipy.ndimage import gaussian_filter, label
from shapely.geometry import shape

from gossan.target.grid import Grid

NODATA = -9999.0
TARGET_DIR = "target"


def target_dir(root: Path) -> Path:
    return root / TARGET_DIR


def write_cog(
    path: Path, grid: Grid, bands: dict[str, np.ndarray], *, nodata: float = NODATA
) -> None:
    """Write named 2-D float arrays (NaN outside the AOI) as a Cloud Optimized GeoTIFF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(bands)
    data = np.stack([np.where(np.isfinite(bands[n]), bands[n], nodata) for n in names]).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_tif = Path(tmp) / "t.tif"
        with rasterio.open(
            tmp_tif, "w", driver="GTiff", width=grid.width, height=grid.height, count=len(names),
            dtype="float32", crs=grid.crs, transform=grid.transform, nodata=nodata, tiled=True,
            blockxsize=256, blockysize=256,
        ) as ds:  # fmt: skip
            ds.write(data)
            for i, n in enumerate(names, start=1):
                ds.set_band_description(i, n)
        rio_copy(tmp_tif, path, driver="COG", compress="DEFLATE")


def smooth_scores(grid: Grid, values: np.ndarray, sigma_cells: float, rows: np.ndarray | None = None) -> np.ndarray:
    """Gaussian-smoothed ``values`` (one per cell of ``rows``, default every in-AOI cell), using
    only those cells: a cell next to the AOI edge, or to a gap, is averaged over the neighbours it
    has instead of being pulled toward zero (normalized convolution)."""
    if sigma_cells <= 0:
        return values
    idx = np.arange(grid.n_cells) if rows is None else rows
    r, c = grid.rows_cols[0][idx], grid.rows_cols[1][idx]
    num = np.zeros(grid.shape)
    den = np.zeros(grid.shape)
    num[r, c] = values
    den[r, c] = 1.0
    num = gaussian_filter(num, sigma_cells, mode="constant")
    den = gaussian_filter(den, sigma_cells, mode="constant")
    return num[r, c] / np.maximum(den[r, c], 1e-12)


def top_percentile_polygons(
    grid: Grid,
    score: np.ndarray,
    spread: np.ndarray,
    positives: np.ndarray,
    percentile: float,
    *,
    smooth_cells: float = 0.0,
    min_cells: int = 0,
) -> tuple[gpd.GeoDataFrame, float]:
    """Polygons of the connected cells at or above the ``percentile`` of the score.

    With ``smooth_cells`` the percentile is taken on the score smoothed by that many cells
    (standard deviation), and polygons of fewer than ``min_cells`` cells are dropped; the scores
    reported per target are always the unsmoothed ones. Returns the targets (ranked by mean score,
    with area, scores, uncertainty and the number of known occurrences they contain) and the
    score threshold used (on the smoothed score if smoothing). ``attrs["n_dropped_small"]`` holds
    how many polygons the size limit removed.
    """
    selection = smooth_scores(grid, score, smooth_cells)
    threshold = float(np.percentile(selection, percentile))
    keep = selection >= threshold
    sc = grid.to_raster(score)
    sp = grid.to_raster(spread)
    known = np.zeros(grid.shape, dtype=bool)
    known[grid.rows_cols[0][positives], grid.rows_cols[1][positives]] = True
    selected = np.zeros(grid.shape, dtype=bool)
    selected[grid.mask] = keep
    labels, n = label(selected)
    n_dropped = 0
    if min_cells > 1 and n:
        sizes = np.bincount(labels.ravel())
        small = np.flatnonzero(sizes < min_cells)
        small = small[small > 0]
        n_dropped = len(small)
        labels = np.where(np.isin(labels, small), 0, labels)
        n -= n_dropped
    if n == 0:
        empty = gpd.GeoDataFrame({"target_id": []}, geometry=[], crs=grid.crs)
        empty.attrs["n_dropped_small"] = n_dropped
        return empty, threshold

    rows = []
    for geom_dict, value in shapes(labels.astype(np.int32), mask=labels > 0, transform=grid.transform):
        lab = int(value)
        cells = labels == lab
        rows.append(
            {
                "geometry": shape(geom_dict),
                "n_cells": int(cells.sum()),
                "area_ha": round(float(cells.sum()) * grid.resolution**2 / 1e4, 3),
                "mean_score": float(np.mean(sc[cells])),
                "max_score": float(np.max(sc[cells])),
                "mean_spread": float(np.mean(sp[cells])),
                "n_known_occurrences": int(known[cells].sum()),
            }
        )
    gdf = gpd.GeoDataFrame(rows, crs=grid.crs)
    gdf = gdf.sort_values("mean_score", ascending=False).reset_index(drop=True)
    gdf.insert(0, "rank", range(1, len(gdf) + 1))
    gdf.insert(1, "target_id", [f"T{i:03d}" for i in gdf["rank"]])
    gdf["percentile"] = percentile
    gdf["score_threshold"] = threshold
    gdf.attrs["n_dropped_small"] = n_dropped
    return gdf, threshold


def write_importances(path: Path, names: list[str], importances: np.ndarray) -> None:
    order = np.argsort(importances)[::-1]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "importance", "rank"])
        for rank, i in enumerate(order, start=1):
            w.writerow([names[i], f"{importances[i]:.6f}", rank])


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
