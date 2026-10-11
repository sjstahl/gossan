"""Stream-sediment catchments from the DEM: what ground each RGS sample actually drains.

A stream-sediment sample is a composite of its whole upstream catchment, not of the ground around
the sample point. Interpolating samples by distance (IDW) smears an anomaly across ridges into
neighbouring valleys it cannot come from, and compares samples that drain very different rocks
and areas as if they were alike. This module routes water over the DEM and gives each sample its
catchment, so that:

* scores can be mapped onto the ground that produced them (each cell takes the score of the
  first sample downstream of it: the sample's *subcatchment*);
* each sample's *full catchment* (everything upstream, including other samples' subcatchments)
  gives its drainage area, to correct for dilution, and its bedrock-unit make-up, to compare its
  concentrations with what those rocks normally shed (``gossan.analyze.geochem``).

Method (all in numpy, no GIS dependency):

1. The DEM (``rasters/dem.tif``) is resampled to ``catchment_resolution_m`` by averaging.
2. Depressions are filled by priority-flood with a small gradient (Barnes, Lehman and Mulla,
   2014), so every cell drains to the DEM edge or to a nodata gap, including across flats.
3. D8 flow directions (steepest descent to one of eight neighbours) and flow accumulation.
4. Each sample is snapped to the nearest stream cell (draining at least ``stream_km2``) within
   ``snap_m``, because sample coordinates and DEM streams rarely coincide; with no stream cell in
   reach it takes the highest-flow cell. (Snapping to the highest flow outright drags samples
   from small creeks onto the trunk river beside them.) Cells are then labelled by the first
   sample downstream.

Catchments that reach the DEM edge or a nodata gap may be cut off (``catchment_truncated``): the
DEM is harvested 10 km beyond the AOI, so this mostly affects large rivers. A 30 m national DEM
resampled to 50 m resolves small streams approximately; where two streams run close together a
sample can snap to the wrong one, so check ``snap_dist_m`` and the catchment polygons in QGIS.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize, shapes
from rasterio.transform import Affine
from scipy.ndimage import binary_dilation
from shapely.geometry import shape
from shapely.ops import unary_union

#: D8 neighbours (row offset, column offset).
OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
#: Height added per step across a filled depression or flat, so water keeps moving.
EPSILON = 1e-4


@dataclass
class Catchments:
    transform: Affine
    crs: str
    resolution: float
    #: (rows, cols): index of the sample whose subcatchment holds the cell, -1 where none.
    labels: np.ndarray
    #: One row per input point (same order): snap distance, areas, downstream sample, truncation.
    samples: pd.DataFrame
    #: Bedrock unit names and each sample's full-catchment share of them (None without geology).
    unit_names: list[str] | None = None
    unit_shares: np.ndarray | None = None

    def label_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Sample index draining each point (-1 where none or outside the DEM)."""
        inv = ~self.transform
        cols, rows = inv * (np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        rows, cols = np.floor(rows).astype(int), np.floor(cols).astype(int)
        h, w = self.labels.shape
        ok = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        out = np.full(len(rows), -1, dtype=np.int64)
        out[ok] = self.labels[rows[ok], cols[ok]]
        return out


# --- hydrology ------------------------------------------------------------------------------


def load_dem(path: Path, resolution: float) -> tuple[np.ndarray, Affine, str]:
    """The DEM averaged to ``resolution`` (never finer than the source), NaN where nodata."""
    with rasterio.open(path) as ds:
        scale = min(ds.res[0] / resolution, 1.0)
        h, w = max(1, round(ds.height * scale)), max(1, round(ds.width * scale))
        dem = ds.read(1, out_shape=(h, w), resampling=Resampling.average, masked=True)
        transform = ds.transform * Affine.scale(ds.width / w, ds.height / h)
        crs = ds.crs.to_string()
    return np.ma.filled(dem.astype(np.float64), np.nan), transform, crs


def fill_depressions(dem: np.ndarray) -> np.ndarray:
    """Priority-flood fill with an epsilon gradient: afterwards every valid cell has a strictly
    lower neighbour on a path to the edge or to nodata. NaN stays NaN."""
    h, w = dem.shape
    z = dem.ravel()
    valid = np.isfinite(z)
    filled = z.copy()
    closed = (~valid).tolist()
    edge = np.zeros((h, w), dtype=bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    edge |= binary_dilation(~valid.reshape(h, w)) & valid.reshape(h, w)
    seeds = np.flatnonzero(edge.ravel() & valid)
    heap = list(zip(z[seeds].tolist(), seeds.tolist(), strict=True))
    heapq.heapify(heap)
    for i in seeds.tolist():
        closed[i] = True
    zl, out = z.tolist(), filled.tolist()
    push, pop = heapq.heappush, heapq.heappop
    while heap:
        e, i = pop(heap)
        r, c = divmod(i, w)
        for dr, dc in OFFSETS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                j = rr * w + cc
                if not closed[j]:
                    closed[j] = True
                    v = zl[j]
                    if v <= e:
                        v = e + EPSILON
                    out[j] = v
                    push(heap, (v, j))
    return np.asarray(out).reshape(h, w)


def flow_directions(filled: np.ndarray, resolution: float = 1.0) -> np.ndarray:
    """Flat index of each cell's steepest-descent D8 neighbour, -1 for outlets and nodata."""
    h, w = filled.shape
    padded = np.pad(filled, 1, constant_values=np.inf)
    best = np.zeros((h, w))
    best_k = np.full((h, w), -1, dtype=np.int64)
    with np.errstate(invalid="ignore"):
        for k, (dr, dc) in enumerate(OFFSETS):
            neighbour = padded[1 + dr : 1 + dr + h, 1 + dc : 1 + dc + w]
            drop = (filled - neighbour) / (resolution * np.hypot(dr, dc))
            better = drop > best
            best[better] = drop[better]
            best_k[better] = k
    rows, cols = np.indices((h, w))
    dr = np.array([o[0] for o in OFFSETS])[best_k]
    dc = np.array([o[1] for o in OFFSETS])[best_k]
    down = (rows + dr) * w + (cols + dc)
    down[best_k < 0] = -1
    return down.ravel()


def accumulate(filled: np.ndarray, down: np.ndarray) -> np.ndarray:
    """Number of cells draining through each cell (itself included); 0 for nodata."""
    z = filled.ravel()
    valid = np.isfinite(z)
    order = np.flatnonzero(valid)
    order = order[np.argsort(-z[order], kind="stable")].tolist()
    acc = valid.astype(np.float64).tolist()
    dl = down.tolist()
    for i in order:
        d = dl[i]
        if d >= 0:
            acc[d] += acc[i]
    return np.asarray(acc).reshape(filled.shape)


# --- catchments -----------------------------------------------------------------------------


def delineate(
    dem_path: Path,
    points: gpd.GeoSeries,
    *,
    resolution: float,
    snap_m: float,
    stream_km2: float = 0.5,
    units: gpd.GeoDataFrame | None = None,
) -> Catchments:
    """Catchments of ``points`` (project CRS) over the DEM. ``units``, if given, needs a ``unit``
    column of bedrock unit names; each sample's full-catchment share of each unit is computed."""
    dem, transform, crs = load_dem(dem_path, resolution)
    cell = abs(transform.a)
    h, w = dem.shape
    filled = fill_depressions(dem)
    down = flow_directions(filled, cell)
    acc = accumulate(filled, down)
    z = filled.ravel()
    n = len(points)

    # Snap each sample to the nearest stream cell within snap_m (else the highest-flow cell).
    stream_cells = stream_km2 * 1e6 / (cell * cell)
    inv = ~transform
    snap_cell = np.full(n, -1, dtype=np.int64)
    snap_dist = np.full(n, np.nan)
    k = int(np.ceil(snap_m / cell))
    for s, (px, py) in enumerate(zip(points.x, points.y, strict=True)):
        col, row = inv * (px, py)
        r0, c0 = int(np.floor(row)), int(np.floor(col))
        r_lo, r_hi, c_lo, c_hi = max(r0 - k, 0), min(r0 + k + 1, h), max(c0 - k, 0), min(c0 + k + 1, w)
        if r_lo >= r_hi or c_lo >= c_hi:
            continue
        rr, cc = np.mgrid[r_lo:r_hi, c_lo:c_hi]
        dist = np.hypot(rr + 0.5 - row, cc + 0.5 - col) * cell
        reach = (dist <= max(snap_m, cell / 2)) & np.isfinite(dem[rr, cc])
        if not reach.any():
            continue
        stream = reach & (acc[rr, cc] >= stream_cells)
        if stream.any():
            best = np.unravel_index(np.argmin(np.where(stream, dist, np.inf)), dist.shape)
        else:
            best = np.unravel_index(np.argmax(np.where(reach, acc[rr, cc], -1.0)), dist.shape)
        snap_cell[s] = rr[best] * w + cc[best]
        snap_dist[s] = dist[best]

    # Label cells by the first sample downstream (process downstream cells first).
    owner = {}
    for s, c in enumerate(snap_cell.tolist()):
        if c >= 0 and c not in owner:
            owner[c] = s
    valid = np.flatnonzero(np.isfinite(z))
    order = valid[np.argsort(z[valid], kind="stable")].tolist()
    lab = [-1] * (h * w)
    dl = down.tolist()
    for i in order:
        if i in owner:
            lab[i] = owner[i]
        else:
            d = dl[i]
            if d >= 0:
                lab[i] = lab[d]
    labels = np.asarray(lab, dtype=np.int64).reshape(h, w)

    # Sample tree: the next sample downstream of each snapped sample.
    alias = np.array([owner.get(c, -1) if c >= 0 else -1 for c in snap_cell.tolist()])  # duplicates
    downstream = np.full(n, -1, dtype=np.int64)
    for s in range(n):
        c = snap_cell[s]
        if c >= 0 and alias[s] == s and down[c] >= 0:
            downstream[s] = labels.ravel()[down[c]]
    tree_order = [s for s in np.argsort(-np.where(snap_cell >= 0, z[np.maximum(snap_cell, 0)], -np.inf)) if alias[s] == s]

    flat = labels.ravel()
    has = flat >= 0
    sub_cells = np.bincount(flat[has], minlength=n).astype(float)
    edge = np.zeros((h, w), dtype=bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    edge |= binary_dilation(~np.isfinite(dem))
    edge_hits = np.bincount(flat[has & edge.ravel()], minlength=n) > 0
    truncated = edge_hits.copy()
    for s in tree_order:
        d = downstream[s]
        if d >= 0:
            truncated[d] |= truncated[s]

    unit_names = unit_shares = None
    if units is not None and len(units):
        unit_names = sorted(set(units["unit"]))
        code = {u: i for i, u in enumerate(unit_names)}
        unit_raster = rasterize(
            [(g, code[u]) for g, u in zip(units.geometry, units["unit"], strict=True) if g is not None and not g.is_empty],
            out_shape=(h, w), transform=transform, fill=-1, dtype="int32",
        ).ravel()  # fmt: skip
        m = has & (unit_raster >= 0)
        sub = np.bincount(flat[m] * len(unit_names) + unit_raster[m], minlength=n * len(unit_names))
        full = sub.reshape(n, len(unit_names)).astype(float)
        for s in tree_order:
            d = downstream[s]
            if d >= 0:
                full[d] += full[s]
        totals = full.sum(axis=1, keepdims=True)
        unit_shares = np.divide(full, totals, out=np.full_like(full, np.nan), where=totals > 0)

    km2 = cell * cell / 1e6
    snapped = snap_cell >= 0
    full_area = np.where(snapped, acc.ravel()[np.maximum(snap_cell, 0)] * km2, np.nan)
    primary = np.where(alias >= 0, alias, np.arange(n))
    table = pd.DataFrame(
        {
            "snap_dist_m": np.round(snap_dist, 0),
            "catchment_area_km2": np.round(full_area, 3),
            "subcatchment_area_km2": np.where(snapped, np.round(sub_cells[primary] * km2, 3), np.nan),
            "downstream_sample": np.where(snapped, downstream[primary], -1),
            "catchment_truncated": np.where(snapped, truncated[primary], False),
            "shares_snap_cell_with": np.where(snapped & (alias != np.arange(n)), alias, -1),
        }
    )
    if unit_shares is not None:
        unit_shares = unit_shares[primary]
    return Catchments(transform, crs, cell, labels, table, unit_names, unit_shares)


def catchment_polygons(c: Catchments) -> gpd.GeoDataFrame:
    """One polygon per sample subcatchment, with the sample index in ``sample``."""
    pieces: dict[int, list] = {}
    for geom, value in shapes(c.labels.astype(np.int32), mask=c.labels >= 0, transform=c.transform):
        pieces.setdefault(int(value), []).append(shape(geom))
    rows = [{"sample": s, "geometry": unary_union(g)} for s, g in sorted(pieces.items())]
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=c.crs)
