"""Stacking harvested data into features on the prediction grid.

Each builder looks for one kind of data in the project and returns a feature (one value per
in-AOI cell) or a reason it was skipped. Missing data is normal, since not every AOI has every
layer: a builder that finds nothing usable skips itself and ``build_stack`` carries on with what
is left. Features:

Evidence of mineralization (always used when the data exist):

``dist_fault``, ``dist_thrust``, ``dist_contact``, ``dist_intrusive``, ``unit_age_ma``
    Structure and lithology from ``faults`` and ``bedrock_geology``: distance (m) to the nearest
    fault, thrust fault, map-unit contact and intrusive body (0 inside one), and the midpoint age
    of the bedrock unit. Faults and contacts outside the grid count (the grid is padded by
    ``EDGE_PAD_M``), so cells near the AOI edge are not given false large distances.
``elevation``, ``slope``, ``relief``
    Terrain from ``rasters/dem.tif`` (``gossan harvest --source dem``): elevation (m), slope
    (degrees) and local relief (m, range within ``RELIEF_WINDOW_M``).
``rgs_idx_<model>``, ``rgs_pc<n>`` (``geochem_features: indices``, the default)
    Stream-sediment geochemistry scored as in ``gossan analyze`` (deposit-model pathfinder
    indices corrected for scavenging and, with a DEM, catchment size and lithology; and the first
    ``geochem_components`` compositional principal components). With ``geochem_spatial:
    catchment`` (the default) and ``rasters/dem.tif``, each cell takes the score of the sample
    its water drains to, which is what a stream sample measures; cells no sample drains, and
    every cell without a DEM, use inverse-distance weighting (``idw_neighbours`` nearest samples
    within ``idw_radius_m``; cells with no sample that close are missing).
``rgs_<element>`` (``geochem_features: elements``)
    The raw ``geochem_elements`` instead, log-scaled and interpolated the same way; for each
    element the analysis column with the most values is used (at least five samples).
``<raster name>``
    Every other single-band raster in ``rasters/`` (regional magnetics and gravity from
    ``gossan harvest`` / ``gossan ingest``), bilinearly resampled to the grid. Georeferenced ARIS
    maps (``aris_*``) and multi-band images are not features and are skipped.

Exploration effort (see ``effort`` in the target config):

``tenure_density``
    Share of the surrounding area (``density_radius_m``) held under mining tenure (``mto_tenure``).
``aris_density``
    Assessment reports per km², smoothed with a Gaussian kernel (``aris_index``).

These measure where people have looked, not where deposits are, and known occurrences sit where
people looked. As features (``effort: feature``) they dominate the model and the map redraws
known camps. By default (``effort: ignore``) they are not used. ``effort: bias`` instead sets how
often each cell is drawn as unlabeled background (``FeatureStack.bias``), so occurrences are
contrasted with ground that was explored about as much ("target-group background", a standard
correction for sampling bias in presence-only modelling); it is opt-in because it scored worst on
the East Kootenay test area. With no evidence layers at all, effort becomes the features again
(and the stack says so) rather than leaving the model nothing to learn from.

Label-derived (rebuilt for every training run from the training occurrences only):

``unit_favourability`` (``unit_favourability: true``, the default)
    How strongly the cell's bedrock unit (``bedrock_geology``) hosts training occurrences
    relative to its area: ``log((n + 1) / (expected + 1))``, where ``expected`` is the number of
    training occurrences the unit would hold if they fell at random by area. The +1 keeps small
    or empty units near zero instead of at minus infinity (a weights-of-evidence-style contrast,
    shrunk toward "no information"). A known occurrence does not count toward its own unit.
``dist_minfile`` (``occurrence_distance: true`` only)
    Distance to the nearest known occurrence. Because the occurrences are also the training
    labels, this feature is *derived from the labels* and is rebuilt for every training run:
    a known occurrence is measured to its nearest *other* occurrence (never to itself), and in
    cross-validation only the training-fold occurrences count. It mostly encodes "near a known
    camp", so it is off by default.

Positives are the MINFILE occurrences in the AOI, optionally only one commodity class
(``occurrence_class``: metallic, industrial or specialty, from the BC deposit-profile code).

After building, features with more than ``max_missing_fraction`` of cells missing, or with no
variation, are dropped; the remaining gaps are filled with the feature's median.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import pyogrio
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import reproject
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    maximum_filter,
    minimum_filter,
    uniform_filter,
)
from scipy.spatial import cKDTree

from gossan.analyze.common import AnalysisError
from gossan.analyze.geochem import score_project
from gossan.analyze.occurrences import commodity_class
from gossan.core.config import TargetConfig
from gossan.core.project import Project
from gossan.target.grid import Grid

LABEL_FEATURE = "dist_minfile"
UNIT_FEATURE = "unit_favourability"
MIN_GEOCHEM_SAMPLES = 5
EFFORT_FEATURES = ("tenure_density", "aris_density")
#: Faults and contacts this far outside the grid still count when measuring distances.
EDGE_PAD_M = 5000.0
RELIEF_WINDOW_M = 500.0
DEM_RASTER = "dem"
#: Unlabeled-draw weight of a cell with no recorded effort, relative to the mean effort.
EFFORT_FLOOR = 0.1


class FeatureError(Exception):
    """The features cannot be built; the message says why."""


@dataclass
class FeatureStack:
    names: list[str]
    X: np.ndarray  # (n_cells, n_features) float32, gaps filled; excludes the label feature
    #: Fraction of cells that were missing before filling, per feature.
    missing_fraction: dict[str, float]
    skipped: dict[str, str]  # feature or source -> reason
    positives: np.ndarray  # compact cell indices of known occurrences (unique)
    n_occurrences_total: int
    n_occurrences_outside: int  # not in an in-AOI cell
    sources: dict[str, str] = field(default_factory=dict)
    #: Per-cell weight for drawing unlabeled cells (exploration effort), or None for uniform.
    bias: np.ndarray | None = None
    #: Exploration-effort weight of every cell (whatever ``effort`` is), for effort-matched CV
    #: metrics; None when there are no effort layers.
    effort_weights: np.ndarray | None = None
    #: Whether the label-derived distance feature is appended (``occurrence_distance``).
    use_label_feature: bool = True
    occurrence_class: str = "all"
    #: Bedrock unit code of every cell (-1: none) when ``unit_favourability`` is used, else None.
    units: np.ndarray | None = None
    #: Datasets behind the geochemistry features (rgs, plus dem and bedrock_geology for catchments).
    geochem_inputs: list[str] = field(default_factory=list)

    @property
    def model_names(self) -> list[str]:
        """Feature names in the order the model sees them (label-derived features last)."""
        return [
            *self.names,
            *([UNIT_FEATURE] if self.units is not None else []),
            *([LABEL_FEATURE] if self.use_label_feature else []),
        ]


# --- helpers ----------------------------------------------------------------------------


def _layer(project: Project, name: str) -> gpd.GeoDataFrame | None:
    gpkg = project.data_gpkg
    if not gpkg.exists() or name not in {str(n) for n in pyogrio.list_layers(gpkg)[:, 0]}:
        return None
    gdf = gpd.read_file(gpkg, layer=name, engine="pyogrio")
    return gdf if len(gdf) else None


def idw(
    grid: Grid, x: np.ndarray, y: np.ndarray, values: np.ndarray, k: int, radius: float, power: float = 2.0
) -> np.ndarray:
    """Inverse-distance-weighted interpolation at the in-AOI cell centres.

    Uses up to ``k`` samples within ``radius``; cells with none get NaN. A cell centre that
    coincides with a sample takes its value exactly.
    """
    tree = cKDTree(np.column_stack([x, y]))
    k = min(k, len(values))
    dist, idx = tree.query(grid.xy, k=k, distance_upper_bound=radius)
    dist, idx = dist.reshape(len(grid.xy), -1), idx.reshape(len(grid.xy), -1)
    valid = np.isfinite(dist)
    vals = np.where(valid, values[np.minimum(idx, len(values) - 1)], 0.0)
    with np.errstate(divide="ignore"):
        w = np.where(valid, 1.0 / np.maximum(dist, 1e-9) ** power, 0.0)
    total = w.sum(axis=1)
    out = np.full(len(grid.xy), np.nan)
    ok = total > 0
    out[ok] = (w[ok] * vals[ok]).sum(axis=1) / total[ok]
    return out


def distance_to_positives(grid: Grid, positives: np.ndarray, exclude_self: bool) -> np.ndarray:
    """Distance (m) from every cell to the nearest cell in ``positives`` (compact indices).

    With ``exclude_self`` a positive cell is measured to its nearest *other* positive, which is
    how the label-derived feature is built for training cells.
    """
    xy = grid.xy
    tree = cKDTree(xy[positives])
    if exclude_self:
        if len(positives) < 2:
            raise FeatureError("need at least two occurrences to measure distances between them")
        d, _ = tree.query(xy, k=2)
        out = d[:, 0].copy()
        is_pos = np.zeros(len(xy), dtype=bool)
        is_pos[positives] = True
        out[is_pos] = d[is_pos, 1]
        return out
    d, _ = tree.query(xy, k=1)
    return d


def unit_favourability(units: np.ndarray, positives: np.ndarray) -> np.ndarray:
    """``log((n + 1) / (expected + 1))`` per cell for its unit, from ``positives`` (compact cell
    indices); each positive cell leaves itself out. Cells with no unit (-1) get 0."""
    n_units = int(units.max()) + 1 if (units >= 0).any() else 0
    out = np.zeros(len(units), dtype=np.float64)
    if n_units == 0:
        return out
    mapped = units >= 0
    area_share = np.bincount(units[mapped], minlength=n_units) / mapped.sum()
    pos_units = units[positives]
    pos_units = pos_units[pos_units >= 0]
    counts = np.bincount(pos_units, minlength=n_units).astype(np.float64)
    n = len(pos_units)
    fav = np.log((counts + 1) / (n * area_share + 1))
    out[mapped] = fav[units[mapped]]
    own = positives[units[positives] >= 0]
    if n > 1 and len(own):
        u = units[own]
        out[own] = np.log((counts[u] - 1 + 1) / ((n - 1) * area_share[u] + 1))
    return out


def assemble_matrix(stack: FeatureStack, grid: Grid, train_positives: np.ndarray) -> np.ndarray:
    """The model matrix: static features, plus the label-derived features (unit favourability,
    distance to occurrences) built from ``train_positives`` when the stack uses them."""
    columns = [stack.X]
    if stack.units is not None:
        columns.append(unit_favourability(stack.units, train_positives).astype(np.float32)[:, None])
    if stack.use_label_feature:
        columns.append(distance_to_positives(grid, train_positives, exclude_self=True).astype(np.float32)[:, None])
    return np.hstack(columns) if len(columns) > 1 else stack.X


# --- builders (each returns a compact vector, or None with a reason) ---------------------


def _tenure_density(project: Project, grid: Grid, cfg: TargetConfig) -> tuple[np.ndarray | None, str]:
    tenure = _layer(project, "mto_tenure")
    if tenure is None:
        return None, "no mto_tenure layer (run 'gossan harvest --source mto_tenure')"
    geoms = [g for g in tenure.to_crs(grid.crs).geometry if g is not None and not g.is_empty]
    if not geoms:
        return None, "mto_tenure has no geometries"
    presence = rasterize(
        [(g, 1) for g in geoms], out_shape=grid.shape, transform=grid.transform, fill=0, all_touched=True, dtype="uint8"
    ).astype(np.float32)
    size = max(1, round(2 * cfg.density_radius_m / grid.resolution) | 1)
    return grid.compact(uniform_filter(presence, size=size, mode="nearest")), ""


def _aris_density(project: Project, grid: Grid, cfg: TargetConfig) -> tuple[np.ndarray | None, str]:
    aris = _layer(project, "aris_index")
    if aris is None:
        return None, "no aris_index layer (run 'gossan harvest --source aris_index')"
    pts = aris.to_crs(grid.crs).geometry
    pts = pts[~pts.is_empty & pts.notna()]
    rows = np.floor((grid.transform.f - pts.y.to_numpy()) / grid.resolution).astype(int)
    cols = np.floor((pts.x.to_numpy() - grid.transform.c) / grid.resolution).astype(int)
    ok = (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)
    counts = np.zeros(grid.shape, dtype=np.float32)
    np.add.at(counts, (rows[ok], cols[ok]), 1.0)
    sigma = max(0.5, cfg.density_radius_m / grid.resolution / 2)
    per_km2 = gaussian_filter(counts, sigma=sigma, mode="nearest") * (1e6 / grid.resolution**2)
    return grid.compact(per_km2.astype(np.float32)), ""


def _geochem(project: Project, grid: Grid, cfg: TargetConfig) -> dict[str, tuple[np.ndarray | None, str]]:
    rgs = _layer(project, "rgs")
    if rgs is None:
        return {"rgs": (None, "no rgs layer (run 'gossan harvest --source rgs')")}
    rgs = rgs.to_crs(grid.crs)
    rgs = rgs[~rgs.geometry.is_empty & rgs.geometry.notna()]
    out: dict[str, tuple[np.ndarray | None, str]] = {}
    for element in cfg.geochem_elements:
        name = f"rgs_{element.lower()}"
        columns = [
            c for c in rgs.columns
            if c.upper().startswith(f"{element.upper()}_") and c.upper().endswith(("_PPM", "_PPB", "_PERCENTAGE"))
        ]  # fmt: skip
        if not columns:
            out[name] = (None, f"no {element} analysis column in the rgs layer")
            continue
        best = max(columns, key=lambda c: rgs[c].notna().sum())
        vals = rgs[best].to_numpy(dtype=float)
        keep = np.isfinite(vals)
        if keep.sum() < MIN_GEOCHEM_SAMPLES:
            out[name] = (None, f"only {int(keep.sum())} {element} value(s) ({best}); need {MIN_GEOCHEM_SAMPLES}")
            continue
        logv = np.log1p(np.maximum(vals[keep], 0.0))
        pts = rgs.geometry[keep]
        out[name] = (
            idw(grid, pts.x.to_numpy(), pts.y.to_numpy(), logv, cfg.idw_neighbours, cfg.idw_radius_m),
            "",
        )
    return out


def distance_to_geometries(grid: Grid, geoms: list, pad_m: float = EDGE_PAD_M) -> np.ndarray:
    """Distance (m) from every in-AOI cell centre to the nearest of ``geoms`` (0 on or inside one),
    on a grid padded by ``pad_m`` so features just outside the AOI count."""
    pad = int(np.ceil(pad_m / grid.resolution))
    shape = (grid.height + 2 * pad, grid.width + 2 * pad)
    t = grid.transform
    transform = from_origin(t.c - pad * grid.resolution, t.f + pad * grid.resolution, grid.resolution, grid.resolution)
    hit = rasterize([(g, 1) for g in geoms], out_shape=shape, transform=transform, fill=0, all_touched=True, dtype="uint8")
    if not hit.any():
        return np.full(grid.n_cells, np.nan)
    dist = distance_transform_edt(hit == 0) * grid.resolution
    return grid.compact(dist[pad : pad + grid.height, pad : pad + grid.width])


def _structure(project: Project, grid: Grid) -> dict[str, tuple[np.ndarray | None, str]]:
    out: dict[str, tuple[np.ndarray | None, str]] = {}
    faults = _layer(project, "faults")
    if faults is None:
        out["dist_fault"] = (None, "no faults layer (run 'gossan harvest --source faults')")
    else:
        faults = faults.to_crs(grid.crs)
        out["dist_fault"] = (distance_to_geometries(grid, list(faults.geometry)), "")
        if "FAULT_TYPE" in faults:
            thrusts = faults[faults["FAULT_TYPE"].astype(str).str.contains("thrust", case=False)]
            out["dist_thrust"] = (
                (distance_to_geometries(grid, list(thrusts.geometry)), "") if len(thrusts) else (None, "no thrust faults")
            )
    bedrock = _layer(project, "bedrock_geology")
    if bedrock is None:
        out["bedrock"] = (None, "no bedrock_geology layer (run 'gossan harvest --source bedrock_geology')")
        return out
    bedrock = bedrock.to_crs(grid.crs)
    out["dist_contact"] = (distance_to_geometries(grid, list(bedrock.geometry.boundary)), "")
    if "ROCK_CLASS" in bedrock:
        intrusive = bedrock[bedrock["ROCK_CLASS"].astype(str).str.contains("intrusive", case=False)]
        out["dist_intrusive"] = (
            (distance_to_geometries(grid, list(intrusive.geometry)), "") if len(intrusive) else (None, "no intrusive units mapped")
        )
    if {"MAXIMUM_AGE_VALUE", "MINIMUM_AGE_VALUE"} <= set(bedrock.columns):
        hi = bedrock["MAXIMUM_AGE_VALUE"].astype(float)
        lo = bedrock["MINIMUM_AGE_VALUE"].astype(float)
        age = ((hi + lo) / 2).where(hi < 4000)  # 4000 Ma / 0 Ma is the "age unknown" code
        shapes_ = [(g, a) for g, a in zip(bedrock.geometry, age, strict=True) if np.isfinite(a)]
        if shapes_:
            arr = rasterize(shapes_, out_shape=grid.shape, transform=grid.transform, fill=np.nan, dtype="float32")
            out["unit_age_ma"] = (grid.compact(arr), "")
        else:
            out["unit_age_ma"] = (None, "no dated units")
    return out


def unit_codes(project: Project, grid: Grid) -> np.ndarray | None:
    """Integer code of each cell's bedrock unit (same unit name, same code; -1 where unmapped),
    or None without a ``bedrock_geology`` layer."""
    from gossan.analyze.occurrences import (
        unit_name,  # local: analyze imports stay one-way at load
    )

    bedrock = _layer(project, "bedrock_geology")
    if bedrock is None:
        return None
    bedrock = bedrock.to_crs(grid.crs)
    names = bedrock.apply(unit_name, axis=1)
    codes = {n: i for i, n in enumerate(sorted(set(names)))}
    shapes_ = [(g, codes[n]) for g, n in zip(bedrock.geometry, names, strict=True) if g is not None and not g.is_empty]
    arr = rasterize(shapes_, out_shape=grid.shape, transform=grid.transform, fill=-1, dtype="int32")
    return grid.compact(arr).astype(np.int64)


def _resample(path, grid: Grid) -> np.ndarray | None:
    """Band 1 of a raster bilinearly resampled to the full grid (NaN where nodata), or None."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
        with rasterio.open(path) as ds:
            if ds.crs is None:
                return None
            dst = np.full(grid.shape, np.nan, dtype=np.float32)
            reproject(
                rasterio.band(ds, 1), dst,
                dst_transform=grid.transform, dst_crs=grid.crs,
                src_nodata=ds.nodata, dst_nodata=np.nan, resampling=Resampling.bilinear,
            )  # fmt: skip
    return dst


def _terrain(project: Project, grid: Grid) -> dict[str, tuple[np.ndarray | None, str]]:
    path = project.root / "rasters" / f"{DEM_RASTER}.tif"
    if not path.exists():
        return {"terrain": (None, "no rasters/dem.tif (run 'gossan harvest --source dem')")}
    dem = _resample(path, grid)
    if dem is None:
        return {"terrain": (None, "dem.tif has no CRS")}
    gy, gx = np.gradient(dem, grid.resolution)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    size = max(3, round(RELIEF_WINDOW_M / grid.resolution) | 1)
    filled = np.where(np.isfinite(dem), dem, np.nanmedian(dem))
    relief = maximum_filter(filled, size=size) - minimum_filter(filled, size=size)
    relief = np.where(np.isfinite(dem), relief, np.nan)
    return {
        "elevation": (grid.compact(dem), ""),
        "slope": (grid.compact(slope), ""),
        "relief": (grid.compact(relief), ""),
    }


def _geochem_indices(
    project: Project, grid: Grid, cfg: TargetConfig, inputs: list[str]
) -> dict[str, tuple[np.ndarray | None, str]]:
    """Sample scores spread over the grid: by catchment (each cell takes the score of the sample
    its water reaches) where the DEM allows, by IDW elsewhere. Datasets used go into ``inputs``."""
    rgs = _layer(project, "rgs")
    if rgs is None:
        return {"rgs": (None, "no rgs layer (run 'gossan harvest --source rgs')")}
    rgs = rgs.to_crs(grid.crs)
    rgs = rgs[~rgs.geometry.is_empty & rgs.geometry.notna()].reset_index(drop=True)
    try:
        project_scores = score_project(project, rgs)
    except AnalysisError as e:
        return {"rgs": (None, str(e))}
    scored = project_scores.result.samples
    inputs += project_scores.inputs
    labels = None
    if cfg.geochem_spatial == "catchment" and project_scores.catchments is not None:
        labels = project_scores.catchments.label_at(grid.xy[:, 0], grid.xy[:, 1])
    columns = [c for c in scored.columns if c.startswith("idx_")]
    columns += [f"PC{i}" for i in range(1, cfg.geochem_components + 1) if f"PC{i}" in scored]
    out: dict[str, tuple[np.ndarray | None, str]] = {}
    x, y = scored.geometry.x.to_numpy(), scored.geometry.y.to_numpy()
    for col in columns:
        vals = scored[col].to_numpy(dtype=float)
        keep = np.isfinite(vals)
        name = f"rgs_{col.lower()}"
        if keep.sum() < MIN_GEOCHEM_SAMPLES:
            out[name] = (None, f"only {int(keep.sum())} scored sample(s)")
            continue
        values = idw(grid, x[keep], y[keep], vals[keep], cfg.idw_neighbours, cfg.idw_radius_m)
        if labels is not None:
            drained = labels >= 0
            by_catchment = vals[np.where(drained, labels, 0)]
            values = np.where(drained & np.isfinite(by_catchment), by_catchment, values)
        out[name] = (values, "")
    return out


def _effort_bias(candidates: dict[str, tuple[np.ndarray | None, str]]) -> np.ndarray | None:
    """Unlabeled-draw weights from the effort features: each scaled to mean 1 and summed, plus a
    floor so unexplored ground is still drawn sometimes. None if no effort layer exists."""
    parts = []
    for name in EFFORT_FEATURES:
        values, _ = candidates.get(name, (None, ""))
        if values is None:
            continue
        v = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0)
        if v.mean() > 0:
            parts.append(v / v.mean())
    if not parts:
        return None
    effort = np.sum(parts, axis=0)
    return effort + EFFORT_FLOOR * effort.mean()


def _rasters(project: Project, grid: Grid) -> dict[str, tuple[np.ndarray | None, str]]:
    out: dict[str, tuple[np.ndarray | None, str]] = {}
    folder = project.root / "rasters"
    for path in sorted(folder.glob("*.tif")) if folder.is_dir() else []:
        name = path.stem
        if name.lower().startswith("aris_") or name == DEM_RASTER:
            continue  # georeferenced report maps are images; the DEM feeds the terrain features
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
            with rasterio.open(path) as ds:
                if ds.count != 1:
                    out[name] = (None, f"{ds.count} bands; only single-band rasters are used")
                    continue
                if ds.crs is None:
                    out[name] = (None, "no CRS")
                    continue
                dst = np.full(grid.shape, np.nan, dtype=np.float32)
                reproject(
                    rasterio.band(ds, 1), dst,
                    dst_transform=grid.transform, dst_crs=grid.crs,
                    src_nodata=ds.nodata, dst_nodata=np.nan, resampling=Resampling.bilinear,
                )  # fmt: skip
        out[name] = (grid.compact(dst), "")
    if not out:
        out["rasters"] = (None, "no rasters in rasters/")
    return out


def _positives(project: Project, grid: Grid, occurrence_class: str = "all") -> tuple[np.ndarray, int, int]:
    minfile = _layer(project, "minfile")
    if minfile is None:
        raise FeatureError("no minfile layer: the known occurrences are the training labels (run 'gossan harvest --source minfile')")
    if occurrence_class != "all":
        if "DEPOSIT_TYPE_CODE1" not in minfile:
            raise FeatureError(
                f"occurrence_class {occurrence_class!r} needs DEPOSIT_TYPE_CODE1 in the minfile layer; use 'all'"
            )
        minfile = minfile[minfile["DEPOSIT_TYPE_CODE1"].map(commodity_class) == occurrence_class]
    pts = minfile.to_crs(grid.crs).geometry
    pts = pts[~pts.is_empty & pts.notna()]
    cells = grid.cell_of(pts.x.to_numpy(), pts.y.to_numpy())
    inside = cells[cells >= 0]
    return np.unique(inside), len(pts), int((cells < 0).sum())


# --- the stack --------------------------------------------------------------------------

#: The ``data.gpkg`` layer (or raster) each built-in feature comes from.
FEATURE_DATASETS = {
    "tenure_density": "mto_tenure",
    "aris_density": "aris_index",
    "dist_fault": "faults",
    "dist_thrust": "faults",
    "dist_contact": "bedrock_geology",
    "dist_intrusive": "bedrock_geology",
    "unit_age_ma": "bedrock_geology",
    "elevation": DEM_RASTER,
    "slope": DEM_RASTER,
    "relief": DEM_RASTER,
}


def input_datasets(stack: FeatureStack) -> list[str]:
    """Every dataset the model used: the labels, each feature's source, and the effort layers when
    they weight the background."""
    used = ["minfile"]
    for name in stack.names:
        used.append("rgs" if name.startswith("rgs_") else FEATURE_DATASETS.get(name, name))
    if stack.units is not None:
        used.append("bedrock_geology")
    if any(n.startswith("rgs_") for n in stack.names):
        used += stack.geochem_inputs
    if stack.bias is not None:
        used += [FEATURE_DATASETS[n] for n in EFFORT_FEATURES]
    return list(dict.fromkeys(used))



def build_stack(project: Project, grid: Grid, cfg: TargetConfig) -> FeatureStack:
    """Build every available feature on ``grid``; skipped features are listed with reasons."""
    positives, n_total, n_outside = _positives(project, grid, cfg.occurrence_class)
    effort = {
        "tenure_density": _tenure_density(project, grid, cfg),
        "aris_density": _aris_density(project, grid, cfg),
    }
    geochem_inputs: list[str] = []
    geochem = (
        _geochem_indices(project, grid, cfg, geochem_inputs)
        if cfg.geochem_features == "indices"
        else _geochem(project, grid, cfg)
    )
    candidates: dict[str, tuple[np.ndarray | None, str]] = {
        **(effort if cfg.effort == "feature" else {}),
        **_structure(project, grid),
        **_terrain(project, grid),
        **geochem,
        **_rasters(project, grid),
    }
    skipped: dict[str, str] = {}
    bias = None
    effort_weights = _effort_bias(effort)
    evidence = [v for v, _ in candidates.values() if v is not None]
    if cfg.effort != "feature" and not evidence and any(v is not None for v, _ in effort.values()):
        # Nothing describes the ground itself (no geology, geochemistry, terrain or rasters):
        # rather than fail, model effort as before and say so.
        candidates = {**effort, **candidates}
        skipped["evidence"] = (
            "no evidence layers (faults, bedrock_geology, rgs, dem, rasters/): fell back to "
            "exploration effort as features, so the map mirrors where people have explored"
        )
    elif cfg.effort == "bias":
        bias = effort_weights
        for name in EFFORT_FEATURES:
            skipped[name] = "exploration effort: weights background sampling, not a feature (effort: bias)"
    elif cfg.effort == "ignore":
        for name in EFFORT_FEATURES:
            skipped[name] = "exploration effort: not used (effort: ignore)"
    names: list[str] = []
    columns: list[np.ndarray] = []
    missing: dict[str, float] = {}
    for name, (values, reason) in candidates.items():
        if values is None:
            skipped[name] = reason
            continue
        values = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(values)
        frac = 1.0 - finite.mean()
        if frac > cfg.max_missing_fraction:
            skipped[name] = f"{frac:.0%} of the AOI has no value (limit {cfg.max_missing_fraction:.0%})"
            continue
        if not finite.any() or np.nanmax(values) == np.nanmin(values):
            skipped[name] = "no variation across the AOI"
            continue
        values = np.where(finite, values, np.nanmedian(values))
        names.append(name)
        columns.append(values.astype(np.float32))
        missing[name] = round(float(frac), 4)
    X = np.column_stack(columns) if columns else np.empty((grid.n_cells, 0), dtype=np.float32)
    units = None
    if cfg.unit_favourability:
        units = unit_codes(project, grid)
        if units is None:
            skipped[UNIT_FEATURE] = "no bedrock_geology layer (run 'gossan harvest --source bedrock_geology')"
        elif len(np.unique(units[units >= 0])) < 2:
            skipped[UNIT_FEATURE] = "fewer than two bedrock units in the AOI"
            units = None
    return FeatureStack(
        names, X, missing, skipped, positives, n_total, n_outside,
        bias=bias, effort_weights=effort_weights, use_label_feature=cfg.occurrence_distance,
        occurrence_class=cfg.occurrence_class, units=units, geochem_inputs=geochem_inputs,
    )  # fmt: skip
