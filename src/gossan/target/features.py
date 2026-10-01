"""Stacking harvested data into features on the prediction grid.

Each builder looks for one kind of data in the project and returns a feature (one value per
in-AOI cell) or a reason it was skipped. Missing data is normal, since not every AOI has every
layer: a builder that finds nothing usable skips itself and ``build_stack`` carries on with what
is left. Features:

``tenure_density``
    Share of the surrounding area (``density_radius_m``) held under mining tenure
    (``mto_tenure``). A proxy for how much of the ground has been of interest to someone.
``aris_density``
    Assessment reports per km², smoothed with a Gaussian kernel (``aris_index``). Exploration
    effort, not mineralisation: it biases toward well-explored ground, and the model will learn
    that bias along with any real signal.
``rgs_<element>``
    Regional stream-sediment geochemistry (``rgs``), interpolated by inverse-distance weighting
    (log scale, ``idw_neighbours`` nearest samples within ``idw_radius_m``; cells with no sample
    that close are missing). For each element in ``geochem_elements`` the analysis column with
    the most values is used; an element needs at least five samples. ARIS reports do not carry
    geochemical values, so they cannot feed this.
``<raster name>``
    Every single-band raster in ``rasters/`` (regional magnetics and gravity from
    ``gossan harvest`` / ``gossan ingest``), bilinearly resampled to the grid. Georeferenced ARIS
    maps (``aris_*``) and multi-band images are not features and are skipped.
``dist_minfile``
    Distance to the nearest known occurrence. Because the occurrences are also the training
    labels, this feature is *derived from the labels* and is rebuilt for every training run:
    a known occurrence is measured to its nearest *other* occurrence (never to itself), and in
    cross-validation only the training-fold occurrences count. Otherwise the model would simply
    learn "distance zero means deposit".

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
from rasterio.warp import reproject
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.spatial import cKDTree

from gossan.core.config import TargetConfig
from gossan.core.project import Project
from gossan.target.grid import Grid

LABEL_FEATURE = "dist_minfile"
MIN_GEOCHEM_SAMPLES = 5


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

    @property
    def model_names(self) -> list[str]:
        """Feature names in the order the model sees them (label-derived feature last)."""
        return [*self.names, LABEL_FEATURE]


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


def assemble_matrix(stack: FeatureStack, grid: Grid, train_positives: np.ndarray) -> np.ndarray:
    """The model matrix: static features plus the label-derived distance for ``train_positives``."""
    dist = distance_to_positives(grid, train_positives, exclude_self=True).astype(np.float32)
    return np.column_stack([stack.X, dist])


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


def _rasters(project: Project, grid: Grid) -> dict[str, tuple[np.ndarray | None, str]]:
    out: dict[str, tuple[np.ndarray | None, str]] = {}
    folder = project.root / "rasters"
    for path in sorted(folder.glob("*.tif")) if folder.is_dir() else []:
        name = path.stem
        if name.lower().startswith("aris_"):
            continue  # georeferenced report maps: images, not features
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


def _positives(project: Project, grid: Grid) -> tuple[np.ndarray, int, int]:
    minfile = _layer(project, "minfile")
    if minfile is None:
        raise FeatureError("no minfile layer: the known occurrences are the training labels (run 'gossan harvest --source minfile')")
    pts = minfile.to_crs(grid.crs).geometry
    pts = pts[~pts.is_empty & pts.notna()]
    cells = grid.cell_of(pts.x.to_numpy(), pts.y.to_numpy())
    inside = cells[cells >= 0]
    return np.unique(inside), len(pts), int((cells < 0).sum())


# --- the stack --------------------------------------------------------------------------


def build_stack(project: Project, grid: Grid, cfg: TargetConfig) -> FeatureStack:
    """Build every available feature on ``grid``; skipped features are listed with reasons."""
    positives, n_total, n_outside = _positives(project, grid)
    candidates: dict[str, tuple[np.ndarray | None, str]] = {
        "tenure_density": _tenure_density(project, grid, cfg),
        "aris_density": _aris_density(project, grid, cfg),
        **_geochem(project, grid, cfg),
        **_rasters(project, grid),
    }
    names: list[str] = []
    columns: list[np.ndarray] = []
    missing: dict[str, float] = {}
    skipped: dict[str, str] = {}
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
    return FeatureStack(names, X, missing, skipped, positives, n_total, n_outside)
