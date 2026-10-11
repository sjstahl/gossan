"""ASTER band-ratio alteration indices from the scenes ``gossan harvest --source aster`` downloads.

This is the basic, dependable end of spectral work: published band ratios on at-sensor radiance, not a
spectral unmixing. For each scene in ``raw/satellite/aster/<scene id>/`` the VNIR (15 m) and SWIR (30 m)
band files are resampled onto one grid in the project CRS (``spectral.resolution_m``), converted from
digital numbers to radiance, and five indices are made where the bands exist:

==============  ======================  ======================================================
raster          formula (ASTER bands)   what a high value suggests
==============  ======================  ======================================================
aster_ferric    B2 / B1                 ferric iron (goethite, hematite, jarosite): gossans
aster_gossan    B4 / B2                 iron oxide against SWIR brightness
aster_ferrous   B5 / B3 + B1 / B2       ferrous iron (chlorite, epidote, some sulphide rinds)
aster_aloh      (B5 + B7) / B6          Al-OH absorption: white mica, clay (sericitic, argillic)
aster_mgoh      (B7 + B9) / B8          Mg-OH and carbonate: chlorite, epidote, carbonate
==============  ======================  ======================================================

Cells under vegetation (vegetation index above ``spectral.ndvi_max``), snow (snow index above
``spectral.snow_ndsi``, which needs SWIR) or with no data are removed, then the scenes are combined with
the median of the cells each has. Several scenes from different dates therefore fill each other's gaps
and drop single-date snow, cloud and noise.

**What this can and cannot do.** The ratios see bare rock and soil only, and most of British Columbia is
forested: outside alpine and burnt or logged ground the masks will remove most cells, and a feature with
less than ``max_missing_fraction`` of the AOI is dropped by ``gossan target`` with that reason. They are
computed on at-sensor radiance, with no atmospheric or solar-angle correction, so absolute values are not
comparable between projects and the summary reports each index's separation of known occurrences from
background instead of asserting that it works. The thermal bands (silica) are not used. SWIR scenes after
April 2008 do not exist (the detectors failed); later scenes give only the ferric index.

Digital numbers become radiance as ``(DN - 1) * UCC`` with the unit conversion coefficients of the ASTER
user handbook; the gain is a project setting because the GeoTIFFs do not say which was used.
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.warp import reproject

from gossan.analyze.common import read_layer, write_json
from gossan.core.licensing import derived_license_from
from gossan.core.project import Project

SCENE_DATASET = "aster"
RASTER_PREFIX = "aster_"
SUMMARY = "spectral_summary"

#: Radiance in W m-2 sr-1 um-1 per digital number, by band and gain (ASTER User Handbook, table of unit
#: conversion coefficients). The 3N band is used for band 3.
UCC: dict[str, dict[int, float]] = {
    "high": {1: 0.676, 2: 0.708, 3: 0.423, 4: 0.1087, 5: 0.0348, 6: 0.0313, 7: 0.0299, 8: 0.0209, 9: 0.0159},
    "normal": {1: 1.688, 2: 1.415, 3: 0.862, 4: 0.2174, 5: 0.0696, 6: 0.0625, 7: 0.0597, 8: 0.0417, 9: 0.0318},
    "low1": {1: 2.25, 2: 1.89, 3: 1.15, 4: 0.2900, 5: 0.0925, 6: 0.0830, 7: 0.0795, 8: 0.0556, 9: 0.0424},
}  # fmt: skip

#: raster name -> (label, formula, bands it needs)
INDICES: dict[str, tuple[str, str, tuple[int, ...]]] = {
    "ferric": ("Ferric iron (iron oxide)", "B2 / B1", (1, 2)),
    "gossan": ("Gossan (iron oxide / SWIR)", "B4 / B2", (2, 4)),
    "ferrous": ("Ferrous iron", "B5 / B3 + B1 / B2", (1, 2, 3, 5)),
    "aloh": ("Al-OH (white mica, clay)", "(B5 + B7) / B6", (5, 6, 7)),
    "mgoh": ("Mg-OH and carbonate", "(B7 + B9) / B8", (7, 8, 9)),
}  # fmt: skip

_BAND_FILE = re.compile(r"_(?:VNIR_B0?([12])|VNIR_B0?(3)N?|SWIR_B0?([4-9]))\.tif$", re.IGNORECASE)
MAX_AUC_BACKGROUND = 200_000
MIN_OCCURRENCES = 10


class SpectralError(Exception):
    """The indices cannot be built; the message says why."""


@dataclass
class SpectralResult:
    summary: dict
    outputs: list[str] = field(default_factory=list)


def scene_bands(folder: Path) -> dict[int, Path]:
    """The band files of one scene folder, by ASTER band number (VNIR 1, 2, 3N and SWIR 4 to 9)."""
    found: dict[int, Path] = {}
    for path in sorted(folder.glob("*.tif")):
        m = _BAND_FILE.search(path.name)
        if m:
            found[int(next(g for g in m.groups() if g))] = path
    return found


def find_scenes(project: Project) -> dict[str, dict[int, Path]]:
    """Scene id -> band files, for every scene folder in ``raw/satellite/aster`` with at least two bands."""
    root = project.raw_dir / "satellite" / SCENE_DATASET
    scenes = {}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
        bands = scene_bands(folder)
        if len(bands) >= 2:
            scenes[folder.name] = bands
    return scenes


def radiance(dn: np.ndarray, band: int, gain: str) -> np.ndarray:
    """At-sensor radiance from digital numbers; 0 is no data."""
    ucc = UCC[gain].get(band)
    if ucc is None:
        raise SpectralError(f"band {band} has no unit conversion coefficient at {gain} gain")
    return np.where(dn > 0, (dn - 1.0) * ucc, np.nan).astype(np.float32)


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    return out.astype(np.float32)


def _normalized(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(a + b > 0, (a - b) / (a + b), np.nan)
    return out.astype(np.float32)


def indices_from_radiance(rad: dict[int, np.ndarray], ndvi_max: float, snow_ndsi: float) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """(masked indices, bare-ground mask, valid-data mask) for one scene's radiances (bands present only).

    The bare-ground mask is True where there is data, no vegetation and no snow; the snow test needs the
    SWIR, so a VNIR-only scene masks vegetation alone.
    """
    shape = next(iter(rad.values())).shape
    valid = np.zeros(shape, dtype=bool)
    for r in rad.values():
        valid |= np.isfinite(r)
    bare = valid.copy()
    if 2 in rad and 3 in rad:
        bare &= ~(_normalized(rad[3], rad[2]) > ndvi_max)  # NaN compares False: a cell with no vegetation value stays
    if 1 in rad and 4 in rad:
        bare &= ~(_normalized(rad[1], rad[4]) > snow_ndsi)
    out: dict[str, np.ndarray] = {}
    b = rad
    if {1, 2} <= b.keys():
        out["ferric"] = _ratio(b[2], b[1])
    if {2, 4} <= b.keys():
        out["gossan"] = _ratio(b[4], b[2])
    if {1, 2, 3, 5} <= b.keys():
        out["ferrous"] = _ratio(b[5], b[3]) + _ratio(b[1], b[2])
    if {5, 6, 7} <= b.keys():
        out["aloh"] = _ratio(b[5] + b[7], b[6])
    if {7, 8, 9} <= b.keys():
        out["mgoh"] = _ratio(b[7] + b[9], b[8])
    return {k: np.where(bare, v, np.nan).astype(np.float32) for k, v in out.items()}, bare, valid


def _grid(project: Project, res: float) -> tuple[object, int, int]:
    minx, miny, maxx, maxy = project.read_aoi().total_bounds
    width, height = int(np.ceil((maxx - minx) / res)), int(np.ceil((maxy - miny) / res))
    return from_origin(minx, maxy, res, res), width, height


def _load_band(path: Path, transform: object, width: int, height: int, project: Project, res: float) -> np.ndarray:
    """One band file resampled onto the index grid; 0 (no data) where the scene does not reach."""
    dst = np.zeros((height, width), dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
        with rasterio.open(path) as ds:
            if ds.crs is None:
                raise SpectralError(f"{path.name} has no CRS")
            finer = abs(ds.transform.a) < 0.9 * res
            reproject(
                rasterio.band(ds, 1), dst, dst_transform=transform, dst_crs=project.config.crs,
                src_nodata=0, dst_nodata=0, resampling=Resampling.average if finer else Resampling.bilinear,
            )  # fmt: skip
    return dst


def _write(project: Project, name: str, data: np.ndarray, transform: object) -> Path:
    folder = project.root / "rasters"
    folder.mkdir(exist_ok=True)
    path = folder / f"{RASTER_PREFIX}{name}.tif"
    profile = {
        "driver": "GTiff", "height": data.shape[0], "width": data.shape[1], "count": 1, "dtype": "float32",
        "crs": project.config.crs, "transform": transform, "nodata": float("nan"),
        "tiled": True, "blockxsize": 256, "blockysize": 256, "compress": "deflate",
    }  # fmt: skip
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(data, 1)
    return path


def _auc(positive: np.ndarray, background: np.ndarray) -> float:
    """Probability that a random occurrence cell scores above a random background cell (ties half)."""
    from scipy.stats import rankdata

    both = np.concatenate([positive, background])
    ranks = rankdata(both)
    n1, n0 = len(positive), len(background)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _occurrence_stats(project: Project, rasters: dict[str, Path]) -> dict[str, dict]:
    """Per index: the share of the AOI with a value, and how well it separates metallic occurrences."""
    from gossan.analyze.occurrences import commodity_class

    aoi = project.read_aoi()
    minfile = read_layer(project, "minfile")
    points = None
    if minfile is not None and "DEPOSIT_TYPE_CODE1" in minfile:
        inside = minfile[minfile.within(aoi.union_all())]
        points = inside[inside["DEPOSIT_TYPE_CODE1"].map(commodity_class) == "metallic"]
    rng = np.random.default_rng(0)
    stats: dict[str, dict] = {}
    for name, path in rasters.items():
        with rasterio.open(path) as ds:
            data = ds.read(1, masked=False)
            transform = ds.transform
            inside_aoi = ~geometry_mask(aoi.geometry, out_shape=data.shape, transform=transform)
            finite = np.isfinite(data)
            n_aoi = int(inside_aoi.sum())
            row: dict = {"valid_pct": round(100.0 * float((finite & inside_aoi).sum()) / max(n_aoi, 1), 1)}
            background = data[finite & inside_aoi]
            if len(background) > MAX_AUC_BACKGROUND:
                background = rng.choice(background, MAX_AUC_BACKGROUND, replace=False)
            row["n_background"] = len(background)
            at = np.array([])
            if points is not None and len(points):
                at = np.array([v[0] for v in ds.sample([(g.x, g.y) for g in points.geometry])], dtype=float)
                at = at[np.isfinite(at)]
            row["n_occurrences"] = len(at)
            if len(at) >= MIN_OCCURRENCES and len(background) >= MIN_OCCURRENCES:
                row["auc"] = round(_auc(at, background), 3)
                row["median_occurrence"] = round(float(np.median(at)), 3)
                row["median_background"] = round(float(np.median(background)), 3)
            else:
                row["auc"] = None
        stats[name] = row
    return stats


def build_aster(project: Project, *, force: bool = False) -> SpectralResult:
    """Build the index rasters in ``rasters/`` from the scenes and write ``analysis/spectral_summary.json``.

    Raises ``SpectralError`` when there are no usable scenes. Rasters newer than every scene file are
    kept unless ``force``; the statistics are always recomputed.
    """
    cfg = project.config.spectral
    scenes = find_scenes(project)
    if not scenes:
        write_json(project, {"available": False, "reason": "no ASTER scenes in raw/satellite/aster (run 'gossan harvest --source aster --limit-scenes N')"}, SUMMARY)
        raise SpectralError("no ASTER scenes in raw/satellite/aster: run 'gossan harvest --source aster --limit-scenes 3' (needs EARTHDATA_TOKEN), or put the scene folders there by hand")
    files = [p for bands in scenes.values() for p in bands.values()]
    transform, width, height = _grid(project, cfg.resolution_m)
    target = {name: project.root / "rasters" / f"{RASTER_PREFIX}{name}.tif" for name in INDICES}
    newest = max(p.stat().st_mtime for p in files)
    stale = force or any(not p.exists() or p.stat().st_mtime < newest for p in target.values())

    per_scene: dict[str, list[np.ndarray]] = {name: [] for name in INDICES}
    bare_fractions: list[float] = []
    notes: list[str] = []
    used: list[str] = []
    if stale:
        for scene_id, bands in scenes.items():
            rad = {}
            for band, path in bands.items():
                if band not in UCC[cfg.gain]:
                    continue
                dn = _load_band(path, transform, width, height, project, cfg.resolution_m)
                rad[band] = radiance(dn, band, cfg.gain)
            if not rad:
                notes.append(f"{scene_id}: no usable bands")
                continue
            layers, bare, valid = indices_from_radiance(rad, cfg.ndvi_max, cfg.snow_ndsi)
            if not valid.any():
                notes.append(f"{scene_id}: does not overlap the AOI")
                continue
            if not layers:
                notes.append(f"{scene_id}: has bands {sorted(rad)}, too few for any index")
                continue
            used.append(scene_id)
            bare_fractions.append(float(bare.sum()) / float(valid.sum()))
            for name, layer in layers.items():
                per_scene[name].append(layer)
        if not used:
            write_json(project, {"available": False, "reason": "; ".join(notes) or "no scene gave an index"}, SUMMARY)
            raise SpectralError("no scene gave an index: " + ("; ".join(notes) or "check the band files"))
        for name, layers in per_scene.items():
            if not layers:
                target[name].unlink(missing_ok=True)  # an earlier run's raster from other scenes
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN cells are the point
                combined = layers[0] if len(layers) == 1 else np.nanmedian(np.stack(layers), axis=0).astype(np.float32)
            _write(project, name, combined, transform)
    else:
        used = list(scenes)
        notes.append("rasters are newer than the scenes; kept")
        earlier = project.root / "analysis" / f"{SUMMARY}.json"
        if earlier.exists():
            bare_fractions = [float(json.loads(earlier.read_text(encoding="utf-8")).get("median_bare_pct") or 0.0) / 100.0]

    rasters = {name: path for name, path in target.items() if path.exists()}
    inputs = [SCENE_DATASET]
    licence = derived_license_from(project.provenance.read(), inputs)
    outputs = []
    stats = _occurrence_stats(project, rasters)
    for name, path in rasters.items():
        rel = path.relative_to(project.root).as_posix()
        outputs.append(rel)
        if stale:
            project.provenance.log(
                dataset=f"{RASTER_PREFIX}{name}", source="gossan", identifier=rel, url=None, license=licence, operation="spectral_index",
                parameters={
                    "inputs": inputs, "formula": INDICES[name][1], "scenes": used, "gain": cfg.gain, "ndvi_max": cfg.ndvi_max,
                    "snow_ndsi": cfg.snow_ndsi, "resolution_m": cfg.resolution_m,
                },
            )  # fmt: skip
    summary = {
        "available": True,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_scenes": len(used),
        "scenes": used,
        "gain": cfg.gain,
        "resolution_m": cfg.resolution_m,
        "ndvi_max": cfg.ndvi_max,
        "snow_ndsi": cfg.snow_ndsi,
        "median_bare_pct": round(100.0 * float(np.median(bare_fractions)), 1) if bare_fractions else None,
        "indices": [{"name": n, "raster": f"{RASTER_PREFIX}{n}", "label": INDICES[n][0], "formula": INDICES[n][1], **stats[n]} for n in rasters],
        "notes": notes,
    }
    outputs.append(write_json(project, summary, SUMMARY))
    return SpectralResult(summary, outputs)


def summary_table(summary: dict) -> pd.DataFrame:
    """The per-index rows of a summary as a table."""
    return pd.DataFrame(summary.get("indices", []))
