"""A knowledge-driven prospectivity score beside the random forest.

The random forest learns from the known occurrences, so it reproduces what they have in common and can
only be checked against them. This score is the other kind: a geologist's reasoning written down. Each
line of evidence is turned into a favourability between 0 and 1 by a stated rule, and the lines are
combined by a weighted mean (an index overlay, the simplest combination whose every number can be
traced). It does not look at the occurrences when it is built, so they are an independent check on it,
and where the two disagree is information: a place geology likes and the forest does not usually lacks
occurrences or data, and a place the forest likes and geology does not may be a camp-shaped artefact.

Evidence layers (each present only when the data are):

=========== ================================================================ ==============================
layer       rule                                                             from
=========== ================================================================ ==============================
fault       1 on a fault, falling to 0 at ``fault_range_m``                  ``dist_fault``
contact     the same for unit contacts, over ``contact_range_m``             ``dist_contact``
intrusive   the same for intrusive rocks, over ``intrusive_range_m``         ``dist_intrusive``
host        1 in the ``host_units`` named by the geologist, 0.2 elsewhere,   ``bedrock_geology``
            or (``host_mode: empirical``) learned from the occurrences
geochem     0 below the AOI median, rising to 1 at the top, of the deposit   ``rgs_idx_*``
            model's pathfinder index (the highest of the indices for
            ``generic``)
magnetics   the same rise for the magnetic analytic signal                   ``mag_as``
alteration  the highest of the same rise for the ASTER Al-OH, Mg-OH,         ``aster_*``
            ferric and gossan ratios
=========== ================================================================ ==============================

The weights come from the deposit model (``PRESETS``: starting points from general exploration geology,
not tuned on any area; the generic mix leaves the intrusive rule out, because across four validated BC areas it
was contradicted at Boundary, silent at two more and informative only at Stewart, and removing it changed the
held-out score by 0.01 or less except 0.02 at Stewart: choose a porphyry, vein or gold model where intrusives matter) and can be replaced one by one. A layer counts for less where it has
gaps (its weight is scaled by the share of the AOI it covers), and the host layer only when the geologist
names host units. Nothing here knows about ARIS or tenure: they measure where people looked.

The host layer is the one place the occurrences can leak in. Named by a geologist who looked at where the
occurrences are, it makes the checks circular (the report says so). With ``host_mode: empirical`` it is
learned from them instead, as the forest's unit favourability is (``log((n + 1) / (expected + 1))`` per
unit), rescaled to 0.2 (no better than chance) up to 1 (``HOST_FAV_MAX`` or more); and the checks relearn it
for every spatial fold from the training blocks only, with the same gap around each test block the forest
uses, so the held-out figures stay honest. The all-cell AUC and the host layer's own AUC are then
in-sample, and the report marks them; the map uses all the occurrences, like the forest's.

The result is checked the way the forest is, on the same spatial folds and metrics (AUC, effort-matched
AUC, top-5% capture of the occurrences in the held-out blocks); there is nothing to fit, so those figures
are simply the score's skill on that ground. Each layer's own AUC shows which of the reasoned rules the
occurrences actually bear out here. ``target/knowledge.tif`` is the score, ``target/knowledge_targets.gpkg``
its top-percentile polygons (with whether each overlaps a forest target), and the forest targets gain
``knowledge_pctile`` and ``knowledge_support``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

from gossan.core.config import KnowledgeConfig, TargetConfig
from gossan.core.project import Project
from gossan.target.cv import (
    default_block_size,
    drop_near,
    effective_buffer,
    fold_metrics,
    make_blocks,
    spatial_folds,
)
from gossan.target.features import FeatureStack, _layer, unit_codes, unit_favourability
from gossan.target.grid import Grid
from gossan.target.outputs import top_percentile_polygons

#: Layer weights by deposit model.
PRESETS: dict[str, dict[str, float]] = {
    "generic": {"fault": 2, "contact": 1, "intrusive": 0, "host": 3, "geochem": 3, "magnetics": 1, "alteration": 1.5},
    "sediment_hosted_zn_pb": {"fault": 3, "contact": 1, "intrusive": 0, "host": 3, "geochem": 3, "magnetics": 0.5, "alteration": 0.5},
    "polymetallic_vein": {"fault": 3, "contact": 0.5, "intrusive": 1.5, "host": 3, "geochem": 3, "magnetics": 1, "alteration": 1.5},
    "cu_porphyry_skarn": {"fault": 1.5, "contact": 1, "intrusive": 3, "host": 3, "geochem": 3, "magnetics": 2, "alteration": 2},
    "gold_vein": {"fault": 3, "contact": 0.5, "intrusive": 1, "host": 3, "geochem": 3, "magnetics": 1, "alteration": 2},
}  # fmt: skip
LABELS = {
    "fault": "Near faults", "contact": "Near unit contacts", "intrusive": "Near intrusive rocks", "host": "In named host units",
    "geochem": "Stream-sediment pathfinders", "magnetics": "Magnetic analytic signal", "alteration": "ASTER alteration ratios",
}  # fmt: skip
ALTERATION_FEATURES = ("aster_aloh", "aster_mgoh", "aster_ferric", "aster_gossan")
OUTSIDE_HOST = 0.2
#: Unit favourability (log scale, see ``unit_favourability``) at which the empirical host rule reaches 1: a
#: unit holding about 1.7 times its share of the occurrences once the +1 shrinkage is allowed for. A stated
#: starting point, not tuned on any area.
HOST_FAV_MAX = 1.0
#: A target whose cells average at or above this percentile of the knowledge score is "supported" by it;
#: between the two limits "partly"; below, "forest only".
SUPPORTED_PCTILE = 80.0
PARTLY_PCTILE = 50.0


@dataclass
class Layer:
    name: str
    mu: np.ndarray  # favourability 0..1 per cell
    weight: float  # the weight used (nominal, scaled by coverage)
    nominal: float
    coverage: float  # share of the AOI with a value
    fitted: bool = False  # learned from the occurrences (the empirical host layer): held-out checks must refit it


@dataclass
class KnowledgeResult:
    score: np.ndarray
    layers: list[Layer]
    report: dict
    targets: gpd.GeoDataFrame
    percentile: np.ndarray = field(default_factory=lambda: np.empty(0))


def near(distance: np.ndarray, range_m: float) -> np.ndarray:
    """1 at the feature, falling linearly to 0 at ``range_m`` and beyond."""
    return np.clip(1.0 - np.asarray(distance, dtype=np.float64) / range_m, 0.0, 1.0)


def upper_ramp(values: np.ndarray) -> np.ndarray:
    """0 for the lower half of the AOI's values, rising linearly to 1 at the highest (by rank)."""
    rank = (rankdata(values) - 0.5) / len(values)  # mid-rank percentile: a constant field sits exactly at 0.5
    return np.clip((rank - 0.5) * 2.0, 0.0, 1.0)


def _column(stack: FeatureStack, name: str) -> np.ndarray | None:
    return stack.X[:, stack.names.index(name)].astype(np.float64) if name in stack.names else None


def _coverage(stack: FeatureStack, names: list[str]) -> float:
    return float(np.mean([1.0 - stack.missing_fraction.get(n, 0.0) for n in names]))


def _host(project: Project, grid: Grid, patterns: list[str]) -> np.ndarray | None:
    """1 in bedrock units named like a pattern, ``OUTSIDE_HOST`` elsewhere; None without the layer or a match."""
    from gossan.analyze.occurrences import unit_name

    bedrock = _layer(project, "bedrock_geology")
    if bedrock is None or not patterns:
        return None
    bedrock = bedrock.to_crs(grid.crs)
    words = [p.lower() for p in patterns]
    names = bedrock.apply(unit_name, axis=1).astype(str).str.lower()
    hit = bedrock[names.map(lambda n: any(w in n for w in words))]
    if hit.empty:
        return None
    raster = rasterize([(g, 1) for g in hit.geometry], out_shape=grid.shape, transform=grid.transform, fill=0, dtype="uint8")
    return np.where(grid.compact(raster) == 1, 1.0, OUTSIDE_HOST)


def empirical_host(units: np.ndarray, positives: np.ndarray) -> np.ndarray:
    """Host favourability learned from ``positives``: 0.2 where the unit holds no more than its share, rising
    to 1 at ``HOST_FAV_MAX``. Each positive leaves itself out, as in the forest's feature."""
    fav = unit_favourability(units, positives)
    return OUTSIDE_HOST + (1.0 - OUTSIDE_HOST) * np.clip(fav / HOST_FAV_MAX, 0.0, 1.0)


def _units(project: Project, stack: FeatureStack, grid: Grid) -> np.ndarray | None:
    """Bedrock unit code of every cell: the stack's, else built here; None without two units."""
    units = stack.units if stack.units is not None else unit_codes(project, grid)
    return units if units is not None and int(units.max()) >= 1 else None


def build_layers(project: Project, stack: FeatureStack, grid: Grid, cfg: KnowledgeConfig) -> tuple[list[Layer], dict[str, str]]:
    """The evidence layers the project's data allow, and the reason for each one that is missing."""
    weights = {**PRESETS[cfg.deposit_model], **cfg.weights}
    layers: list[Layer] = []
    skipped: dict[str, str] = {}

    def add(name: str, mu: np.ndarray, used: list[str], fitted: bool = False) -> None:
        nominal = float(weights.get(name, 0.0))
        if nominal <= 0:
            skipped[name] = "weight 0"
            return
        cover = 1.0 if not used else _coverage(stack, used)
        layers.append(Layer(name, np.asarray(mu, dtype=np.float64), nominal * cover, nominal, cover, fitted))

    for name, feature, rng in (
        ("fault", "dist_fault", cfg.fault_range_m), ("contact", "dist_contact", cfg.contact_range_m), ("intrusive", "dist_intrusive", cfg.intrusive_range_m),
    ):  # fmt: skip
        col = _column(stack, feature)
        if col is None:
            skipped[name] = stack.skipped.get(feature, f"no {feature} feature")
        else:
            add(name, near(col, rng), [feature])
    if cfg.host_mode == "empirical":
        units = _units(project, stack, grid)
        if units is None:
            skipped["host"] = "empirical host layer needs a bedrock_geology layer with at least two units"
        else:
            add("host", empirical_host(units, stack.positives), [], fitted=True)
    elif cfg.host_units:
        host = _host(project, grid, cfg.host_units)
        if host is None:
            skipped["host"] = f"no bedrock unit named like {', '.join(cfg.host_units)}"
        else:
            add("host", host, [])
    else:
        skipped["host"] = "no host_units named"
    wanted = f"rgs_idx_{cfg.deposit_model}"
    indices = [wanted] if wanted in stack.names else [n for n in stack.names if n.startswith("rgs_idx_")]
    if indices:
        add("geochem", np.max([upper_ramp(_column(stack, n)) for n in indices], axis=0), indices)
    else:
        skipped["geochem"] = "no pathfinder index features (needs RGS samples and geochem_features: indices)"
    if "mag_as" in stack.names:
        add("magnetics", upper_ramp(_column(stack, "mag_as")), ["mag_as"])
    else:
        skipped["magnetics"] = stack.skipped.get("magnetics", "no magnetic grid")
    alteration = [n for n in ALTERATION_FEATURES if n in stack.names]
    if alteration:
        add("alteration", np.max([upper_ramp(_column(stack, n)) for n in alteration], axis=0), alteration)
    else:
        skipped["alteration"] = "no ASTER index features (run 'gossan spectral')"
    return layers, skipped


def combine(layers: list[Layer]) -> np.ndarray:
    """The weighted mean of the layers' favourabilities."""
    total = sum(layer.weight for layer in layers)
    return sum(layer.weight * layer.mu for layer in layers) / total


def _summ(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values))


def _checks(
    score: np.ndarray, y: np.ndarray, stack: FeatureStack, grid: Grid, cfg: TargetConfig,
    layers: list[Layer] | None = None, units: np.ndarray | None = None,
) -> dict:  # fmt: skip
    """AUC of the score over every cell, and over the held-out blocks of the forest's spatial folds.

    When a layer is ``fitted`` (the empirical host layer) each fold relearns it from the training blocks only
    (training cells within the CV gap of a test block are dropped, as for the forest) and rescores the test
    cells, so the fold figures are held out; ``auc`` over every cell is then in-sample.
    """
    out: dict = {"auc": float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else None}
    effort = stack.effort_weights
    refit = bool(units is not None and any(layer.fitted for layer in layers or []))
    block = cfg.block_size_m or default_block_size(grid)
    try:
        folds = spatial_folds(make_blocks(grid, block), y, cfg.folds, cfg.seed)
    except ValueError:
        folds = []
    gap = effective_buffer(cfg.cv_buffer_m, block)
    rows: list[dict] = []
    host_aucs: list[float] = []
    skipped: list[str] = []
    for i, (train, te) in enumerate(folds, start=1):
        fold_score = score
        if refit:
            train = drop_near(train, te, grid.xy, gap)
            train_pos = train[y[train] == 1]
            if len(train_pos) < 2 or int(y[te].sum()) == 0:
                skipped.append(f"fold {i}: {len(train_pos)} training and {int(y[te].sum())} test occurrences")
                continue
            host = empirical_host(units, train_pos)
            fold_score = combine([replace(layer, mu=host) if layer.fitted else layer for layer in layers])
            if 0 < y[te].sum() < len(te):
                host_aucs.append(float(roc_auc_score(y[te], host[te])))
        metrics = fold_metrics(fold_score[te], y[te], None if effort is None else effort[te])
        if metrics is not None:
            rows.append(metrics)
    out["n_folds_scored"] = len(rows)
    for key in ("auc", "auc_effort_matched", "capture_top5"):
        vals = [r[key] for r in rows if key in r]
        out[f"{key}_fold_mean"], out[f"{key}_fold_std"] = _summ(vals) if vals else (None, None)
    if refit:
        out["held_out_host"] = True
        out["host_layer_auc_fold_mean"] = float(np.mean(host_aucs)) if host_aucs else None
        out["cv_gap_m"] = gap
        out["folds_skipped"] = skipped
    return out


def _zonal_mean(grid: Grid, values: np.ndarray, polygons: gpd.GeoSeries) -> list[float]:
    raster = grid.to_raster(values)
    means = []
    for geom in polygons:
        inside = rasterize([(geom, 1)], out_shape=grid.shape, transform=grid.transform, fill=0, dtype="uint8") == 1
        cells = raster[inside & grid.mask]
        means.append(float(np.nanmean(cells)) if cells.size else float("nan"))
    return means


def support_label(pctile: float) -> str:
    if pctile >= SUPPORTED_PCTILE:
        return "supported"
    return "partly" if pctile >= PARTLY_PCTILE else "forest only"


def run_knowledge(project: Project, stack: FeatureStack, grid: Grid, y: np.ndarray, rf_score: np.ndarray, cfg: TargetConfig) -> KnowledgeResult | None:
    """The knowledge-driven score, its checks and comparison with ``rf_score``, and its target polygons.
    None when switched off; a result with an ``available: False`` report when no layer can be built."""
    kcfg = cfg.knowledge
    if not kcfg.enabled:
        return None
    layers, skipped = build_layers(project, stack, grid, kcfg)
    empty = gpd.GeoDataFrame({"target_id": []}, geometry=[], crs=grid.crs)
    if not layers:
        report = {"available": False, "reason": "no evidence layer could be built: " + "; ".join(f"{k}: {v}" for k, v in skipped.items()), "skipped": skipped}
        return KnowledgeResult(np.empty(0), [], report, empty)
    score = combine(layers)
    pct = 100.0 * rankdata(score) / len(score)
    cut = np.quantile(score, cfg.top_percentile / 100.0)
    rf_cut = np.quantile(rf_score, cfg.top_percentile / 100.0)
    top, rf_top = score >= cut, rf_score >= rf_cut
    min_cells = int(np.ceil(cfg.min_target_ha * 1e4 / grid.resolution**2))
    polygons, threshold = top_percentile_polygons(
        grid, score, np.zeros_like(score), stack.positives, cfg.top_percentile, smooth_cells=cfg.smooth_m / grid.resolution, min_cells=min_cells
    )  # fmt: skip
    if len(polygons):
        polygons["target_id"] = [f"K{i:03d}" for i in polygons["rank"]]
        polygons = polygons.drop(columns=["mean_spread"], errors="ignore")
    per_layer = []
    for layer in layers:
        uniform = np.ptp(layer.mu) == 0
        per_layer.append(
            {
                "layer": layer.name, "label": LABELS[layer.name], "weight": round(layer.nominal, 2), "weight_used": round(layer.weight, 2), "coverage_pct": round(100 * layer.coverage, 1),
                "auc": None if uniform or not 0 < y.sum() < len(y) else round(float(roc_auc_score(y, layer.mu)), 3), "in_sample": layer.fitted,
            }
        )
    rho = spearmanr(score, rf_score).statistic
    report = {
        "available": True,
        "deposit_model": kcfg.deposit_model,
        "host_units": kcfg.host_units,
        "layers": per_layer,
        "skipped": skipped,
        "checks": _checks(score, y, stack, grid, cfg, layers, _units(project, stack, grid) if kcfg.host_mode == "empirical" else None),
        "n_positives": int(y.sum()),
        "rank_correlation_with_forest": None if not np.isfinite(rho) else round(float(rho), 3),
        "top_percentile": cfg.top_percentile,
        "top_overlap_with_forest": round(float((top & rf_top).sum() / max((top | rf_top).sum(), 1)), 3),
        "n_knowledge_targets": len(polygons),
        "n_dropped_small": int(polygons.attrs.get("n_dropped_small", 0)),
        "score_threshold": float(threshold),
        "host_mode": kcfg.host_mode,
        "circular_host_units": kcfg.host_mode == "named" and bool(kcfg.host_units),
        "host_learned_from_occurrences": any(layer.fitted for layer in layers),
        "note": _note(kcfg, layers),
    }  # fmt: skip
    return KnowledgeResult(score, layers, report, polygons, pct)


def _note(kcfg: KnowledgeConfig, layers: list[Layer]) -> str:
    if any(layer.fitted for layer in layers):
        return (
            "the host layer was learned from the occurrences: the fold figures relearn it from training blocks only and are held out; "
            "the all-cell AUC and the host layer's own AUC are in-sample"
        )
    if kcfg.host_units:
        return "host_units were chosen by the geologist; if they were chosen because occurrences sit in them, the checks above flatter the score"
    return ""


def annotate_targets(targets: gpd.GeoDataFrame, grid: Grid, result: KnowledgeResult) -> gpd.GeoDataFrame:
    """The forest's targets with the knowledge score they sit on and whether it supports them."""
    if result.score.size == 0 or targets.empty:
        return targets
    out = targets.copy()
    out["knowledge_mean"] = np.round(_zonal_mean(grid, result.score, out.geometry), 4)
    out["knowledge_pctile"] = np.round(_zonal_mean(grid, result.percentile, out.geometry), 1)
    out["knowledge_support"] = out["knowledge_pctile"].map(support_label)
    return out


def annotate_knowledge_targets(polygons: gpd.GeoDataFrame, forest: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Knowledge polygons with whether they overlap a forest target (and which)."""
    if polygons.empty:
        return polygons
    out = polygons.copy()
    ids = []
    for geom in out.geometry:
        hit = forest[forest.intersects(geom)] if len(forest) else forest
        ids.append("; ".join(hit["target_id"].astype(str)) if len(hit) else "")
    out["forest_targets"] = ids
    out["overlaps_forest_target"] = [bool(i) for i in ids]
    return out
