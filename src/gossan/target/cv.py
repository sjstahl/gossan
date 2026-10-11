"""Spatial cross-validation.

Neighbouring cells look alike (spatial autocorrelation), so a random train/test split puts a
test cell right beside a training cell with almost the same features and label, and the model
scores well by memorising places. Here the grid is cut into square blocks, whole blocks are held
out together, and the training cells within ``cv_buffer_m`` of any held-out cell are dropped as
well. The model never sees the neighbourhood of what it is tested on.

Because the data are positive-unlabeled, a held-out unlabeled cell is not a confirmed negative,
so the usual accuracy is not available. Two ranking measures are reported instead, per fold and
averaged, from the held-out cells only:

* ``auc``: the chance that a held-out known occurrence scores higher than a held-out unlabeled
  cell. Undiscovered deposits among the unlabeled cells pull it down, so read it as a lower
  bound and for comparing models, not as an absolute figure.
* ``capture_top5``: the share of held-out occurrences that fall in the top 5% of held-out
  cells by score. A model with no skill captures about 5%.
* ``auc_effort_matched``: the AUC with each held-out unlabeled cell weighted by its exploration
  effort (``FeatureStack.effort_weights``), i.e. occurrences against ground explored about as
  much. Known occurrences sit where people looked, so the plain AUC rewards a model for
  predicting exploration itself; this one does not. It is the fairer figure for comparing
  ``effort`` modes, and the one that says whether the model finds deposits rather than camps.

Because the distance-to-occurrence feature is rebuilt from the training occurrences only in
each fold, held-out occurrences never inform their own features.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from gossan.core.config import TargetConfig
from gossan.target.features import FeatureStack, assemble_matrix
from gossan.target.grid import Grid
from gossan.target.model import pu_bagging
from gossan.target.outputs import smooth_scores

CAPTURE_FRACTION = 0.05


AUTO_BUFFER_FRACTION = 0.15
AUTO_BUFFER_MAX_M = 2000.0


def effective_buffer(cfg_buffer_m: float | None, block_size_m: float) -> float:
    """The gap used between test blocks and training cells: ``cfg_buffer_m`` if set, else
    ``AUTO_BUFFER_FRACTION`` of the block edge, at most ``AUTO_BUFFER_MAX_M``."""
    if cfg_buffer_m is not None:
        return cfg_buffer_m
    return min(AUTO_BUFFER_MAX_M, AUTO_BUFFER_FRACTION * block_size_m)


def default_block_size(grid: Grid, resolution_floor: float = 5.0) -> float:
    """A quarter of the AOI's shorter side, at least ``resolution_floor`` cells."""
    minx, miny, maxx, maxy = grid.bounds
    return max((min(maxx - minx, maxy - miny)) / 4.0, resolution_floor * grid.resolution)


def make_blocks(grid: Grid, block_size_m: float) -> np.ndarray:
    """Block id of every in-AOI cell (square blocks of ``block_size_m``)."""
    xy = grid.xy
    bx = np.floor((xy[:, 0] - grid.bounds[0]) / block_size_m).astype(int)
    by = np.floor((grid.bounds[3] - xy[:, 1]) / block_size_m).astype(int)
    return bx * (by.max() + 1) + by


def spatial_folds(
    blocks: np.ndarray, y: np.ndarray, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train rows, test rows) per fold; whole blocks go to one side, positives spread evenly."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in splitter.split(np.zeros(len(y)), y, groups=blocks)]


def drop_near(train: np.ndarray, test: np.ndarray, xy: np.ndarray, buffer_m: float) -> np.ndarray:
    """Training rows farther than ``buffer_m`` from every test row."""
    if buffer_m <= 0 or len(test) == 0:
        return train
    d, _ = cKDTree(xy[test]).query(xy[train], k=1)
    return train[d > buffer_m]


def fold_metrics(
    score: np.ndarray, y_test: np.ndarray, effort: np.ndarray | None = None, smoothed: np.ndarray | None = None
) -> dict[str, float] | None:
    """AUC, top-5% capture of held-out positives, (with ``effort``, one weight per test cell) the
    effort-matched AUC and (with ``smoothed``, the test cells' smoothed score) the capture when
    targets are chosen on the smoothed score; None if the fold has no positives or no unlabeled
    cells."""
    n_pos = int(y_test.sum())
    if n_pos == 0 or n_pos == len(y_test):
        return None
    cutoff = np.quantile(score, 1 - CAPTURE_FRACTION)
    extra = {}
    if smoothed is not None:
        cut_s = np.quantile(smoothed, 1 - CAPTURE_FRACTION)
        extra["capture_top5_smoothed"] = float(((smoothed >= cut_s) & (y_test == 1)).sum() / n_pos)
    if effort is not None:
        weights = np.where(y_test == 1, 1.0, effort)
        extra["auc_effort_matched"] = float(roc_auc_score(y_test, score, sample_weight=weights))
    return {
        **extra,
        "auc": float(roc_auc_score(y_test, score)),
        "capture_top5": float(((score >= cutoff) & (y_test == 1)).sum() / n_pos),
        "n_test_positives": n_pos,
        "n_test_cells": len(y_test),
    }


MIN_TEST_BARREN = 3


def auc_vs_barren(score: np.ndarray, y_test: np.ndarray, barren_test: np.ndarray) -> dict[str, float] | None:
    """AUC of held-out known occurrences against held-out cells holding a barren drill hole: the one
    comparison against real negatives rather than unlabeled cells. None without a held-out occurrence
    or at least ``MIN_TEST_BARREN`` held-out barren cells."""
    pos = y_test == 1
    neg = barren_test & ~pos  # a cell that is both is an occurrence
    if pos.sum() < 1 or neg.sum() < MIN_TEST_BARREN:
        return None
    labels = np.r_[np.ones(int(pos.sum())), np.zeros(int(neg.sum()))]
    return {"auc_vs_barren": float(roc_auc_score(labels, np.r_[score[pos], score[neg]])), "n_test_barren": int(neg.sum())}


@dataclass
class CvResult:
    folds: list[dict] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    block_size_m: float | None = None
    buffer_m: float | None = None
    n_barren_cells: int = 0

    def summary(self) -> dict[str, object]:
        out: dict[str, object] = {"n_folds_scored": len(self.folds), "skipped": self.skipped, "folds": self.folds}
        for key in ("auc", "auc_effort_matched", "auc_vs_barren", "capture_top5", "capture_top5_smoothed"):
            vals = [f[key] for f in self.folds if key in f]
            out[f"{key}_mean"] = float(np.mean(vals)) if vals else None
            out[f"{key}_std"] = float(np.std(vals)) if vals else None
        out["capture_expected_at_random"] = CAPTURE_FRACTION
        out["n_barren_cells"] = self.n_barren_cells
        out["block_size_m"] = self.block_size_m
        out["buffer_m"] = self.buffer_m
        return out


def run_cv(stack: FeatureStack, grid: Grid, cfg: TargetConfig, y: np.ndarray) -> CvResult:
    """Spatial K-fold cross-validation of the whole PU-bagging model."""
    result = CvResult()
    n_pos = int(y.sum())
    n_splits = min(cfg.folds, n_pos // 2)
    if n_splits < 2:
        result.skipped.append(f"{n_pos} occurrences are too few for cross-validation")
        return result
    block = cfg.block_size_m or default_block_size(grid)
    buffer_m = effective_buffer(cfg.cv_buffer_m, block)
    result.block_size_m, result.buffer_m = block, buffer_m
    result.n_barren_cells = 0 if stack.barren_cells is None else len(stack.barren_cells)
    blocks = make_blocks(grid, block)
    if len(np.unique(blocks)) < n_splits:
        result.skipped.append(f"only {len(np.unique(blocks))} spatial block(s) for {n_splits} folds")
        return result
    xy = grid.xy
    for i, (train, test) in enumerate(spatial_folds(blocks, y, n_splits, cfg.seed), start=1):
        train = drop_near(train, test, xy, buffer_m)
        train_pos = train[y[train] == 1]
        if len(train_pos) < 2 or int(y[test].sum()) == 0:
            result.skipped.append(f"fold {i}: {len(train_pos)} training and {int(y[test].sum())} test occurrences")
            continue
        X = assemble_matrix(stack, grid, train_pos)  # distance feature from training occurrences only
        res = pu_bagging(
            X, y, train, test,
            n_bags=cfg.n_bags, n_trees=cfg.n_trees, min_samples_leaf=cfg.min_samples_leaf,
            unlabeled_ratio=cfg.unlabeled_ratio, seed=cfg.seed + i, unlabeled_weights=stack.bias, negatives=stack.negatives,
        )  # fmt: skip
        effort = None if stack.effort_weights is None else stack.effort_weights[test]
        smoothed = None
        if cfg.smooth_m > 0:  # smooth within the held-out blocks only, so no training cell leaks in
            smoothed = smooth_scores(grid, res.score, cfg.smooth_m / grid.resolution, rows=test)
        metrics = fold_metrics(res.score, y[test], effort, smoothed)
        if metrics is not None and stack.barren_cells is not None:
            vs = auc_vs_barren(res.score, y[test], np.isin(test, stack.barren_cells))
            if vs is not None:
                metrics.update(vs)
        if metrics is None:
            result.skipped.append(f"fold {i}: no positives or no unlabeled cells in the test blocks")
            continue
        result.folds.append({"fold": i, "n_train_cells": len(train), "n_train_positives": len(train_pos), **metrics})
    return result
