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

CAPTURE_FRACTION = 0.05


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
    score: np.ndarray, y_test: np.ndarray, effort: np.ndarray | None = None
) -> dict[str, float] | None:
    """AUC, top-5% capture of held-out positives and (with ``effort``, one weight per test cell)
    the effort-matched AUC; None if the fold has no positives or no unlabeled cells."""
    n_pos = int(y_test.sum())
    if n_pos == 0 or n_pos == len(y_test):
        return None
    cutoff = np.quantile(score, 1 - CAPTURE_FRACTION)
    extra = {}
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


@dataclass
class CvResult:
    folds: list[dict] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, object]:
        out: dict[str, object] = {"n_folds_scored": len(self.folds), "skipped": self.skipped, "folds": self.folds}
        for key in ("auc", "auc_effort_matched", "capture_top5"):
            vals = [f[key] for f in self.folds if key in f]
            out[f"{key}_mean"] = float(np.mean(vals)) if vals else None
            out[f"{key}_std"] = float(np.std(vals)) if vals else None
        out["capture_expected_at_random"] = CAPTURE_FRACTION
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
    blocks = make_blocks(grid, block)
    if len(np.unique(blocks)) < n_splits:
        result.skipped.append(f"only {len(np.unique(blocks))} spatial block(s) for {n_splits} folds")
        return result
    xy = grid.xy
    for i, (train, test) in enumerate(spatial_folds(blocks, y, n_splits, cfg.seed), start=1):
        train = drop_near(train, test, xy, cfg.cv_buffer_m)
        train_pos = train[y[train] == 1]
        if len(train_pos) < 2 or int(y[test].sum()) == 0:
            result.skipped.append(f"fold {i}: {len(train_pos)} training and {int(y[test].sum())} test occurrences")
            continue
        X = assemble_matrix(stack, grid, train_pos)  # distance feature from training occurrences only
        res = pu_bagging(
            X, y, train, test,
            n_bags=cfg.n_bags, n_trees=cfg.n_trees, min_samples_leaf=cfg.min_samples_leaf,
            unlabeled_ratio=cfg.unlabeled_ratio, seed=cfg.seed + i, unlabeled_weights=stack.bias,
        )  # fmt: skip
        effort = None if stack.effort_weights is None else stack.effort_weights[test]
        metrics = fold_metrics(res.score, y[test], effort)
        if metrics is None:
            result.skipped.append(f"fold {i}: no positives or no unlabeled cells in the test blocks")
            continue
        result.folds.append({"fold": i, "n_train_cells": len(train), "n_train_positives": len(train_pos), **metrics})
    return result
