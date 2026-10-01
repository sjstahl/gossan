"""Positive-unlabeled bagging with random forests.

The data are *positive-unlabeled*: known occurrences (positives) are real deposits, but every
other cell is merely unlabeled, since it may hold an undiscovered one. Treating the unlabeled
cells as confirmed negatives would teach the model that anywhere nobody has found a deposit is
barren. PU bagging (Mordelet and Vert, 2014) avoids that: each bag trains a random forest on a
subsample of the positives against an equal-sized random draw of unlabeled cells, then scores
only the cells *not* in its training sample. A cell's score is the mean of its out-of-bag scores
over the bags, so a cell is never scored by a model that saw it.

What the outputs mean, and do not:

* ``score`` is a relative ranking of how much a cell looks like known occurrences in the
  features supplied. It is not the probability that a deposit exists: that needs the true
  prevalence of undiscovered deposits, which is unknown. Use it to rank, not as a probability.
* ``spread`` is the standard deviation of a cell's score across the bags that scored it. It
  measures how much the answer depends on which cells were drawn, a part of the model's
  uncertainty. It says nothing about error in the input data or about the labels.
* The forest is only as good as the features: where an area lacks the datasets that matter,
  scores follow whatever proxies remain (tenure density and report density track exploration
  effort).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier

POSITIVE_FRACTION = 0.8  # share of the positives each bag trains on


@dataclass
class PuResult:
    score: np.ndarray  # (n_score,) mean out-of-bag score, 0 to 1
    spread: np.ndarray  # (n_score,) std of the score across bags
    n_scoring_bags: np.ndarray  # (n_score,) how many bags scored the cell
    importances: np.ndarray  # (n_features,) mean impurity importance across bags


def pu_bagging(
    X: np.ndarray,
    y: np.ndarray,
    train_rows: np.ndarray,
    score_rows: np.ndarray,
    *,
    n_bags: int,
    n_trees: int,
    min_samples_leaf: int,
    unlabeled_ratio: float,
    seed: int,
) -> PuResult:
    """Train ``n_bags`` positive-versus-unlabeled forests and score ``score_rows``.

    ``y`` is 1 for known occurrences and 0 for unlabeled cells, for every row of ``X``.
    Bags are drawn from ``train_rows``. A scored row that is also a training row is scored only
    by bags whose sample left it out; rows outside ``train_rows`` are scored by every bag.
    Raises ``ValueError`` if there are fewer than two positives or no unlabeled cells to train on.
    """
    rng = np.random.default_rng(seed)
    pos = train_rows[y[train_rows] == 1]
    unl = train_rows[y[train_rows] == 0]
    if len(pos) < 2:
        raise ValueError("need at least two positive cells to train")
    if len(unl) == 0:
        raise ValueError("no unlabeled cells to train against")
    n_pos_bag = max(2, round(POSITIVE_FRACTION * len(pos)))
    n_unl_bag = int(min(len(unl), max(1, round(unlabeled_ratio * n_pos_bag))))

    total = np.zeros(len(score_rows))
    total_sq = np.zeros(len(score_rows))
    count = np.zeros(len(score_rows), dtype=int)
    importances = np.zeros(X.shape[1])
    models: list[RandomForestClassifier] = []
    in_bag = np.zeros(len(y), dtype=bool)
    for b in range(n_bags):
        sample_pos = rng.choice(pos, size=n_pos_bag, replace=False)
        sample_unl = rng.choice(unl, size=n_unl_bag, replace=False)
        rows = np.concatenate([sample_pos, sample_unl])
        labels = np.concatenate([np.ones(len(sample_pos)), np.zeros(len(sample_unl))])
        forest = RandomForestClassifier(
            n_estimators=n_trees,
            min_samples_leaf=min_samples_leaf,
            max_features="sqrt",
            class_weight="balanced",
            random_state=int(rng.integers(0, 2**31 - 1)),
            n_jobs=1,
        ).fit(X[rows], labels)
        models.append(forest)
        importances += forest.feature_importances_

        in_bag[:] = False
        in_bag[rows] = True
        out = ~in_bag[score_rows]
        if out.any():
            p = forest.predict_proba(X[score_rows[out]])[:, 1]
            total[out] += p
            total_sq[out] += p * p
            count[out] += 1

    score = np.full(len(score_rows), np.nan)
    spread = np.full(len(score_rows), np.nan)
    seen = count > 0
    score[seen] = total[seen] / count[seen]
    spread[seen] = np.sqrt(np.maximum(total_sq[seen] / count[seen] - score[seen] ** 2, 0.0))

    # A row that was in every bag's sample has no out-of-bag score: use all the forests on it.
    missing = ~seen
    if missing.any():
        preds = np.mean([m.predict_proba(X[score_rows[missing]])[:, 1] for m in models], axis=0)
        score[missing] = preds
        spread[missing] = np.std([m.predict_proba(X[score_rows[missing]])[:, 1] for m in models], axis=0)
    # One scoring bag gives no spread; report the largest observed rather than a false zero.
    single = count == 1
    if single.any() and (count > 1).any():
        spread[single] = np.nanmax(spread[count > 1])
    return PuResult(score, spread, count, importances / n_bags)
