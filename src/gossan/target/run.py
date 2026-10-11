"""Running the prospectivity model for a project."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gossan.core.config import TargetConfig
from gossan.core.licensing import derived_license_from
from gossan.core.project import Project
from gossan.target.cv import run_cv
from gossan.target.features import (
    FeatureError,
    FeatureStack,
    assemble_matrix,
    build_stack,
    input_datasets,
)
from gossan.target.grid import Grid, GridError, build_grid
from gossan.target.model import pu_bagging
from gossan.target.outputs import (
    NODATA,
    target_dir,
    top_percentile_polygons,
    write_cog,
    write_importances,
    write_json,
)


class TargetError(Exception):
    """The model cannot be run; the message says why."""


@dataclass
class TargetResult:
    grid: Grid
    stack: FeatureStack
    cfg: TargetConfig
    cv: dict[str, Any] | None
    outputs: dict[str, str] = field(default_factory=dict)
    n_targets: int = 0
    n_dropped_small: int = 0
    threshold: float = 0.0
    importances: dict[str, float] = field(default_factory=dict)


def run_target(
    project: Project,
    cfg: TargetConfig,
    *,
    cross_validate: bool = True,
) -> TargetResult:
    """Build the grid and feature stack, run spatial CV, fit the final model and write outputs
    to ``<project>/target/``. Provenance is logged for every output."""
    try:
        grid = build_grid(project.read_aoi(), project.config.crs, cfg.resolution_m, cfg.max_cells)
    except GridError as e:
        raise TargetError(str(e)) from e
    try:
        stack = build_stack(project, grid, cfg)
    except FeatureError as e:
        raise TargetError(str(e)) from e
    n_pos = len(stack.positives)
    if n_pos < cfg.min_positives:
        raise TargetError(
            f"{n_pos} known {'' if cfg.occurrence_class == 'all' else cfg.occurrence_class + ' '}occurrence(s) fall in the AOI's {grid.resolution:g} m cells "
            f"({stack.n_occurrences_total} in the layer, {stack.n_occurrences_outside} outside the "
            f"AOI); at least {cfg.min_positives} are needed (target.min_positives). Use a larger "
            "AOI or a coarser --resolution (several occurrences in one cell count once)"
        )
    if n_pos >= grid.n_cells:
        raise TargetError("every cell holds an occurrence; there is nothing unlabeled to learn from")

    y = np.zeros(grid.n_cells, dtype=int)
    y[stack.positives] = 1

    cv = run_cv(stack, grid, cfg, y).summary() if cross_validate else None

    # Final model: every cell trains and is scored out-of-bag; the label-derived distance feature
    # is built from all occurrences, each measured to its nearest other occurrence.
    X = assemble_matrix(stack, grid, stack.positives)
    everything = np.arange(grid.n_cells)
    fit = pu_bagging(
        X, y, everything, everything,
        n_bags=cfg.n_bags, n_trees=cfg.n_trees, min_samples_leaf=cfg.min_samples_leaf,
        unlabeled_ratio=cfg.unlabeled_ratio, seed=cfg.seed, unlabeled_weights=stack.bias, negatives=stack.negatives,
    )  # fmt: skip

    out = target_dir(project.root)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "prospectivity": out / "prospectivity.tif",
        "uncertainty": out / "uncertainty.tif",
        "feature_stack": out / "feature_stack.tif",
        "targets": out / "targets.gpkg",
        "feature_importance": out / "feature_importance.csv",
        "cv_report": out / "cv_report.json",
        "grid": out / "grid.json",
    }
    write_cog(paths["prospectivity"], grid, {"prospectivity": grid.to_raster(fit.score)})
    write_cog(paths["uncertainty"], grid, {"uncertainty": grid.to_raster(fit.spread)})
    bands = {n: grid.to_raster(X[:, i]) for i, n in enumerate(stack.model_names)}
    write_cog(paths["feature_stack"], grid, bands)

    min_cells = int(np.ceil(cfg.min_target_ha * 1e4 / grid.resolution**2))
    targets, threshold = top_percentile_polygons(
        grid, fit.score, fit.spread, stack.positives, cfg.top_percentile,
        smooth_cells=cfg.smooth_m / grid.resolution, min_cells=min_cells,
    )  # fmt: skip
    n_dropped = int(targets.attrs.get("n_dropped_small", 0))
    if len(targets):
        targets.to_file(paths["targets"], layer="targets_top", driver="GPKG", engine="pyogrio")
    importances = dict(zip(stack.model_names, (float(v) for v in fit.importances), strict=True))
    write_importances(paths["feature_importance"], stack.model_names, fit.importances)
    write_json(paths["cv_report"], cv if cv is not None else {"skipped": ["cross-validation was switched off"]})

    description = {
        **grid.describe(),
        "nodata": NODATA,
        "n_known_occurrences_in_aoi": n_pos,
        "smooth_m": cfg.smooth_m,
        "min_target_ha": cfg.min_target_ha,
        "n_targets_dropped_small": n_dropped,
        "occurrence_class": cfg.occurrence_class,
        "effort": cfg.effort,
        "drill_labels": cfg.drill_labels,
        "n_barren_drill_cells": 0 if stack.barren_cells is None else len(stack.barren_cells),
        "n_drill_positives_added": stack.n_drill_positives_added,
        "drill_note": stack.drill_note,
        "background_weighted_by_effort": stack.bias is not None,
        "n_occurrences_in_layer": stack.n_occurrences_total,
        "n_occurrences_outside_aoi": stack.n_occurrences_outside,
        "features_used": stack.model_names,
        "features_skipped": stack.skipped,
        "feature_missing_fraction": stack.missing_fraction,
        "min_scoring_bags": int(fit.n_scoring_bags.min()),
        "mean_scoring_bags": float(fit.n_scoring_bags.mean()),
    }
    write_json(paths["grid"], description)

    params = {
        "resolution_m": grid.resolution,
        "grid": [grid.height, grid.width],
        "n_cells_in_aoi": grid.n_cells,
        "n_positives": n_pos,
        "occurrence_class": cfg.occurrence_class,
        "effort": cfg.effort,
        "geochem_features": cfg.geochem_features,
        "features_used": stack.model_names,
        "features_skipped": stack.skipped,
        "model": {
            "type": "PU-bagging random forest",
            "n_bags": cfg.n_bags,
            "n_trees": cfg.n_trees,
            "min_samples_leaf": cfg.min_samples_leaf,
            "unlabeled_ratio": cfg.unlabeled_ratio,
            "seed": cfg.seed,
        },
        "cv": None
        if cv is None
        else {
            k: cv[k]
            for k in (
                "n_folds_scored", "auc_mean", "auc_std", "auc_effort_matched_mean", "auc_effort_matched_std",
                "capture_top5_mean", "capture_top5_std", "capture_top5_smoothed_mean", "capture_top5_smoothed_std", "skipped",
                "block_size_m", "buffer_m", "auc_vs_barren_mean", "auc_vs_barren_std", "n_barren_cells",
            )
        },
        "top_percentile": cfg.top_percentile,
        "smooth_m": cfg.smooth_m,
        "min_target_ha": cfg.min_target_ha,
        "n_targets_dropped_small": n_dropped,
        "score_threshold": threshold,
        "n_targets": len(targets),
    }
    for name, path in paths.items():
        if name == "targets" and not len(targets):
            continue
        project.provenance.log(
            dataset=f"target_{name}",
            source="gossan",
            identifier=path.relative_to(project.root).as_posix(),
            url=None,
            license=derived_license_from(project.provenance.read(), input_datasets(stack)),
            operation="target_run",
            parameters={"output": name, **params},
        )
    return TargetResult(
        grid, stack, cfg, cv,
        {k: v.relative_to(project.root).as_posix() for k, v in paths.items()},
        len(targets), n_dropped, threshold, importances,
    )  # fmt: skip
