"""The evidence-first target defaults: geology/terrain/geochemistry features, effort as background
bias, unit favourability, occurrence classes and the effort-matched CV metric."""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import LineString, Point, box
from synthetic import CRS, X0, Y0, H, W, build_project, runner, small_cfg

from gossan.cli import app
from gossan.core.config import TargetConfig
from gossan.core.project import Project
from gossan.target.cv import fold_metrics
from gossan.target.features import (
    FeatureError,
    build_stack,
    distance_to_geometries,
    input_datasets,
    unit_favourability,
)
from gossan.target.grid import build_grid
from gossan.target.model import pu_bagging

EAST = X0 + 3000


def evidence_cfg(**kw) -> TargetConfig:
    """Fast, but with the evidence-first defaults (unlike ``small_cfg``)."""
    return TargetConfig(**{"resolution_m": 100.0, "n_bags": 4, "n_trees": 15, "folds": 3, "seed": 1, **kw})


def add_evidence(project: Project, seed: int = 0) -> Project:
    """Faults, two bedrock units, a DEM, a multi-element RGS layer and deposit codes."""
    rng = np.random.default_rng(seed)

    def write(name, gdf):
        gdf.to_file(project.data_gpkg, layer=name, driver="GPKG", engine="pyogrio")

    minfile = gpd.read_file(project.data_gpkg, layer="minfile")
    minfile["DEPOSIT_TYPE_CODE1"] = np.where(np.arange(len(minfile)) % 4 == 0, "F02", "I05")  # 3/4 metallic
    write("minfile", minfile)
    write("faults", gpd.GeoDataFrame(
        {"FAULT_TYPE": ["Thrust", "Normal Fault"]},
        geometry=[LineString([(X0 + 3500, Y0 - 100), (X0 + 3500, Y0 + H + 100)]),
                  LineString([(X0 - 2000, Y0 + 1500), (X0 - 1000, Y0 + 1500)])],  # outside the AOI
        crs=CRS,
    ))  # fmt: skip
    write("bedrock_geology", gpd.GeoDataFrame(
        {
            "STRATIGRAPHIC_NAME": ["Aldridge Formation", "Creston Formation", "Unnamed"],
            "GEOLOGICAL_PERIOD": ["Middle Proterozoic", "Middle Proterozoic", "Cretaceous"],
            "ROCK_CLASS": ["sedimentary rocks", "sedimentary rocks", "intrusive rocks"],
            "MAXIMUM_AGE_VALUE": [1500.0, 1400.0, 4000.0],
            "MINIMUM_AGE_VALUE": [1400.0, 1300.0, 0.0],
        },
        geometry=[box(EAST, Y0 - 500, X0 + W + 500, Y0 + H + 500), box(X0 - 500, Y0 - 500, EAST, Y0 + H + 500),
                  box(X0 + 500, Y0 + 500, X0 + 900, Y0 + 900)],
        crs=CRS,
    ))  # fmt: skip
    n = 40
    gx, gy = rng.uniform(X0, X0 + W, n), rng.uniform(Y0, Y0 + H, n)
    east = (gx - X0) / W
    write("rgs", gpd.GeoDataFrame(
        {
            "ZINC_AAS_PPM": np.round(80 * np.exp(east + rng.normal(0, 0.2, n))),
            "LEAD_AAS_PPM": np.round(12 * np.exp(east + rng.normal(0, 0.2, n))),
            "COPPER_AAS_PPM": np.round(20 * np.exp(rng.normal(0, 0.3, n))),
            "ARSENIC_INAA_PPM": np.round(5 * np.exp(rng.normal(0, 0.3, n)), 1),
            "IRON_AAS_PERCENTAGE": np.round(2.5 * np.exp(rng.normal(0, 0.2, n)), 2),
            "MANGANESE_AAS_PPM": np.round(600 * np.exp(rng.normal(0, 0.3, n))),
        },
        geometry=[Point(a, b) for a, b in zip(gx, gy, strict=True)], crs=CRS,
    ))  # fmt: skip
    data = (1000 + np.tile(np.linspace(0, 500, 80, dtype=np.float32), (60, 1))).astype(np.float32)
    with rasterio.open(
        project.root / "rasters" / "dem.tif", "w", driver="GTiff", width=80, height=60, count=1, dtype="float32",
        crs=CRS, transform=from_bounds(X0 - 200, Y0 - 200, X0 + W + 200, Y0 + H + 200, 80, 60), nodata=-9999.0,
    ) as ds:  # fmt: skip
        ds.write(data, 1)
    return project


@pytest.fixture
def project(tmp_path) -> Project:
    proj = add_evidence(build_project(tmp_path, n_occ=60, seed=4))
    proj.config.target = evidence_cfg()
    proj.config.save(proj.root)
    return Project(proj.root)


@pytest.fixture
def grid(project):
    return build_grid(project.read_aoi(), CRS, 100.0, 10**6)


# --- building blocks ---------------------------------------------------------------------


def test_distance_to_geometries_counts_features_outside_the_grid(grid):
    inside = distance_to_geometries(grid, [LineString([(X0 + 2000, Y0), (X0 + 2000, Y0 + H)])])
    xy = grid.xy
    assert np.allclose(inside, np.abs(xy[:, 0] - (X0 + 2000)), atol=grid.resolution)
    outside = distance_to_geometries(grid, [LineString([(X0 - 1000, Y0), (X0 - 1000, Y0 + H)])])
    assert outside.min() == pytest.approx(1000 + grid.resolution / 2, abs=grid.resolution)
    assert np.isnan(distance_to_geometries(grid, [LineString([(X0 - 50_000, Y0), (X0 - 50_000, Y0 + 1)])])).all()


def test_unit_favourability_math_and_leave_one_out():
    units = np.array([0] * 25 + [1] * 75)  # unit 0: a quarter of the area
    positives = np.array([0, 1, 2, 3, 30])  # 4 in unit 0, 1 in unit 1
    fav = unit_favourability(units, positives)
    assert fav[10] == pytest.approx(np.log((4 + 1) / (5 * 0.25 + 1)))  # unlabeled cell in unit 0
    assert fav[50] == pytest.approx(np.log((1 + 1) / (5 * 0.75 + 1)))
    assert fav[0] == pytest.approx(np.log((3 + 1) / (4 * 0.25 + 1)))  # a positive leaves itself out
    assert fav[10] > 0 > fav[50]
    assert (unit_favourability(np.array([-1, -1, 0]), np.array([2])) [:2] == 0).all()  # fmt: skip


def test_unlabeled_weights_steer_background_draws():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = np.zeros(200, dtype=int)
    y[:20] = 1
    rows = np.arange(200)
    weights = np.where(np.arange(200) < 110, 1.0, 0.0)  # rows 110+ are never drawn as background
    res = pu_bagging(X, y, rows, rows, n_bags=6, n_trees=5, min_samples_leaf=1, unlabeled_ratio=1.0, seed=1,
                     unlabeled_weights=weights)  # fmt: skip
    assert (res.n_scoring_bags[110:] == 6).all()  # never in a bag, so scored by every bag
    assert (res.n_scoring_bags[20:110] < 6).any()


def test_effort_matched_auc_discounts_unexplored_background():
    score = np.array([0.9, 0.8, 0.85, 0.1, 0.1, 0.1])  # model ranks explored ground high
    y = np.array([1, 0, 0, 0, 0, 0])
    effort = np.array([1.0, 10.0, 10.0, 0.01, 0.01, 0.01])
    m = fold_metrics(score, y, effort)
    assert m["auc"] == pytest.approx(1.0)
    assert m["auc_effort_matched"] == pytest.approx(1.0)  # still beats the explored background
    m2 = fold_metrics(np.array([0.5, 0.8, 0.85, 0.1, 0.1, 0.1]), y, effort)
    assert m2["auc"] == pytest.approx(0.6)
    assert m2["auc_effort_matched"] < 0.01  # loses to every well-explored cell, which dominate


# --- the stack ---------------------------------------------------------------------------


def test_default_stack_uses_evidence_and_keeps_effort_out(project, grid):
    stack = build_stack(project, grid, evidence_cfg())
    for name in ("dist_fault", "dist_thrust", "dist_contact", "dist_intrusive", "unit_age_ma",
                 "elevation", "slope", "relief", "synth_magnetics"):  # fmt: skip
        assert name in stack.names, name
    assert any(n.startswith("rgs_idx_") for n in stack.names) and "rgs_pc1" in stack.names
    assert "tenure_density" not in stack.names and "aris_density" not in stack.names
    assert "not used" in stack.skipped["tenure_density"]
    assert stack.bias is None and stack.effort_weights is not None  # kept for the effort-matched metric
    assert stack.model_names[-1] == "unit_favourability" and "dist_minfile" not in stack.model_names
    assert "dem" not in stack.names  # the DEM feeds terrain features, not a raw raster feature
    assert set(input_datasets(stack)) >= {"minfile", "faults", "bedrock_geology", "rgs", "dem"}
    assert "aris_index" not in input_datasets(stack)


def test_effort_modes(project, grid):
    feature = build_stack(project, grid, evidence_cfg(effort="feature"))
    assert {"tenure_density", "aris_density"} <= set(feature.names) and feature.bias is None
    assert feature.effort_weights is not None  # still available for the effort-matched metric
    bias = build_stack(project, grid, evidence_cfg(effort="bias"))
    assert "tenure_density" not in bias.names and "weights background" in bias.skipped["tenure_density"]
    assert bias.bias is not None and bias.bias.min() > 0
    assert {"aris_index", "mto_tenure"} <= set(input_datasets(bias))


def test_occurrence_classes(project, grid):
    everything = build_stack(project, grid, evidence_cfg())
    metallic = build_stack(project, grid, evidence_cfg(occurrence_class="metallic"))
    industrial = build_stack(project, grid, evidence_cfg(occurrence_class="industrial"))
    assert 0 < len(industrial.positives) < len(metallic.positives) < len(everything.positives)
    assert set(metallic.positives).isdisjoint(industrial.positives) or len(everything.positives) < 60


def test_occurrence_class_needs_deposit_codes(tmp_path):
    project = build_project(tmp_path)  # the plain synthetic minfile has no DEPOSIT_TYPE_CODE1
    g = build_grid(project.read_aoi(), CRS, 100.0, 10**6)
    with pytest.raises(FeatureError, match="DEPOSIT_TYPE_CODE1"):
        build_stack(project, g, evidence_cfg(occurrence_class="metallic"))


def test_no_evidence_falls_back_to_effort_features(tmp_path):
    project = build_project(tmp_path, layers=("minfile", "mto_tenure", "aris_index"), raster=False)
    g = build_grid(project.read_aoi(), CRS, 100.0, 10**6)
    stack = build_stack(project, g, evidence_cfg())
    assert stack.names == ["tenure_density", "aris_density"] and stack.bias is None
    assert "fell back to exploration effort" in stack.skipped["evidence"]


def test_legacy_config_reproduces_the_old_feature_set(project, grid):
    stack = build_stack(project, grid, small_cfg(unit_favourability=False))
    assert stack.names[:2] == ["tenure_density", "aris_density"]
    assert "rgs_zinc" in stack.names and not any(n.startswith("rgs_idx_") for n in stack.names)  # raw elements
    assert stack.model_names[-1] == "dist_minfile"


# --- the command --------------------------------------------------------------------------


def test_target_command_with_class_and_effort(project):
    result = runner.invoke(app, ["target", "demo", "--class", "metallic", "--effort", "bias", "--folds", "3"])
    assert result.exit_code == 0, result.output
    assert "Known occurrences used as positives (metallic):" in result.output
    assert "Background cells drawn in proportion to exploration effort" in result.output
    assert "effort-matched AUC" in result.output
    out = project.root / "target"
    grid_json = json.loads((out / "grid.json").read_text())
    assert grid_json["occurrence_class"] == "metallic" and grid_json["background_weighted_by_effort"] is True
    assert "unit_favourability" in grid_json["features_used"]
    cv = json.loads((out / "cv_report.json").read_text())
    assert cv["auc_effort_matched_mean"] is not None
    with rasterio.open(out / "feature_stack.tif") as ds:
        assert ds.descriptions[-1] == "unit_favourability"
    imp = pd.read_csv(out / "feature_importance.csv")
    assert set(imp["feature"]) == set(grid_json["features_used"])
    record = next(r for r in project.provenance.read() if r.dataset == "target_prospectivity")
    assert "Access Only" in record.license and "aris_index" in record.license  # effort weights use ARIS


def test_target_command_rejects_bad_options(project):
    result = runner.invoke(app, ["target", "demo", "--effort", "sometimes"])
    assert result.exit_code != 0 and "effort" in result.output


# --- tiny targets: smoothing and the minimum size ---------------------------------------------


@pytest.fixture
def tiny_grid():
    """A 10 x 10 grid of 100 m cells."""
    return build_grid(gpd.GeoDataFrame(geometry=[box(X0, Y0, X0 + 1000, Y0 + 1000)], crs=CRS), CRS, 100.0, 10**6)


def _spotted_score(tiny_grid):
    """A 4 x 4 plateau of 0.8 with two stray 0.9 cells elsewhere, zeros everywhere else."""
    score = np.zeros(tiny_grid.n_cells)
    cell = lambda r, c: int(tiny_grid.index[r, c])
    for r in range(2, 6):
        for c in range(2, 6):
            score[cell(r, c)] = 0.8
    score[cell(8, 8)] = score[cell(0, 9)] = 0.9
    return score, np.full(tiny_grid.n_cells, 0.05), cell


def test_minimum_size_drops_stray_cells_and_counts_them(tiny_grid):
    from gossan.target.outputs import top_percentile_polygons

    score, spread, cell = _spotted_score(tiny_grid)
    positives = np.array([cell(3, 3), cell(8, 8)])
    kept, _ = top_percentile_polygons(tiny_grid, score, spread, positives, 85.0)  # keeps the 18 cells >= 0.8
    assert len(kept) == 3 and kept.attrs["n_dropped_small"] == 0  # no limit: strays are targets
    filtered, _ = top_percentile_polygons(tiny_grid, score, spread, positives, 85.0, min_cells=4)
    assert len(filtered) == 1 and filtered.attrs["n_dropped_small"] == 2
    only = filtered.iloc[0]
    assert only["rank"] == 1 and only["target_id"] == "T001" and only["n_cells"] == 16
    assert only["n_known_occurrences"] == 1  # the occurrence in the dropped stray cell is not counted


def test_everything_below_the_minimum_gives_no_targets_and_a_count(tiny_grid):
    from gossan.target.outputs import top_percentile_polygons

    score, spread, cell = _spotted_score(tiny_grid)
    none, _ = top_percentile_polygons(tiny_grid, score, spread, np.array([cell(3, 3)]), 85.0, min_cells=100)
    assert len(none) == 0 and none.attrs["n_dropped_small"] == 3


def test_smoothing_merges_cells_that_sit_close_together(tiny_grid):
    from gossan.target.outputs import smooth_scores, top_percentile_polygons

    score = np.zeros(tiny_grid.n_cells)
    for r, c in [(4, 3), (4, 5), (5, 4), (3, 4)]:  # four high cells around an empty centre
        score[int(tiny_grid.index[r, c])] = 0.9
    spread = np.zeros(tiny_grid.n_cells)
    raw, _ = top_percentile_polygons(tiny_grid, score, spread, np.array([], dtype=int), 97.0)
    assert len(raw) == 4  # four separate single cells
    merged, thr = top_percentile_polygons(tiny_grid, score, spread, np.array([], dtype=int), 97.0, smooth_cells=1.2)
    assert len(merged) == 1 and merged.iloc[0]["n_cells"] == 3  # the top 3% of 100 cells, now one polygon
    x, y = tiny_grid.xy[int(tiny_grid.index[4, 4])]  # the empty centre of the ring (raw score 0) is selected
    assert merged.geometry.iloc[0].intersects(Point(x, y)) or merged.geometry.iloc[0].distance(Point(x, y)) <= 100
    assert 0 <= merged.iloc[0]["max_score"] <= 0.9  # reported scores are the unsmoothed ones
    smoothed = smooth_scores(tiny_grid, score, 1.2)
    assert smoothed.max() < 0.9 and thr == pytest.approx(np.percentile(smoothed, 97.0))


def test_smoothing_does_not_pull_edge_cells_toward_zero(tiny_grid):
    from gossan.target.outputs import smooth_scores

    flat = np.full(tiny_grid.n_cells, 0.5)
    assert np.allclose(smooth_scores(tiny_grid, flat, 2.0), 0.5)  # corners and edges included
    assert smooth_scores(tiny_grid, flat, 0.0) is flat
    rows = np.arange(0, tiny_grid.n_cells, 2)
    assert np.allclose(smooth_scores(tiny_grid, flat[rows], 2.0, rows=rows), 0.5)  # a subset of cells, same rule


def test_target_command_applies_the_default_minimum_and_reports_it(project):
    result = runner.invoke(app, ["target", "demo", "--folds", "3"])
    assert result.exit_code == 0, result.output  # even if nothing survives the 20 ha limit
    grid_json = json.loads((project.root / "target" / "grid.json").read_text())
    assert grid_json["min_target_ha"] == 20.0 and grid_json["smooth_m"] == 0.0
    assert f"{grid_json['n_targets_dropped_small']} under 20 ha dropped" in result.output
    gpkg = project.root / "target" / "targets.gpkg"
    if gpkg.exists():
        assert (gpd.read_file(gpkg, layer="targets_top")["area_ha"] >= 20.0).all()


def test_target_command_options_set_minimum_size_and_smoothing(project):
    result = runner.invoke(app, ["target", "demo", "--folds", "3", "--min-target-ha", "2", "--smooth-m", "300"])
    assert result.exit_code == 0, result.output
    grid_json = json.loads((project.root / "target" / "grid.json").read_text())
    assert grid_json["min_target_ha"] == 2.0 and grid_json["smooth_m"] == 300.0
    assert "score smoothed over 300 m" in result.output and "under 2 ha dropped" in result.output
    assert "smoothed over 300 m: top-5% capture" in result.output
    cv = json.loads((project.root / "target" / "cv_report.json").read_text())
    assert cv["capture_top5_smoothed_mean"] is not None
    targets = gpd.read_file(project.root / "target" / "targets.gpkg", layer="targets_top")
    assert len(targets) >= 1 and (targets["area_ha"] >= 2.0).all()
