import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import Point, Polygon, box
from typer.testing import CliRunner

from gossan.cli import app
from gossan.core.config import TargetConfig
from gossan.core.project import Project
from gossan.target import cv as cvmod
from gossan.target import features as feat
from gossan.target.features import (
    FeatureError,
    FeatureStack,
    build_stack,
    distance_to_positives,
    idw,
)
from gossan.target.grid import GridError, build_grid
from gossan.target.model import pu_bagging
from gossan.target.outputs import top_percentile_polygons, write_cog

runner = CliRunner()
CRS = "EPSG:32610"
X0, Y0, W, H = 690_000.0, 5_617_000.0, 4000.0, 3000.0  # a 4 km x 3 km AOI


def small_cfg(**kw) -> TargetConfig:
    base = {"resolution_m": 100.0, "n_bags": 4, "n_trees": 15, "folds": 3, "seed": 1}
    return TargetConfig(**{**base, **kw})


def make_aoi(path: Path, geometry=None) -> Path:
    gpd.GeoDataFrame({"id": [1]}, geometry=[geometry or box(X0, Y0, X0 + W, Y0 + H)], crs=CRS).to_file(path, driver="GPKG")
    return path


@pytest.fixture
def make_project(tmp_path):
    def build(*, layers=("minfile", "mto_tenure", "aris_index", "rgs"), raster=True, n_occ=40, geometry=None, seed=0) -> Project:
        aoi = make_aoi(tmp_path / "aoi.gpkg", geometry)
        assert runner.invoke(app, ["new", "demo", "--aoi", str(aoi)]).exit_code == 0
        project = Project(Path("demo"))
        assert project.config.crs == CRS
        rng = np.random.default_rng(seed)
        gpkg = project.data_gpkg

        def write(name, gdf):
            gdf.to_file(gpkg, layer=name, driver="GPKG", engine="pyogrio")

        if "minfile" in layers:  # occurrences prefer the east, where the planted signal is high
            xs = np.concatenate([rng.uniform(X0 + 0.55 * W, X0 + W, int(n_occ * 0.85)), rng.uniform(X0, X0 + W, n_occ - int(n_occ * 0.85))])
            ys = rng.uniform(Y0, Y0 + H, n_occ)
            write("minfile", gpd.GeoDataFrame({"MINFILE_NUMBER": range(n_occ)}, geometry=[Point(x, y) for x, y in zip(xs, ys, strict=True)], crs=CRS))
        if "mto_tenure" in layers:
            write("mto_tenure", gpd.GeoDataFrame({"TENURE_NUMBER_ID": [1, 2]}, geometry=[box(X0 + 2500, Y0 + 500, X0 + 3600, Y0 + 1500), box(X0 + 200, Y0 + 2000, X0 + 900, Y0 + 2800)], crs=CRS))
        if "aris_index" in layers:
            ax, ay = rng.uniform(X0, X0 + W, 30), rng.uniform(Y0, Y0 + H, 30)
            write("aris_index", gpd.GeoDataFrame({"ARIS_NUMBER": range(30)}, geometry=[Point(x, y) for x, y in zip(ax, ay, strict=True)], crs=CRS))
        if "rgs" in layers:
            gx, gy = rng.uniform(X0, X0 + W, 25), rng.uniform(Y0, Y0 + H, 25)
            write(
                "rgs",
                gpd.GeoDataFrame(
                    {"COPPER_AAS_PPM": 20 + (gx - X0) / 40, "COPPER_ICPMS_PPM": [np.nan] * 25, "GOLD_INAA_PPB": rng.uniform(1, 9, 25)},
                    geometry=[Point(x, y) for x, y in zip(gx, gy, strict=True)], crs=CRS,
                ),
            )  # fmt: skip
        if raster:
            (project.root / "rasters").mkdir()
            cols = np.linspace(0, 1, 80, dtype=np.float32)
            data = np.tile(cols, (60, 1))  # values rise from west to east
            with rasterio.open(
                project.root / "rasters" / "synth_magnetics.tif", "w", driver="GTiff", width=80, height=60, count=1,
                dtype="float32", crs=CRS, transform=from_bounds(X0 - 200, Y0 - 200, X0 + W + 200, Y0 + H + 200, 80, 60), nodata=-9999.0,
            ) as ds:  # fmt: skip
                ds.write(data, 1)
        project.config.target = small_cfg()
        project.config.save(project.root)
        return Project(project.root)

    return build


# --- grid ---------------------------------------------------------------------------------


def test_grid_is_snapped_and_masks_the_aoi(tmp_path):
    aoi = gpd.GeoDataFrame(geometry=[box(X0 + 30, Y0 + 20, X0 + W, Y0 + H)], crs=CRS)
    g = build_grid(aoi, CRS, 100.0, 10**6)
    assert (g.width, g.height) == (40, 30) and g.n_cells == 1200
    assert g.transform.c == X0 and g.transform.f == Y0 + H  # snapped to whole multiples of the resolution
    assert g.bounds == (X0, Y0, X0 + W, Y0 + H)
    xy = g.xy
    assert xy.shape == (1200, 2) and tuple(xy[0]) == (X0 + 50, Y0 + H - 50)  # first cell: top-left centre
    cells = g.cell_of(np.array([X0 + 50, X0 + W - 10, X0 - 5]), np.array([Y0 + H - 50, Y0 + 10, Y0 + 10]))
    assert list(cells) == [0, g.n_cells - 1, -1]
    values = np.arange(g.n_cells, dtype=float)
    assert np.array_equal(g.compact(g.to_raster(values)), values)  # to_raster/compact round-trip


def test_cell_centres_decide_membership_in_an_irregular_aoi():
    triangle = Polygon([(X0, Y0), (X0 + W, Y0), (X0, Y0 + H)])
    g = build_grid(gpd.GeoDataFrame(geometry=[triangle], crs=CRS), CRS, 100.0, 10**6)
    full = g.width * g.height
    assert 0.45 * full < g.n_cells < 0.55 * full and not g.mask[0, -1]  # top-right corner is outside
    assert g.cell_of(np.array([X0 + W - 50]), np.array([Y0 + H - 50]))[0] == -1


def test_grid_refuses_too_many_cells_or_none(tmp_path):
    aoi = gpd.GeoDataFrame(geometry=[box(X0, Y0, X0 + W, Y0 + H)], crs=CRS)
    with pytest.raises(GridError, match="coarser --resolution"):
        build_grid(aoi, CRS, 1.0, 1_000_000)
    tiny = gpd.GeoDataFrame(geometry=[box(X0 + 10, Y0 + 10, X0 + 20, Y0 + 20)], crs=CRS)
    with pytest.raises(GridError, match="no 100 m cell centre"):
        build_grid(tiny, CRS, 100.0, 10**6)


# --- feature building blocks ---------------------------------------------------------------


@pytest.fixture
def grid():
    return build_grid(gpd.GeoDataFrame(geometry=[box(X0, Y0, X0 + 1000, Y0 + 1000)], crs=CRS), CRS, 100.0, 10**6)  # 10 x 10


def test_idw_is_exact_at_samples_and_nan_beyond_the_radius(grid):
    sx, sy = np.array([X0 + 50, X0 + 950]), np.array([Y0 + 950, Y0 + 50])  # top-left and bottom-right cell centres
    out = idw(grid, sx, sy, np.array([10.0, 20.0]), k=2, radius=800)
    assert out[grid.cell_of(sx[:1], sy[:1])[0]] == pytest.approx(10.0)
    assert out[grid.cell_of(sx[1:], sy[1:])[0]] == pytest.approx(20.0)
    centre = out[grid.cell_of(np.array([X0 + 550]), np.array([Y0 + 450]))[0]]
    assert 10.0 < centre < 20.0  # a weighted blend of the two
    far = idw(grid, sx[:1], sy[:1], np.array([10.0]), k=1, radius=300)
    assert np.isnan(far).sum() > 70 and np.isfinite(far).sum() >= 1  # most cells are beyond 300 m of the one sample


def test_distance_feature_never_measures_an_occurrence_to_itself(grid):
    pos = np.array([0, 55, 99])
    loo = distance_to_positives(grid, pos, exclude_self=True)
    plain = distance_to_positives(grid, pos, exclude_self=False)
    assert plain[pos].tolist() == [0.0, 0.0, 0.0]  # naive distance would reveal the label
    assert (loo[pos] > 0).all()
    xy = grid.xy
    assert loo[0] == pytest.approx(min(np.hypot(*(xy[0] - xy[55])), np.hypot(*(xy[0] - xy[99]))))
    others = np.setdiff1d(np.arange(grid.n_cells), pos)
    assert np.allclose(loo[others], plain[others])  # every other cell: distance to the nearest occurrence
    with pytest.raises(FeatureError, match="at least two"):
        distance_to_positives(grid, np.array([3]), exclude_self=True)


# --- the stack ----------------------------------------------------------------------------


def test_stack_builds_every_available_feature(make_project):
    project = make_project()
    g = build_grid(project.read_aoi(), CRS, 100.0, 10**6)
    stack = build_stack(project, g, small_cfg())
    assert stack.names == ["tenure_density", "aris_density", "rgs_copper", "rgs_gold", "synth_magnetics"]
    assert stack.X.shape == (1200, 5) and stack.X.dtype == np.float32 and np.isfinite(stack.X).all()
    assert stack.model_names[-1] == "dist_minfile"  # the label-derived feature is last
    # Skipped, with reasons: elements with no analysis column in the rgs layer.
    assert "no SILVER analysis column" in stack.skipped["rgs_silver"]
    assert 0 < stack.n_occurrences_total == 40 and stack.n_occurrences_outside == 0 and 20 < len(stack.positives) <= 40
    mag = stack.X[:, stack.names.index("synth_magnetics")]
    xs = g.xy[:, 0]
    assert np.corrcoef(mag, xs)[0, 1] > 0.99  # resampled raster reproduces the west-to-east ramp
    tenure = stack.X[:, stack.names.index("tenure_density")]
    assert 0 <= tenure.min() and tenure.max() <= 1 and tenure.max() > 0.2  # smoothed over a 2 km window: peaks inside a tenure box
    aris = stack.X[:, stack.names.index("aris_density")]
    assert aris.min() >= 0 and aris.max() > aris.mean() * 2  # peaks around report points


def test_missing_layers_are_skipped_not_fatal(make_project):
    project = make_project(layers=("minfile",), raster=False)
    g = build_grid(project.read_aoi(), CRS, 100.0, 10**6)
    stack = build_stack(project, g, small_cfg())
    assert stack.names == [] and stack.X.shape == (1200, 0)
    for key in ("tenure_density", "aris_density", "rgs", "rasters"):
        assert key in stack.skipped
    assert "gossan harvest --source mto_tenure" in stack.skipped["tenure_density"]


def test_no_minfile_layer_means_no_labels(make_project):
    project = make_project(layers=("mto_tenure",), raster=False)
    g = build_grid(project.read_aoi(), CRS, 100.0, 10**6)
    with pytest.raises(FeatureError, match="training labels"):
        build_stack(project, g, small_cfg())


def test_occurrences_outside_the_aoi_and_sharing_a_cell_are_counted_once(make_project):
    project = make_project(layers=(), raster=False)
    pts = [Point(X0 + 50, Y0 + 50), Point(X0 + 60, Y0 + 60), Point(X0 + 5000, Y0 + 50), Point(X0 + 950, Y0 + 950)]  # 2 in one cell, 1 outside
    gpd.GeoDataFrame({"a": range(4)}, geometry=pts, crs=CRS).to_file(project.data_gpkg, layer="minfile", driver="GPKG", engine="pyogrio")
    g = build_grid(project.read_aoi(), CRS, 100.0, 10**6)
    stack = build_stack(project, g, small_cfg())
    assert (stack.n_occurrences_total, stack.n_occurrences_outside, len(stack.positives)) == (4, 1, 2)


def test_rasters_that_are_not_features_are_skipped(make_project):
    project = make_project(layers=("minfile",), raster=False)
    folder = project.root / "rasters"
    folder.mkdir()

    def write(name, count, bounds, value=1.0):
        data = np.full((count, 20, 20), value, dtype=np.float32)
        data += np.arange(20, dtype=np.float32)  # some variation
        with rasterio.open(folder / name, "w", driver="GTiff", width=20, height=20, count=count, dtype="float32", crs=CRS, transform=from_bounds(*bounds, 20, 20), nodata=-9999.0) as ds:
            ds.write(data)

    full = (X0 - 100, Y0 - 100, X0 + W + 100, Y0 + H + 100)
    write("aris_00123_p3.tif", 1, full)  # a georeferenced report map
    write("rgb_image.tif", 3, full)  # not a single band
    write("small_patch.tif", 1, (X0, Y0, X0 + 600, Y0 + 600))  # covers about 3% of the AOI
    write("half_cover.tif", 1, (X0, Y0, X0 + W / 2, Y0 + H))  # covers half: kept, gaps filled
    g = build_grid(project.read_aoi(), CRS, 100.0, 10**6)
    stack = build_stack(project, g, small_cfg())
    assert stack.names == ["half_cover"] and 0.45 < stack.missing_fraction["half_cover"] < 0.55
    assert "3 bands" in stack.skipped["rgb_image"]
    assert "of the AOI has no value" in stack.skipped["small_patch"]
    assert "aris_00123_p3" not in stack.skipped and "aris_00123_p3" not in stack.names
    assert np.isfinite(stack.X).all()


# --- PU bagging ---------------------------------------------------------------------------


def synthetic_pu(seed=0, n_unlabeled=400, n_known=25, n_hidden=25):
    """Known positives and *hidden* positives share a signal in feature 0; the rest do not."""
    rng = np.random.default_rng(seed)
    n = n_known + n_hidden + n_unlabeled
    X = rng.normal(0, 1, (n, 3))
    X[: n_known + n_hidden, 0] += 4.0  # positives (known first, then hidden) have high feature 0
    y = np.zeros(n, dtype=int)
    y[:n_known] = 1
    return X, y, np.arange(n_known, n_known + n_hidden), np.arange(n_known + n_hidden, n)


def run_pu(X, y, **kw):
    rows = np.arange(len(y))
    params = {"n_bags": 8, "n_trees": 25, "min_samples_leaf": 2, "unlabeled_ratio": 1.0, "seed": 3, **kw}
    return pu_bagging(X, y, rows, rows, **params)


def test_pu_bagging_ranks_hidden_positives_above_the_unlabeled():
    X, y, hidden, unlabeled = synthetic_pu()
    res = run_pu(X, y)
    assert res.score.shape == (len(y),) and (res.score >= 0).all() and (res.score <= 1).all()
    assert res.score[hidden].mean() > res.score[unlabeled].mean() + 0.3  # unlabeled hidden positives are found
    assert res.importances.argmax() == 0 and res.importances.sum() == pytest.approx(1.0, abs=1e-6)
    assert (res.spread >= 0).all() and np.isfinite(res.spread).all()
    assert res.n_scoring_bags.max() <= 8
    # Positives are scored only by bags that left them out, so an in-bag score cannot memorise them.
    assert (res.n_scoring_bags[:25] < 8).all()


def test_pu_bagging_is_reproducible_and_seed_dependent():
    X, y, *_ = synthetic_pu()
    a, b, c = run_pu(X, y), run_pu(X, y), run_pu(X, y, seed=99)
    assert np.array_equal(a.score, b.score) and not np.array_equal(a.score, c.score)


def test_rows_outside_the_training_set_are_scored_by_every_bag():
    X, y, *_ = synthetic_pu()
    train = np.arange(0, len(y), 2)
    test = np.arange(1, len(y), 2)
    res = pu_bagging(X, y, train, test, n_bags=5, n_trees=10, min_samples_leaf=2, unlabeled_ratio=1.0, seed=1)
    assert (res.n_scoring_bags == 5).all()


def test_pu_bagging_input_checks():
    X, y, *_ = synthetic_pu()
    rows = np.arange(len(y))
    one_pos = np.zeros_like(y)
    one_pos[0] = 1
    with pytest.raises(ValueError, match="two positive"):
        pu_bagging(X, one_pos, rows, rows, n_bags=2, n_trees=5, min_samples_leaf=1, unlabeled_ratio=1.0, seed=0)
    with pytest.raises(ValueError, match="unlabeled"):
        pu_bagging(X, np.ones_like(y), rows, rows, n_bags=2, n_trees=5, min_samples_leaf=1, unlabeled_ratio=1.0, seed=0)


# --- spatial cross-validation -------------------------------------------------------------


def test_blocks_are_held_out_whole_and_every_cell_is_tested_once(grid):
    blocks = cvmod.make_blocks(grid, 300.0)
    assert len(np.unique(blocks)) == 16  # 1 km / 300 m -> 4 x 4 blocks
    rng = np.random.default_rng(0)
    y = np.zeros(grid.n_cells, dtype=int)
    y[rng.choice(grid.n_cells, 16, replace=False)] = 1
    folds = cvmod.spatial_folds(blocks, y, 4, seed=0)
    assert len(folds) == 4
    all_test = np.concatenate([te for _tr, te in folds])
    assert sorted(all_test) == list(range(grid.n_cells))  # a partition: each cell is tested exactly once
    for train, test in folds:
        assert set(blocks[train]).isdisjoint(blocks[test])  # no block is on both sides: no neighbours leak
        assert y[test].sum() >= 1  # positives are spread across folds


def test_buffer_removes_training_cells_near_the_test_blocks(grid):
    blocks = cvmod.make_blocks(grid, 500.0)  # 2 x 2 blocks
    y = np.zeros(grid.n_cells, dtype=int)
    train, test = np.flatnonzero(blocks != 0), np.flatnonzero(blocks == 0)
    kept = cvmod.drop_near(train, test, grid.xy, buffer_m=250.0)
    assert len(kept) < len(train)
    d = np.min(np.linalg.norm(grid.xy[kept][:, None, :] - grid.xy[test][None, :, :], axis=2), axis=1)
    assert d.min() > 250.0
    assert np.array_equal(cvmod.drop_near(train, test, grid.xy, 0.0), train) and y.sum() == 0


def test_fold_metrics():
    y = np.zeros(100, dtype=int)
    y[:5] = 1
    perfect = np.where(y == 1, 0.9, 0.1)
    m = cvmod.fold_metrics(perfect, y)
    assert m["auc"] == 1.0 and m["capture_top5"] == 1.0 and m["n_test_positives"] == 5
    inverted = cvmod.fold_metrics(1 - perfect, y)
    assert inverted["auc"] == 0.0 and inverted["capture_top5"] == 0.0
    assert cvmod.fold_metrics(perfect, np.zeros(100, dtype=int)) is None  # no held-out positives
    assert cvmod.fold_metrics(perfect, np.ones(100, dtype=int)) is None  # nothing unlabeled


def stack_for(project: Project, cfg: TargetConfig):
    g = build_grid(project.read_aoi(), CRS, cfg.resolution_m, 10**6)
    stack = build_stack(project, g, cfg)
    y = np.zeros(g.n_cells, dtype=int)
    y[stack.positives] = 1
    return g, stack, y


def test_cv_runs_and_the_distance_feature_uses_training_occurrences_only(make_project, monkeypatch):
    project = make_project()
    cfg = small_cfg()
    g, stack, y = stack_for(project, cfg)
    seen = []
    real = cvmod.pu_bagging

    def spy(X, y_, train, test, **kw):
        seen.append((X[:, -1].copy(), train.copy(), test.copy()))
        return real(X, y_, train, test, **kw)

    monkeypatch.setattr(cvmod, "pu_bagging", spy)
    result = cvmod.run_cv(stack, g, cfg, y)
    summary = result.summary()
    assert summary["n_folds_scored"] == len(result.folds) >= 2
    assert 0 <= summary["auc_mean"] <= 1 and 0 <= summary["capture_top5_mean"] <= 1
    assert summary["capture_expected_at_random"] == 0.05
    for dist, train, test in seen:
        held_out_positives = test[y[test] == 1]
        train_positives = train[y[train] == 1]
        # Held-out occurrences are not in the distance feature's reference set, so they are not at distance 0,
        # and training occurrences are measured to another occurrence, never to themselves.
        assert (dist[held_out_positives] > 0).all() and (dist[train_positives] > 0).all()
        # A held-out occurrence must be at least as far as the nearest *training* occurrence: no peeking.
        xy = g.xy
        nearest = np.min(np.linalg.norm(xy[held_out_positives][:, None, :] - xy[train_positives][None, :, :], axis=2), axis=1)
        assert np.allclose(dist[held_out_positives], nearest, atol=1e-3)


def test_cv_skips_politely_when_there_are_too_few_occurrences(make_project):
    project = make_project(n_occ=3)
    cfg = small_cfg(min_positives=3)
    g, stack, y = stack_for(project, cfg)
    result = cvmod.run_cv(stack, g, cfg, y)
    assert result.folds == [] and "too few for cross-validation" in result.skipped[0]
    assert result.summary()["auc_mean"] is None


def test_cv_is_better_than_chance_when_the_signal_is_real(make_project):
    project = make_project(n_occ=60, seed=4)
    cfg = small_cfg(n_bags=6, n_trees=30, folds=4)
    g, stack, y = stack_for(project, cfg)
    summary = cvmod.run_cv(stack, g, cfg, y).summary()
    assert summary["auc_mean"] > 0.6  # the planted east-west ramp is learnable even across blocks


# --- target polygons and the COG writer ----------------------------------------------------


def test_top_percentile_polygons_rank_and_count_known_occurrences(grid):
    score = np.zeros(grid.n_cells)
    cell = lambda r, c: int(grid.index[r, c])
    for r in range(1, 4):  # a 3 x 3 high blob
        for c in range(1, 4):
            score[cell(r, c)] = 0.9
    for r in (7, 8):  # a smaller 2 x 2 blob, lower score
        for c in (7, 8):
            score[cell(r, c)] = 0.7
    spread = np.full(grid.n_cells, 0.05)
    positives = np.array([cell(2, 2), cell(7, 7), cell(0, 9)])
    targets, threshold = top_percentile_polygons(grid, score, spread, positives, 90.0)
    assert threshold == pytest.approx(0.7)  # 13 of 100 cells are non-zero: the 90th percentile falls in the 0.7 group
    assert len(targets) == 2 and list(targets["rank"]) == [1, 2] and list(targets["target_id"]) == ["T001", "T002"]
    big = targets.iloc[0]
    assert big["n_cells"] == 9 and big["area_ha"] == pytest.approx(9 * 100 * 100 / 1e4)
    assert big["mean_score"] == pytest.approx(0.9) and big["mean_spread"] == pytest.approx(0.05)
    assert big["n_known_occurrences"] == 1 and targets.iloc[1]["n_known_occurrences"] == 1
    assert big.geometry.area == pytest.approx(9 * 100 * 100) and targets.crs.to_string() == CRS
    assert (targets["percentile"] == 90.0).all()


def test_cog_writer_keeps_names_nodata_and_layout(tmp_path, grid):
    arr = np.full(grid.shape, np.nan, dtype=np.float32)
    arr[:5] = 0.5
    path = tmp_path / "out" / "x.tif"
    write_cog(path, grid, {"a": arr, "b": arr * 2})
    with rasterio.open(path) as ds:
        assert ds.count == 2 and ds.descriptions == ("a", "b") and ds.nodata == -9999.0
        assert ds.crs.to_string() == CRS and ds.transform == grid.transform
        assert ds.tags(ns="IMAGE_STRUCTURE")["LAYOUT"] == "COG" and ds.dtypes == ("float32", "float32")
        data = ds.read(1)
        assert (data[:5] == 0.5).all() and (data[5:] == -9999.0).all()


# --- the command, end to end on a small synthetic project ----------------------------------


def run(*args):
    return runner.invoke(app, ["target", "demo", *args])


def test_target_command_writes_every_output(make_project):
    project = make_project(n_occ=60, seed=4)
    result = run("--resolution", "100", "--n-bags", "5", "--folds", "3")
    assert result.exit_code == 0, result.output
    out = project.root / "target"
    assert sorted(p.name for p in out.iterdir()) == [
        "cv_report.json", "feature_importance.csv", "feature_stack.tif", "grid.json",
        "prospectivity.tif", "targets.gpkg", "uncertainty.tif",
    ]  # fmt: skip
    assert "Grid: 40 x 30 cells (1,200) at 100 m" in result.output
    assert "Features (6): tenure_density, aris_density, rgs_copper, rgs_gold, synth_magnetics, dist_minfile" in result.output
    assert "Spatial CV" in result.output and "polygon(s) at or above the 95th percentile" in result.output

    with rasterio.open(out / "prospectivity.tif") as ds:
        assert (ds.height, ds.width, ds.count, ds.dtypes[0], ds.nodata) == (30, 40, 1, "float32", -9999.0)
        assert ds.crs.to_string() == CRS and ds.tags(ns="IMAGE_STRUCTURE")["LAYOUT"] == "COG"
        prob = ds.read(1)
    inside = prob != -9999.0
    assert inside.all()  # a rectangular AOI: every cell is scored
    assert prob.min() >= 0 and prob.max() <= 1
    east, west = prob[:, 30:].mean(), prob[:, :10].mean()
    assert east > west + 0.1  # the planted signal (occurrences cluster where the ramp is high) is recovered
    with rasterio.open(out / "uncertainty.tif") as ds:
        unc = ds.read(1)
    assert (unc >= 0).all() and unc.max() < 0.6

    with rasterio.open(out / "feature_stack.tif") as ds:
        assert ds.descriptions == ("tenure_density", "aris_density", "rgs_copper", "rgs_gold", "synth_magnetics", "dist_minfile")

    layer = gpd.read_file(out / "targets.gpkg", layer="targets_top")
    assert len(layer) >= 1 and layer.crs.to_epsg() == 32610
    assert layer["n_cells"].sum() >= 0.05 * 1200 and (layer["mean_score"] >= layer["score_threshold"]).all()
    assert list(layer["rank"]) == sorted(layer["rank"])

    importance = (out / "feature_importance.csv").read_text().splitlines()
    assert importance[0] == "feature,importance,rank" and len(importance) == 7
    assert importance[1].startswith(("synth_magnetics", "rgs_copper"))  # both carry the planted west-east signal

    cv = json.loads((out / "cv_report.json").read_text())
    assert cv["n_folds_scored"] >= 2 and cv["auc_mean"] > 0.55

    grid = json.loads((out / "grid.json").read_text())
    assert (grid["width"], grid["height"], grid["n_cells_in_aoi"], grid["resolution_m"]) == (40, 30, 1200, 100.0)
    assert grid["features_used"][-1] == "dist_minfile" and "rgs_silver" in grid["features_skipped"]
    assert grid["bounds"] == [X0, Y0, X0 + W, Y0 + H]


def test_target_run_is_logged_to_provenance(make_project):
    project = make_project(n_occ=60, seed=4)
    assert run("--n-bags", "4", "--folds", "3").exit_code == 0
    records = [r for r in project.provenance.read() if r.operation == "target_run"]
    assert {r.dataset for r in records} == {
        "target_prospectivity", "target_uncertainty", "target_feature_stack", "target_targets",
        "target_feature_importance", "target_cv_report", "target_grid",
    }  # fmt: skip
    p = records[0].parameters
    assert p["resolution_m"] == 100.0 and p["grid"] == [30, 40] and p["n_cells_in_aoi"] == 1200
    assert p["model"]["type"] == "PU-bagging random forest" and p["model"]["n_bags"] == 4 and p["model"]["seed"] == 1
    assert p["features_used"][-1] == "dist_minfile" and p["cv"]["n_folds_scored"] >= 2
    assert p["top_percentile"] == 95.0 and p["n_positives"] > 10
    assert "Access Only" in records[0].license


def test_the_same_seed_gives_the_same_map(make_project):
    project = make_project(n_occ=40)
    assert run("--n-bags", "3", "--no-cv").exit_code == 0
    with rasterio.open(project.root / "target" / "prospectivity.tif") as ds:
        first = ds.read(1)
    assert run("--n-bags", "3", "--no-cv").exit_code == 0
    with rasterio.open(project.root / "target" / "prospectivity.tif") as ds:
        assert np.array_equal(ds.read(1), first)


def test_no_cv_switch_and_resolution_argument(make_project):
    project = make_project(n_occ=40)
    result = run("--resolution", "200", "--n-bags", "3", "--no-cv")
    assert result.exit_code == 0, result.output
    assert "Spatial CV" not in result.output and "Grid: 20 x 15 cells (300) at 200 m" in result.output
    cv = json.loads((project.root / "target" / "cv_report.json").read_text())
    assert cv == {"skipped": ["cross-validation was switched off"]}


def test_irregular_aoi_leaves_nodata_outside(make_project):
    triangle = Polygon([(X0, Y0), (X0 + W, Y0), (X0, Y0 + H)])
    project = make_project(geometry=triangle, n_occ=80)
    # keep occurrences inside the triangle
    pts = gpd.read_file(project.data_gpkg, layer="minfile", engine="pyogrio")
    pts = pts[pts.within(triangle)]
    pts.to_file(project.data_gpkg, layer="minfile", driver="GPKG", engine="pyogrio")
    result = run("--n-bags", "3", "--no-cv")
    assert result.exit_code == 0, result.output
    with rasterio.open(project.root / "target" / "prospectivity.tif") as ds:
        prob = ds.read(1)
    assert prob[0, -1] == -9999.0 and prob[-1, 0] != -9999.0  # far corner outside the AOI is nodata, near corner scored
    assert 0.4 < (prob != -9999.0).mean() < 0.6


def test_runs_with_only_the_occurrences(make_project):
    project = make_project(layers=("minfile",), raster=False, n_occ=40)
    result = run("--n-bags", "3", "--no-cv")
    assert result.exit_code == 0, result.output
    assert "Features (1): dist_minfile" in result.output and "skipped tenure_density" in result.output
    grid = json.loads((project.root / "target" / "grid.json").read_text())
    assert grid["features_used"] == ["dist_minfile"]


def test_too_few_occurrences_is_a_clear_error_and_writes_nothing(make_project):
    project = make_project(n_occ=6)
    result = run()
    assert result.exit_code == 1
    assert "at least 10 are needed" in result.output and "larger AOI" in result.output
    assert not (project.root / "target").exists()


def test_a_grid_too_fine_for_the_aoi_is_refused(make_project):
    project = make_project()
    result = run("--resolution", "0.5")
    assert result.exit_code == 1 and "coarser --resolution" in result.output
    assert not (project.root / "target").exists()


def test_missing_minfile_layer_is_reported(make_project):
    make_project(layers=("mto_tenure",), raster=False)
    result = run()
    assert result.exit_code == 1 and "no minfile layer" in result.output


def test_stack_type(make_project):
    project = make_project()
    g = build_grid(project.read_aoi(), CRS, 100.0, 10**6)
    assert isinstance(build_stack(project, g, small_cfg()), FeatureStack) and feat.LABEL_FEATURE == "dist_minfile"
