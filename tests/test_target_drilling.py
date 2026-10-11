"""Digitized drill holes in the prospectivity model: reliable negatives, extra positives, and the
cross-validation measure against known-barren cells."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyogrio
import pytest
import rasterio
from sklearn.metrics import roc_auc_score
from synthetic import CRS, X0, build_project, runner, small_cfg

from gossan.cli import app
from gossan.core.project import Project
from gossan.digitize.db import COLUMNS
from gossan.target.cv import MIN_TEST_BARREN, auc_vs_barren
from gossan.target.features import build_stack, input_datasets
from gossan.target.grid import build_grid
from gossan.target.model import pu_bagging
from gossan.target.run import run_target


def long_rows(aris: int, hole: str, items: list[tuple]) -> list[dict]:
    return [
        {
            "hole_id": hole, "from_m": f, "to_m": t, "attribute": a, "unit": u, "value": v, "flag": "", "raw_value": str(v),
            "aris_number": aris, "page": 1, "review_item": "R0001", "source_file": "x.csv", "approved_at": "2026-10-11", "approved_by": "t",
        }
        for a, v, u, f, t in items
    ]  # fmt: skip


def drill_rows(holes: list[tuple[str, float, float, str]]) -> tuple[list[dict], list[dict]]:
    """(collar rows, assay rows) for holes given as (id, x, y, 'min'|'barren')."""
    collars, assays = [], []
    for hole, x, y, kind in holes:
        collars += long_rows(1, hole, [("Easting", x, "m", None, None), ("Northing", y, "m", None, None)])
        grades = [900] + [10] * 5 if kind == "min" else [10] * 6  # Au in ppb
        assays += long_rows(1, hole, [("Au", g, "ppb", i * 5.0, (i + 1) * 5.0) for i, g in enumerate(grades)])
    return collars, assays


def write_drilling(project: Project, holes) -> None:
    collars, assays = drill_rows(holes)
    for name, rows in (("collar", collars), ("assay", assays)):
        df = pd.DataFrame(rows, columns=list(COLUMNS)).astype({"aris_number": "Int64", "page": "Int64"})
        pyogrio.write_dataframe(df, project.data_gpkg, layer=name, driver="GPKG")


def east_cells(project: Project, n: int, *, free_of: np.ndarray, west: bool = False) -> list[tuple[float, float]]:
    """Centres of ``n`` cells in the east (or west) third that hold no known occurrence."""
    grid = build_grid(project.read_aoi(), CRS, 100.0, 10**6)
    xy = grid.xy
    side = (xy[:, 0] < X0 + 1300) if west else (xy[:, 0] > X0 + 2700)
    choose = np.flatnonzero(side & ~np.isin(np.arange(grid.n_cells), free_of))
    picked = choose[np.linspace(0, len(choose) - 1, n).astype(int)]
    return [(float(xy[i, 0]), float(xy[i, 1])) for i in picked]


@pytest.fixture
def project(tmp_path) -> Project:
    proj = build_project(tmp_path, n_occ=60, seed=4)
    proj.config.target = small_cfg(drill_labels="evaluate", min_drill_negatives=5)
    proj.config.save(proj.root)
    return Project(proj.root)


def positives_of(project: Project) -> np.ndarray:
    grid = build_grid(project.read_aoi(), CRS, 100.0, 10**6)
    return build_stack(project, grid, small_cfg(drill_labels="off")).positives


def with_drilling(project: Project, mode: str, *, barren_in_east=True, n_barren=10, extra_min=1) -> tuple[Project, list]:
    pos = positives_of(project)
    barren = east_cells(project, n_barren, free_of=pos, west=not barren_in_east)
    grid = build_grid(project.read_aoi(), CRS, 100.0, 10**6)
    barren_idx = grid.cell_of(np.array([x for x, _ in barren]), np.array([y for _, y in barren]))
    mineral = east_cells(project, extra_min, free_of=np.union1d(pos, barren_idx)) if extra_min else []  # never in a barren hole's cell
    holes = [(f"B{i}", x, y, "barren") for i, (x, y) in enumerate(barren)] + [(f"M{i}", x, y, "min") for i, (x, y) in enumerate(mineral)]
    write_drilling(project, holes)
    project.config.target = small_cfg(drill_labels=mode, min_drill_negatives=5)
    project.config.save(project.root)
    return Project(project.root), holes


def stack_for(project: Project):
    return build_stack(project, build_grid(project.read_aoi(), CRS, 100.0, 10**6), project.config.target)


# --- building blocks ------------------------------------------------------------------------------------------


def test_auc_against_barren_needs_enough_held_out_barren_cells():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    barren = np.array([0, 0, 1, 1, 1, 0, 0, 0], dtype=bool)
    score = np.array([0.9, 0.8, 0.1, 0.2, 0.3, 0.95, 0.6, 0.85])  # unlabeled cells 5 and 7 outrank the occurrences
    out = auc_vs_barren(score, y, barren)
    assert out == {"auc_vs_barren": 1.0, "n_test_barren": 3}  # against barren cells only, not the unlabeled ones
    assert roc_auc_score(y, score) < 1.0  # the plain AUC is pulled down by those unlabeled cells
    assert auc_vs_barren(score, y, np.array([0, 0, 1, 1, 0, 0, 0, 0], dtype=bool)) is None  # only 2 < MIN_TEST_BARREN
    assert MIN_TEST_BARREN == 3 and auc_vs_barren(score, np.zeros(8), barren) is None  # no held-out occurrence
    both = barren.copy()
    both[0] = True  # a cell that is both an occurrence and barren counts as the occurrence
    assert auc_vs_barren(score, y, both)["n_test_barren"] == 3


def test_negatives_join_every_bag_and_are_scored_by_all_forests():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = np.zeros(300, dtype=int)
    y[:20] = 1
    rows = np.arange(300)
    neg = np.arange(100, 110)
    res = pu_bagging(X, y, rows, rows, n_bags=6, n_trees=5, min_samples_leaf=1, unlabeled_ratio=1.0, seed=1, negatives=neg)
    assert (res.n_scoring_bags[neg] == 0).all() and np.isfinite(res.score[neg]).all()  # always in the bag: no out-of-bag score, so all forests score them
    assert (res.n_scoring_bags[20:100] < 6).any()  # ordinary unlabeled cells are still out of some bags
    outside = pu_bagging(X, y, np.arange(150), rows, n_bags=4, n_trees=5, min_samples_leaf=1, unlabeled_ratio=1.0, seed=1, negatives=np.array([200, 201]))
    assert (outside.n_scoring_bags[[200, 201]] == 4).all()  # negatives outside the training rows are ignored, not trained on


# --- the stack ----------------------------------------------------------------------------------------------------


def test_evaluate_mode_finds_the_barren_cells_but_leaves_the_training_alone(project):
    base_positives = positives_of(project)
    project, _ = with_drilling(project, "evaluate")
    stack = stack_for(project)
    assert len(stack.barren_cells) == 10 and stack.negatives is None and stack.n_drill_positives_added == 0
    assert np.array_equal(stack.positives, base_positives)  # the mineralized hole is not added in evaluate mode
    assert "aris_pdfs" in input_datasets(stack)  # drill tables come from ARIS: Access Only


def test_train_mode_adds_negatives_and_the_mineralized_hole_as_a_positive(project):
    base_positives = positives_of(project)
    project, _ = with_drilling(project, "train")
    stack = stack_for(project)
    assert np.array_equal(stack.negatives, stack.barren_cells) and len(stack.negatives) == 10
    assert stack.n_drill_positives_added == 1 and len(stack.positives) == len(base_positives) + 1
    assert not np.isin(stack.negatives, stack.positives).any()


def test_conflicting_evidence_resolves_to_mineralization(project):
    pos = positives_of(project)
    grid = build_grid(project.read_aoi(), CRS, 100.0, 10**6)
    cells = east_cells(project, 8, free_of=pos)
    holes = [(f"B{i}", x, y, "barren") for i, (x, y) in enumerate(cells)]
    ox, oy = (float(v) for v in grid.xy[pos[0]])
    holes += [("ONOCC", ox, oy, "barren"), ("SAME-B", *cells[0], "barren"), ("SAME-M", *cells[0], "min")]  # barren on an occurrence; both in one cell
    write_drilling(project, holes)
    project.config.target = small_cfg(drill_labels="train", min_drill_negatives=3)
    project.config.save(project.root)
    stack = stack_for(Project(project.root))
    cell0 = grid.cell_of(np.array([cells[0][0]]), np.array([cells[0][1]]))[0]
    assert cell0 not in stack.barren_cells and cell0 in stack.positives  # the mineralized hole wins its cell
    assert not np.isin(stack.barren_cells, pos).any()  # nor is a known occurrence ever a negative


def test_too_few_barren_cells_are_not_used_and_say_why(project):
    project, _ = with_drilling(project, "train", n_barren=3, extra_min=0)
    stack = stack_for(project)
    assert stack.barren_cells is None and stack.negatives is None
    assert "only 3 cell(s) hold a barren drill hole (need 5)" in stack.drill_note


def test_off_ignores_drilling_and_industrial_models_ignore_mineralized_holes(project):
    project, _ = with_drilling(project, "off")
    stack = stack_for(project)
    assert stack.barren_cells is None and stack.drill_note == "" and stack.n_drill_positives_added == 0
    import geopandas as gpd

    minfile = gpd.read_file(project.data_gpkg, layer="minfile")  # deposit-type codes, so occurrence classes exist
    minfile["DEPOSIT_TYPE_CODE1"] = np.where(np.arange(len(minfile)) % 2 == 0, "F02", "I05")  # half industrial, half metallic
    minfile.to_file(project.data_gpkg, layer="minfile", driver="GPKG", engine="pyogrio")
    for cls, added in (("all", 1), ("metallic", 1), ("industrial", 0)):  # drill grades say nothing about gypsum or REE
        project.config.target = small_cfg(drill_labels="train", occurrence_class=cls)
        project.config.save(project.root)
        assert stack_for(Project(project.root)).n_drill_positives_added == added, cls


def test_no_digitized_tables_changes_nothing(project):
    stack = stack_for(project)
    assert stack.barren_cells is None and stack.drill_note == "no approved drill tables"


# --- the model and the command ----------------------------------------------------------------------------------------


def test_barren_holes_pull_down_the_score_where_they_sit_when_trained_on(project):
    project, _ = with_drilling(project, "evaluate", extra_min=0)
    grid = build_grid(project.read_aoi(), CRS, 100.0, 10**6)
    cells = stack_for(project).barren_cells

    def score_at_barren(mode: str) -> float:
        project.config.target = small_cfg(drill_labels=mode, min_drill_negatives=5, n_bags=8)
        project.config.save(project.root)
        run_target(Project(project.root), project.config.target, cross_validate=False)
        with rasterio.open(project.root / "target" / "prospectivity.tif") as ds:
            raster = ds.read(1, masked=True).filled(np.nan)
        return float(np.nanmean(raster[grid.rows_cols[0][cells], grid.rows_cols[1][cells]]))

    assert score_at_barren("train") < score_at_barren("evaluate")  # in the high-scoring east, where it matters


def test_cross_validation_reports_the_auc_against_barren_cells(project):
    project, _ = with_drilling(project, "evaluate", barren_in_east=False, n_barren=14, extra_min=0)  # barren holes in the low-scoring west
    result = run_target(Project(project.root), project.config.target, cross_validate=True)
    cv = result.cv
    assert cv["auc_vs_barren_mean"] is not None and cv["n_barren_cells"] == 14
    assert cv["auc_vs_barren_mean"] > 0.7  # occurrences (east) rank above barren ground (west) in held-out blocks
    grid_json = json.loads((project.root / "target" / "grid.json").read_text())
    assert grid_json["drill_labels"] == "evaluate" and grid_json["n_barren_drill_cells"] == 14
    record = next(r for r in project.provenance.read() if r.dataset == "target_prospectivity")
    assert "Access Only" in record.license  # the drill tables are digitized ARIS


def test_target_command_says_how_the_drilling_was_used(project):
    project, _ = with_drilling(project, "evaluate", barren_in_east=False, n_barren=14, extra_min=0)
    out = runner.invoke(app, ["target", "demo", "--folds", "3"])
    assert out.exit_code == 0, out.output
    assert "Drill results: 14 cell(s) hold a barren hole; used to evaluate only" in out.output
    assert "AUC against known-barren drill cells" in out.output
    project.config.target = small_cfg(drill_labels="train", min_drill_negatives=5)
    project.config.save(project.root)
    out = runner.invoke(app, ["target", "demo", "--no-cv"])
    assert "training on 14 barren cell(s) as negatives and 0 extra positive(s)" in out.output
