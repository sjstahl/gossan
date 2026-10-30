"""The knowledge-driven score beside the random forest: rules, weights, checks, outputs, report."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pymupdf
import pytest
from analysis_project import build_analysis_project, runner

from gossan.cli import app
from gossan.core.config import KnowledgeConfig, TargetConfig
from gossan.core.project import Project
from gossan.target import knowledge as kn
from gossan.target.run import run_target


def config(**knowledge) -> TargetConfig:
    return TargetConfig(
        resolution_m=100.0, n_bags=4, n_trees=15, folds=3, seed=1, min_target_ha=0.0, knowledge=KnowledgeConfig(**knowledge)
    )  # fmt: skip


@pytest.fixture
def project(tmp_path) -> Project:
    return build_analysis_project(tmp_path)


def run(project: Project, **knowledge):
    return run_target(project, config(**knowledge), cross_validate=True)


# --- the rules ------------------------------------------------------------------------------------------------------------


def test_proximity_falls_linearly_to_nothing_at_the_range():
    assert kn.near(np.array([0.0, 750.0, 1500.0, 9000.0]), 1500.0).tolist() == [1.0, 0.5, 0.0, 0.0]


def test_rank_ramp_ignores_the_lower_half_and_reaches_one_at_the_top():
    ramp = kn.upper_ramp(np.arange(100.0))
    assert (ramp[:50] == 0).all() and ramp[-1] == pytest.approx(0.99) and np.all(np.diff(ramp[50:]) >= 0)
    assert (kn.upper_ramp(np.full(10, 3.0)) == 0).all()  # no variation, no evidence


def test_combine_is_the_weighted_mean_and_support_labels_follow_the_percentile():
    a = kn.Layer("fault", np.array([1.0, 0.0]), 3.0, 3.0, 1.0)
    b = kn.Layer("contact", np.array([0.0, 1.0]), 1.0, 1.0, 1.0)
    assert kn.combine([a, b]).tolist() == [0.75, 0.25]
    assert [kn.support_label(p) for p in (95, 80, 60, 49)] == ["supported", "supported", "partly", "forest only"]


def test_the_generic_mix_leaves_out_the_intrusive_rule_but_the_deposit_models_that_need_it_keep_it():
    assert kn.PRESETS["generic"]["intrusive"] == 0  # contradicted at Boundary, silent at two areas, informative at Stewart (docs/validation)
    assert kn.PRESETS["cu_porphyry_skarn"]["intrusive"] == 3 and kn.PRESETS["polymetallic_vein"]["intrusive"] == 1.5 and kn.PRESETS["gold_vein"]["intrusive"] == 1


def test_presets_cover_every_deposit_model_and_layer():
    assert set(kn.PRESETS) == set(KnowledgeConfig.model_fields["deposit_model"].annotation.__args__)
    assert all(set(w) == set(kn.LABELS) for w in kn.PRESETS.values())


# --- on a project ------------------------------------------------------------------------------------------------------------


def test_the_score_is_written_with_its_layers_and_independent_checks(project):
    cfg = config(fault_range_m=800.0).model_copy(update={"occurrence_class": "metallic"})  # the veins hug the thrust; the gypsum beds do not; 3 km would reach most of this AOI
    result = run_target(project, cfg, cross_validate=True)
    k = result.knowledge
    assert k["available"] and {layer["layer"] for layer in k["layers"]} >= {"fault", "contact", "geochem"}
    assert not k["circular_host_units"] and k["skipped"]["host"] == "no host_units named"
    out = project.root / "target"
    for name in ("knowledge.tif", "knowledge_report.json"):
        assert (out / name).exists(), name
    layers = {layer["layer"]: layer for layer in k["layers"]}
    assert layers["fault"]["auc"] > 0.8 and layers["contact"]["auc"] > 0.7  # the veins hug the thrust and the unit contact
    assert layers["geochem"]["auc"] < 0.5  # this synthetic geochemistry is noise, and the table says so rather than hiding it
    assert k["checks"]["n_folds_scored"] >= 2 and k["checks"]["auc_fold_mean"] is not None
    without = run_target(project, cfg.model_copy(update={"knowledge": KnowledgeConfig(fault_range_m=800.0, weights={"geochem": 0})}), cross_validate=True).knowledge
    assert without["checks"]["auc"] > 0.8 > k["checks"]["auc"]  # dropping the contradicted rule restores the score: the weights act
    assert -1 <= k["rank_correlation_with_forest"] <= 1 and 0 <= k["top_overlap_with_forest"] <= 1
    records = {r.dataset for r in project.provenance.read() if r.operation == "target_run"}
    assert {"target_knowledge", "target_knowledge_report"} <= records


def test_named_host_units_add_a_layer_and_flag_the_check_as_open_to_circularity(project):
    k = run(project, host_units=["aldridge"]).knowledge
    host = next(layer for layer in k["layers"] if layer["layer"] == "host")
    assert host["weight"] == 3 and host["auc"] > 0.6 and k["circular_host_units"] and "flatter" in k["note"]
    nothing = run(project, host_units=["nonexistent rock"]).knowledge
    assert "host" not in {layer["layer"] for layer in nothing["layers"]} and "no bedrock unit named like" in nothing["skipped"]["host"]


def test_weights_and_deposit_model_change_the_mix(project):
    base = {layer["layer"]: layer["weight"] for layer in run(project).knowledge["layers"]}
    vein = {layer["layer"]: layer["weight"] for layer in run(project, deposit_model="polymetallic_vein").knowledge["layers"]}
    assert base["fault"] == 2 and vein["fault"] == 3
    dropped = run(project, weights={"fault": 0, "contact": 5}).knowledge
    layers = {layer["layer"]: layer["weight"] for layer in dropped["layers"]}
    assert "fault" not in layers and dropped["skipped"]["fault"] == "weight 0" and layers["contact"] == 5


def test_forest_targets_gain_the_knowledge_columns_and_the_polygons_say_whether_they_overlap(project):
    run(project)
    targets = gpd.read_file(project.root / "target" / "targets.gpkg", layer="targets_top")
    assert {"knowledge_mean", "knowledge_pctile", "knowledge_support"} <= set(targets.columns)
    assert set(targets["knowledge_support"]) <= {"supported", "partly", "forest only"} and targets["knowledge_pctile"].between(0, 100).all()
    path = project.root / "target" / "knowledge_targets.gpkg"
    assert path.exists()
    polygons = gpd.read_file(path, layer="knowledge_top")
    assert polygons["target_id"].str.startswith("K").all() and {"forest_targets", "overlaps_forest_target"} <= set(polygons.columns)
    overlap = polygons[polygons["overlaps_forest_target"]]
    assert all(set(ids.split("; ")) <= set(targets["target_id"]) for ids in overlap["forest_targets"])


def test_the_decision_table_carries_the_support_and_notes_unsupported_targets(project):
    run(project)
    assert runner.invoke(app, ["analyze", "demo", "--steps", "targets"]).exit_code == 0
    import pandas as pd

    table = pd.read_csv(project.root / "analysis" / "target_decisions.csv")
    assert {"knowledge_pctile", "knowledge_support"} <= set(table.columns)
    weak = table[table["knowledge_support"] == "forest only"]
    assert all("little support" in str(n) for n in weak["notes"])


def test_switching_it_off_removes_an_earlier_runs_files(project):
    run(project)
    out = project.root / "target"
    assert (out / "knowledge.tif").exists()
    result = run(project, enabled=False)
    assert result.knowledge is None
    assert not any((out / n).exists() for n in ("knowledge.tif", "knowledge_report.json", "knowledge_targets.gpkg"))
    assert "knowledge_pctile" not in gpd.read_file(out / "targets.gpkg", layer="targets_top")


def test_without_any_evidence_layer_the_report_says_why_and_the_run_still_finishes(project, monkeypatch):
    monkeypatch.setattr(kn, "build_layers", lambda *a, **k: ([], {"fault": "no faults layer"}))
    result = run(project)
    assert result.knowledge["available"] is False and "no faults layer" in result.knowledge["reason"]
    assert not (project.root / "target" / "knowledge.tif").exists() and (project.root / "target" / "knowledge_report.json").exists()
    assert result.n_targets > 0  # the forest is untouched


def test_the_command_and_the_report_show_it(project):
    cli = runner.invoke(app, ["target", "demo", "--resolution", "100", "--n-bags", "4", "--folds", "3"])
    assert cli.exit_code == 0, cli.output
    assert "Knowledge score (generic;" in cli.output and "rank correlation with the forest" in cli.output
    assert runner.invoke(app, ["analyze", "demo", "--steps", "occurrences,targets"]).exit_code == 0
    assert runner.invoke(app, ["report", "demo", "--kind", "analysis", "--format", "pdf"]).exit_code == 0
    with pymupdf.open(project.root / "outputs" / "analysis_report.pdf") as pdf:
        text = " ".join(" ".join(page.get_text() for page in pdf).split())
    assert "A second opinion: the knowledge-driven score" in text and "It never looks at the occurrences" in text
    assert "Near faults" in text and "AUC alone" in text and "Random forest" in text and "Knowledge score" in text


# --- the empirical host layer and its held-out checks ----------------------------------------------------------------------


def test_the_empirical_host_rule_rises_with_the_units_share_of_occurrences():
    units = np.array([0] * 50 + [1] * 50 + [-1] * 10)
    positives = np.arange(50, 60)  # ten occurrences, all in unit 1
    mu = kn.empirical_host(units, positives)
    assert mu[:50].tolist() == [kn.OUTSIDE_HOST] * 50  # holds none: no better than chance
    assert mu[-10:].tolist() == [kn.OUTSIDE_HOST] * 10  # no unit: no information
    assert kn.OUTSIDE_HOST < mu[60] < 1  # holds all ten
    assert mu[55] < mu[60]  # an occurrence leaves itself out
    assert kn.empirical_host(np.array([0, 0, 1, 1]), np.array([], dtype=int)).tolist() == [kn.OUTSIDE_HOST] * 4  # no occurrences, no evidence


def test_the_default_is_still_the_named_host_units():
    assert KnowledgeConfig().host_mode == "named"


def test_an_empirical_host_layer_is_fitted_marked_in_sample_and_not_circular(project):
    k = run(project, host_mode="empirical").knowledge
    host = next(layer for layer in k["layers"] if layer["layer"] == "host")
    assert host["in_sample"] is True and host["weight"] == 3 and host["auc"] is not None
    assert k["host_mode"] == "empirical" and k["host_learned_from_occurrences"] and not k["circular_host_units"]
    assert "in-sample" in k["note"] and "flatter" not in k["note"]
    c = k["checks"]
    assert c["held_out_host"] and c["cv_gap_m"] >= 0 and c["n_folds_scored"] >= 2 and c["auc_fold_mean"] is not None
    assert c["host_layer_auc_fold_mean"] is not None
    named = run(project, host_units=["aldridge"]).knowledge
    assert not any(layer.get("in_sample") for layer in named["layers"]) and "held_out_host" not in named["checks"]


def test_empirical_mode_ignores_named_host_units(project):
    k = run(project, host_mode="empirical", host_units=["aldridge"]).knowledge
    assert k["host_learned_from_occurrences"] and not k["circular_host_units"]


def test_the_held_out_figure_does_not_see_the_test_blocks_occurrences(project):
    """Give every spatial block its own 'unit'. Learned from all the occurrences the host layer ranks them perfectly
    (in-sample); relearned from the training blocks only it knows nothing about the test blocks, and the fold figure
    falls to chance. A host layer that was not refitted would keep the flattering figure."""
    result = run_target(project, config(), cross_validate=False)
    stack, grid, cfg = result.stack, result.grid, result.cfg
    y = np.zeros(grid.n_cells, dtype=int)
    y[stack.positives] = 1
    block = cfg.block_size_m or kn.default_block_size(grid)
    units = np.unique(kn.make_blocks(grid, block), return_inverse=True)[1]
    mu = kn.empirical_host(units, stack.positives)
    for fitted in (True, False):
        layers = [kn.Layer("host", mu, 3.0, 3.0, 1.0, fitted=fitted)]
        checks = kn._checks(kn.combine(layers), y, stack, grid, cfg, layers, units)
        assert checks["auc"] > 0.7  # in-sample either way
        if fitted:
            assert checks["auc_fold_mean"] == pytest.approx(0.5, abs=0.02) and checks["host_layer_auc_fold_mean"] == pytest.approx(0.5, abs=0.02)
        else:
            assert checks["auc_fold_mean"] > 0.7  # the circular figure


def test_without_two_bedrock_units_the_empirical_layer_is_skipped_with_a_reason(project, monkeypatch):
    monkeypatch.setattr(kn, "_units", lambda *a, **k: None)
    k = run(project, host_mode="empirical").knowledge
    assert "host" not in {layer["layer"] for layer in k["layers"]} and "at least two units" in k["skipped"]["host"]
    assert "held_out_host" not in k["checks"]


def test_the_report_says_the_host_rule_was_learned(project):
    run(project, host_mode="empirical")
    assert runner.invoke(app, ["analyze", "demo", "--steps", "targets"]).exit_code == 0
    from gossan.report.analysis_render import build_analysis_report

    path = build_analysis_report(project, "pdf")[0]
    text = " ".join("\n".join(p.get_text() for p in pymupdf.open(path)).split())
    assert "host-rock rule is learned from the occurrences" in text and "(in-sample)" in text and "refitted inside each fold" in text
