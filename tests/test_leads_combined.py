"""BC leads beside the model's targets and the tenure watch: overlap, prospectivity, combined score."""

from __future__ import annotations

import json
from datetime import date

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box
from synthetic import CRS, X0, Y0, H, W, build_project, runner

from gossan.cli import app
from gossan.core.project import Project
from gossan.leads.combine import link_evidence, model_skill
from gossan.leads.generator import CSV_COLUMNS, run_leads

AS_OF = date(2026, 10, 1)
#: The east block (tenure 1001) is x 2500 to 3600, y 500 to 1500 above the AOI corner: 110 ha. This target
#: covers its x 3000 to 3600, y 700 to 1500 part: 48 ha, 43.6%.
TARGET = box(X0 + 3000, Y0 + 700, X0 + 4000, Y0 + 1500)


@pytest.fixture
def project(tmp_path) -> Project:
    proj = build_project(tmp_path, n_occ=10, seed=2, raster=False)
    return Project(proj.root)


def add_model_outputs(project: Project, auc: float | None = 0.72) -> None:
    out = project.root / "target"
    out.mkdir(exist_ok=True)
    gpd.GeoDataFrame({"rank": [1], "target_id": ["T001"], "mean_score": [0.9]}, geometry=[TARGET], crs=CRS).to_file(out / "targets.gpkg", layer="targets_top", driver="GPKG")
    ncol, nrow = int(W / 100), int(H / 100)
    score = np.tile(np.linspace(0.0, 1.0, ncol, dtype=np.float32), (nrow, 1))  # prospectivity rises from west to east
    with rasterio.open(out / "prospectivity.tif", "w", driver="GTiff", width=ncol, height=nrow, count=1, dtype="float32", crs=CRS, transform=from_origin(X0, Y0 + H, 100, 100), nodata=-9999.0) as ds:
        ds.write(score, 1)
    cv = {"folds": 3} if auc is None else {"folds": 3, "auc_mean": 0.8, "auc_effort_matched_mean": auc}
    (out / "cv_report.json").write_text(json.dumps(cv))


def run(project: Project, **leads_cfg):
    if leads_cfg:
        project.config.parameters["leads"] = leads_cfg
        project.config.save(project.root)
        project = Project(project.root)
    return run_leads(project, AS_OF)


def test_leads_alone_are_unchanged_and_say_how_to_get_the_rest(project):
    result = run(project)
    assert list(pd.read_csv(project.outputs_dir / "leads_ranked.csv").columns) == CSV_COLUMNS
    assert "combined_score" not in result.leads and any("gossan target" in n for n in result.notes)
    assert any("gossan analyze --steps tenure" in n for n in result.notes)


def test_overlap_and_prospectivity_are_measured_per_tenure(project):
    add_model_outputs(project)
    leads = run(project).leads.set_index("lead_id")
    east, west = leads.loc["1001"], leads.loc["1002"]
    assert east.target_overlap_pct == pytest.approx(43.6, abs=0.5) and east.target_overlap_ha == pytest.approx(48.0, abs=0.5)
    assert east.n_targets == 1 and east.best_target == "T001" and east.best_target_rank == 1
    assert west.target_overlap_pct == 0 and west.n_targets == 0 and west.best_target == "" and pd.isna(west.best_target_rank)
    assert east.prospectivity_p90 == pytest.approx(0.87, abs=0.04) and 80 < east.prospectivity_pctile < 95  # the hottest tenth of the east block
    assert west.prospectivity_p90 < 0.25 and west.prospectivity_pctile < 25


def test_combined_score_blends_lead_score_and_prospectivity_percentile(project):
    add_model_outputs(project)
    leads = run(project).leads.set_index("lead_id")
    for _, r in leads.iterrows():
        assert r.combined_score == pytest.approx(0.7 * r.score + 0.3 * r.prospectivity_pctile, abs=0.01)
    assert sorted(leads.combined_rank) == list(range(1, len(leads) + 1))
    assert leads.loc["1001", "combined_rank"] < leads.loc["1002", "combined_rank"]  # new holder and hot ground


def test_the_weight_decides_which_half_leads(project):
    add_model_outputs(project)
    by_score = run(project, target_weight=0.0).leads.set_index("lead_id")
    assert (by_score.combined_rank == by_score["rank"]).all() and (by_score.combined_score == by_score.score).all()
    by_merit = run(project, target_weight=1.0).leads.set_index("lead_id")
    assert by_merit.loc["1001", "combined_rank"] == 1  # the east block is the hottest ground
    assert by_merit.loc["1001", "rank"] == by_score.loc["1001", "rank"]  # the lead rank itself never moves
    with pytest.raises(ValueError, match="target_weight"):
        run(project, target_weight=1.5)


def test_watchlist_flags_and_recent_changes_come_from_the_tenure_analysis(project):
    add_model_outputs(project)
    folder = project.root / "analysis"
    folder.mkdir()
    pd.DataFrame({"tenure_number": [1001], "flags": ["recent_staking; churn"]}).to_csv(folder / "tenure_watch.csv", index=False)
    pd.DataFrame(
        {"tenure_number": [1001, 1001, 1002], "change": ["work_filed", "owner_changed", "extended"], "by": ["2026-06-01T00:00:00+00:00", "2026-09-01T00:00:00+00:00", "2026-05-02T00:00:00+00:00"]}
    ).to_csv(folder / "tenure_changes.csv", index=False)
    leads = run(project).leads.set_index("lead_id")
    assert leads.loc["1001", "watch_flags"] == "recent_staking; churn" and leads.loc["1002", "watch_flags"] == ""
    assert leads.loc["1001", "latest_change"] == "owner_changed (by 2026-09-01)"  # the latest of its two
    assert leads.loc["1002", "latest_change"] == "extended (by 2026-05-02)"


def test_a_weak_model_is_called_out_and_its_skill_is_recorded(project):
    add_model_outputs(project, auc=0.55)
    result = run(project)
    assert any("weak or unmeasured (AUC 0.55)" in n for n in result.notes)
    record = [r for r in project.provenance.read() if r.operation == "write_leads_csv"][-1]
    assert record.parameters["model_auc"] == 0.55 and "combined_score" in record.parameters["linked_columns"]
    add_model_outputs(project, auc=0.72)
    assert not any("weak" in n for n in run(project).notes)
    add_model_outputs(project, auc=None)
    assert model_skill(project) is None and any("unmeasured" in n for n in run(project).notes)


def test_csv_and_layer_carry_the_new_columns(project):
    add_model_outputs(project)
    run(project)
    csv = pd.read_csv(project.outputs_dir / "leads_ranked.csv")
    assert {"target_overlap_pct", "combined_score", "combined_rank"} <= set(csv.columns)  # no tenure analysis in this project: no watch columns
    layer = gpd.read_file(project.data_gpkg, layer="leads")
    assert "combined_score" in layer and "best_target" in layer
    assert list(csv.columns[: len(CSV_COLUMNS)]) == CSV_COLUMNS  # the old columns keep their places


def test_a_damaged_targets_file_does_not_stop_the_leads(project):
    add_model_outputs(project)
    (project.root / "target" / "targets.gpkg").write_bytes(b"not a geopackage")
    leads = run(project).leads
    assert "target_overlap_pct" not in leads and "combined_score" in leads  # prospectivity still there


def test_empty_leads_pass_through(project):
    empty = gpd.GeoDataFrame({"lead_id": [], "score": []}, geometry=[], crs=CRS)
    linked = link_evidence(project, empty, 0.3)
    assert linked.columns == [] and linked.leads.empty


def test_the_command_lists_the_best_by_combined_score(project):
    add_model_outputs(project, auc=0.55)
    result = runner.invoke(app, ["leads", "run", "demo", "--as-of", "2026-10-01"])
    assert result.exit_code == 0, result.output
    assert "Best by combined score" in result.output and "in targets" in result.output and "prospectivity" in result.output
    assert "weak or unmeasured" in result.output
