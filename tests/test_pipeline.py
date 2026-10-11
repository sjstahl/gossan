"""gossan run: harvest, model, analyze and report in one command, degrading step by step."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from analysis_project import build_analysis_project, runner

from gossan import pipeline
from gossan.cli import app
from gossan.core.config import TargetConfig
from gossan.core.project import Project
from gossan.harvest.bc_provincial.wfs import WfsConnector
from gossan.pipeline import (
    FAILED,
    OK,
    PARTIAL,
    SKIPPED,
    Reporter,
    parse_names,
    run_pipeline,
    summary_lines,
)


class Recorder(Reporter):
    def __init__(self) -> None:
        self.lines: list[tuple[str, str]] = []
        self.steps: list[str] = []

    def say(self, level: str, text: str) -> None:
        self.lines.append((level, text))

    def step(self, name: str) -> None:
        self.steps.append(name)


def fast_target_config(project: Project, **kw) -> None:
    """Small, quick model settings saved to project.yaml (the synthetic AOI is only 4 x 3 km)."""
    project.config.target = TargetConfig(resolution_m=100.0, n_bags=4, n_trees=15, folds=3, seed=1, min_target_ha=0.0, **kw)
    project.config.save(project.root)


# --- argument handling --------------------------------------------------------------------------


def test_parse_names_orders_validates_and_defaults():
    assert parse_names("report,harvest", pipeline.STEPS, "step") == ["report", "harvest"]  # order is applied later
    assert parse_names(None, pipeline.STEPS, "step") == list(pipeline.STEPS)
    with pytest.raises(ValueError, match=r"unknown step\(s\): magic"):
        parse_names("harvest,magic", pipeline.STEPS, "step")


def test_pipeline_runs_steps_in_dependency_order_whatever_the_request(tmp_path, monkeypatch):
    project = build_analysis_project(tmp_path)
    calls: list[str] = []
    for name in ("_harvest_step", "_target_step", "_analyze_step", "_report_step"):
        monkeypatch.setattr(pipeline, name, lambda *a, _n=name, **k: (calls.append(_n), pipeline.StepOutcome(_n[1:-5], OK))[1])
    ticks = iter(range(0, 100, 5))
    rec = Recorder()
    outcomes = run_pipeline(project, ["report", "target", "harvest"], reporter=rec, clock=lambda: next(ticks))
    assert calls == ["_harvest_step", "_target_step", "_report_step"]  # canonical order, analyze not requested
    assert rec.steps == ["harvest", "target", "report"] and [o.seconds for o in outcomes] == [5, 5, 5]
    record = next(r for r in project.provenance.read() if r.operation == "run_pipeline")
    assert [s["step"] for s in record.parameters["steps"]] == ["harvest", "target", "report"]
    assert record.license == "N/A (run log)"


def test_summary_table_lines():
    lines = summary_lines([pipeline.StepOutcome("harvest", OK, "25 source(s)", 48.2), pipeline.StepOutcome("target", SKIPPED, "too few", 0.4)])
    assert lines[0].split() == ["Step", "Status", "Time", "Detail"]
    assert lines[1].split()[:3] == ["harvest", "ok", "48s"] and "25 source(s)" in lines[1]
    assert lines[2].split()[:3] == ["target", "skipped", "0s"]


# --- the command: success, degradation, errors ------------------------------------------------------


def test_run_on_an_existing_project_completes_every_step(tmp_path):
    project = build_analysis_project(tmp_path)
    fast_target_config(project)
    result = runner.invoke(app, ["run", "demo", "--steps", "target,analyze,report"])
    assert result.exit_code == 0, result.output
    table = result.output.split("#### summary")[1]
    rows = {line.split()[0]: line.split()[1] for line in table.splitlines()[2:] if line.strip()}
    assert rows == {"target": "ok", "analyze": "ok", "report": "ok"}
    for path in ("target/targets.gpkg", "analysis/target_decisions.csv", "outputs/analysis_report.pdf", "outputs/analysis_report.docx"):
        assert (project.root / path).exists(), path
    record = next(r for r in project.provenance.read() if r.operation == "run_pipeline")
    assert [s["status"] for s in record.parameters["steps"]] == [OK, OK, OK]
    assert "AUC" in table  # the model's skill figure is in the summary (plain AUC: this project has no effort layers)


def test_run_creates_the_project_from_an_aoi_and_degrades_gracefully(aoi_file, wfs):
    # The fake network serves a handful of features: far too few to train a model.
    result = runner.invoke(app, ["run", "fresh", "--aoi", str(aoi_file), "--no-cv", "--format", "pdf"])
    assert "Created project fresh" in result.output
    assert result.exit_code == 1  # not everything completed
    table = result.output.split("#### summary")[1]
    rows = {line.split()[0]: line.split()[1] for line in table.splitlines()[2:] if line.strip()}
    assert rows["harvest"] == OK and rows["target"] == SKIPPED and rows["report"] == OK
    assert "at least 10" in table  # the reason, in words
    assert rows["analyze"] in {PARTIAL, SKIPPED}
    assert (Path("fresh") / "outputs" / "analysis_report.pdf").exists()
    assert not (Path("fresh") / "outputs" / "analysis_report.docx").exists()  # --format pdf


def test_a_failing_source_makes_harvest_partial_and_the_run_carries_on(aoi_file, wfs, monkeypatch):
    real = WfsConnector._get

    def flaky(self, url):
        if self.name == "roads":
            raise OSError("network down")
        return real(self, url)

    monkeypatch.setattr(WfsConnector, "_get", flaky)
    assert runner.invoke(app, ["new", "demo", "--aoi", str(aoi_file)]).exit_code == 0
    result = runner.invoke(app, ["run", "demo", "--steps", "harvest,report", "--format", "pdf"])
    assert result.exit_code == 1
    table = result.output.split("#### summary")[1]
    assert "partial" in table and "unavailable: roads" in table
    assert "report" in table and (Path("demo") / "outputs" / "analysis_report.pdf").exists()  # the report still ran
    assert "Fallback: download the data manually" in result.output


def test_sources_option_limits_the_harvest(aoi_file, wfs):
    assert runner.invoke(app, ["new", "demo", "--aoi", str(aoi_file)]).exit_code == 0
    result = runner.invoke(app, ["run", "demo", "--steps", "harvest", "--sources", "minfile,faults"])
    assert result.exit_code == 0, result.output
    assert len(wfs.calls) == 2 and "2 source(s)" in result.output


def test_unexpected_report_error_is_reported_not_raised(tmp_path, monkeypatch):
    project = build_analysis_project(tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("matplotlib exploded")

    monkeypatch.setattr(pipeline, "build_analysis_report", boom)
    rec = Recorder()
    (outcome,) = run_pipeline(project, ["report"], reporter=rec)
    assert outcome.status == FAILED and "RuntimeError: matplotlib exploded" in outcome.detail
    assert any(level == "error" for level, _ in rec.lines)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["run", "demo", "--steps", "magic"], "unknown step"),
        (["run", "demo", "--sources", "nope", "--steps", "harvest"], "unknown source"),
        (["run", "demo", "--format", "odt"], "use pdf, docx or both"),
        (["run", "demo", "--class", "gems"], "occurrence_class"),
    ],
)
def test_bad_arguments_are_rejected_before_anything_runs(tmp_path, args, message):
    project = build_analysis_project(tmp_path)
    before = (project.root / "provenance.jsonl").read_text(encoding="utf-8")
    result = runner.invoke(app, args)
    assert result.exit_code != 0 and message in result.output
    assert (project.root / "provenance.jsonl").read_text(encoding="utf-8") == before  # nothing ran


def test_project_and_aoi_conflicts(tmp_path, aoi_file):
    result = runner.invoke(app, ["run", "ghost"])
    assert result.exit_code == 1 and "not a Gossan project" in result.output
    build_analysis_project(tmp_path)
    result = runner.invoke(app, ["run", "demo", "--aoi", str(aoi_file)])
    assert result.exit_code != 0 and "already exists" in result.output


def test_class_and_resolution_options_reach_the_model(tmp_path):
    project = build_analysis_project(tmp_path)
    fast_target_config(project)
    result = runner.invoke(app, ["run", "demo", "--steps", "target", "--class", "metallic", "--resolution", "200", "--no-cv"])
    assert result.exit_code == 0, result.output
    grid = json.loads((project.root / "target" / "grid.json").read_text())
    assert grid["occurrence_class"] == "metallic" and grid["resolution_m"] == 200.0
