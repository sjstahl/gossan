"""Drilling in the decision table and the analysis report."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pymupdf
import pytest
from analysis_project import CRS, X0, Y0, build_analysis_project, runner
from docx import Document
from shapely.geometry import Point, box
from test_drill_results import CX, CY, assays, collar, write_tables

from gossan.cli import app
from gossan.core.project import Project


def write_targets(project: Project) -> None:
    out = project.root / "target"
    out.mkdir(exist_ok=True)
    gpd.GeoDataFrame(
        {
            "rank": [1, 2, 3], "target_id": ["T001", "T002", "T003"], "n_cells": [4, 4, 4], "area_ha": [4.0, 4.0, 4.0],
            "mean_score": [0.9, 0.8, 0.7], "max_score": [0.95, 0.85, 0.75], "mean_spread": [0.05] * 3, "n_known_occurrences": [0, 0, 0],
        },
        geometry=[
            box(CX - 100, CY - 100, CX + 100, CY + 100),  # one mineralized hole inside, one barren 200 m away
            box(CX + 1000, CY, CX + 1200, CY + 200),  # two barren holes inside
            box(X0 + 3300, Y0 + 300, X0 + 3500, Y0 + 500),  # no placed hole; a report with drilling lies 700 m off
        ],
        crs=CRS,
    ).to_file(out / "targets.gpkg", layer="targets_top", driver="GPKG")  # fmt: skip


@pytest.fixture
def project(tmp_path) -> Project:
    proj = build_analysis_project(tmp_path)
    write_targets(proj)
    gpd.GeoDataFrame({"ARIS_NUMBER": [9], "REPORT_YEAR": [1988]}, geometry=[Point(X0 + 3400, Y0 + 1200)], crs=CRS).to_file(proj.data_gpkg, layer="aris_index", driver="GPKG")
    write_tables(
        proj,
        [
            *collar(1, "MIN", CX + 50, CY), *collar(1, "B1", CX + 300, CY),
            *collar(1, "B2", CX + 1100, CY + 100), *collar(1, "B3", CX + 1150, CY + 150),
            *collar(9, "REPORT-ONLY"),  # no coordinates: only the report's location is known
        ],
        [
            *assays(1, "MIN", "Au", [10, 900, 10, 10, 10, 10], "ppb"), *assays(1, "B1", "Au", [10] * 6, "ppb"),
            *assays(1, "B2", "Au", [10] * 6, "ppb"), *assays(1, "B3", "Au", [10] * 6, "ppb"),
            *assays(9, "REPORT-ONLY", "Cu", [0.5] * 6, "%"),
        ],
    )  # fmt: skip
    return proj


def decisions(project: Project) -> pd.DataFrame:
    out = runner.invoke(app, ["analyze", "demo", "--steps", "targets"])
    assert out.exit_code == 0, out.output
    return pd.read_csv(project.root / "analysis" / "target_decisions.csv").set_index("target_id")


def test_each_target_says_what_drilling_already_tested_it(project):
    t = decisions(project)
    assert list(t["n_drill_holes"]) == [2, 2, 0]
    assert set(t.loc["T001", "drill_result"].split(", ")) == {"1 mineralized", "1 barren"}
    assert t.loc["T002", "drill_result"] == "2 barren" and pd.isna(t.loc["T003", "drill_result"])
    assert "drilled: 2 hole(s), 1 mineralized (best Au 0.9 g/t over 5-10 m, ARIS 1)" in t.loc["T001", "notes"]
    assert "already drilled: 2 hole(s), 2 barren and none mineralized (a hole tests only its own line)" in t.loc["T002", "notes"]
    assert "ARIS report(s) 9 describe drilling near it, but the holes could not be placed" in t.loc["T003", "notes"]


def test_distance_limits_come_from_the_project_settings(project):
    project.config.drill.near_m = 50.0  # B1 is 200 m from T001: no longer near
    project.config.save(project.root)
    t = decisions(project)
    assert t.loc["T001", "n_drill_holes"] == 1 and t.loc["T001", "drill_result"] == "1 mineralized"
    project.config.drill.near_m = 500.0
    project.config.drill.report_radius_m = 100.0  # the report's location is 700 m away
    project.config.save(project.root)
    assert "ARIS report" not in str(decisions(project).loc["T003", "notes"])


def test_decisions_without_drill_tables_have_no_drill_columns(tmp_path):
    proj = build_analysis_project(tmp_path)
    write_targets(proj)
    t = decisions(proj)
    assert "n_drill_holes" not in t.columns and "drilled" not in " ".join(str(n) for n in t["notes"])


def test_the_decision_table_inherits_the_access_only_licence_from_the_drill_tables(project):
    decisions(project)
    record = next(r for r in project.provenance.read() if r.dataset == "analysis_targets")
    assert "Access Only" in record.license and "aris_pdfs" in record.license


def test_the_report_has_a_drilling_section_with_the_results(project):
    assert runner.invoke(app, ["analyze", "demo", "--steps", "drilling,targets"]).exit_code == 0
    assert runner.invoke(app, ["report", "demo", "--kind", "analysis"]).exit_code == 0
    with pymupdf.open(project.root / "outputs" / "analysis_report.pdf") as pdf:
        text = " ".join(" ".join(page.get_text() for page in pdf).split())  # line wraps are not part of the wording
    assert "Drilling evidence" in text and "5 drill hole(s) from 2 assessment report(s)" in text
    assert "2 mineralized, 3 barren" in text and "tests only its own line" in text
    assert "Mineralized holes" in text and "MIN (ARIS 1)" in text and "Drilling: 5 digitized hole(s)" in text
    assert "2 of 3 targets already have a digitized drill hole" in text
    doc = Document(project.root / "outputs" / "analysis_report.docx")
    assert "Drilling evidence" in [p.text for p in doc.paragraphs if p.style.name == "Heading 1"]


def test_without_digitized_holes_the_report_says_so(tmp_path):
    proj = build_analysis_project(tmp_path)
    assert runner.invoke(app, ["analyze", "demo", "--steps", "drilling"]).exit_code == 0
    assert runner.invoke(app, ["report", "demo", "--kind", "analysis", "--format", "pdf"]).exit_code == 0
    with pymupdf.open(proj.root / "outputs" / "analysis_report.pdf") as pdf:
        text = "\n".join(page.get_text() for page in pdf)
    assert "No digitized drill holes yet" in text and "gossan review" in text
