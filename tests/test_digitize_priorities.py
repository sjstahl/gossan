"""Which ARIS drilling reports to digitize next, and downloading them in that order."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pymupdf
import pytest
from analysis_project import CRS, X0, Y0, build_analysis_project, runner
from shapely.geometry import Point
from test_drill_decisions import write_targets
from test_drill_results import CX, CY, assays, collar, write_tables

from gossan.analyze.digitize_priorities import build_digitize_priorities
from gossan.cli import app
from gossan.core.project import Project
from gossan.harvest.bc_provincial.aris_pdf import ArisPdfDownloader

# Target T001 is around (CX, CY); T002 about 1.1 km east of it; T003 near (X0 + 3400, Y0 + 400).


def report(number, x, y, *, year=2000, work="Diamond surface; Soil", general="Drilling; Geochemical", spend=10_000.0, title=None) -> dict:
    return {
        "ARIS_NUMBER": number, "REPORT_YEAR": year, "ASSESSMENT_REPORT_TITLE": title or f"Report {number}", "GENERAL_WORK_TYPES": general,
        "SPECIFIC_WORK_TYPES": work, "OPERATOR_NAMES": "Op Ltd", "MERGE_SPENT_2002": spend,
        "ARIS_REPORT_URL": f"https://apps.nrs.gov.bc.ca/pub/aris/detail/{number}", "geometry": Point(x, y),
    }  # fmt: skip


def write_aris(project: Project, reports: list[dict]) -> None:
    gpd.GeoDataFrame(reports, geometry="geometry", crs=CRS).to_file(project.data_gpkg, layer="aris_index", driver="GPKG")


@pytest.fixture
def project(tmp_path) -> Project:
    proj = build_analysis_project(tmp_path)
    write_targets(proj)
    write_aris(
        proj,
        [
            report(1, CX + 400, CY, spend=50_000, year=1990),  # tier 1 (400 m from T001), mid spend
            report(2, CX + 100, CY, spend=200_000, year=1985),  # tier 1, biggest spend: first
            report(3, X0 + 100, Y0 + 2800, spend=900_000),  # about 3 km from any target: tier 2
            report(4, X0 + 9000, Y0 + 9000, spend=5_000),  # well outside the area, beyond 5 km: tier 3
            report(5, CX, CY, general="Geochemical; Geological", work="Soil; Rock", spend=999_999),  # no drilling: not listed
            report(6, CX + 200, CY + 200, spend=100_000, year=1999),  # tier 1, already digitized below
            report(7, CX - 300, CY, spend=50_000, year=2005, work="Reverse circulation", general="Geophysical"),  # drilling by its specific type; newer than 1
        ],
    )
    write_tables(proj, collar(6, "D1", CX + 200, CY + 200), assays(6, "D1", "Au", [10] * 6, "ppb"))
    return proj


def table(project: Project) -> pd.DataFrame:
    return build_digitize_priorities(project).table.set_index("aris_number")


def test_drilling_reports_are_ranked_by_closeness_to_a_target_then_spend(project):
    t = table(project)
    assert set(t.index) == {1, 2, 3, 4, 6, 7}  # report 5 describes no drilling
    todo = t[t["priority"].notna()].sort_values("priority")
    assert list(todo.index) == [2, 7, 1, 3, 4]  # tier 1 by spend (2: 200k; 7 and 1 tie at 50k, newer first), then tier 2, then 3
    assert list(todo["tier"]) == [1, 1, 1, 2, 3]
    assert t.loc[6, "status"] == "digitized (1 hole(s))" and pd.isna(t.loc[6, "priority"])  # done: no priority


def test_distance_and_nearest_target_are_reported(project):
    t = table(project)
    assert t.loc[2, "nearest_target"] == "T001" and t.loc[2, "distance_to_nearest_target_m"] < 150
    assert t.loc[4, "distance_to_nearest_target_m"] > 5000 and t.loc[4, "tier"] == 3


def test_summary_counts_and_outputs(project):
    result = build_digitize_priorities(project)
    s = result.summary
    assert s["n_reports"] == 7 and s["n_drilling_reports"] == 6 and s["n_digitized"] == 1
    assert s["n_tier1"] == 3 and s["n_tier2"] == 1 and s["reference"] == "target"
    assert [r["aris_number"] for r in s["top"]][:3] == [2, 7, 1]
    out = project.root / "analysis"
    assert (out / "digitize_priorities.csv").exists() and (out / "digitize_priorities_summary.json").exists()
    layer = gpd.read_file(out / "analysis.gpkg", layer="digitize_priorities")
    assert len(layer) == 6 and layer.geometry.notna().all()
    record = next(r for r in project.provenance.read() if r.dataset == "analysis_priorities")
    assert "Access Only" in record.license


def test_without_a_target_model_known_occurrences_are_the_reference(project):
    (project.root / "target" / "targets.gpkg").unlink()
    s = build_digitize_priorities(project).summary
    assert s["reference"] == "known occurrence" and s["n_drilling_reports"] == 6


def test_without_the_aris_index_it_says_how_to_get_it(tmp_path):
    proj = build_analysis_project(tmp_path)
    result = build_digitize_priorities(proj)
    assert result.summary["available"] is False and "harvest --source aris_index" in result.summary["reason"]
    out = runner.invoke(app, ["analyze", "demo", "--steps", "priorities"])
    assert out.exit_code == 0 and "harvest --source aris_index" in out.output


def test_command_output_names_the_first_reports_and_the_download_command(project):
    out = runner.invoke(app, ["analyze", "demo", "--steps", "priorities"])
    assert out.exit_code == 0, out.output
    assert "6 of 7 ARIS reports describe drilling; 1 digitized; 3 not yet digitized lie within 1.5 km of a target" in out.output
    assert "#1 ARIS 2" in out.output and "--pdf-order priority" in out.output


# --- downloading in priority order -------------------------------------------------------------------------------------


def test_pdfs_download_in_priority_order_then_newest_first(project):
    build_digitize_priorities(project)
    reports = ArisPdfDownloader(project)._reports()
    ordered = [n for n, _ in ArisPdfDownloader(project)._by_priority(reports)]
    assert ordered[:5] == ["2", "7", "1", "3", "4"]  # ranked reports first
    assert set(ordered[5:]) == {"5", "6"}  # the rest follow, in the downloader's own newest-first order
    assert sorted(ordered) == sorted(n for n, _ in reports)  # nothing lost or repeated


def test_priority_order_needs_the_ranking_to_exist(project):
    with pytest.raises(FileNotFoundError, match="analyze --steps priorities"):
        ArisPdfDownloader(project)._by_priority([("1", "u")])
    with pytest.raises(ValueError, match="newest or priority"):
        ArisPdfDownloader(project).run(1, order="random")
    out = runner.invoke(app, ["harvest", "demo", "--source", "faults", "--pdf-order", "sideways"])
    assert out.exit_code != 0 and "newest or priority" in out.output


def test_the_first_download_is_the_top_ranked_report(project, monkeypatch):
    build_digitize_priorities(project)
    got: list[str] = []

    def fake_download(self, url: str, part: Path) -> None:
        got.append(url)
        part.write_bytes(b"%PDF-1.4 fake")

    monkeypatch.setattr(ArisPdfDownloader, "_download", fake_download)
    result = ArisPdfDownloader(project, pause=0).run(2, order="priority")
    assert result.downloaded == ["2", "7"] and got[0].endswith("00002.pdf/")


# --- the report -----------------------------------------------------------------------------------------------------------------


def test_the_report_lists_the_reports_worth_digitizing_next(project):
    assert runner.invoke(app, ["analyze", "demo", "--steps", "drilling,priorities,targets"]).exit_code == 0
    assert runner.invoke(app, ["report", "demo", "--kind", "analysis", "--format", "pdf"]).exit_code == 0
    with pymupdf.open(project.root / "outputs" / "analysis_report.pdf") as pdf:
        text = " ".join(" ".join(page.get_text() for page in pdf).split())
    assert "Reports worth digitizing next" in text and "6 of the 7 assessment reports in the area describe drilling; 1 have been digitized" in text
    assert "--pdf-order priority" in text and "Report 2" in text
