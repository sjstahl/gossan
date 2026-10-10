"""BC drilling indicator: the Notice of Work connector, the flag, and the small score bonus."""

from datetime import date
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import geopandas as gpd
import pandas as pd
import pymupdf
import pytest
from docx import Document
from typer.testing import CliRunner

from gossan.cli import app
from gossan.core.project import Project
from gossan.leads.drilling import classify_notices, parse_bc_date
from gossan.leads.generator import run_leads
from gossan.leads.scoring import LeadsConfig, read_layer, score_leads

runner = CliRunner()
AS_OF = date(2026, 10, 1)
NOW_TYPE = "pub:WHSE_MINERAL_TENURE.HSP_NOTICE_OF_WORK_PA_POINT_SV"


@pytest.fixture
def project(aoi_file, seed_leads_layers) -> Project:
    assert runner.invoke(app, ["new", "demo", "--aoi", str(aoi_file)]).exit_code == 0
    proj = Project(Path("demo"))
    seed_leads_layers(proj)
    return proj


@pytest.fixture
def with_notices(project, fixtures_dir) -> Project:
    gdf = gpd.read_file(fixtures_dir / "drilling_notices_wfs.geojson").to_crs(project.config.crs)
    gdf.to_file(project.data_gpkg, layer="drilling_notices", driver="GPKG", engine="pyogrio")
    return project


def scored(project: Project, config: LeadsConfig | None = None, notices: bool = True) -> gpd.GeoDataFrame:
    tenure = read_layer(project, "mto_tenure")
    aris = read_layer(project, "aris_index")
    layer = read_layer(project, "drilling_notices") if notices else None
    return score_leads(tenure, aris, config or LeadsConfig(), AS_OF, layer).set_index("lead_id")


# --- the connector ----------------------------------------------------------------------------


def test_harvest_drilling_notices(project, wfs):
    result = runner.invoke(app, ["harvest", "demo", "--source", "drilling_notices"])
    assert result.exit_code == 0, result.output
    query = parse_qs(urlparse(wfs.calls[0]).query)
    assert query["typeNames"] == [NOW_TYPE]
    cql = query["CQL_FILTER"][0]
    assert cql.startswith("BBOX(SHAPE,") and cql.endswith(" AND EXP_SURFACE_DRILLING_HA>0")  # only notices that drill

    gdf = gpd.read_file(project.data_gpkg, layer="drilling_notices")
    assert gdf.crs.to_epsg() == 32610
    assert sorted(gdf["NOW_NUMBER"]) == ["A-1", "B-1", "C-1", "D-1", "E-1", "F-1"]  # G-1 is outside the AOI
    fetch = next(r for r in project.provenance.read() if r.dataset == "drilling_notices")
    assert fetch.operation == "fetch" and "Open Government Licence" in fetch.license
    assert "cql_filter" in fetch.parameters
    assert len(list((project.raw_dir / "drilling_notices").glob("drilling_notices_*.geojson"))) == 1


def test_a_default_bc_harvest_includes_drilling_notices(project, wfs):
    assert runner.invoke(app, ["harvest", "demo"]).exit_code == 0
    assert NOW_TYPE in {parse_qs(urlparse(u).query)["typeNames"][0] for u in wfs.calls}


# --- classification ---------------------------------------------------------------------------


def test_bc_dates_with_a_trailing_z_parse():
    parsed = parse_bc_date(pd.Series(["2026-09-14Z", None, "junk"]))
    assert parsed.iloc[0] == pd.Timestamp("2026-09-14", tz="UTC") and parsed.iloc[1:].isna().all()


def test_notice_status_as_of_a_date(fixtures_dir):
    notices = gpd.read_file(fixtures_dir / "drilling_notices_wfs.geojson").set_index("NOW_NUMBER")
    status = classify_notices(notices, AS_OF)
    assert status.to_dict() == {
        "A-1": "active", "B-1": "pending", "C-1": "lapsed", "D-1": "ignored",
        "E-1": "lapsed", "F-1": "active", "G-1": "active",
    }  # fmt: skip
    # The same permit is active before its authorisation ends and lapsed after (E-1 ended 2025-01-01).
    assert classify_notices(notices, date(2024, 6, 1))["E-1"] == "active"


# --- the flag and the bonus -------------------------------------------------------------------


def test_flags_and_bonus_by_status(with_notices):
    leads = scored(with_notices)
    assert leads.loc["1001", "drilling_flag"] == "active" and leads.loc["1001", "drilling_points"] == 10  # A-1, by tenure number
    assert leads.loc["1003", "drilling_flag"] == "pending" and leads.loc["1003", "drilling_points"] == 5  # B-1, by location
    assert leads.loc["1007", "drilling_flag"] == "active"  # F-1 is about 390 m away, inside the 500 m buffer
    # 1004: C-1 (lapsed, by number) plus B-1 (pending) and E-1 (lapsed) within the buffer: the best status wins.
    row = leads.loc["1004"]
    assert row.drilling_flag == "pending" and (row.drilling_pending_count, row.drilling_lapsed_count) == (1, 2)
    assert row.drilling_notice_numbers == "B-1; C-1; E-1"
    # 1002: its own notice D-1 was rejected and is ignored; A-1 and B-1 are within the buffer.
    assert leads.loc["1002", "drilling_flag"] == "active" and "D-1" not in leads.loc["1002", "drilling_notice_numbers"]
    row = leads.loc["1001"]
    assert row.drilling_active_count == 1 and row.drilling_notice_numbers == "A-1"
    assert row.drilling_operator == "Alpha Gold Corp." and row.drilling_latest_filed == "2026-03-10"


def test_the_bonus_is_added_to_the_unchanged_base_score(project, with_notices):
    with_ = scored(with_notices)
    without = scored(project, notices=False)
    assert (without["drilling_flag"] == "not assessed").all() and (without["drilling_points"] == 0).all()
    for lead_id in with_.index:
        assert with_.loc[lead_id, "score"] == pytest.approx(without.loc[lead_id, "score"] + with_.loc[lead_id, "drilling_points"])
    assert with_.loc["1003", "score"] == 67.5 and with_.loc["1001", "score"] == 60  # 62.5 + 5 and 50 + 10
    assert list(with_.index[:2]) == ["1003", "1001"]


def test_the_bonus_is_configurable_and_capped_at_100(with_notices):
    off = scored(with_notices, LeadsConfig(drilling_active_points=0, drilling_pending_points=0))
    assert (off["drilling_points"] == 0).all() and off.loc["1001", "drilling_flag"] == "active"  # the flag stays
    assert off.loc["1003", "score"] == 62.5
    big = scored(with_notices, LeadsConfig(drilling_pending_points=50))
    assert big.loc["1003", "drilling_points"] == 50 and big.loc["1003", "score"] == 100


def test_a_zero_buffer_matches_by_tenure_number_and_exact_location_only(with_notices):
    leads = scored(with_notices, LeadsConfig(drilling_match_buffer_m=0))
    assert leads.loc["1001", "drilling_flag"] == "active"  # number
    assert leads.loc["1004", "drilling_flag"] == "lapsed"  # C-1, by number only
    assert leads.loc["1003", "drilling_flag"] == "pending"  # B-1 lies inside the polygon
    assert leads.loc["1002", "drilling_flag"] == "none"  # D-1 was rejected
    assert leads.loc["1007", "drilling_flag"] == "none"  # F-1 is outside the polygon


# --- outputs ----------------------------------------------------------------------------------


def test_leads_run_writes_the_drilling_columns(with_notices):
    result = runner.invoke(app, ["leads", "run", "demo", "--as-of", "2026-10-01"])
    assert result.exit_code == 0, result.output
    assert "drilling active" in result.output
    csv = pd.read_csv(with_notices.outputs_dir / "leads_ranked.csv", dtype={"lead_id": str})
    assert {"drilling_points", "drilling_flag", "drilling_notice_numbers"} <= set(csv.columns)
    assert csv.set_index("lead_id").loc["1001", "drilling_flag"] == "active"
    layer = gpd.read_file(with_notices.data_gpkg, layer="leads")
    assert "drilling_flag" in layer.columns
    rec = [r for r in with_notices.provenance.read() if r.operation == "write_leads_csv"][-1]
    assert rec.parameters["n_drilling_notices_in"] == 7


def test_leads_run_without_the_layer_is_unchanged_apart_from_not_assessed(project):
    assert run_leads(project, AS_OF).leads["score"].iloc[0] == 62.5
    csv = pd.read_csv(project.outputs_dir / "leads_ranked.csv")
    assert (csv["drilling_flag"] == "not assessed").all() and (csv["drilling_points"] == 0).all()


def test_pack_and_report_show_the_flag(with_notices):
    assert runner.invoke(app, ["leads", "run", "demo", "--as-of", "2026-10-01"]).exit_code == 0
    assert runner.invoke(app, ["leads", "pack", "demo", "1001"]).exit_code == 0
    cells = [c.text for t in Document(str(with_notices.outputs_dir / "lead_1001_pack.docx")).tables for r in t.rows for c in r.cells]
    assert any("active" in c and "A-1" in c and "permit signal only" in c for c in cells)

    assert runner.invoke(app, ["report", "demo", "--lead", "1001"]).exit_code == 0
    doc = pymupdf.open(with_notices.outputs_dir / "report.pdf")
    text = "\n".join(p.get_text() for p in doc).replace("\n", " ")
    assert "Drilling" in text and "not that a rig is on site" in text and "Alpha Gold Corp." in text
