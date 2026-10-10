"""Nevada NDEP reclamation permit notices: parsing, the connector, the manual route, and US lead flags."""

import json
from datetime import date
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import geopandas as gpd
import pandas as pd
import pymupdf
import pytest
from shapely.geometry import box
from test_us_pipeline import NV, blm_features, mrds_gml
from typer.testing import CliRunner

from gossan.cli import app
from gossan.core.project import Project
from gossan.harvest.us_state import ndep_notices
from gossan.harvest.us_state.ndep_notices import (
    LISTING_URL,
    NdepNoticesConnector,
    parse_listing,
    pdf_links,
    wanted,
)
from gossan.harvest.us_state.ndep_parse import parse_notice
from gossan.harvest.us_state.plss import PLSS_SECTION_URL, Township
from gossan.leads.drilling import classify_notices
from gossan.leads.scoring_us import UsLeadsConfig, score_us_leads

runner = CliRunner()
AS_OF = date(2026, 10, 1)


def make_pdf(lines: list[str]) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    for i, line in enumerate(lines):
        page.insert_text((40, 60 + 14 * i), line, fontsize=9)
    return doc.tobytes()


def decision(permit: str, who: str, project: str, kind: str, posted: str, trs: str, county: str = "White Pine") -> list[str]:
    return [
        "NOTICE OF FINAL DECISION - Bureau of Mining Regulation and Reclamation",
        f"Date of Posting: {posted}",
        "Deadline for Appeals: 04/23/2026",
        f"Reclamation Permit No. {permit}",
        who,
        project,
        f"The Administrator has decided to issue Reclamation Permit ({permit}) for {kind} to {who} (Permittee).",
        f"The project is located within portions of {trs}: Mount Diablo Baseline and Meridian, {county} County, Nevada.",
        "This Permit authorizes 11.6 acres of Phase I disturbance on USFS administered land.",
        "The total proposed disturbed acreage is 15.0 acres on public lands.",
    ]


MUNCY = decision("0456", "Falcon Copper Corporation", "Muncy Creek Drilling Project", "an Exploration Project", "04/13/2026", "Sections 8, 9, and 16, Township 20 North, Range 66 East")
NOPA = [
    "NOTICE OF PROPOSED ACTION - Bureau of Mining Regulation and Reclamation",
    "Date of Posting: 09/20/2026",
    "Reclamation Permit No. 0460",
    "Bell Minerals Inc.",
    "Bell Drilling Project",
    "The Division proposes to issue Reclamation Permit (0460) for an Exploration Project to Bell Minerals Inc. (Permittee).",
    "The project is located within Section 1, Township 21 North, Range 67 East: Mount Diablo Baseline and Meridian, Nye County, Nevada.",
    "The proposed disturbance is 9.5 acres.",
]
ARGENTA = decision("0301", "Baker Hughes Drilling Fluids", "Argenta Mine", "a mine project", "03/30/2023", "Section 5, Township 19 North, Range 65 East", "Lander")
OLD = decision("0302", "Old Timers LLC", "Old Exploration Project", "an Exploration Project", "01/15/2025", "Sections 3 and 4, Township 23 North, Range 69 East", "Elko")
OTHER = ["Memo", "This is a notice about something else entirely."]

PDFS = {
    "https://ndep.nv.gov/uploads/documents/20260413lb_0456_NOFD_Muncy_Ck_Drilling.pdf": make_pdf(MUNCY),
    "https://ndep.nv.gov/uploads/documents/20260920_0460_NoPA.pdf": make_pdf(NOPA),
    "https://ndep.nv.gov/uploads/documents/0301_NOFD.pdf": make_pdf(ARGENTA),
    "https://ndep.nv.gov/uploads/documents/0302_NOFD.pdf": make_pdf(OLD),
    "https://ndep.nv.gov/uploads/documents/memo.pdf": make_pdf(OTHER),
}
PLSS_BOXES = {  # township id -> west, south, east, north
    "NV210200N0660E0": (-117.30, 39.10, -117.25, 39.25),
    "NV210210N0670E0": (-117.25, 39.10, -117.20, 39.25),
    "NV210190N0650E0": (-117.20, 39.10, -117.15, 39.25),
    "NV210230N0690E0": (-117.15, 39.10, -117.10, 39.25),
}


def article(date_text: str, title: str, desc: str, slug: str) -> str:
    return (
        '<article class="row postList-post"><section class="large-2 columns postList-post-details">'
        f"<p>{date_text}</p></section>"
        f'<section class="large-10 columns postList-post-summary"><h3 class="postList-post-summary-title">{title}</h3>'
        f'<p class="postList-post-summary-desc"> {desc} </p><a href="/posts/{slug}/" class="btn">Read More</a></section></article>'
    )


PAGE0 = "<html>" + "".join([
    article("September 20, 2026", "Notice of Proposed Action - BMRR - Bell Drilling Project", "BMRR - Reclamation Permit 0460 to Bell Minerals Inc.", "bell"),
    article("April 13, 2026", "Notice of Final Decision - BMRR - Muncy Creek Drilling Project", "BMRR - Notice of Final Decision - Reclamation Permit 0456", "muncy"),
    article("April 01, 2026", "Notice of Decision - BMRR - Reward Project #2", "BMRR - Issue renewed Water Pollution Control Permit NEV2007101.", "reward"),
    article("March 01, 2026", "Notice of Final Decision - BMRR - Odd Memo", "BMRR - Notice of Final Decision - Reclamation", "memo"),
    article("February 01, 2026", "Public meeting on air quality", "No mining here", "air"),
]) + "</html>"
PAGE5 = "<html>" + "".join([
    article("January 15, 2025", "Notice of Final Decision - BMRR - Old Exploration Project", "BMRR - Reclamation Permit 0302", "old"),
    article("March 30, 2023", "Notice of Final Decision - BMRR - Argenta Mine", "BMRR - Reclamation Permit 0301", "argenta"),
]) + "</html>"
POSTS = {
    "bell": ["20260920_0460_NoPA.pdf"],
    "muncy": ["20260413lb_0456_NOFD_Muncy_Ck_Drilling.pdf"],
    "memo": ["memo.pdf"],
    "old": ["0302_NOFD.pdf"],
    "argenta": ["0301_NOFD.pdf"],
}


class FakeNdep:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.plss_down = False


@pytest.fixture
def ndep(monkeypatch) -> FakeNdep:
    fake = FakeNdep()
    monkeypatch.setattr(ndep_notices, "PAUSE_SECONDS", 0.0)

    def fake_get(self, url: str) -> bytes:
        fake.calls.append(url)
        if url == LISTING_URL:
            return PAGE0.encode()
        if url == f"{LISTING_URL}/P5":
            return PAGE5.encode()
        if url.startswith(f"{LISTING_URL}/P"):
            return b"<html></html>"
        if url.startswith("https://ndep.nv.gov/posts/"):
            slug = url.rstrip("/").rsplit("/", 1)[1]
            links = "".join(f'<a href="/uploads/documents/{n}">pdf</a>' for n in POSTS[slug])
            return f"<html>{links}</html>".encode()
        if url in PDFS:
            return PDFS[url]
        if url.startswith(PLSS_SECTION_URL):
            if fake.plss_down:
                raise OSError("PLSS service unavailable")
            where = parse_qs(urlparse(url).query)["where"][0]
            west, south, east, north = PLSS_BOXES[where.split("'")[1]]
            feature = {"type": "Feature", "geometry": json.loads(gpd.GeoSeries([box(west, south, east, north)]).to_json())["features"][0]["geometry"], "properties": {}}
            return json.dumps({"type": "FeatureCollection", "features": [feature]}).encode()
        raise AssertionError(f"unexpected request {url}")

    monkeypatch.setattr(NdepNoticesConnector, "_get", fake_get)
    return fake


@pytest.fixture
def nevada_project(us) -> Project:
    us.mrds_bytes = mrds_gml(40)
    us.blm_features = blm_features()
    assert runner.invoke(app, ["new", "nv", "--aoi", str(NV)]).exit_code == 0
    return Project(Path("nv"))


# --- reading a notice -------------------------------------------------------------------------


def test_a_final_decision_for_an_exploration_project_is_read():
    n = parse_notice("\n".join(MUNCY))
    assert (n.permit_no, n.kind, n.project_type) == ("0456", "decision", "exploration")
    assert (n.permittee, n.project, n.county, n.acres) == ("Falcon Copper Corporation", "Muncy Creek Drilling Project", "White Pine", 15.0)
    assert n.posted == date(2026, 4, 13)
    assert n.townships == [Township(township=20, ns="N", range=66, ew="E", sections=(8, 9, 16))]
    assert n.townships[0].plss_id() == "NV210200N0660E0"


def test_a_proposed_action_and_a_mine_permit_are_read():
    n = parse_notice("\n".join(NOPA))
    assert (n.kind, n.project_type, n.county, n.acres) == ("proposed", "exploration", "Nye", 9.5)
    assert n.townships[0].sections == (1,)
    mine = parse_notice("\n".join(ARGENTA))
    assert mine.project_type == "mine" and mine.townships[0].sections == (5,)


def test_other_documents_are_not_notices():
    assert parse_notice("\n".join(OTHER)) is None
    assert parse_notice("NOTICE OF DECISION Water Pollution Control Permit NEV2007101 renewed") is None


def test_the_list_page_is_read_and_filtered():
    items = parse_listing(PAGE0)
    assert [i.path for i in items][:2] == ["/posts/bell/", "/posts/muncy/"]
    assert [i.path for i in items if wanted(i)] == ["/posts/bell/", "/posts/muncy/", "/posts/memo/"]  # not water, not air
    assert pdf_links('<a href="/uploads/documents/a.pdf">x</a><a href="/x">y</a>') == ["https://ndep.nv.gov/uploads/documents/a.pdf"]


# --- the connector ----------------------------------------------------------------------------


def test_harvest_reads_notices_into_the_drilling_layer(nevada_project, ndep):
    result = runner.invoke(app, ["harvest", "nv", "--source", "ndep_notices"])
    assert result.exit_code == 0, result.output
    gdf = gpd.read_file(nevada_project.data_gpkg, layer="drilling_notices")
    assert gdf.crs.to_epsg() == 32611
    assert sorted(gdf["PERMIT_NO"]) == ["0301", "0302", "0456", "0460"]  # the memo was skipped
    row = gdf.set_index("PERMIT_NO")
    assert row.loc["0456", "NOW_APPLICATION_STATUS_DESC"] == "Approved" and str(row.loc["0456", "DECISION_DATE"])[:10] == "2026-04-13"
    assert row.loc["0460", "NOW_APPLICATION_STATUS_DESC"] == "Received" and pd.isna(row.loc["0460", "DECISION_DATE"])
    assert row.loc["0301", "PROJECT_TYPE"] == "mine" and row.loc["0456", "PROJECT_TYPE"] == "exploration"
    assert row.loc["0456", "PLSS"] == "T20N R66E sec 8,9,16" and row.loc["0456", "PERMITTEE_NAME"] == "Falcon Copper Corporation"
    assert (gdf.geom_type == "Polygon").all()

    pdfs = sorted((nevada_project.raw_dir / "ndep_notices" / "pdfs").glob("*.pdf"))
    assert len(pdfs) == 5  # every PDF read is kept, including the memo
    records = nevada_project.provenance.read()
    assert sum(r.dataset == "ndep_notices_pdf" and r.operation == "fetch" for r in records) == 5
    fetch = next(r for r in records if r.dataset == "drilling_notices" and r.operation == "fetch")
    assert fetch.parameters["n_features"] == 4 and fetch.parameters["n_skipped"] == 1 and "Access Only" not in fetch.license
    assert "BLM" in fetch.parameters["plss_source"]


def test_notices_accumulate_across_harvests_and_a_decision_replaces_its_proposal(nevada_project, ndep):
    assert runner.invoke(app, ["harvest", "nv", "--source", "ndep_notices"]).exit_code == 0
    # The next list shows only the final decision on the Bell permit; the earlier proposal is gone from the list.
    bell_decision = decision("0460", "Bell Minerals Inc.", "Bell Drilling Project", "an Exploration Project", "09/28/2026", "Section 1, Township 21 North, Range 67 East", "Nye")
    PDFS["https://ndep.nv.gov/uploads/documents/20260920_0460_NoPA.pdf"] = make_pdf(bell_decision)
    try:
        assert runner.invoke(app, ["harvest", "nv", "--source", "ndep_notices"]).exit_code == 0
    finally:
        PDFS["https://ndep.nv.gov/uploads/documents/20260920_0460_NoPA.pdf"] = make_pdf(NOPA)
    gdf = gpd.read_file(nevada_project.data_gpkg, layer="drilling_notices").set_index("PERMIT_NO")
    assert len(gdf) == 4 and gdf.loc["0460", "NOTICE_KIND"] == "decision" and str(gdf.loc["0460", "DECISION_DATE"])[:10] == "2026-09-28"
    assert len(list((nevada_project.raw_dir / "ndep_notices").glob("ndep_notices_*.geojson"))) == 2  # originals are never overwritten


def test_the_plss_service_being_down_is_reported_not_hidden(nevada_project, ndep):
    ndep.plss_down = True
    result = runner.invoke(app, ["harvest", "nv", "--source", "ndep_notices"])
    assert result.exit_code == 1 and "PLSS service unavailable" in result.output + (result.stderr or "")


def test_ndep_notices_is_not_part_of_a_plain_harvest():
    from gossan.harvest import default_sources

    assert "ndep_notices" not in default_sources("us") and "ndep_notices" not in default_sources("bc")


def test_ingest_notices_loads_saved_pdfs(nevada_project, ndep, tmp_path):
    folder = tmp_path / "emails"
    folder.mkdir()
    (folder / "a.pdf").write_bytes(make_pdf(MUNCY))
    (folder / "b.pdf").write_bytes(make_pdf(OTHER))
    (folder / "c.txt").write_text("not a pdf")
    result = runner.invoke(app, ["ingest-notices", "nv", str(folder)])
    assert result.exit_code == 0, result.output
    assert "1 NDEP notice area(s)" in result.output and "1 file(s) skipped" in result.output
    gdf = gpd.read_file(nevada_project.data_gpkg, layer="drilling_notices")
    assert gdf["PERMIT_NO"].tolist() == ["0456"]
    assert [r.operation for r in nevada_project.provenance.read() if r.dataset == "ndep_notices_pdf"] == ["manual_ingest", "manual_ingest"]
    assert (nevada_project.raw_dir / "ndep_notices" / "pdfs" / "a.pdf").read_bytes() == (folder / "a.pdf").read_bytes()


# --- scoring ----------------------------------------------------------------------------------


def claims_and_notices():
    def claim(i, x0):
        return {"TENURE_NUMBER_ID": i, "CLAIM_NAME": f"C{i}", "TENURE_TYPE_DESCRIPTION": "Lode", "OWNER_NAME": None,
                "AREA_IN_HECTARES": 100.0, "geometry": box(x0, 0, x0 + 1000, 1000)}  # fmt: skip

    claims = gpd.GeoDataFrame([claim(1, 0), claim(2, 5000), claim(3, 10000), claim(4, 15000), claim(5, 20000)], crs="EPSG:32611")

    def notice(number, kind, posted, ptype, x0):
        return {"NOW_NUMBER": f"NDEP {number}", "NOW_APPLICATION_STATUS_DESC": "Approved" if kind == "decision" else "Received",
                "PERMIT_STATUS_CODE_DESC": None, "AMENDMENT_AUTH_END_DATE": None, "NOW_APPLICATION_SUBMITTED_DATE": posted,
                "TENURE_NUMBER": None, "PERMITTEE_NAME": f"Op {number}", "MINE_NAME": f"Project {number}", "NOTICE_SOURCE": "NDEP BMRR",
                "NOTICE_KIND": kind, "DECISION_DATE": posted if kind == "decision" else None, "PROJECT_TYPE": ptype,
                "geometry": box(x0 - 200, -200, x0 + 1200, 1200)}  # fmt: skip

    notices = gpd.GeoDataFrame(
        [
            notice("0456", "decision", "2026-04-13", "exploration", 0),  # active
            notice("0460", "proposed", "2026-09-20", "exploration", 5000),  # pending
            notice("0302", "decision", "2025-01-15", "exploration", 10000),  # lapsed (20 months old)
            notice("0301", "decision", "2026-06-01", "mine", 15000),  # a mine: context only
        ],
        crs="EPSG:32611",
    )
    targets = gpd.GeoDataFrame(geometry=[box(0, 0, 500, 1000)], crs="EPSG:32611")
    occ = gpd.GeoDataFrame(geometry=[], crs="EPSG:32611")
    return claims, targets, occ, notices


def test_status_by_posting_date_and_window():
    _, _, _, notices = claims_and_notices()
    status = classify_notices(notices.set_index("NOW_NUMBER"), AS_OF)
    assert status.to_dict() == {"NDEP 0456": "active", "NDEP 0460": "pending", "NDEP 0302": "lapsed", "NDEP 0301": "active"}
    assert classify_notices(notices.set_index("NOW_NUMBER"), AS_OF, window_months=24)["NDEP 0302"] == "active"
    assert classify_notices(notices.set_index("NOW_NUMBER"), AS_OF, window_months=3)["NDEP 0456"] == "lapsed"


def test_us_flags_and_bonus():
    claims, targets, occ, notices = claims_and_notices()
    out = score_us_leads(claims, targets, occ, UsLeadsConfig(), notices, AS_OF).set_index("lead_id")
    assert out.loc["1", "drilling_flag"] == "active" and out.loc["1", "drilling_points"] == 10
    assert out.loc["2", "drilling_flag"] == "pending" and out.loc["2", "drilling_points"] == 5
    assert out.loc["3", "drilling_flag"] == "lapsed" and out.loc["3", "drilling_points"] == 0
    assert out.loc["4", "drilling_flag"] == "none"  # a mine permit does not flag
    assert out.loc["5", "drilling_flag"] == "none"
    row = out.loc["1"]
    assert row.score == row.overlap_points + row.proximity_points + 10 == 40 and row.drilling_operator == "Op 0456"
    assert row.drilling_notice_numbers == "NDEP 0456" and row.drilling_latest_filed == "2026-04-13"
    assert out.loc["2", "score"] == 5


def test_us_bonus_is_configurable_and_the_window_moves_the_flag():
    claims, targets, occ, notices = claims_and_notices()
    off = score_us_leads(claims, targets, occ, UsLeadsConfig(drilling_active_points=0, drilling_pending_points=0), notices, AS_OF).set_index("lead_id")
    assert (off["drilling_points"] == 0).all() and off.loc["1", "drilling_flag"] == "active"
    wide = score_us_leads(claims, targets, occ, UsLeadsConfig(drilling_window_months=24), notices, AS_OF).set_index("lead_id")
    assert wide.loc["3", "drilling_flag"] == "active"
    big = score_us_leads(claims, targets, occ, UsLeadsConfig(drilling_active_points=90), notices, AS_OF).set_index("lead_id")
    assert big.loc["1", "score"] == 100  # capped
    unassessed = score_us_leads(claims, targets, occ, UsLeadsConfig(), None, AS_OF)
    assert (unassessed["drilling_flag"] == "not assessed").all() and (unassessed["drilling_points"] == 0).all()


# --- end to end -------------------------------------------------------------------------------


def test_us_leads_carry_the_flag_end_to_end(nevada_project, ndep):
    assert runner.invoke(app, ["harvest", "nv"]).exit_code == 0
    assert runner.invoke(app, ["harvest", "nv", "--source", "ndep_notices"]).exit_code == 0
    assert runner.invoke(app, ["target", "nv", "--resolution", "500", "--n-bags", "3", "--no-cv"]).exit_code == 0
    result = runner.invoke(app, ["leads", "run", "nv", "--as-of", "2026-10-01"])
    assert result.exit_code == 0, result.output
    csv = pd.read_csv(nevada_project.outputs_dir / "leads_ranked.csv", dtype={"lead_id": str}).set_index("lead_id")
    claims = gpd.read_file(nevada_project.data_gpkg, layer="mto_tenure").drop_duplicates("TENURE_NUMBER_ID")
    notices = gpd.read_file(nevada_project.data_gpkg, layer="drilling_notices")
    rank = {"active": 3, "pending": 2, "lapsed": 1, "none": 0}
    status = classify_notices(notices, AS_OF)
    for c in claims.itertuples():
        hit = [s for s, n, t in zip(status, notices["PROJECT_TYPE"], notices.geometry, strict=True) if n == "exploration" and t.intersects(c.geometry)]
        expected = max(hit, key=rank.get) if hit else "none"
        assert csv.loc[str(int(c.TENURE_NUMBER_ID)), "drilling_flag"] == expected
    assert (csv["drilling_flag"] != "none").any(), "the fixture notices should touch at least one claim"
    top = str(csv.index[0])
    assert runner.invoke(app, ["leads", "pack", "nv", top]).exit_code == 0
    assert runner.invoke(app, ["report", "nv", "--lead", top]).exit_code == 0
    text = " ".join("\n".join(p.get_text() for p in pymupdf.open(nevada_project.outputs_dir / "report.pdf")).split())
    assert "Drilling" in text and "not that a rig is on site" in text
