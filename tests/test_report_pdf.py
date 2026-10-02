import getpass
import hashlib
import re
from datetime import UTC, datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import pymupdf
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_bounds
from synthetic import build_project, runner

from gossan.cli import app
from gossan.core.project import Project
from gossan.digitize import review
from gossan.report import pdf as report_pdf
from gossan.report.pdf import friendly_feature, tenure_summary
from gossan.target.run import run_target

FOOTER = "INTERNAL DRAFT | Contains ARIS data (Access Only): not for distribution"
ASSAY_CSV = "hole_id,from_m,to_m,Au (ppb),Cu (ppm)\nR-1,0,1.5,12,40\nR-1,1.5,3.0,<5,55\nR-2,0,2,8,61\n"


@pytest.fixture(autouse=True)
def reviewer(monkeypatch):
    monkeypatch.setattr(getpass, "getuser", lambda: "tester")


@pytest.fixture
def make_project(tmp_path):
    def build(**kw) -> Project:
        return build_project(tmp_path, **kw)

    return build


def full_project(make_project, **kw) -> Project:
    """Synthetic layers, leads, a (small) targeting run, and one approved map and table."""
    project = make_project(n_occ=50, seed=4, **kw)
    assert runner.invoke(app, ["leads", "run", str(project.root)]).exit_code == 0
    run_target(project, project.config.target, cross_validate=True)
    lead = gpd.read_file(project.data_gpkg, layer="leads").set_index("lead_id").loc["1001"]
    minx, miny, maxx, maxy = lead.geometry.bounds

    tif = "rasters/aris_00502_p3.tif"
    data = np.full((4, 60, 90), 180, dtype=np.uint8)
    data[3] = 255
    with rasterio.open(project.root / tif, "w", driver="GTiff", width=90, height=60, count=4, dtype="uint8", crs=project.config.crs,
                       transform=from_bounds(minx - 30, miny - 30, maxx + 30, maxy + 30, 90, 60)) as ds:  # fmt: skip
        ds.write(data)
    project.provenance.log(dataset="aris_maps", source="QGIS Georeferencer points (user)", identifier="ARIS 502 p3", url=None, license="x",
                           operation="georef_manual", parameters={"output": tif, "n_gcps": 9, "rms_m": 0.4, "pixel_size_m": 0.5})  # fmt: skip
    for rel, colour in (("digitize/maps/00502_p3.png", (235, 230, 215)), ("digitize/pages/00503_p5.png", (210, 225, 240))):
        (project.root / rel).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (500, 330), colour).save(project.root / rel)  # different colours: identical images would merge in the PDF
    m = review.add_item(project, 502, "map", page=3, source=tif, image="digitize/maps/00502_p3.png", note="n")

    csv_rel = "digitize/tables/scanned/00503_p5.csv"
    (project.root / csv_rel).parent.mkdir(parents=True, exist_ok=True)
    (project.root / csv_rel).write_text(ASSAY_CSV, encoding="utf-8")
    project.provenance.log(dataset="aris_tables_scanned", source="x", identifier="ARIS 503 p5", url=None, license="x",
                           operation="extract_scanned_table", parameters={"output": csv_rel, "table_type": "assay"})  # fmt: skip
    t = review.add_item(project, 503, "table", page=5, source=csv_rel, image="digitize/pages/00503_p5.png", note="n")
    for item in (m, t):
        assert runner.invoke(app, ["review", "approve", str(project.root), item.item_id, "--note", "checked"]).exit_code == 0
    return Project(project.root)


def make_pdf(project: Project, *args: str) -> pymupdf.Document:
    result = runner.invoke(app, ["report", str(project.root), *args])
    assert result.exit_code == 0, result.output
    return pymupdf.open(project.outputs_dir / "report.pdf")


def text_of(doc: pymupdf.Document) -> str:
    return "\n".join(page.get_text() for page in doc)


# --- the full pitch pack -------------------------------------------------------------------


def test_full_pack_has_every_section_in_order(make_project):
    project = full_project(make_project)
    doc = make_pdf(project, "--lead", "1001")
    text = text_of(doc)
    headings = ["Executive summary", "Property and tenure", "Lead scoring", "Prospectivity targeting",
                "Before & After: digitized samples", "Data sources and provenance"]  # fmt: skip
    positions = [text.index(h) for h in headings]
    assert positions == sorted(positions)  # sections appear in the intended order
    assert doc.page_count >= 6
    assert doc.metadata["title"] == "Gossan pitch pack: demo"
    assert "INTERNAL DRAFT" in doc.metadata["subject"] and "Access Only" in doc.metadata["subject"]


def test_every_page_carries_the_internal_marking_and_page_one_the_disclaimer(make_project):
    doc = make_pdf(full_project(make_project), "--lead", "1001")
    for page in doc:
        assert FOOTER in page.get_text(), f"page {page.number + 1} lacks the internal footer"
    first = doc[0].get_text().replace("\n", " ")
    assert "INTERNAL DRAFT - NOT FOR DISTRIBUTION" in first
    assert "Access Only" in first and "Assessment Report Indexing System (ARIS)" in first
    assert "written permission" in first and "client, prospect or other third party" in first
    assert "ARIS data: Access Only" in text_of(doc)  # and again in the notices


def test_executive_summary_reports_the_numbers_from_each_phase(make_project):
    project = make_project(n_occ=50, seed=4)
    runner.invoke(app, ["leads", "run", str(project.root)])
    run_target(project, project.config.target)
    text = text_of(make_pdf(project)).replace("\n", " ")
    assert "50 MINFILE mineral occurrence(s)" in text and "Most common commodities: Gold (17), Copper (17), Silver (16)" in text
    assert "2 mineral/placer tenure(s) covering 166 ha" in text and "2 active, 0 past their good-to date" in text
    assert "30 assessment report(s) on file in ARIS" in text
    assert "2 tenure(s) scored as leads; the highest is EAST BLOCK (tenure 1001" in text
    assert "Prospectivity model: 1,200 cells at 100 m, trained on" in text and "spatial cross-validation AUC" in text
    assert "random: 5%" in text and "Digitized samples approved by a reviewer: 0 georeferenced map(s), 0 extracted table(s)." in text


def test_property_section_shows_the_overview_map_and_tenure_table(make_project):
    doc = make_pdf(full_project(make_project))
    text = text_of(doc).replace("\n", " ")
    assert "Figure 1. Area of interest with tenure" in text
    for fragment in ("Tenures 2", "Total area 166 ha", "Types Mineral: 2", "Status 2 active", "BIG MINING LTD (1)"):
        assert fragment in text
    assert "snapshot of current Mineral Titles Online data" in text


def test_lead_section_lists_ranked_leads_and_the_featured_lead(make_project):
    text = text_of(make_pdf(full_project(make_project), "--lead", "1001")).replace("\n", " ")
    assert "Each tenure is scored out of 100" in text and "Top 2 of 2 scored tenures" in text
    assert "Featured lead: 1001" in text and "EAST BLOCK" in text and "BIG MINING LTD" in text
    assert "Report 501" in text and "Report 503" in text  # the ARIS reports on that tenure
    assert "Report 600" not in text.split("Featured lead")[1].split("Prospectivity targeting")[0]  # not those elsewhere


def test_targeting_section_has_maps_and_clean_tables(make_project):
    doc = make_pdf(full_project(make_project))
    text = text_of(doc).replace("\n", " ")
    assert "Figure 2. Prospectivity score (a relative ranking, not a probability of a deposit)" in text
    assert "Figure 3. How much each cell's score varies between the model's bags" in text
    assert "Grid 40 x 30 cells at 100 m (EPSG:32610)" in text and "Cells in the area 1,200 (12.0 km²)" in text
    assert "Distance to known occurrences" in text and "Share of model importance" in text
    # Importance shares add up to 100%.
    section = text.split("What drives the model")[1].split("Spatial cross-validation")[0]
    shares = [float(x) for x in re.findall(r"(\d+\.\d)%", section)]
    assert len(shares) == 6 and sum(shares) == pytest.approx(100.0, abs=0.7)
    assert "Held-out occurrences" in text and "Top-5% capture" in text and "Mean +/-" in text
    assert "undiscovered deposits among the unlabeled cells pull it down" in text
    assert "Top targets" in text and "Known occurrences" in text and "T001" in text


def test_before_and_after_shows_the_approved_map_and_table(make_project):
    doc = make_pdf(full_project(make_project))
    text = text_of(doc).replace("\n", " ")
    assert "ARIS 502, page 3: georeferenced map" in text and "ARIS 503, page 5: extracted table" in text
    assert "After: georeferenced, tenure outlined" in text and "After: parsed data (excerpt)" in text
    assert "9 control points, RMS 0.4 m" in text and "Approved by tester" in text and "Reviewer's note: checked" in text
    assert "loaded into the assay table as 6 record(s)" in text
    after = text.split("After: parsed data (excerpt)")[1]
    assert "Au (ppb)" in after and "<5" in after and "R-2" in after  # the parsed data excerpt, flags intact
    # Images: overview, prospectivity, uncertainty, map before + after, table before.
    assert sum(len(page.get_images()) for page in doc) == 6


def test_sources_section_lists_datasets_with_licences_and_notices(make_project):
    project = full_project(make_project)
    project.provenance.log(dataset="minfile", source="BC Geological Survey (DataBC WFS)", identifier="pub:WHSE.MINFILE", url="u",
                           license="Open Government Licence - British Columbia", operation="fetch", parameters={})  # fmt: skip
    project.provenance.log(dataset="aris_index", source="BC Geological Survey, ARIS", identifier="pub:WHSE.ARIS", url="u",
                           license="Access Only (BC Data Catalogue): no reproduction", operation="fetch", parameters={})  # fmt: skip
    text = text_of(make_pdf(project)).replace("\n", " ")
    sources = text.split("Data sources and provenance")[1]
    assert "pub:WHSE.MINFILE" in sources and "pub:WHSE.ARIS" in sources
    assert "Contains information licensed under the Open Government Licence - British Columbia." in sources
    assert "The provenance log (provenance.jsonl) holds" in sources and "target_run" in sources


def test_build_is_logged_with_a_hash_of_the_file(make_project):
    project = full_project(make_project)
    make_pdf(project, "--lead", "1001")
    rec = [r for r in project.provenance.read() if r.operation == "build_report_pdf"][-1]
    assert rec.identifier == "outputs/report.pdf" and "Access Only" in rec.license
    p = rec.parameters
    assert p["lead_id"] == "1001" and p["targeting_included"] and p["leads_included"]
    assert (p["n_approved_maps"], p["n_approved_tables"]) == (1, 1)
    assert p["sections"] == ["summary", "property", "leads", "targeting", "samples", "sources"]
    assert p["sha256"] == hashlib.sha256((project.outputs_dir / "report.pdf").read_bytes()).hexdigest() and p["pages"] >= 6
    assets = project.outputs_dir / "report_assets"
    assert {"overview_map.png", "prospectivity_map.png", "uncertainty_map.png"} <= {f.name for f in assets.iterdir()}


# --- sparse projects and failures ----------------------------------------------------------


def test_a_bare_project_still_produces_a_pdf_that_says_what_is_missing(make_project):
    """No ARIS data in provenance, so this is a plain draft: the internal marking does not apply
    (see test_report_conditional.py); everything else about a sparse project is unchanged."""
    project = make_project(layers=(), raster=False)
    doc = make_pdf(project)
    text = text_of(doc).replace("\n", " ")
    assert all(report_pdf.PLAIN_FOOTER in page.get_text() for page in doc) and "INTERNAL DRAFT" not in text
    assert "Tenure information is not available" in text and "gossan harvest --source mto_tenure" in text
    assert "Lead scoring is not available" in text and "gossan leads run" in text
    assert "Prospectivity targeting is not available" in text and "gossan target" in text
    assert "No digitized samples have been approved yet." in text
    assert "No external datasets have been retrieved for this project." in text


def test_partial_target_outputs_are_handled(make_project):
    project = full_project(make_project)
    (project.root / "target" / "cv_report.json").unlink()
    (project.root / "target" / "targets.gpkg").unlink()
    (project.root / "target" / "feature_importance.csv").unlink()
    text = text_of(make_pdf(project)).replace("\n", " ")
    assert "Not available: cross-validation was not run." in text and "No target polygons were produced." in text


def test_pending_items_are_mentioned_but_never_shown(make_project):
    project = make_project(n_occ=50, seed=4)
    review.add_item(project, 502, "map", page=3, source="rasters/x.tif", image="digitize/maps/x.png")
    doc = make_pdf(project)
    text = text_of(doc).replace("\n", " ")
    assert "No digitized samples have been approved yet. 1 item(s) are waiting for review" in text
    assert "ARIS 502" not in text.split("Before & After")[1].split("Data sources")[0]
    assert sum(len(p.get_images()) for p in doc) == 1  # only the overview map


def test_unknown_lead_and_missing_leads_layer_are_errors(make_project):
    project = make_project(n_occ=50, seed=4)
    result = runner.invoke(app, ["report", str(project.root), "--lead", "1001"])
    assert result.exit_code == 1 and "gossan leads run" in result.output
    runner.invoke(app, ["leads", "run", str(project.root)])
    result = runner.invoke(app, ["report", str(project.root), "--lead", "9999"])
    assert result.exit_code == 1 and "not found" in result.output
    assert not (project.outputs_dir / "report.pdf").exists()


def test_untrusted_text_cannot_break_the_pdf(make_project):
    project = make_project(n_occ=50, seed=4, owner="A & B <Mining> Co. </para>")
    runner.invoke(app, ["leads", "run", str(project.root)])
    text = text_of(make_pdf(project)).replace("\n", " ")
    assert "A & B <Mining> Co. </para>" in text  # shown literally, not interpreted as markup


def test_format_option(make_project):
    project = make_project(layers=("minfile",), raster=False, n_occ=12)
    assert runner.invoke(app, ["report", str(project.root), "--format", "docx"]).exit_code == 0
    assert (project.outputs_dir / "report.docx").exists() and not (project.outputs_dir / "report.pdf").exists()
    assert runner.invoke(app, ["report", str(project.root), "--format", "both"]).exit_code == 0
    assert (project.outputs_dir / "report.pdf").exists()
    assert runner.invoke(app, ["report", str(project.root), "--format", "html"]).exit_code != 0


# --- helpers -------------------------------------------------------------------------------


def test_tenure_summary_counts_owners_and_status(make_project):
    project = make_project(layers=("mto_tenure",), raster=False)
    tenure = gpd.read_file(project.data_gpkg, layer="mto_tenure")
    repeat = tenure.iloc[[0]].copy()  # the layer has one row per owner: the same owner twice must count once
    joint = pd.concat([tenure, repeat], ignore_index=True)
    s = tenure_summary(joint, datetime(2095, 1, 1, tzinfo=UTC))  # 1001 is good to 2099, 1002 to 2090
    assert s["n"] == 2 and s["hectares"] == 166.0 and s["types"] == {"Mineral": 2}
    assert (s["active"], s["lapsed"]) == (1, 1)
    assert s["issued_first"] == "2005-01-01" and s["issued_last"] == "2026-03-01"
    assert dict(s["owners"]) == {"BIG MINING LTD": 1, "OLD OWNER INC": 1}


def test_feature_labels_are_readable():
    assert friendly_feature("dist_minfile") == "Distance to known occurrences"
    assert friendly_feature("rgs_copper") == "Stream-sediment geochemistry: Copper"
    assert friendly_feature("nrcan_magnetics") == "Regional magnetics (NRCan)"
    assert friendly_feature("something_else") == "something_else"
    assert report_pdf.DISCLAIMER.count("Access Only") == 1 and report_pdf.FOOTER == FOOTER
