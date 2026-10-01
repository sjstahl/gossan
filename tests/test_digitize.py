import csv
import shutil
from pathlib import Path

import geopandas as gpd
import pymupdf
import pytest
from typer.testing import CliRunner

from gossan.cli import app
from gossan.core.project import Project
from gossan.digitize import review
from gossan.digitize.catalogue import classify_page, inspect_pdf
from gossan.digitize.text import has_text, page_ranges

runner = CliRunner()


@pytest.fixture
def sample_pdf(fixtures_dir) -> Path:
    return fixtures_dir / "aris_sample.pdf"


@pytest.fixture
def project(aoi_file) -> Project:
    assert runner.invoke(app, ["new", "demo", "--aoi", str(aoi_file)]).exit_code == 0
    return Project(Path("demo"))


def add_pdf(project: Project, sample: Path, number: int) -> Path:
    dest = project.raw_dir / "aris_pdfs" / f"ARIS_{number:05d}.pdf"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(sample, dest)
    return dest


# --- map heuristic -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("width", "height", "oversize", "landscape"),
    [
        (612, 792, False, False),  # US letter portrait
        (595, 842, False, False),  # A4 portrait
        (842, 595, False, True),  # A4 landscape: small but landscape
        (792, 612, False, True),  # letter landscape
        (612, 1008, True, False),  # legal: longer than A4
        (800, 1200, True, False),  # oversize portrait
        (1224, 792, True, True),  # tabloid landscape
        (615, 795, False, False),  # within the 6 pt tolerance
    ],
)
def test_classify_page(width, height, oversize, landscape):
    assert classify_page(width, height) == (oversize, landscape)


def test_rotated_page_uses_displayed_orientation(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.set_rotation(90)  # displayed landscape
    path = tmp_path / "rot.pdf"
    doc.save(path)
    assert inspect_pdf(path)["n_landscape_pages"] == 1


def test_inspect_sample_pdf(sample_pdf):
    facts = inspect_pdf(sample_pdf)
    assert facts["pages"] == 5
    assert facts["n_oversize_pages"] == 1  # page 4
    assert facts["n_landscape_pages"] == 1  # page 3
    assert facts["n_map_pages"] == 2
    assert facts["map_pages"] == "3-4"
    assert facts["n_no_text_pages"] == 1  # page 2
    assert facts["n_text_pages"] == 4
    assert facts["pdf_bytes"] == sample_pdf.stat().st_size


def test_page_ranges():
    assert page_ranges([]) == ""
    assert page_ranges([7]) == "7"
    assert page_ranges([5, 1, 2, 3]) == "1-3, 5"
    assert page_ranges([1, 3, 4, 5, 9]) == "1, 3-5, 9"


def test_has_text_threshold():
    doc = pymupdf.open()
    page = doc.new_page()
    assert not has_text(page)
    page.insert_text((72, 72), "12")  # a bare page number is not a text page
    assert not has_text(page)
    page.insert_text((72, 100), "A sentence long enough to count as text.")
    assert has_text(page)


# --- review queue ------------------------------------------------------------------------


def test_queue_init_is_idempotent_and_never_overwrites(project):
    path = review.init_queue(project)
    assert path == project.root / "digitize" / "review_queue.csv"
    assert path.read_text().strip() == ",".join(review.COLUMNS)
    review.add_item(project, 8038, "map", page=4)
    before = path.read_text()
    review.init_queue(project)  # does not reset an existing queue
    assert path.read_text() == before


def test_add_item_defaults_and_dedup(project):
    item = review.add_item(project, 8038, review.ItemType.MAP, page=4, source="digitize/x.png", note="n")
    assert item.item_id == "R0001"
    assert (item.status, item.type, item.report, item.page) == ("PENDING", "map", "8038", "4")
    second = review.add_item(project, 8038, "text", note="scanned pages 2, 9")
    assert second.item_id == "R0002" and second.page == ""

    assert review.add_item(project, 8038, "map", page=4, source="digitize/x.png") is None  # duplicate
    assert [i.item_id for i in review.read_queue(project)] == ["R0001", "R0002"]

    with (project.root / "digitize" / "review_queue.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["status"] == "PENDING" and rows[0]["created_at"].endswith("Z")
    assert review.summarize(review.read_queue(project)) == {"PENDING": {"map": 1, "text": 1}}

    ops = [r.operation for r in project.provenance.read()]
    assert ops.count("queue_item") == 2


def test_add_item_rejects_unknown_type(project):
    with pytest.raises(ValueError):
        review.add_item(project, 1, "photo")
    assert review.read_queue(project) == []


def test_read_queue_before_init_is_empty(project):
    assert review.read_queue(project) == []
    assert not (project.root / "digitize").exists()  # reading creates nothing


# --- CLI ---------------------------------------------------------------------------------


def test_digitize_catalogues_and_extracts_text(project, sample_pdf, wfs):
    assert runner.invoke(app, ["harvest", "demo", "--source", "aris_index"]).exit_code == 0
    add_pdf(project, sample_pdf, 8038)  # in the (mocked) index
    add_pdf(project, sample_pdf, 123)  # not in the index

    result = runner.invoke(app, ["digitize", "demo"])
    assert result.exit_code == 0, result.output
    assert "ARIS 8038: 5 page(s), 2 likely map page(s)" in result.output
    assert "2 report(s) processed, 0 failed" in result.output
    # One warning per report, naming the page without a text layer.
    assert result.output.count("no embedded text on page(s) 2") == 2

    # Text files: <aris_id>.txt with page separators and the scan marker.
    text = (project.root / "digitize" / "text" / "08038.txt").read_text(encoding="utf-8")
    assert "===== page 1 =====" in text and "===== page 5 =====" in text
    assert "Diamond drilling on the Reg property" in text
    assert "Geology map of the property" in text
    page2 = text.split("===== page 2 =====")[1].split("===== page 3 =====")[0]
    assert "[no embedded text" in page2
    assert (project.root / "digitize" / "text" / "00123.txt").is_file()

    # Catalogue layer: attributes from the PDF, location and metadata from the index.
    cat = gpd.read_file(project.data_gpkg, layer="catalogue").set_index("aris_number")
    assert sorted(cat.index) == [123, 8038]
    row = cat.loc[8038]
    assert (row.pages, row.n_map_pages, row.map_pages, row.n_no_text_pages) == (5, 2, "3-4", 1)
    assert row.report_year == 1979 and row.title == "Drilling Report"
    assert row.geometry is not None and cat.crs.to_epsg() == 32610
    assert cat.loc[123].geometry is None  # not in aris_index: no location
    assert row.pdf_file == "raw/aris_pdfs/ARIS_08038.pdf"

    # Review queue initialised, nothing queued yet.
    assert (project.root / "digitize" / "review_queue.csv").is_file()
    assert review.read_queue(project) == []

    # Provenance: one record per extraction plus the catalogue.
    records = project.provenance.read()
    texts = [r for r in records if r.operation == "extract_text"]
    assert sorted(r.identifier for r in texts) == ["ARIS 123", "ARIS 8038"]
    assert texts[0].parameters["pages_without_text"] == [2]
    assert texts[0].parameters["ocr"] is False
    assert "Access Only" in texts[0].license
    (cat_rec,) = [r for r in records if r.operation == "write_catalogue"]
    assert cat_rec.parameters["reports_updated"] == [123, 8038]
    # raw/ is untouched.
    assert (project.raw_dir / "aris_pdfs" / "ARIS_08038.pdf").read_bytes() == sample_pdf.read_bytes()


def test_digitize_without_aris_index_still_catalogues(project, sample_pdf):
    add_pdf(project, sample_pdf, 123)
    result = runner.invoke(app, ["digitize", "demo"])
    assert result.exit_code == 0, result.output
    cat = gpd.read_file(project.data_gpkg, layer="catalogue")
    assert len(cat) == 1 and cat.geometry.isna().all()
    assert cat["pages"].item() == 5


def test_digitize_one_report_updates_only_that_row(project, sample_pdf):
    add_pdf(project, sample_pdf, 123)
    add_pdf(project, sample_pdf, 456)
    assert runner.invoke(app, ["digitize", "demo"]).exit_code == 0
    result = runner.invoke(app, ["digitize", "demo", "--report", "456"])
    assert result.exit_code == 0, result.output
    assert "ARIS 456:" in result.output and "ARIS 123:" not in result.output

    cat = gpd.read_file(project.data_gpkg, layer="catalogue")
    assert sorted(cat["aris_number"]) == [123, 456]  # replaced, not duplicated or dropped
    cat_records = [r for r in project.provenance.read() if r.operation == "write_catalogue"]
    assert cat_records[-1].parameters["reports_updated"] == [456]
    assert cat_records[-1].parameters["n_reports_total"] == 2


def test_digitize_skips_unreadable_pdf_and_exits_1(project, sample_pdf):
    add_pdf(project, sample_pdf, 123)
    bad = project.raw_dir / "aris_pdfs" / "ARIS_00999.pdf"
    bad.write_bytes(b"%PDF-1.4 this is not really a pdf")
    result = runner.invoke(app, ["digitize", "demo"])
    assert result.exit_code == 1
    assert "ARIS 999" in result.output and "1 report(s) processed, 1 failed" in result.output
    assert gpd.read_file(project.data_gpkg, layer="catalogue")["aris_number"].tolist() == [123]
    assert not (project.root / "digitize" / "text" / "00999.txt").exists()


def test_digitize_errors_when_no_pdfs(project):
    result = runner.invoke(app, ["digitize", "demo"])
    assert result.exit_code == 1 and "gossan harvest --limit-pdfs" in result.output
    result = runner.invoke(app, ["digitize", "demo", "--report", "8038"])
    assert result.exit_code == 1 and "no PDF for ARIS 8038" in result.output
    assert not project.data_gpkg.exists()


def test_review_command(project):
    result = runner.invoke(app, ["review", "demo"])
    assert result.exit_code == 0 and "No items pending review." in result.output

    review.add_item(project, 8038, "map", page=4, source="digitize/maps/08038_p4.png")
    review.add_item(project, 8038, "table", page="7-8", note="assay sheet")
    review.add_item(project, 724, "text", note="pages 2-9 scanned")
    result = runner.invoke(app, ["review", "demo"])
    assert result.exit_code == 0, result.output
    assert "3 item(s) pending review (1 map, 1 table, 1 text)" in result.output
    assert "R0001  map    ARIS 8038 p.4  digitize/maps/08038_p4.png" in result.output
    assert "R0003  text   ARIS 724  " in result.output


def test_review_ignores_non_pending_items(project):
    review.add_item(project, 1, "map")
    path = review.queue_path(project)
    path.write_text(path.read_text().replace("PENDING", "APPROVED"))
    result = runner.invoke(app, ["review", "demo"])
    assert "No items pending review." in result.output
