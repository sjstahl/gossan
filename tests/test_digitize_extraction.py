import base64
import csv
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gossan.cli import app
from gossan.core.project import Project
from gossan.digitize import review
from gossan.digitize.claude_vision import (
    Attribute,
    ClaudeVisionError,
    PageTable,
    TableRow,
    table_to_rows,
)
from gossan.digitize.pages import parse_pages

runner = CliRunner()


@pytest.fixture
def project(aoi_file) -> Project:
    assert runner.invoke(app, ["new", "demo", "--aoi", str(aoi_file)]).exit_code == 0
    return Project(Path("demo"))


def add_pdf(project: Project, fixture: Path, number: int) -> Path:
    dest = project.raw_dir / "aris_pdfs" / f"ARIS_{number:05d}.pdf"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(fixture, dest)
    return dest


@pytest.fixture
def scan_project(project, fixtures_dir) -> Project:
    """ARIS 77 = tables_sample.pdf: p1 ruled table with text, p2 image-only scan, p3 prose."""
    add_pdf(project, fixtures_dir / "tables_sample.pdf", 77)
    return project


@pytest.fixture
def sample_project(project, fixtures_dir) -> Project:
    """ARIS 123 = aris_sample.pdf: five pages, page 2 has no text layer."""
    add_pdf(project, fixtures_dir / "aris_sample.pdf", 123)
    return project


def text_of(project: Project, number: int) -> str:
    return (project.root / "digitize" / "text" / f"{number:05d}.txt").read_text(encoding="utf-8")


# --- page specs ---------------------------------------------------------------------------


def test_parse_pages():
    assert parse_pages("43-47") == [43, 44, 45, 46, 47]
    assert parse_pages("3, 43 - 45,3") == [3, 43, 44, 45]
    assert parse_pages("2", n_pages=5) == [2]
    for bad in ["", "a", "0", "5-3", "1,,2", "-4"]:
        with pytest.raises(ValueError):
            parse_pages(bad)
    with pytest.raises(ValueError, match="past the end"):
        parse_pages("4-9", n_pages=5)


# --- OCR ----------------------------------------------------------------------------------


def test_ocr_appends_marked_text(sample_project, fake_easyocr):
    result = runner.invoke(app, ["digitize", "demo"])
    assert result.exit_code == 0, result.output
    assert "OCR: 1 page(s) appended" in result.output

    text = text_of(sample_project, 123)
    assert "===== page 2 [OCR EXTRACTED] =====" in text
    ocr_part = text.split("===== page 2 [OCR EXTRACTED] =====")[1]
    assert "Scanned text from p2.png" in ocr_part and "Au 12 ppb Cu 40 ppm" in ocr_part
    # The embedded-text part is untouched, and OCR sits after it.
    assert "Diamond drilling on the Reg property" in text.split("[OCR EXTRACTED]")[0]
    assert text.index("===== page 5 =====") < text.index("[OCR EXTRACTED]")

    assert fake_easyocr.reader_args == [(("en",), False)]  # English, CPU
    assert fake_easyocr.images == ["p2.png"]  # only the page without a text layer
    record = next(r for r in sample_project.provenance.read() if r.operation == "ocr")
    assert record.parameters["pages"] == [2]
    assert record.parameters["engine"] == "easyocr"
    assert record.parameters["output"] == "digitize/text/00123.txt"
    assert "Access Only" in record.license


def test_ocr_is_not_repeated_and_survives_reextraction(sample_project, fake_easyocr):
    assert runner.invoke(app, ["digitize", "demo"]).exit_code == 0
    result = runner.invoke(app, ["digitize", "demo"])  # text is re-extracted from the PDF
    assert result.exit_code == 0, result.output
    assert "1 already done" in result.output
    assert fake_easyocr.images == ["p2.png"]  # not OCR'd a second time
    assert text_of(sample_project, 123).count("[OCR EXTRACTED]") == 1  # and still there


def test_ocr_respects_target_pages_and_no_ocr(sample_project, fake_easyocr):
    result = runner.invoke(app, ["digitize", "demo", "--report", "123", "--target-pages", "3-4"])
    assert result.exit_code == 0, result.output
    assert fake_easyocr.images == []  # page 2 is outside the target

    result = runner.invoke(app, ["digitize", "demo", "--report", "123", "--no-ocr"])
    assert result.exit_code == 0, result.output
    assert fake_easyocr.images == []

    result = runner.invoke(app, ["digitize", "demo", "--report", "123", "--target-pages", "2"])
    assert result.exit_code == 0, result.output
    assert fake_easyocr.images == ["p2.png"]


def test_missing_easyocr_is_a_warning_not_a_failure(sample_project):
    result = runner.invoke(app, ["digitize", "demo"])  # easyocr is blocked by the autouse fixture
    assert result.exit_code == 0, result.output
    assert "EasyOCR is not installed" in result.output
    assert "[OCR EXTRACTED]" not in text_of(sample_project, 123)
    assert not [r for r in sample_project.provenance.read() if r.operation == "ocr"]


# --- digital tables -----------------------------------------------------------------------


def test_digital_tables_saved_as_csv(scan_project):
    result = runner.invoke(app, ["digitize", "demo"])
    assert result.exit_code == 0, result.output

    path = scan_project.root / "digitize" / "tables" / "digital" / "00077_p1.csv"
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows == [
        ["Hole", "Depth (m)", "Au (ppb)"],
        ["R-1", "45.7", "12"],
        ["R-2", "60.9", "<5"],  # detection-limit flag kept as printed
        ["R-3", "30.5", "8"],
    ]
    # Page 3 has text but no table; page 2 has no text layer, so pdfplumber is not run on it.
    assert sorted(p.name for p in path.parent.iterdir()) == ["00077_p1.csv"]
    assert "digital tables: 1 file(s) from 2 text page(s)" in result.output

    (record,) = [r for r in scan_project.provenance.read() if r.operation == "extract_table"]
    assert record.identifier == "ARIS 77 p1"
    assert record.parameters["output"] == "digitize/tables/digital/00077_p1.csv"
    assert (record.parameters["rows"], record.parameters["cols"]) == (4, 3)
    assert record.parameters["engine"].startswith("pdfplumber")
    assert review.read_queue(scan_project) == []  # digital tables are not queued


def test_digital_tables_rerun_replaces_and_respects_target_pages(scan_project):
    assert runner.invoke(app, ["digitize", "demo"]).exit_code == 0
    path = scan_project.root / "digitize" / "tables" / "digital" / "00077_p1.csv"
    path.write_text("stale")
    assert runner.invoke(app, ["digitize", "demo", "--report", "77", "--target-pages", "3"]).exit_code == 0
    assert path.read_text() == "stale"  # page 1 was not targeted, so left alone
    assert runner.invoke(app, ["digitize", "demo"]).exit_code == 0
    assert path.read_text().startswith("Hole,")  # rewritten


# --- Claude vision ------------------------------------------------------------------------


def test_scanned_table_goes_through_claude_to_csv_and_review_queue(scan_project, fake_claude):
    result = runner.invoke(
        app,
        ["digitize", "demo", "--report", "77", "--target-pages", "2", "--extract-scanned-tables"],
    )
    assert result.exit_code == 0, result.output
    assert "1 table(s) queued for review" in result.output
    assert "1500 in / 300 out tokens" in result.output

    # One API call, for the targeted page only, with that page's image and a schema.
    (call,) = fake_claude.calls
    assert call["model"] == "anthropic/claude-opus-5.5"
    assert call["output_format"] is PageTable
    assert "never infer" in call["system"]
    image_block = call["messages"][0]["content"][0]
    assert image_block["type"] == "image" and image_block["source"]["media_type"] == "image/png"
    image_path = scan_project.root / "digitize" / "pages" / "00077_p2.png"
    sent = base64.standard_b64decode(image_block["source"]["data"])
    assert sent == image_path.read_bytes() and sent.startswith(b"\x89PNG")

    # CSV: required columns first, then attributes in order of first appearance.
    csv_path = scan_project.root / "digitize" / "tables" / "scanned" / "00077_p2.csv"
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == ["hole_id", "from_m", "to_m", "Au (ppb)", "Lithology", "Cu (ppm)"]
    assert rows[0] == {
        "hole_id": "R-1", "from_m": "0", "to_m": "1.5",
        "Au (ppb)": "12", "Lithology": "andesite", "Cu (ppm)": "",
    }  # fmt: skip
    assert rows[1]["Au (ppb)"] == "<5" and rows[1]["Cu (ppm)"] == "40"

    # Review queue: PENDING, with both the page image and the CSV.
    (item,) = review.read_queue(scan_project)
    assert (item.status, item.type, item.report, item.page) == ("PENDING", "table", "77", "2")
    assert item.source == "digitize/tables/scanned/00077_p2.csv"
    assert item.image == "digitize/pages/00077_p2.png"
    assert "assay table, 2 row(s)" in item.note and "Bottom rows smudged." in item.note

    # Provenance: model, tokens, and that nobody has reviewed it yet.
    record = next(r for r in scan_project.provenance.read() if r.operation == "extract_scanned_table")
    assert record.source == "OpenRouter (Claude vision)"
    assert record.parameters["model"] == "anthropic/claude-opus-5.5"
    assert record.parameters["reviewed"] is False
    assert record.parameters["page_image"] == "digitize/pages/00077_p2.png"
    assert record.parameters["output"] == "digitize/tables/scanned/00077_p2.csv"
    assert record.parameters["input_tokens"] == 1500

    # `gossan review` shows the item with both paths.
    shown = runner.invoke(app, ["review", "demo"]).output
    assert "1 item(s) pending review (1 table)" in shown
    assert "digitize/tables/scanned/00077_p2.csv" in shown
    assert "image: digitize/pages/00077_p2.png" in shown


def test_model_can_be_overridden_by_env(scan_project, fake_claude, monkeypatch):
    monkeypatch.setenv("GOSSAN_CLAUDE_MODEL", "anthropic/claude-sonnet-5.5")
    args = ["digitize", "demo", "--report", "77", "--target-pages", "2", "--extract-scanned-tables"]
    assert runner.invoke(app, args).exit_code == 0
    assert fake_claude.calls[0]["model"] == "anthropic/claude-sonnet-5.5"


def test_claude_is_never_called_by_default(scan_project, fake_claude):
    assert runner.invoke(app, ["digitize", "demo"]).exit_code == 0
    assert runner.invoke(app, ["digitize", "demo", "--report", "77", "--target-pages", "2"]).exit_code == 0
    assert fake_claude.calls == []
    assert review.read_queue(scan_project) == []


def test_scanned_extraction_needs_explicit_pages(scan_project, fake_claude):
    result = runner.invoke(app, ["digitize", "demo", "--extract-scanned-tables"])
    assert result.exit_code != 0 and "--target-pages" in result.output
    result = runner.invoke(app, ["digitize", "demo", "--target-pages", "2"])  # no --report
    assert result.exit_code != 0 and "--target-pages" in result.output
    assert fake_claude.calls == []
    assert not (scan_project.root / "digitize" / "pages").exists()


def test_page_cap_and_bad_pages_stop_before_any_api_call(sample_project, fake_claude):
    base = ["digitize", "demo", "--report", "123", "--extract-scanned-tables", "--target-pages"]
    result = runner.invoke(app, [*base, "9-12"])  # the PDF has 5 pages
    assert result.exit_code == 1 and "past the end" in result.output
    assert fake_claude.calls == []


def test_more_than_25_pages_refused(project, fake_claude, tmp_path):
    import pymupdf

    doc = pymupdf.open()
    for _ in range(40):
        doc.new_page()
    big = tmp_path / "big.pdf"
    doc.save(big)
    add_pdf(project, big, 500)
    args = ["digitize", "demo", "--report", "500", "--extract-scanned-tables", "--target-pages"]
    result = runner.invoke(app, [*args, "1-30"])
    assert result.exit_code == 1 and "limit is 25" in result.output
    assert fake_claude.calls == []
    assert runner.invoke(app, [*args, "1-25"]).exit_code == 0  # exactly the limit is allowed
    assert len(fake_claude.calls) == 25


def test_missing_api_key_is_a_clear_error(scan_project):
    args = ["digitize", "demo", "--report", "77", "--target-pages", "2", "--extract-scanned-tables"]
    result = runner.invoke(app, args)  # no key, and the live client is blocked
    assert result.exit_code == 1 and "OPENROUTER_API_KEY is not set" in result.output
    assert review.read_queue(scan_project) == []


def test_extracted_pages_are_not_sent_twice(scan_project, fake_claude):
    args = ["digitize", "demo", "--report", "77", "--target-pages", "2", "--extract-scanned-tables"]
    assert runner.invoke(app, args).exit_code == 0
    result = runner.invoke(app, args)
    assert result.exit_code == 0 and "1 already extracted" in result.output
    assert len(fake_claude.calls) == 1
    assert len(review.read_queue(scan_project)) == 1


def test_page_without_a_table_is_logged_not_queued(scan_project, fake_claude):
    fake_claude.default = PageTable(table_type="other", depth_unit="unknown", rows=[], notes="A plan view map.")
    args = ["digitize", "demo", "--report", "77", "--target-pages", "2", "--extract-scanned-tables"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "1 page(s) without a table" in result.output
    assert not (scan_project.root / "digitize" / "tables" / "scanned").exists()
    assert review.read_queue(scan_project) == []
    record = next(r for r in scan_project.provenance.read() if r.operation == "extract_scanned_table_none")
    assert record.parameters["notes"] == "A plan view map."


def test_api_failure_on_one_page_does_not_stop_the_rest(sample_project, fake_claude):
    import anthropic
    import httpx2

    request = httpx2.Request("POST", "https://api.anthropic.test/v1/messages")
    fake_claude.queue = [anthropic.APIConnectionError(request=request)]  # page 3 fails; page 4 uses the default
    args = ["digitize", "demo", "--report", "123", "--extract-scanned-tables", "--target-pages", "3-4"]
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "Claude failed on page 3" in result.output
    scanned = sample_project.root / "digitize" / "tables" / "scanned"
    assert sorted(p.name for p in scanned.iterdir()) == ["00123_p4.csv"]
    assert [i.page for i in review.read_queue(sample_project)] == ["4"]


def test_feet_are_converted_to_metres_and_attributes_flattened():
    table = PageTable(
        table_type="assay",
        depth_unit="ft",
        notes="",
        rows=[
            TableRow(
                hole_id="A", from_m=100.0, to_m=110.0,
                attributes=[Attribute(name="Au", value="1"), Attribute(name="Au", value="2")],
            ),
            TableRow(hole_id="B", from_m=None, to_m=None, attributes=[Attribute(name="Cu", value="5")]),
        ],
    )  # fmt: skip
    flat = table_to_rows(table)
    assert flat.converted_from_feet is True
    assert flat.columns == ["hole_id", "from_m", "to_m", "Au", "Au_2", "Cu"]  # repeated name kept
    assert flat.rows[0]["from_m"] == "30.480" and flat.rows[0]["to_m"] == "33.528"
    assert flat.rows[1]["from_m"] == "" and flat.rows[1]["to_m"] == ""  # no interval (collar row)


def test_refusal_is_a_page_failure(scan_project, fake_claude, monkeypatch):
    from types import SimpleNamespace

    def refuse(**kwargs):
        return SimpleNamespace(parsed_output=None, stop_reason="refusal", usage=SimpleNamespace(input_tokens=1, output_tokens=1))

    fake_claude.messages.parse = refuse
    args = ["digitize", "demo", "--report", "77", "--target-pages", "2", "--extract-scanned-tables"]
    result = runner.invoke(app, args)
    assert result.exit_code == 1 and "stopped early (refusal)" in result.output
    assert review.read_queue(scan_project) == []
    assert ClaudeVisionError  # (class is public for callers)


# --- review queue: columns changed --------------------------------------------------------


def test_old_queue_without_image_column_is_upgraded_not_lost(project):
    path = review.queue_path(project)
    path.parent.mkdir(parents=True)
    path.write_text(
        "item_id,created_at,report,type,page,status,source,note\n"
        "R0001,2026-10-01T00:00:00Z,8038,map,4,PENDING,digitize/x.png,old item\n",
        encoding="utf-8",
    )
    review.init_queue(project)
    assert path.read_text().splitlines()[0] == ",".join(review.COLUMNS)
    (item,) = review.read_queue(project)
    assert (item.item_id, item.note, item.image) == ("R0001", "old item", "")
    new = review.add_item(project, 8038, "table", page=5, source="a.csv", image="a.png")
    assert new.item_id == "R0002" and len(review.read_queue(project)) == 2


def test_client_is_routed_through_openrouter(monkeypatch):
    """The real client factory, with the SDK constructor replaced to capture its arguments."""
    from gossan.digitize import claude_vision

    captured = {}
    monkeypatch.setattr(claude_vision.anthropic, "Anthropic", lambda **kw: captured.update(kw) or "client")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-be-used")
    assert claude_vision.make_client() == "client"
    assert captured == {
        "base_url": "https://openrouter.ai/api",  # the SDK appends /v1/messages
        "auth_token": "sk-or-test",
        "api_key": None,
    }
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(ClaudeVisionError, match="OPENROUTER_API_KEY is not set"):
        claude_vision.make_client()  # an ANTHROPIC_API_KEY alone is not accepted
