"""Branded PDF reports: accent colour, header and logo, cover page, and the marking branding cannot remove."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest
from analysis_project import build_analysis_project, runner
from PIL import Image

from gossan.cli import app
from gossan.core.config import BrandingConfig
from gossan.core.project import Project
from gossan.report import pdf as pitch
from gossan.report.analysis_render import (
    MARKED_FOOTER,
    PLAIN_FOOTER,
    build_analysis_report,
)
from gossan.report.branding import BrandingError, resolve
from gossan.report.branding_pdf import brand_styles, footer_line, is_branded, top_margin
from gossan.report.licensing import decide_marking
from gossan.report.pdf import _styles, build_pdf

ACCENT = "#c2410c"
ACCENT_INT = int(ACCENT.lstrip("#"), 16)
CONTACT = "ops@example.org"


@pytest.fixture
def project(tmp_path, monkeypatch) -> Project:
    monkeypatch.chdir(tmp_path)
    return build_analysis_project(tmp_path, with_inventory=False)  # no restricted data: an unmarked report


@pytest.fixture
def marked(tmp_path, monkeypatch) -> Project:
    monkeypatch.chdir(tmp_path)
    return build_analysis_project(tmp_path)  # holds an Access Only layer: a marked report


def logo(path: Path, size=(300, 80)) -> Path:
    Image.new("RGB", size, "#334155").save(path)
    return path


def brand(project: Project, **settings) -> Project:
    project.config.client = "Example Resources Ltd."
    project.config.branding = BrandingConfig(**settings)
    project.config.save(project.root)
    return Project(project.root)


def full(project: Project, tmp_path: Path, **over) -> Project:
    return brand(project, **{"company": "Acme Geo", "logo": str(logo(tmp_path / "logo.png")), "accent": ACCENT, "contact": CONTACT, **over})


def analysis_pdf(project: Project) -> pymupdf.Document:
    return pymupdf.open(build_analysis_report(project, "pdf")[0])


def pitch_pdf(project: Project) -> pymupdf.Document:
    return pymupdf.open(build_pdf(project))


def flat(page: pymupdf.Page) -> str:
    return " ".join(page.get_text().split())


def span_colours(page: pymupdf.Page, text: str) -> set[int]:
    return {s["color"] for b in page.get_text("dict")["blocks"] for line in b.get("lines", []) for s in line["spans"] if text in s["text"]}


# --- unbranded: as before -------------------------------------------------------------------------------------------------------


def test_with_no_branding_the_pdfs_have_no_header_cover_or_logo(project):
    assert not is_branded(resolve(project)) and top_margin(resolve(project)) == pytest.approx(0.7 * 72)
    for doc in (analysis_pdf(project), pitch_pdf(project)):
        assert not any(info["bbox"][3] < 72 for page in doc for info in page.get_image_info())  # nothing in the header band (report figures are lower)
        assert PLAIN_FOOTER in flat(doc[0]) or pitch.PLAIN_FOOTER in flat(doc[0])  # the title page, not a cover
        assert "Prepared for" not in flat(doc[0])
        assert doc.metadata["author"] == "Gossan"
        assert span_colours(doc[0], "Summary") <= {int("1f3a5f", 16)} or not span_colours(doc[0], "Summary")


def test_accent_alone_is_a_branding_setting_and_the_default_accent_is_not(project):
    assert not is_branded(resolve(brand(project)))
    assert brand_styles(_styles(), resolve(brand(project))) is not None
    branded = resolve(brand(project, accent=ACCENT))
    assert is_branded(branded)
    styles = brand_styles(_styles(), branded)
    assert styles["header_fill"].startswith("#") and styles["h1"].textColor.hexval() == "0xc2410c"


# --- branded --------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("make", [analysis_pdf, pitch_pdf], ids=["analysis", "pitch"])
def test_the_header_carries_the_company_logo_and_a_rule_on_every_page(project, tmp_path, make):
    doc = make(full(project, tmp_path))
    assert doc.page_count >= 2
    for page in doc:
        assert "Acme Geo" in flat(page) and len(page.get_images()) >= 1, page.number
    assert doc.metadata["author"] == "Acme Geo"
    # the accent colours the headings and the company name
    assert ACCENT_INT in span_colours(doc[0], "Acme Geo")
    heading = "Summary" if make is analysis_pdf else "Executive summary"
    assert ACCENT_INT in span_colours(doc[0], heading)


@pytest.mark.parametrize("make", [analysis_pdf, pitch_pdf], ids=["analysis", "pitch"])
def test_a_cover_page_holds_the_logo_title_client_date_contact_and_banner(project, tmp_path, make):
    doc = make(full(project, tmp_path, cover_page=True))
    cover, second = flat(doc[0]), flat(doc[1])
    assert "Acme Geo" in cover and "Prepared for Example Resources Ltd." in cover and CONTACT in cover and "Draft" in cover
    assert len(doc[0].get_images()) >= 1 and "demo" in cover
    assert "Prepared for" not in second and "Acme Geo" in second  # the next page is the report, with the header
    heading = "Summary" if make is analysis_pdf else "Executive summary"
    assert heading not in cover and heading in second


@pytest.mark.parametrize("cover", [False, True])
def test_footer_text_replaces_the_plain_draft_line_and_page_numbers_run_on(project, tmp_path, cover):
    doc = analysis_pdf(full(project, tmp_path, footer_text="Confidential to Acme", cover_page=cover))
    for page in doc:
        text = flat(page)
        assert "Confidential to Acme" in text and PLAIN_FOOTER not in text and f"Page {page.number + 1}" in text


@pytest.mark.parametrize("cover", [False, True])
def test_the_access_only_marking_stays_on_every_page_whatever_the_branding(marked, tmp_path, cover):
    doc = analysis_pdf(full(marked, tmp_path, footer_text="Confidential to Acme", cover_page=cover))
    for page in doc:
        text = flat(page)
        assert MARKED_FOOTER in text and "Confidential to Acme" in text, page.number
        assert text.index(MARKED_FOOTER) < text.index("Confidential to Acme")  # added after the marking, never in its place
    assert "INTERNAL DRAFT" in flat(doc[0])  # the banner is on the cover or the first page


def test_the_pitch_pack_keeps_its_marking_on_every_page_when_branded(marked, tmp_path):
    assert decide_marking(marked).marked
    doc = pitch_pdf(full(marked, tmp_path, cover_page=True, footer_text="Confidential to Acme"))
    assert all(pitch.FOOTER in flat(page) for page in doc)
    assert doc.metadata["subject"].startswith("INTERNAL DRAFT")
    assert "INTERNAL DRAFT - NOT FOR DISTRIBUTION" in flat(doc[0])


def test_footer_line_logic():
    brand_ = resolve_like(footer_text="X")
    assert footer_line(True, "MARK", "PLAIN", brand_) == "MARK   |   X"
    assert footer_line(False, "MARK", "PLAIN", brand_) == "X"
    assert footer_line(True, "MARK", "PLAIN", None) == "MARK" and footer_line(False, "MARK", "PLAIN", None) == "PLAIN"


def resolve_like(**kw):
    from gossan.report.branding import Branding

    return Branding(**kw)


# --- problems -------------------------------------------------------------------------------------------------------------------


def test_a_missing_logo_is_an_error_not_an_unbranded_client_document(project):
    brand(project, company="Acme Geo", logo="nope.png")
    with pytest.raises(BrandingError, match="not found"):
        build_analysis_report(Project(project.root), "pdf")
    assert not (project.outputs_dir / "analysis_report.pdf").exists()
    for args in (["report", "demo", "--kind", "analysis", "--format", "pdf"], ["report", "demo"]):
        result = runner.invoke(app, args)
        assert result.exit_code == 1 and "branding.logo" in result.output + (result.stderr or ""), args


def test_a_branded_pdf_is_logged_with_its_branding(project, tmp_path):
    full(project, tmp_path, cover_page=True)
    reloaded = Project(project.root)
    build_analysis_report(reloaded, "pdf")
    build_pdf(reloaded)
    records = {r.dataset: r for r in reloaded.provenance.read() if r.operation in {"build_analysis_report", "build_report_pdf"}}
    assert records["report_analysis_pdf"].parameters["branding"]["company"] == "Acme Geo"
    assert records["report_pdf"].parameters["branding"] == {"company": "Acme Geo", "logo": "logo.png", "cover_page": True, "accent": "C2410C"}
