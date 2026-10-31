"""The company's look for PDF reports: logo, accent colour, header, footer, cover page.

The same ``branding:`` settings that style the Word reports (``gossan.report.branding``) style the PDFs, so a
pitch pack or analysis report in either format looks like the company's:

* the accent colour sets the headings and the table header fill;
* the page header carries the company name and logo above a rule in the accent colour (not on a cover page);
* the footer carries the marking line, then the company and project and the page number;
* ``cover_page`` puts a cover first: logo, company, title, subtitle, client, date, contact, and the report's banner.

Branding changes appearance, not the disclaimers. The footer's marking line is drawn on every page, the cover
included: on a report that uses Access Only data it is "INTERNAL DRAFT | Contains Access Only data: not for
distribution", and ``branding.footer_text`` is added after it, never in its place; on a report with no restricted
data ``footer_text`` replaces the plain draft line. The banner is part of the report body (the cover when there is
one). With no branding set, nothing here changes the page: the output is the unbranded layout.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader, simpleSplit
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import Image, PageBreak, Paragraph, Spacer

from gossan.report.branding import Branding, tint

PAGE_W, PAGE_H = letter
MARGIN = 0.75 * inch
CONTENT_W = PAGE_W - 2 * MARGIN
DEFAULT_ACCENT = "2A78D6"  # the setting's default; with nothing else set, the unbranded layout is kept
UNBRANDED_HEADING = "#1f3a5f"
UNBRANDED_HEADER_FILL = "#e6ecf3"
HEADER_LOGO_HEIGHT = 0.42 * inch
HEADER_LOGO_MAX_WIDTH = 3.0 * inch
HEADER_TOP_MARGIN = 1.0 * inch  # room for the header on top of the unbranded 0.7 inch
PLAIN_TOP_MARGIN = 0.7 * inch
COVER_LOGO_MAX = (CONTENT_W, 1.7 * inch)
FOOTER_Y = 0.5 * inch
MARKED_COLOUR = "#7b1d12"
PLAIN_COLOUR = "#555555"


def is_branded(brand: Branding | None) -> bool:
    """Whether any branding setting changes the PDF."""
    if brand is None:
        return False
    return bool(brand.company or brand.logo or brand.footer_text or brand.contact or brand.cover_page) or brand.accent.upper() != DEFAULT_ACCENT


def has_header(brand: Branding | None) -> bool:
    return bool(brand and (brand.company or brand.logo))


def top_margin(brand: Branding | None) -> float:
    """Page top margin: deeper when there is a header to fit."""
    return HEADER_TOP_MARGIN if has_header(brand) else PLAIN_TOP_MARGIN


def brand_styles(styles: dict, brand: Branding | None) -> dict:
    """The report styles with the accent colour on headings, and the table header fill under ``header_fill``."""
    if not is_branded(brand) or brand is None:
        return styles
    accent = colors.HexColor(f"#{brand.accent}")
    out = dict(styles)
    for key in ("h1", "h2"):
        out[key] = ParagraphStyle(f"{key}_brand", parent=styles[key], textColor=accent)
    out["header_fill"] = f"#{tint(brand.accent)}"
    return out


def footer_line(marked: bool, marked_text: str, plain_text: str, brand: Branding | None) -> str:
    """The footer's marking line. On a marked report the marking is always first and ``footer_text`` follows it;
    on a plain one ``footer_text`` replaces the draft line."""
    extra = brand.footer_text if brand and brand.footer_text else None
    if marked:
        return f"{marked_text}   |   {extra}" if extra else marked_text
    return extra or plain_text


def _aspect(path: Path) -> float:
    with PILImage.open(path) as im:
        w, h = im.size
    return w / h


def _fit(path: Path, max_w: float, max_h: float) -> tuple[float, float]:
    aspect = _aspect(path)
    h = min(max_h, max_w / aspect)
    return h * aspect, h


def page_decorations(
    project_name: str, marked: bool, marked_text: str, plain_text: str, brand: Branding | None = None
) -> Callable:
    """The ``onFirstPage`` and ``onLaterPages`` callback: footer with the marking on every page, the header on
    pages that are not a cover."""
    line = footer_line(marked, marked_text, plain_text, brand)
    company = brand.company if brand and brand.company else None
    accent = colors.HexColor(f"#{brand.accent}") if brand and is_branded(brand) else colors.HexColor("#c8c8c8")
    right = f"{company + ' | ' if company else ''}{project_name} | Page "

    def draw(canvas, doc) -> None:
        canvas.saveState()
        right_text = f"{right}{doc.page}"
        left_width = CONTENT_W - stringWidth(right_text, "Helvetica", 7) - 12
        canvas.setFont("Helvetica-Bold", 7)
        canvas.setFillColor(colors.HexColor(MARKED_COLOUR if marked else PLAIN_COLOUR))
        for i, text in enumerate(simpleSplit(line, "Helvetica-Bold", 7, left_width)[:3]):
            canvas.drawString(MARGIN, FOOTER_Y - 8 * i, text)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor(PLAIN_COLOUR))
        canvas.drawRightString(PAGE_W - MARGIN, FOOTER_Y, right_text)
        canvas.setStrokeColor(colors.HexColor("#c8c8c8"))
        canvas.line(MARGIN, 0.65 * inch, PAGE_W - MARGIN, 0.65 * inch)
        if has_header(brand) and brand is not None and not (brand.cover_page and doc.page == 1):
            top = PAGE_H - 0.35 * inch
            if brand.logo:
                w, h = _fit(brand.logo, HEADER_LOGO_MAX_WIDTH, HEADER_LOGO_HEIGHT)
                canvas.drawImage(ImageReader(str(brand.logo)), PAGE_W - MARGIN - w, top - h, width=w, height=h, mask="auto")
            if company:
                canvas.setFont("Helvetica-Bold", 11)
                canvas.setFillColor(accent)
                canvas.drawString(MARGIN, top - HEADER_LOGO_HEIGHT / 2 - 4, company)
            canvas.setStrokeColor(accent)
            canvas.setLineWidth(1.2)
            canvas.line(MARGIN, top - HEADER_LOGO_HEIGHT - 6, PAGE_W - MARGIN, top - HEADER_LOGO_HEIGHT - 6)
        canvas.restoreState()

    return draw


def cover_story(brand: Branding, title: str, subtitle: str, banner, styles: dict, today: date | None = None) -> list:
    """A cover page's flowables: logo, company, title, subtitle, client, date, contact, the banner, then a page break."""
    today = today or datetime.now(UTC).date()
    accent = colors.HexColor(f"#{brand.accent}")
    big = ParagraphStyle("cover_company", parent=styles["body"], fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=accent, spaceAfter=18)
    line = ParagraphStyle("cover_line", parent=styles["body"], fontSize=11, leading=15, spaceAfter=3)
    story: list = [Spacer(1, 0.4 * inch)]
    if brand.logo:
        w, h = _fit(brand.logo, *COVER_LOGO_MAX)
        story += [Image(str(brand.logo), width=w, height=h, hAlign="LEFT"), Spacer(1, 0.45 * inch)]
    if brand.company:
        story.append(Paragraph(escape(brand.company), big))
    story += [Paragraph(escape(title), styles["title"]), Paragraph(escape(subtitle), styles["subtitle"]), Spacer(1, 0.25 * inch)]
    for text in (f"Prepared for {brand.client}" if brand.client else "", f"{today:%d %B %Y}", brand.contact or ""):
        if text:
            story.append(Paragraph(escape(text), line))
    story += [Spacer(1, 0.5 * inch), banner, PageBreak()]
    return story
