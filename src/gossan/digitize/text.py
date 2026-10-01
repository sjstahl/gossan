"""Embedded-text extraction from report PDFs.

Only the text layer already in the PDF is read (PyMuPDF). Pages without one (scanned
images) are reported and left marked in the output; OCR is not implemented yet.

PyMuPDF is dual-licensed AGPL-3.0 / Artifex commercial. Internal use is fine; review the
licence before distributing Gossan or offering it as a service.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from gossan.core.project import Project

#: A page with fewer non-space characters than this counts as having no embedded text.
MIN_TEXT_CHARS = 25
NO_TEXT_MARKER = "[no embedded text: scanned or image-only page]"
OCR_TAG = "[OCR EXTRACTED]"
SECTION = re.compile(r"^===== page (\d+)( \[OCR EXTRACTED\])? =====$", re.MULTILINE)
DERIVED_LICENSE = "Derived from ARIS reports (Access Only): internal use only"


@dataclass
class TextResult:
    path: Path
    n_pages: int
    no_text_pages: list[int]  # 1-based
    n_chars: int


def has_text(page: pymupdf.Page) -> bool:
    return len("".join(page.get_text().split())) >= MIN_TEXT_CHARS


def page_ranges(pages: list[int]) -> str:
    """[1, 2, 3, 5] -> '1-3, 5'."""
    ranges: list[list[int]] = []
    for p in sorted(pages):
        if ranges and p == ranges[-1][1] + 1:
            ranges[-1][1] = p
        else:
            ranges.append([p, p])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in ranges)


def read_ocr_blocks(path: Path) -> dict[int, str]:
    """OCR text already appended to a text file, by page number."""
    if not path.exists():
        return {}
    content = path.read_text(encoding="utf-8")
    marks = list(SECTION.finditer(content))
    blocks = {}
    for i, m in enumerate(marks):
        if m[2]:
            end = marks[i + 1].start() if i + 1 < len(marks) else len(content)
            blocks[int(m[1])] = content[m.end() : end].strip()
    return blocks


def write_with_ocr(path: Path, base: str, ocr: dict[int, str]) -> None:
    """Write ``base`` (the embedded-text pages) followed by the OCR blocks, in page order."""
    parts = [base.rstrip("\n") + "\n"]
    parts += [f"\n===== page {p} {OCR_TAG} =====\n{ocr[p]}\n" for p in sorted(ocr)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts), encoding="utf-8")


def text_base(path: Path) -> str:
    """A text file's embedded-text part, without the appended OCR blocks."""
    content = path.read_text(encoding="utf-8")
    first_ocr = next((m for m in SECTION.finditer(content) if m[2]), None)
    return content if first_ocr is None else content[: first_ocr.start()]


def text_path(project: Project, aris_number: int) -> Path:
    return project.root / "digitize" / "text" / f"{aris_number:05d}.txt"


def extract_text(project: Project, aris_number: int, pdf: Path) -> TextResult:
    """Write ``digitize/text/<aris_id>.txt`` (replacing any earlier extraction)."""
    parts: list[str] = []
    no_text: list[int] = []
    with pymupdf.open(pdf) as doc:
        n_pages = doc.page_count
        for i, page in enumerate(doc, start=1):
            if has_text(page):
                body = page.get_text().strip()
            else:
                no_text.append(i)
                body = NO_TEXT_MARKER
            parts.append(f"===== page {i} =====\n{body}\n")
    out = text_path(project, aris_number)
    # A re-run must not throw away OCR done earlier, for pages that still have no text layer.
    kept_ocr = {p: t for p, t in read_ocr_blocks(out).items() if p in no_text}
    content = "\n".join(parts)
    write_with_ocr(out, content, kept_ocr)

    result = TextResult(out, n_pages, no_text, len(content))
    project.provenance.log(
        dataset="aris_text",
        source="gossan",
        identifier=f"ARIS {aris_number}",
        url=None,
        license=DERIVED_LICENSE,
        operation="extract_text",
        parameters={
            "input": f"raw/aris_pdfs/{pdf.name}",
            "output": out.relative_to(project.root).as_posix(),
            "pages": n_pages,
            "pages_without_text": no_text,
            "min_text_chars": MIN_TEXT_CHARS,
            "ocr_pages_kept": sorted(kept_ocr),
        },
    )
    return result
