"""OCR for pages with no embedded text, using EasyOCR (optional dependency).

Install with ``pip install "gossan[ocr]"`` (EasyOCR pulls in PyTorch, which is large). The
first run downloads EasyOCR's recognition models (about 100 MB) into ``~/.EasyOCR``; later
runs work offline. CPU only: expect several seconds per page.

Each OCR'd page is appended to ``digitize/text/<aris_id>.txt`` under a
``===== page N [OCR EXTRACTED] =====`` header. OCR text is machine output: expect errors
in numbers and tables, and do not treat it as checked.

Licences: EasyOCR is Apache-2.0 and PyTorch is BSD-3.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gossan.core.project import Project
from gossan.digitize.pages import DEFAULT_DPI, render_page
from gossan.digitize.text import (
    DERIVED_LICENSE,
    read_ocr_blocks,
    text_base,
    text_path,
    write_with_ocr,
)

LANGUAGES = ["en"]
_reader: Any = None


class OcrUnavailableError(Exception):
    """EasyOCR is not installed."""


def load_reader() -> Any:
    """The shared EasyOCR reader (created on first use; loading models is slow)."""
    global _reader
    if _reader is None:
        try:
            import easyocr
        except ImportError as e:
            raise OcrUnavailableError(
                "EasyOCR is not installed (install gossan's ocr extra, or pip install easyocr); skipping OCR"
            ) from e
        _reader = easyocr.Reader(LANGUAGES, gpu=False)
    return _reader


@dataclass
class OcrResult:
    path: Path
    pages_done: list[int]
    pages_skipped: list[int]  # already OCR'd earlier


def ocr_pages(project: Project, aris_number: int, pdf: Path, pages: list[int]) -> OcrResult:
    """OCR ``pages`` (1-based) and append the text to the report's text file.

    Pages that already have OCR text in the file are skipped. The text file must exist
    (run text extraction first). Raises ``OcrUnavailableError`` if EasyOCR is missing.
    """
    path = text_path(project, aris_number)
    existing = read_ocr_blocks(path)
    todo = [p for p in pages if p not in existing]
    skipped = [p for p in pages if p in existing]
    if not todo:
        return OcrResult(path, [], skipped)

    reader = load_reader()
    new: dict[int, str] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for page in todo:
            image = render_page(pdf, page, Path(tmp) / f"p{page}.png", DEFAULT_DPI)
            lines = reader.readtext(str(image), detail=0, paragraph=True)
            new[page] = "\n".join(str(line) for line in lines).strip()

    write_with_ocr(path, text_base(path), {**existing, **new})
    project.provenance.log(
        dataset="aris_text",
        source="gossan",
        identifier=f"ARIS {aris_number}",
        url=None,
        license=DERIVED_LICENSE,
        operation="ocr",
        parameters={
            "input": f"raw/aris_pdfs/{pdf.name}",
            "output": path.relative_to(project.root).as_posix(),
            "engine": "easyocr",
            "languages": LANGUAGES,
            "dpi": DEFAULT_DPI,
            "pages": todo,
            "chars_per_page": {str(p): len(t) for p, t in new.items()},
        },
    )
    return OcrResult(path, todo, skipped)
