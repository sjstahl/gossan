"""Page selection and rendering shared by the OCR and scanned-table steps."""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf

from gossan.core.project import Project

#: Longest image side sent to OCR or Claude, in pixels (Claude downsizes beyond ~1568 px
#: anyway; the cap keeps files and requests small).
MAX_SIDE_PX = 3000
DEFAULT_DPI = 200


def parse_pages(spec: str, n_pages: int | None = None) -> list[int]:
    """'43-47, 50' -> [43, 44, 45, 46, 47, 50] (1-based, sorted, unique).

    Raises ``ValueError`` on bad syntax, page 0, reversed ranges, or pages past ``n_pages``.
    """
    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        m = re.fullmatch(r"(\d+)(?:\s*-\s*(\d+))?", part)
        if not m:
            raise ValueError(f"bad page spec {part!r}: use numbers and ranges like '3, 43-47'")
        start, end = int(m[1]), int(m[2] or m[1])
        if start < 1 or end < start:
            raise ValueError(f"bad page range {part!r}")
        pages.update(range(start, end + 1))
    if n_pages is not None and (too_big := [p for p in pages if p > n_pages]):
        raise ValueError(f"page {min(too_big)} is past the end of the document ({n_pages} pages)")
    return sorted(pages)


def page_image_path(project: Project, aris_number: int, page: int) -> Path:
    return project.root / "digitize" / "pages" / f"{aris_number:05d}_p{page}.png"


def render_page(
    pdf: Path, page: int, dest: Path, dpi: int = DEFAULT_DPI, max_side: int = MAX_SIDE_PX
) -> Path:
    """Render 1-based ``page`` of ``pdf`` to a PNG at ``dest`` (parent folders are created)."""
    with pymupdf.open(pdf) as doc:
        pg = doc[page - 1]
        scale = dpi / 72
        longest = max(pg.rect.width, pg.rect.height) * scale
        if longest > max_side:
            scale *= max_side / longest
        pix = pg.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    dest.parent.mkdir(parents=True, exist_ok=True)
    pix.save(dest)
    return dest
