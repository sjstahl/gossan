"""Tables from pages that have a text layer, using pdfplumber.

pdfplumber finds tables from ruling lines (and aligned text); it does not read scans. Each
table goes to ``digitize/tables/digital/<aris_id>_p<N>.csv`` (a page's second and later
tables get ``_t2``, ``_t3``, ...). The output is as extracted: header rows, merged cells and
split rows are not interpreted, and nothing is queued for review yet.

pdfplumber and pdfminer.six are MIT-licensed.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

from gossan.core.project import Project
from gossan.digitize.text import DERIVED_LICENSE

#: Smaller grids are almost always layout artefacts (boxed text, rules), not tables.
MIN_ROWS = 2
MIN_COLS = 2


@dataclass
class TableResult:
    files: list[Path]
    pages_without_tables: list[int]
    errors: dict[int, str]  # page -> reason, for pages pdfplumber could not process


def digital_dir(project: Project) -> Path:
    return project.root / "digitize" / "tables" / "digital"


def _clean(cell: str | None) -> str:
    return " ".join((cell or "").split())


def extract_digital_tables(
    project: Project, aris_number: int, pdf: Path, pages: list[int]
) -> TableResult:
    """Extract tables from ``pages`` (1-based) of ``pdf`` and write one CSV per table.

    Earlier CSVs for the same pages are replaced.
    """
    out_dir = digital_dir(project)
    files: list[Path] = []
    empty: list[int] = []
    errors: dict[int, str] = {}
    with pdfplumber.open(pdf) as doc:
        for page in pages:
            for stale in out_dir.glob(f"{aris_number:05d}_p{page}.csv"):
                stale.unlink()
            for stale in out_dir.glob(f"{aris_number:05d}_p{page}_t*.csv"):
                stale.unlink()
            try:
                found = doc.pages[page - 1].extract_tables()
            except Exception as e:  # noqa: BLE001 - pdfminer raises a wide variety on malformed pages
                errors[page] = f"{type(e).__name__}: {e}"
                continue
            tables = [
                [[_clean(c) for c in row] for row in t]
                for t in found
                if len(t) >= MIN_ROWS and max(len(r) for r in t) >= MIN_COLS
            ]
            if not tables:
                empty.append(page)
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            for k, rows in enumerate(tables, start=1):
                suffix = "" if k == 1 else f"_t{k}"
                path = out_dir / f"{aris_number:05d}_p{page}{suffix}.csv"
                with path.open("w", encoding="utf-8", newline="") as f:
                    csv.writer(f).writerows(rows)
                files.append(path)
                project.provenance.log(
                    dataset="aris_tables_digital",
                    source="gossan",
                    identifier=f"ARIS {aris_number} p{page}" + (f" t{k}" if k > 1 else ""),
                    url=None,
                    license=DERIVED_LICENSE,
                    operation="extract_table",
                    parameters={
                        "input": f"raw/aris_pdfs/{pdf.name}",
                        "output": path.relative_to(project.root).as_posix(),
                        "engine": f"pdfplumber {pdfplumber.__version__}",
                        "page": page,
                        "rows": len(rows),
                        "cols": max(len(r) for r in rows),
                    },
                )
    return TableResult(files, empty, errors)
