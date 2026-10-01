"""Finding the downloaded ARIS PDFs that digitize works on."""

from __future__ import annotations

import re
from pathlib import Path

from gossan.core.project import Project
from gossan.harvest.bc.aris_pdf import RAW_DIR

PDF_NAME = re.compile(r"^ARIS_(\d+)\.pdf$", re.IGNORECASE)
INSUFFICIENT_PDFS = "no ARIS PDFs in raw/aris_pdfs; download some with 'gossan harvest --limit-pdfs N'"


def list_aris_pdfs(project: Project) -> dict[int, Path]:
    """ARIS number -> PDF path for every downloaded report, in number order."""
    folder = project.raw_dir / RAW_DIR
    found = {}
    for p in folder.glob("*.pdf") if folder.is_dir() else []:
        m = PDF_NAME.match(p.name)
        if m:
            found[int(m[1])] = p
    return dict(sorted(found.items()))


def select_pdfs(project: Project, report: int | None) -> dict[int, Path]:
    """All downloaded PDFs, or just ``report``. Raises ``FileNotFoundError`` with advice."""
    pdfs = list_aris_pdfs(project)
    if report is None:
        if not pdfs:
            raise FileNotFoundError(INSUFFICIENT_PDFS)
        return pdfs
    if report not in pdfs:
        raise FileNotFoundError(
            f"no PDF for ARIS {report} in raw/aris_pdfs; download it with 'gossan harvest "
            "--limit-pdfs N' (it must be in the aris_index layer)"
        )
    return {report: pdfs[report]}
