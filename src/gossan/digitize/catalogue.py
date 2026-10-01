"""Catalogue of downloaded reports: page counts, likely maps, text-layer coverage.

One row per report in the ``catalogue`` layer of ``data.gpkg``. Where the report is in the
``aris_index`` layer, the row carries its location (point), year, operator, title and work
types, so the inventory maps in QGIS and can feed ``leads``; otherwise the geometry is empty.

Map heuristic (a guess, to be confirmed by a person): a page is a likely map when it is
larger than letter/A4 (either side beyond letter 612 x 792 pt / A4 595 x 842 pt, by more
than a 6 pt tolerance) or landscape. Both counts are kept so the guess can be tuned.
Page size is as displayed (``/Rotate`` applied).

PyMuPDF is dual-licensed AGPL-3.0 / Artifex commercial; see ``text.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pymupdf
import pyogrio

from gossan.core.project import Project
from gossan.digitize.text import DERIVED_LICENSE, has_text, page_ranges

CATALOGUE_LAYER = "catalogue"
TOLERANCE_PT = 6.0
# A page fits "standard" if its short side <= letter's and long side <= A4's (the larger).
STANDARD_SHORT_PT = 612.0
STANDARD_LONG_PT = 842.0
INDEX_COLUMNS = {
    "REPORT_YEAR": "report_year",
    "OPERATOR_NAMES": "operator",
    "ASSESSMENT_REPORT_TITLE": "title",
    "GENERAL_WORK_TYPES": "work_types",
    "PROPERTY_NAME": "property_name",
}


def classify_page(width: float, height: float) -> tuple[bool, bool]:
    """(oversize, landscape) for a page of the given size in points."""
    short, long_ = sorted((width, height))
    oversize = short > STANDARD_SHORT_PT + TOLERANCE_PT or long_ > STANDARD_LONG_PT + TOLERANCE_PT
    return oversize, width > height


def inspect_pdf(pdf: Path) -> dict[str, object]:
    """Facts about one PDF; raises on a file PyMuPDF cannot open."""
    oversize, landscape, maps, no_text = [], [], [], []
    with pymupdf.open(pdf) as doc:
        if doc.needs_pass:
            raise ValueError("PDF is password protected")
        n_pages = doc.page_count
        for i, page in enumerate(doc, start=1):
            big, wide = classify_page(page.rect.width, page.rect.height)
            if big:
                oversize.append(i)
            if wide:
                landscape.append(i)
            if big or wide:
                maps.append(i)
            if not has_text(page):
                no_text.append(i)
    return {
        "pages": n_pages,
        "n_oversize_pages": len(oversize),
        "n_landscape_pages": len(landscape),
        "n_map_pages": len(maps),
        "map_pages": page_ranges(maps),
        "n_text_pages": n_pages - len(no_text),
        "n_no_text_pages": len(no_text),
        "pdf_bytes": pdf.stat().st_size,
    }


def build_catalogue(project: Project, pdfs: dict[int, Path]) -> tuple[gpd.GeoDataFrame, dict[int, str]]:
    """Inspect ``pdfs`` and update the ``catalogue`` layer (rows for other reports are kept).

    Returns the rows written this run and ``{aris_number: error}`` for unreadable PDFs.
    """
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows, failed = [], {}
    for number, pdf in pdfs.items():
        try:
            facts = inspect_pdf(pdf)
        except (RuntimeError, ValueError, OSError) as e:  # pymupdf raises RuntimeError subclasses
            failed[number] = str(e)
            continue
        rows.append({"aris_number": number, "pdf_file": f"raw/aris_pdfs/{pdf.name}", **facts, "cataloged_at": now})
    new = _with_index_metadata(project, pd.DataFrame(rows))
    if new.empty:
        return new, failed

    layers = pyogrio.list_layers(project.data_gpkg)[:, 0] if project.data_gpkg.exists() else []
    if CATALOGUE_LAYER in layers:
        old = gpd.read_file(project.data_gpkg, layer=CATALOGUE_LAYER, engine="pyogrio")
        old = old[~old["aris_number"].isin(new["aris_number"])]
        all_rows = gpd.GeoDataFrame(pd.concat([old, new], ignore_index=True), crs=new.crs)
    else:
        all_rows = new
    all_rows = all_rows.sort_values("aris_number").reset_index(drop=True)
    all_rows.to_file(project.data_gpkg, layer=CATALOGUE_LAYER, driver="GPKG", engine="pyogrio")

    project.provenance.log(
        dataset=CATALOGUE_LAYER,
        source="gossan",
        identifier=CATALOGUE_LAYER,
        url=None,
        license=DERIVED_LICENSE,
        operation="write_catalogue",
        parameters={
            "gpkg": project.data_gpkg.name,
            "reports_updated": [int(n) for n in new["aris_number"]],
            "n_reports_total": len(all_rows),
            "map_heuristic": "larger than letter/A4 (6 pt tolerance) or landscape",
            "failed": {str(k): v for k, v in failed.items()},
        },
    )
    return new, failed


def _with_index_metadata(project: Project, rows: pd.DataFrame) -> gpd.GeoDataFrame:
    """Add location and descriptive fields from ``aris_index``, where the report is in it."""
    crs = project.config.crs
    if rows.empty:
        return gpd.GeoDataFrame(rows, geometry=[], crs=crs)
    layers = pyogrio.list_layers(project.data_gpkg)[:, 0] if project.data_gpkg.exists() else []
    index = pd.DataFrame(columns=["ARIS_NUMBER", "geometry", *INDEX_COLUMNS])
    if "aris_index" in layers:
        index = gpd.read_file(project.data_gpkg, layer="aris_index", engine="pyogrio").to_crs(crs)
    index = index.drop_duplicates("ARIS_NUMBER").set_index("ARIS_NUMBER")
    out = rows.copy()
    for src, dest in INDEX_COLUMNS.items():
        out[dest] = out["aris_number"].map(index[src]) if src in index else None
    geometry = out["aris_number"].map(index["geometry"]) if "geometry" in index else None
    return gpd.GeoDataFrame(out, geometry=geometry, crs=crs)
