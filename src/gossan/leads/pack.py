"""Sample pack skeleton: a one-property summary of a lead and the reports available.

This is a stub for the Phase 4 pitch pack. It lists what exists (from the tenure and the
ARIS index) and whether each report PDF is already in ``raw/aris_pdfs/``. It does not
open the PDFs: text, table and image extraction, and the before/after samples, belong to
the ``digitize`` module.

ARIS is Access Only, so the pack is marked internal; get the Province's permission
before sending anything derived from ARIS to a prospect.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from docx import Document

from gossan.core.project import Project
from gossan.harvest.bc.aris_pdf import RAW_DIR as ARIS_PDF_DIR
from gossan.leads.generator import DERIVED_LICENSE, LEADS_LAYER
from gossan.leads.scoring import ARIS_LAYER, MissingLayerError, read_layer


class LeadNotFoundError(Exception):
    """No lead with that id."""


def reports_for_lead(lead: gpd.GeoSeries, aris: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """ARIS reports whose location falls inside the lead's tenure, oldest first."""
    inside = aris[aris.intersects(lead.geometry)]
    return inside.sort_values(["REPORT_YEAR", "ARIS_NUMBER"]).reset_index(drop=True)


def build_pack(project: Project, lead_id: str) -> Path:
    """Write ``outputs/lead_<id>_pack.docx`` for a lead from the ``leads`` layer."""
    try:
        leads = read_layer(project, LEADS_LAYER)
    except MissingLayerError as e:
        raise MissingLayerError("no leads layer yet; run 'gossan leads run' first") from e
    match = leads[leads["lead_id"] == str(lead_id)]
    if match.empty:
        raise LeadNotFoundError(f"lead {lead_id!r} not found; see outputs/leads_ranked.csv")
    lead = match.iloc[0]
    aris = read_layer(project, ARIS_LAYER).to_crs(leads.crs)
    reports = reports_for_lead(lead, aris)

    doc = Document()
    doc.add_heading(f"Lead {lead['lead_id']}: {lead['claim_name']}", level=0)
    doc.add_paragraph(
        "INTERNAL DRAFT. Contains ARIS data (Access Only): do not send to a prospect "
        "without the Province's written permission."
    )

    doc.add_heading("Property", level=1)
    fields = [
        ("Tenure type", lead["tenure_type"]),
        ("Owner", lead["owner"]),
        ("Issued", lead["issue_date"]),
        ("Good to", lead["good_to_date"]),
        ("Area (ha)", lead["area_ha"]),
        ("Lead rank / score", f"{lead['rank']} / {lead['score']}"),
        ("Recency points", lead["recency_points"]),
        ("Report-count points", lead["count_points"]),
        ("Legacy-report points", lead["legacy_points"]),
    ]
    table = doc.add_table(rows=0, cols=2, style="Light List")
    for label, value in fields:
        cells = table.add_row().cells
        cells[0].text, cells[1].text = label, "" if pd.isna(value) else str(value)

    doc.add_heading("Assessment reports on the property", level=1)
    if reports.empty:
        doc.add_paragraph("No ARIS reports are located on this tenure.")
    else:
        doc.add_paragraph(
            f"{len(reports)} report(s), {int(reports['REPORT_YEAR'].min())} to "
            f"{int(reports['REPORT_YEAR'].max())}. A report's location is a single point; "
            "it may cover ground beyond this tenure."
        )
        table = doc.add_table(rows=1, cols=6, style="Light List")
        for cell, head in zip(
            table.rows[0].cells,
            ["ARIS", "Year", "Title", "Operator", "Work", "PDF in project"],
            strict=True,
        ):
            cell.text = head
        for r in reports.itertuples():
            number = int(r.ARIS_NUMBER)
            have_pdf = (project.raw_dir / ARIS_PDF_DIR / f"ARIS_{number:05d}.pdf").exists()
            row = table.add_row().cells
            values = [
                str(number),
                str(int(r.REPORT_YEAR)),
                r.ASSESSMENT_REPORT_TITLE,
                r.OPERATOR_NAMES,
                r.GENERAL_WORK_TYPES,
                "yes" if have_pdf else "no",
            ]
            for cell, value in zip(row, values, strict=True):
                cell.text = "" if pd.isna(value) else str(value)

    doc.add_heading("Next steps", level=1)
    doc.add_paragraph(
        "Pick two or three reports with maps or drill tables for a before/after sample "
        "(digitization is not part of this pack yet)."
    )

    project.outputs_dir.mkdir(exist_ok=True)
    path = project.outputs_dir / f"lead_{lead['lead_id']}_pack.docx"
    doc.save(path)
    project.provenance.log(
        dataset="lead_pack",
        source="gossan",
        identifier=f"outputs/{path.name}",
        url=None,
        license=DERIVED_LICENSE,
        operation="build_lead_pack",
        parameters={"lead_id": str(lead["lead_id"]), "n_reports": len(reports)},
    )
    return path
