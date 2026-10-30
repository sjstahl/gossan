"""Run the checks, write the QA report (Markdown, PDF and JSON) and log it to provenance.

The report is an internal record for whoever signs off the deliverable: it names layers and counts, not
client-facing content. Its licence is derived from the datasets the checks read, so a project holding
Access Only data produces a report that says so.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from gossan import __version__
from gossan.core.licensing import derived_license_from
from gossan.core.project import Project
from gossan.qa.checks import SEVERITIES, QaResult, Severity, run_checks

QA_MD = "qa_report.md"
QA_PDF = "qa_report.pdf"
QA_JSON = "qa_report.json"
NOTE = "Internal QA record. Not for distribution to clients."
LABEL: dict[Severity, str] = {"error": "ERROR", "warn": "WARNING", "info": "info"}
COLOUR: dict[Severity, str] = {"error": "#b3261e", "warn": "#9a6700", "info": "#555555"}


def verdict(result: QaResult) -> str:
    if result.count("error"):
        return f"NOT READY: {result.count('error')} error(s) to fix before delivery"
    if result.count("warn"):
        return f"READY WITH CARE: no errors, {result.count('warn')} warning(s) to look at"
    return "READY: no errors or warnings"


def markdown(project: Project, result: QaResult) -> str:
    cfg = project.config
    lines = [
        f"# QA report: {cfg.name}",
        "",
        f"_{NOTE}_",
        "",
        f"Checked {result.checked_at} with Gossan {__version__}. CRS {cfg.crs}, jurisdiction {cfg.jurisdiction}.",
        "",
        f"**{verdict(result)}**",
        "",
        f"{result.count('error')} error(s), {result.count('warn')} warning(s), {result.count('info')} note(s).",
        "",
    ]
    for severity in SEVERITIES:
        found = [f for f in result.findings if f.severity == severity]
        if not found:
            continue
        lines += [f"## {LABEL[severity].capitalize() if severity == 'info' else LABEL[severity].title()}s" if severity != "info" else "## Notes", ""]
        for f in found:
            lines.append(f"- **{f.subject}** ({f.check}): {f.message}")
            if f.ids:
                lines.append(f"  - {'; '.join(f.ids)}")
        lines.append("")
    lines += ["## Layer inventory", "", "| Layer | Kind | Geometry | Features | CRS |", "|---|---|---|---|---|"]
    lines += [f"| {r.name} | {r.kind} | {r.geometry} | {r.features:,} | {r.crs} |" for r in result.layers]
    lines += ["", "## Thresholds", "", ", ".join(f"{k} {v:g}" if isinstance(v, int | float) else f"{k} {v}" for k, v in result.config.items()), ""]
    return "\n".join(lines)


def build_pdf(project: Project, result: QaResult, path) -> None:
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9, leading=12)
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=10, textColor=colors.HexColor("#555555"))
    story: list = [
        Paragraph(f"QA report: {escape(project.config.name)}", styles["Title"]),
        Paragraph(escape(NOTE), small),
        Paragraph(f"Checked {result.checked_at} with Gossan {__version__}. CRS {project.config.crs}, jurisdiction {project.config.jurisdiction}.", body),
        Spacer(1, 6),
        Paragraph(f"<b>{escape(verdict(result))}</b>", ParagraphStyle("v", parent=body, fontSize=11, leading=14, textColor=colors.HexColor(COLOUR["error" if result.count("error") else "warn" if result.count("warn") else "info"]))),
        Spacer(1, 8),
    ]
    for severity in SEVERITIES:
        found = [f for f in result.findings if f.severity == severity]
        if not found:
            continue
        story.append(Paragraph({"error": "Errors", "warn": "Warnings", "info": "Notes"}[severity], styles["Heading2"]))
        rows = []
        for f in found:
            detail = escape(f.message) + (f"<br/><font size=7 color='#555555'>{escape('; '.join(f.ids))}</font>" if f.ids else "")
            rows.append([Paragraph(f"<b>{escape(f.subject)}</b><br/><font size=7>{escape(f.check)}</font>", body), Paragraph(detail, body)])
        table = Table(rows, colWidths=[1.7 * inch, 5.3 * inch])
        table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")), ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(COLOUR[severity]))]))
        story += [table, Spacer(1, 8)]
    story.append(Paragraph("Layer inventory", styles["Heading2"]))
    inv = [["Layer", "Kind", "Geometry", "Features", "CRS"]] + [[Paragraph(escape(r.name), small), r.kind, Paragraph(escape(r.geometry), small), f"{r.features:,}", Paragraph(escape(r.crs), small)] for r in result.layers]
    table = Table(inv, colWidths=[2.4 * inch, 0.7 * inch, 1.5 * inch, 0.9 * inch, 1.5 * inch], repeatRows=1)
    table.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(table)

    def footer(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(0.75 * inch, 0.5 * inch, f"{NOTE}  Page {doc.page}")
        canvas.restoreState()

    SimpleDocTemplate(str(path), pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch, topMargin=0.7 * inch, bottomMargin=0.8 * inch, title=f"QA report: {project.config.name}", author="Gossan").build(story, onFirstPage=footer, onLaterPages=footer)


def run_qa(project: Project, *, write: bool = True, package: bool = True) -> QaResult:
    """Run every check; with ``write`` also save ``outputs/qa_report.{md,pdf,json}`` and log them."""
    result = run_checks(project, package=package)
    if not write:
        return result
    out = project.outputs_dir
    out.mkdir(exist_ok=True)
    (out / QA_MD).write_text(markdown(project, result), encoding="utf-8")
    (out / QA_JSON).write_text(
        json.dumps({"verdict": verdict(result), "passed": result.passed, "checked_at": result.checked_at, "config": result.config, "findings": [vars(f) for f in result.findings], "layers": [vars(r) for r in result.layers]}, indent=2),
        encoding="utf-8",
    )
    build_pdf(project, result, out / QA_PDF)
    inputs = [r.name for r in result.layers if r.kind == "vector"]
    licence = derived_license_from(project.provenance.read(), inputs)
    for name in (QA_MD, QA_PDF, QA_JSON):
        project.provenance.log(
            dataset="qa_report",
            source="gossan",
            identifier=f"outputs/{name}",
            url=None,
            license=licence,
            operation="qa_check",
            parameters={
                "errors": result.count("error"), "warnings": result.count("warn"), "notes": result.count("info"),
                "passed": result.passed, "config": result.config, "sha256": hashlib.sha256((out / name).read_bytes()).hexdigest(),
                "checked_at": result.checked_at or datetime.now(UTC).isoformat(),
            },
        )  # fmt: skip
    return result
