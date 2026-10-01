"""Gossan command-line entry point."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Annotated

import typer

from gossan.core.aoi import read_aoi, utm_zone_for, write_aoi
from gossan.core.config import AOI_FILE, ProjectConfig
from gossan.core.ingest import IngestError
from gossan.core.ingest import ingest as ingest_file
from gossan.core.project import Project
from gossan.core.provenance import ProvenanceLog
from gossan.digitize.catalogue import build_catalogue
from gossan.digitize.claude_vision import ClaudeVisionError, model_name
from gossan.digitize.claude_vision import (
    extract_scanned_tables as extract_scanned_tables_for,
)
from gossan.digitize.inputs import select_pdfs
from gossan.digitize.ocr import OcrUnavailableError, ocr_pages
from gossan.digitize.pages import parse_pages
from gossan.digitize.review import init_queue, read_queue, summarize
from gossan.digitize.tables import extract_digital_tables
from gossan.digitize.text import extract_text, page_ranges
from gossan.harvest import CONNECTORS
from gossan.harvest.bc.aris_pdf import ArisPdfDownloader
from gossan.harvest.national.nrcan_geophysics import ManualDownloadRequired
from gossan.leads.generator import run_leads
from gossan.leads.pack import LeadNotFoundError, build_pack
from gossan.leads.scoring import MissingLayerError
from gossan.report.builder import build_report

app = typer.Typer(
    help="Gossan: turn a claim polygon into client-ready deliverables.",
    no_args_is_help=True,
)


leads_app = typer.Typer(help="Find and package properties likely to need digitization.", no_args_is_help=True)
app.add_typer(leads_app, name="leads")


@app.callback()
def main() -> None:
    """Gossan: turn a claim polygon into client-ready deliverables."""


@app.command()
def new(
    name: Annotated[str, typer.Argument(help="Project name; also the new folder's name.")],
    aoi: Annotated[
        Path,
        typer.Option(
            "--aoi",
            exists=True,
            dir_okay=False,
            readable=True,
            help="AOI polygon file (shp, gpkg, geojson, ...).",
        ),
    ],
    crs: Annotated[
        str | None,
        typer.Option(help="Project CRS, e.g. EPSG:26910. Default: UTM zone of the AOI centroid."),
    ] = None,
    project: Annotated[
        bool,
        typer.Option(
            "--project/--no-project",
            help="Reproject the AOI to the project CRS. --no-project keeps the AOI's own CRS.",
        ),
    ] = True,
    client: Annotated[str | None, typer.Option(help="Client name to record.")] = None,
) -> None:
    """Create a project folder with aoi.gpkg and a default project.yaml.

    If the AOI file has no CRS, assign one in QGIS first (manual fallback).
    """
    project_dir = Path(name)
    try:
        config = ProjectConfig(name=name, client=client, crs="EPSG:0")
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="NAME") from e
    if project_dir.exists():
        typer.secho(f"Error: {project_dir} already exists", fg="red", err=True)
        raise typer.Exit(1)

    try:
        gdf = read_aoi(aoi)
    except ValueError as e:
        typer.secho(f"Error: {e}", fg="red", err=True)
        raise typer.Exit(1) from e

    gdf_source_crs = gdf.crs.to_string()
    reprojected = project
    zone, utm_crs = utm_zone_for(gdf)
    if not project:
        config.crs, config.utm_zone = gdf.crs.to_string(), None
    elif crs is not None:
        gdf = gdf.to_crs(crs)
        config.crs, config.utm_zone = gdf.crs.to_string(), None
    else:
        gdf = gdf.to_crs(utm_crs)
        config.crs, config.utm_zone = utm_crs, zone

    project_dir.mkdir(parents=True)
    write_aoi(gdf, project_dir / AOI_FILE)
    config.save(project_dir)
    ProvenanceLog(project_dir).log(
        dataset="aoi",
        source="user-supplied",
        identifier=aoi.name,
        url=None,
        license="Client-supplied; confidential",
        operation="import_aoi",
        parameters={"source_crs": gdf_source_crs, "project_crs": config.crs, "reprojected": reprojected},
    )
    typer.echo(f"Created project {name} (CRS {config.crs})")


def _open_project(path: Path) -> Project:
    try:
        return Project(path)
    except FileNotFoundError as e:
        typer.secho(f"Error: {e}", fg="red", err=True)
        raise typer.Exit(1) from e


@app.command()
def harvest(
    project: Annotated[Path, typer.Argument(help="Project folder created by 'gossan new'.")],
    source: Annotated[
        str | None,
        typer.Option(
            help="Comma-separated sources to harvest. "
            f"Available: {', '.join(CONNECTORS)}. "
            f"Default: {', '.join(n for n, c in CONNECTORS.items() if c.default)}."
        ),
    ] = None,
    limit_pdfs: Annotated[
        int,
        typer.Option(
            min=0,
            help="Download up to this many ARIS report PDFs (newest first) from the aris_index "
            "layer into raw/aris_pdfs/. 0 (default) downloads none; reports can be tens of MB.",
        ),
    ] = 0,
) -> None:
    """Download public data for the project AOI into data.gpkg and rasters/.

    Originals are kept in raw/ and every retrieval is logged to provenance.jsonl.
    A source that fails does not stop the others; the command exits 1 if any failed.

    Manual fallback: if a source is unavailable, download the data yourself and load
    it with 'gossan ingest' (vector files and .tif grids). nrcan_geophysics needs a
    configured WCS (see its module docs) and otherwise prints manual instructions.
    """
    proj = _open_project(project)
    names = (
        [s.strip() for s in source.split(",") if s.strip()]
        if source
        else [n for n, c in CONNECTORS.items() if c.default]
    )
    unknown = [n for n in names if n not in CONNECTORS]
    if unknown:
        raise typer.BadParameter(f"unknown source(s): {', '.join(unknown)}", param_hint="--source")

    failed = False
    for n in names:
        connector = CONNECTORS[n](proj)
        try:
            count = connector.run()
        except ManualDownloadRequired as e:
            typer.secho(f"{n}: {e}", fg="yellow", err=True)
            failed = True
        except OSError as e:  # network failures (URLError is an OSError)
            typer.secho(f"Error: {n}: {e}", fg="red", err=True)
            typer.echo("Fallback: download the data manually and use 'gossan ingest'.", err=True)
            failed = True
        else:
            typer.echo(f"{n}: {count} {connector.output_description}")

    if limit_pdfs:
        try:
            result = ArisPdfDownloader(proj).run(limit_pdfs)
        except FileNotFoundError as e:
            typer.secho(f"Error: {e}", fg="red", err=True)
            raise typer.Exit(1) from e
        typer.echo(
            f"aris_pdfs: {len(result.downloaded)} downloaded, {result.skipped_existing} already "
            f"present, {len(result.failed)} failed, {result.remaining} more not fetched "
            "(raise --limit-pdfs)"
        )
        for number, reason in result.failed:
            typer.secho(f"  ARIS {number}: {reason}", fg="yellow", err=True)
    if failed:
        raise typer.Exit(1)


@app.command()
def ingest(
    project: Annotated[Path, typer.Argument(help="Project folder created by 'gossan new'.")],
    file_path: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False, readable=True, help="Local vector file or .tif raster."
        ),
    ],
    layer: Annotated[
        str, typer.Option("--layer", help="Layer name in data.gpkg, or raster name in rasters/.")
    ],
    source: Annotated[
        str, typer.Option(help="Who supplied the data, for provenance and the report.")
    ] = "user-supplied",
    license: Annotated[
        str, typer.Option(help="Licence or terms of the data, for provenance and the report.")
    ] = "unspecified (user-supplied)",
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Replace the layer or raster if it exists.")
    ] = False,
) -> None:
    """Load a local file into the project; the manual fallback for any harvest source.

    Vector files (shp, geojson, gpkg, ...) are reprojected to the project CRS, clipped
    to the AOI and written to data.gpkg. A .tif raster is clipped, reprojected and
    written as a Cloud Optimized GeoTIFF to rasters/<layer>.tif. A copy of the original
    goes to raw/ingest/ and the ingest is logged to provenance.jsonl. The file must
    have a CRS defined.
    """
    proj = _open_project(project)
    try:
        message = ingest_file(
            proj, file_path, layer, source=source, license=license, overwrite=overwrite
        )
    except IngestError as e:
        typer.secho(f"Error: {e}", fg="red", err=True)
        raise typer.Exit(1) from e
    typer.echo(message)


@app.command()
def report(
    project: Annotated[Path, typer.Argument(help="Project folder created by 'gossan new'.")],
) -> None:
    """Build outputs/map.png and outputs/report.docx (a draft for geologist review)."""
    proj = _open_project(project)
    path = build_report(proj)
    typer.echo(f"Wrote {path}")


@app.command("digitize")
def digitize_cmd(
    project: Annotated[Path, typer.Argument(help="Project folder with downloaded ARIS PDFs.")],
    report: Annotated[
        int | None,
        typer.Option("--report", help="ARIS number to process. Default: every PDF in raw/aris_pdfs."),
    ] = None,
    target_pages: Annotated[
        str | None,
        typer.Option(
            "--target-pages",
            help="Pages to limit OCR, table extraction and Claude calls to, e.g. '43-47' or "
            "'3, 43-47'. Needs --report.",
        ),
    ] = None,
    extract_scanned_tables: Annotated[
        bool,
        typer.Option(
            "--extract-scanned-tables",
            help="Send the --target-pages page images to Claude (Anthropic API: costs money, "
            "needs ANTHROPIC_API_KEY) to read drillhole/assay tables. Results go to "
            "digitize/tables/scanned/ and the review queue as PENDING.",
        ),
    ] = False,
    ocr: Annotated[
        bool, typer.Option("--ocr/--no-ocr", help="OCR pages with no embedded text (EasyOCR).")
    ] = True,
) -> None:
    """Catalogue the downloaded ARIS PDFs, extract text and tables.

    Always: the 'catalogue' layer in data.gpkg, digitize/text/<aris_id>.txt, and digital
    tables (pdfplumber) in digitize/tables/digital/. Pages with no embedded text are OCR'd
    with EasyOCR (an optional install, the "ocr" extra; skipped with a warning if missing)
    and appended to the text file as [OCR EXTRACTED]. --target-pages limits the page-level
    steps to the pages you name. Scanned drillhole/assay tables are read by Claude only with
    --extract-scanned-tables, only on --target-pages, and never more than 25 pages per run;
    every table it extracts is queued for human review ('gossan review'). Get PDFs first with
    'gossan harvest --limit-pdfs N'. Unreadable PDFs are reported and skipped.
    """
    proj = _open_project(project)
    if target_pages and report is None:
        raise typer.BadParameter("page numbers only make sense for one report", param_hint="--target-pages")
    if extract_scanned_tables and not target_pages:
        raise typer.BadParameter(
            "name the pages to send to Claude, e.g. --target-pages 43-47 "
            "(the whole document is never sent by default)",
            param_hint="--extract-scanned-tables",
        )
    try:
        pdfs = select_pdfs(proj, report)
    except FileNotFoundError as e:
        typer.secho(f"Error: {e}", fg="red", err=True)
        raise typer.Exit(1) from e

    init_queue(proj)
    rows, failed = build_catalogue(proj, pdfs)
    facts = {int(r.aris_number): r for r in rows.itertuples()}
    ocr_missing = False
    n_done = 0
    for number, pdf in pdfs.items():
        if number in failed:
            continue
        try:
            targets = parse_pages(target_pages, facts[number].pages) if target_pages else None
        except ValueError as e:
            failed[number] = str(e)
            continue
        try:
            result = extract_text(proj, number, pdf)
        except (RuntimeError, ValueError, OSError) as e:
            failed[number] = str(e)
            continue
        n_done += 1
        r = facts[number]
        typer.echo(
            f"ARIS {number}: {r.pages} page(s), {r.n_map_pages} likely map page(s), "
            f"text -> {result.path.relative_to(proj.root)}"
        )
        if result.no_text_pages:
            typer.secho(
                f"  warning: ARIS {number}: no embedded text on page(s) "
                f"{page_ranges(result.no_text_pages)} (scanned)",
                fg="yellow",
                err=True,
            )

        scope = set(targets) if targets else set(range(1, r.pages + 1))
        scanned = [p for p in result.no_text_pages if p in scope]
        if ocr and scanned and not ocr_missing:
            try:
                done = ocr_pages(proj, number, pdf, scanned)
                typer.echo(
                    f"  OCR: {len(done.pages_done)} page(s) appended"
                    + (f", {len(done.pages_skipped)} already done" if done.pages_skipped else "")
                )
            except OcrUnavailableError as e:
                ocr_missing = True
                typer.secho(f"  warning: {e}", fg="yellow", err=True)

        with_text = sorted(scope - set(result.no_text_pages))
        tables = extract_digital_tables(proj, number, pdf, with_text)
        typer.echo(f"  digital tables: {len(tables.files)} file(s) from {len(with_text)} text page(s)")
        for page, reason in tables.errors.items():
            typer.secho(f"  warning: table extraction failed on page {page}: {reason}", fg="yellow", err=True)

        if extract_scanned_tables:
            try:
                sc = extract_scanned_tables_for(proj, number, pdf, targets or [])
            except ClaudeVisionError as e:
                typer.secho(f"Error: {e}", fg="red", err=True)
                raise typer.Exit(1) from e
            typer.echo(
                f"  Claude ({model_name()}): {len(sc.written)} table(s) queued for review, "
                f"{len(sc.no_table)} page(s) without a table, {len(sc.already_done)} already "
                f"extracted, {len(sc.failed)} failed; {sc.input_tokens} in / {sc.output_tokens} out tokens"
            )
            for page, reason in sc.failed.items():
                typer.secho(f"  warning: Claude failed on page {page}: {reason}", fg="yellow", err=True)
                failed[number] = f"scanned-table extraction failed on page(s) {sorted(sc.failed)}"

    for number, reason in failed.items():
        typer.secho(f"Error: ARIS {number}: {reason}", fg="red", err=True)
    typer.echo(f"{n_done} report(s) processed, {len(failed)} failed; catalogue updated")
    if failed:
        raise typer.Exit(1)


@app.command()
def review(
    project: Annotated[Path, typer.Argument(help="Project folder created by 'gossan new'.")],
) -> None:
    """Summarise the items waiting for human review (digitize/review_queue.csv)."""
    proj = _open_project(project)
    items = read_queue(proj)
    pending = [i for i in items if i.status == "PENDING"]
    if not pending:
        typer.echo("No items pending review.")
        return
    counts = ", ".join(f"{n} {t}" for t, n in sorted(summarize(items)["PENDING"].items()))
    typer.echo(f"{len(pending)} item(s) pending review ({counts})")
    for i in pending:
        where = f" p.{i.page}" if i.page else ""
        typer.echo(f"  {i.item_id}  {i.type:<5}  ARIS {i.report}{where}  {i.source}  {i.note}".rstrip())
        if i.image:
            typer.echo(f"         image: {i.image}")


@leads_app.command("run")
def leads_run(
    project: Annotated[Path, typer.Argument(help="Project folder with mto_tenure and aris_index.")],
    as_of: Annotated[
        str | None,
        typer.Option(help="Score as of this date (YYYY-MM-DD) to reproduce a run. Default: today."),
    ] = None,
) -> None:
    """Score tenures and write outputs/leads_ranked.csv and the 'leads' layer in data.gpkg.

    Needs the mto_tenure and aris_index layers: run 'gossan harvest --source
    mto_tenure,aris_index' first, or load them with 'gossan ingest'. Weights live in
    project.yaml under parameters: leads: (see gossan.leads.scoring). Leads use ARIS
    data, which is Access Only: keep the outputs internal.
    """
    proj = _open_project(project)
    try:
        when = date.fromisoformat(as_of) if as_of else None
    except ValueError as e:
        raise typer.BadParameter("use YYYY-MM-DD", param_hint="--as-of") from e
    try:
        result = run_leads(proj, when)
    except MissingLayerError as e:
        typer.secho(f"Error: {e}", fg="red", err=True)
        raise typer.Exit(1) from e
    leads = result.leads
    typer.echo(f"{len(leads)} lead(s) scored; wrote {result.csv_path} and the 'leads' layer")
    for r in leads.head(5).itertuples():
        name = r.claim_name if isinstance(r.claim_name, str) else "(unnamed)"
        typer.echo(f"  {r.rank:>3}  {r.score:>6}  {r.lead_id}  {name}  ({r.aris_count} ARIS)")


@leads_app.command("pack")
def leads_pack(
    project: Annotated[Path, typer.Argument(help="Project folder where 'gossan leads run' was run.")],
    lead_id: Annotated[str, typer.Argument(help="Tenure number of the lead (lead_id in the CSV).")],
) -> None:
    """Write a summary document of a lead and its ARIS reports to outputs/.

    A skeleton: it lists the property and the reports available, and does not read the
    PDFs (digitization comes later). It is an internal draft because ARIS is Access Only.
    """
    proj = _open_project(project)
    try:
        path = build_pack(proj, lead_id)
    except (MissingLayerError, LeadNotFoundError) as e:
        typer.secho(f"Error: {e}", fg="red", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"Wrote {path}")
