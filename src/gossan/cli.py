"""Gossan command-line entry point."""

from __future__ import annotations

import getpass
from datetime import date
from pathlib import Path
from typing import Annotated

import pymupdf
import typer
from typer.core import TyperGroup

from gossan.core.aoi import read_aoi, utm_zone_for, write_aoi
from gossan.core.config import AOI_FILE, ProjectConfig
from gossan.core.ingest import IngestError
from gossan.core.ingest import ingest as ingest_file
from gossan.core.project import Project
from gossan.core.provenance import ProvenanceLog
from gossan.core.raster import raster_path
from gossan.digitize.catalogue import build_catalogue
from gossan.digitize.claude_vision import ClaudeVisionError, model_name
from gossan.digitize.claude_vision import (
    extract_scanned_tables as extract_scanned_tables_for,
)
from gossan.digitize.db import DrillholeDbError, load_approved_table
from gossan.digitize.georef import georeference_map_page, map_output_name
from gossan.digitize.inputs import select_pdfs
from gossan.digitize.ocr import OcrUnavailableError, ocr_pages
from gossan.digitize.pages import parse_pages
from gossan.digitize.qgis_fallback import PointsError, parse_points
from gossan.digitize.review import (
    ReviewError,
    Status,
    get_item,
    init_queue,
    read_queue,
    set_status,
    summarize,
)
from gossan.digitize.tables import extract_digital_tables
from gossan.digitize.text import extract_text, page_ranges
from gossan.harvest import CONNECTORS
from gossan.harvest.bc.aris_pdf import ArisPdfDownloader
from gossan.harvest.national.nrcan_geophysics import ManualDownloadRequired
from gossan.harvest.satellite.stac import SatelliteAuthError
from gossan.leads.generator import run_leads
from gossan.leads.pack import LeadNotFoundError, build_pack
from gossan.leads.scoring import MissingLayerError
from gossan.report.builder import build_report
from gossan.report.pdf import ReportError, build_pdf
from gossan.target.run import TargetError, run_target

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
    limit_scenes: Annotated[
        int,
        typer.Option(
            min=0,
            help="For the satellite sources (emit, enmap, aster): download the assets of up to "
            "this many scenes (best AOI coverage, least cloud first) into raw/satellite/. 0 "
            "(default) only searches and writes the scene footprints; a scene is 0.5 to several GB.",
        ),
    ] = 0,
) -> None:
    """Download public data for the project AOI into data.gpkg and rasters/.

    Originals are kept in raw/ and every retrieval is logged to provenance.jsonl.
    A source that fails does not stop the others; the command exits 1 if any failed.

    Manual fallback: if a source is unavailable, download the data yourself and load
    it with 'gossan ingest' (vector files and .tif grids). nrcan_geophysics needs a
    configured WCS (see its module docs) and otherwise prints manual instructions.

    Satellite scenes (--source emit,enmap,aster) are searched without credentials; downloading
    (--limit-scenes N) needs EARTHDATA_TOKEN (emit, aster) or EOC_USERNAME and EOC_PASSWORD
    (enmap). They are not part of a plain harvest. Search settings: satellite: in project.yaml.
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
        if hasattr(connector, "limit_scenes"):
            connector.limit_scenes = limit_scenes
        try:
            count = connector.run()
        except (ManualDownloadRequired, SatelliteAuthError) as e:
            typer.secho(f"{n}: {e}", fg="yellow", err=True)
            failed = True
        except OSError as e:  # network failures (URLError is an OSError)
            typer.secho(f"Error: {n}: {e}", fg="red", err=True)
            typer.echo("Fallback: download the data manually and use 'gossan ingest'.", err=True)
            failed = True
        else:
            typer.echo(f"{n}: {count} {connector.output_description}")
            if hasattr(connector, "report"):
                typer.echo(f"  {connector.report.text(n)}")

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
    fmt: Annotated[
        str,
        typer.Option(
            "--format",
            help="pdf (default): the pitch pack, outputs/report.pdf. docx: the short Word data-sources "
            "report, outputs/report.docx. both: write both.",
        ),
    ] = "pdf",
    lead: Annotated[
        str | None,
        typer.Option(help="Feature this lead (tenure number from 'gossan leads run') in the PDF."),
    ] = None,
) -> None:
    """Build the pitch pack PDF: a draft executive summary of everything in the project.

    Sections (each appears only with what exists, and says which command makes the rest): property and
    tenure, lead scoring, prospectivity targeting (map with targets, uncertainty, importance and
    spatial cross-validation tables), before/after digitized samples a reviewer has approved,
    and data sources. Every page is marked INTERNAL DRAFT because it contains ARIS data, which is
    Access Only: do not send it outside the company without the Province's written permission.
    """
    proj = _open_project(project)
    if fmt not in {"pdf", "docx", "both"}:
        raise typer.BadParameter("use pdf, docx or both", param_hint="--format")
    if fmt in {"docx", "both"}:
        typer.echo(f"Wrote {build_report(proj)}")
    if fmt in {"pdf", "both"}:
        try:
            typer.echo(f"Wrote {build_pdf(proj, lead)}")
        except ReportError as e:
            typer.secho(f"Error: {e}", fg="red", err=True)
            raise typer.Exit(1) from e


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
            help="Send the --target-pages page images to Claude (via OpenRouter: costs money, "
            "needs OPENROUTER_API_KEY) to read drillhole/assay tables. Results go to "
            "digitize/tables/scanned/ and the review queue as PENDING.",
        ),
    ] = False,
    ocr: Annotated[
        bool, typer.Option("--ocr/--no-ocr", help="OCR pages with no embedded text (EasyOCR).")
    ] = True,
    georef: Annotated[
        bool,
        typer.Option(
            "--georef/--no-georef",
            help="Try to georeference pages flagged as maps (needs a printed grid and, to read "
            "its labels, EasyOCR). Fit limit: georef.rms_threshold_m in project.yaml.",
        ),
    ] = True,
) -> None:
    """Catalogue the downloaded ARIS PDFs, extract text and tables.

    Always: the 'catalogue' layer in data.gpkg, digitize/text/<aris_id>.txt, and digital
    tables (pdfplumber) in digitize/tables/digital/. Pages with no embedded text are OCR'd
    with EasyOCR (an optional install, the "ocr" extra; skipped with a warning if missing)
    and appended to the text file as [OCR EXTRACTED]. --target-pages limits the page-level
    steps to the pages you name. Scanned drillhole/assay tables are read by Claude only with
    --extract-scanned-tables, only on --target-pages, and never more than 25 pages per run;
    every table it extracts is queued for human review ('gossan review'). Pages flagged as
    maps are georeferenced automatically when possible (OpenCV finds the grid, EasyOCR reads
    its labels, GDAL fits and warps; the GeoTIFF goes to rasters/ and is queued for review).
    Manual fallback when that fails: georeference the page image in QGIS and load the saved
    points with 'gossan ingest-points'. Get PDFs first with 'gossan harvest --limit-pdfs N'.
    Unreadable PDFs are reported and skipped.
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

        map_pages = [p for p in (parse_pages(r.map_pages) if r.map_pages else []) if p in scope]
        if georef and map_pages:
            _georeference_maps(proj, number, pdf, map_pages)

    for number, reason in failed.items():
        typer.secho(f"Error: ARIS {number}: {reason}", fg="red", err=True)
    typer.echo(f"{n_done} report(s) processed, {len(failed)} failed; catalogue updated")
    if failed:
        raise typer.Exit(1)


def _georeference_maps(proj: Project, number: int, pdf: Path, pages: list[int]) -> None:
    """Run the automatic georeferencing on map pages, reporting each outcome."""
    done = skipped = 0
    for page in pages:
        if raster_path(proj, map_output_name(number, page)).exists():
            skipped += 1  # keep an existing (possibly manual, possibly reviewed) result
            continue
        res = georeference_map_page(proj, number, pdf, page)
        if res.ok:
            done += 1
            typer.echo(
                f"  map p.{page}: georeferenced, {res.n_gcps} GCPs, RMS {res.rms_m:.1f} m "
                f"-> {res.output.relative_to(proj.root)} (queued for review)"
            )
        else:
            typer.secho(f"  map p.{page}: not georeferenced: {res.reason}", fg="yellow", err=True)
            where = f"digitize/maps/{number:05d}_p{page}.png"
            hint = f"georeference {where} in QGIS, save the points, then: gossan ingest-points {proj.root} {number} {page} <file.points>"
            if res.candidate_points:
                hint = f"correct {res.candidate_points.relative_to(proj.root)} in QGIS (Layer > Georeferencer, open {where}), save, then: gossan ingest-points {proj.root} {number} {page} <file.points>"
            typer.secho(f"    manual fallback: {hint}", fg="yellow", err=True)
    typer.echo(f"  georeferencing: {done} done, {skipped} already done, {len(pages) - done - skipped} need the manual route")


@app.command("ingest-points")
def ingest_points_cmd(
    project: Annotated[Path, typer.Argument(help="Project folder created by 'gossan new'.")],
    report_id: Annotated[int, typer.Argument(help="ARIS number of the report.")],
    page_num: Annotated[int, typer.Argument(help="Page number of the map in the PDF (1-based).")],
    points_file: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="QGIS Georeferencer .points file."),
    ],
    image: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="The image the points were placed on, if not digitize/maps/<id>_p<N>.png.",
        ),
    ] = None,
    rms_threshold: Annotated[
        float | None,
        typer.Option(
            min=0.001,
            help="Accept a fit up to this RMS (metres) for this map, instead of "
            "georef.rms_threshold_m; recorded in provenance and the review note.",
        ),
    ] = None,
) -> None:
    """Georeference a map page from a QGIS .points file (manual fallback).

    In QGIS (Layer > Georeferencer) open the page image digitize/maps/<id>_p<N>.png (made by
    'gossan digitize', or by this command if absent), place at least 3 points (4 or more so the
    fit can be checked), and use 'Save GCP Points'. The pixel positions in the file refer to
    that image, so do not re-render or crop it. Map coordinates are read in the file's #CRS
    (else the project's UTM zone in georef.datum). The fit is warped to a COG in rasters/,
    checked against the RMS threshold, and queued for review.
    """
    proj = _open_project(project)
    try:
        pdfs = select_pdfs(proj, report_id)
        points = parse_points(points_file)
    except (FileNotFoundError, PointsError) as e:
        typer.secho(f"Error: {e}", fg="red", err=True)
        raise typer.Exit(1) from e
    pdf = pdfs[report_id]
    with pymupdf.open(pdf) as doc:
        n_pages = doc.page_count
    if not 1 <= page_num <= n_pages:
        typer.secho(f"Error: page {page_num} is not in the PDF ({n_pages} pages)", fg="red", err=True)
        raise typer.Exit(1)

    res = georeference_map_page(
        proj, report_id, pdf, page_num, points=points, image=image, rms_threshold_m=rms_threshold
    )
    for note in res.notes:
        typer.echo(f"note: {note}")
    if not res.ok:
        typer.secho(f"Not georeferenced: {res.reason}", fg="red", err=True)
        raise typer.Exit(1)
    typer.echo(
        f"Georeferenced ARIS {report_id} p.{page_num}: {res.n_gcps} GCPs, RMS {res.rms_m:.1f} m "
        f"-> {res.output.relative_to(proj.root)} (queued for review)"
    )


class _ReviewGroup(TyperGroup):
    """`gossan review <project>` keeps working as `gossan review list <project>`."""

    def resolve_command(self, ctx, args):
        if args and args[0] not in self.commands and not args[0].startswith("-"):
            args = ["list", *args]
        return super().resolve_command(ctx, args)


review_app = typer.Typer(
    cls=_ReviewGroup,
    help="Review what the pipeline extracted: list, approve or reject queue items.",
    no_args_is_help=True,
)
app.add_typer(review_app, name="review")

_Project = Annotated[Path, typer.Argument(help="Project folder created by 'gossan new'.")]
_ItemId = Annotated[str, typer.Argument(help="Review item id, e.g. R0003 (see 'gossan review list').")]
_By = Annotated[
    str, typer.Option("--by", help="Who is deciding, for the record. Default: your login name.")
]
_Note = Annotated[str, typer.Option("--note", help="Why (kept in the queue and provenance).")]


@review_app.command("list")
def review_list(project: _Project) -> None:
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


@review_app.command("approve")
def review_approve(
    project: _Project, item_id: _ItemId, by: _By = "", note: _Note = ""
) -> None:
    """Approve a PENDING item; an approved table is loaded into the drillhole tables.

    Check the item first against its page image (see 'gossan review list'); for a table,
    correct the CSV before approving: it is read at this moment. The CSV must have hole_id,
    from_m and to_m columns, with depths numeric (or both empty for a surface sample), and no
    '?' cells left. If validation fails nothing is loaded and the item stays PENDING. Maps and
    text items are only marked. The decision is final and logged to provenance.
    """
    proj = _open_project(project)
    by = by or getpass.getuser()
    details: dict[str, object] = {}
    try:
        item = get_item(proj, item_id)
        if item.status != "PENDING":
            raise ReviewError(f"{item_id} is already {item.status}; decisions are final")
        message = f"Approved {item_id} ({item.type}, ARIS {item.report}" + (f" p.{item.page})" if item.page else ")")
        if item.type == "table":
            loaded = load_approved_table(proj, item, by)
            details = {
                "loaded_into": loaded.table,
                "records": loaded.n_records,
                "csv_sha256": loaded.csv_sha256,
            }
            message += (
                f": {loaded.n_rows} row(s) -> {loaded.n_records} record(s) in the {loaded.table} table"
            )
            if loaded.n_no_interval:
                message += f" ({loaded.n_no_interval} row(s) have no depth interval: surface samples)"
        set_status(proj, item_id, Status.APPROVED, by=by, note=note, details=details)
    except (ReviewError, DrillholeDbError) as e:
        typer.secho(f"Error: {e}", fg="red", err=True)
        raise typer.Exit(1) from e
    typer.echo(message)


@review_app.command("reject")
def review_reject(
    project: _Project, item_id: _ItemId, by: _By = "", note: _Note = ""
) -> None:
    """Reject a PENDING item; nothing is loaded. The decision is final and logged."""
    proj = _open_project(project)
    try:
        item = set_status(proj, item_id, Status.REJECTED, by=by or getpass.getuser(), note=note)
    except ReviewError as e:
        typer.secho(f"Error: {e}", fg="red", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"Rejected {item.item_id} ({item.type}, ARIS {item.report})")


@app.command("target")
def target_cmd(
    project: Annotated[Path, typer.Argument(help="Project folder with harvested layers.")],
    resolution: Annotated[
        float | None,
        typer.Option(
            "--resolution", min=0.001,
            help="Grid cell size in metres (project CRS). Default: target.resolution_m in project.yaml (100).",
        ),
    ] = None,
    n_bags: Annotated[int | None, typer.Option(min=2, help="PU-bagging rounds. Default: target.n_bags (25).")] = None,
    folds: Annotated[int | None, typer.Option(min=2, help="Spatial CV folds. Default: target.folds (5).")] = None,
    seed: Annotated[int | None, typer.Option(help="Random seed. Default: target.seed (42).")] = None,
    cv: Annotated[
        bool, typer.Option("--cv/--no-cv", help="Run spatial cross-validation (slower; reports how well it generalises).")
    ] = True,
) -> None:
    """Rank the AOI by prospectivity: a PU-bagging random forest on harvested layers.

    Known MINFILE occurrences in the AOI are the positives; every other cell is unlabeled, not
    barren. Features are built from whatever the project has (tenure, ARIS report density, RGS
    geochemistry, regional geophysics rasters, distance to occurrences); missing layers are
    skipped and listed. Needs at least target.min_positives (10) occurrences inside the AOI, so
    use a regional AOI. Writes target/prospectivity.tif (a relative score, not a probability of
    a deposit), target/uncertainty.tif (spread across bags), target/targets.gpkg (the top
    percentile as polygons), feature_stack.tif, feature_importance.csv, cv_report.json and
    grid.json, and logs the run to provenance. The ARIS index is Access Only: keep outputs internal.
    """
    proj = _open_project(project)
    overrides = {
        k: v
        for k, v in {"resolution_m": resolution, "n_bags": n_bags, "folds": folds, "seed": seed}.items()
        if v is not None
    }
    cfg = proj.config.target.model_copy(update=overrides)
    try:
        r = run_target(proj, cfg, cross_validate=cv)
    except TargetError as e:
        typer.secho(f"Error: {e}", fg="red", err=True)
        raise typer.Exit(1) from e

    g = r.grid
    d = g.describe()
    typer.echo(
        f"Grid: {g.width} x {g.height} cells ({d['n_cells_bbox']:,}) at {g.resolution:g} m in {g.crs}; "
        f"{g.n_cells:,} cells ({d['area_km2']} km2) inside the AOI"
    )
    typer.echo(f"Known occurrences used as positives: {len(r.stack.positives)}")
    typer.echo(f"Features ({len(r.stack.model_names)}): {', '.join(r.stack.model_names)}")
    for name, reason in r.stack.skipped.items():
        typer.secho(f"  skipped {name}: {reason}", fg="yellow")
    if r.cv is not None:
        if r.cv["folds"]:
            typer.echo(
                f"Spatial CV ({r.cv['n_folds_scored']} fold(s)): AUC {r.cv['auc_mean']:.2f} +/- {r.cv['auc_std']:.2f}, "
                f"top-5% capture {r.cv['capture_top5_mean']:.0%} +/- {r.cv['capture_top5_std']:.0%} (random: 5%)"
            )
        for note in r.cv["skipped"]:
            typer.secho(f"  CV: {note}", fg="yellow")
    typer.echo(f"Targets: {r.n_targets} polygon(s) at or above the {r.cfg.top_percentile:g}th percentile (score {r.threshold:.3f})")
    for key, rel in r.outputs.items():
        if key == "targets" and not r.n_targets:
            continue
        typer.echo(f"  {key}: {rel}")


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
