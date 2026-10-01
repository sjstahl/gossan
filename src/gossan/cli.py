"""Gossan command-line entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from gossan.core.aoi import read_aoi, utm_zone_for, write_aoi
from gossan.core.config import AOI_FILE, ProjectConfig
from gossan.core.ingest import IngestError
from gossan.core.ingest import ingest as ingest_file
from gossan.core.project import Project
from gossan.core.provenance import ProvenanceLog
from gossan.harvest import CONNECTORS
from gossan.harvest.bc.aris_pdf import ArisPdfDownloader
from gossan.harvest.national.nrcan_geophysics import ManualDownloadRequired
from gossan.report.builder import build_report

app = typer.Typer(
    help="Gossan: turn a claim polygon into client-ready deliverables.",
    no_args_is_help=True,
)


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
