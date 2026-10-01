"""Gossan command-line entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from gossan.core.aoi import read_aoi, utm_zone_for, write_aoi
from gossan.core.config import AOI_FILE, ProjectConfig
from gossan.core.ingest import IngestError, ingest_vector
from gossan.core.project import Project
from gossan.core.provenance import ProvenanceLog
from gossan.harvest import CONNECTORS
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
            f"Available: {', '.join(CONNECTORS)}. Default: all."
        ),
    ] = None,
) -> None:
    """Download public data for the project AOI into data.gpkg.

    Originals are kept in raw/ and every retrieval is logged to provenance.jsonl.

    Manual fallback: if a source is unavailable, download the data yourself and load
    it with 'gossan ingest'.
    """
    proj = _open_project(project)
    names = [s.strip() for s in source.split(",") if s.strip()] if source else list(CONNECTORS)
    unknown = [n for n in names if n not in CONNECTORS]
    if unknown:
        raise typer.BadParameter(f"unknown source(s): {', '.join(unknown)}", param_hint="--source")
    for n in names:
        try:
            count = CONNECTORS[n](proj).run()
        except OSError as e:  # network failures (URLError is an OSError)
            typer.secho(f"Error: {n}: {e}", fg="red", err=True)
            typer.echo("Fallback: download the data manually and use 'gossan ingest'.", err=True)
            raise typer.Exit(1) from e
        typer.echo(f"{n}: {count} feature(s) written to data.gpkg")


@app.command()
def ingest(
    project: Annotated[Path, typer.Argument(help="Project folder created by 'gossan new'.")],
    file_path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Local vector file.")
    ],
    layer: Annotated[str, typer.Option("--layer", help="Layer name to write in data.gpkg.")],
    source: Annotated[
        str, typer.Option(help="Who supplied the data, for provenance and the report.")
    ] = "user-supplied",
    license: Annotated[
        str, typer.Option(help="Licence or terms of the data, for provenance and the report.")
    ] = "unspecified (user-supplied)",
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Replace the layer if it already exists.")
    ] = False,
) -> None:
    """Load a local vector file (shp, geojson, gpkg, ...) into data.gpkg.

    The file is reprojected to the project CRS and clipped to the AOI. A copy of the
    original goes to raw/ingest/ and the ingest is logged to provenance.jsonl. This is
    the manual fallback for any harvest source. The file must have a CRS defined.
    """
    proj = _open_project(project)
    try:
        count = ingest_vector(
            proj, file_path, layer, source=source, license=license, overwrite=overwrite
        )
    except IngestError as e:
        typer.secho(f"Error: {e}", fg="red", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"{layer}: {count} feature(s) written to data.gpkg")


@app.command()
def report(
    project: Annotated[Path, typer.Argument(help="Project folder created by 'gossan new'.")],
) -> None:
    """Build outputs/map.png and outputs/report.docx (a draft for geologist review)."""
    proj = _open_project(project)
    path = build_report(proj)
    typer.echo(f"Wrote {path}")
