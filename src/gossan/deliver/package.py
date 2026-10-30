"""Build the client package: ``outputs/deliverable/`` (or ``--out``).

::

    <name>.qgs            QGIS project (open it, everything is laid out and styled)
    map.html              interactive map; one file, works offline, no basemap
    data/deliverable.gpkg every packaged vector layer, in the project CRS
    rasters/              the packaged score and index rasters (Cloud Optimized GeoTIFF where made so)
    <name>_analysis_report.pdf / .docx   when the report carries no restricted data
    README.txt            what is in it, how to open it, the licences, what was left out and why
    manifest.json         the same, for machines, with a SHA-256 for every file

What goes in is decided by ``gossan.deliver.items``: each item by its licence on record, the rest
withheld and listed. Nothing is ever deleted from the project; the package folder is rebuilt each time
(and refuses to overwrite a folder it did not make).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio

from gossan import __version__
from gossan.analyze.targets import build_decisions
from gossan.core.project import Project
from gossan.core.revision import code_revision
from gossan.deliver import webmap
from gossan.deliver.items import Item, Plan, _layers, output_licence, plan, restricts
from gossan.deliver.qgis_project import RasterLayer, VectorLayer, write_project
from gossan.deliver.styles import STYLES
from gossan.report.analysis_render import build_analysis_report

MANIFEST = "manifest.json"
GPKG = "data/deliverable.gpkg"
GEOMETRY = {"Polygon": "Polygon", "MultiPolygon": "Polygon", "LineString": "Line", "MultiLineString": "Line", "Point": "Point", "MultiPoint": "Point"}
#: Top of the layer list first, as QGIS draws it.
QGIS_ORDER = ["aoi", "minfile", "faults", "targets", "knowledge_targets", "target_decisions", "mto_tenure"]
VISIBLE_IN_QGIS = {"aoi", "minfile", "faults", "targets", "knowledge_targets", "mto_tenure", "prospectivity"}
DISCLAIMER = (
    "Scores rank ground by how much it resembles known occurrences in the supplied data; they are relative, not the probability of a deposit. "
    "Confirm land status on Mineral Titles Online before acting."
)


class DeliverError(Exception):
    """The package cannot be built; the message says why."""


@dataclass
class DeliverResult:
    folder: Path
    included: list[Item] = field(default_factory=list)
    withheld: list[Item] = field(default_factory=list)
    files: list[Path] = field(default_factory=list)
    zip_path: Path | None = None
    internal: bool = False


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _prepare(folder: Path) -> None:
    if folder.exists() and any(folder.iterdir()):
        if not (folder / MANIFEST).is_file():
            raise DeliverError(f"{folder} exists and is not a Gossan package (no {MANIFEST}); choose another --out or empty it")
        shutil.rmtree(folder)
    for sub in ("data", "rasters"):
        (folder / sub).mkdir(parents=True, exist_ok=True)


def _raster_range(path: Path) -> tuple[float, float]:
    with rasterio.open(path) as ds:
        data = ds.read(1, masked=True).astype(float).filled(np.nan)
    finite = data[np.isfinite(data)]
    if not finite.size:
        return 0.0, 1.0
    lo, hi = (float(v) for v in np.nanpercentile(finite, [2, 98]))
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


def _licence_lines(items: list[Item]) -> list[str]:
    seen: dict[str, list[str]] = {}
    for item in items:
        seen.setdefault(item.licence or "licence not recorded", []).append(item.title)
    return [f"{lic}: {', '.join(titles)}" for lic, titles in seen.items()]


def _readme(project: Project, result: DeliverResult, project_file: str, notice: str) -> str:
    cfg = project.config
    lines = [f"{cfg.name}" + (f" - prepared for {cfg.client}" if cfg.client else ""), f"Package built {datetime.now(UTC):%Y-%m-%d} with Gossan {__version__}", ""]
    if result.internal:
        lines += ["INTERNAL: this package was built with --include-internal and may contain Access Only or otherwise restricted data. Do not send it outside the company.", ""]
    lines += [
        "WHAT IS HERE",
        f"  {project_file}      QGIS project: open it in QGIS 3.x; layers are styled and the prospectivity score is shown.",
        "  map.html            interactive map; open in any browser. One file, works offline, no basemap. Click a feature for its attributes.",
        f"  {GPKG}  every vector layer (QGIS, ArcGIS and others read GeoPackage), coordinates in {cfg.crs}.",
        "  rasters/            score and index rasters (GeoTIFF).",
    ]
    if any(i.kind == "report" for i in result.included):
        lines.append("  *_analysis_report.*  the analysis report.")
    lines += ["", "LAYERS"]
    for i in result.included:
        lines.append(f"  - {i.title} ({i.kind}, {i.group})")
    lines += ["", "LICENCES AND SOURCES", *(f"  {line}" for line in _licence_lines(result.included))]
    if result.withheld:
        lines += ["", "LEFT OUT"]
        lines += [f"  - {i.title}: {i.withheld}" for i in result.withheld]
    lines += ["", "READ THIS BEFORE USING THE RESULTS", f"  {DISCLAIMER}", f"  {notice}" if notice else ""]
    return "\n".join(lines).rstrip() + "\n"


def _rebuild_open_outputs(project: Project) -> None:
    """Rebuild the decision table and any existing analysis report from open-licensed layers only."""
    root = project.root
    if "targets_top" in _layers(root / "target" / "targets.gpkg") and (
        restricts(output_licence(project.provenance.read(), "analysis/analysis.gpkg:target_decisions")) is not None
        or not (root / "analysis" / "analysis.gpkg").is_file()
    ):
        build_decisions(project, open_only=True)
    has_pdf = (project.outputs_dir / "analysis_report.pdf").is_file()
    has_docx = (project.outputs_dir / "analysis_report.docx").is_file()
    if has_pdf or has_docx:
        records = project.provenance.read()
        pdf_restricted = has_pdf and restricts(output_licence(records, "outputs/analysis_report.pdf")) is not None
        docx_restricted = has_docx and restricts(output_licence(records, "outputs/analysis_report.docx")) is not None
        if pdf_restricted or docx_restricted:
            fmt = "both" if (has_pdf and has_docx) else ("pdf" if has_pdf else "docx")
            build_analysis_report(project, fmt, open_only=True)


def build_deliverable(
    project: Project, out: Path | None = None, *, include_internal: bool = False, make_zip: bool = False, open_only: bool = False
) -> DeliverResult:
    """Build the package. Raises ``DeliverError`` when there is nothing to deliver or the folder is not ours to replace."""
    if open_only and include_internal:
        raise DeliverError("cannot combine --open-only and --include-internal")
    if open_only:
        _rebuild_open_outputs(project)
    the_plan: Plan = plan(project, include_internal=include_internal)
    results = [i for i in the_plan.included if i.group == "results"]
    if not results:
        reasons = "; ".join(f"{i.title}: {i.withheld}" for i in the_plan.withheld) or "no model output yet"
        raise DeliverError(f"nothing to deliver: no results can be packaged ({reasons}). Run 'gossan target' first")
    folder = Path(out) if out else project.outputs_dir / "deliverable"
    _prepare(folder)
    cfg = project.config
    crs = cfg.crs
    internal = include_internal and any(i.withheld is None and (i.licence is None or "access only" in (i.licence or "").lower() or "internal use only" in (i.licence or "").lower()) for i in the_plan.items)
    result = DeliverResult(folder, the_plan.included, the_plan.withheld, internal=internal)

    # --- vectors -> one GeoPackage ---------------------------------------------------------------------------------
    aoi = project.read_aoi()
    gpkg = folder / GPKG
    aoi.to_file(gpkg, layer="aoi", driver="GPKG", engine="pyogrio")
    vectors: list[tuple[Item, gpd.GeoDataFrame]] = [(Item("aoi", "vector", "Area of interest", project.aoi_path, identifier="aoi"), aoi)]
    for item in the_plan.included:
        if item.kind != "vector":
            continue
        gdf = gpd.read_file(item.source, layer=item.layer, engine="pyogrio").to_crs(crs)
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
        if gdf.empty:
            item.withheld = "no features inside the area"
            result.included.remove(item)
            result.withheld.append(item)
            continue
        gdf.to_file(gpkg, layer=item.key, driver="GPKG", engine="pyogrio")
        vectors.append((item, gdf))

    # --- rasters and reports ---------------------------------------------------------------------------------------
    raster_items = [i for i in result.included if i.kind == "raster"]
    for item in raster_items:
        shutil.copy2(item.source, folder / "rasters" / f"{item.key}.tif")
    for item in result.included:
        if item.kind == "report":
            shutil.copy2(item.source, folder / f"{cfg.name}_{item.source.name}")

    # --- the QGIS project --------------------------------------------------------------------------------------------
    qgis_layers: list[VectorLayer | RasterLayer] = []
    by_key = {item.key: (item, gdf) for item, gdf in vectors}
    ordered = [k for k in QGIS_ORDER if k in by_key] + [k for k in by_key if k not in QGIS_ORDER and k not in STYLES]
    for key in ordered:
        item, gdf = by_key[key]
        qgis_layers.append(VectorLayer(key, item.title, GEOMETRY.get(gdf.geom_type.iloc[0], "Polygon"), GPKG, key, key in VISIBLE_IN_QGIS))
    for item in raster_items:
        vmin, vmax = _raster_range(folder / "rasters" / f"{item.key}.tif")
        qgis_layers.append(RasterLayer(item.key, item.title, f"rasters/{item.key}.tif", vmin, vmax, visible=item.key in VISIBLE_IN_QGIS))
    for key in ("bedrock_geology",):
        if key in by_key:
            item, gdf = by_key[key]
            qgis_layers.append(VectorLayer(key, item.title, GEOMETRY.get(gdf.geom_type.iloc[0], "Polygon"), GPKG, key, False))
    minx, miny, maxx, maxy = aoi.total_bounds
    pad = 0.05 * max(maxx - minx, maxy - miny)
    notice = "; ".join(_licence_lines(result.included))
    project_file = f"{cfg.name}.qgs"
    write_project(folder / project_file, cfg.name, crs, (minx - pad, miny - pad, maxx + pad, maxy + pad), qgis_layers, notice=notice + " " + DISCLAIMER)

    # --- the web map ---------------------------------------------------------------------------------------------------
    web_vectors = [(i, g) for i, g in vectors if i.key == "aoi" or (i.web and i.key != "target_decisions")]
    decisions = by_key.get("target_decisions")
    if decisions is not None:  # one target layer on the map, with the ground status, access and history in its popup
        extra = decisions[1].drop(columns="geometry")
        web_vectors = [(i, g.merge(extra.drop(columns=[c for c in extra if c in g.columns and c != "target_id"]), on="target_id", how="left") if i.key == "targets" and "target_id" in g and "target_id" in extra else g) for i, g in web_vectors]
    web_rasters = [i for i in raster_items if i.web]
    footer = f"{DISCLAIMER}  Sources and licences: {'; '.join(sorted({i.licence or 'licence not recorded' for i in result.included}))}."
    data = webmap.build_map_data(project, web_vectors, web_rasters, footer, internal)
    webmap.write_map(folder / "map.html", data)

    (folder / "README.txt").write_text(_readme(project, result, project_file, ""), encoding="utf-8")
    files = sorted(p for p in folder.rglob("*") if p.is_file() and p.name != MANIFEST)
    manifest = {
        "project": cfg.name,
        "client": cfg.client,
        "crs": crs,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "gossan": {"version": __version__, "revision": code_revision()},
        "internal": internal,
        "included": [{"key": i.key, "title": i.title, "kind": i.kind, "group": i.group, "licence": i.licence} for i in result.included],
        "withheld": [{"key": i.key, "title": i.title, "reason": i.withheld} for i in result.withheld],
        "files": [{"path": p.relative_to(folder).as_posix(), "bytes": p.stat().st_size, "sha256": _sha256(p)} for p in files],
    }
    (folder / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    result.files = [*files, folder / MANIFEST]
    if make_zip:
        result.zip_path = folder.with_suffix(".zip")
        with zipfile.ZipFile(result.zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for p in result.files:
                z.write(p, f"{folder.name}/{p.relative_to(folder).as_posix()}")
    project.provenance.log(
        dataset="deliverable",
        source="gossan",
        identifier=folder.relative_to(project.root).as_posix() if folder.is_relative_to(project.root) else str(folder),
        url=None,
        license="; ".join(_licence_lines(result.included)) or "N/A",
        operation="build_deliverable",
        parameters={
            "included": [i.key for i in result.included], "withheld": {i.key: i.withheld for i in result.withheld},
            "include_internal": include_internal, "internal": internal, "files": len(result.files),
        },
    )  # fmt: skip
    return result
