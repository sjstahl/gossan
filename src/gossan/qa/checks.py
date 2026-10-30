"""Quality checks on a project before anything goes to a client.

Each check returns findings of one of three severities:

* ``error``: do not hand this over until it is fixed (invalid claim geometry, data that does not lie
  in the area of interest, a layer without a CRS, a delivery package whose files changed or that
  carries restricted data);
* ``warn``: look at it (overlapping or sliver claims, gaps between claims, empty layers, data with no
  provenance record);
* ``info``: for the record (CRS stored differently from the project's, mostly-empty attributes,
  expired tenures).

What is checked:

* the layer inventory: features, geometry type, stored CRS, empty and invalid geometries, duplicates,
  empty attribute columns, whether any feature lies in the AOI, whether provenance names the layer;
* the claim block (the AOI polygons): validity, overlaps, slivers, and gaps between claims (closed
  holes and thin open gaps), and the same for the harvested tenure layer;
* rasters: CRS, valid cells, overlap with the AOI;
* a delivery package, if one has been built: every file present and unchanged since its SHA-256 was
  recorded, every item with a licence on record, and no restricted item in a package not marked internal.

Thresholds live under ``parameters: qa:`` in ``project.yaml`` (see ``QaConfig``). BLM claims are
quarter-section geocodes and overlap by design, so on a US project overlaps in the tenure layer are
reported as information. Not checked: a generic "unclosed polyline" test (faults and other lines are
open by nature) and zero-length lines are reported as degenerate instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from pydantic import BaseModel, ConfigDict, Field
from pyproj import CRS
from pyproj.exceptions import CRSError
from rasterio.features import geometry_mask
from shapely import STRtree
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.validation import explain_validity

from gossan.core.project import Project
from gossan.deliver.items import RESTRICTING
from gossan.report.licensing import decide_marking

Severity = Literal["error", "warn", "info"]
SEVERITIES: tuple[Severity, ...] = ("error", "warn", "info")
MANIFEST = "manifest.json"
#: Layers Gossan writes itself; a defect in one of these is its own to fix, so invalid geometry is an error.
GOSSAN_LAYERS = {"aoi", "leads"}
MAX_IDS = 8


class QaConfig(BaseModel):
    """Thresholds; override under ``parameters: qa:`` in ``project.yaml``."""

    model_config = ConfigDict(extra="forbid")

    #: Overlap between two claims smaller than this (m2) is edge noise and ignored.
    overlap_tolerance_m2: float = Field(default=100, ge=0)
    #: A claim smaller than this (m2), or narrower than ``min_width_m``, is a sliver.
    min_area_m2: float = Field(default=1000, ge=0)
    min_width_m: float = Field(default=5, ge=0)
    #: A gap between claims of at least ``gap_min_m2`` and below ``gap_max_m2`` is probably a digitising error.
    gap_min_m2: float = Field(default=25, ge=0)
    gap_max_m2: float = Field(default=10_000, ge=0)
    #: Thin gaps up to this width (m) are found even where they are open at one end.
    gap_close_m: float = Field(default=5, gt=0)
    #: An attribute with at least this share of empty values is reported.
    empty_share: float = Field(default=0.5, ge=0, le=1)
    #: A raster covering less than this share of the AOI is a warning.
    min_raster_coverage: float = Field(default=0.5, ge=0, le=1)

    @classmethod
    def from_project(cls, project: Project) -> QaConfig:
        return cls.model_validate(project.config.parameters.get("qa", {}))


@dataclass
class Finding:
    severity: Severity
    check: str
    subject: str
    message: str
    ids: list[str] = field(default_factory=list)


@dataclass
class LayerRow:
    name: str
    kind: str  # "vector" or "raster"
    geometry: str
    features: int
    crs: str
    detail: str = ""


@dataclass
class QaResult:
    findings: list[Finding] = field(default_factory=list)
    layers: list[LayerRow] = field(default_factory=list)
    checked_at: str = ""
    config: dict = field(default_factory=dict)

    def count(self, severity: Severity) -> int:
        return sum(f.severity == severity for f in self.findings)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def passed(self) -> bool:
        return not self.errors

    def add(self, severity: Severity, check: str, subject: str, message: str, ids: list[str] | None = None) -> None:
        self.findings.append(Finding(severity, check, subject, message, [str(i) for i in (ids or [])][:MAX_IDS]))


# --- helpers ----------------------------------------------------------------------------------------


def _same_crs(a: object, b: object) -> bool:
    try:
        return CRS.from_user_input(a) == CRS.from_user_input(b)
    except CRSError:  # an unparseable CRS is simply "not the same"
        return False


def _label(gdf: gpd.GeoDataFrame, index: int, columns: tuple[str, ...]) -> str:
    for c in columns:
        if c in gdf.columns and pd.notna(gdf.iloc[index][c]):
            return str(gdf.iloc[index][c])
    return f"row {index}"


def _is_polygonal(gdf: gpd.GeoDataFrame) -> bool:
    kinds = set(gdf.geom_type.dropna())
    return bool(kinds) and kinds <= {"Polygon", "MultiPolygon"}


def _width(geom) -> float:
    """Short side of the minimum rotated rectangle: how narrow a polygon is."""
    rect = geom.minimum_rotated_rectangle
    if rect.geom_type != "Polygon":
        return 0.0
    xs, ys = rect.exterior.coords.xy
    sides = [float(np.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])) for i in range(4)]
    return min(sides)


def _provenance_datasets(project: Project) -> set[str]:
    return {r.dataset for r in project.provenance.read()}


# --- layers -----------------------------------------------------------------------------------------


def check_layers(project: Project, cfg: QaConfig, result: QaResult) -> None:
    """The inventory of ``data.gpkg``, with the generic checks on each layer."""
    gpkg = project.data_gpkg
    if not gpkg.is_file():
        result.add("warn", "inventory", "data.gpkg", "the project has no data.gpkg yet (run 'gossan harvest')")
        return
    aoi = project.read_aoi()
    aoi_zone = aoi.union_all().buffer(1000)
    known = _provenance_datasets(project)
    for name in (str(n) for n in pyogrio.list_layers(gpkg)[:, 0]):
        info = pyogrio.read_info(gpkg, layer=name)
        stored = info.get("crs")
        n_features = int(info["features"])
        gdf = gpd.read_file(gpkg, layer=name, engine="pyogrio")
        shown = "none" if not stored else str(CRS.from_user_input(stored).to_string())
        kinds = "/".join(sorted(set(gdf.geom_type.dropna()))) or "none"
        result.layers.append(LayerRow(name, "vector", kinds, n_features, shown))
        if not stored:
            result.add("error", "crs", name, "the layer has no CRS, so its position cannot be trusted; assign one in QGIS or re-ingest with --crs")
            continue
        if not _same_crs(stored, project.config.crs):
            result.add("info", "crs", name, f"stored in {shown}; Gossan reprojects it to {project.config.crs} when it reads it")
        if n_features == 0:
            result.add("warn", "inventory", name, "the layer is empty")
            continue
        empty = gdf.geometry.isna() | gdf.geometry.is_empty
        if empty.any():
            result.add("warn", "geometry", name, f"{int(empty.sum())} feature(s) have no geometry", list(gdf.index[empty]))
        gdf = gdf[~empty]
        if gdf.empty:
            continue
        invalid = ~gdf.geometry.is_valid
        if invalid.any():
            severity: Severity = "error" if name in GOSSAN_LAYERS else "warn"
            reasons = sorted({explain_validity(g).split("[")[0].strip() for g in gdf.geometry[invalid]})
            result.add(severity, "geometry", name, f"{int(invalid.sum())} invalid geometr{'y' if invalid.sum() == 1 else 'ies'} ({'; '.join(reasons[:3])})", list(gdf.index[invalid]))
        dup = gdf.geometry.map(lambda g: g.wkb).duplicated(keep="first")
        if dup.any() and name != "aoi":
            result.add("warn", "geometry", name, f"{int(dup.sum())} feature(s) repeat another feature's geometry exactly", list(gdf.index[dup]))
        lines = gdf[gdf.geom_type.isin(["LineString", "MultiLineString"])]
        zero = lines[lines.to_crs(project.config.crs).geometry.length == 0] if stored else lines.iloc[0:0]
        if len(zero):
            result.add("warn", "geometry", name, f"{len(zero)} line(s) have zero length", list(zero.index))
        if name != "aoi":
            inside = gdf.to_crs(project.config.crs).intersects(aoi_zone)
            if not inside.any():
                result.add("error", "location", name, "no feature lies in or within 1 km of the AOI: a wrong CRS or the wrong area (swapped x/y, another UTM zone, NAD27 against NAD83 gives a small shift only)")
        sparse = []
        for col in gdf.columns.drop(gdf.geometry.name):
            column = gdf[col]
            blank = column.isna()
            if column.dtype == object or pd.api.types.is_string_dtype(column):
                blank |= column.astype("string").str.strip().eq("").fillna(False)
            share = float(blank.mean())
            if share > 0 and share >= cfg.empty_share:
                sparse.append(f"{col} {share:.0%}")
        if sparse:
            result.add("info", "attributes", name, "mostly empty attributes: " + ", ".join(sparse[:8]) + (" ..." if len(sparse) > 8 else ""))
        if name not in known and name != "aoi":
            result.add("warn", "provenance", name, "no provenance record names this layer (where did it come from?)")


# --- claims -----------------------------------------------------------------------------------------


def _claim_checks(gdf: gpd.GeoDataFrame, subject: str, labels: tuple[str, ...], cfg: QaConfig, result: QaResult, *, overlaps: Literal["warn", "info", "skip"] = "warn") -> None:
    """Overlaps, slivers and gaps among polygons that are all in a projected CRS."""
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].reset_index(drop=True)
    if gdf.empty or not _is_polygonal(gdf):
        return
    geoms = [g if g.is_valid else g.buffer(0) for g in gdf.geometry]  # repaired copies, for measuring only
    # slivers
    slivers = [i for i, g in enumerate(geoms) if g.area < cfg.min_area_m2 or _width(g) < cfg.min_width_m]
    if slivers:
        result.add("warn", "sliver", subject, f"{len(slivers)} polygon(s) are slivers (under {cfg.min_area_m2:g} m2 or narrower than {cfg.min_width_m:g} m)", [_label(gdf, i, labels) for i in slivers])
    # overlaps between different polygons
    if overlaps != "skip" and len(geoms) > 1:
        tree = STRtree(geoms)
        pairs, total = [], 0.0
        for i, g in enumerate(geoms):
            for j in tree.query(g, predicate="intersects"):
                if j <= i:
                    continue
                area = g.intersection(geoms[j]).area
                if area > cfg.overlap_tolerance_m2:
                    pairs.append((i, int(j), area))
                    total += area
        if pairs:
            names = [f"{_label(gdf, i, labels)} / {_label(gdf, j, labels)} ({a:,.0f} m2)" for i, j, a in sorted(pairs, key=lambda p: -p[2])]
            note = "" if overlaps == "warn" else " (expected for this source)"
            result.add(overlaps, "overlap", subject, f"{len(pairs)} pair(s) of polygons overlap by more than {cfg.overlap_tolerance_m2:g} m2, {total:,.0f} m2 in all{note}", names)
    # gaps between neighbours: closed holes and thin open gaps
    if len(geoms) > 1:
        union = unary_union(geoms)
        closed = union.buffer(cfg.gap_close_m).buffer(-cfg.gap_close_m)
        gap_geom = closed.difference(union)
        parts = list(gap_geom.geoms) if hasattr(gap_geom, "geoms") else ([gap_geom] if not gap_geom.is_empty else [])
        found_already = unary_union(parts) if parts else None
        for poly in (union.geoms if hasattr(union, "geoms") else [union]):
            for ring in poly.interiors:  # closed holes; a narrow one was already found by the closing above
                hole = Polygon(ring)
                if found_already is None or not found_already.intersects(hole.representative_point()):
                    parts.append(hole)
        small = [p for p in parts if cfg.gap_min_m2 <= p.area < cfg.gap_max_m2]
        if small:
            result.add("warn", "gap", subject, f"{len(small)} gap(s) between polygons, {sum(p.area for p in small):,.0f} m2 in all (a digitising error, or ground nobody holds)", [f"{p.area:,.0f} m2 near ({p.centroid.x:,.0f}, {p.centroid.y:,.0f})" for p in sorted(small, key=lambda p: -p.area)])


def check_claims(project: Project, cfg: QaConfig, result: QaResult) -> None:
    aoi = project.read_aoi()
    result.add("info", "claims", "aoi", f"{len(aoi)} claim polygon(s), {aoi.union_all().area / 1e4:,.1f} ha in all, CRS {project.config.crs}")
    invalid = ~aoi.geometry.is_valid
    if invalid.any():
        reasons = sorted({explain_validity(g).split("[")[0].strip() for g in aoi.geometry[invalid]})
        result.add("error", "geometry", "aoi", f"{int(invalid.sum())} claim polygon(s) are invalid ({'; '.join(reasons[:3])}); repair them before anything is built from them", list(aoi.index[invalid]))
    _claim_checks(aoi, "aoi (claim block)", ("claim", "name", "CLAIM_NAME", "id"), cfg, result)

    gpkg = project.data_gpkg
    if not gpkg.is_file() or "mto_tenure" not in [str(n) for n in pyogrio.list_layers(gpkg)[:, 0]]:
        return
    tenure = gpd.read_file(gpkg, layer="mto_tenure", engine="pyogrio")
    if tenure.empty or tenure.crs is None:
        return
    tenure = tenure.to_crs(project.config.crs)
    if "TENURE_NUMBER_ID" in tenure.columns:
        unique = tenure.drop_duplicates("TENURE_NUMBER_ID")  # one row per owner of a tenure
        dup = unique[unique.geometry.map(lambda g: g.wkb).duplicated(keep=False)]
        if len(dup) > 1:
            result.add("warn", "geometry", "mto_tenure", f"{len(dup)} different tenure numbers share an identical outline", [int(v) for v in dup["TENURE_NUMBER_ID"]])
        tenure = unique
    mode: Literal["warn", "info"] = "info" if project.config.jurisdiction == "us" else "warn"
    _claim_checks(tenure, "mto_tenure", ("CLAIM_NAME", "TENURE_NUMBER_ID"), cfg, result, overlaps=mode)
    if "GOOD_TO_DATE" in tenure.columns:
        good_to = pd.to_datetime(tenure["GOOD_TO_DATE"], utc=True, errors="coerce")
        expired = tenure[good_to < pd.Timestamp(datetime.now(UTC))]
        if len(expired):
            result.add("info", "tenure", "mto_tenure", f"{len(expired)} tenure(s) are past their good-to date; confirm status on Mineral Titles Online", [int(v) for v in expired.get("TENURE_NUMBER_ID", expired.index)])


# --- rasters ----------------------------------------------------------------------------------------


def check_rasters(project: Project, cfg: QaConfig, result: QaResult) -> None:
    aoi = project.read_aoi()
    paths = sorted((project.root / "rasters").glob("*.tif")) + sorted((project.root / "target").glob("*.tif"))
    for path in paths:
        rel = path.relative_to(project.root).as_posix()
        try:
            with rasterio.open(path) as ds:
                crs = ds.crs
                shown = crs.to_string() if crs else "none"
                result.layers.append(LayerRow(rel, "raster", f"{ds.count} band(s)", ds.width * ds.height, shown, f"{ds.res[0]:g} x {ds.res[1]:g}"))
                if not crs:
                    result.add("error", "crs", rel, "the raster has no CRS")
                    continue
                if not _same_crs(crs, project.config.crs):
                    result.add("info", "crs", rel, f"stored in {shown}; Gossan reprojects it when it reads it")
                data = ds.read(1, masked=True)
                if not (~np.ma.getmaskarray(data)).any() or not np.isfinite(data.filled(np.nan)).any():
                    result.add("error", "raster", rel, "every cell is nodata")
                    continue
                shape = aoi.to_crs(crs).union_all()
                if not shape.intersects(Polygon.from_bounds(*ds.bounds)):
                    result.add("error", "location", rel, "the raster does not overlap the AOI: a wrong CRS or the wrong area")
                    continue
                inside = ~geometry_mask([shape], out_shape=(ds.height, ds.width), transform=ds.transform, invert=False)
                valid_in = (~np.ma.getmaskarray(data) & np.isfinite(data.filled(np.nan)) & inside).sum()
                footprint = Polygon.from_bounds(*ds.bounds)
                share = (valid_in / max(int(inside.sum()), 1)) * (shape.intersection(footprint).area / shape.area)  # valid cells, and how much of the AOI the grid reaches
                if share < cfg.min_raster_coverage:
                    result.add("warn", "raster", rel, f"valid cells cover only {share:.0%} of the AOI")
        except rasterio.errors.RasterioIOError as e:
            result.add("error", "raster", rel, f"cannot be opened: {e}")


# --- delivery package -------------------------------------------------------------------------------


def check_package(project: Project, result: QaResult, folder: Path | None = None) -> None:
    """The delivery package, if one exists: files unchanged, licences recorded, nothing restricted slipped in."""
    import hashlib
    import json

    folder = folder or project.outputs_dir / "deliverable"
    manifest_path = folder / MANIFEST
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    subject = folder.relative_to(project.root).as_posix() if folder.is_relative_to(project.root) else str(folder)
    changed, missing = [], []
    for entry in manifest.get("files", []):
        path = folder / entry["path"]
        if not path.is_file():
            missing.append(entry["path"])
            continue
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != entry["sha256"]:
            changed.append(entry["path"])
    if missing:
        result.add("error", "package", subject, f"{len(missing)} file(s) in the manifest are missing", missing)
    if changed:
        result.add("error", "package", subject, f"{len(changed)} file(s) changed since the package was built; rebuild it with 'gossan deliver'", changed)
    no_licence = [i["title"] for i in manifest.get("included", []) if not i.get("licence")]
    if no_licence:
        result.add("error", "package", subject, f"{len(no_licence)} item(s) have no licence on record", no_licence)
    restricted = [i["title"] for i in manifest.get("included", []) if any(w in (i.get("licence") or "").lower() for w in RESTRICTING)]
    if restricted and not manifest.get("internal"):
        result.add("error", "package", subject, "restricted items are in a package not marked internal", restricted)
    if decide_marking(project).marked and not manifest.get("internal"):
        extra = [i["title"] for i in manifest.get("included", []) if "aris" in (i.get("key", "") + i.get("title", "")).lower()]
        if extra:
            result.add("error", "package", subject, "ARIS-derived items are in a package not marked internal", extra)
    n_files = len(manifest.get("files", []))
    on_disk = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()} - {MANIFEST, "README.txt"}
    extras = sorted(on_disk - {e["path"] for e in manifest.get("files", [])})
    if extras:
        result.add("warn", "package", subject, f"{len(extras)} file(s) in the folder are not in the manifest", extras)
    result.add("info", "package", subject, f"{n_files} file(s) checked against the manifest built {manifest.get('built_at', '?')}")


def run_checks(project: Project, *, package: bool = True) -> QaResult:
    cfg = QaConfig.from_project(project)
    result = QaResult(checked_at=datetime.now(UTC).isoformat(timespec="seconds"), config=cfg.model_dump(mode="json"))
    check_claims(project, cfg, result)
    check_layers(project, cfg, result)
    check_rasters(project, cfg, result)
    if package:
        check_package(project, result)
    order = {s: i for i, s in enumerate(SEVERITIES)}
    result.findings.sort(key=lambda f: order[f.severity])
    return result
