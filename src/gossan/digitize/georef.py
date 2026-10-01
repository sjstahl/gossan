"""Georeferencing scanned report maps.

Two routes into the same fit-and-warp step:

* **Automatic**: OpenCV finds long horizontal and vertical grid lines in the rendered page,
  EasyOCR (optional) reads the coordinate labels beyond their ends, and every
  vertical x horizontal line crossing becomes a ground control point (GCP).
* **Manual (fallback)**: a QGIS Georeferencer ``.points`` file (see ``qgis_fallback``).

The fit is a first-order (affine) transform of the GCPs, computed by GDAL (through
rasterio); the image is then warped to the project CRS and written as a Cloud Optimized
GeoTIFF in ``rasters/``. The root-mean-square GCP error, in metres, must not exceed
``georef.rms_threshold_m`` in ``project.yaml`` (default 15) or nothing is written. With only
three GCPs an affine fit is exact, so its RMS says nothing; the automatic route therefore
needs ``georef.min_gcps`` (default 4).

Limits, so nobody trusts this more than it deserves:

* Only grid lines that run a good fraction of the page width/height are detected. Margin
  ticks without lines, curved or heavily skewed grids, and maps with no printed grid need
  the QGIS route.
* The fit is affine; strongly distorted scans may need a higher-order fit in QGIS.
* A map in NAD27 labelled in UTM is fitted in NAD27 (``georef.datum``) and then reprojected,
  which is only as good as the datum-shift grids PROJ has installed. A wrong ``datum`` is
  *not* caught by the RMS (it shifts every point together); the review step must check
  the placement against known features.
* Every georeferenced map is added to the review queue as PENDING. Nothing here is checked.

GDAL is used through rasterio's bundled build, so the separate ``osgeo`` Python bindings
(which need a system GDAL) are not a dependency.
"""

from __future__ import annotations

import math
import re
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import rasterio
from pyproj import CRS
from rasterio.control import GroundControlPoint
from rasterio.enums import ColorInterp, Resampling
from rasterio.shutil import copy as rio_copy
from rasterio.transform import Affine, from_gcps
from rasterio.warp import calculate_default_transform, reproject

from gossan.core.config import GeorefConfig
from gossan.core.project import Project
from gossan.core.raster import raster_path
from gossan.digitize import review
from gossan.digitize.ocr import OcrUnavailableError, load_reader
from gossan.digitize.pages import render_page
from gossan.digitize.qgis_fallback import Gcp, PointsFile, write_points
from gossan.digitize.text import DERIVED_LICENSE

MAP_MAX_SIDE_PX = 6000
#: Labels are only trusted within this distance of the AOI (metres): rejects OCR misreads.
WINDOW_PAD_M = 50_000
#: Largest GeoTIFF (pixels) a warp may produce; a bad fit can otherwise explode the size.
MAX_OUTPUT_PIXELS = 150_000_000
_OCR_ALLOW = "0123456789 ,.NE"


class GeorefError(Exception):
    """Georeferencing could not proceed; the message says why."""


# --- CRS ----------------------------------------------------------------------------------


def gcp_crs_for(project: Project) -> str:
    """CRS of the map's grid labels: ``georef.gcp_crs``, else the project's UTM zone in
    ``georef.datum``."""
    cfg = project.config.georef
    if cfg.gcp_crs:
        return CRS.from_user_input(cfg.gcp_crs).to_string()
    zone = project.config.utm_zone
    if zone is None:
        raise GeorefError(
            "the project CRS is not UTM, so the grid-label CRS is unknown: set georef.gcp_crs "
            "in project.yaml"
        )
    base = {"NAD83": 26900, "NAD27": 26700, "WGS84": 32600}[cfg.datum]
    if cfg.datum == "NAD27" and zone > 22:
        raise GeorefError(f"NAD27 has no UTM zone {zone}; set georef.gcp_crs")
    return f"EPSG:{base + zone}"


# --- fit ----------------------------------------------------------------------------------


@dataclass
class Fit:
    transform: Affine  # image (col, row) -> map (x, y)
    rms_m: float
    max_residual_m: float
    n_gcps: int

    @property
    def redundant(self) -> int:
        """GCPs beyond the 3 an affine fit needs; 0 means the RMS is meaningless."""
        return self.n_gcps - 3


def fit_affine(gcps: list[Gcp]) -> Fit:
    """Least-squares affine fit (GDAL) of image pixels to map coordinates, with its residuals."""
    if len(gcps) < 3:
        raise GeorefError(f"{len(gcps)} GCP(s): at least 3 are needed")
    design = np.array([[g.col, g.row, 1.0] for g in gcps])
    if np.linalg.matrix_rank(design) < 3:
        raise GeorefError("the GCPs lie on one line (or coincide); an affine fit is undefined")
    transform = from_gcps([GroundControlPoint(row=g.row, col=g.col, x=g.x, y=g.y) for g in gcps])
    if not math.isfinite(transform.determinant) or transform.determinant == 0:
        raise GeorefError("the fitted transform is degenerate")
    residuals = []
    for g in gcps:
        px, py = transform * (g.col, g.row)
        residuals.append(math.hypot(px - g.x, py - g.y))
    rms = math.sqrt(sum(r * r for r in residuals) / len(residuals))
    return Fit(transform, rms, max(residuals), len(gcps))


# --- warp ---------------------------------------------------------------------------------


def warp_to_cog(image: Path, transform: Affine, src_crs: str, dst_crs: str, dst: Path) -> dict[str, Any]:
    """Warp ``image`` (positioned by ``transform`` in ``src_crs``) into ``dst_crs`` as a COG.

    The output has an alpha band so the area outside the map is transparent.
    """
    with warnings.catch_warnings():  # a rendered page image has no georeferencing, by design
        warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
        with rasterio.open(image) as src:
            data = src.read()
    bands, h, w = data.shape
    source = np.concatenate([data, np.full((1, h, w), 255, dtype=data.dtype)])
    corners = [transform * c for c in ((0, 0), (w, 0), (0, h), (w, h))]
    xs, ys = zip(*corners, strict=True)
    resolution = math.sqrt(abs(transform.determinant))
    dst_transform, dw, dh = calculate_default_transform(
        src_crs, dst_crs, w, h,
        left=min(xs), bottom=min(ys), right=max(xs), top=max(ys), resolution=resolution,
    )  # fmt: skip
    if dw * dh > MAX_OUTPUT_PIXELS:
        raise GeorefError(f"the warped image would be {dw} x {dh} pixels; the fit is implausible")
    out = np.zeros((bands + 1, dh, dw), dtype=source.dtype)
    reproject(
        source, out,
        src_transform=transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=dst_crs,
        resampling=Resampling.bilinear,
    )  # fmt: skip

    dst.parent.mkdir(parents=True, exist_ok=True)
    interp = {
        2: [ColorInterp.gray, ColorInterp.alpha],
        4: [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha],
    }.get(bands + 1)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_tif = Path(tmp) / "warped.tif"
        profile = {
            "driver": "GTiff", "width": dw, "height": dh, "count": bands + 1,
            "dtype": str(out.dtype), "crs": dst_crs, "transform": dst_transform, "tiled": True,
        }  # fmt: skip
        with rasterio.open(tmp_tif, "w", **profile) as dataset:
            dataset.write(out)
            if interp:
                dataset.colorinterp = interp
        rio_copy(tmp_tif, dst, driver="COG", compress="DEFLATE")
    return {"width": dw, "height": dh, "pixel_size_m": round(abs(dst_transform.a), 3)}


@dataclass
class GeorefOutcome:
    ok: bool
    reason: str = ""
    fit: Fit | None = None
    output: Path | None = None
    details: dict[str, Any] = field(default_factory=dict)


def georeference(
    image: Path,
    gcps: list[Gcp],
    gcp_crs: str,
    dst_crs: str,
    dst: Path,
    *,
    rms_threshold_m: float,
    min_gcps: int,
) -> GeorefOutcome:
    """Fit, validate and warp. Fails (without writing ``dst``) on too few GCPs, a degenerate
    fit, or an RMS above ``rms_threshold_m``."""
    if len(gcps) < min_gcps:
        return GeorefOutcome(False, f"{len(gcps)} GCP(s); at least {min_gcps} are required")
    try:
        fit = fit_affine(gcps)
    except GeorefError as e:
        return GeorefOutcome(False, str(e))
    if fit.rms_m > rms_threshold_m:
        return GeorefOutcome(
            False,
            f"RMS {fit.rms_m:.1f} m exceeds the {rms_threshold_m:g} m threshold "
            f"(worst point {fit.max_residual_m:.1f} m, {fit.n_gcps} GCPs)",
            fit,
        )
    try:
        details = warp_to_cog(image, fit.transform, gcp_crs, dst_crs, dst)
    except GeorefError as e:
        return GeorefOutcome(False, str(e), fit)
    return GeorefOutcome(True, "", fit, dst, details)


# --- automatic GCPs: OpenCV grid detection ------------------------------------------------


@dataclass
class GridLine:
    pos: float  # position across the line, in continuous pixel coordinates (edge = integer)
    start: float  # extent along the line
    end: float


@dataclass
class GridLines:
    vertical: list[GridLine]
    horizontal: list[GridLine]
    width: int
    height: int


def _runs(indices: np.ndarray, max_gap: int = 2) -> list[tuple[int, int]]:
    """Group sorted integer indices into (first, last) runs allowing small gaps."""
    runs: list[tuple[int, int]] = []
    for i in indices.tolist():
        if runs and i - runs[-1][1] <= max_gap + 1:
            runs[-1] = (runs[-1][0], i)
        else:
            runs.append((i, i))
    return runs


def detect_grid_lines(image: Path, min_length_frac: float = 0.35) -> GridLines:
    """Find long horizontal and vertical lines (map grid, neatline) with OpenCV.

    A line must span at least ``min_length_frac`` of the page in its direction; dashed lines
    are bridged. Slightly rotated scans are not deskewed.
    """
    import cv2  # imported here: OpenCV is only needed for this step

    gray = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise GeorefError(f"cannot read image {image}")
    h, w = gray.shape
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)

    def lines_along(kernel_shape: tuple[int, int], axis: int, extent: int) -> list[GridLine]:
        # Close small gaps (dashes), then keep only long straight runs.
        bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1) if axis == 1 else (1, 15))
        closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, bridge)
        long_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_shape)
        mask = cv2.morphologyEx(closed, cv2.MORPH_OPEN, long_kernel) > 0
        profile = mask.sum(axis=axis)
        hits = np.flatnonzero(profile >= min_length_frac * extent)
        found = []
        for first, last in _runs(hits):
            band = mask[first : last + 1, :] if axis == 1 else mask[:, first : last + 1]
            along = np.flatnonzero(band.any(axis=0 if axis == 1 else 1))
            found.append(GridLine((first + last + 1) / 2, float(along.min()), float(along.max() + 1)))
        return found

    horizontal = lines_along((max(25, int(w * min_length_frac * 0.6)), 1), 1, w)
    vertical = lines_along((1, max(25, int(h * min_length_frac * 0.6))), 0, h)
    return GridLines(vertical, horizontal, w, h)


# --- automatic GCPs: reading labels, assembling points -----------------------------------


@dataclass
class LineLabel:
    axis: Literal["x", "y"]  # "x": easting of a vertical line; "y": northing of a horizontal one
    pixel: float
    value: float
    confidence: float
    text: str


def resolve_value(digits: str, lo: float, hi: float) -> float | None:
    """Turn OCR'd digits into a map coordinate inside ``[lo, hi]``.

    Grid labels are often abbreviated ("689" for 689000 m); the scale (x1, x10, x100, x1000)
    that lands the value in the plausible window is used, smallest first.
    """
    if not digits:
        return None
    n = int(digits)
    for scale in (1, 10, 100, 1000):
        if lo <= n * scale <= hi:
            return float(n * scale)
    return None


def _drop_outliers(pairs: list[tuple[float, float]], tolerance: float) -> list[tuple[float, float]]:
    """Remove labels that disagree with the straight line the others lie on."""
    pairs = list(pairs)
    while len(pairs) > 2:
        px = np.array([p[0] for p in pairs])
        val = np.array([p[1] for p in pairs])
        slope, intercept = np.polyfit(px, val, 1)
        resid = np.abs(val - (slope * px + intercept))
        worst = int(resid.argmax())
        if resid[worst] <= tolerance:
            break
        pairs.pop(worst)
    return pairs


def assemble_gcps(
    labels: list[LineLabel], tolerance_m: float
) -> tuple[list[Gcp], list[str]]:
    """GCPs from every vertical x horizontal label pair, after dropping inconsistent labels.

    A line labelled at both ends keeps its more confident reading. Returns the GCPs and notes.
    """
    notes: list[str] = []
    axes: dict[str, list[tuple[float, float]]] = {}
    for axis in ("x", "y"):
        best: dict[float, LineLabel] = {}
        for lab in (lab for lab in labels if lab.axis == axis):
            if lab.pixel not in best or lab.confidence > best[lab.pixel].confidence:
                best[lab.pixel] = lab
        pairs = sorted((p, best[p].value) for p in best)
        kept = _drop_outliers(pairs, tolerance_m)
        if len(kept) < len(pairs):
            notes.append(f"dropped {len(pairs) - len(kept)} inconsistent {axis} label(s)")
        axes[axis] = kept
    if len(axes["x"]) < 2 or len(axes["y"]) < 2:
        notes.append(
            f"readable labels: {len(axes['x'])} vertical and {len(axes['y'])} horizontal "
            "line(s); at least 2 of each are needed"
        )
        return [], notes
    gcps = [
        Gcp(col=px, row=py, x=easting, y=northing, label=f"{easting:.0f}/{northing:.0f}")
        for px, easting in axes["x"]
        for py, northing in axes["y"]
    ]
    return gcps, notes


def _number_candidates(reader: Any, crop: np.ndarray) -> list[tuple[str, float]]:
    """Digit strings (with confidence) OCR finds in ``crop`` upright and rotated both ways."""
    import cv2

    found: list[tuple[str, float]] = []
    for rotated in (crop, cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)):
        for _box, text, conf in reader.readtext(rotated, allowlist=_OCR_ALLOW, detail=1):
            for run in re.findall(r"\d[\d ,.]*\d|\d", str(text)):
                digits = re.sub(r"\D", "", run)
                if len(digits) >= 2:
                    found.append((digits, float(conf)))
    return found


def read_labels(
    image: Path,
    grid: GridLines,
    reader: Any,
    easting_window: tuple[float, float],
    northing_window: tuple[float, float],
) -> list[LineLabel]:
    """OCR the coordinate label beyond each end of every detected grid line."""
    import cv2

    gray = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    h, w = gray.shape
    band = max(40, int(0.035 * max(h, w)))

    def crop(x0: float, y0: float, x1: float, y1: float) -> np.ndarray | None:
        x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(w, int(x1)), min(h, int(y1))
        return gray[y0:y1, x0:x1] if x1 - x0 > 8 and y1 - y0 > 8 else None

    labels: list[LineLabel] = []
    for line in grid.vertical:  # easting labels sit above or below the line
        for c in (
            crop(line.pos - band, line.start - 2 * band, line.pos + band, line.start),
            crop(line.pos - band, line.end, line.pos + band, line.end + 2 * band),
        ):
            labels += _best_labels(reader, c, "x", line.pos, easting_window)
    for line in grid.horizontal:  # northing labels sit left or right of the line
        for c in (
            crop(line.start - 2 * band, line.pos - band, line.start, line.pos + band),
            crop(line.end, line.pos - band, line.end + 2 * band, line.pos + band),
        ):
            labels += _best_labels(reader, c, "y", line.pos, northing_window)
    return labels


def _best_labels(reader, crop, axis, pixel, window) -> list[LineLabel]:
    if crop is None:
        return []
    resolved = [
        (v, conf, digits)
        for digits, conf in _number_candidates(reader, crop)
        if (v := resolve_value(digits, *window)) is not None
    ]
    if not resolved:
        return []
    value, conf, digits = max(resolved, key=lambda r: r[1])
    return [LineLabel(axis, pixel, value, conf, digits)]


@dataclass
class AutoGcps:
    gcps: list[Gcp]
    notes: list[str]
    n_vertical: int = 0
    n_horizontal: int = 0


def auto_gcps(project: Project, image: Path, gcp_crs: str, reader: Any = None) -> AutoGcps:
    """Candidate GCPs for ``image`` from its printed grid; empty (with notes saying why) if the
    grid or its labels cannot be read."""
    grid = detect_grid_lines(image)
    notes = [f"grid lines found: {len(grid.vertical)} vertical, {len(grid.horizontal)} horizontal"]
    result = AutoGcps([], notes, len(grid.vertical), len(grid.horizontal))
    if len(grid.vertical) < 2 or len(grid.horizontal) < 2:
        notes.append("need at least 2 grid lines in each direction")
        return result
    if reader is None:
        try:
            reader = load_reader()
        except OcrUnavailableError as e:
            notes.append(f"cannot read coordinate labels: {e}")
            return result

    aoi = project.read_aoi().to_crs(gcp_crs)
    minx, miny, maxx, maxy = aoi.total_bounds
    labels = read_labels(
        image, grid, reader,
        (minx - WINDOW_PAD_M, maxx + WINDOW_PAD_M), (miny - WINDOW_PAD_M, maxy + WINDOW_PAD_M),
    )  # fmt: skip
    notes.append(f"coordinate labels read: {len(labels)}")
    gcps, more = assemble_gcps(labels, project.config.georef.rms_threshold_m)
    notes += more
    result.gcps = gcps
    return result


# --- one map page, end to end -------------------------------------------------------------


def map_image_path(project: Project, aris_number: int, page: int) -> Path:
    return project.root / "digitize" / "maps" / f"{aris_number:05d}_p{page}.png"


def points_path(project: Project, aris_number: int, page: int, kind: str) -> Path:
    return project.root / "digitize" / "gcps" / f"{aris_number:05d}_p{page}_{kind}.points"


def map_output_name(aris_number: int, page: int) -> str:
    return f"aris_{aris_number:05d}_p{page}"


@dataclass
class PageGeoref:
    page: int
    ok: bool
    method: str
    reason: str = ""
    output: Path | None = None
    image: Path | None = None
    rms_m: float | None = None
    n_gcps: int = 0
    candidate_points: Path | None = None
    notes: list[str] = field(default_factory=list)


def georeference_map_page(
    project: Project,
    aris_number: int,
    pdf: Path,
    page: int,
    *,
    points: PointsFile | None = None,
    image: Path | None = None,
    rms_threshold_m: float | None = None,
    reader: Any = None,
) -> PageGeoref:
    """Georeference one map page, by the automatic route or from a parsed ``.points`` file.

    On success the COG is in ``rasters/``, provenance is written and the map is queued for
    review. On failure nothing is queued, the reason is returned (and logged), and for a
    failed automatic fit the candidate GCPs are saved as a QGIS ``.points`` file to correct.
    """
    cfg: GeorefConfig = project.config.georef
    method = "manual_qgis" if points is not None else "auto"
    threshold = rms_threshold_m if rms_threshold_m is not None else cfg.rms_threshold_m
    out = PageGeoref(page, False, method)
    try:
        if image is None:
            image = map_image_path(project, aris_number, page)
            if not image.exists():  # reuse an existing render: QGIS pixel positions refer to it
                render_page(pdf, page, image, cfg.dpi, max_side=MAP_MAX_SIDE_PX)
        out.image = image
        if points is not None:
            gcps = points.gcps
            gcp_crs = CRS.from_user_input(points.crs).to_string() if points.crs else gcp_crs_for(project)
            out.notes += points.notes
            min_gcps = 3
        else:
            gcp_crs = gcp_crs_for(project)
            auto = auto_gcps(project, image, gcp_crs, reader)
            gcps, out.notes, min_gcps = auto.gcps, auto.notes, cfg.min_gcps
    except (GeorefError, OSError, ValueError, RuntimeError) as e:
        out.reason = str(e)
        _log_failure(project, aris_number, out)
        return out

    out.n_gcps = len(gcps)
    dst = raster_path(project, map_output_name(aris_number, page))
    outcome = (
        georeference(image, gcps, gcp_crs, project.config.crs, dst, rms_threshold_m=threshold, min_gcps=min_gcps)
        if gcps
        else GeorefOutcome(False, "no GCPs could be derived (" + "; ".join(out.notes[-2:]) + ")")
    )
    out.rms_m = outcome.fit.rms_m if outcome.fit else None
    if not outcome.ok:
        out.reason = outcome.reason
        if method == "auto" and gcps:
            out.candidate_points = points_path(project, aris_number, page, "auto")
            write_points(out.candidate_points, gcps, gcp_crs)
        _log_failure(project, aris_number, out)
        return out

    fit = outcome.fit
    out.ok, out.output = True, outcome.output
    rel = lambda p: p.relative_to(project.root).as_posix()
    project.provenance.log(
        dataset="aris_maps",
        source="gossan (OpenCV + EasyOCR)" if method == "auto" else "QGIS Georeferencer points (user)",
        identifier=f"ARIS {aris_number} p{page}",
        url=None,
        license=DERIVED_LICENSE,
        operation="georef_auto" if method == "auto" else "georef_manual",
        parameters={
            "input": f"raw/aris_pdfs/{pdf.name}",
            "page": page,
            "image": rel(image) if image.is_relative_to(project.root) else str(image),
            "output": rel(dst),
            "gcp_crs": gcp_crs,
            "project_crs": project.config.crs,
            "n_gcps": fit.n_gcps,
            "rms_m": round(fit.rms_m, 3),
            "max_residual_m": round(fit.max_residual_m, 3),
            "rms_threshold_m": threshold,
            "redundant_gcps": fit.redundant,
            "transform": "affine (first order)",
            "notes": out.notes,
            **outcome.details,
            "reviewed": False,
        },
    )
    caveat = "" if fit.redundant > 0 else " RMS is not meaningful: no redundant GCPs."
    review.add_item(
        project,
        aris_number,
        review.ItemType.MAP,
        page=page,
        source=rel(dst),
        image=rel(image) if image.is_relative_to(project.root) else str(image),
        note=(
            f"georeferenced {'automatically' if method == 'auto' else 'from QGIS points'}, "
            f"{fit.n_gcps} GCPs, RMS {fit.rms_m:.1f} m (limit {threshold:g} m, datum "
            f"{gcp_crs}); check placement against known features.{caveat}"
        ),
    )
    return out


def _log_failure(project: Project, aris_number: int, out: PageGeoref) -> None:
    project.provenance.log(
        dataset="aris_maps",
        source="gossan (OpenCV + EasyOCR)" if out.method == "auto" else "QGIS Georeferencer points (user)",
        identifier=f"ARIS {aris_number} p{out.page}",
        url=None,
        license=DERIVED_LICENSE,
        operation="georef_failed",
        parameters={
            "method": out.method,
            "page": out.page,
            "reason": out.reason,
            "n_gcps": out.n_gcps,
            "rms_m": None if out.rms_m is None else round(out.rms_m, 3),
            "notes": out.notes,
            "candidate_points": out.candidate_points.relative_to(project.root).as_posix()
            if out.candidate_points
            else None,
        },
    )
