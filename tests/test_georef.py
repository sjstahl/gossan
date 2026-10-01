from pathlib import Path

import cv2
import numpy as np
import pytest
import rasterio
from typer.testing import CliRunner

from gossan.cli import app
from gossan.core.project import Project
from gossan.digitize import georef, review
from gossan.digitize.georef import (
    GridLine,
    GridLines,
    LineLabel,
    assemble_gcps,
    detect_grid_lines,
    fit_affine,
    georeference,
    resolve_value,
)
from gossan.digitize.qgis_fallback import Gcp, PointsError, parse_points, write_points

runner = CliRunner()

# ------------------------------------------------------------------------------------------
# QGIS .points parser
# ------------------------------------------------------------------------------------------

QGIS3 = """\
#CRS: PROJCRS["NAD83 / UTM zone 10N",BASEGEOGCRS["NAD83",DATUM["North American Datum 1983"]],CONVERSION["UTM zone 10N"],ID["EPSG",26910]]
mapX,mapY,sourceX,sourceY,enable,dX,dY,residual
682400,5613200,366.0,-165.0,1,0,0,0
683100,5613200,2570.5,-165.0,1,0,0,0
682400,5612600,366.0,-2054.5,1,0,0,0
683100,5612600,2570.5,-2054.5,1,0,0,0
682900,5612900,1941.0,-1110.0,0,0,0,0
"""


def test_parse_qgis3_points():
    pts = parse_points(QGIS3)
    assert pts.crs.startswith("PROJCRS") and "26910" in pts.crs  # the WKT's commas do not matter
    assert len(pts.gcps) == 4 and pts.n_disabled == 1  # the enable=0 point is skipped
    first = pts.gcps[0]
    assert (first.x, first.y, first.col) == (682400.0, 5613200.0, 366.0)
    assert first.row == 165.0  # QGIS stores the row as a negative sourceY


def test_parse_older_header_names_and_epsg_crs():
    text = "#CRS: EPSG:26910\nmapX,mapY,pixelX,pixelY,enable\n1,2,10,-20,1\n3,4,30,-40,1\n5,7,50,-90,1\n"
    pts = parse_points(text)
    assert pts.crs == "EPSG:26910"
    assert [(g.col, g.row) for g in pts.gcps] == [(10, 20), (30, 40), (50, 90)]


def test_parse_is_not_tied_to_header_names_or_order():
    # Different names and column order, semicolons, a BOM, mixed case, no CRS line.
    text = "﻿source_col;SOURCE_ROW;Easting;Northing\n10;-20;1000;2000\n30;-40;1100;2100\n50;-90;1200;2300\n"
    pts = parse_points(text)
    assert pts.crs is None
    assert [(g.x, g.y, g.col, g.row) for g in pts.gcps] == [
        (1000, 2000, 10, 20), (1100, 2100, 30, 40), (1200, 2300, 50, 90)
    ]  # fmt: skip


def test_parse_without_header_uses_qgis_column_order():
    pts = parse_points("1000,2000,10,-20\n1100,2100,30,-40\n1200,2300,50,-90,1\n")
    first = pts.gcps[0]
    assert (first.x, first.y, first.col, first.row) == (1000, 2000, 10, 20)
    assert not pts.notes


def test_parse_unrecognised_header_falls_back_to_position_and_says_so():
    pts = parse_points("a,b,c,d\n1000,2000,10,-20\n1100,2100,30,-40\n1200,2300,50,-90\n")
    assert len(pts.gcps) == 3
    assert any("by position" in n for n in pts.notes)


def test_parse_non_negative_source_y_is_a_plain_row():
    pts = parse_points("mapX,mapY,sourceX,sourceY\n1,2,10,20\n3,4,30,40\n5,7,50,90\n")
    assert [g.row for g in pts.gcps] == [20, 40, 90]


def test_parse_errors():
    with pytest.raises(PointsError, match="mixed signs"):
        parse_points("mapX,mapY,sourceX,sourceY\n1,2,10,20\n3,4,30,-40\n5,7,50,-90\n")
    with pytest.raises(PointsError, match="at least 3"):
        parse_points("mapX,mapY,sourceX,sourceY\n1,2,10,-20\n3,4,30,-40\n")
    with pytest.raises(PointsError, match="no points"):
        parse_points("#CRS: EPSG:26910\n")
    with pytest.raises(PointsError, match="at least 3"):  # a disabled point does not count
        parse_points("mapX,mapY,sourceX,sourceY,enable\n1,2,1,-1,1\n3,4,3,-3,1\n5,7,5,-5,0\n")


def test_parse_counts_unreadable_rows():
    pts = parse_points("mapX,mapY,sourceX,sourceY\n1,2,10,-20\nx,y,z,w\n3,4,30,-40\n5,7,50,-90\n9,9\n")
    assert len(pts.gcps) == 3 and pts.n_unreadable == 2


def test_write_points_roundtrip(tmp_path):
    gcps = [Gcp(10.5, 20.25, 1000.0, 2000.0), Gcp(30, 40, 1100, 2100), Gcp(50, 90, 1200, 2300)]
    path = tmp_path / "x.points"
    write_points(path, gcps, "EPSG:26910")
    assert "sourceY" in path.read_text() and "-20.25" in path.read_text()  # QGIS's negative rows
    back = parse_points(path)
    assert back.crs == "EPSG:26910"
    assert [(g.col, g.row, g.x, g.y) for g in back.gcps] == [(g.col, g.row, g.x, g.y) for g in gcps]


# ------------------------------------------------------------------------------------------
# Fit, RMS and the threshold
# ------------------------------------------------------------------------------------------


def grid_gcps(noise: dict[int, tuple[float, float]] | None = None) -> list[Gcp]:
    """A 3 x 3 grid of GCPs for a north-up map at 2 m per pixel, optionally displaced."""
    gcps = []
    for i, (col, row) in enumerate((c, r) for r in (0, 40, 80) for c in (0, 50, 100)):
        dx, dy = (noise or {}).get(i, (0.0, 0.0))
        gcps.append(Gcp(col, row, 500_000 + 2.0 * col + dx, 5_600_000 - 2.0 * row + dy))
    return gcps


@pytest.fixture
def tiny_image(tmp_path) -> Path:
    """A 100 x 80 RGB raster standing in for a rendered map page."""
    path = tmp_path / "map.tif"
    data = np.random.default_rng(1).integers(0, 255, (3, 80, 100), dtype=np.uint8)
    with rasterio.open(path, "w", driver="GTiff", width=100, height=80, count=3, dtype="uint8") as ds:
        ds.write(data)
    return path


def test_exact_fit_has_zero_rms():
    fit = fit_affine(grid_gcps())
    assert fit.rms_m == pytest.approx(0, abs=1e-6)
    assert fit.n_gcps == 9 and fit.redundant == 6
    assert fit.transform.a == pytest.approx(2.0) and fit.transform.e == pytest.approx(-2.0)


def test_rms_is_the_root_mean_square_of_point_errors():
    # One of 9 points is 30 m off in x: the least-squares fit spreads some of it around, but
    # the RMS must lie between 0 and the displaced point's own error, and be clearly nonzero.
    fit = fit_affine(grid_gcps({4: (30.0, 0.0)}))
    assert 5 < fit.rms_m < 30
    assert fit.max_residual_m > fit.rms_m


def test_degenerate_gcps_are_rejected():
    line = [Gcp(c, c, 1000 + c, 2000 + c) for c in (0, 10, 20, 30)]  # all on one line
    with pytest.raises(georef.GeorefError, match="one line"):
        fit_affine(line)
    with pytest.raises(georef.GeorefError, match="at least 3"):
        fit_affine(line[:2])


def run_georeference(image, gcps, tmp_path, threshold=15.0, min_gcps=4):
    dst = tmp_path / "out" / "map.tif"
    outcome = georeference(
        image, gcps, "EPSG:26910", "EPSG:32610", dst, rms_threshold_m=threshold, min_gcps=min_gcps
    )
    return outcome, dst


def test_good_fit_writes_a_cog_in_the_project_crs(tiny_image, tmp_path):
    outcome, dst = run_georeference(tiny_image, grid_gcps(), tmp_path)
    assert outcome.ok and outcome.output == dst
    with rasterio.open(dst) as ds:
        assert ds.crs.to_epsg() == 32610
        assert ds.count == 4 and ds.colorinterp[-1].name == "alpha"  # transparent outside the map
        assert ds.tags(ns="IMAGE_STRUCTURE")["LAYOUT"] == "COG"
        assert ds.res[0] == pytest.approx(2.0, rel=0.05)  # native 2 m per pixel kept
        # 100 x 80 px at 2 m = 200 x 160 m (NAD83 -> WGS84 UTM shifts by about a metre)
        assert (ds.bounds.right - ds.bounds.left) == pytest.approx(200, abs=10)
        assert (ds.bounds.top - ds.bounds.bottom) == pytest.approx(160, abs=10)
        centre = ds.read(1, masked=False)[40, 50]
        assert ds.read(4)[40, 50] == 255 and isinstance(centre, np.uint8)


def test_rms_above_threshold_fails_gracefully_and_writes_nothing(tiny_image, tmp_path):
    bad = grid_gcps({4: (90.0, 0.0), 1: (0.0, 80.0), 7: (-70.0, 0.0)})
    outcome, dst = run_georeference(tiny_image, bad, tmp_path, threshold=15.0)
    assert not outcome.ok
    assert "exceeds the 15 m threshold" in outcome.reason
    assert outcome.fit is not None and outcome.fit.rms_m > 15
    assert not dst.exists() and not dst.parent.exists()


def test_threshold_is_the_deciding_line(tiny_image, tmp_path):
    gcps = grid_gcps({4: (30.0, 0.0)})
    rms = fit_affine(gcps).rms_m
    assert not run_georeference(tiny_image, gcps, tmp_path, threshold=rms - 0.5)[0].ok
    assert run_georeference(tiny_image, gcps, tmp_path, threshold=rms + 0.5)[0].ok


def test_too_few_gcps_for_the_route(tiny_image, tmp_path):
    every = grid_gcps()
    three = [every[0], every[2], every[6]]  # three corners: not collinear
    outcome, _ = run_georeference(tiny_image, [*three[:1], *three[1:2]], tmp_path, min_gcps=3)
    assert not outcome.ok and "at least 3" in outcome.reason
    auto, _ = run_georeference(tiny_image, three, tmp_path, min_gcps=4)  # automatic: needs 4
    assert not auto.ok and "at least 4" in auto.reason
    manual, _ = run_georeference(tiny_image, three, tmp_path, min_gcps=3)  # QGIS: 3 is allowed
    assert manual.ok and manual.fit.redundant == 0


# ------------------------------------------------------------------------------------------
# Automatic GCPs: label logic (pure) and grid detection (real OpenCV on a synthetic image)
# ------------------------------------------------------------------------------------------


def test_resolve_value_handles_abbreviated_labels():
    window = (680_000.0, 690_000.0)
    assert resolve_value("682400", *window) == 682_400
    assert resolve_value("6824", *window) == 682_400  # hundreds of metres
    assert resolve_value("682", *window) == 682_000  # kilometres
    assert resolve_value("123456", *window) is None  # misread: outside the AOI's neighbourhood
    assert resolve_value("", *window) is None


def labels_for(vertical: dict[float, float], horizontal: dict[float, float]) -> list[LineLabel]:
    return [LineLabel("x", p, v, 0.9, str(int(v))) for p, v in vertical.items()] + [
        LineLabel("y", p, v, 0.9, str(int(v))) for p, v in horizontal.items()
    ]


def test_assemble_gcps_crosses_every_line_pair():
    labels = labels_for({100: 682_400, 400: 682_500, 700: 682_600}, {50: 5_613_200, 350: 5_613_100})
    gcps, notes = assemble_gcps(labels, tolerance_m=15)
    assert len(gcps) == 6 and not notes
    assert Gcp(400, 350, 682_500, 5_613_100, "682500/5613100") in gcps


def test_assemble_gcps_drops_a_misread_label():
    # 682900 is OCR'd for the third line where 682600 belongs: off the line the others define.
    labels = labels_for({100: 682_400, 400: 682_500, 700: 682_900, 1000: 682_700}, {50: 5_613_200, 350: 5_613_100})
    gcps, notes = assemble_gcps(labels, tolerance_m=15)
    assert any("dropped 1 inconsistent x label" in n for n in notes)
    assert sorted({g.col for g in gcps}) == [100, 400, 1000]


def test_assemble_gcps_needs_two_lines_each_way():
    gcps, notes = assemble_gcps(labels_for({100: 682_400, 400: 682_500}, {50: 5_613_200}), 15)
    assert gcps == [] and any("at least 2 of each" in n for n in notes)


def test_a_line_labelled_at_both_ends_keeps_the_better_reading():
    labels = [
        LineLabel("x", 100, 682_400, 0.4, "x"), LineLabel("x", 100, 682_400, 0.9, "x"),
        LineLabel("x", 400, 682_500, 0.9, "x"), LineLabel("y", 50, 5_613_200, 0.9, "y"),
        LineLabel("y", 350, 5_613_100, 0.9, "y"),
    ]  # fmt: skip
    gcps, _ = assemble_gcps(labels, 15)
    assert len(gcps) == 4


def test_detect_grid_lines_on_a_synthetic_map(tmp_path):
    img = np.full((800, 1200, 3), 255, np.uint8)
    for x in (200, 500, 800):  # long vertical grid lines
        cv2.line(img, (x, 60), (x, 740), (120, 0, 0), 2)
    for y in (150, 400, 650):  # long horizontal grid lines
        cv2.line(img, (60, y), (1140, y), (120, 0, 0), 2)
    cv2.line(img, (900, 100), (900, 160), (0, 0, 0), 2)  # a short stroke: not a grid line
    cv2.putText(img, "682400E", (150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)  # text: ignored
    path = tmp_path / "grid.png"
    cv2.imwrite(str(path), img)

    grid = detect_grid_lines(path)
    # positions are line centres in continuous pixel coordinates (a 2 px stroke at 200 spans 199-201)
    assert [line.pos for line in grid.vertical] == pytest.approx([200.5, 500.5, 800.5], abs=1.1)
    assert [line.pos for line in grid.horizontal] == pytest.approx([150.5, 400.5, 650.5], abs=1.1)
    assert (grid.width, grid.height) == (1200, 800)
    assert grid.vertical[0].start <= 62 and grid.vertical[0].end >= 738


# ------------------------------------------------------------------------------------------
# One map page end to end (OpenCV detection and EasyOCR mocked; the fit and warp are real)
# ------------------------------------------------------------------------------------------

from gossan.core.raster import raster_path


@pytest.fixture
def project(aoi_file, fixtures_dir) -> Project:
    assert runner.invoke(app, ["new", "demo", "--aoi", str(aoi_file)]).exit_code == 0
    proj = Project(Path("demo"))
    dest = proj.raw_dir / "aris_pdfs" / "ARIS_00123.pdf"
    dest.parent.mkdir(parents=True)
    dest.write_bytes((fixtures_dir / "aris_sample.pdf").read_bytes())  # page 3 is a landscape page
    return proj


def aoi_corner(project: Project) -> tuple[float, float]:
    minx, _, _, maxy = project.read_aoi().to_crs("EPSG:26910").total_bounds
    return minx, maxy


def make_points_text(project: Project, jitter: dict[int, float] | None = None) -> str:
    """A .points file for the project's AOI: 0.5 m per pixel, a 3 x 3 grid of points."""
    e0, n0 = aoi_corner(project)
    rows = ["#CRS: EPSG:26910", "mapX,mapY,sourceX,sourceY,enable,dX,dY,residual"]
    for i, (col, row) in enumerate((c, r) for r in (300, 800, 1300) for c in (400, 1000, 1600)):
        dx = (jitter or {}).get(i, 0.0)
        rows.append(f"{e0 + 0.5 * col + dx:.2f},{n0 - 0.5 * row:.2f},{col},{-row},1,0,0,0")
    return "\n".join(rows) + "\n"


def mock_grid(monkeypatch, project: Project, n=3):
    """Pretend OpenCV found an n x n grid and EasyOCR read its labels (0.5 m per pixel)."""
    e0, n0 = aoi_corner(project)
    cols, rows = [400 + 600 * i for i in range(n)], [300 + 500 * i for i in range(n)]
    grid = GridLines(
        [GridLine(c, 100, 1600) for c in cols], [GridLine(r, 100, 2100) for r in rows], 2200, 1700
    )
    labels = labels_for(
        {c: round(e0 + 0.5 * c) for c in cols}, {r: round(n0 - 0.5 * r) for r in rows}
    )
    monkeypatch.setattr(georef, "detect_grid_lines", lambda image: grid)
    monkeypatch.setattr(georef, "read_labels", lambda *a, **k: labels)
    monkeypatch.setattr(georef, "load_reader", lambda: object())


def test_auto_georeference_success_queues_the_map_for_review(project, monkeypatch):
    mock_grid(monkeypatch, project)
    pdf = project.raw_dir / "aris_pdfs" / "ARIS_00123.pdf"
    res = georef.georeference_map_page(project, 123, pdf, 3)
    assert res.ok and res.method == "auto" and res.n_gcps == 9
    assert res.rms_m is not None and res.rms_m < 1

    tif = raster_path(project, "aris_00123_p3")
    assert res.output == tif and tif.exists()
    with rasterio.open(tif) as ds:
        assert ds.crs.to_epsg() == 32610
        aoi = project.read_aoi().total_bounds
        assert abs(ds.bounds.left - aoi[0]) < 5  # the map's corner sits on the AOI corner
    assert (project.root / "digitize" / "maps" / "00123_p3.png").exists()

    (item,) = review.read_queue(project)
    assert (item.status, item.type, item.report, item.page) == ("PENDING", "map", "123", "3")
    assert item.source == "rasters/aris_00123_p3.tif"
    assert item.image == "digitize/maps/00123_p3.png"
    assert "9 GCPs" in item.note and "check placement" in item.note

    rec = next(r for r in project.provenance.read() if r.operation == "georef_auto")
    assert rec.parameters["n_gcps"] == 9 and rec.parameters["reviewed"] is False
    assert rec.parameters["gcp_crs"] == "EPSG:26910" and rec.parameters["rms_threshold_m"] == 15.0
    assert "Access Only" in rec.license


def test_auto_failure_saves_candidate_points_and_queues_nothing(project, monkeypatch):
    mock_grid(monkeypatch, project)
    project.config.georef.min_gcps = 10  # the 3 x 3 grid gives 9
    res = georef.georeference_map_page(project, 123, project.raw_dir / "aris_pdfs" / "ARIS_00123.pdf", 3)
    assert not res.ok and "at least 10" in res.reason
    assert res.candidate_points == project.root / "digitize" / "gcps" / "00123_p3_auto.points"
    assert len(parse_points(res.candidate_points).gcps) == 9  # ready to correct in QGIS
    assert not raster_path(project, "aris_00123_p3").exists()
    assert review.read_queue(project) == []
    rec = next(r for r in project.provenance.read() if r.operation == "georef_failed")
    assert rec.parameters["method"] == "auto" and "at least 10" in rec.parameters["reason"]
    assert rec.parameters["candidate_points"] == "digitize/gcps/00123_p3_auto.points"


def test_auto_without_a_readable_grid_or_ocr(project, monkeypatch):
    pdf = project.raw_dir / "aris_pdfs" / "ARIS_00123.pdf"
    res = georef.georeference_map_page(project, 123, pdf, 3)  # real OpenCV: a text page has no grid
    assert not res.ok and "no GCPs" in res.reason and any("grid lines found: 0" in n for n in res.notes)

    mock_grid(monkeypatch, project)  # grid found, but EasyOCR is not installed

    def no_ocr():
        raise georef.OcrUnavailableError("EasyOCR is not installed")

    monkeypatch.setattr(georef, "load_reader", no_ocr)
    res = georef.georeference_map_page(project, 123, pdf, 3)
    assert not res.ok and any("cannot read coordinate labels" in n for n in res.notes)
    assert review.read_queue(project) == []


def test_gcp_crs_follows_datum_and_zone(project):
    assert georef.gcp_crs_for(project) == "EPSG:26910"
    project.config.georef.datum = "NAD27"
    assert georef.gcp_crs_for(project) == "EPSG:26710"
    project.config.georef.gcp_crs = "EPSG:26911"
    assert georef.gcp_crs_for(project) == "EPSG:26911"
    project.config.georef.gcp_crs = None
    project.config.utm_zone = None
    with pytest.raises(georef.GeorefError, match="gcp_crs"):
        georef.gcp_crs_for(project)


def test_rms_threshold_is_part_of_the_default_project_yaml(project):
    text = (project.root / "project.yaml").read_text()
    assert "georef:" in text and "rms_threshold_m: 15.0" in text
    assert Project(project.root).config.georef.rms_threshold_m == 15.0


# ------------------------------------------------------------------------------------------
# CLI: ingest-points (the QGIS fallback) and digitize integration
# ------------------------------------------------------------------------------------------


def test_ingest_points_georeferences_from_a_qgis_file(project, tmp_path):
    points = tmp_path / "p3.points"
    points.write_text(make_points_text(project))
    result = runner.invoke(app, ["ingest-points", "demo", "123", "3", str(points)])
    assert result.exit_code == 0, result.output
    assert "9 GCPs" in result.output and "queued for review" in result.output

    assert raster_path(project, "aris_00123_p3").exists()
    (item,) = review.read_queue(project)
    assert (item.status, item.type, item.page) == ("PENDING", "map", "3")
    assert item.source == "rasters/aris_00123_p3.tif" and item.image == "digitize/maps/00123_p3.png"
    assert "from QGIS points" in item.note
    rec = next(r for r in project.provenance.read() if r.operation == "georef_manual")
    assert rec.source.startswith("QGIS Georeferencer") and rec.parameters["gcp_crs"] == "EPSG:26910"


def test_ingest_points_rejects_a_bad_fit_unless_the_threshold_is_raised(project, tmp_path):
    points = tmp_path / "bad.points"
    points.write_text(make_points_text(project, jitter={4: 60.0, 2: -40.0}))
    result = runner.invoke(app, ["ingest-points", "demo", "123", "3", str(points)])
    assert result.exit_code == 1 and "exceeds the 15 m threshold" in result.output
    assert not raster_path(project, "aris_00123_p3").exists()
    assert review.read_queue(project) == []
    assert any(r.operation == "georef_failed" for r in project.provenance.read())

    ok = runner.invoke(app, ["ingest-points", "demo", "123", "3", str(points), "--rms-threshold", "100"])
    assert ok.exit_code == 0, ok.output
    rec = next(r for r in project.provenance.read() if r.operation == "georef_manual")
    assert rec.parameters["rms_threshold_m"] == 100.0  # the override is on the record
    assert len(review.read_queue(project)) == 1


def test_ingest_points_uses_a_supplied_image_and_the_projects_datum_without_a_crs_line(project, tmp_path, tiny_image):
    # No #CRS line: the project's UTM zone in georef.datum applies. The image is the user's own.
    e0, n0 = aoi_corner(project)
    lines = ["mapX,mapY,sourceX,sourceY"]
    lines += [f"{e0 + 2.0 * c},{n0 - 2.0 * r},{c},{-r}" for r in (0, 40, 80) for c in (0, 50, 100)]
    points = tmp_path / "x.points"
    points.write_text("\n".join(lines))
    result = runner.invoke(app, ["ingest-points", "demo", "123", "3", str(points), "--image", str(tiny_image)])
    assert result.exit_code == 0, result.output
    (item,) = review.read_queue(project)
    assert item.image.endswith("map.tif")  # the image the points were placed on, not a new render
    assert not (project.root / "digitize" / "maps").exists()


def test_ingest_points_errors(project, tmp_path):
    points = tmp_path / "p.points"
    points.write_text(make_points_text(project))
    assert runner.invoke(app, ["ingest-points", "demo", "123", "99", str(points)]).exit_code == 1  # no page 99
    assert runner.invoke(app, ["ingest-points", "demo", "555", "3", str(points)]).exit_code == 1  # no such PDF
    short = tmp_path / "short.points"
    short.write_text("mapX,mapY,sourceX,sourceY\n1,2,3,-4\n")
    result = runner.invoke(app, ["ingest-points", "demo", "123", "3", str(short)])
    assert result.exit_code == 1 and "at least 3" in result.output
    assert review.read_queue(project) == []


def test_digitize_georeferences_map_pages_and_queues_them(project, monkeypatch):
    mock_grid(monkeypatch, project)
    result = runner.invoke(app, ["digitize", "demo", "--report", "123", "--target-pages", "3"])
    assert result.exit_code == 0, result.output
    assert "map p.3: georeferenced, 9 GCPs" in result.output
    assert "georeferencing: 1 done" in result.output
    assert [i.page for i in review.read_queue(project)] == ["3"]  # page 4 was out of scope

    again = runner.invoke(app, ["digitize", "demo", "--report", "123", "--target-pages", "3"])
    assert again.exit_code == 0 and "1 already done" in again.output
    assert len(review.read_queue(project)) == 1  # a finished map is not redone or re-queued


def test_digitize_reports_the_manual_route_when_auto_fails(project):
    result = runner.invoke(app, ["digitize", "demo", "--report", "123"])  # pages 3-4 are 'maps'
    assert result.exit_code == 0, result.output  # an expected fallback is not a failure
    assert "map p.3: not georeferenced" in result.output
    assert "gossan ingest-points" in result.output and "digitize/maps/00123_p3.png" in result.output
    assert review.read_queue(project) == []

    off = runner.invoke(app, ["digitize", "demo", "--report", "123", "--no-georef"])
    assert "map p." not in off.output


def test_opencv_and_rasterio_are_declared_dependencies():
    import tomllib

    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    names = " ".join(tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["dependencies"])
    assert "opencv-python-headless" in names and "rasterio" in names  # GDAL comes with rasterio
