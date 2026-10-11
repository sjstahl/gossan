"""Regional magnetics end to end: grids ingested in the formats geophysics arrives in, turned into
derivative features, used by the model and shown in the report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pymupdf
import pytest
import rasterio
from pyproj import Transformer
from rasterio.shutil import copy as rio_copy
from rasterio.transform import from_origin
from synthetic import CRS, X0, Y0, build_project, runner, small_cfg

from gossan.cli import app
from gossan.core.project import Project
from gossan.target.features import build_stack, find_magnetic_raster, input_datasets
from gossan.target.grid import build_grid
from gossan.target.run import run_target

DX = 100.0
MARGIN = 3000.0
#: The grid covers the AOI (4 x 3 km) plus a 3 km margin on every side.
W, H = 4000.0 + 2 * MARGIN, 3000.0 + 2 * MARGIN
LEFT, TOP = X0 - MARGIN, Y0 + 3000.0 + MARGIN
SOURCES = [(X0 + 3000.0, Y0 + 1000.0), (X0 + 3300.0, Y0 + 2100.0)]  # magnetic bodies in the east, where the occurrences are


def synthetic_field() -> np.ndarray:
    """A regional gradient plus the vertical field of two buried dipoles at ``SOURCES``."""
    ncol, nrow = int(W / DX), int(H / DX)
    xs = LEFT + (np.arange(ncol) + 0.5) * DX
    ys = TOP - (np.arange(nrow) + 0.5) * DX
    xx, yy = np.meshgrid(xs, ys)
    field = 55_000.0 + 0.01 * (xx - X0)  # a gentle regional trend, nT
    for sx, sy in SOURCES:
        r2 = (xx - sx) ** 2 + (yy - sy) ** 2
        field = field + 4e9 * (2 * 300.0**2 - r2) / (r2 + 300.0**2) ** 2.5
    return field.astype(np.float32)


def write_grid(path: Path, field: np.ndarray, *, crs: str | None = CRS, driver: str = "GTiff") -> Path:
    profile = {"driver": driver, "width": field.shape[1], "height": field.shape[0], "count": 1, "dtype": "float32", "transform": from_origin(LEFT, TOP, DX, DX), "nodata": -99999.0}
    if crs:
        profile["crs"] = crs
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(field, 1)
    return path


@pytest.fixture
def project(tmp_path) -> Project:
    proj = build_project(tmp_path, n_occ=60, seed=4, raster=False)  # no stand-in raster: magnetics are the only grid
    proj.config.target = small_cfg(effort="ignore")
    proj.config.save(proj.root)
    return Project(proj.root)


def ingest(project: Project, grid: Path, *extra: str):
    return runner.invoke(app, ["ingest", "demo", str(grid), "--layer", "magnetics", *extra])


def stack_for(project: Project, **cfg):
    project.config.target = small_cfg(effort="ignore", **cfg)
    project.config.save(project.root)
    proj = Project(project.root)
    return build_stack(proj, build_grid(proj.read_aoi(), CRS, 100.0, 10**6), proj.config.target), proj


# --- ingest ---------------------------------------------------------------------------------------------------------------


def test_a_grid_without_a_crs_needs_one_and_a_margin_is_kept(project, tmp_path):
    grid = write_grid(tmp_path / "mag_nocrs.tif", synthetic_field(), crs=None)
    failed = ingest(project, grid)
    assert failed.exit_code == 1 and "has no CRS" in failed.output and "--crs" in failed.output
    assert ingest(project, grid, "--crs", CRS, "--buffer-m", "1500").exit_code == 0
    with rasterio.open(project.root / "rasters" / "magnetics.tif") as ds:
        b = ds.bounds
    assert b.left <= X0 - 1400 and b.right >= X0 + 4000 + 1400 and b.bottom <= Y0 - 1400  # AOI plus about 1.5 km each way
    assert b.left >= X0 - 1700  # but not the whole grid
    record = next(r for r in project.provenance.read() if r.operation == "manual_ingest")
    assert record.parameters["aoi_buffer_m"] == 1500.0 and record.parameters["crs_assigned"] == CRS


def test_ascii_and_er_mapper_grids_load_and_keep_their_companion_files(project, tmp_path):
    src = write_grid(tmp_path / "mag_src.tif", synthetic_field())
    asc = tmp_path / "grid.asc"
    rio_copy(src, asc, driver="AAIGrid")
    assert asc.with_suffix(".prj").exists()  # the CRS travels in a sidecar
    ers = tmp_path / "grid.ers"
    rio_copy(src, ers, driver="ERS")
    for grid in (asc, ers):
        result = ingest(project, grid, "--overwrite", "--buffer-m", "500")
        assert result.exit_code == 0, f"{grid.name}: {result.output}"
        with rasterio.open(project.root / "rasters" / "magnetics.tif") as ds:
            assert ds.crs.to_epsg() == 32610 and np.isfinite(ds.read(1, masked=True).filled(np.nan)).any()
    raw = sorted(p for d in (project.raw_dir / "ingest" / "magnetics").iterdir() for p in d.iterdir())
    names = {p.name for p in raw}
    assert {"grid.asc", "grid.prj", "grid.ers", "grid"} <= names  # the ER Mapper data file has no extension


def test_an_unreadable_grid_says_what_to_do(project, tmp_path):
    fake = tmp_path / "geosoft.grd"
    fake.write_bytes(b"this is not a grid")
    result = ingest(project, fake)
    assert result.exit_code == 1 and "cannot read geosoft.grd" in result.output and "export it to GeoTIFF" in result.output
    assert runner.invoke(app, ["ingest", "demo", str(tmp_path / "points.geojson"), "--layer", "x", "--buffer-m", "10"]).exit_code != 0


def test_buffer_applies_to_rasters_only(project, tmp_path):
    vector = tmp_path / "pts.geojson"
    vector.write_text('{"type":"FeatureCollection","features":[]}')
    result = runner.invoke(app, ["ingest", "demo", str(vector), "--layer", "pts", "--buffer-m", "100"])
    assert result.exit_code == 1 and "raster grids only" in result.output


# --- features ------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def with_grid(project, tmp_path) -> Project:
    write_grid(tmp_path / "mag.tif", synthetic_field())
    assert ingest(project, tmp_path / "mag.tif", "--buffer-m", "2500").exit_code == 0
    return project


def test_derivatives_become_features_and_replace_the_bare_field(with_grid):
    stack, proj = stack_for(with_grid)
    mags = [n for n in stack.names if n.startswith("mag_")]
    assert mags == ["mag_field", "mag_local", "mag_vd1", "mag_as", "mag_tilt", "mag_texture"]
    assert "magnetics" not in stack.names  # the bare raster is not a second, duplicate feature
    assert stack.magnetic_dataset == "magnetics" and "magnetics" in input_datasets(stack)
    grid = build_grid(proj.read_aoi(), CRS, 100.0, 10**6)
    cell = int(grid.cell_of(np.array([SOURCES[0][0]]), np.array([SOURCES[0][1]]))[0])
    values = dict(zip(stack.names, stack.X[cell], strict=True))
    assert values["mag_tilt"] > 60 and values["mag_vd1"] > 0  # on top of a source
    away = int(grid.cell_of(np.array([X0 + 500.0]), np.array([Y0 + 1500.0]))[0])
    assert dict(zip(stack.names, stack.X[away], strict=True))["mag_as"] < 0.5 * values["mag_as"]  # quiet away from it
    assert values["mag_field"] > 55_000 + 0.01 * 3000  # the field itself is passed through


def test_modes_and_the_raster_name_are_configurable(with_grid):
    field_only, _ = stack_for(with_grid, magnetics="field")
    assert "magnetics" in field_only.names and not any(n.startswith("mag_") for n in field_only.names)  # the bare grid, as any raster
    off, _ = stack_for(with_grid, magnetics="off")
    assert not any(n.startswith("mag_") for n in off.names)
    renamed = with_grid.root / "rasters" / "my_tmi.tif"
    (with_grid.root / "rasters" / "magnetics.tif").rename(renamed)
    assert find_magnetic_raster(with_grid, with_grid.config.target.model_copy(update={"magnetic_raster": "my_tmi"})) == renamed
    named, _ = stack_for(with_grid, magnetic_raster="my_tmi")
    assert "mag_tilt" in named.names and named.magnetic_dataset == "my_tmi"


def test_other_usual_names_are_found_and_no_grid_says_where_to_get_one(project, tmp_path):
    stack, _ = stack_for(project)
    assert "nrcan_geophysics" in stack.skipped["magnetics"] and "gossan ingest" in stack.skipped["magnetics"]
    write_grid(tmp_path / "t.tif", synthetic_field())
    runner.invoke(app, ["ingest", "demo", str(tmp_path / "t.tif"), "--layer", "rtf", "--buffer-m", "2500"])
    stack, _ = stack_for(project)
    assert "mag_tilt" in stack.names and stack.magnetic_dataset == "rtf"


def test_a_geographic_grid_cannot_be_differentiated_and_says_so(with_grid):
    rasters = with_grid.root / "rasters"
    (rasters / "magnetics.tif").unlink()
    lon0, lat0 = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform(X0, Y0)  # a real geographic grid over the AOI
    field = synthetic_field()[:60, :60]
    profile = {"driver": "GTiff", "width": 60, "height": 60, "count": 1, "dtype": "float32", "crs": "EPSG:4326", "transform": from_origin(lon0 - 0.05, lat0 + 0.1, 0.002, 0.002)}
    with rasterio.open(rasters / "magnetics.tif", "w", **profile) as ds:
        ds.write(field, 1)
    stack, _ = stack_for(with_grid)
    assert "not in a projected CRS in metres" in stack.skipped["magnetics"]
    assert not any(n.startswith("mag_") for n in stack.names)


def test_a_tiny_grid_is_reported_not_a_crash(project):
    (project.root / "rasters").mkdir(exist_ok=True)
    write_grid(project.root / "rasters" / "magnetics.tif", np.full((5, 5), 55_000.0, dtype=np.float32))
    stack, _ = stack_for(project)
    assert "too small" in stack.skipped["magnetics"]


# --- the model and the report -----------------------------------------------------------------------------------------------------


def test_the_model_uses_the_derivatives_and_finds_the_sources(with_grid):
    project, _ = with_grid, None
    project.config.target = small_cfg(effort="ignore", n_bags=6, folds=3)
    project.config.save(project.root)
    result = run_target(Project(project.root), project.config.target, cross_validate=True)
    importances = result.importances
    assert any(name.startswith("mag_") for name in importances) and sum(v for k, v in importances.items() if k.startswith("mag_")) > 0.15
    assert result.cv["auc_mean"] is not None
    grid_json = json.loads((project.root / "target" / "grid.json").read_text())
    assert "mag_tilt" in grid_json["features_used"]
    record = next(r for r in project.provenance.read() if r.dataset == "target_prospectivity")
    assert "unspecified" in record.license or "Derived" in record.license  # the user-supplied licence flows through


def test_the_report_shows_the_magnetics_and_numbers_its_figures(with_grid):
    assert runner.invoke(app, ["analyze", "demo", "--steps", "occurrences"]).exit_code == 0
    assert runner.invoke(app, ["report", "demo", "--kind", "analysis", "--format", "pdf"]).exit_code == 0
    with pymupdf.open(with_grid.root / "outputs" / "analysis_report.pdf") as pdf:
        text = " ".join(" ".join(page.get_text() for page in pdf).split())
    assert "Regional magnetic field and its tilt angle" in text and "Figure 1." in text
    figures = [int(n) for n in __import__("re").findall(r"Figure (\d+)\.", text)]
    assert figures == sorted(set(figures)) == list(range(1, len(set(figures)) + 1))  # no gaps, no repeats, no "1b"
    assert "Figure 1b" not in text


def test_the_manual_instructions_point_to_the_current_sources(project):
    result = runner.invoke(app, ["harvest", "demo", "--source", "nrcan_geophysics"])
    assert result.exit_code == 1
    for needle in ("p1xpddgswc", "Open File 2024-08", "dapid 140", "--buffer-m 5000", "Geosoft", "tilt angle", "752fe3fc"):
        assert needle in result.output, needle
