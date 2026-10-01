from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from typer.testing import CliRunner

from gossan.cli import app
from gossan.core.project import Project

runner = CliRunner()


@pytest.fixture
def project(aoi_file) -> Project:
    assert runner.invoke(app, ["new", "demo", "--aoi", str(aoi_file)]).exit_code == 0
    return Project(Path("demo"))


@pytest.fixture
def grid(fixtures_dir) -> str:
    return str(fixtures_dir / "grid_wgs84.tif")


def test_ingest_tif_writes_clipped_projected_cog(project, grid, fixtures_dir):
    result = runner.invoke(app, ["ingest", "demo", grid, "--layer", "magnetics"])
    assert result.exit_code == 0, result.output
    assert "rasters/magnetics.tif" in result.output

    cog = project.root / "rasters" / "magnetics.tif"
    with rasterio.open(cog) as ds:
        assert ds.crs.to_epsg() == 32610
        assert ds.nodata == -9999
        assert ds.tags(ns="IMAGE_STRUCTURE")["LAYOUT"] == "COG"
        # Clipped to the AOI: the source covers 0.08 x 0.07 deg, the AOI 0.04 x 0.03 deg.
        aoi = project.read_aoi()
        minx, miny, maxx, maxy = aoi.total_bounds
        left, bottom, right, top = ds.bounds
        assert (right - left) < (maxx - minx) * 1.2
        assert (top - bottom) < (maxy - miny) * 1.2
        assert left <= minx + 200 and right >= maxx - 200
        data = ds.read(1, masked=True)
        assert data.count() > 0
        assert 0 <= data.min() and data.max() <= 38  # source values are 0..38 (nodata excluded)

    # Original kept, provenance written.
    (copy,) = (project.raw_dir / "ingest" / "magnetics").glob("*/grid_wgs84.tif")
    assert copy.read_bytes() == (fixtures_dir / "grid_wgs84.tif").read_bytes()
    record = project.provenance.read()[-1]
    assert record.operation == "manual_ingest"
    assert record.dataset == "magnetics"
    assert record.parameters["kind"] == "raster"
    assert record.parameters["format"] == "COG"
    assert record.parameters["output"] == "rasters/magnetics.tif"
    assert record.parameters["source_crs"] == "EPSG:4326"
    assert record.parameters["project_crs"] == "EPSG:32610"


def test_ingest_tif_refuses_existing_unless_overwrite(project, grid):
    assert runner.invoke(app, ["ingest", "demo", grid, "--layer", "mag"]).exit_code == 0
    again = runner.invoke(app, ["ingest", "demo", grid, "--layer", "mag"])
    assert again.exit_code == 1 and "--overwrite" in again.output
    assert runner.invoke(app, ["ingest", "demo", grid, "--layer", "mag", "--overwrite"]).exit_code == 0


def write_tif(path: Path, *, crs, bounds, dtype="uint8", nodata=None) -> Path:
    data = np.arange(100, dtype=dtype).reshape(10, 10)
    with rasterio.open(
        path, "w", driver="GTiff", width=10, height=10, count=1, dtype=dtype,
        crs=crs, nodata=nodata, transform=from_bounds(*bounds, 10, 10),
    ) as ds:  # fmt: skip
        ds.write(data, 1)
    return path


def test_ingest_tif_in_utm_without_nodata_gets_one(project, tmp_path):
    # An integer grid already in a UTM CRS (NAD83 zone 10) covering the AOI, no nodata set.
    aoi = project.read_aoi().to_crs("EPSG:26910")
    minx, miny, maxx, maxy = aoi.total_bounds
    src = write_tif(
        tmp_path / "ints.tif", crs="EPSG:26910", bounds=(minx - 500, miny - 500, maxx + 500, maxy + 500)
    )
    result = runner.invoke(app, ["ingest", "demo", str(src), "--layer", "ints"])
    assert result.exit_code == 0, result.output
    with rasterio.open(project.root / "rasters" / "ints.tif") as ds:
        assert ds.crs.to_epsg() == 32610
        assert ds.dtypes[0] == "uint8"  # integer data stay integer (nearest-neighbour)
        assert ds.nodata == 255  # a nodata value was defined


def test_ingest_tif_without_crs_is_rejected(project, tmp_path):
    src = write_tif(tmp_path / "nocrs.tif", crs=None, bounds=(0, 0, 10, 10))
    result = runner.invoke(app, ["ingest", "demo", str(src), "--layer", "x"])
    assert result.exit_code == 1
    assert "no CRS" in result.output
    assert not (project.root / "rasters").exists()
    assert not (project.raw_dir / "ingest" / "x").exists()


def test_ingest_tif_outside_aoi_is_rejected(project, tmp_path):
    src = write_tif(tmp_path / "far.tif", crs="EPSG:4326", bounds=(-100.1, 45.0, -100.0, 45.1))
    result = runner.invoke(app, ["ingest", "demo", str(src), "--layer", "far"])
    assert result.exit_code == 1
    assert "does not overlap the AOI" in result.output
    assert not (project.root / "rasters" / "far.tif").exists()
    assert not (project.raw_dir / "ingest" / "far").exists()
