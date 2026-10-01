import json
from pathlib import Path

import geopandas as gpd
import pytest
from typer.testing import CliRunner

from gossan.cli import app
from gossan.core.project import Project

runner = CliRunner()


@pytest.fixture
def project(aoi_file) -> Project:
    assert runner.invoke(app, ["new", "demo", "--aoi", str(aoi_file)]).exit_code == 0
    return Project(Path("demo"))


def test_ingest_geojson_clips_and_projects(project, fixtures_dir):
    lines = fixtures_dir / "ingest_lines.geojson"
    result = runner.invoke(app, ["ingest", "demo", str(lines), "--layer", "roads"])
    assert result.exit_code == 0, result.output
    assert "roads: 2 feature(s)" in result.output

    gdf = gpd.read_file(project.data_gpkg, layer="roads")
    assert gdf.crs.to_epsg() == 32610
    assert sorted(gdf["road"]) == ["crossing", "inside"]  # the outside line is dropped
    # The crossing line is cut at the AOI edge: 0.03 deg of latitude is ~3.3 km, not 5.5 km.
    crossing = gdf[gdf["road"] == "crossing"].geometry.iloc[0]
    assert 3000 < crossing.length < 3500
    assert project.read_aoi().union_all().buffer(1).contains(crossing)

    # Original preserved under raw/, and a provenance record written.
    copies = list((project.raw_dir / "ingest" / "roads").glob("*/ingest_lines.geojson"))
    assert len(copies) == 1
    assert copies[0].read_bytes() == lines.read_bytes()
    record = project.provenance.read()[-1]
    assert record.operation == "manual_ingest"
    assert record.dataset == "roads"
    assert record.identifier == "ingest_lines.geojson"
    assert record.parameters["n_read"] == 3
    assert record.parameters["n_written"] == 2
    assert record.parameters["project_crs"] == "EPSG:32610"
    assert len(record.parameters["sha256"]) == 64


def test_ingest_shapefile_in_other_crs(project, fixtures_dir, tmp_path):
    src = gpd.read_file(fixtures_dir / "ingest_lines.geojson").to_crs("EPSG:26910")
    shp = tmp_path / "src" / "roads_nad83.shp"
    shp.parent.mkdir()
    src.to_file(shp)

    result = runner.invoke(
        app,
        [
            "ingest", "demo", str(shp), "--layer", "roads",
            "--source", "Client X", "--license", "Client confidential",
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert gpd.read_file(project.data_gpkg, layer="roads").crs.to_epsg() == 32610

    (raw,) = (project.raw_dir / "ingest" / "roads").iterdir()
    assert {f.suffix for f in raw.iterdir()} >= {".shp", ".shx", ".dbf", ".prj"}  # sidecars kept
    record = project.provenance.read()[-1]
    assert (record.source, record.license) == ("Client X", "Client confidential")
    assert record.parameters["source_crs"].endswith("26910")


def test_ingest_refuses_existing_layer_without_overwrite(project, fixtures_dir):
    lines = str(fixtures_dir / "ingest_lines.geojson")
    assert runner.invoke(app, ["ingest", "demo", lines, "--layer", "roads"]).exit_code == 0
    again = runner.invoke(app, ["ingest", "demo", lines, "--layer", "roads"])
    assert again.exit_code == 1
    assert "--overwrite" in again.output
    forced = runner.invoke(app, ["ingest", "demo", lines, "--layer", "roads", "--overwrite"])
    assert forced.exit_code == 0


def test_ingest_rejects_bad_input(project, fixtures_dir, tmp_path):
    lines = str(fixtures_dir / "ingest_lines.geojson")
    assert runner.invoke(app, ["ingest", "demo", lines, "--layer", "bad name"]).exit_code == 1

    outside = tmp_path / "outside.geojson"
    point = {"type": "Point", "coordinates": [-100.0, 45.0]}
    outside.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [{"type": "Feature", "properties": {}, "geometry": point}],
            }
        )
    )
    result = runner.invoke(app, ["ingest", "demo", str(outside), "--layer", "far"])
    assert result.exit_code == 1
    assert "intersect the AOI" in result.output
    assert not project.data_gpkg.exists()
    assert not (project.raw_dir / "ingest" / "far").exists()


def test_ingest_into_non_project(tmp_path, fixtures_dir):
    lines = fixtures_dir / "ingest_lines.geojson"
    result = runner.invoke(app, ["ingest", str(tmp_path), str(lines), "--layer", "x"])
    assert result.exit_code == 1
