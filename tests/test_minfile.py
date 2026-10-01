from pathlib import Path
from urllib.parse import parse_qs, urlparse

import geopandas as gpd
from typer.testing import CliRunner

from gossan.cli import app
from gossan.core.project import Project

runner = CliRunner()


def make_project(aoi_file) -> Project:
    result = runner.invoke(app, ["new", "demo", "--aoi", str(aoi_file)])
    assert result.exit_code == 0, result.output
    return Project(Path("demo"))


def test_harvest_minfile(aoi_file, wfs_calls):
    project = make_project(aoi_file)
    result = runner.invoke(app, ["harvest", "demo", "--source", "minfile"])
    assert result.exit_code == 0, result.output

    # Request: right layer, bbox filter around the AOI, no live network.
    assert len(wfs_calls) == 1
    query = parse_qs(urlparse(wfs_calls[0]).query)
    assert query["typeNames"] == ["pub:WHSE_MINERAL_TENURE.MINFIL_MINERAL_FILE"]
    assert query["CQL_FILTER"][0].startswith("BBOX(SHAPE,-120.320000,50.700000")

    # Layer: project CRS, clipped to the AOI (the fixture's outside point is dropped), text stripped.
    gdf = gpd.read_file(project.data_gpkg, layer="minfile")
    assert gdf.crs.to_epsg() == 32610
    assert sorted(gdf["MINFILE_NUMBER"]) == ["092INE900", "092INE901"]
    assert "TEST ALPHA" in set(gdf["MINFILE_NAME1"])

    # Raw original kept; provenance logged for the fetch and the layer write.
    assert len(list((project.raw_dir / "minfile").glob("*.geojson"))) == 1
    records = project.provenance.read()
    fetch = next(r for r in records if r.operation == "fetch")
    assert fetch.dataset == "minfile"
    assert "Open Government Licence" in fetch.license
    assert fetch.parameters["n_features"] == 3
    write = next(r for r in records if r.operation == "write_layer")
    assert write.parameters["n_features"] == 2


def test_harvest_twice_keeps_all_originals(aoi_file, wfs_calls, monkeypatch):
    project = make_project(aoi_file)
    runner.invoke(app, ["harvest", "demo"])
    first = list((project.raw_dir / "minfile").glob("*.geojson"))
    import time

    time.sleep(1.1)  # file names carry a one-second timestamp
    runner.invoke(app, ["harvest", "demo"])
    assert len(list((project.raw_dir / "minfile").glob("*.geojson"))) == len(first) + 1


def test_harvest_unknown_source(aoi_file):
    make_project(aoi_file)
    result = runner.invoke(app, ["harvest", "demo", "--source", "nope"])
    assert result.exit_code != 0


def test_harvest_not_a_project(tmp_path):
    result = runner.invoke(app, ["harvest", str(tmp_path)])
    assert result.exit_code == 1
