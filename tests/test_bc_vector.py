from pathlib import Path
from urllib.parse import parse_qs, urlparse

import geopandas as gpd
import pyogrio
import pytest
from typer.testing import CliRunner

from gossan.cli import app
from gossan.core.project import Project
from gossan.harvest.bc.wfs import WfsConnector

runner = CliRunner()


@pytest.fixture
def project(aoi_file) -> Project:
    assert runner.invoke(app, ["new", "demo", "--aoi", str(aoi_file)]).exit_code == 0
    return Project(Path("demo"))


def layer_names(project: Project) -> set[str]:
    return set(pyogrio.list_layers(project.data_gpkg)[:, 0])


def test_harvest_mto_tenure(project, wfs):
    result = runner.invoke(app, ["harvest", "demo", "--source", "mto_tenure"])
    assert result.exit_code == 0, result.output

    query = parse_qs(urlparse(wfs.calls[0]).query)
    assert query["typeNames"] == ["pub:WHSE_MINERAL_TENURE.MTA_ACQUIRED_TENURE_SVW"]
    # This layer's geometry property is GEOMETRY, not SHAPE.
    assert query["CQL_FILTER"][0].startswith("BBOX(GEOMETRY,")

    gdf = gpd.read_file(project.data_gpkg, layer="mto_tenure")
    assert gdf.crs.to_epsg() == 32610
    # Whole polygons that intersect the AOI are kept (inside + straddling); the far one is dropped.
    assert sorted(gdf["TENURE_NUMBER_ID"]) == [100001, 100002]
    assert set(gdf["CLAIM_NAME"]) == {"INSIDE CLAIM", "STRADDLING CLAIM"}
    assert (gdf.geom_type == "Polygon").all()

    assert len(list((project.raw_dir / "mto_tenure").glob("mto_tenure_*.geojson"))) == 1
    fetch = next(r for r in project.provenance.read() if r.operation == "fetch")
    assert fetch.dataset == "mto_tenure"
    assert fetch.identifier == "pub:WHSE_MINERAL_TENURE.MTA_ACQUIRED_TENURE_SVW"
    assert "Open Government Licence" in fetch.license


def test_harvest_aris_index(project, wfs):
    result = runner.invoke(app, ["harvest", "demo", "--source", "aris_index"])
    assert result.exit_code == 0, result.output

    query = parse_qs(urlparse(wfs.calls[0]).query)
    assert query["typeNames"] == ["pub:WHSE_MINERAL_TENURE.ARIS_MINERAL_REPORTS"]
    assert len(wfs.calls) == 1  # index only: no report PDF requests

    gdf = gpd.read_file(project.data_gpkg, layer="aris_index")
    assert gdf.crs.to_epsg() == 32610
    assert sorted(gdf["ARIS_NUMBER"]) == [724, 8038]
    assert "Reg" in set(gdf["PROPERTY_NAME"])
    prefix = "https://apps.nrs.gov.bc.ca/pub/aris/detail/"
    assert all(u.startswith(prefix) for u in gdf["ARIS_REPORT_URL"])

    assert len(list((project.raw_dir / "aris_index").glob("aris_index_*.geojson"))) == 1
    fetch = next(r for r in project.provenance.read() if r.operation == "fetch")
    assert fetch.dataset == "aris_index"
    assert "Access Only" in fetch.license


def test_harvest_defaults_to_all_sources_in_one_gpkg(project, wfs):
    result = runner.invoke(app, ["harvest", "demo"])
    assert result.exit_code == 0, result.output
    assert layer_names(project) == {"minfile", "mto_tenure", "aris_index"}  # layers coexist
    datasets = [r.dataset for r in project.provenance.read() if r.operation == "fetch"]
    assert sorted(datasets) == ["aris_index", "minfile", "mto_tenure"]


def test_harvest_empty_result_writes_no_layer(project, wfs):
    wfs.responses["pub:WHSE_MINERAL_TENURE.MTA_ACQUIRED_TENURE_SVW"] = "empty_wfs.geojson"
    result = runner.invoke(app, ["harvest", "demo", "--source", "mto_tenure"])
    assert result.exit_code == 0, result.output
    assert "0 feature(s)" in result.output
    assert not project.data_gpkg.exists()
    write = next(r for r in project.provenance.read() if r.operation == "write_layer")
    assert write.parameters["n_features"] == 0


def test_harvest_network_failure_points_to_ingest(project, monkeypatch):
    def boom(self, url):
        raise OSError("network down")

    monkeypatch.setattr(WfsConnector, "_get", boom)
    result = runner.invoke(app, ["harvest", "demo", "--source", "aris_index"])
    assert result.exit_code == 1
    assert "gossan ingest" in result.output
