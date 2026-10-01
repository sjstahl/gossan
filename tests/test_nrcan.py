from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
import rasterio
from typer.testing import CliRunner

from gossan.cli import app
from gossan.core.project import Project
from gossan.harvest.national.nrcan_geophysics import NrcanGeophysicsConnector

runner = CliRunner()
ENDPOINT = "https://wcs.example.test/ows"


@pytest.fixture
def project(aoi_file) -> Project:
    assert runner.invoke(app, ["new", "demo", "--aoi", str(aoi_file)]).exit_code == 0
    return Project(Path("demo"))


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("GOSSAN_NRCAN_WCS_URL", "GOSSAN_NRCAN_MAG_COVERAGE", "GOSSAN_NRCAN_GRAV_COVERAGE"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("GOSSAN_NRCAN_WCS_URL", ENDPOINT)
    monkeypatch.setenv("GOSSAN_NRCAN_MAG_COVERAGE", "mag_residual_200m")


@pytest.fixture
def wcs(monkeypatch, fixtures_dir):
    calls: list[str] = []
    body = {"data": (fixtures_dir / "grid_wgs84.tif").read_bytes()}

    def fake_get(self, url):
        calls.append(url)
        return body["data"]

    monkeypatch.setattr(NrcanGeophysicsConnector, "_get", fake_get)
    return calls, body


def test_not_configured_prints_manual_instructions(project):
    result = runner.invoke(app, ["harvest", "demo", "--source", "nrcan_geophysics"])
    assert result.exit_code == 1
    assert "GOSSAN_NRCAN_WCS_URL is not set" in result.output
    assert "gossan ingest" in result.output
    assert "open.canada.ca/data/en/dataset/752fe3fc" in result.output  # magnetics record
    assert "open.canada.ca/data/en/dataset/5a4e46fe" in result.output  # gravity record
    assert "Open Government Licence - Canada" in result.output
    assert not (project.root / "rasters").exists()


def test_plain_harvest_leaves_nrcan_out(project, wfs):
    result = runner.invoke(app, ["harvest", "demo"])
    assert result.exit_code == 0, result.output
    assert "nrcan" not in result.output


def test_wcs_fetch_writes_cog_and_provenance(project, configured, wcs):
    calls, _ = wcs
    result = runner.invoke(app, ["harvest", "demo", "--source", "nrcan_geophysics"])
    assert result.exit_code == 0, result.output
    assert "1 raster(s) written to rasters/" in result.output

    # One request (gravity has no coverage configured), subset to the AOI in WGS84.
    (url,) = calls
    query = parse_qs(urlparse(url).query)
    assert url.startswith(ENDPOINT + "?")
    assert query["request"] == ["GetCoverage"] and query["version"] == ["2.0.1"]
    assert query["coverageId"] == ["mag_residual_200m"]
    assert query["subset"] == ["Long(-120.320000,-120.280000)", "Lat(50.700000,50.730000)"]

    with rasterio.open(project.root / "rasters" / "nrcan_magnetics.tif") as ds:
        assert ds.crs.to_epsg() == 32610
        assert ds.tags(ns="IMAGE_STRUCTURE")["LAYOUT"] == "COG"
    assert len(list((project.raw_dir / "nrcan_geophysics").glob("nrcan_magnetics_*.tif"))) == 1

    records = project.provenance.read()
    fetch = next(r for r in records if r.operation == "fetch")
    assert fetch.dataset == "nrcan_magnetics"
    assert fetch.identifier == "mag_residual_200m"
    assert fetch.license == "Open Government Licence - Canada"
    out = next(r for r in records if r.operation == "write_raster")
    assert out.identifier == "rasters/nrcan_magnetics.tif"
    assert out.parameters["project_crs"] == "EPSG:32610"


def test_service_returning_xml_exception_falls_back(project, configured, wcs):
    _, body = wcs
    body["data"] = b'<?xml version="1.0"?><ows:ExceptionReport>NoSuchCoverage</ows:ExceptionReport>'
    result = runner.invoke(app, ["harvest", "demo", "--source", "nrcan_geophysics"])
    assert result.exit_code == 1
    assert "did not return a GeoTIFF" in result.output
    assert "NoSuchCoverage" in result.output
    assert "gossan ingest" in result.output
    assert not (project.root / "rasters").exists()
    assert not list((project.raw_dir / "nrcan_geophysics").glob("*"))  # nothing kept


def test_unreachable_service_falls_back(project, configured, monkeypatch):
    def boom(self, url):
        raise OSError("timed out")

    monkeypatch.setattr(NrcanGeophysicsConnector, "_get", boom)
    result = runner.invoke(app, ["harvest", "demo", "--source", "nrcan_geophysics"])
    assert result.exit_code == 1
    assert "timed out" in result.output
    assert "gossan ingest" in result.output


def test_url_without_coverages_falls_back(project, monkeypatch):
    monkeypatch.setenv("GOSSAN_NRCAN_WCS_URL", ENDPOINT)
    result = runner.invoke(app, ["harvest", "demo", "--source", "nrcan_geophysics"])
    assert result.exit_code == 1
    assert "no coverage ids are configured" in result.output
