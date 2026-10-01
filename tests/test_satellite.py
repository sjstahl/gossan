import hashlib
import urllib.error
import urllib.request
from pathlib import Path

import geopandas as gpd
import pytest
from typer.testing import CliRunner

from gossan.cli import app
from gossan.core.project import Project
from gossan.harvest.satellite import stac as stac_module
from gossan.harvest.satellite.stac import (
    SatelliteAuthError,
    StacSatelliteConnector,
    _signature_problem,
)

runner = CliRunner()
EMIT_IDS = [f"EMIT_L2A_RFL_001_2024010{n}T000000_000000{n}_001" for n in (1, 2, 3, 4)]


@pytest.fixture
def project(aoi_file) -> Project:
    assert runner.invoke(app, ["new", "demo", "--aoi", str(aoi_file)]).exit_code == 0
    return Project(Path("demo"))


def harvest(*args: str):
    return runner.invoke(app, ["harvest", "demo", *args])


def set_config(project: Project, **satellite) -> None:
    for key, value in satellite.items():
        setattr(project.config.satellite, key, value)
    project.config.save(project.root)


def scenes(project: Project, name: str) -> gpd.GeoDataFrame:
    return gpd.read_file(project.data_gpkg, layer=f"{name}_scenes")


# --- EMIT: search, catalogue, provenance ---------------------------------------------------


def test_emit_search_catalogues_scenes_best_first_and_downloads_nothing(project, stac):
    result = harvest("--source", "emit")
    assert result.exit_code == 0, result.output
    assert "emit: 4 scene(s) catalogued" in result.output
    assert "4 scene(s) found, 4 after the cloud/date filters" in result.output

    # One search request for the AOI, then the next-page link followed with GET.
    (m1, u1, body), (m2, u2, _) = stac.requests
    assert (m1, u1) == ("POST", "https://cmr.earthdata.nasa.gov/stac/LPCLOUD/search")
    assert body["collections"] == ["EMITL2ARFL_001"] and body["limit"] == 25 and "datetime" not in body
    ring = body["intersects"]["coordinates"][0]
    assert min(x for x, _ in ring) == pytest.approx(-120.32, abs=1e-3) and max(y for _, y in ring) == pytest.approx(50.73, abs=1e-3)
    assert (m2, u2) == ("GET", "https://stac.test/emit/search?page=2")

    gdf = scenes(project, "emit")
    assert gdf.crs.to_epsg() == 32610
    # Most AOI coverage first, then least cloud (no cloud value last), then the half-covering scene.
    assert list(gdf["scene_id"]) == [EMIT_IDS[0], EMIT_IDS[1], EMIT_IDS[3], EMIT_IDS[2]]
    assert list(gdf["aoi_coverage_pct"]) == [100.0, 100.0, 100.0, pytest.approx(50.0, abs=1)]
    assert gdf.loc[2, "cloud_cover"] != gdf.loc[2, "cloud_cover"]  # NaN: the scene with no cloud value is kept
    assert set(gdf["collection"]) == {"EMITL2ARFL_001"}
    assert gdf.loc[0, "stac_url"] == f"https://stac.test/items/{EMIT_IDS[0]}"

    records = project.provenance.read()
    search = next(r for r in records if r.operation == "search")
    assert (search.dataset, search.identifier) == ("emit_scenes", "EMITL2ARFL_001")
    assert search.parameters["n_found"] == 4 and search.url.endswith("/stac/LPCLOUD/search")
    assert "CC0" in search.license
    assert any(r.operation == "write_layer" and r.parameters["n_features"] == 4 for r in records)
    assert not (project.raw_dir / "satellite").exists() and stac.downloads == []


def test_cloud_filter_keeps_scenes_without_a_cloud_value(project, stac):
    set_config(project, max_cloud_percent=30)
    assert harvest("--source", "emit").exit_code == 0
    assert list(scenes(project, "emit")["scene_id"]) == [EMIT_IDS[0], EMIT_IDS[3], EMIT_IDS[2]]  # the 60% scene is gone
    search = next(r for r in Project(project.root).provenance.read() if r.operation == "search")
    assert (search.parameters["n_found"], search.parameters["n_after_filters"]) == (4, 3)


def test_date_range_and_collection_come_from_project_yaml(project, stac):
    set_config(project, datetime="2023-01-01/2024-12-31", emit_collection="EMITL1BRAD_001")
    assert harvest("--source", "emit").exit_code == 0
    body = stac.requests[0][2]
    assert body["datetime"] == "2023-01-01/2024-12-31" and body["collections"] == ["EMITL1BRAD_001"]


def test_satellite_defaults_are_visible_in_project_yaml(project):
    text = (project.root / "project.yaml").read_text()
    assert "satellite:" in text and "max_cloud_percent: 100.0" in text and "emit_collection: EMITL2ARFL_001" in text


def test_a_plain_harvest_never_touches_the_satellite_services(project, stac, wfs):
    assert harvest().exit_code == 0
    assert stac.requests == [] and stac.downloads == []


def test_empty_search_writes_no_layer(project, stac):
    stac.responses["emit"] = ["stac_empty.json"]
    result = harvest("--source", "emit")
    assert result.exit_code == 0 and "emit: 0 scene(s) catalogued" in result.output
    assert not project.data_gpkg.exists()
    search = next(r for r in project.provenance.read() if r.operation == "search")
    assert search.parameters["n_found"] == 0


# --- EMIT: downloads -----------------------------------------------------------------------


def test_limit_scenes_downloads_the_best_scene_with_provenance(project, stac, monkeypatch):
    monkeypatch.setenv("EARTHDATA_TOKEN", "tok-123")
    result = harvest("--source", "emit", "--limit-scenes", "1")
    assert result.exit_code == 0, result.output
    assert "1 scene(s) downloaded to raw/satellite/emit/" in result.output

    folder = project.raw_dir / "satellite" / "emit" / EMIT_IDS[0]
    # Reflectance and mask only: the uncertainty file and the preview are skipped.
    assert sorted(p.name for p in folder.iterdir()) == [
        f"{EMIT_IDS[0].replace('RFL_001', 'MASK_001')}.nc",
        f"{EMIT_IDS[0]}.nc",
    ]
    assert len(stac.downloads) == 2
    assert all(h == {"Authorization": "Bearer tok-123"} for _u, h in stac.downloads)
    assert all(u.startswith("https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/") for u, _h in stac.downloads)

    fetches = [r for r in project.provenance.read() if r.operation == "fetch"]
    assert len(fetches) == 2
    rfl = next(r for r in fetches if r.parameters["file"].endswith(f"{EMIT_IDS[0]}.nc"))
    assert (rfl.dataset, rfl.identifier) == ("emit", EMIT_IDS[0])  # explicit dataset + scene identifier
    assert rfl.url.endswith(f"{EMIT_IDS[0]}.nc") and rfl.retrieved_at.year >= 2026
    assert rfl.parameters["bytes"] == 72 and rfl.parameters["cloud_cover"] == 10
    assert rfl.parameters["sha256"] == hashlib.sha256((project.root / rfl.parameters["file"]).read_bytes()).hexdigest()
    assert rfl.parameters["collection"] == "EMITL2ARFL_001" and "CC0" in rfl.license


def test_downloaded_scenes_are_not_fetched_or_replaced_again(project, stac, monkeypatch):
    monkeypatch.setenv("EARTHDATA_TOKEN", "tok")
    assert harvest("--source", "emit", "--limit-scenes", "1").exit_code == 0
    original = project.raw_dir / "satellite" / "emit" / EMIT_IDS[0] / f"{EMIT_IDS[0]}.nc"
    original.write_bytes(b"\x89HDF-edited-by-hand")
    stac.downloads.clear()
    result = harvest("--source", "emit", "--limit-scenes", "1")
    assert result.exit_code == 0 and "1 already there" in result.output
    assert stac.downloads == [] and original.read_bytes() == b"\x89HDF-edited-by-hand"
    assert len([r for r in project.provenance.read() if r.operation == "fetch"]) == 2  # no new records

    stac.downloads.clear()
    assert harvest("--source", "emit", "--limit-scenes", "2").exit_code == 0  # the next scene is fetched
    assert len(stac.downloads) == 2 and (project.raw_dir / "satellite" / "emit" / EMIT_IDS[1]).is_dir()


def test_download_without_credentials_stops_before_any_request(project, stac):
    result = harvest("--source", "emit", "--limit-scenes", "1")
    assert result.exit_code == 1
    assert "EARTHDATA_TOKEN is not set" in result.output and "Manual fallback" in result.output
    assert "raw/satellite/emit/<scene id>/" in result.output
    assert stac.downloads == [] and not (project.raw_dir / "satellite").exists()
    assert len(scenes(project, "emit")) == 4  # the catalogue was still written, so scenes can be fetched by hand


def test_a_login_page_instead_of_data_is_rejected_and_leaves_nothing(project, stac, monkeypatch):
    monkeypatch.setenv("EARTHDATA_TOKEN", "bad")
    first_url = "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/" + f"{EMIT_IDS[0]}/{EMIT_IDS[0]}.nc"
    stac.html_for.add(first_url)
    result = harvest("--source", "emit", "--limit-scenes", "1")
    assert result.exit_code == 1
    assert "HTML page instead of the file" in result.output and "credentials were probably rejected" in result.output
    folder = project.raw_dir / "satellite" / "emit" / EMIT_IDS[0]
    assert not list(folder.glob("*"))  # neither the file nor a .part leftover
    assert not [r for r in project.provenance.read() if r.operation == "fetch"]


# --- ASTER ---------------------------------------------------------------------------------


def test_aster_is_capped_at_the_swir_failure_and_takes_vnir_and_swir_bands(project, stac, monkeypatch):
    monkeypatch.setenv("EARTHDATA_TOKEN", "tok")
    result = harvest("--source", "aster", "--limit-scenes", "1")
    assert result.exit_code == 0, result.output
    body = stac.requests[0][2]
    assert body["collections"] == ["AST_L1T_004"] and body["datetime"] == "../2008-04-30T23:59:59Z"

    scene = "AST_L1T_00407212005191137_20250519103107"  # covers the AOI fully with 0% cloud
    files = sorted(p.name.removeprefix(scene + "_") for p in (project.raw_dir / "satellite" / "aster" / scene).iterdir())
    assert files == [
        "SWIR_B04.tif", "SWIR_B05.tif", "SWIR_B06.tif", "SWIR_B07.tif", "SWIR_B08.tif", "SWIR_B09.tif",
        "VNIR_B01.tif", "VNIR_B02.tif", "VNIR_B03N.tif",
    ]  # fmt: skip
    fetch = next(r for r in project.provenance.read() if r.operation == "fetch")
    assert fetch.dataset == "aster" and fetch.identifier == scene and "CC0" in fetch.license
    assert len([r for r in project.provenance.read() if r.operation == "fetch"]) == 9


def test_aster_date_cap_can_be_overridden(project, stac):
    set_config(project, datetime="2000-01-01/..")
    assert harvest("--source", "aster").exit_code == 0
    assert stac.requests[0][2]["datetime"] == "2000-01-01/.."


# --- EnMAP ---------------------------------------------------------------------------------


def test_enmap_follows_post_paging_by_merging_the_next_body(project, stac):
    result = harvest("--source", "enmap")
    assert result.exit_code == 0, result.output
    (m1, u1, b1), (m2, u2, b2) = stac.requests
    assert (m1, m2) == ("POST", "POST")
    assert u1 == "https://geoservice.dlr.de/eoc/ogc/stac/v1/search"
    assert u2 == "https://stac.test/enmap/search"  # the next link's own href is used
    assert b1["collections"] == ["ENMAP_HSI_L2A"]
    assert b2["collections"] == ["ENMAP_HSI_L2A"] and b2["startIndex"] == 2 and "intersects" in b2  # merged, not replaced
    assert len(scenes(project, "enmap")) == 3
    search = next(r for r in project.provenance.read() if r.operation == "search")
    assert "EnMAP Data License" in search.license


def test_enmap_downloads_metadata_image_and_quality_with_basic_auth(project, stac, monkeypatch):
    monkeypatch.setenv("EOC_USERNAME", "user")
    monkeypatch.setenv("EOC_PASSWORD", "pw")
    assert harvest("--source", "enmap", "--limit-scenes", "1").exit_code == 0
    best = "ENMAP01-____L2A-DT0000140722_20250630T103935Z_006_V010502_20250702T131216Z"  # 0% cloud, full cover
    names = sorted(p.name.removeprefix(best + "-") for p in (project.raw_dir / "satellite" / "enmap" / best).iterdir())
    assert names == ["METADATA.XML", "QL_QUALITY_CLASSES_COG.TIF", "SPECTRAL_IMAGE_COG.TIF"]
    assert all(h["Authorization"].startswith("Basic ") for _u, h in stac.downloads)
    fetch = next(r for r in project.provenance.read() if r.operation == "fetch")
    assert fetch.dataset == "enmap" and "no sale or redistribution" in fetch.license


def test_enmap_without_credentials_explains_registration(project, stac):
    result = harvest("--source", "enmap", "--limit-scenes", "1")
    assert result.exit_code == 1
    assert "EOC_USERNAME and EOC_PASSWORD are not set" in result.output and "planning.enmap.org" in result.output
    assert stac.downloads == []


# --- building blocks -----------------------------------------------------------------------


def test_authorization_is_not_forwarded_to_another_host():
    handler = stac_module._SameHostRedirects()
    request = urllib.request.Request("https://data.example.test/file.nc", headers={"Authorization": "Bearer secret"})
    other = handler.redirect_request(request, None, 303, "See Other", {}, "https://cdn.example.net/signed?x=1")
    assert "Authorization" not in other.headers and "authorization" not in {k.lower() for k in other.unredirected_hdrs}
    same = handler.redirect_request(request, None, 302, "Found", {}, "https://data.example.test/other.nc")
    assert same.headers.get("Authorization") == "Bearer secret"


def test_file_signature_checks(tmp_path):
    def check(name: str, data: bytes):
        path = tmp_path / name
        path.write_bytes(data)
        return _signature_problem(path)

    assert check("a.nc.part", b"\x89HDF\r\n\x1a\n") is None
    assert check("a.nc.part", b"CDF\x01") is None
    assert "not NetCDF" in check("a.nc.part", b"PK\x03\x04zip")
    assert check("a.tif.part", b"II*\x00....") is None and check("a.tif.part", b"MM\x00*....") is None
    assert "not a TIFF" in check("a.TIF.part", b"GIF89a")
    assert "HTML" in check("a.tif.part", b"  <!DOCTYPE html><html>")
    assert "empty" in check("a.nc.part", b"")
    assert check("a.xml.part", b"<?xml version='1.0'?>") is None


def test_transient_gateway_errors_are_retried(monkeypatch, project):
    # The real _request_json, with urlopen scripted to fail once.
    real = StacSatelliteConnector.__dict__["_request_json"]
    outcomes = [urllib.error.HTTPError("u", 502, "Bad Gateway", {}, None), b'{"features": [], "links": []}']
    calls = []

    class Response:
        def __init__(self, data):
            self.data = data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, *a):
            return self.data

    def urlopen(request, timeout):
        calls.append(request.full_url)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return Response(outcome)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(stac_module.time, "sleep", lambda s: None)
    connector = stac_module.StacSatelliteConnector.__subclasses__()[0](project)
    assert real(connector, "POST", "https://stac.test/search", {"a": 1}) == {"features": [], "links": []}
    assert len(calls) == 2

    outcomes[:] = [urllib.error.HTTPError("u", 404, "Not Found", {}, None)]
    with pytest.raises(urllib.error.HTTPError):  # anything but a gateway error is raised at once
        real(connector, "POST", "https://stac.test/search", {"a": 1})


def test_unknown_auth_error_type_is_a_clean_failure(project):
    assert issubclass(SatelliteAuthError, Exception) and not issubclass(SatelliteAuthError, OSError)
