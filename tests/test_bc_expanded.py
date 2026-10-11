"""The expanded BC harvest: geology, mineral inventory, activity, land status, access, terrain."""

from pathlib import Path
from urllib.parse import parse_qs, urlparse

import geopandas as gpd
import pyogrio
import pytest
import rasterio
from typer.testing import CliRunner

from gossan.cli import app
from gossan.core.licensing import uses_aris
from gossan.core.project import Project
from gossan.harvest import CONNECTORS, default_sources
from gossan.harvest.bc_provincial.wfs import (
    ACCESS_ONLY_BC,
    OGL_BC,
    OGL_CANADA,
    WfsConnector,
)
from gossan.harvest.national.mrdem import MRDEM_URL

runner = CliRunner()

# name -> (WFS type name, geometry property, fixture, licence)
NEW_WFS = {
    "bedrock_geology": ("pub:WHSE_MINERAL_TENURE.GEOL_BEDROCK_UNIT_POLY_SVW", "GEOMETRY", "bc_polygons_wfs.geojson", OGL_BC),
    "faults": ("pub:WHSE_MINERAL_TENURE.GEOL_FAULT_LINE", "GEOMETRY", "bc_lines_wfs.geojson", OGL_BC),
    "quaternary_cover": ("pub:WHSE_MINERAL_TENURE.GEOL_QUATERNARY_POLY", "GEOMETRY", "bc_polygons_wfs.geojson", ACCESS_ONLY_BC),
    "mineral_potential": ("pub:WHSE_MINERAL_TENURE.MINPOT_MINERAL_POTENTIAL", "GEOMETRY", "bc_polygons_wfs.geojson", ACCESS_ONLY_BC),
    "minfile_production": ("pub:WHSE_MINERAL_TENURE.MINFIL_PRODUCT", "SHAPE", "bc_points_wfs.geojson", ACCESS_ONLY_BC),
    "minfile_inventory": ("pub:WHSE_MINERAL_TENURE.MINFIL_RESERVE", "SHAPE", "bc_points_wfs.geojson", ACCESS_ONLY_BC),
    "rgs_survey_areas": ("pub:WHSE_MINERAL_TENURE.GEOL_REG_GEOCHMCL_SURV_AREA_SP", "GEOMETRY", "bc_polygons_wfs.geojson", ACCESS_ONLY_BC),
    "historic_notices": ("pub:WHSE_MINERAL_TENURE.MMS_NOTICE_OF_WORK", "GEOMETRY", "bc_points_wfs.geojson", ACCESS_ONLY_BC),
    "permitted_mine_areas": ("pub:WHSE_MINERAL_TENURE.HSP_NOTICE_OF_WORK_PA_POLY_SV", "SHAPE", "bc_polygons_wfs.geojson", OGL_BC),
    "major_mine_areas": ("pub:WHSE_MINERAL_TENURE.HSP_MJR_MINES_PERMTTD_AREAS_SP", "SHAPE", "bc_polygons_wfs.geojson", ACCESS_ONLY_BC),
    "crown_grants": ("pub:WHSE_MINERAL_TENURE.MTA_CROWN_GRANT_MIN_CLAIM_SVW", "SHAPE", "bc_polygons_wfs.geojson", ACCESS_ONLY_BC),
    "parks_protected": ("pub:WHSE_TANTALIS.TA_PARK_ECORES_PA_SVW", "SHAPE", "bc_polygons_wfs.geojson", OGL_BC),
    "conservancies": ("pub:WHSE_TANTALIS.TA_CONSERVANCY_AREAS_SVW", "SHAPE", "bc_polygons_wfs.geojson", OGL_BC),
    "national_parks": ("pub:WHSE_ADMIN_BOUNDARIES.CLAB_NATIONAL_PARKS", "GEOMETRY", "bc_polygons_wfs.geojson", ACCESS_ONLY_BC),
    "indian_reserves": ("pub:WHSE_ADMIN_BOUNDARIES.CLAB_INDIAN_RESERVES", "GEOMETRY", "bc_polygons_wfs.geojson", OGL_CANADA),
    "treaty_lands": ("pub:WHSE_LEGAL_ADMIN_BOUNDARIES.FNT_TREATY_LAND_SP", "GEOMETRY", "bc_polygons_wfs.geojson", ACCESS_ONLY_BC),
    "reserve_sites": ("pub:WHSE_MINERAL_TENURE.MTA_SITE_SP", "GEOMETRY", "bc_polygons_wfs.geojson", OGL_BC),
    "roads": ("pub:WHSE_BASEMAPPING.DRA_DGTL_ROAD_ATLAS_MPAR_SP", "GEOMETRY", "bc_lines_wfs.geojson", ACCESS_ONLY_BC),
    "transmission_lines": ("pub:WHSE_BASEMAPPING.GBA_TRANSMISSION_LINES_SP", "SHAPE", "bc_lines_wfs.geojson", OGL_BC),
}


@pytest.fixture
def project(aoi_file) -> Project:
    assert runner.invoke(app, ["new", "demo", "--aoi", str(aoi_file)]).exit_code == 0
    return Project(Path("demo"))


def test_new_sources_are_registered_bc_defaults():
    defaults = default_sources("bc")
    for name in [*NEW_WFS, "dem"]:
        assert name in CONNECTORS
        assert name in defaults
        assert name not in default_sources("us")
    for name, (type_name, geometry, _, licence) in NEW_WFS.items():
        c = CONNECTORS[name]
        assert issubclass(c, WfsConnector)
        assert (c.type_name, c.geometry_column, c.license) == (type_name, geometry, licence)


@pytest.mark.parametrize("name", sorted(NEW_WFS))
def test_harvest_new_wfs_layer(project, wfs, name):
    type_name, geometry, fixture, licence = NEW_WFS[name]
    wfs.responses[type_name] = fixture
    result = runner.invoke(app, ["harvest", "demo", "--source", name])
    assert result.exit_code == 0, result.output

    query = parse_qs(urlparse(wfs.calls[0]).query)
    assert query["typeNames"] == [type_name]
    assert query["CQL_FILTER"][0].startswith(f"BBOX({geometry},")

    gdf = gpd.read_file(project.data_gpkg, layer=name)
    assert gdf.crs.to_epsg() == 32610
    assert "OUTSIDE" not in set(gdf["NAME"])  # selected to the AOI
    assert len(gdf) == 2
    assert all(n == n.strip() for n in gdf["NAME"])  # padding stripped

    assert len(list((project.raw_dir / name).glob(f"{name}_*.geojson"))) == 1
    fetch = next(r for r in project.provenance.read() if r.operation == "fetch")
    assert (fetch.dataset, fetch.identifier, fetch.license) == (name, type_name, licence)


def test_access_only_layers_mark_the_project(project, wfs):
    wfs.responses[NEW_WFS["roads"][0]] = "bc_lines_wfs.geojson"
    assert runner.invoke(app, ["harvest", "demo", "--source", "roads"]).exit_code == 0
    assert uses_aris(project.provenance.read())  # "Access Only" licence makes outputs internal


def test_open_layers_do_not_mark_the_project(project, wfs):
    wfs.responses[NEW_WFS["faults"][0]] = "bc_lines_wfs.geojson"
    assert runner.invoke(app, ["harvest", "demo", "--source", "faults"]).exit_code == 0
    assert not uses_aris(project.provenance.read())


def test_harvest_dem(project, wfs):
    result = runner.invoke(app, ["harvest", "demo", "--source", "dem"])
    assert result.exit_code == 0, result.output
    assert "dem: 1 raster(s) written to rasters/" in result.output

    url, bounds = wfs.dem_calls[0]
    assert url == MRDEM_URL
    minx, miny, maxx, maxy = bounds
    assert minx < maxx and miny < maxy
    # Window is the AOI in EPSG:3979 plus a buffer: a ~3 x 3 km AOI gives a > 4 km window.
    assert maxx - minx > 4000 and maxy - miny > 4000

    raws = list((project.raw_dir / "dem").glob("mrdem30_*.tif"))
    assert len(raws) == 1
    with rasterio.open(project.root / "rasters" / "dem.tif") as ds:
        assert ds.crs.to_epsg() == 32610
        assert ds.nodata is not None

    records = project.provenance.read()
    fetch = next(r for r in records if r.operation == "fetch")
    assert fetch.dataset == "dem" and fetch.url == MRDEM_URL
    assert fetch.license == "Open Government Licence - Canada"
    assert fetch.parameters["file"] == f"raw/dem/{raws[0].name}"
    write = next(r for r in records if r.operation == "write_raster")
    assert write.identifier == "rasters/dem.tif"


def test_reharvest_keeps_raw_originals(project, wfs):
    for _ in range(2):
        assert runner.invoke(app, ["harvest", "demo", "--source", "dem"]).exit_code == 0
    assert len(list((project.raw_dir / "dem").glob("mrdem30_*.tif"))) == 2


def test_plain_harvest_runs_new_sources_without_failing_on_empty_layers(project, wfs):
    result = runner.invoke(app, ["harvest", "demo"])
    assert result.exit_code == 0, result.output
    assert "bedrock_geology: 0 feature(s)" in result.output
    layers = set(pyogrio.list_layers(project.data_gpkg)[:, 0])
    assert "bedrock_geology" not in layers  # empty layers are logged, not written
    assert (project.root / "rasters" / "dem.tif").exists()
