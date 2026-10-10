from pathlib import Path
from urllib.parse import parse_qs, urlparse

import geopandas as gpd
import pandas as pd
import pyogrio
import pytest
from typer.testing import CliRunner

from gossan.cli import app
from gossan.core.project import Project
from gossan.core.schemas import (
    MINFILE_COLUMNS,
    TENURE_COLUMNS,
    conform,
    schema_problems,
)
from gossan.harvest import CONNECTORS, default_sources
from gossan.harvest.us_federal.blm_mlrs import (
    BlmClaimsConnector,
    claim_type,
    serial_digits,
)
from gossan.harvest.us_federal.mrds import COMMODITY_NAMES, commodity_names

runner = CliRunner()
NV = Path(__file__).parent / "fixtures" / "aoi_nv.geojson"


@pytest.fixture
def project() -> Project:
    result = runner.invoke(app, ["new", "nv", "--aoi", str(NV)])
    assert result.exit_code == 0, result.output
    return Project(Path("nv"))


def harvest(*args):
    return runner.invoke(app, ["harvest", "nv", *args])


def layers(project: Project) -> set[str]:
    return {str(n) for n in pyogrio.list_layers(project.data_gpkg)[:, 0]}


# --- registry and the BC alias ---------------------------------------------------------------


def test_connectors_declare_their_jurisdiction_and_defaults_follow_it():
    assert default_sources("bc") == ["minfile", "mto_tenure", "aris_index", "drilling_notices", "rgs"]  # BC gains the drilling indicator
    assert default_sources("us") == ["mrds", "blm_claims"]
    assert CONNECTORS["mrds"].jurisdiction == CONNECTORS["blm_claims"].jurisdiction == "us"
    assert CONNECTORS["aris_index"].jurisdiction == "bc"
    assert CONNECTORS["emit"].jurisdiction == "any" and not CONNECTORS["emit"].default  # opt-in, anywhere


def test_old_bc_import_paths_are_the_same_modules():
    import gossan.harvest.bc.wfs as old

    import gossan.harvest.bc_provincial.wfs as new

    assert old is new
    from gossan.harvest.bc.aris_pdf import ArisPdfDownloader as old_cls

    from gossan.harvest.bc_provincial.aris_pdf import ArisPdfDownloader as new_cls

    assert old_cls is new_cls


def test_us_connectors_write_the_same_layers_as_bc():
    assert CONNECTORS["mrds"].canonical_layer == "minfile"
    assert CONNECTORS["blm_claims"].canonical_layer == "mto_tenure"
    assert CONNECTORS["mrds"].name == "mrds" and CONNECTORS["blm_claims"].name == "blm_claims"  # raw/ folders differ


# --- schema helpers --------------------------------------------------------------------------


def test_conform_adds_types_and_orders_the_canonical_columns_first():
    gdf = gpd.GeoDataFrame({"extra": [1], "TENURE_NUMBER_ID": ["42"], "AREA_IN_HECTARES": ["3.5"]}, geometry=gpd.points_from_xy([0], [0]).buffer(1), crs="EPSG:32611")
    out = conform(gdf, "mto_tenure")
    assert list(out.columns)[: len(TENURE_COLUMNS)] == list(TENURE_COLUMNS) and list(out.columns)[-2:] == ["extra", "geometry"]
    assert out["TENURE_NUMBER_ID"].dtype == "Int64" and out["TENURE_NUMBER_ID"].iloc[0] == 42
    assert out["AREA_IN_HECTARES"].iloc[0] == 3.5 and out["OWNER_NAME"].iloc[0] is None and out["ISSUE_DATE"].isna().all()
    assert schema_problems(out, "mto_tenure") == []


def test_schema_problems_name_what_is_wrong():
    bad = gpd.GeoDataFrame({"TENURE_NUMBER_ID": [1.5, None]}, geometry=gpd.points_from_xy([0, 1], [0, 1]), crs="EPSG:32611")
    problems = schema_problems(bad, "mto_tenure")
    assert any("missing column(s)" in p for p in problems) and any("geometry types" in p for p in problems)
    assert any("empty values" in p for p in problems) and any("not integer" in p for p in problems)


# --- MRDS ------------------------------------------------------------------------------------


def test_commodity_codes_become_names_and_unknown_codes_stay():
    assert commodity_names(" AU AG CU") == ["Gold", "Silver", "Copper"]
    assert commodity_names(" XX cu") == ["XX", "Copper"]  # lower case is accepted
    assert commodity_names("") == [] and commodity_names(None) == [] and commodity_names(float("nan")) == []
    assert COMMODITY_NAMES["AU"] == "Gold" and COMMODITY_NAMES["PB"] == "Lead"


def test_mrds_harvest_writes_the_minfile_layer_in_the_canonical_schema(project, us):
    result = harvest("--source", "mrds")
    assert result.exit_code == 0, result.output
    assert "mrds: 5 feature(s) written to data.gpkg" in result.output  # the outside record is dropped locally

    ((_service, url),) = us.calls
    q = parse_qs(urlparse(url).query)
    assert url.startswith("https://mrdata.usgs.gov/services/mrds?") and q["typename"] == ["mrds-high"] and q["version"] == ["1.0.0"]
    assert q["bbox"] == ["-117.300000,39.100000,-117.100000,39.250000"]  # lon,lat order for WFS 1.0.0

    gdf = gpd.read_file(project.data_gpkg, layer="minfile")
    assert gdf.crs.to_epsg() == 32611  # Nevada: UTM zone 11
    assert schema_problems(gdf, "minfile") == [] and list(gdf.columns)[: len(MINFILE_COLUMNS)] == list(MINFILE_COLUMNS)
    by_id = gdf.set_index("MINFILE_NUMBER")
    apex = by_id.loc["10022723"]
    assert (apex["MINFILE_NAME1"], apex["STATUS_DESCRIPTION"]) == ("Apex Nos 1 and 2", "Occurrence")
    assert (apex["COMMODITY_DESCRIPTION1"], apex["COMMODITY_DESCRIPTION2"], apex["COMMODITY_DESCRIPTION3"]) == ("Gold", "Silver", "Copper")
    assert apex["COMMODITY_CODES"] == "AU AG CU" and apex["MINFILE_SUMMARY_URL"].endswith("dep_id=10022723")
    assert apex["LONGITUDE_DECIMAL"] == pytest.approx(-117.20) and apex["LATITUDE_DECIMAL"] == pytest.approx(39.15)
    five = by_id.loc["10040483"]
    assert five["COMMODITY_DESCRIPTION3"] == "Silver" and five["COMMODITY_CODES"] == "HG AU AG PB ZN"  # codes beyond three are kept
    assert by_id.loc["10099999"]["COMMODITY_DESCRIPTION1"] == "XX"  # an unknown code is not lost
    assert pd.isna(by_id.loc["10088888"]["COMMODITY_DESCRIPTION1"]) and pd.isna(by_id.loc["10088888"]["DEPOSIT_TYPE_DESCRIPTION1"])
    assert "10077777" not in by_id.index and set(gdf.geom_type) == {"Point"}


def test_mrds_raw_copy_is_kept_untouched_and_logged(project, us):
    assert harvest("--source", "mrds").exit_code == 0
    raw = list((project.raw_dir / "mrds").iterdir())
    assert [p.suffix for p in raw] == [".gml"]  # the original response; no .gfs sidecar written beside it
    assert raw[0].read_bytes() == (Path(__file__).parent / "fixtures" / "mrds_wfs.gml").read_bytes()

    records = project.provenance.read()
    fetch = next(r for r in records if r.operation == "fetch")
    assert (fetch.dataset, fetch.identifier) == ("minfile", "mrds-high")  # the canonical layer is the dataset
    assert "USGS" in fetch.source and "Public domain" in fetch.license and "Mason and Arndt" in fetch.license
    assert fetch.parameters["n_features"] == 6 and fetch.parameters["file"].startswith("raw/mrds/mrds_")
    write = next(r for r in records if r.operation == "write_layer")
    assert write.dataset == "minfile" and write.parameters["n_features"] == 5
    assert not any("Access Only" in r.license for r in records)  # nothing here restricts the data


def test_mrds_empty_result_and_service_errors(project, us):
    us.mrds_file = "mrds_empty.gml"
    result = harvest("--source", "mrds")
    assert result.exit_code == 0 and "0 feature(s)" in result.output and not project.data_gpkg.exists()

    us.mrds_bytes = b'<?xml version="1.0"?><ows:ExceptionReport><ows:Exception>bad bbox</ows:Exception></ows:ExceptionReport>'
    result = harvest("--source", "mrds")
    assert result.exit_code == 1 and "MRDS WFS returned an error" in result.output and "gossan ingest" in result.output


# --- BLM MLRS --------------------------------------------------------------------------------


def test_blm_helpers():
    assert serial_digits("NV105223401") == 105223401 and serial_digits("AMC123456") == 123456
    assert serial_digits("") is None and serial_digits(None) is None and serial_digits("ABC") is None
    assert claim_type("Lode Claim") == "Mineral" and claim_type("PLACER CLAIM") == "Placer"
    assert claim_type("Mill Site") == "Mill site" and claim_type("Tunnel Site") == "Tunnel site"
    assert claim_type("Geothermal Lease") == "Geothermal Lease" and claim_type(None) == "Unknown"


def test_blm_harvest_pages_the_service_and_writes_the_tenure_layer(project, us, monkeypatch):
    monkeypatch.setattr(BlmClaimsConnector, "page_size", 2)  # force paging on the 8-feature fixture
    result = harvest("--source", "blm_claims")
    assert result.exit_code == 0, result.output

    urls = [u for s, u in us.calls if s == "blm"]
    assert len(urls) == 5  # 4 full pages and an empty one that ends the loop
    first = urls[0]
    assert first.startswith("https://gis.blm.gov/nlsdb/rest/services/HUB/BLM_Natl_MLRS_Mining_Claims_Not_Closed/MapServer/0/query?")
    q = parse_qs(urlparse(first).query)
    assert q["geometryType"] == ["esriGeometryEnvelope"] and q["inSR"] == ["4326"] and q["outSR"] == ["4326"] and q["f"] == ["geojson"]
    assert q["geometry"] == ["-117.300000,39.100000,-117.100000,39.250000"] and q["orderByFields"] == ["OBJECTID"]
    assert [parse_qs(urlparse(u).query)["resultOffset"] for u in urls] == [["0"], ["2"], ["4"], ["6"], ["8"]]

    gdf = gpd.read_file(project.data_gpkg, layer="mto_tenure")
    assert gdf.crs.to_epsg() == 32611 and schema_problems(gdf, "mto_tenure") == []
    assert list(gdf.columns)[: len(TENURE_COLUMNS)] == list(TENURE_COLUMNS)
    # 8 features: one outside the AOI, one with no geometry, and two polygons of the same case are merged.
    assert len(gdf) == 5
    assert sorted(gdf["CLAIM_NAME"]) == ["AZ STYLE", "MILL ONE", "NO SERIAL", "PLACER ONE", "TG001"]
    by = gdf.set_index("CLAIM_NAME")
    assert by.loc["TG001"]["TENURE_NUMBER_ID"] == 105223401 and by.loc["TG001"].geometry.geom_type in {"Polygon", "MultiPolygon"}
    assert by.loc["AZ STYLE"]["TENURE_NUMBER_ID"] == 123456  # the digits of AMC123456
    assert by.loc["NO SERIAL"]["TENURE_NUMBER_ID"] == 809142547606777805  # no serial: the service's own numeric id
    assert dict(zip(gdf["CLAIM_NAME"], gdf["TENURE_TYPE_DESCRIPTION"], strict=True)) == {
        "TG001": "Mineral", "PLACER ONE": "Placer", "MILL ONE": "Mill site", "NO SERIAL": "Mineral", "AZ STYLE": "Mineral",
    }  # fmt: skip
    assert by.loc["PLACER ONE"]["AREA_IN_HECTARES"] == pytest.approx(160.0 * 0.40468564224)
    assert by.loc["AZ STYLE"]["AREA_IN_HECTARES"] == pytest.approx(by.loc["AZ STYLE"].geometry.area / 1e4, rel=1e-6)  # no recorded acres: from the polygon
    # The public layer has no staking date, expiry or claimant: they stay empty, and the record-creation date is not a staking date.
    assert gdf["ISSUE_DATE"].isna().all() and gdf["GOOD_TO_DATE"].isna().all() and gdf["OWNER_NAME"].isna().all()
    assert gdf["RECORD_CREATED"].notna().all()
    assert by.loc["TG001"]["BLM_CASE_SERIAL"] == "NV105223401" and by.loc["TG001"]["STATE"] == "NV" and by.loc["TG001"]["DISPOSITION"] == "Active"
    assert by.loc["PLACER ONE"]["POSITION_QUALITY"] == "8: SIMPLE" and by.loc["MILL ONE"]["DISPOSITION"] == "Pending"
    assert gdf["TENURE_NUMBER_ID"].is_unique


def test_blm_provenance_and_raw_copy(project, us, monkeypatch):
    monkeypatch.setattr(BlmClaimsConnector, "page_size", 2)
    assert harvest("--source", "blm_claims").exit_code == 0
    (raw,) = (project.raw_dir / "blm_claims").iterdir()
    assert raw.suffix == ".geojson" and len(__import__("json").loads(raw.read_text())["features"]) == 8  # every page, merged, unfiltered
    records = project.provenance.read()
    fetch = next(r for r in records if r.operation == "fetch")
    assert (fetch.dataset, fetch.identifier) == ("mto_tenure", BlmClaimsConnector.layer_url)
    assert "BLM" in fetch.source and "Public domain" in fetch.license and "not a legal document" in fetch.license
    assert fetch.parameters["pages"] == 5 and fetch.parameters["n_features"] == 8 and fetch.parameters["where"] == "1=1"
    assert next(r for r in records if r.operation == "write_layer").parameters["n_features"] == 5
    assert not any("Access Only" in r.license for r in records)


def test_blm_service_error_in_a_200_response_is_reported(project, us):
    us.blm_file_override = "blm_claims_error.json"
    result = harvest("--source", "blm_claims")
    assert result.exit_code == 1 and "ArcGIS service error 400" in result.output and "gossan ingest" in result.output
    assert not project.data_gpkg.exists()


def test_blm_empty_result(project, us):
    us.blm_file_override = "blm_claims_empty.json"
    result = harvest("--source", "blm_claims")
    assert result.exit_code == 0 and "0 feature(s)" in result.output and not project.data_gpkg.exists()


def test_blm_cases_with_equal_digits_fall_back_to_the_service_id(project, us):
    from gossan.harvest.us_federal import (
        blm_mlrs,  # noqa: F401  (documented behaviour below)
    )

    def feature(oid, serial, x):
        return {
            "type": "Feature", "id": oid,
            "geometry": {"type": "Polygon", "coordinates": [[[x, 39.12], [x + 0.001, 39.12], [x + 0.001, 39.13], [x, 39.13], [x, 39.12]]]},
            "properties": {"OBJECTID": oid, "ID": str(9000 + oid), "CSE_NAME": f"C{oid}", "BLM_PROD": "Lode Claim", "CSE_NR": serial, "CSE_DISP": "Active", "RCRD_ACRS": 20.0, "QLTY": "0: x", "CSE_META": "", "Created": 1612915200000},
        }  # fmt: skip

    us.blm_features = [feature(1, "NV100200", -117.28), feature(2, "AZ100200", -117.26)]  # same digits, different cases
    assert harvest("--source", "blm_claims").exit_code == 0
    gdf = gpd.read_file(project.data_gpkg, layer="mto_tenure")
    assert sorted(gdf["TENURE_NUMBER_ID"]) == [9001, 9002] and gdf["TENURE_NUMBER_ID"].is_unique


# --- running them ----------------------------------------------------------------------------


def test_a_us_project_harvests_only_us_sources_by_default(project, us):
    result = harvest()
    assert result.exit_code == 0, result.output
    assert {s for s, _ in us.calls} == {"mrds", "blm"}
    assert layers(project) == {"minfile", "mto_tenure"}  # the same layer names target and leads read
    fetched = sorted(r.dataset for r in project.provenance.read() if r.operation == "fetch")
    assert fetched == ["minfile", "mto_tenure"]


def test_naming_a_source_from_the_other_jurisdiction_is_allowed_with_a_note(project, us, wfs):
    result = harvest("--source", "aris_index")
    assert "aris_index is a bc source and this is a us project" in result.output
    assert pd.Series(us.calls).empty  # and the US services were not touched


def test_merged_layers_from_both_us_sources_coexist(project, us):
    assert harvest("--source", "mrds").exit_code == 0
    assert harvest("--source", "blm_claims").exit_code == 0
    assert layers(project) == {"minfile", "mto_tenure"}
    assert len(gpd.read_file(project.data_gpkg, layer="minfile")) == 5
