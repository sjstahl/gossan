from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box
from typer.testing import CliRunner

from gossan.cli import app
from gossan.core.aoi import detect_jurisdiction, utm_zone_for, utm_zones_spanned
from gossan.core.config import ProjectConfig

runner = CliRunner()


def aoi_at(lon: float, lat: float, half: float = 0.05) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"id": [1]}, geometry=[box(lon - half, lat - half, lon + half, lat + half)], crs="EPSG:4326")


def new_project(tmp_path: Path, lon: float, lat: float, *args: str, half: float = 0.05, name: str = "p"):
    path = tmp_path / f"{name}_aoi.gpkg"
    aoi_at(lon, lat, half).to_file(path, driver="GPKG")
    return runner.invoke(app, ["new", name, "--aoi", str(path), *args])


# --- the project CRS follows the AOI, anywhere ---------------------------------------------


@pytest.mark.parametrize(
    ("place", "lon", "lat", "epsg", "zone"),
    [
        ("Kamloops, BC", -120.3, 50.7, 32610, 10),
        ("Reno, NV", -119.8, 39.5, 32611, 11),
        ("Battle Mountain, NV", -116.9, 40.6, 32611, 11),
        ("Las Vegas, NV", -115.1, 36.2, 32611, 11),
        ("Tucson, AZ", -110.97, 32.22, 32612, 12),
        ("Flagstaff, AZ", -111.65, 35.2, 32612, 12),
        ("Yuma, AZ", -114.6, 32.7, 32611, 11),  # Arizona's west edge is zone 11, not 12
        ("Santiago, Chile", -70.6, -33.4, 32719, 19),  # southern hemisphere
    ],
)
def test_the_project_gets_the_utm_zone_of_the_aoi_centroid(tmp_path, place, lon, lat, epsg, zone):
    result = new_project(tmp_path, lon, lat)
    assert result.exit_code == 0, result.output
    config = ProjectConfig.load(Path("p"))
    assert (config.crs, config.utm_zone) == (f"EPSG:{epsg}", zone), place
    assert gpd.read_file("p/aoi.gpkg").crs.to_epsg() == epsg  # the stored AOI is reprojected into it
    assert "3005" not in config.crs  # no fixed provincial projection anywhere


def test_nothing_in_the_code_defaults_to_bc_albers():
    src = Path(__file__).parent.parent / "src"
    offenders = [p.name for p in src.rglob("*.py") if "3005" in p.read_text(encoding="utf-8") or "Albers" in p.read_text(encoding="utf-8")]
    assert offenders == []
    assert "crs" in ProjectConfig.model_fields and ProjectConfig.model_fields["crs"].is_required()  # there is no default CRS to override


def test_zone_helpers():
    assert utm_zone_for(aoi_at(-116.9, 40.6)) == (11, "EPSG:32611")
    assert utm_zones_spanned(aoi_at(-116.9, 40.6)) == [11]
    assert utm_zones_spanned(aoi_at(-120.0, 39.0, half=0.5)) == [10, 11]  # straddles the -120 meridian


def test_an_aoi_across_two_zones_warns_and_uses_the_centroid_zone(tmp_path):
    result = new_project(tmp_path, -120.0, 39.5, half=0.5)
    assert result.exit_code == 0
    assert "spans UTM zones 10 and 11" in result.output and "using zone 11" in result.output
    assert ProjectConfig.load(Path("p")).crs == "EPSG:32611"
    clean = new_project(tmp_path, -116.9, 40.6, name="q")
    assert "spans UTM zones" not in clean.output


def test_an_explicit_crs_is_respected_and_does_not_warn_about_zones(tmp_path):
    result = new_project(tmp_path, -120.0, 39.5, "--crs", "EPSG:26911", half=0.5)
    assert result.exit_code == 0 and "spans UTM zones" not in result.output
    assert ProjectConfig.load(Path("p")).crs == "EPSG:26911"


# --- jurisdiction ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("place", "lon", "lat", "expected"),
    [
        ("Kamloops", -120.3, 50.7, "bc"),
        ("Vancouver", -123.1, 49.28, "bc"),
        ("Victoria", -123.37, 48.43, "bc"),
        ("Prince George", -122.75, 53.9, "bc"),
        ("Bellingham", -122.48, 48.75, "us"),
        ("Seattle", -122.3, 47.6, "us"),
        ("Reno", -119.8, 39.5, "us"),
        ("Tucson", -110.97, 32.22, "us"),
        ("Elko", -115.8, 40.8, "us"),
        ("Whitehorse", -135.05, 60.7, None),  # Yukon: not BC
        ("Anchorage", -149.9, 61.2, None),  # Alaska: not the contiguous US
        ("Denver", -105.0, 39.7, "us"),
        ("Toronto", -79.4, 43.7, None),  # eastern Canada: not guessed
        ("Atlanta", -84.4, 33.7, None),  # the eastern US: pass --jurisdiction us
    ],
)
def test_jurisdiction_detection(place, lon, lat, expected):
    assert detect_jurisdiction(aoi_at(lon, lat)) == expected, place


def test_new_records_the_detected_jurisdiction(tmp_path):
    result = new_project(tmp_path, -116.9, 40.6)
    assert result.exit_code == 0 and "jurisdiction us" in result.output
    assert ProjectConfig.load(Path("p")).jurisdiction == "us"
    assert "jurisdiction: us" in Path("p/project.yaml").read_text()
    bc = new_project(tmp_path, -120.3, 50.7, name="b")
    assert ProjectConfig.load(Path("b")).jurisdiction == "bc" and "jurisdiction bc" in bc.output


def test_new_logs_the_jurisdiction_and_zones_to_provenance(tmp_path):
    from gossan.core.provenance import ProvenanceLog

    new_project(tmp_path, -120.0, 39.5, half=0.5)
    (record,) = ProvenanceLog(Path("p")).read()
    assert record.parameters["jurisdiction"] == "us" and record.parameters["utm_zones_spanned"] == [10, 11]


def test_an_unclear_aoi_assumes_bc_and_says_so_and_the_flag_overrides(tmp_path):
    result = new_project(tmp_path, -149.9, 61.2)  # Anchorage
    assert result.exit_code == 0 and "could not tell the AOI's jurisdiction; assuming bc" in result.output
    assert ProjectConfig.load(Path("p")).jurisdiction == "bc"
    forced = new_project(tmp_path, -149.9, 61.2, "--jurisdiction", "us", name="u")
    assert "could not tell" not in forced.output and ProjectConfig.load(Path("u")).jurisdiction == "us"


def test_the_flag_wins_over_detection_and_bad_values_are_refused(tmp_path):
    result = new_project(tmp_path, -116.9, 40.6, "--jurisdiction", "bc")  # Nevada, but told bc
    assert ProjectConfig.load(Path("p")).jurisdiction == "bc" and result.exit_code == 0
    assert new_project(tmp_path, -116.9, 40.6, "--jurisdiction", "mx", name="x").exit_code != 0


def test_projects_made_before_jurisdiction_existed_default_to_bc(tmp_path):
    new_project(tmp_path, -120.3, 50.7)
    path = Path("p/project.yaml")
    path.write_text("".join(line for line in path.read_text().splitlines(keepends=True) if not line.startswith("jurisdiction")))
    assert ProjectConfig.load(Path("p")).jurisdiction == "bc"  # an old project.yaml keeps working
