from pathlib import Path

import geopandas as gpd
from typer.testing import CliRunner

from gossan.cli import app
from gossan.core.config import ProjectConfig
from gossan.core.provenance import ProvenanceLog

runner = CliRunner()


def test_new_creates_project(aoi_file):
    result = runner.invoke(app, ["new", "demo", "--aoi", str(aoi_file)])
    assert result.exit_code == 0, result.output

    project = Path("demo")
    assert (project / "aoi.gpkg").is_file()
    assert (project / "project.yaml").is_file()

    config = ProjectConfig.load(project)
    assert config.name == "demo"
    assert config.utm_zone == 10
    assert config.crs == "EPSG:32610"
    assert config.aoi_path == "aoi.gpkg"
    assert [m.value for m in config.modules] == ["harvest", "report"]

    aoi = gpd.read_file(project / "aoi.gpkg")
    assert aoi.crs.to_epsg() == 32610
    assert len(aoi) == 1

    (record,) = ProvenanceLog(project).read()
    assert record.operation == "import_aoi"


def test_new_crs_override_and_no_project(aoi_file):
    result = runner.invoke(app, ["new", "a", "--aoi", str(aoi_file), "--crs", "EPSG:26910"])
    assert result.exit_code == 0, result.output
    assert gpd.read_file("a/aoi.gpkg").crs.to_epsg() == 26910
    assert ProjectConfig.load(Path("a")).utm_zone is None

    result = runner.invoke(app, ["new", "b", "--aoi", str(aoi_file), "--no-project"])
    assert result.exit_code == 0, result.output
    assert gpd.read_file("b/aoi.gpkg").crs.to_epsg() == 4326
    assert ProjectConfig.load(Path("b")).crs == "EPSG:4326"


def test_new_refuses_existing_folder(aoi_file):
    Path("demo").mkdir()
    result = runner.invoke(app, ["new", "demo", "--aoi", str(aoi_file)])
    assert result.exit_code == 1
    assert not Path("demo/aoi.gpkg").exists()


def test_new_rejects_bad_name(aoi_file):
    result = runner.invoke(app, ["new", "a/b", "--aoi", str(aoi_file)])
    assert result.exit_code != 0
    assert not Path("a").exists()
