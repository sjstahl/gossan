"""A small synthetic project (invented geometries and arrays) shared by the target and report tests."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import Point, box
from typer.testing import CliRunner

from gossan.cli import app
from gossan.core.config import TargetConfig
from gossan.core.project import Project

CRS = "EPSG:32610"
X0, Y0, W, H = 690_000.0, 5_617_000.0, 4000.0, 3000.0  # a 4 km x 3 km AOI
runner = CliRunner()


def small_cfg(**kw) -> TargetConfig:
    """A fast config. It pins the original feature set (effort layers as features, raw elements, the
    occurrence-distance feature), which these tests were written against; the evidence-first
    defaults are tested in test_target_evidence.py."""
    base = {
        "resolution_m": 100.0, "n_bags": 4, "n_trees": 15, "folds": 3, "seed": 1,
        "effort": "feature", "occurrence_distance": True, "geochem_features": "elements",
        "smooth_m": 0.0, "min_target_ha": 0.0,
    }  # fmt: skip
    return TargetConfig(**{**base, **kw})


def make_aoi(path: Path, geometry=None) -> Path:
    gpd.GeoDataFrame({"id": [1]}, geometry=[geometry or box(X0, Y0, X0 + W, Y0 + H)], crs=CRS).to_file(path, driver="GPKG")
    return path


def build_project(
    tmp_path: Path,
    *,
    layers=("minfile", "mto_tenure", "aris_index", "rgs"),
    raster=True,
    n_occ=40,
    geometry=None,
    seed=0,
    owner="BIG MINING LTD",
) -> Project:
    """Create ``demo/`` in the current directory with the requested harvested layers."""
    aoi = make_aoi(tmp_path / "aoi.gpkg", geometry)
    assert runner.invoke(app, ["new", "demo", "--aoi", str(aoi)]).exit_code == 0
    project = Project(Path("demo"))
    assert project.config.crs == CRS
    rng = np.random.default_rng(seed)

    def write(name, gdf):
        gdf.to_file(project.data_gpkg, layer=name, driver="GPKG", engine="pyogrio")

    if "minfile" in layers:  # occurrences prefer the east, where the planted signal is high
        n_east = int(n_occ * 0.85)
        xs = np.concatenate([rng.uniform(X0 + 0.55 * W, X0 + W, n_east), rng.uniform(X0, X0 + W, n_occ - n_east)])
        ys = rng.uniform(Y0, Y0 + H, n_occ)
        commodities = np.where(np.arange(n_occ) % 3 == 0, "Gold", np.where(np.arange(n_occ) % 3 == 1, "Copper", "Silver"))
        write(
            "minfile",
            gpd.GeoDataFrame(
                {"MINFILE_NUMBER": range(n_occ), "COMMODITY_DESCRIPTION1": commodities},
                geometry=[Point(x, y) for x, y in zip(xs, ys, strict=True)], crs=CRS,
            ),
        )  # fmt: skip
    if "mto_tenure" in layers:
        boxes = [box(X0 + 2500, Y0 + 500, X0 + 3600, Y0 + 1500), box(X0 + 200, Y0 + 2000, X0 + 900, Y0 + 2800)]
        write(
            "mto_tenure",
            gpd.GeoDataFrame(
                {
                    "TENURE_NUMBER_ID": [1001, 1002],
                    "CLAIM_NAME": ["EAST BLOCK", "NORTH WEST"],
                    "TENURE_TYPE_DESCRIPTION": ["Mineral", "Mineral"],
                    "ISSUE_DATE": ["2026-03-01", "2005-01-01"],
                    "GOOD_TO_DATE": ["2099-03-01", "2090-01-01"],  # far ahead so tests do not age out
                    "AREA_IN_HECTARES": [110.0, 56.0],
                    "OWNER_NAME": [owner, "OLD OWNER INC"],
                },
                geometry=boxes, crs=CRS,
            ),
        )  # fmt: skip
    if "aris_index" in layers:
        ax, ay = rng.uniform(X0, X0 + W, 30), rng.uniform(Y0, Y0 + H, 30)
        # Reports 501-503 sit inside the east tenure so it has a history.
        ax[:3], ay[:3] = [X0 + 2700, X0 + 3000, X0 + 3300], [Y0 + 800, Y0 + 900, Y0 + 1000]
        write(
            "aris_index",
            gpd.GeoDataFrame(
                {
                    "ARIS_NUMBER": [501, 502, 503, *range(600, 627)],
                    "REPORT_YEAR": [1965, 1985, 1998, *rng.integers(1960, 2020, 27)],
                    "ASSESSMENT_REPORT_TITLE": [f"Report {n}" for n in [501, 502, 503, *range(600, 627)]],
                    "OPERATOR_NAMES": "Test Exploration Ltd.",
                    "GENERAL_WORK_TYPES": "Drilling; Geochemical",
                },
                geometry=[Point(x, y) for x, y in zip(ax, ay, strict=True)], crs=CRS,
            ),
        )  # fmt: skip
    if "rgs" in layers:
        gx, gy = rng.uniform(X0, X0 + W, 25), rng.uniform(Y0, Y0 + H, 25)
        write(
            "rgs",
            gpd.GeoDataFrame(
                {"COPPER_AAS_PPM": 20 + (gx - X0) / 40, "COPPER_ICPMS_PPM": [np.nan] * 25, "GOLD_INAA_PPB": rng.uniform(1, 9, 25)},
                geometry=[Point(x, y) for x, y in zip(gx, gy, strict=True)], crs=CRS,
            ),
        )  # fmt: skip
    if raster:
        (project.root / "rasters").mkdir()
        data = np.tile(np.linspace(0, 1, 80, dtype=np.float32), (60, 1))  # values rise from west to east
        with rasterio.open(
            project.root / "rasters" / "synth_magnetics.tif", "w", driver="GTiff", width=80, height=60, count=1,
            dtype="float32", crs=CRS, transform=from_bounds(X0 - 200, Y0 - 200, X0 + W + 200, Y0 + H + 200, 80, 60), nodata=-9999.0,
        ) as ds:  # fmt: skip
            ds.write(data, 1)
    project.config.target = small_cfg()
    project.config.save(project.root)
    return Project(project.root)
