"""A synthetic BC-like project shared by the analysis and report tests (invented data)."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, box
from typer.testing import CliRunner

from gossan.cli import app
from gossan.core.project import Project
from gossan.harvest.bc_provincial.wfs import ACCESS_ONLY_BC, OGL_BC

CRS = "EPSG:32610"
X0, Y0, W, H = 690_000.0, 5_617_000.0, 4000.0, 3000.0
EAST = X0 + 3000  # the "Aldridge" unit is the eastern quarter
runner = CliRunner()


def _write(project: Project, name: str, gdf: gpd.GeoDataFrame, licence: str = OGL_BC) -> None:
    gdf.to_file(project.data_gpkg, layer=name, driver="GPKG", engine="pyogrio")
    project.provenance.log(
        dataset=name, source="test", identifier=name, url=None, license=licence, operation="fetch"
    )


def _rgs(rng: np.random.Generator, n: int = 60) -> gpd.GeoDataFrame:
    x, y = rng.uniform(X0, X0 + W, n), rng.uniform(Y0, Y0 + H, n)
    iron = rng.lognormal(np.log(2.5), 0.2, n)
    mn = rng.lognormal(np.log(600), 0.3, n)
    df = pd.DataFrame(
        {
            "MASTER_ID": [f"S{i:03d}" for i in range(n)],
            "ZINC_AAS_PPM": np.round(rng.lognormal(np.log(80), 0.3, n)),
            "LEAD_AAS_PPM": np.round(rng.lognormal(np.log(12), 0.3, n)),
            "COPPER_AAS_PPM": np.round(rng.lognormal(np.log(20), 0.3, n) * iron / 2.5),
            "NICKEL_AAS_PPM": np.round(rng.lognormal(np.log(30), 0.3, n)),
            "BARIUM_INAA_PPM": np.round(rng.lognormal(np.log(700), 0.3, n)),
            "IRON_AAS_PERCENTAGE": iron.round(2),
            "MANGANESE_AAS_PPM": np.round(mn),
            "CADMIUM_AAS_PPM": np.zeros(n),  # below detection except a few
            "SILVER_AAS_PPM": np.zeros(n),  # never detected: dropped
            "GOLD_INAA_PPB": np.round(rng.lognormal(np.log(4), 0.4, n)),
            "GOLD_ICPMS_PPB": np.full(n, np.nan),  # a method nobody used here
            "WATER_URANIUM_PPB": rng.uniform(0, 1, n),  # water chemistry: never a sediment element
        }
    )
    df.loc[[5, 17, 33], "CADMIUM_AAS_PPM"] = [1.0, 1.5, 2.0]
    # Planted sediment-hosted Zn-Pb anomaly, ordinary Fe and Mn.
    df.loc[0, ["ZINC_AAS_PPM", "LEAD_AAS_PPM", "CADMIUM_AAS_PPM", "BARIUM_INAA_PPM"]] = [2500, 400, 30, 4000]
    df.loc[0, ["IRON_AAS_PERCENTAGE", "MANGANESE_AAS_PPM"]] = [2.5, 600]
    return gpd.GeoDataFrame(df, geometry=[Point(a, b) for a, b in zip(x, y, strict=True)], crs=CRS)


def build_analysis_project(tmp_path: Path, *, with_inventory: bool = True) -> Project:
    """A small BC-like project in the current directory: two bedrock units, a thrust fault,
    20 occurrences (12 metallic veins in the east, 8 gypsum beds in the west), 60 RGS samples with
    a planted Zn-Pb anomaly, and (unless ``with_inventory`` is false) an Access Only inventory layer."""
    aoi = tmp_path / "aoi.gpkg"
    gpd.GeoDataFrame({"id": [1]}, geometry=[box(X0, Y0, X0 + W, Y0 + H)], crs=CRS).to_file(aoi, driver="GPKG")
    assert runner.invoke(app, ["new", "demo", "--aoi", str(aoi)]).exit_code == 0
    proj = Project(Path("demo"))
    rng = np.random.default_rng(3)

    # 12 metallic veins in the east (Aldridge) near the fault; 8 gypsum beds in the west.
    mx, my = rng.uniform(EAST + 50, X0 + W - 50, 12), rng.uniform(Y0 + 50, Y0 + H - 50, 12)
    ix, iy = rng.uniform(X0 + 50, EAST - 50, 8), rng.uniform(Y0 + 50, Y0 + H - 50, 8)
    minfile = gpd.GeoDataFrame(
        {
            "MINFILE_NUMBER": [f"M{i:02d}" for i in range(20)],
            "MINFILE_NAME1": [f"OCC {i}" for i in range(20)],
            "DEPOSIT_TYPE_CODE1": ["I05"] * 12 + ["F02"] * 8,
            "DEPOSIT_TYPE_DESCRIPTION1": ["Polymetallic veins"] * 12 + ["Bedded gypsum"] * 8,
        },
        geometry=[Point(a, b) for a, b in zip([*mx, *ix], [*my, *iy], strict=True)],
        crs=CRS,
    )
    _write(proj, "minfile", minfile)
    bedrock = gpd.GeoDataFrame(
        {
            "STRATIGRAPHIC_NAME": ["Purcell Supergroup - Aldridge Formation", "Purcell Supergroup - Creston Formation"],
            "GEOLOGICAL_PERIOD": ["Middle Proterozoic", "Middle Proterozoic"],
            "ROCK_CLASS": ["sedimentary rocks", "sedimentary rocks"],
        },
        geometry=[box(EAST, Y0 - 500, X0 + W + 500, Y0 + H + 500), box(X0 - 500, Y0 - 500, EAST, Y0 + H + 500)],
        crs=CRS,
    )
    _write(proj, "bedrock_geology", bedrock)
    faults = gpd.GeoDataFrame(
        {"FAULT_TYPE": ["Thrust"]}, geometry=[LineString([(X0 + 3500, Y0 - 100), (X0 + 3500, Y0 + H + 100)])], crs=CRS
    )
    _write(proj, "faults", faults)
    _write(proj, "rgs", _rgs(rng))
    if with_inventory:  # the project's one Access Only layer
        inventory = gpd.GeoDataFrame({"MINFILE_NUMBER": ["M00", "M00", "M03"]}, geometry=[Point(mx[0], my[0])] * 3, crs=CRS)
        _write(proj, "minfile_inventory", inventory, ACCESS_ONLY_BC)
    return proj

