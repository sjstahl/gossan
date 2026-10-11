"""Stream-sediment catchments: depression filling, D8 routing, delineation, and their use in the
geochemistry scores and target features."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box
from typer.testing import CliRunner

from gossan.analyze.catchments import (
    accumulate,
    catchment_polygons,
    delineate,
    fill_depressions,
    flow_directions,
)
from gossan.analyze.geochem import background_residuals, catchment_covariates
from gossan.cli import app
from gossan.core.project import Project

CRS = "EPSG:32610"
X0, Y0 = 690_000.0, 5_617_000.0
CELL = 50.0
NX, NY = 80, 60  # a 4 km x 3 km DEM
XA, XB = X0 + 1025, X0 + 3025  # two valley axes on cell centres, a ridge at X0 + 2025
runner = CliRunner()


def two_valleys() -> np.ndarray:
    """Two parallel valleys draining south, separated by a ridge."""
    cols, rows = np.meshgrid(np.arange(NX), np.arange(NY))
    x = X0 + (cols + 0.5) * CELL
    y_from_south = (NY - rows - 0.5) * CELL
    return (1000 + 0.05 * y_from_south + 0.2 * np.minimum(np.abs(x - XA), np.abs(x - XB))).astype(np.float32)


def write_dem(path: Path, z: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", width=z.shape[1], height=z.shape[0], count=1, dtype="float32",
        crs=CRS, transform=from_origin(X0, Y0 + NY * CELL, CELL, CELL), nodata=-9999.0,
    ) as ds:  # fmt: skip
        ds.write(z, 1)
    return path


# --- hydrology ------------------------------------------------------------------------------


def test_fill_removes_pits_and_keeps_every_cell_draining():
    z = np.full((7, 7), 10.0)
    z[1:-1, 1:-1] = 8.0  # a closed basin
    z[3, 3] = 5.0  # with a deeper pit
    z[0, 3] = 4.0  # the only outlet, on the edge
    filled = fill_depressions(z)
    assert (filled >= z).all()
    down = flow_directions(filled)
    interior = [(r, c) for r in range(1, 6) for c in range(1, 6)]
    for r, c in interior:  # every interior cell has a strictly lower downstream neighbour
        d = down[r * 7 + c]
        assert d >= 0 and filled.ravel()[d] < filled[r, c]
    acc = accumulate(filled, down)
    assert acc.ravel()[down[1 * 7 + 3]] >= 25  # the whole basin leaves past the outlet


def test_nodata_is_left_alone_and_acts_as_an_outlet():
    z = np.arange(25, dtype=float).reshape(5, 5)
    z[2, 2] = np.nan
    filled = fill_depressions(z)
    assert np.isnan(filled[2, 2])
    down = flow_directions(filled)
    assert down[12] == -1
    assert accumulate(filled, down)[2, 2] == 0


def test_accumulation_of_a_plane_counts_upslope_cells():
    z = np.tile(np.arange(5, 0, -1, dtype=float), (3, 1))  # slopes down to the east
    acc = accumulate(fill_depressions(z), flow_directions(fill_depressions(z)))
    assert acc[1].tolist() == [1, 2, 3, 4, 5]


# --- delineation ----------------------------------------------------------------------------


@pytest.fixture
def dem(tmp_path) -> Path:
    return write_dem(tmp_path / "dem.tif", two_valleys())


def test_delineate_nested_and_separate_catchments(dem):
    points = gpd.GeoSeries(
        [
            Point(XA + 120, Y0 + 2000),  # A: upper west valley, 120 m off the stream
            Point(XA, Y0 + 500),  # B: lower west valley
            Point(XB, Y0 + 1000),  # C: east valley
        ],
        crs=CRS,
    )
    units = gpd.GeoDataFrame(
        {"unit": ["West", "East"]},
        geometry=[box(X0, Y0, X0 + 2025, Y0 + NY * CELL), box(X0 + 2025, Y0, X0 + NX * CELL, Y0 + NY * CELL)],
        crs=CRS,
    )
    c = delineate(dem, points, resolution=CELL, snap_m=200, stream_km2=0.1, units=units)
    t = c.samples
    assert t.loc[0, "snap_dist_m"] == pytest.approx(120, abs=CELL)  # moved onto the stream
    assert t.loc[0, "downstream_sample"] == 1 and t.loc[1, "downstream_sample"] == -1
    assert t.loc[0, "catchment_area_km2"] < t.loc[1, "catchment_area_km2"]
    assert t.loc[1, "subcatchment_area_km2"] == pytest.approx(
        t.loc[1, "catchment_area_km2"] - t.loc[0, "catchment_area_km2"], rel=0.05
    )
    # Upper west valley: 2 km wide x about 1 km above A.
    assert t.loc[0, "catchment_area_km2"] == pytest.approx(2.0, rel=0.25)
    # The ridge separates the valleys.
    assert c.label_at(np.array([XB]), np.array([Y0 + 2500]))[0] == 2
    assert c.label_at(np.array([XA]), np.array([Y0 + 2500]))[0] == 0
    assert c.unit_names == ["East", "West"]
    assert c.unit_shares[0].tolist() == [0.0, 1.0] and c.unit_shares[2].tolist() == [1.0, 0.0]
    polys = catchment_polygons(c)
    assert sorted(polys["sample"]) == [0, 1, 2]
    assert polys.set_index("sample").area[0] / 1e6 == pytest.approx(t.loc[0, "subcatchment_area_km2"], rel=0.01)


def test_samples_off_the_dem_have_no_catchment(dem):
    c = delineate(dem, gpd.GeoSeries([Point(X0 - 5000, Y0)], crs=CRS), resolution=CELL, snap_m=200)
    assert np.isnan(c.samples.loc[0, "catchment_area_km2"]) and (c.labels == -1).all()


# --- background model ------------------------------------------------------------------------


def test_lithology_levelling_finds_the_anomaly_a_rich_unit_hides():
    rng = np.random.default_rng(0)
    n = 80
    beta = np.r_[np.zeros(40), np.ones(40)]  # share of the Zn-rich unit in each catchment
    log_zn = 1.9 + 0.6 * beta + rng.normal(0, 0.05, n)
    log_zn[5] += 0.3  # a real anomaly in the poor unit, still below the rich unit's background
    logs = pd.DataFrame({
        "ZINC": log_zn,
        "IRON": rng.normal(0.4, 0.05, n),
        "MANGANESE": rng.normal(2.8, 0.1, n),
    })  # fmt: skip
    table = pd.DataFrame({"catchment_area_km2": rng.uniform(1, 20, n)})
    cov = catchment_covariates(table, ["Alpha", "Beta"], np.column_stack([1 - beta, beta]), min_unit_share=0.02)
    assert list(cov.columns) == ["log_catchment_area", "unit: Beta"] or list(cov.columns) == ["log_catchment_area", "unit: Alpha"]
    plain = background_residuals(logs, None)["ZINC"]
    levelled = background_residuals(logs, cov)["ZINC"]
    assert plain.idxmax() != 5  # Fe-Mn alone cannot see past the rich unit
    assert levelled.idxmax() == 5


def test_rows_without_catchments_keep_the_scavenging_residual():
    rng = np.random.default_rng(1)
    logs = pd.DataFrame({"ZINC": rng.normal(2, 0.2, 40), "IRON": rng.normal(0.4, 0.05, 40), "MANGANESE": rng.normal(2.8, 0.1, 40)})
    cov = pd.DataFrame({"log_catchment_area": rng.normal(0.7, 0.3, 40)})
    cov.iloc[:3] = np.nan
    plain, extended = background_residuals(logs, None), background_residuals(logs, cov)
    assert np.allclose(extended["ZINC"].iloc[:3], plain["ZINC"].iloc[:3])
    assert not np.allclose(extended["ZINC"].iloc[3:], plain["ZINC"].iloc[3:])


# --- through the commands --------------------------------------------------------------------


@pytest.fixture
def project(tmp_path) -> Project:
    aoi = tmp_path / "aoi.gpkg"
    gpd.GeoDataFrame({"id": [1]}, geometry=[box(X0 + 100, Y0 + 100, X0 + 3900, Y0 + 2900)], crs=CRS).to_file(aoi, driver="GPKG")
    assert runner.invoke(app, ["new", "demo", "--aoi", str(aoi)]).exit_code == 0
    proj = Project(Path("demo"))
    write_dem(proj.root / "rasters" / "dem.tif", two_valleys())
    rng = np.random.default_rng(2)
    n = 40
    x = np.where(np.arange(n) % 2 == 0, XA, XB) + rng.normal(0, 30, n)
    y = rng.uniform(Y0 + 300, Y0 + 2700, n)
    rgs = gpd.GeoDataFrame(
        {
            "MASTER_ID": [f"S{i:02d}" for i in range(n)],
            "ZINC_AAS_PPM": np.round(rng.lognormal(np.log(80), 0.3, n)),
            "LEAD_AAS_PPM": np.round(rng.lognormal(np.log(12), 0.3, n)),
            "COPPER_AAS_PPM": np.round(rng.lognormal(np.log(20), 0.3, n)),
            "IRON_AAS_PERCENTAGE": np.round(rng.lognormal(np.log(2.5), 0.2, n), 2),
            "MANGANESE_AAS_PPM": np.round(rng.lognormal(np.log(600), 0.3, n)),
        },
        geometry=[Point(a, b) for a, b in zip(x, y, strict=True)],
        crs=CRS,
    )
    rgs.to_file(proj.data_gpkg, layer="rgs", driver="GPKG")
    proj.config.geochem.stream_km2 = 0.1
    proj.config.geochem.dilution_correction = True
    proj.config.save(proj.root)
    return Project(proj.root)


def test_analyze_geochem_writes_catchments(project):
    result = runner.invoke(app, ["analyze", "demo", "--steps", "geochem"])
    assert result.exit_code == 0, result.output
    assert "catchments from the DEM at 50 m" in result.output
    corrected = result.output.split("scores corrected for:")[1].splitlines()[0]
    assert "log_catchment_area" in corrected and "unit:" not in corrected
    out = project.root / "analysis" / "analysis.gpkg"
    scores = gpd.read_file(out, layer="rgs_scores")
    assert scores["catchment_area_km2"].notna().all() and (scores["snap_dist_m"] <= 200).all()
    polys = gpd.read_file(out, layer="rgs_catchments")
    assert len(polys) >= 30 and {"MASTER_ID", "idx_sediment_hosted_zn_pb", "catchment_area_km2"} <= set(polys.columns)
    record = next(r for r in project.provenance.read() if r.dataset == "analysis_geochem")
    assert record.parameters["inputs"] == ["rgs", "dem"]


def test_without_a_dem_samples_stay_points(project):
    (project.root / "rasters" / "dem.tif").unlink()
    result = runner.invoke(app, ["analyze", "demo", "--steps", "geochem"])
    assert result.exit_code == 0, result.output
    assert "harvest --source dem" in result.output
    layers = set(gpd.list_layers(project.root / "analysis" / "analysis.gpkg")["name"])
    assert "rgs_catchments" not in layers


def test_target_features_follow_catchments(project):
    from gossan.analyze.geochem import score_project
    from gossan.core.config import TargetConfig
    from gossan.target.features import _geochem_indices
    from gossan.target.grid import build_grid

    grid = build_grid(project.read_aoi(), CRS, 100.0, 10**6)
    rgs = gpd.read_file(project.data_gpkg, layer="rgs").to_crs(CRS)
    scored = score_project(project, rgs)
    labels = scored.catchments.label_at(grid.xy[:, 0], grid.xy[:, 1])
    idx = scored.result.samples["idx_sediment_hosted_zn_pb"].to_numpy()

    inputs: list[str] = []
    by_catchment = _geochem_indices(project, grid, TargetConfig(), inputs)["rgs_idx_sediment_hosted_zn_pb"][0]
    by_idw = _geochem_indices(project, grid, TargetConfig(geochem_spatial="idw"), [])["rgs_idx_sediment_hosted_zn_pb"][0]
    drained = labels >= 0
    assert drained.mean() > 0.3
    assert np.allclose(by_catchment[drained], idx[labels[drained]])
    assert np.allclose(by_catchment[~drained], by_idw[~drained])  # IDW where no sample drains the cell
    assert not np.allclose(by_catchment, by_idw)
    assert inputs == ["rgs", "dem"]


def test_corrections_are_opt_in_and_lithology_is_always_reported(project):
    import json

    project.config.geochem.dilution_correction = False
    project.config.save(project.root)
    result = runner.invoke(app, ["analyze", "demo", "--steps", "geochem"])
    assert result.exit_code == 0, result.output
    summary = json.loads((project.root / "analysis" / "geochem_summary.json").read_text())
    assert summary["background_predictors"] == ["log Iron", "log Manganese"]
    assert {"Zinc", "Lead", "Copper"} <= set(summary["catchment_r2_adjusted"])
