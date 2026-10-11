"""gossan analyze: inventory, geochemistry and occurrences-versus-geology on a synthetic project."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from analysis_project import (
    ACCESS_ONLY_BC,
    CRS,
    EAST,
    X0,
    Y0,
    H,
    W,
    _rgs,
    _write,
    build_analysis_project,
    runner,
)
from shapely.geometry import LineString, Point, box

from gossan.analyze.geochem import choose_columns, normal_scores
from gossan.analyze.occurrences import commodity_class, deposit_family, enrichment
from gossan.cli import app
from gossan.core.licensing import uses_aris
from gossan.core.project import Project


@pytest.fixture
def project(tmp_path) -> Project:
    return build_analysis_project(tmp_path)

# --- unit tests -------------------------------------------------------------------------


def test_choose_columns_tiers():
    df = _rgs(np.random.default_rng(0))
    choices = {c.element: c for c in choose_columns(df)}
    assert choices["ZINC"].tier == "continuous"
    assert choices["CADMIUM"].tier == "censored"  # 4 detections above zero
    assert choices["SILVER"].tier == "dropped" and "floor" in choices["SILVER"].dropped
    assert choices["GOLD"].column == "GOLD_INAA_PPB"  # the method with values wins
    assert "WATER" not in choices


def test_normal_scores_handle_ties_and_gaps():
    x = pd.DataFrame({"a": [0.0, 0.0, 0.0, 1.0, 2.0, np.nan]})
    z = normal_scores(x)["a"]
    assert np.isnan(z.iloc[5])
    assert z.iloc[0] == z.iloc[1] == z.iloc[2]  # ties share a score
    assert z.iloc[4] > z.iloc[3] > z.iloc[0]
    assert z.abs().max() < 3


def test_deposit_family_and_class():
    assert deposit_family("E12") == "Sediment-hosted"
    assert deposit_family(" l04") == "Porphyry"
    assert deposit_family(None) == "Unclassified"
    assert commodity_class("F04") == "industrial"
    assert commodity_class("K01") == "metallic"
    assert commodity_class("N01") == "specialty"


def test_enrichment_ratio_and_p_value():
    groups = pd.Series(["A"] * 8 + ["B"] * 2)
    areas = pd.Series({"A": 25.0, "B": 75.0})
    t = enrichment(groups, areas, 100.0, 10)
    assert t.loc["A", "expected"] == pytest.approx(2.5)
    assert t.loc["A", "ratio"] == pytest.approx(3.2)
    assert t.loc["A", "p_value"] < 0.01 < t.loc["B", "p_value"]


# --- the command ------------------------------------------------------------------------


def test_analyze_all_steps(project):
    result = runner.invoke(app, ["analyze", "demo"])
    assert result.exit_code == 0, result.output
    out = project.root / "analysis"
    for f in ("inventory.csv", "geochem_elements.csv", "geochem_pca_loadings.csv", "geochem_summary.json",
              "occurrence_geology.csv", "occurrence_faults.csv", "occurrence_summary.json", "analysis.gpkg"):  # fmt: skip
        assert (out / f).exists(), f
    assert "Aldridge Formation" in result.output

    # Geochemistry: the planted Zn-Pb-Cd-Ba sample leads the sediment-hosted index and is flagged.
    scores = gpd.read_file(out / "analysis.gpkg", layer="rgs_scores")
    top = scores.nlargest(1, "idx_sediment_hosted_zn_pb").iloc[0]
    assert top["MASTER_ID"] == "S000"
    assert "Zinc" in top["top_elements"] and "Cadmium" in top["top_elements"]
    assert bool(scores.set_index("MASTER_ID").loc["S000", "multivariate_anomaly"])
    assert {"zinc_ppm", "zinc_z", "zinc_zc", "cadmium_z", "PC1", "anomaly_d2"} <= set(scores.columns)
    assert "silver_ppm" not in scores

    # Occurrences: metallic veins are enriched in the Aldridge and close to the fault.
    geo = pd.read_csv(out / "occurrence_geology.csv")
    ald = geo[(geo["level"] == "unit") & (geo["occurrences"] == "metallic") & geo["group"].str.contains("Aldridge")].iloc[0]
    assert ald["n_occurrences"] == 12 and ald["ratio"] == pytest.approx(4.0, rel=0.05)
    occ = gpd.read_file(out / "analysis.gpkg", layer="occurrences")
    assert set(occ["commodity_class"]) == {"metallic", "industrial"}
    metallic = occ[occ["commodity_class"] == "metallic"]
    assert metallic["dist_all_fault_m"].max() <= 500
    assert occ.set_index("MINFILE_NUMBER").loc["M00", "n_inventory_records"] == 2
    assert occ["nearest_rgs_master_id"].notna().all()  # geochem ran first

    inv = pd.read_csv(out / "inventory.csv")
    assert set(inv["dataset"]) >= {"minfile", "bedrock_geology", "faults", "rgs", "minfile_inventory"}
    assert inv.set_index("dataset").loc["minfile_inventory", "license"] == ACCESS_ONLY_BC


def test_licences_follow_inputs(project):
    assert runner.invoke(app, ["analyze", "demo", "--steps", "geochem,occurrences"]).exit_code == 0
    records = {r.dataset: r for r in project.provenance.read() if r.operation.startswith("analyze_")}
    assert "Access Only" not in records["analysis_geochem"].license  # rgs is open
    assert records["analysis_geochem"].parameters["inputs"] == ["rgs"]
    occ = records["analysis_occurrences"]
    assert "Access Only" in occ.license and "minfile_inventory" in occ.license
    assert uses_aris(project.provenance.read())


def test_missing_layer_is_skipped_with_harvest_hint(tmp_path):
    aoi = tmp_path / "aoi.gpkg"
    gpd.GeoDataFrame({"id": [1]}, geometry=[box(X0, Y0, X0 + W, Y0 + H)], crs=CRS).to_file(aoi, driver="GPKG")
    assert runner.invoke(app, ["new", "bare", "--aoi", str(aoi)]).exit_code == 0
    result = runner.invoke(app, ["analyze", "bare", "--steps", "geochem"])
    assert result.exit_code == 1
    assert "harvest --source rgs" in result.output


def test_unknown_step(project):
    result = runner.invoke(app, ["analyze", "demo", "--steps", "magic"])
    assert result.exit_code != 0
    assert "unknown step" in result.output


# --- target decision table -------------------------------------------------------------


def _targets(project: Project) -> None:
    """Three targets: one in a park, one under tenure, one on open ground."""
    out = project.root / "target"
    out.mkdir()
    gpd.GeoDataFrame(
        {
            "rank": [1, 2, 3],
            "target_id": ["T001", "T002", "T003"],
            "n_cells": [4, 4, 4],
            "area_ha": [4.0, 4.0, 4.0],
            "mean_score": [0.9, 0.8, 0.7],
            "max_score": [0.95, 0.85, 0.75],
            "mean_spread": [0.05, 0.05, 0.05],
            "n_known_occurrences": [0, 1, 0],
        },
        geometry=[
            box(X0 + 200, Y0 + 200, X0 + 400, Y0 + 400),  # park
            box(EAST + 200, Y0 + 200, EAST + 400, Y0 + 400),  # tenure
            box(X0 + 1500, Y0 + 2000, X0 + 1700, Y0 + 2200),  # open
        ],
        crs=CRS,
    ).to_file(out / "targets.gpkg", layer="targets_top", driver="GPKG")
    _write(project, "parks_protected", gpd.GeoDataFrame(
        {"PROTECTED_LANDS_NAME": ["TEST PARK"]}, geometry=[box(X0, Y0, X0 + 1000, Y0 + 1000)], crs=CRS))
    _write(project, "mto_tenure", gpd.GeoDataFrame(
        {"TENURE_NUMBER_ID": [1], "TENURE_TYPE_DESCRIPTION": ["Mineral"], "OWNER_NAME": ["ACME LTD"],
         "GOOD_TO_DATE": ["2027-01-31"]},
        geometry=[box(EAST, Y0, X0 + W, Y0 + 1000)], crs=CRS))
    _write(project, "crown_grants", gpd.GeoDataFrame(
        {"LOT_STATUS": ["NOT GRANTED"]}, geometry=[box(X0 + 1400, Y0 + 1900, X0 + 1800, Y0 + 2300)], crs=CRS),
        ACCESS_ONLY_BC)
    _write(project, "roads", gpd.GeoDataFrame(
        {"ROAD_CLASS": ["resource"]}, geometry=[LineString([(X0, Y0 + 2100), (X0 + 1000, Y0 + 2100)])], crs=CRS),
        ACCESS_ONLY_BC)
    _write(project, "aris_index", gpd.GeoDataFrame(
        {"REPORT_YEAR": [1990, 2012], "MERGE_SPENT_2002": [10000.0, 50000.0]},
        geometry=[Point(EAST + 300, Y0 + 300), Point(EAST + 900, Y0 + 300)], crs=CRS), ACCESS_ONLY_BC)  # fmt: skip


def test_target_decisions(project):
    _targets(project)
    result = runner.invoke(app, ["analyze", "demo", "--steps", "geochem,targets"])
    assert result.exit_code == 0, result.output
    t = pd.read_csv(project.root / "analysis" / "target_decisions.csv").set_index("target_id")
    assert list(t["ground_status"]) == ["excluded", "held", "open"]
    assert t.loc["T001", "excluded_share"] == 1.0 and "TEST PARK" in t.loc["T001", "notes"]
    assert t.loc["T002", "tenure_holders"] == "ACME LTD" and t.loc["T002", "earliest_good_to"] == "2027-01-31"
    assert "Aldridge" in t.loc["T002", "main_unit"] and t.loc["T001", "main_unit_share"] == 1.0
    assert t.loc["T002", "n_aris_reports"] == 2 and t.loc["T002", "latest_report_year"] == 2012
    assert t.loc["T002", "reported_spend_2002_dollars"] == 60000
    assert t.loc["T003", "crown_grant_share"] == 0.0  # a "not granted" lot is not private title
    assert t.loc["T003", "dist_road_m"] == pytest.approx(500, abs=1)
    assert t.loc["T003", "n_aris_reports"] == 0 and "no assessment reports" in t.loc["T003", "notes"]
    assert t["best_geochem_index"].notna().all() and t["n_geochem_anomalies"].ge(0).all()
    assert t.loc["T002", "dist_fault_m"] == pytest.approx(100, abs=1)  # fault at X0 + 3500, target edge at X0 + 3400
    assert "open 1" in result.output and "#3 T003" in result.output
    record = next(r for r in project.provenance.read() if r.dataset == "analysis_targets")
    assert "Access Only" in record.license


def test_target_decisions_need_targets(project):
    result = runner.invoke(app, ["analyze", "demo", "--steps", "targets"])
    assert result.exit_code == 1 and "gossan target" in result.output
