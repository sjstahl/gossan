"""Digitized drill holes: placement, grades, units and classification."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pyogrio
import pytest
from analysis_project import CRS, X0, Y0, build_analysis_project, runner
from pyproj import Transformer
from shapely.geometry import Point

from gossan.analyze.drill_results import (
    build_drill_results,
    collar_crs,
    grade_intervals,
)
from gossan.cli import app
from gossan.core.config import DrillConfig
from gossan.core.project import Project
from gossan.digitize.db import COLUMNS

CX, CY = X0 + 1500.0, Y0 + 1500.0  # the middle of the AOI


def long_rows(aris: int, hole: str, items: list[tuple]) -> list[dict]:
    """Long-format records: items are (attribute, value, unit, from_m, to_m, flag)."""
    rows = []
    for attribute, value, unit, from_m, to_m, flag in items:
        rows.append({
            "hole_id": hole, "from_m": from_m, "to_m": to_m, "attribute": attribute, "unit": unit, "value": value,
            "flag": flag, "raw_value": str(value), "aris_number": aris, "page": 1, "review_item": "R0001",
            "source_file": "x.csv", "approved_at": "2026-10-11", "approved_by": "tester",
        })  # fmt: skip
    return rows


def collar(aris: int, hole: str, east=None, north=None, unit="m", length=None) -> list[dict]:
    items = []
    if east is not None:
        items += [("Easting", east, unit, None, None, ""), ("Northing", north, unit, None, None, "")]
    if length is not None:
        items.append(("Length", length, "m", None, None, ""))
    return long_rows(aris, hole, items or [("Dip", -60, "deg", None, None, "")])


def assays(aris: int, hole: str, element: str, grades: list[float], unit: str, *, step=5.0, flag="") -> list[dict]:
    return long_rows(aris, hole, [(element, g, unit, i * step, (i + 1) * step, flag) for i, g in enumerate(grades)])


def write_tables(project: Project, collar_rows: list[dict], assay_rows: list[dict]) -> None:
    for name, rows in (("collar", collar_rows), ("assay", assay_rows)):
        df = pd.DataFrame(rows, columns=list(COLUMNS)).astype({"aris_number": "Int64", "page": "Int64"})
        pyogrio.write_dataframe(df, project.data_gpkg, layer=name, driver="GPKG")


@pytest.fixture
def project(tmp_path) -> Project:
    return build_analysis_project(tmp_path)


def classes(result) -> dict[str, str]:
    return dict(zip(result.holes["key"], result.holes["class"], strict=True))


# --- classification -----------------------------------------------------------------------------------------


def test_holes_are_classified_from_their_assays(project):
    write_tables(
        project,
        [row for h in ("MIN", "BAR", "MARG", "FEW", "CEN", "NOUNIT") for row in collar(1, h, CX, CY)],
        [
            *assays(1, "MIN", "Au", [20, 800, 30, 10, 10, 10], "ppb"),  # 0.8 g/t: above 0.3
            *assays(1, "BAR", "Au", [10, 20, 15, 30, 10, 20], "ppb"),  # 30 ppb: well under half of 0.3 g/t, 30 m sampled
            *assays(1, "MARG", "Au", [10, 200, 50, 10, 10, 10], "ppb"),  # 0.2 g/t: over half the threshold, not over it
            *assays(1, "FEW", "Au", [10, 20, 10], "ppb"),  # 15 m, 3 samples: too little to call barren
            *assays(1, "CEN", "Au", [5, 5, 5, 5, 5, 5], "ppb", flag="<"),  # all below detection
            *assays(1, "NOUNIT", "Au", [900, 900, 900, 900, 900, 900], ""),  # could be ppb or g/t: not used
        ],
    )
    result = build_drill_results(project)
    assert classes(result) == {
        "1:MIN": "mineralized", "1:BAR": "barren", "1:MARG": "inconclusive", "1:FEW": "inconclusive",
        "1:CEN": "barren", "1:NOUNIT": "no_assays",
    }  # fmt: skip
    by_key = result.holes.set_index("key")
    assert by_key.loc["1:MIN", "best_element"] == "Au" and by_key.loc["1:MIN", "best_grade"] == pytest.approx(0.8)
    assert by_key.loc["1:MIN", "best_interval"] == "5-10 m" and by_key.loc["1:MIN", "exceeded"] == "Au"
    assert by_key.loc["1:BAR", "metres_sampled"] == 30.0 and by_key.loc["1:BAR", "n_samples"] == 6
    assert result.summary["left_out_of_grades"]["no_unit"] == 6
    assert result.summary["by_class"] == {"mineralized": 1, "barren": 2, "inconclusive": 2, "no_assays": 1}


def test_units_are_converted_to_one_scale_per_element():
    items = [
        ("Cu", 1500, "ppm", 0, 5, ""), ("Cu", 0.2, "%", 5, 10, ""), ("Au", 0.02, "oz/t", 0, 5, ""),
        ("Au", 50, "ppb", 5, 10, ""), ("Gold", 1, "g/t", 10, 15, ""), ("Zn", 3, "mg/kg", 0, 5, ""),
    ]  # fmt: skip
    rows = pd.DataFrame(long_rows(1, "H", items)).assign(key="1:H")
    graded, left = grade_intervals(rows)
    by = {(r.element, r.from_m): r.grade for r in graded.itertuples()}
    assert by[("Cu", 0.0)] == pytest.approx(0.15) and by[("Cu", 5.0)] == pytest.approx(0.2)
    assert by[("Au", 0.0)] == pytest.approx(0.02 * 34.2857) and by[("Au", 5.0)] == pytest.approx(0.05) and by[("Au", 10.0)] == 1.0
    assert left["unknown_unit"] == 1  # mg/kg is not understood: left out, not guessed


def test_censored_values_count_as_zero_and_only_listed_elements_are_graded():
    rows = pd.DataFrame(
        long_rows(1, "H", [("Au", 5, "ppb", 0, 5, "<"), ("Au", 120, "ppb", 5, 10, ">"), ("Cd", 4, "ppm", 0, 5, ""), ("Lithology", None, "", 0, 5, "")])
    ).assign(key="1:H")
    graded, _ = grade_intervals(rows)
    assert list(graded["element"]) == ["Au", "Au"] and list(graded["grade"]) == [0.0, pytest.approx(0.12)]


def test_surface_samples_without_depths_are_counted_not_graded(project):
    surface = long_rows(2, "S1", [("Au", 900, "ppb", None, None, "")])
    write_tables(project, collar(2, "D1", CX, CY), [*assays(2, "D1", "Au", [10, 10, 10, 10, 10, 10], "ppb"), *surface])
    result = build_drill_results(project)
    assert result.summary["n_surface_sample_ids"] == 1 and result.summary["left_out_of_grades"]["not_interval"] == 1
    assert list(result.holes["key"]) == ["2:D1"]  # the surface sample is not a hole


def test_the_same_hole_id_in_two_reports_is_two_holes(project):
    write_tables(
        project,
        [*collar(1, "DDH-1", CX, CY), *collar(2, "DDH-1", CX + 100, CY)],
        [*assays(1, "DDH-1", "Cu", [0.5] * 6, "%"), *assays(2, "DDH-1", "Cu", [0.001] * 6, "%")],
    )
    assert classes(build_drill_results(project)) == {"1:DDH-1": "mineralized", "2:DDH-1": "barren"}


def test_thresholds_come_from_the_project_settings(project):
    write_tables(project, collar(1, "H", CX, CY), assays(1, "H", "Au", [10, 200, 10, 10, 10, 10], "ppb"))  # 0.2 g/t
    assert classes(build_drill_results(project))["1:H"] == "inconclusive"
    project.config.drill = DrillConfig(thresholds={"Au": 0.1})
    assert classes(build_drill_results(project))["1:H"] == "mineralized"


# --- placement --------------------------------------------------------------------------------------------------


def test_collars_are_placed_by_coordinates_or_flagged(project):
    far_e, far_n = X0 + 60_000.0, Y0 + 60_000.0
    write_tables(
        project,
        [
            *collar(1, "IN", CX, CY), *collar(1, "OUT", far_e, far_n), *collar(1, "SWAP", CY, CX),  # easting and northing swapped
            *collar(1, "NOXY"),
        ],
        [row for h in ("IN", "OUT", "SWAP", "NOXY") for row in assays(1, h, "Au", [10] * 6, "ppb")],
    )
    result = build_drill_results(project)
    by = result.holes.set_index("key")
    assert by.loc["1:IN", "location_quality"] == "collar_coordinates" and by.loc["1:IN"].geometry.distance(Point(CX, CY)) < 3
    assert by.loc["1:SWAP", "location_quality"] == "collar_coordinates" and "swapped" in by.loc["1:SWAP", "location_note"]
    assert by.loc["1:SWAP"].geometry.distance(Point(CX, CY)) < 3
    assert by.loc["1:OUT", "location_quality"] == "unplaced" and "outside the AOI" in by.loc["1:OUT", "location_note"]
    assert by.loc["1:NOXY", "location_quality"] == "unplaced"
    assert result.summary["by_location"] == {"collar_coordinates": 2, "report_location": 0, "unplaced": 2}


def test_an_unplaced_hole_falls_back_to_its_reports_location(project):
    aris = gpd.GeoDataFrame({"ARIS_NUMBER": [7], "REPORT_YEAR": [1990]}, geometry=[Point(X0 + 800, Y0 + 900)], crs=CRS)
    aris.to_file(project.data_gpkg, layer="aris_index", driver="GPKG")
    write_tables(project, collar(7, "H1"), assays(7, "H1", "Au", [900] * 6, "ppb"))
    hole = build_drill_results(project).holes.iloc[0]
    assert hole["location_quality"] == "report_location" and hole.geometry.distance(Point(X0 + 800, Y0 + 900)) < 1
    assert "ARIS report" in hole["location_note"]


def test_nad27_coordinates_are_read_in_the_datum_set_for_the_project(project):
    project.config.georef.datum = "NAD27"
    assert collar_crs(project) == "EPSG:26710"
    e27, n27 = Transformer.from_crs(CRS, "EPSG:26710", always_xy=True).transform(CX, CY)
    assert abs(e27 - CX) + abs(n27 - CY) > 20  # NAD27 really does differ here, so reading it as NAD83 would misplace the hole
    write_tables(project, collar(1, "H", e27, n27), assays(1, "H", "Au", [10] * 6, "ppb"))
    assert build_drill_results(project).holes.iloc[0].geometry.distance(Point(CX, CY)) < 3
    project.config.georef.gcp_crs = "EPSG:32610"  # an explicit CRS wins over the datum
    assert collar_crs(project) == "EPSG:32610"


def test_coordinates_in_feet_are_converted(project):
    write_tables(project, collar(1, "H", CX / 0.3048, CY / 0.3048, unit="ft"), assays(1, "H", "Au", [10] * 6, "ppb"))
    assert build_drill_results(project).holes.iloc[0].geometry.distance(Point(CX, CY)) < 3


# --- the step, outputs and command ---------------------------------------------------------------------------------


def test_outputs_inherit_the_access_only_licence_and_only_placed_holes_are_mapped(project):
    write_tables(project, [*collar(1, "IN", CX, CY), *collar(1, "NOXY")], [*assays(1, "IN", "Au", [900] * 6, "ppb"), *assays(1, "NOXY", "Au", [10] * 6, "ppb")])
    result = build_drill_results(project)
    out = project.root / "analysis"
    assert len(pd.read_csv(out / "drill_holes.csv")) == 2  # the table lists every hole
    mapped = gpd.read_file(out / "analysis.gpkg", layer="drill_holes")
    assert list(mapped["key"]) == ["1:IN"]  # the layer only the placed one
    assert result.summary["mineralized_holes"][0]["hole"] == "IN (ARIS 1)"
    record = next(r for r in project.provenance.read() if r.dataset == "analysis_drilling")
    assert "Access Only" in record.license and "aris_pdfs" in record.license


def test_no_approved_tables_is_reported_not_an_error(project):
    result = build_drill_results(project)
    assert result.summary["available"] is False and "gossan review" in result.summary["reason"]
    assert result.outputs == ["analysis/drill_summary.json"]
    out = runner.invoke(app, ["analyze", "demo", "--steps", "drilling"])
    assert out.exit_code == 0 and "approve the extracted tables" in out.output


def test_command_output_summarises_the_holes(project):
    write_tables(project, [*collar(1, "A", CX, CY), *collar(1, "B", CX + 200, CY)], [*assays(1, "A", "Cu", [0.5] * 6, "%"), *assays(1, "B", "Cu", [0.001] * 6, "%")])
    out = runner.invoke(app, ["analyze", "demo", "--steps", "drilling"])
    assert out.exit_code == 0, out.output
    assert "2 hole(s) from 1 report(s): 1 mineralized, 1 barren" in out.output
    assert "placed by collar coordinates 2" in out.output and "mineralized: A (ARIS 1) Cu 0.5 %" in out.output
