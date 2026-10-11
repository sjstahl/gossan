"""Tenure activity: snapshot history from raw/, change detection, and the watchlist."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from analysis_project import CRS, X0, Y0, build_analysis_project, runner
from shapely.geometry import box

from gossan.analyze.tenure import build_tenure_changes, list_snapshots, per_tenure
from gossan.cli import app
from gossan.core.project import Project

TODAY = date(2026, 10, 11)
T0 = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)


def tenure(number, *, owner="ACME LTD", x=0, area=100.0, issue="2020-01-01", good_to="2030-01-01", transfers=None, work=None,
           cash=None, reduction=None, updated="2024-01-01", claim=None) -> dict:  # fmt: skip
    """One tenure row; ``x`` places it 300 m apart from its neighbours inside the 4 x 3 km AOI."""
    return {
        "TENURE_NUMBER_ID": number, "CLAIM_NAME": claim or f"CLAIM {number}", "TENURE_TYPE_DESCRIPTION": "Mineral",
        "OWNER_NAME": owner, "ISSUE_DATE": issue, "GOOD_TO_DATE": good_to, "AREA_IN_HECTARES": area,
        "OWNERSHIP_TRANSFER_EVENT_COUNT": transfers, "STATEMENT_OF_WORK_EVENT_COUNT": work,
        "CASH_IN_LIEU_EVENT_COUNT": cash, "REDUCTION_EVENT_COUNT": reduction, "UPDATE_TIMESTAMP": updated,
        "_box": box(X0 + 100 + 300 * x, Y0 + 100, X0 + 350 + 300 * x, Y0 + 350),
    }  # fmt: skip


def save_snapshot(project: Project, when: datetime, rows: list[dict], *, log: bool = True) -> Path:
    """Write a raw snapshot (WGS84 GeoJSON, as the WFS gives it) and, if ``log``, the sighting."""
    folder = project.raw_dir / "mto_tenure"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"mto_tenure_{when:%Y%m%dT%H%M%S}000000Z.geojson"
    if rows:
        gdf = gpd.GeoDataFrame(
            [{k: v for k, v in r.items() if k != "_box"} for r in rows], geometry=[r["_box"] for r in rows], crs=CRS
        ).to_crs("EPSG:4326")
        path.write_text(gdf.to_json(), encoding="utf-8")
    else:
        path.write_text(json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8")
    if log:
        sight(project, path, when)
    return path


def sight(project: Project, path: Path, when: datetime) -> None:
    project.provenance.log(
        dataset="mto_tenure", source="test", identifier="mto", url=None, license="Open Government Licence - British Columbia",
        retrieved_at=when, operation="fetch", parameters={"file": f"raw/mto_tenure/{path.name}"},
    )  # fmt: skip


def set_layer(project: Project, rows: list[dict]) -> None:
    """The data.gpkg layer (what the latest harvest loaded), in the project CRS."""
    gdf = gpd.GeoDataFrame([{k: v for k, v in r.items() if k != "_box"} for r in rows], geometry=[r["_box"] for r in rows], crs=CRS)
    gdf.to_file(project.data_gpkg, layer="mto_tenure", driver="GPKG", engine="pyogrio")


@pytest.fixture
def project(tmp_path) -> Project:
    return build_analysis_project(tmp_path)


def events(result) -> set[tuple[int, str]]:
    return {(int(r.tenure_number), r.change) for r in result.events.itertuples()}


# --- snapshots -----------------------------------------------------------------------------------


def test_snapshots_collapse_repeat_sightings_and_keep_a_b_a_apart(project):
    a = save_snapshot(project, T0, [tenure(1)])
    sight(project, a, T0 + timedelta(days=1))  # unchanged on the next harvest: the same file again
    b = save_snapshot(project, T0 + timedelta(days=7), [tenure(1, owner="NEW LTD")])
    sight(project, a, T0 + timedelta(days=14))  # the old state is back (e.g. a transfer was reversed)
    snaps = list_snapshots(project)
    assert [s.file for s in snaps] == [a, b, a]
    assert (snaps[0].first_seen, snaps[0].last_seen) == (T0, T0 + timedelta(days=1))
    assert snaps[2].first_seen == snaps[2].last_seen == T0 + timedelta(days=14)


def test_files_without_a_provenance_record_are_dated_by_their_name(project):
    save_snapshot(project, T0, [tenure(1)], log=False)
    save_snapshot(project, T0 + timedelta(days=3), [tenure(1)], log=False)
    snaps = list_snapshots(project)
    assert [s.first_seen for s in snaps] == [T0, T0 + timedelta(days=3)]


def test_a_tenure_with_two_holders_is_one_tenure_whatever_the_row_order():
    rows = [tenure(7, owner="B INC"), tenure(7, owner="A LTD")]
    gdf = gpd.GeoDataFrame([{k: v for k, v in r.items() if k != "_box"} for r in rows], geometry=[r["_box"] for r in rows], crs=CRS)
    merged = per_tenure(gdf)
    assert len(merged) == 1 and merged.loc[7, "owner"] == "A LTD; B INC"
    assert per_tenure(gdf.iloc[::-1]).loc[7, "owner"] == "A LTD; B INC"


# --- changes between snapshots ---------------------------------------------------------------------


def test_every_kind_of_change_is_found_with_its_observation_window(project):
    before = [
        tenure(1, owner="ACME LTD"), tenure(2, x=1), tenure(4, x=3, work=1, good_to="2027-01-01"),
        tenure(5, x=4, area=100.0), tenure(6, x=5, cash=1), tenure(7, x=6, transfers=1), tenure(8, x=7),
        tenure(9, x=8, reduction=None),
    ]  # fmt: skip
    after = [
        tenure(1, owner="NEW OWNER INC"),  # owner changed
        # 2 is gone
        tenure(3, x=2, issue="2026-09-20"),  # new
        tenure(4, x=3, work=2, good_to="2028-01-01"),  # work filed, good-to extended
        tenure(5, x=4, area=80.0),  # reduced
        tenure(6, x=5, cash=2),  # cash in lieu
        tenure(7, x=6, transfers=2),  # transferred
        tenure(8, x=7),  # unchanged
        tenure(9, x=8, area=120.0),  # enlarged
    ]
    save_snapshot(project, T0, before)
    save_snapshot(project, T0 + timedelta(days=10), after)
    set_layer(project, after)
    result = build_tenure_changes(project, today=TODAY)
    assert events(result) == {
        (1, "owner_changed"), (2, "gone"), (3, "new"), (4, "work_filed"), (4, "extended"), (5, "reduced"),
        (6, "cash_in_lieu"), (7, "transferred"), (9, "area_changed"),
    }  # fmt: skip
    by_key = {(int(r.tenure_number), r.change): r for r in result.events.itertuples()}
    assert by_key[(1, "owner_changed")].detail == "ACME LTD -> NEW OWNER INC"
    assert by_key[(4, "work_filed")].detail == "1 -> 2 work filings" and "2027-01-01 -> 2028-01-01" in by_key[(4, "extended")].detail
    assert by_key[(5, "reduced")].detail == "100.0 -> 80.0 ha"
    assert (by_key[(3, "new")].after, by_key[(3, "new")].by) == ("2026-09-01", "2026-09-11")  # it happened in between
    assert result.summary["n_events"] == 9 and result.summary["events_by_type"]["extended"] == 1


def test_a_failed_empty_download_is_not_mistaken_for_every_claim_lapsing(project):
    one = [tenure(1), tenure(2, x=1)]
    save_snapshot(project, T0, one)
    save_snapshot(project, T0 + timedelta(days=1), [])  # the service returned nothing
    save_snapshot(project, T0 + timedelta(days=2), one + [tenure(3, x=2)])
    set_layer(project, one + [tenure(3, x=2)])
    result = build_tenure_changes(project, today=TODAY)
    assert events(result) == {(3, "new")}  # compared with the last good snapshot, across the gap
    assert any("lists no tenure" in n for n in result.summary["notes"]) and result.summary["n_snapshots"] == 2


def test_identical_snapshots_give_no_changes(project):
    rows = [tenure(1), tenure(2, x=1)]
    save_snapshot(project, T0, rows)
    save_snapshot(project, T0 + timedelta(days=5), rows)
    set_layer(project, rows)
    result = build_tenure_changes(project, today=TODAY)
    assert len(result.events) == 0 and result.summary["n_snapshots"] == 2 and result.summary["events_by_type"] == {}


# --- the watchlist -------------------------------------------------------------------------------------


def test_watchlist_flags_expiring_recent_updated_and_churned_tenures(project):
    rows = [
        tenure(1, good_to="2026-11-05"),  # 25 days: expiring
        tenure(2, x=1, good_to="2027-06-01", issue="2026-08-01"),  # recently staked
        tenure(3, x=2, updated="2026-09-20"),  # recently updated
        tenure(4, x=3, transfers=3),  # churn
        tenure(5, x=4, good_to="2026-10-20", issue="2026-09-01", updated="2026-10-01"),  # three flags
        tenure(6, x=5),  # nothing to see
        tenure(7, x=6, good_to="2026-09-01"),  # already past good-to: not "expiring"
    ]
    set_layer(project, rows)
    result = build_tenure_changes(project, today=TODAY)
    watch = result.watch.set_index("tenure_number")
    assert set(watch.index) == {1, 2, 3, 4, 5}
    assert watch.loc[1, "flags"] == "expiring" and watch.loc[1, "days_to_good_to"] == 25
    assert watch.loc[2, "flags"] == "recent_staking" and watch.loc[3, "flags"] == "recently_updated" and watch.loc[4, "flags"] == "churn"
    assert watch.loc[5, "flags"] == "expiring; recent_staking; recently_updated"
    assert result.watch.iloc[0]["tenure_number"] == 5  # the most flags first
    assert result.summary["watch_counts"] == {"expiring": 2, "recent_staking": 2, "recently_updated": 2, "churn": 1}


def test_thresholds_come_from_the_project_settings(project):
    set_layer(project, [tenure(1, good_to="2026-11-05")])
    assert len(build_tenure_changes(project, today=TODAY).watch) == 1
    project.config.tenure.expiring_days = 10
    assert len(build_tenure_changes(project, today=TODAY).watch) == 0


def test_recent_staking_is_summarised_by_owner(project):
    rows = [tenure(1, owner="BIG CORP", area=500, issue="2026-08-01"), tenure(2, x=1, owner="BIG CORP", area=700, issue="2026-09-01"),
            tenure(3, x=2, owner="SMALL", area=50, issue="2026-07-01"), tenure(4, x=3, owner="OLD HAND", issue="2015-01-01")]  # fmt: skip
    set_layer(project, rows)
    top = build_tenure_changes(project, today=TODAY).summary["recent_staking_by_owner"]
    assert top[0] == {"owner": "BIG CORP", "tenures": 2, "hectares": 1200.0} and [o["owner"] for o in top] == ["BIG CORP", "SMALL"]


# --- the step, its outputs and the commands -----------------------------------------------------------------


def test_outputs_licence_and_layers(project):
    save_snapshot(project, T0, [tenure(1)])
    save_snapshot(project, T0 + timedelta(days=2), [tenure(1, owner="OTHER")])
    set_layer(project, [tenure(1, owner="OTHER", good_to="2026-11-01")])
    result = build_tenure_changes(project, today=TODAY)
    out = project.root / "analysis"
    for name in ("tenure_changes.csv", "tenure_watch.csv", "tenure_snapshots.csv", "tenure_summary.json"):
        assert (out / name).exists(), name
    assert pd.read_csv(out / "tenure_snapshots.csv")["tenures_in_aoi"].tolist() == [1, 1]
    layers = set(gpd.list_layers(out / "analysis.gpkg")["name"])
    assert {"tenure_changes", "tenure_watch"} <= layers
    record = next(r for r in project.provenance.read() if r.dataset == "analysis_tenure")
    assert "Open Government Licence" in record.license and "Access Only" not in record.license
    assert record.parameters["snapshots"] == 2 and record.parameters["events"] == len(result.events)


def test_no_tenure_layer_is_reported_not_an_error(project):
    result = build_tenure_changes(project, today=TODAY)
    assert result.summary == {"available": False, "reason": "no mto_tenure layer (run 'gossan harvest --source mto_tenure')"}
    assert result.outputs == ["analysis/tenure_summary.json"]  # recorded, so a report can say why
    out = runner.invoke(app, ["analyze", "demo", "--steps", "tenure"])
    assert out.exit_code == 0 and "no mto_tenure layer" in out.output


def test_command_output_says_when_only_one_snapshot_exists(project):
    save_snapshot(project, T0, [tenure(1, good_to="2026-11-05")])
    set_layer(project, [tenure(1, good_to="2026-11-05")])
    out = runner.invoke(app, ["analyze", "demo", "--steps", "tenure"])
    assert out.exit_code == 0, out.output
    assert "1 saved snapshot(s)" in out.output and "only one snapshot so far" in out.output
    assert "watchlist (" in out.output and "expiring" in out.output


def test_command_output_summarises_changes(project):
    save_snapshot(project, T0, [tenure(1), tenure(2, x=1)])
    save_snapshot(project, T0 + timedelta(days=3), [tenure(1)])
    set_layer(project, [tenure(1)])
    out = runner.invoke(app, ["analyze", "demo", "--steps", "tenure"])
    assert "changes between snapshots: 1 gone" in out.output
