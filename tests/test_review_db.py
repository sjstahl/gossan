import getpass
import sqlite3
from pathlib import Path

import geopandas as gpd
import pyogrio
import pytest
from shapely.geometry import Point
from typer.testing import CliRunner

from gossan.cli import app
from gossan.core.project import Project
from gossan.digitize import db, review
from gossan.digitize.db import (
    DrillholeDbError,
    load_approved_table,
    parse_value,
    split_heading,
)
from gossan.digitize.review import ReviewError, Status

runner = CliRunner()

GOOD_CSV = (
    "hole_id,from_m,to_m,Au (ppb),Lithology,Cu (ppm)\n"
    "R-1,0,1.5,12,andesite,\n"
    "R-1,1.5,3.0,<5,tuff,40\n"
    "R-2,0,2,nil,andesite,1200\n"
)


@pytest.fixture(autouse=True)
def reviewer(monkeypatch):
    monkeypatch.setattr(getpass, "getuser", lambda: "tester")


@pytest.fixture
def project(aoi_file) -> Project:
    assert runner.invoke(app, ["new", "demo", "--aoi", str(aoi_file)]).exit_code == 0
    return Project(Path("demo"))


def queue_table(project, csv_text=GOOD_CSV, table_type="assay", report=77, page=2) -> review.ReviewItem:
    """A table the way the scanned-table step leaves it: CSV, provenance (with its type), PENDING item."""
    rel = f"digitize/tables/scanned/{report:05d}_p{page}.csv"
    path = project.root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(csv_text, encoding="utf-8")
    if table_type is not None:
        project.provenance.log(
            dataset="aris_tables_scanned", source="Anthropic API (Claude vision)",
            identifier=f"ARIS {report} p{page}", url=None, license="x",
            operation="extract_scanned_table",
            parameters={"output": rel, "table_type": table_type, "page": page},
        )  # fmt: skip
    return review.add_item(
        project, report, "table", page=page, source=rel, image=f"digitize/pages/{report:05d}_p{page}.png", note="n"
    )


def rows(project: Project, table: str) -> list[dict]:
    con = sqlite3.connect(project.data_gpkg)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(f"SELECT * FROM {table} ORDER BY fid")]
    finally:
        con.close()


def approve(project, item_id, *extra):
    return runner.invoke(app, ["review", "approve", str(project.root), item_id, *extra])


# --- value parsing ------------------------------------------------------------------------


def test_split_heading():
    assert split_heading("Au (ppb)") == ("Au", "ppb")
    assert split_heading("Lithology") == ("Lithology", "")
    assert split_heading("  Cu  (ppm) ") == ("Cu", "ppm")
    assert split_heading("Recvd Wt. (kg)") == ("Recvd Wt.", "kg")
    assert split_heading("(ppm)")[0] == "(ppm)"  # nothing but a unit: keep the text, not an empty name


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("12", (12.0, "")), ("0.011", (0.011, "")), ("<5", (5.0, "<")), ("< 0.2", (0.2, "<")),
        (">10000", (10000.0, ">")), ("1,200", (1200.0, "")), ("-3.5", (-3.5, "")),
        ("nil", (None, "")), ("andesite", (None, "")), ("12 ppm", (None, "")), ("<LOD", (None, "")),
    ],
)  # fmt: skip
def test_parse_value(raw, expected):
    assert parse_value(raw) == expected


# --- approve / reject ---------------------------------------------------------------------


def test_approve_and_reject_update_the_queue_and_provenance(project):
    a = review.add_item(project, 8038, "map", page=4, source="rasters/a.tif", image="digitize/maps/a.png")
    b = review.add_item(project, 8038, "text", note="scanned pages")
    result = approve(project, a.item_id, "--note", "placement checked against the road")
    assert result.exit_code == 0, result.output
    assert "Approved R0001 (map, ARIS 8038 p.4)" in result.output

    item = review.get_item(project, "R0001")
    assert item.status == "APPROVED" and item.reviewed_by == "tester"
    assert item.decision_note == "placement checked against the road" and item.reviewed_at.endswith("Z")
    rec = next(r for r in project.provenance.read() if r.operation == "review_approve")
    assert rec.identifier == "R0001" and rec.parameters["previous_status"] == "PENDING"
    assert rec.parameters["reviewed_by"] == "tester" and rec.source == "manual review"

    result = runner.invoke(app, ["review", "reject", str(project.root), b.item_id, "--by", "Sam", "--note", "wrong page"])
    assert result.exit_code == 0 and "Rejected R0002" in result.output
    item = review.get_item(project, "R0002")
    assert (item.status, item.reviewed_by, item.decision_note) == ("REJECTED", "Sam", "wrong page")
    assert any(r.operation == "review_reject" for r in project.provenance.read())

    assert "No items pending review." in runner.invoke(app, ["review", "list", str(project.root)]).output
    assert not project.data_gpkg.exists()  # maps and text do not touch the database


def test_decisions_are_final_and_ids_are_checked(project):
    item = review.add_item(project, 1, "map")
    assert approve(project, item.item_id).exit_code == 0
    again = approve(project, item.item_id)
    assert again.exit_code == 1 and "already APPROVED" in again.output
    flip = runner.invoke(app, ["review", "reject", str(project.root), item.item_id])
    assert flip.exit_code == 1 and "decisions are final" in flip.output
    assert review.get_item(project, item.item_id).status == "APPROVED"
    assert approve(project, "R0099").exit_code == 1
    with pytest.raises(ReviewError):
        review.set_status(project, item.item_id, Status.PENDING, by="x")


def test_review_still_works_without_a_subcommand(project):
    review.add_item(project, 8038, "map", page=4, source="s", image="i")
    old_style = runner.invoke(app, ["review", str(project.root)])  # the Part 1 form
    assert old_style.exit_code == 0 and "1 item(s) pending review (1 map)" in old_style.output
    listed = runner.invoke(app, ["review", "list", str(project.root)])
    assert listed.output == old_style.output


def test_old_queue_without_decision_columns_is_upgraded(project):
    path = review.queue_path(project)
    path.parent.mkdir(parents=True)
    path.write_text(
        "item_id,created_at,report,type,page,status,source,note,image\n"
        "R0001,2026-10-01T00:00:00Z,8038,map,4,PENDING,s,n,i\n",
        encoding="utf-8",
    )
    assert approve(project, "R0001").exit_code == 0
    assert path.read_text().splitlines()[0] == ",".join(review.COLUMNS)
    assert review.get_item(project, "R0001").reviewed_by == "tester"


# --- drillhole tables ---------------------------------------------------------------------


def test_approving_a_table_creates_the_tables_and_appends_long_records(project):
    item = queue_table(project)
    result = approve(project, item.item_id)
    assert result.exit_code == 0, result.output
    assert "3 row(s) -> 8 record(s) in the assay table" in result.output

    layers = {str(n) for n in pyogrio.list_layers(project.data_gpkg)[:, 0]}
    assert {"collar", "survey", "assay", "lithology"} <= layers
    con = sqlite3.connect(project.data_gpkg)  # GeoPackage is SQLite; QGIS lists these as attribute tables
    kinds = dict(con.execute("SELECT table_name, data_type FROM gpkg_contents").fetchall())
    con.close()
    assert all(kinds[t] == "attributes" for t in db.TABLES)
    assert rows(project, "collar") == rows(project, "survey") == rows(project, "lithology") == []

    recs = rows(project, "assay")
    assert len(recs) == 8  # 3 rows x attributes, empty cells skipped
    first = recs[0]
    assert (first["hole_id"], first["from_m"], first["to_m"]) == ("R-1", 0.0, 1.5)
    assert (first["attribute"], first["unit"], first["value"], first["flag"], first["raw_value"]) == (
        "Au", "ppb", 12.0, "", "12",
    )  # fmt: skip
    by = {(r["hole_id"], r["from_m"], r["attribute"]): r for r in recs}
    censored = by[("R-1", 1.5, "Au")]
    assert (censored["value"], censored["flag"], censored["raw_value"]) == (5.0, "<", "<5")  # never plain 5
    text = by[("R-1", 0.0, "Lithology")]
    assert text["value"] is None and text["raw_value"] == "andesite" and text["unit"] == ""
    nil = by[("R-2", 0.0, "Au")]
    assert nil["value"] is None and nil["raw_value"] == "nil"
    assert by[("R-2", 0.0, "Cu")]["value"] == 1200.0 and by[("R-2", 0.0, "Cu")]["unit"] == "ppm"
    assert {r["aris_number"] for r in recs} == {77} and {r["page"] for r in recs} == {2}
    assert {r["review_item"] for r in recs} == {"R0001"} and {r["approved_by"] for r in recs} == {"tester"}
    assert first["source_file"] == "digitize/tables/scanned/00077_p2.csv"

    ops = [r.operation for r in project.provenance.read()]
    assert ops.count("init_drillhole_tables") == 1
    load = next(r for r in project.provenance.read() if r.operation == "load_drillhole_table")
    assert load.parameters["table"] == "assay" and load.parameters["records"] == 8
    assert load.parameters["censored_values"] == 1 and len(load.parameters["csv_sha256"]) == 64
    approve_rec = next(r for r in project.provenance.read() if r.operation == "review_approve")
    assert approve_rec.parameters["loaded_into"] == "assay" and approve_rec.parameters["records"] == 8


def test_second_table_appends_and_other_layers_survive(project):
    seed = gpd.GeoDataFrame({"n": [1]}, geometry=[Point(0, 0)], crs="EPSG:32610")
    seed.to_file(project.data_gpkg, layer="minfile", driver="GPKG", engine="pyogrio")
    one = queue_table(project, page=2)
    two = queue_table(project, "hole_id,from_m,to_m,Zn (%)\nR-9,5,6,0.4\n", page=3)
    assert approve(project, one.item_id).exit_code == 0
    assert approve(project, two.item_id).exit_code == 0
    assert len(rows(project, "assay")) == 9
    assert {r["review_item"] for r in rows(project, "assay")} == {one.item_id, two.item_id}
    assert "minfile" in {str(n) for n in pyogrio.list_layers(project.data_gpkg)[:, 0]}
    assert [r.operation for r in project.provenance.read()].count("init_drillhole_tables") == 1


def test_the_csv_is_read_at_approval_so_reviewer_corrections_count(project):
    item = queue_table(project)
    path = project.root / item.source
    path.write_text(path.read_text().replace("R-1,0,1.5,12,", "R-1,0,1.5,21,"))  # reviewer fixes a digit
    assert approve(project, item.item_id).exit_code == 0
    au = [r for r in rows(project, "assay") if r["attribute"] == "Au" and r["from_m"] == 0.0 and r["hole_id"] == "R-1"]
    assert au[0]["value"] == 21.0


def test_missing_required_keys_blocks_approval(project):
    item = queue_table(project, "hole,from,to,Au (ppb)\nR-1,0,1,5\n")
    result = approve(project, item.item_id)
    assert result.exit_code == 1
    assert "missing required column(s): hole_id, from_m, to_m" in result.output
    assert review.get_item(project, item.item_id).status == "PENDING"  # stays in the queue
    assert not project.data_gpkg.exists()  # no tables created, nothing written
    assert not [r for r in project.provenance.read() if r.operation == "review_approve"]


def test_each_validation_problem_is_reported_and_nothing_is_loaded(project):
    bad = (
        "hole_id,from_m,to_m,Au (ppb)\n"
        ",0,1,5\n"  # no hole
        "R-1,2,1,5\n"  # to <= from
        "R-1,x,3,5\n"  # not a number
        "R-1,-1,3,5\n"  # negative
        "R-1,4,,5\n"  # only one depth
        "R-1,5,6,?\n"  # still illegible
    )
    item = queue_table(project, bad)
    result = approve(project, item.item_id)
    assert result.exit_code == 1
    for fragment in ("hole_id is empty", "not greater than from_m", "'x' is not a number", "is negative",
                     "both be given or both empty", "illegible"):  # fmt: skip
        assert fragment in result.output
    assert review.get_item(project, item.item_id).status == "PENDING"
    assert not project.data_gpkg.exists()


def test_surface_samples_without_depths_are_accepted_and_counted(project):
    csv_text = "hole_id,from_m,to_m,Au (ppm),Cu (ppm)\nH608016,,,0.011,17\nH608017,,,0.008,107\n"
    item = queue_table(project, csv_text, page=44)
    result = approve(project, item.item_id)
    assert result.exit_code == 0, result.output
    assert "2 row(s) have no depth interval: surface samples" in result.output
    recs = rows(project, "assay")
    assert len(recs) == 4 and all(r["from_m"] is None and r["to_m"] is None for r in recs)


@pytest.mark.parametrize(("table_type", "target"), [("lithology", "lithology"), ("collar", "collar"), ("survey", "survey")])
def test_table_type_chooses_the_table(project, table_type, target):
    item = queue_table(project, "hole_id,from_m,to_m,Azimuth\nR-1,,,45\n", table_type=table_type)
    assert approve(project, item.item_id).exit_code == 0
    assert len(rows(project, target)) == 1
    assert all(not rows(project, t) for t in db.TABLES if t != target)


@pytest.mark.parametrize("table_type", ["other", None])
def test_untyped_or_other_tables_are_refused(project, table_type):
    item = queue_table(project, table_type=table_type)
    result = approve(project, item.item_id)
    assert result.exit_code == 1 and "can be loaded into the drillhole tables" in result.output
    assert review.get_item(project, item.item_id).status == "PENDING"


def test_an_item_cannot_be_loaded_twice(project):
    item = queue_table(project)
    load_approved_table(project, item, "tester")
    with pytest.raises(DrillholeDbError, match="already loaded into assay"):
        load_approved_table(project, item, "tester")
    assert len(rows(project, "assay")) == 8


def test_missing_csv_blocks_approval(project):
    item = queue_table(project)
    (project.root / item.source).unlink()
    result = approve(project, item.item_id)
    assert result.exit_code == 1 and "does not exist" in result.output


def test_init_tables_is_idempotent_and_keeps_data(project):
    assert db.init_tables(project) == list(db.TABLES)
    item = queue_table(project)
    load_approved_table(project, item, "tester")
    assert db.init_tables(project) == []
    assert len(rows(project, "assay")) == 8
