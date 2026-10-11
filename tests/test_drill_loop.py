"""The whole loop: extracted CSVs -> reviewer approval -> drill tables -> analysis -> model and decisions.

The CSVs are shaped the way the scanned-table extraction leaves them (headings with units, ``<5`` for
below detection, blank depths on collar rows); approval goes through ``gossan review approve``, the
only way rows enter the tables.
"""

from __future__ import annotations

import getpass
from pathlib import Path

import pytest
from pyproj import Transformer
from typer.testing import CliRunner

from gossan.cli import app
from gossan.core.project import Project
from gossan.digitize import review

runner = CliRunner()


@pytest.fixture(autouse=True)
def reviewer(monkeypatch):
    monkeypatch.setattr(getpass, "getuser", lambda: "tester")


@pytest.fixture
def project(aoi_file) -> Project:
    assert runner.invoke(app, ["new", "demo", "--aoi", str(aoi_file)]).exit_code == 0
    return Project(Path("demo"))


def queue(project: Project, csv_text: str, table_type: str, report: int, page: int) -> str:
    rel = f"digitize/tables/scanned/{report:05d}_p{page}.csv"
    path = project.root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(csv_text, encoding="utf-8")
    project.provenance.log(
        dataset="aris_tables_scanned", source="Anthropic API (Claude vision)", identifier=f"ARIS {report} p{page}", url=None,
        license="x", operation="extract_scanned_table", parameters={"output": rel, "table_type": table_type, "page": page},
    )  # fmt: skip
    return review.add_item(project, report, "table", page=page, source=rel, image=f"digitize/pages/{report:05d}_p{page}.png", note="n").item_id


def test_approved_tables_become_classified_holes_on_the_map(project):
    # The AOI is at about 50.715 N, 120.30 W (UTM zone 10); a collar sheet printed in NAD83 UTM.
    east, north = Transformer.from_crs("EPSG:4326", "EPSG:26910", always_xy=True).transform(-120.30, 50.715)
    collars = (
        "hole_id,from_m,to_m,Easting (m),Northing (m),Length (m)\n"
        f"H-1,,,{east:.1f},{north:.1f},30\n"
        f"H-2,,,{east + 80:.1f},{north + 20:.1f},30\n"
        "H-3,,,12500,4300,45\n"  # a local mine grid: cannot be placed
    )
    assay_rows = "".join(f"{h},{i * 5},{(i + 1) * 5},{v}\n" for h, vals in {"H-1": [10, 900, 10, 10, 10, 10], "H-2": ["<5"] * 6, "H-3": [12] * 6}.items() for i, v in enumerate(vals))
    c = queue(project, collars, "collar", 77, 1)
    a = queue(project, "hole_id,from_m,to_m,Au (ppb)\n" + assay_rows, "assay", 77, 2)
    for item in (c, a):
        out = runner.invoke(app, ["review", "approve", str(project.root), item])
        assert out.exit_code == 0, out.output

    out = runner.invoke(app, ["analyze", "demo", "--steps", "drilling"])
    assert out.exit_code == 0, out.output
    assert "3 hole(s) from 1 report(s)" in out.output and "1 mineralized, 2 barren" in out.output
    assert "placed by collar coordinates 2, by report location 0, not placed 1" in out.output
    assert "mineralized: H-1 (ARIS 77) Au 0.9 g/t over 5-10 m" in out.output

    import geopandas as gpd

    holes = gpd.read_file(project.root / "analysis" / "analysis.gpkg", layer="drill_holes").set_index("key")
    assert set(holes.index) == {"77:H-1", "77:H-2"}  # the unplaced one is in the CSV, not on the map
    assert holes.loc["77:H-1", "class"] == "mineralized" and holes.loc["77:H-2", "class"] == "barren"
    assert holes.geometry.iloc[0].distance(holes.geometry.iloc[1]) == pytest.approx(82.5, abs=3)  # 80 m east, 20 m north
