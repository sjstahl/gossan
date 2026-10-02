"""A Nevada project, start to finish, on mocked USGS and BLM services: new, harvest, target, report."""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pymupdf
import pytest
import rasterio
from typer.testing import CliRunner

from gossan.cli import app
from gossan.core.licensing import uses_aris
from gossan.core.project import Project
from gossan.report import pdf as report_pdf

runner = CliRunner()
NV = Path(__file__).parent / "fixtures" / "aoi_nv.geojson"
W, S, E, N = -117.30, 39.10, -117.10, 39.25


def mrds_gml(n: int, seed: int = 3) -> bytes:
    """MRDS-style GML with ``n`` occurrences, mostly in the eastern half of the AOI."""
    rng = np.random.default_rng(seed)
    lons = np.concatenate([rng.uniform(W + 0.6 * (E - W), E, int(n * 0.85)), rng.uniform(W, E, n - int(n * 0.85))])
    lats = rng.uniform(S, N, n)
    members = "".join(
        f"""<gml:featureMember><ms:mrds-high fid="mrds-high.{i}"><ms:geometry><gml:Point srsName="EPSG:4326">
        <gml:coordinates>{lon:.6f},{lat:.6f}</gml:coordinates></gml:Point></ms:geometry>
        <ms:dep_id>{10000000 + i}</ms:dep_id><ms:site_name>Prospect {i}</ms:site_name><ms:dev_stat>Occurrence</ms:dev_stat>
        <ms:fips_code>f32023</ms:fips_code><ms:huc_code>h1</ms:huc_code><ms:quad_code>q1</ms:quad_code>
        <ms:url>https://mrdata.usgs.gov/mrds/show-mrds.php?dep_id={10000000 + i}</ms:url>
        <ms:code_list> {"AU AG" if i % 2 else "CU MO"}</ms:code_list></ms:mrds-high></gml:featureMember>"""
        for i, (lon, lat) in enumerate(zip(lons, lats, strict=True))
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?><wfs:FeatureCollection xmlns:ms="http://mapserver.gis.umn.edu/mapserver" '
        'xmlns:gml="http://www.opengis.net/gml" xmlns:wfs="http://www.opengis.net/wfs">' + members + "</wfs:FeatureCollection>"
    ).encode()


def blm_features(n: int = 12, seed: int = 5) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        x, y = rng.uniform(W, E - 0.01), rng.uniform(S, N - 0.01)
        out.append(
            {
                "type": "Feature", "id": i,
                "geometry": {"type": "Polygon", "coordinates": [[[x, y], [x + 0.006, y], [x + 0.006, y + 0.004], [x, y + 0.004], [x, y]]]},
                "properties": {
                    "OBJECTID": i, "ID": str(7000 + i), "CSE_NAME": f"CLAIM {i}", "BLM_PROD": "Lode Claim", "CSE_NR": f"NV10500{i:04d}",
                    "CSE_DISP": "Active", "RCRD_ACRS": 20.66, "QLTY": "0: ok", "CSE_META": "NV 21", "Created": 1612915200000,
                },
            }
        )
    return out


@pytest.fixture
def nevada(us, monkeypatch) -> Project:
    us.mrds_bytes = mrds_gml(40)
    us.blm_features = blm_features()
    result = runner.invoke(app, ["new", "nv", "--aoi", str(NV)])
    assert result.exit_code == 0, result.output
    assert "jurisdiction us" in result.output and "EPSG:32611" in result.output
    assert runner.invoke(app, ["harvest", "nv"]).exit_code == 0
    return Project(Path("nv"))


def all_licenses(project: Project) -> str:
    return " | ".join(r.license for r in project.provenance.read())


def test_the_us_layers_feed_the_target_module_unchanged(nevada):
    result = runner.invoke(app, ["target", "nv", "--resolution", "500", "--n-bags", "3", "--folds", "3"])
    assert result.exit_code == 0, result.output
    assert "Known occurrences used as positives: " in result.output and "dist_minfile" in result.output
    assert "tenure_density" in result.output  # BLM claims arrive as the tenure layer
    assert "aris_density" not in result.output.split("skipped")[0]  # there is no ARIS index to use
    out = nevada.root / "target"
    with rasterio.open(out / "prospectivity.tif") as ds:
        assert ds.crs.to_epsg() == 32611 and ds.count == 1
        prob = ds.read(1, masked=True)
    assert 0 <= prob.min() and prob.max() <= 1
    grid = json.loads((out / "grid.json").read_text())
    assert grid["crs"] == "EPSG:32611" and grid["n_known_occurrences_in_aoi"] >= 10
    assert "aris_density" in grid["features_skipped"] and "tenure_density" in grid["features_used"]
    assert len(gpd.read_file(out / "targets.gpkg", layer="targets_top")) >= 1


def test_target_outputs_for_a_us_project_are_not_labelled_access_only(nevada):
    assert runner.invoke(app, ["target", "nv", "--resolution", "500", "--n-bags", "3", "--no-cv"]).exit_code == 0
    records = [r for r in nevada.provenance.read() if r.operation == "target_run"]
    assert records and all("Access Only" not in r.license and "Public domain" in r.license for r in records)
    assert "Access Only" not in all_licenses(nevada) and not uses_aris(nevada.provenance.read())


def test_the_report_for_a_us_project_has_no_aris_marking_and_uses_us_names(nevada):
    assert runner.invoke(app, ["target", "nv", "--resolution", "500", "--n-bags", "3", "--no-cv"]).exit_code == 0
    result = runner.invoke(app, ["report", "nv"])
    assert result.exit_code == 0, result.output
    doc = pymupdf.open(nevada.outputs_dir / "report.pdf")
    text = "\n".join(page.get_text() for page in doc).replace("\n", " ")
    assert doc.page_count >= 4
    assert all(report_pdf.PLAIN_FOOTER in page.get_text() for page in doc)
    for forbidden in ("INTERNAL DRAFT", "Access Only", "ARIS", "MINFILE", "Mineral Titles Online"):
        assert forbidden not in text, forbidden
    assert "40 MRDS mineral occurrence(s)" in text and "Most common commodities: Gold" in text or "Copper" in text
    assert "12 mineral/placer tenure(s)" in text and "Claims are BLM MLRS cases that are not closed" in text
    assert "Status not recorded in the source data" in text and "no issue dates" in text
    assert "Prospectivity targeting" in text and "known MRDS occurrences" in text
    assert "Lead scoring is not available" in text  # until 'gossan leads run'
    rec = [r for r in nevada.provenance.read() if r.operation == "build_report_pdf"][-1]
    assert rec.parameters["aris_marking"] is False and "Public domain" in rec.license


def test_us_leads_need_the_target_polygons_first(nevada):
    result = runner.invoke(app, ["leads", "run", "nv"])
    assert result.exit_code == 1 and "gossan target" in result.output


def test_us_leads_score_on_spatial_evidence_only(nevada):
    assert runner.invoke(app, ["target", "nv", "--resolution", "500", "--n-bags", "3", "--no-cv"]).exit_code == 0
    result = runner.invoke(app, ["leads", "run", "nv"])
    assert result.exit_code == 0, result.output
    assert "12 lead(s) scored" in result.output and "in targets" in result.output
    csv = pd.read_csv(nevada.outputs_dir / "leads_ranked.csv")
    assert len(csv) == 12 and csv["rank"].tolist() == list(range(1, 13))
    for absent in ("issue_date", "good_to_date", "recency_points", "aris_count"):
        assert absent not in csv.columns
    assert np.allclose(csv["score"], csv["overlap_points"] + csv["proximity_points"], atol=0.02)
    assert csv["score"].is_monotonic_decreasing and csv["score"].between(0, 100).all()
    assert csv["overlap_points"].between(0, 60).all() and csv["proximity_points"].between(0, 40).all()
    layer = gpd.read_file(nevada.data_gpkg, layer="leads")
    assert len(layer) == 12 and layer.crs.to_epsg() == 32611
    rec = [r for r in nevada.provenance.read() if r.operation == "write_leads_csv"][-1]
    assert "Access Only" not in rec.license and rec.parameters["scoring"] == "us_spatial"


def test_us_lead_scores_follow_overlap_and_proximity():
    from shapely.geometry import Point, box

    from gossan.leads.scoring_us import UsLeadsConfig, score_us_leads

    def claim(i, x0):
        return {"TENURE_NUMBER_ID": i, "CLAIM_NAME": f"C{i}", "TENURE_TYPE_DESCRIPTION": "Lode", "OWNER_NAME": None,
                "AREA_IN_HECTARES": 100.0, "geometry": box(x0, 0, x0 + 1000, 1000)}  # fmt: skip

    claims = gpd.GeoDataFrame([claim(1, 0), claim(2, 5000), claim(3, 20000)], crs="EPSG:32611")
    targets = gpd.GeoDataFrame(geometry=[box(0, 0, 500, 1000)], crs="EPSG:32611")  # half of claim 1
    occ = gpd.GeoDataFrame(geometry=[Point(500, 500), Point(6500, 500)], crs="EPSG:32611")
    out = score_us_leads(claims, targets, occ, UsLeadsConfig()).set_index("lead_id")
    assert out.loc["1", "target_overlap_pct"] == 50 and out.loc["1", "overlap_points"] == 30
    assert out.loc["1", "occurrences_inside"] == 1 and out.loc["1", "proximity_points"] == 40
    assert out.loc["2", "overlap_points"] == 0 and out.loc["2", "proximity_points"] == 30  # 500 m of 2000 m away
    assert out.loc["3", "score"] == 0 and out.index.tolist()[0] == "1"
    assert claims is not None and score_us_leads(claims, targets.iloc[0:0], occ.iloc[0:0], UsLeadsConfig())["score"].eq(0).all()


def test_us_lead_pack_and_report_use_spatial_wording(nevada):
    assert runner.invoke(app, ["target", "nv", "--resolution", "500", "--n-bags", "3", "--no-cv"]).exit_code == 0
    assert runner.invoke(app, ["leads", "run", "nv"]).exit_code == 0
    top = str(pd.read_csv(nevada.outputs_dir / "leads_ranked.csv").loc[0, "lead_id"])
    result = runner.invoke(app, ["leads", "pack", "nv", top])
    assert result.exit_code == 0, result.output
    from docx import Document

    text = "\n".join(p.text for p in Document(str(nevada.outputs_dir / f"lead_{top}_pack.docx")).paragraphs)
    assert "quarter-section" in text and "MRDS is a historical inventory" in text and "Access Only" not in text
    assert runner.invoke(app, ["report", "nv", "--lead", top]).exit_code == 0
    doc = pymupdf.open(nevada.outputs_dir / "report.pdf")
    flat = "\n".join(p.get_text() for p in doc).replace("\n", " ")
    assert "scored out of 100 on spatial evidence only" in flat and "Featured lead" in flat
    assert "Lead scoring is not available" not in flat and "ARIS" not in flat


def test_a_bc_project_is_unaffected(aoi_file, wfs):
    """The BC path through the same code: BC defaults, BC layers, and the ARIS marking."""
    assert runner.invoke(app, ["new", "bc", "--aoi", str(aoi_file)]).exit_code == 0
    assert runner.invoke(app, ["harvest", "bc"]).exit_code == 0
    project = Project(Path("bc"))
    assert project.config.jurisdiction == "bc" and project.config.crs == "EPSG:32610"
    assert uses_aris(project.provenance.read())  # aris_index was harvested and is listed
    assert runner.invoke(app, ["report", "bc"]).exit_code == 0
    doc = pymupdf.open(project.outputs_dir / "report.pdf")
    assert all(report_pdf.FOOTER in page.get_text() for page in doc)
    assert "INTERNAL DRAFT - NOT FOR DISTRIBUTION" in doc[0].get_text().replace("\n", " ")
