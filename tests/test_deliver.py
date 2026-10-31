"""The client package: what goes in and what is withheld, the QGIS project, the offline map, the command."""

from __future__ import annotations

import base64
import glob
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from analysis_project import build_analysis_project, runner
from PIL import Image
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import MultiPolygon, Point, Polygon, box

from gossan.analyze import STEPS
from gossan.cli import app
from gossan.core.config import KnowledgeConfig, TargetConfig
from gossan.core.project import Project
from gossan.deliver import DeliverError, build_deliverable, webmap
from gossan.deliver.items import Item, candidates, plan, restricts
from gossan.report.analysis_render import build_analysis_report
from gossan.target.run import run_target

CRS = "EPSG:32610"


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> Path:
    """A project with a model run and the analysis steps done, built once and copied for each test."""
    base = tmp_path_factory.mktemp("template")
    here = Path.cwd()
    os.chdir(base)
    try:
        proj = build_analysis_project(base)
        cfg = TargetConfig(resolution_m=100.0, n_bags=4, n_trees=15, folds=3, seed=1, min_target_ha=0.0, knowledge=KnowledgeConfig(fault_range_m=800.0))
        run_target(proj, cfg)
        for name in ("occurrences", "targets"):
            STEPS[name](proj)
        build_analysis_report(proj, "pdf")
    finally:
        os.chdir(here)
    return base / "demo"


@pytest.fixture
def project(template, tmp_path, monkeypatch) -> Project:
    shutil.copytree(template, tmp_path / "demo")
    monkeypatch.chdir(tmp_path)
    return Project(Path("demo"))


def restrict(project: Project, identifier: str, licence: str = "Derived product including Access Only data (aris_index): internal use only") -> None:
    """Log a later record that gives an output a restrictive licence, as a run that used ARIS would."""
    project.provenance.log(dataset="restricted_test", source="gossan", identifier=identifier, url=None, license=licence, operation="analyze_test", parameters={})


def gpkg_layers(folder: Path) -> set[str]:
    import pyogrio

    return {str(n) for n in pyogrio.list_layers(folder / "data" / "deliverable.gpkg")[:, 0]}


# --- what goes in ----------------------------------------------------------------------------------------------------------


def test_the_package_holds_the_project_the_map_the_data_and_the_rasters(project):
    result = build_deliverable(project)
    folder = result.folder
    names = {p.name for p in folder.iterdir()}
    assert {"demo.qgs", "map.html", "README.txt", "manifest.json", "data", "rasters"} <= names
    assert not any(n.endswith("analysis_report.pdf") for n in names)  # this project holds an Access Only layer, so its report is internal
    assert gpkg_layers(folder) >= {"aoi", "targets", "knowledge_targets", "target_decisions", "minfile", "faults", "bedrock_geology"}
    assert {p.name for p in (folder / "rasters").iterdir()} >= {"prospectivity.tif", "knowledge.tif", "uncertainty.tif"}
    with rasterio.open(folder / "rasters" / "prospectivity.tif") as ds:
        assert ds.crs.to_epsg() == 32610
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["project"] == "demo" and not manifest["internal"] and [w["key"] for w in manifest["withheld"]] == ["report_pdf"]
    for entry in manifest["files"]:  # every file is listed with a hash that matches
        assert hashlib.sha256((folder / entry["path"]).read_bytes()).hexdigest() == entry["sha256"], entry["path"]
    readme = (folder / "README.txt").read_text()
    assert "Open Government Licence - British Columbia" in readme and "relative, not the probability of a deposit" in readme and "LEFT OUT" in readme and "Analysis report (PDF)" in readme
    record = next(r for r in project.provenance.read() if r.operation == "build_deliverable")
    assert "targets" in record.parameters["included"]


def test_open_only_rebuilds_decisions_and_report_from_open_layers_for_client_delivery(project):
    from gossan.harvest.bc_provincial.wfs import ACCESS_ONLY_BC
    from gossan.qa.checks import QaResult, check_package

    aris = gpd.GeoDataFrame(
        {"ARIS_NUMBER": [12345], "REPORT_YEAR": [2020], "MERGE_SPENT_2002": [50000.0]},
        geometry=[Point(693200, 5618500)],
        crs=CRS,
    )
    aris.to_file(project.data_gpkg, layer="aris_index", driver="GPKG", engine="pyogrio")
    project.provenance.log(dataset="aris_index", source="BC Data Catalogue", identifier="aris_index", url=None, license=ACCESS_ONLY_BC, operation="fetch")
    STEPS["targets"](project)
    build_analysis_report(project, "pdf")

    blocked = build_deliverable(project)
    assert {i.key for i in blocked.withheld} == {"target_decisions", "report_pdf"}

    clean = build_deliverable(project, open_only=True)
    assert not clean.withheld and not clean.internal
    assert "target_decisions" in gpkg_layers(clean.folder)
    assert (clean.folder / "demo_analysis_report.pdf").is_file()
    decisions_df = gpd.read_file(clean.folder / "data" / "deliverable.gpkg", layer="target_decisions", engine="pyogrio")
    assert "n_aris_reports" not in decisions_df.columns and "ground_status" in decisions_df.columns
    qa_res = QaResult()
    check_package(project, qa_res)
    assert qa_res.passed


def test_a_report_without_restricted_data_goes_with_the_package(project):
    project.provenance.log(
        dataset="report_analysis_pdf", source="gossan", identifier="outputs/analysis_report.pdf", url=None, license="Derived product from harvested layers (Open Government Licence - British Columbia)",
        operation="build_analysis_report", parameters={"access_only_marking": False},
    )  # fmt: skip
    result = build_deliverable(project)
    assert (result.folder / "demo_analysis_report.pdf").is_file() and "Analysis report (PDF)" in (result.folder / "README.txt").read_text()


def test_leads_are_never_packaged_even_when_the_project_has_them(project):
    gpd.GeoDataFrame({"lead_id": ["1"], "score": [50.0]}, geometry=[box(690000, 5617000, 690100, 5617100)], crs=CRS).to_file(project.data_gpkg, layer="leads", driver="GPKG", engine="pyogrio")
    result = build_deliverable(project, include_internal=True)  # not even then
    assert "leads" not in gpkg_layers(result.folder) and all(i.key != "leads" for i in candidates(project))


# --- what is withheld ----------------------------------------------------------------------------------------------------


def test_restricted_outputs_stay_out_and_the_package_says_what_and_why(project):
    restrict(project, "analysis/analysis.gpkg:target_decisions")
    restrict(project, "outputs/analysis_report.pdf")
    result = build_deliverable(project)
    assert "target_decisions" not in gpkg_layers(result.folder) and not any(p.name.endswith(".pdf") for p in result.folder.iterdir())
    withheld = {i.key: i.withheld for i in result.withheld}
    assert "Access Only" in withheld["target_decisions"] and "report_pdf" in withheld
    readme = (result.folder / "README.txt").read_text()
    assert "LEFT OUT" in readme and "Targets with ground status" in readme
    html = (result.folder / "map.html").read_text()
    assert "ground status" not in html  # the withheld table's columns never reach the map either
    qgs = (result.folder / "demo.qgs").read_text()
    assert "target_decisions" not in qgs


def test_an_item_with_no_provenance_record_is_withheld_not_guessed_at(project):
    ledger = project.root / "provenance.jsonl"
    kept = [line for line in ledger.read_text().splitlines() if '"dataset":"faults"' not in line.replace(" ", "")]
    ledger.write_text("\n".join(kept) + "\n")
    result = build_deliverable(project)
    assert "faults" not in gpkg_layers(result.folder)
    assert "no provenance record" in {i.key: i.withheld for i in result.withheld}["faults"]


def test_aris_names_are_withheld_whatever_the_ledger_says():
    assert "ARIS" in restricts("Open Government Licence - British Columbia", "aris_index")
    assert restricts("Open Government Licence - British Columbia", "faults") is None
    assert "no provenance record" in restricts(None, "faults") and "restricts" in restricts("Access Only", "x")


def test_include_internal_lifts_the_licence_test_and_marks_every_face(project):
    restrict(project, "analysis/analysis.gpkg:target_decisions")
    result = build_deliverable(project, include_internal=True)
    assert result.internal and "target_decisions" in gpkg_layers(result.folder)
    assert (result.folder / "README.txt").read_text().startswith(("demo", "INTERNAL")) and "INTERNAL" in (result.folder / "README.txt").read_text()
    assert json.loads((result.folder / "manifest.json").read_text())["internal"] is True
    data = _map_data(result.folder)
    assert data["internal"] is True and "Do not send it outside the company" in (result.folder / "map.html").read_text()
    clean = build_deliverable(project)  # and a normal rebuild over it is clean again
    assert not clean.internal and "INTERNAL" not in (clean.folder / "README.txt").read_text()


def test_nothing_to_deliver_without_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    proj = build_analysis_project(tmp_path)  # harvested data, no model run
    with pytest.raises(DeliverError, match="nothing to deliver"):
        build_deliverable(proj)


def test_a_folder_that_is_not_ours_is_never_overwritten_and_ours_is_rebuilt(project, tmp_path):
    foreign = tmp_path / "client_files"
    foreign.mkdir()
    (foreign / "contract.docx").write_text("do not delete")
    with pytest.raises(DeliverError, match="not a Gossan package"):
        build_deliverable(project, foreign)
    assert (foreign / "contract.docx").read_text() == "do not delete"
    first = build_deliverable(project, tmp_path / "pkg")
    (first.folder / "stray.txt").write_text("x")
    again = build_deliverable(project, tmp_path / "pkg", make_zip=True)  # rebuilt: the stray file is gone
    assert not (again.folder / "stray.txt").exists()
    with zipfile.ZipFile(again.zip_path) as z:
        assert "pkg/map.html" in z.namelist() and "pkg/manifest.json" in z.namelist()


# --- the QGIS project ---------------------------------------------------------------------------------------------------


def test_the_qgis_project_references_the_package_by_relative_path_and_styles_every_layer(project):
    folder = build_deliverable(project).folder
    root = ET.fromstring((folder / "demo.qgs").read_text(encoding="utf-8"))
    assert root.find("projectCrs/spatialrefsys/authid").text == "EPSG:32610"
    sources = [e.text for e in root.iter("datasource")]
    assert sources and all(s.lstrip("./").startswith(("data", "rasters")) for s in sources)
    assert not any(":" in s.split("|")[0] for s in sources)  # no absolute paths
    assert root.find("properties/SpatialRefSys/ProjectionsEnabled").text == "1"
    kinds = {m.get("type") for m in root.iter("maplayer")}
    assert kinds == {"vector", "raster"}
    assert len(list(root.iter("colorrampshader"))) == 3 and all(len(list(c.iter("item"))) == 5 for c in root.iter("colorrampshader"))
    assert len(list(root.iter("renderer-v2"))) == len([m for m in root.iter("maplayer") if m.get("type") == "vector"])
    tree = {n.get("name"): n.get("checked") for n in root.iter("layer-tree-layer")}
    assert tree["Prospectivity score (random forest)"] == "Qt::Checked" and tree["Bedrock geology"] == "Qt::Unchecked"


def find_python_qgis() -> Path | None:
    explicit = os.environ.get("GOSSAN_QGIS_PYTHON_BAT")
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    for pattern in (r"C:\Program Files\QGIS*\bin\python-qgis*.bat", r"C:\OSGeo4W*\bin\python-qgis*.bat"):
        found = sorted(glob.glob(pattern))
        if found:
            return Path(found[-1])
    return None


QGIS_CHECK = """
import json, sys, glob
from pathlib import Path
from qgis.core import QgsApplication, QgsProject
prefix = sorted(glob.glob(str(Path(sys.argv[2]) / "apps" / "qgis*")))[-1]
QgsApplication.setPrefixPath(prefix, True)
app = QgsApplication([], False); app.initQgis()
p = QgsProject.instance(); ok = p.read(sys.argv[1])
print("RESULT" + json.dumps({"read": ok, "crs": p.crs().authid(), "layers": {l.name(): [l.isValid(), l.featureCount() if hasattr(l, "featureCount") else None] for l in p.mapLayers().values()}}))
app.exitQgis()
"""


@pytest.mark.skipif(sys.platform != "win32" or find_python_qgis() is None, reason="needs a Windows QGIS install (python-qgis.bat)")
def test_a_real_qgis_opens_the_project_with_every_layer_valid(project, tmp_path):
    folder = build_deliverable(project).folder
    script = tmp_path / "check.py"
    script.write_text(QGIS_CHECK, encoding="utf-8")
    bat = find_python_qgis()
    done = subprocess.run([str(bat), str(script), str(folder / "demo.qgs"), str(bat.parent.parent)], capture_output=True, text=True, timeout=240, check=False)
    out = json.loads(next(line for line in done.stdout.splitlines() if line.startswith("RESULT"))[len("RESULT"):])
    assert out["read"] and out["crs"] == "EPSG:32610"
    assert len(out["layers"]) >= 9 and all(valid for valid, _ in out["layers"].values())
    assert out["layers"]["Known mineral occurrences (MINFILE)"][1] == 20


# --- the web map ---------------------------------------------------------------------------------------------------------


def _map_data(folder: Path) -> dict:
    html = (folder / "map.html").read_text(encoding="utf-8")
    blob = re.search(r"const DATA=(\{.*?\});\n", html, re.DOTALL).group(1)
    return json.loads(blob.replace("<\\/", "</"))


def test_the_map_is_one_self_contained_file_with_no_network_references(project):
    folder = build_deliverable(project).folder
    html = (folder / "map.html").read_text(encoding="utf-8")
    assert not re.search(r"(src|href)=[\"']https?:", html) and "fetch(" not in html and "XMLHttpRequest" not in html and "@import" not in html
    assert html.count("<script>") == 1 and "</script" not in _blob(html)
    data = _map_data(folder)
    keys = [layer["key"] for layer in data["layers"]]
    assert keys[:3] == ["uncertainty", "knowledge", "prospectivity"] or set(keys[:3]) == {"uncertainty", "knowledge", "prospectivity"}  # rasters sit under the vectors
    assert keys[-1] == "aoi" and "bedrock_geology" not in keys and "target_decisions" not in keys  # heavy or duplicated layers stay out
    by = {layer["key"]: layer for layer in data["layers"]}
    assert by["aoi"]["pick"] is False and by["targets"]["geom"] == "polygon" and by["minfile"]["geom"] == "point" and by["faults"]["geom"] == "line"
    assert data["utm"] == {"zone": 10, "north": True} and data["crs"] == CRS and data["extent"][0] < data["extent"][2]
    props = by["targets"]["features"][0]["p"]
    assert props["target id"].startswith("T") and "knowledge support" in props and "ground status" in props  # the decision table rides on the targets


def _blob(html: str) -> str:
    return re.search(r"const DATA=(\{.*?\});\n", html, re.DOTALL).group(1)


def test_the_embedded_data_cannot_close_the_script_tag():
    gdf = gpd.GeoDataFrame({"name": ["</script><b>x"]}, geometry=[Point(1, 2)], crs=CRS)
    layer = webmap.vector_layer(Item("minfile", "vector", "t", Path()), gdf)
    path = Path(os.getcwd()) / "t_map.html"
    try:
        webmap.write_map(path, {"title": "a & <b>", "layers": [layer], "extent": [0, 0, 1, 1], "utm": None, "crs": CRS, "client": "", "notice": "", "internal": False})
        html = path.read_text(encoding="utf-8")
    finally:
        path.unlink(missing_ok=True)
    assert "</script><b>x" not in html and "<title>a &amp; &lt;b&gt;</title>" in html


def test_vectors_keep_holes_parts_and_clean_attributes():
    holed = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)], [[(40, 40), (60, 40), (60, 60), (40, 60)]])
    gdf = gpd.GeoDataFrame({"a_b": [1.23456, np.nan], "txt": ["x" * 500, None]}, geometry=[MultiPolygon([holed, box(200, 0, 300, 100)]), box(0, 200, 10, 210)], crs=CRS)
    layer = webmap.vector_layer(Item("thing", "vector", "Thing", Path()), gdf)
    first, second = layer["features"]
    assert len(first["g"]) == 2 and len(first["g"][0]) == 2 and len(first["g"][1]) == 1  # two parts; the first has its hole
    assert first["p"]["a b"] == 1.235 and len(first["p"]["txt"]) == 160 and second["p"] == {}  # NaN and None are dropped
    assert webmap.vector_layer(Item("x", "vector", "x", Path()), gdf.iloc[0:0]) is None
    mixed = gpd.GeoDataFrame(geometry=[Point(0, 0), box(0, 0, 1, 1)], crs=CRS)
    assert webmap.vector_layer(Item("x", "vector", "x", Path()), mixed) is None  # a layer of mixed geometry kinds is not drawn


def test_a_raster_becomes_a_png_that_is_transparent_where_there_is_no_data(tmp_path):
    data = np.tile(np.linspace(0, 1, 20, dtype="float32"), (10, 1))
    data[:, :3] = -9999.0
    path = tmp_path / "r.tif"
    with rasterio.open(path, "w", driver="GTiff", width=20, height=10, count=1, dtype="float32", crs=CRS, transform=from_origin(0, 1000, 100, 100), nodata=-9999.0) as ds:
        ds.write(data, 1)
    layer = webmap.raster_layer(Item("prospectivity", "raster", "P", path))
    img = Image.open(io.BytesIO(base64.b64decode(layer["png"].split(",", 1)[1]))).convert("RGBA")
    assert img.size == (20, 10) and img.getpixel((0, 0))[3] == 0 and img.getpixel((10, 5))[3] == 255
    assert layer["bounds"] == [0.0, 0.0, 2000.0, 1000.0] and layer["labels"] == ["low", "high"] and layer["visible"]
    assert webmap.raster_layer(Item("magnetics", "raster", "M", path))["labels"][0] != "low"  # a physical quantity shows its numbers


NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="needs node")
def test_the_viewers_utm_conversion_matches_pyproj_everywhere_in_bc(tmp_path):
    points = [(690000, 5617000, 10, "EPSG:32610"), (520000, 5400000, 10, "EPSG:32610"), (350000, 6100000, 11, "EPSG:32611"), (700000, 5900000, 9, "EPSG:32609"), (500000, 5500000, 10, "EPSG:32610")]
    script = webmap.UTM_JS + "\nconst pts=" + json.dumps(points) + ";console.log(JSON.stringify(pts.map(p=>utmToLatLon(p[0],p[1],p[2],true))));"
    f = tmp_path / "utm.js"
    f.write_text(script)
    got = json.loads(subprocess.run([NODE, str(f)], capture_output=True, text=True, check=True).stdout)
    for (e, n, _, epsg), (lat, lon) in zip(points, got, strict=True):
        ref_lon, ref_lat = Transformer.from_crs(epsg, "EPSG:4326", always_xy=True).transform(e, n)
        assert abs(lat - ref_lat) < 1e-6 and abs(lon - ref_lon) < 1e-6, (e, n)  # about 10 cm


@pytest.mark.skipif(NODE is None, reason="needs node")
def test_the_viewer_script_is_valid_javascript(project, tmp_path):
    html = (build_deliverable(project).folder / "map.html").read_text(encoding="utf-8")
    script = re.search(r"<script>(.*)</script>", html, re.DOTALL).group(1)
    f = tmp_path / "viewer.js"
    f.write_text(script, encoding="utf-8")
    done = subprocess.run([NODE, "--check", str(f)], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr


def test_only_utm_projects_get_a_latitude_longitude_readout():
    assert webmap.utm_zone("EPSG:32610") == {"zone": 10, "north": True}
    assert webmap.utm_zone("EPSG:26911") == {"zone": 11, "north": True}
    assert webmap.utm_zone("EPSG:3005") is None  # BC Albers


# --- the command -----------------------------------------------------------------------------------------------------------


def test_the_command_lists_what_went_in_and_what_was_left_out(project):
    restrict(project, "analysis/analysis.gpkg:target_decisions")
    result = runner.invoke(app, ["deliver", "demo", "--zip"])
    assert result.exit_code == 0, result.output
    assert "Package:" in result.output and "in    Prospectivity targets (random forest)" in result.output
    assert "left out  Targets with ground status, access and history" in result.output and "zipped to" in result.output
    assert (project.outputs_dir / "deliverable.zip").exists()
    internal = runner.invoke(app, ["deliver", "demo", "--include-internal"])
    assert internal.exit_code == 0 and "INTERNAL" in internal.output


def test_the_command_fails_cleanly_without_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    build_analysis_project(tmp_path)
    result = runner.invoke(app, ["deliver", "demo"])
    assert result.exit_code == 1 and "nothing to deliver" in result.output
    assert not [i for i in plan(Project(Path("demo"))).included if i.group == "results"]


def test_a_package_passes_its_own_qa_check_until_a_file_changes(project):
    """``gossan qa`` verifies a package built by ``gossan deliver``: intact now, an error after an edit."""
    from gossan.qa.checks import QaResult, check_package

    def check() -> QaResult:
        result = QaResult()
        check_package(project, result)
        return result

    build_deliverable(project)
    result = check()
    assert result.passed and any("checked against the manifest" in f.message for f in result.findings)
    victim = next((project.outputs_dir / "deliverable").glob("*.qgs"))
    victim.write_text(victim.read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert not check().passed


@pytest.mark.parametrize(
    "module",
    ["gossan.pipeline", "gossan.report.analysis_render", "gossan.report.analysis_content", "gossan.deliver", "gossan.deliver.items", "gossan.analyze", "gossan.qa"],
)
def test_every_module_imports_on_its_own(module):
    """The tests import ``gossan.cli`` first, which hid an import cycle between the analysis report and the package
    builder; each of these must import in a fresh interpreter, whatever was imported before it."""
    import subprocess
    import sys

    result = subprocess.run([sys.executable, "-I", "-c", f"import {module}"], capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stderr[-600:]
    assert subprocess.run([sys.executable, "-I", "-c", "from gossan.deliver import build_deliverable, DeliverError, DeliverResult"], capture_output=True, check=False).returncode == 0
