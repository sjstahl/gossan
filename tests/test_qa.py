"""QA/QC: geometry and claim checks, layer and raster checks, the delivery package check, and the CLI."""

import hashlib
import json

import geopandas as gpd
import numpy as np
import pymupdf
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import Point, Polygon, box
from synthetic import CRS, X0, Y0, build_project
from typer.testing import CliRunner

from gossan.cli import app
from gossan.core.aoi import write_aoi
from gossan.core.project import Project
from gossan.qa import QA_JSON, QA_MD, QA_PDF, QaConfig, run_qa
from gossan.qa.checks import check_package, run_checks

runner = CliRunner()


@pytest.fixture
def project(tmp_path) -> Project:
    return build_project(tmp_path, layers=("minfile", "mto_tenure"), raster=False, n_occ=20)


def set_claims(project: Project, geoms: list, names: list[str] | None = None) -> None:
    gdf = gpd.GeoDataFrame({"claim": names or [f"C{i}" for i in range(len(geoms))]}, geometry=geoms, crs=CRS)
    write_aoi(gdf, project.aoi_path)


def found(result, check: str, severity: str | None = None, subject: str | None = None):
    return [f for f in result.findings if f.check == check and (severity is None or f.severity == severity) and (subject is None or f.subject == subject)]


def bowtie() -> Polygon:
    return Polygon([(X0, Y0), (X0 + 100, Y0 + 100), (X0 + 100, Y0), (X0, Y0 + 100)])


# --- the claim block --------------------------------------------------------------------------------------------------------


def test_a_clean_claim_block_has_no_findings(project):
    set_claims(project, [box(X0, Y0, X0 + 1000, Y0 + 1000), box(X0 + 1000, Y0, X0 + 2000, Y0 + 1000)])
    result = run_checks(project)
    assert not found(result, "overlap", subject="aoi (claim block)") and not found(result, "gap") and not found(result, "sliver")
    assert not found(result, "geometry", "error")
    assert any(f.check == "claims" and "2 claim polygon(s)" in f.message for f in result.findings)


def test_an_invalid_claim_is_an_error(project):
    set_claims(project, [bowtie()])
    result = run_checks(project)
    (error,) = found(result, "geometry", "error", "aoi")
    assert "invalid" in error.message and "Self-intersection" in error.message and not result.passed


def test_overlapping_claims_are_reported_with_the_area(project):
    set_claims(project, [box(X0, Y0, X0 + 1000, Y0 + 1000), box(X0 + 900, Y0, X0 + 2000, Y0 + 1000)], ["WEST", "EAST"])
    (overlap,) = found(run_checks(project), "overlap", "warn", "aoi (claim block)")
    assert "1 pair(s)" in overlap.message and "100,000 m2" in overlap.message and overlap.ids == ["WEST / EAST (100,000 m2)"]


def test_edge_noise_is_not_an_overlap(project):
    set_claims(project, [box(X0, Y0, X0 + 1000, Y0 + 1000), box(X0 + 999.99, Y0, X0 + 2000, Y0 + 1000)])
    assert not found(run_checks(project), "overlap", subject="aoi (claim block)")


def test_slivers_are_reported(project):
    set_claims(project, [box(X0, Y0, X0 + 1000, Y0 + 1000), box(X0 + 1000, Y0, X0 + 1002, Y0 + 1000), box(X0 + 2000, Y0, X0 + 2020, Y0 + 20)], ["BIG", "THIN", "TINY"])
    (sliver,) = found(run_checks(project), "sliver", "warn", "aoi (claim block)")
    assert sorted(sliver.ids) == ["THIN", "TINY"]


def test_gaps_between_claims_closed_and_thin(project):
    ring = Polygon([(X0, Y0), (X0 + 3000, Y0), (X0 + 3000, Y0 + 3000), (X0, Y0 + 3000)], [[(X0 + 1000, Y0 + 1000), (X0 + 1000, Y0 + 1050), (X0 + 1050, Y0 + 1050), (X0 + 1050, Y0 + 1000)]])
    set_claims(project, [ring, box(X0 + 3000, Y0, X0 + 4000, Y0 + 3000), box(X0 + 4003, Y0, X0 + 5000, Y0 + 3000)])
    (gap,) = found(run_checks(project), "gap", "warn", "aoi (claim block)")
    assert "2 gap(s)" in gap.message  # the 2,500 m2 hole in the first claim and the 3 m strip between the last two
    areas = sorted(float(i.split(" m2")[0].replace(",", "")) for i in gap.ids)
    assert areas == pytest.approx([2500, 9000], rel=0.05)


def test_a_large_gap_is_not_flagged(project):
    set_claims(project, [box(X0, Y0, X0 + 1000, Y0 + 1000), box(X0 + 1200, Y0, X0 + 2200, Y0 + 1000)])  # 200 m apart: ground nobody holds
    assert not found(run_checks(project), "gap")


def test_thresholds_can_be_changed(project):
    set_claims(project, [box(X0, Y0, X0 + 1000, Y0 + 1000), box(X0 + 999, Y0, X0 + 2000, Y0 + 1000)])  # 1,000 m2 overlap
    assert found(run_checks(project), "overlap", subject="aoi (claim block)")
    project.config.parameters["qa"] = {"overlap_tolerance_m2": 5000}
    assert not found(run_checks(project), "overlap", subject="aoi (claim block)")
    project.config.parameters["qa"] = {"nonsense": 1}
    with pytest.raises(ValueError):
        QaConfig.from_project(project)


def test_tenure_overlaps_are_a_warning_in_bc_and_a_note_in_the_us(project):
    gpd.GeoDataFrame({"TENURE_NUMBER_ID": [1, 2], "CLAIM_NAME": ["A", "B"]}, geometry=[box(X0, Y0, X0 + 1000, Y0 + 1000), box(X0 + 500, Y0, X0 + 1500, Y0 + 1000)], crs=CRS).to_file(project.data_gpkg, layer="mto_tenure", driver="GPKG")
    assert found(run_checks(project), "overlap", "warn", "mto_tenure")
    project.config.jurisdiction = "us"
    (note,) = found(run_checks(project), "overlap", "info", "mto_tenure")
    assert "expected for this source" in note.message


def test_expired_tenures_are_noted(project):
    gpd.GeoDataFrame({"TENURE_NUMBER_ID": [1, 2], "GOOD_TO_DATE": ["2001-01-01", "2099-01-01"]}, geometry=[box(X0, Y0, X0 + 1000, Y0 + 1000), box(X0 + 2000, Y0, X0 + 3000, Y0 + 1000)], crs=CRS).to_file(project.data_gpkg, layer="mto_tenure", driver="GPKG")
    (note,) = found(run_checks(project), "tenure", "info")
    assert "1 tenure(s)" in note.message and note.ids == ["1"]


# --- layers -----------------------------------------------------------------------------------------------------------------


def test_a_layer_outside_the_aoi_is_an_error(project):
    far = gpd.GeoDataFrame({"a": [1]}, geometry=[Point(X0 + 500_000, Y0 + 500_000)], crs=CRS)  # another zone's worth away: a wrong CRS looks like this
    far.to_file(project.data_gpkg, layer="legacy_points", driver="GPKG")
    (error,) = found(run_checks(project), "location", "error", "legacy_points")
    assert "wrong CRS" in error.message


@pytest.mark.filterwarnings("ignore:.crs. was not provided")
def test_a_layer_with_no_crs_is_an_error(project):
    gpd.GeoDataFrame({"a": [1]}, geometry=[Point(X0 + 10, Y0 + 10)], crs=None).to_file(project.data_gpkg, layer="no_crs", driver="GPKG")
    (error,) = found(run_checks(project), "crs", "error", "no_crs")
    assert "no CRS" in error.message


def test_a_layer_in_another_crs_is_noted_not_failed(project):
    gpd.GeoDataFrame({"a": [1]}, geometry=[Point(X0 + 10, Y0 + 10)], crs=CRS).to_crs("EPSG:4326").to_file(project.data_gpkg, layer="wgs84_points", driver="GPKG")
    (note,) = found(run_checks(project), "crs", "info", "wgs84_points")
    assert "EPSG:4326" in note.message and not found(run_checks(project), "location", subject="wgs84_points")


def test_empty_invalid_and_duplicate_geometries(project):
    good = box(X0, Y0, X0 + 500, Y0 + 500)
    gpd.GeoDataFrame({"a": [1, 2, 3, 4]}, geometry=[good, good, bowtie(), None], crs=CRS).to_file(project.data_gpkg, layer="messy", driver="GPKG")
    result = run_checks(project)
    messages = " | ".join(f.message for f in result.findings if f.subject == "messy")
    assert "1 feature(s) have no geometry" in messages and "invalid geometry" in messages and "repeat another feature's geometry" in messages
    assert not found(result, "geometry", "error", "messy")  # harvested data is as published: a warning, not a gate


def test_mostly_empty_attributes_are_noted(project):
    gpd.GeoDataFrame({"full": [1, 2, 3, 4], "sparse": [None, None, None, 1.0], "blank": ["", " ", "x", ""]}, geometry=[Point(X0 + i, Y0) for i in range(4)], crs=CRS).to_file(project.data_gpkg, layer="attrs", driver="GPKG")
    (note,) = found(run_checks(project), "attributes", "info", "attrs")
    assert "sparse 75%" in note.message and "blank 75%" in note.message and "full" not in note.message


def test_provenance_naming_a_layer_clears_the_warning(project):
    assert any(f.check == "provenance" and f.subject == "minfile" for f in run_checks(project).findings)
    project.provenance.log(dataset="minfile", source="test", identifier="x", url=None, license="OGL", operation="fetch", parameters={})
    assert not any(f.check == "provenance" and f.subject == "minfile" for f in run_checks(project).findings)


# --- rasters ----------------------------------------------------------------------------------------------------------------


def write_raster(project: Project, name: str, data: np.ndarray, bounds: tuple, crs: str | None = CRS, nodata: float = -9999.0) -> None:
    folder = project.root / "rasters"
    folder.mkdir(exist_ok=True)
    with rasterio.open(folder / name, "w", driver="GTiff", width=data.shape[1], height=data.shape[0], count=1, dtype="float32", crs=crs, transform=from_bounds(*bounds, data.shape[1], data.shape[0]), nodata=nodata) as ds:
        ds.write(data.astype("float32"), 1)


def test_raster_problems(project):
    aoi_bounds = tuple(project.read_aoi().total_bounds)
    write_raster(project, "good.tif", np.ones((10, 10)), aoi_bounds)
    write_raster(project, "empty.tif", np.full((10, 10), -9999.0), aoi_bounds)
    write_raster(project, "elsewhere.tif", np.ones((10, 10)), (X0 + 900_000, Y0, X0 + 910_000, Y0 + 10_000))
    write_raster(project, "partial.tif", np.ones((10, 10)), (aoi_bounds[0] - 6000, aoi_bounds[1], aoi_bounds[0] + 1000, aoi_bounds[3]))
    result = run_checks(project)
    assert not [f for f in result.findings if f.subject == "rasters/good.tif" and f.severity != "info"]
    assert found(result, "raster", "error", "rasters/empty.tif")
    assert found(result, "location", "error", "rasters/elsewhere.tif")
    (partial,) = found(result, "raster", "warn", "rasters/partial.tif")
    assert "of the AOI" in partial.message
    assert {r.name for r in result.layers if r.kind == "raster"} >= {"rasters/good.tif", "rasters/empty.tif"}


# --- the delivery package ---------------------------------------------------------------------------------------------------


@pytest.fixture
def packaged(project) -> Project:
    project.outputs_dir.mkdir(exist_ok=True)
    folder = project.outputs_dir / "deliverable"
    (folder / "data").mkdir(parents=True)
    (folder / "data" / "deliverable.gpkg").write_bytes(b"gpkg bytes")
    manifest = {
        "project": "demo", "built_at": "2026-01-01T00:00:00", "internal": False,
        "included": [{"key": "minfile", "title": "Occurrences", "kind": "vector", "group": "context", "licence": "OGL"}],
        "withheld": [],
        "files": [{"path": "data/deliverable.gpkg", "bytes": 10, "sha256": hashlib.sha256(b"gpkg bytes").hexdigest()}],
    }  # fmt: skip
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return project


def package_findings(project: Project):
    from gossan.qa.checks import QaResult

    result = QaResult()
    check_package(project, result)
    return result


def edit_manifest(project: Project, **changes) -> None:
    path = project.outputs_dir / "deliverable" / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(changes)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_an_intact_package_passes(packaged):
    result = package_findings(packaged)
    assert result.passed and found(result, "package", "info")


def test_a_changed_or_missing_file_is_an_error(packaged):
    (packaged.outputs_dir / "deliverable" / "data" / "deliverable.gpkg").write_bytes(b"edited")
    (error,) = found(package_findings(packaged), "package", "error")
    assert "changed since the package was built" in error.message
    (packaged.outputs_dir / "deliverable" / "data" / "deliverable.gpkg").unlink()
    (error,) = found(package_findings(packaged), "package", "error")
    assert "missing" in error.message


def test_an_item_with_no_licence_is_an_error(packaged):
    edit_manifest(packaged, included=[{"key": "x", "title": "Mystery layer", "licence": None}])
    (error,) = found(package_findings(packaged), "package", "error")
    assert "no licence on record" in error.message and error.ids == ["Mystery layer"]


def test_restricted_items_need_an_internal_package(packaged):
    edit_manifest(packaged, included=[{"key": "leads", "title": "Leads", "licence": "Derived product including Access Only data (aris_index): internal use only"}])
    (error,) = found(package_findings(packaged), "package", "error")
    assert "not marked internal" in error.message
    edit_manifest(packaged, internal=True)
    assert package_findings(packaged).passed


def test_aris_files_in_a_package_not_marked_internal_are_an_error(packaged):
    (packaged.root / "raw" / "aris_pdfs").mkdir(parents=True)
    (packaged.root / "raw" / "aris_pdfs" / "ARIS_00001.pdf").write_bytes(b"x")  # the fail-safe sees this even with a clean ledger
    edit_manifest(packaged, included=[{"key": "aris_index", "title": "ARIS reports", "licence": "OGL"}])
    assert any("ARIS-derived" in f.message for f in found(package_findings(packaged), "package", "error"))


def test_files_not_in_the_manifest_are_a_warning(packaged):
    (packaged.outputs_dir / "deliverable" / "extra.txt").write_text("hi")
    (warning,) = found(package_findings(packaged), "package", "warn")
    assert warning.ids == ["extra.txt"]


# --- the report, provenance and the commands --------------------------------------------------------------------------------


def test_the_report_is_written_logged_and_licensed(project):
    set_claims(project, [box(X0, Y0, X0 + 1000, Y0 + 1000), box(X0 + 900, Y0, X0 + 2000, Y0 + 1000)])
    result = run_qa(project)
    out = project.outputs_dir
    assert {p.name for p in out.iterdir()} >= {QA_PDF, QA_MD, QA_JSON}
    text = " ".join("\n".join(p.get_text() for p in pymupdf.open(out / QA_PDF)).split())
    assert "READY WITH CARE" in text and "Not for distribution to clients" in text and "overlap" in text and "Layer inventory" in text
    data = json.loads((out / QA_JSON).read_text())
    assert data["passed"] is True and data["verdict"].startswith("READY WITH CARE") and len(data["findings"]) == len(result.findings)
    records = [r for r in project.provenance.read() if r.operation == "qa_check"]
    assert {r.identifier for r in records} == {f"outputs/{QA_PDF}", f"outputs/{QA_MD}", f"outputs/{QA_JSON}"}
    assert all(r.parameters["errors"] == 0 and r.parameters["warnings"] >= 1 and len(r.parameters["sha256"]) == 64 for r in records)


def test_a_project_holding_access_only_data_gets_an_internal_licence(project):
    project.provenance.log(dataset="minfile", source="BC ARIS", identifier="x", url=None, license="Access Only (BC)", operation="fetch", parameters={})
    run_qa(project)
    rec = [r for r in project.provenance.read() if r.operation == "qa_check"][-1]
    assert "internal use only" in rec.license


def test_qa_command_exit_codes(project):
    set_claims(project, [box(X0, Y0, X0 + 1000, Y0 + 1000), box(X0 + 900, Y0, X0 + 2000, Y0 + 1000)])
    result = runner.invoke(app, ["qa", str(project.root)])
    assert result.exit_code == 0 and "WARN" in result.output
    assert runner.invoke(app, ["qa", str(project.root), "--strict"]).exit_code == 2
    set_claims(project, [bowtie()])
    result = runner.invoke(app, ["qa", str(project.root)])
    assert result.exit_code == 1 and "ERROR" in result.output and "invalid" in result.output
    before = len(list(project.outputs_dir.glob("qa_report.*")))
    assert runner.invoke(app, ["qa", str(project.root), "--no-report"]).exit_code == 1
    assert len(list(project.outputs_dir.glob("qa_report.*"))) == before


def test_deliver_stops_on_a_qa_error_unless_told_otherwise(tmp_path, monkeypatch):
    project = build_project(tmp_path, layers=("minfile", "mto_tenure"), raster=False, n_occ=20)
    set_claims(project, [bowtie()])
    called = []
    monkeypatch.setattr("gossan.cli.build_deliverable", lambda *a, **k: called.append(1) or (_ for _ in ()).throw(RuntimeError("built")))
    result = runner.invoke(app, ["deliver", str(project.root)])
    assert result.exit_code == 1 and "Not built" in result.output + (result.stderr or "") and called == []
    result = runner.invoke(app, ["deliver", str(project.root), "--skip-qa"])
    assert called == [1] and "QA error" in result.output + (result.stderr or "")
