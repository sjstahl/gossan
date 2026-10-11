"""ASTER band-ratio alteration indices: radiance, the five ratios, masking, combining scenes, the
rasters and summary, the target features, the command, the pipeline and the report section."""

from __future__ import annotations

import os
import time

import geopandas as gpd
import numpy as np
import pymupdf
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin
from synthetic import CRS, X0, Y0, H, W, build_project, runner, small_cfg

from gossan import pipeline
from gossan.cli import app
from gossan.core.config import ProjectConfig
from gossan.core.project import Project
from gossan.spectral import SpectralError, build_aster
from gossan.spectral.aster import (
    INDICES,
    UCC,
    find_scenes,
    indices_from_radiance,
    radiance,
    scene_bands,
)
from gossan.target.features import build_stack
from gossan.target.grid import build_grid
from gossan.target.run import TargetError

#: At-sensor radiance of bare, unaltered ground in each band, and what changes in the altered half.
BASE = {1: 40.0, 2: 50.0, 3: 60.0, 4: 70.0, 5: 40.0, 6: 38.0, 7: 36.0, 8: 34.0, 9: 30.0}
ALTERED_B6 = 30.0  # deeper absorption in band 6: the Al-OH index rises from 2.0 to 2.53
EAST = X0 + 2000.0  # the altered ground is east of this
EXPECTED_WEST = {"ferric": 1.25, "gossan": 1.4, "ferrous": 40 / 60 + 40 / 50, "aloh": 2.0, "mgoh": 66 / 34}
EXPECTED_EAST = {**EXPECTED_WEST, "aloh": 76 / 30}
MARGIN = 150.0
#: VNIR digital numbers are small integers (a step is about 3.5% of band 2 radiance), so ratios are only this exact.


def band_array(band: int, res: float, *, veg, snow) -> np.ndarray:
    """Radiance of one band over the AOI plus a margin at ``res`` metres, as a scene would hold it."""
    ncol, nrow = int((W + 2 * MARGIN) / res), int((H + 2 * MARGIN) / res)
    xs = X0 - MARGIN + (np.arange(ncol) + 0.5) * res
    ys = Y0 + H + MARGIN - (np.arange(nrow) + 0.5) * res
    xx, yy = np.meshgrid(xs, ys)
    rad = np.full(xx.shape, BASE[band])
    if band == 6:
        rad = np.where(xx > EAST, ALTERED_B6, rad)
    if veg is not None:  # vegetated: red absorbs, near infrared reflects
        inside = (xx > veg[0]) & (xx < veg[1]) & (yy < Y0 + veg[2])
        rad = np.where(inside, {2: 10.0, 3: 100.0}.get(band, rad), rad)
    if snow:  # snow: bright in the green, dark in the first SWIR band
        inside = (xx > X0 + 3200) & (yy > Y0 + 2200)
        rad = np.where(inside, {1: 200.0, 4: 20.0}.get(band, rad), rad)
    return rad


def write_scene(project: Project, scene_id: str, *, bands=tuple(range(1, 10)), veg=(X0, X0 + 800, 800), snow=True, gain="normal") -> None:
    """A scene folder named the way the ASTER connector writes it, with digital numbers from ``band_array``."""
    folder = project.raw_dir / "satellite" / "aster" / scene_id
    folder.mkdir(parents=True, exist_ok=True)
    for band in bands:
        res = 15.0 if band <= 3 else 30.0
        dn = np.rint(band_array(band, res, veg=veg, snow=snow) / UCC[gain][band] + 1).astype(np.uint16)
        label = {1: "VNIR_B01", 2: "VNIR_B02", 3: "VNIR_B03N"}.get(band, f"SWIR_B0{band}")
        transform = from_origin(X0 - MARGIN, Y0 + H + MARGIN, res, res)
        with rasterio.open(folder / f"{scene_id}_{label}.tif", "w", driver="GTiff", width=dn.shape[1], height=dn.shape[0], count=1, dtype="uint16", crs=CRS, transform=transform, nodata=0) as ds:
            ds.write(dn, 1)


def log_scene(project: Project, scene_id: str) -> None:
    project.provenance.log(dataset="aster", source="NASA LP DAAC", identifier=scene_id, url=None, license="CC0-1.0 (test)", operation="fetch", parameters={})


def value_at(project: Project, name: str, dx: float, dy: float) -> float:
    with rasterio.open(project.root / "rasters" / f"aster_{name}.tif") as ds:
        return float(next(ds.sample([(X0 + dx, Y0 + dy)]))[0])


@pytest.fixture
def project(tmp_path) -> Project:
    proj = build_project(tmp_path, n_occ=60, seed=4, raster=False)
    minfile = gpd.read_file(proj.data_gpkg, layer="minfile")
    minfile["DEPOSIT_TYPE_CODE1"] = "I05"  # a metallic family, as the real layer has
    minfile.to_file(proj.data_gpkg, layer="minfile", driver="GPKG", engine="pyogrio")
    return Project(proj.root)


@pytest.fixture
def built(project) -> Project:
    log_scene(project, "SCENE_A")
    write_scene(project, "SCENE_A")
    build_aster(project)
    return project


# --- the arithmetic -----------------------------------------------------------------------------------------------------------


def test_radiance_is_dn_minus_one_times_the_coefficient_and_zero_is_no_data():
    got = radiance(np.array([0, 1, 11]), 1, "normal")
    assert np.isnan(got[0]) and got[1] == 0 and got[2] == pytest.approx(16.88)
    assert radiance(np.array([11]), 1, "high")[0] == pytest.approx(6.76)  # gain changes the scale
    with pytest.raises(Exception, match="no unit conversion"):
        radiance(np.array([5]), 12, "normal")


def test_the_five_indices_are_the_published_ratios():
    rad = {b: np.full((3, 3), v, dtype=np.float32) for b, v in BASE.items()}
    layers, bare, valid = indices_from_radiance(rad, 0.3, 0.4)
    assert bare.all() and valid.all()
    for name, expected in EXPECTED_WEST.items():
        assert np.allclose(layers[name], expected), name
    assert set(layers) == set(INDICES)


def test_vegetation_and_snow_are_masked_and_a_vnir_only_scene_cannot_see_snow():
    rad = {b: np.full((2, 2), v, dtype=np.float32) for b, v in BASE.items()}
    rad[2][0, 0], rad[3][0, 0] = 10.0, 100.0  # NDVI 0.82
    rad[1][1, 1], rad[4][1, 1] = 200.0, 20.0  # NDSI 0.82
    layers, bare, _ = indices_from_radiance(rad, 0.3, 0.4)
    assert bare.tolist() == [[False, True], [True, False]]
    assert all(np.isnan(v[0, 0]) and np.isnan(v[1, 1]) and np.isfinite(v[0, 1]) for v in layers.values())
    vnir = {b: rad[b] for b in (1, 2, 3)}
    layers, bare, _ = indices_from_radiance(vnir, 0.3, 0.4)
    assert set(layers) == {"ferric"} and bare.tolist() == [[False, True], [True, True]]  # no SWIR, no snow test


def test_a_zero_denominator_gives_no_value_not_infinity():
    rad = {1: np.array([[0.0, 40.0]]), 2: np.array([[50.0, 50.0]])}
    out, _, _ = indices_from_radiance(rad, 0.3, 0.4)
    assert np.isnan(out["ferric"][0, 0]) and out["ferric"][0, 1] == pytest.approx(1.25)


# --- scenes --------------------------------------------------------------------------------------------------------------------


def test_scene_files_are_recognised_by_band_and_stray_folders_ignored(project):
    write_scene(project, "SCENE_A")
    (project.raw_dir / "satellite" / "aster" / "EMPTY").mkdir()
    (project.raw_dir / "satellite" / "aster" / "ONE").mkdir()
    (project.raw_dir / "satellite" / "aster" / "ONE" / "x_VNIR_B01.tif").write_bytes(b"")
    bands = scene_bands(project.raw_dir / "satellite" / "aster" / "SCENE_A")
    assert sorted(bands) == list(range(1, 10)) and bands[3].name.endswith("VNIR_B03N.tif")
    assert list(find_scenes(project)) == ["SCENE_A"]  # a folder needs two bands


def test_no_scenes_is_an_error_that_says_how_to_get_them(project):
    with pytest.raises(SpectralError, match="harvest --source aster"):
        build_aster(project)
    summary = yaml.safe_load((project.root / "analysis" / "spectral_summary.json").read_text())
    assert summary["available"] is False and "no ASTER scenes" in summary["reason"]


# --- the rasters and summary -------------------------------------------------------------------------------------------


def test_rasters_hold_the_ratios_and_the_masks_remove_vegetation_and_snow(built):
    for name in INDICES:
        assert value_at(built, name, 1000, 1500) == pytest.approx(EXPECTED_WEST[name], rel=0.04), name  # west, plain ground
        assert value_at(built, name, 2800, 1500) == pytest.approx(EXPECTED_EAST[name], rel=0.04), name  # east, altered
        assert np.isnan(value_at(built, name, 300, 300)), name  # under vegetation
        assert np.isnan(value_at(built, name, 3600, 2600)), name  # under snow
    with rasterio.open(built.root / "rasters" / "aster_aloh.tif") as ds:
        assert ds.crs.to_epsg() == 32610 and abs(ds.transform.a) == 30.0 and ds.dtypes[0] == "float32"


def test_summary_reports_coverage_and_how_well_each_index_separates_occurrences(built):
    s = yaml.safe_load((built.root / "analysis" / "spectral_summary.json").read_text())
    assert s["available"] and s["n_scenes"] == 1 and s["gain"] == "normal" and s["resolution_m"] == 30.0
    rows = {r["name"]: r for r in s["indices"]}
    assert set(rows) == set(INDICES)
    assert 85 < rows["aloh"]["valid_pct"] < 95  # about 10% of the AOI is vegetation or snow
    assert rows["aloh"]["n_occurrences"] >= 10 and rows["aloh"]["auc"] > 0.7  # the occurrences sit in the altered half
    assert rows["ferric"]["auc"] == pytest.approx(0.5, abs=0.15)  # the ferric ratio is flat across the AOI, so it cannot separate anything
    assert rows["aloh"]["median_occurrence"] > rows["aloh"]["median_background"]
    assert 85 < s["median_bare_pct"] < 95


def test_outputs_are_logged_with_the_scenes_licence_and_the_settings_used(built):
    records = [r for r in built.provenance.read() if r.operation == "spectral_index"]
    assert {r.dataset for r in records} == {f"aster_{n}" for n in INDICES}
    aloh = next(r for r in records if r.dataset == "aster_aloh")
    assert "CC0-1.0 (test)" in aloh.license and aloh.parameters["formula"] == "(B5 + B7) / B6"
    assert aloh.parameters["scenes"] == ["SCENE_A"] and aloh.parameters["gain"] == "normal" and aloh.parameters["inputs"] == ["aster"]


def test_scenes_fill_each_others_gaps_and_a_vnir_only_scene_adds_only_ferric(project):
    write_scene(project, "SCENE_A", snow=True)
    write_scene(project, "SCENE_B", snow=False)  # the same ground a season later, without the snow
    build_aster(project)
    assert value_at(project, "aloh", 3600, 2600) == pytest.approx(EXPECTED_EAST["aloh"], rel=0.02)  # the median of one valid cell
    assert np.isnan(value_at(project, "aloh", 300, 300))  # vegetation in both
    only_vnir = project.raw_dir / "satellite" / "aster"
    for scene in ("SCENE_A", "SCENE_B"):
        for f in (only_vnir / scene).glob("*SWIR*"):
            f.unlink()
    result = build_aster(project, force=True)
    assert [i["name"] for i in result.summary["indices"]] == ["ferric"]  # the SWIR indices of the earlier run are gone
    assert not (project.root / "rasters" / "aster_aloh.tif").exists()


def test_rasters_are_kept_until_a_scene_is_newer_or_forced(built):
    path = built.root / "rasters" / "aster_aloh.tif"
    old = time.time() - 100_000
    for p in (built.root / "rasters").glob("aster_*.tif"):
        os.utime(p, (old, old))
    assert "kept" not in " ".join(build_aster(built).summary["notes"])  # older than the scene: rebuilt
    rebuilt = path.stat().st_mtime
    assert rebuilt > old + 1000
    result = build_aster(built)
    assert "rasters are newer than the scenes; kept" in result.summary["notes"] and path.stat().st_mtime == rebuilt
    assert result.summary["median_bare_pct"] is not None  # carried over from the earlier run
    build_aster(built, force=True)
    assert path.stat().st_mtime >= rebuilt


def test_the_gain_setting_scales_the_radiance_but_barely_moves_a_ratio(project):
    write_scene(project, "SCENE_A", gain="high")
    project.config.spectral.gain = "high"
    project.config.save(project.root)
    build_aster(Project(project.root))
    assert value_at(project, "aloh", 2800, 1500) == pytest.approx(EXPECTED_EAST["aloh"], rel=0.02)  # consistent gain: same ratios
    bad = Project(project.root)
    bad.config.spectral.gain = "normal"  # the scene was made at high gain but read as normal: the ratios shift, as the docs warn
    bad.config.save(project.root)
    build_aster(Project(project.root), force=True)
    assert value_at(project, "ferric", 2800, 1500) != pytest.approx(EXPECTED_EAST["ferric"], rel=0.02)


def test_settings_validate_and_old_projects_without_them_load(project):
    with pytest.raises(ValueError, match="gain"):
        ProjectConfig.model_validate({**project.config.model_dump(), "spectral": {"gain": "medium"}})
    data = yaml.safe_load((project.root / "project.yaml").read_text())
    data.pop("spectral")
    (project.root / "project.yaml").write_text(yaml.safe_dump(data))
    assert Project(project.root).config.spectral.ndvi_max == 0.3


# --- the model, the command, the pipeline and the report -------------------------------------------------------------------


def stack_names(project: Project):
    project.config.target = small_cfg(effort="ignore")
    project.config.save(project.root)
    proj = Project(project.root)
    return build_stack(proj, build_grid(proj.read_aoi(), CRS, 100.0, 10**6), proj.config.target)


def test_the_indices_become_features_when_they_cover_enough_ground(built):
    stack = stack_names(built)
    assert "aster_aloh" in stack.names  # the one index this scene varies
    for name in ("aster_mgoh", "aster_ferric", "aster_gossan", "aster_ferrous"):  # flat in the synthetic scene: dropped for that, not for coverage
        assert name in stack.names or stack.skipped[name] == "no variation across the AOI"


def test_a_mostly_forested_area_drops_the_indices_and_says_why(project):
    write_scene(project, "FOREST", veg=(X0, X0 + 3600, 3000), snow=False)  # nine tenths of the AOI under vegetation
    build_aster(project)
    stack = stack_names(project)
    assert not any(n.startswith("aster_") for n in stack.names)
    assert "no value" in stack.skipped["aster_aloh"] and "limit" in stack.skipped["aster_aloh"]


def test_the_command_builds_and_prints_the_numbers_and_fails_cleanly_without_scenes(project):
    failed = runner.invoke(app, ["spectral", "demo"])
    assert failed.exit_code == 1 and "no ASTER scenes" in failed.output
    write_scene(project, "SCENE_A")
    ok = runner.invoke(app, ["spectral", "demo"])
    assert ok.exit_code == 0, ok.output
    assert "1 scene(s)" in ok.output and "aster_aloh" in ok.output and "(B5 + B7) / B6" in ok.output and "AUC" in ok.output


def test_the_pipeline_builds_the_indices_before_the_model_runs(project, monkeypatch):
    write_scene(project, "SCENE_A")
    seen = {}

    def fake_target(proj, cfg, cross_validate=True):
        seen["rasters"] = sorted(p.name for p in (proj.root / "rasters").glob("aster_*.tif"))
        raise TargetError("stop here")

    monkeypatch.setattr(pipeline, "run_target", fake_target)
    lines = []

    class Rec(pipeline.Reporter):
        def say(self, level, text):
            lines.append(text)

    outcomes = pipeline.run_pipeline(project, ["target"], reporter=Rec())
    assert seen["rasters"] == [f"aster_{n}.tif" for n in sorted(INDICES)]
    assert outcomes[0].status == pipeline.SKIPPED and any(line.startswith("spectral: 5 ASTER index raster(s)") for line in lines)


def report_text(project: Project) -> str:
    assert runner.invoke(app, ["analyze", "demo", "--steps", "occurrences"]).exit_code == 0
    assert runner.invoke(app, ["report", "demo", "--kind", "analysis", "--format", "pdf"]).exit_code == 0
    with pymupdf.open(project.root / "outputs" / "analysis_report.pdf") as pdf:
        return " ".join(" ".join(page.get_text() for page in pdf).split())


def test_the_report_has_an_honest_spectral_section_only_when_there_is_something_to_say(built, tmp_path):
    text = report_text(built)
    assert "Satellite alteration indices (ASTER)" in text and "grey cell is unknown, not unaltered" in text
    assert "Al-OH (white mica, clay)" in text and "(B5 + B7) / B6" in text and "Occurrence AUC" in text
    assert "A typical scene leaves" in text


def test_the_report_has_no_spectral_section_without_the_step(project):
    assert "Satellite alteration indices" not in report_text(project)
