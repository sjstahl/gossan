"""Tests for qgis/gossan_toolbox.py with a stand-in for QGIS (no QGIS needed to run them)."""

import importlib.util
import io
import re
import sys
import types
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "qgis" / "gossan_toolbox.py"


# --- a minimal fake of the parts of QGIS the script uses ----------------------------------


class Recorder:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs

    def name(self):
        return self.args[0]


def build_fake_qgis():
    core = types.ModuleType("qgis.core")

    class QgsProcessingException(Exception):
        pass

    class QgsProcessing:
        TypeVectorPolygon = "polygon"

    class QgsProcessingAlgorithm:
        def __init__(self):
            self.params, self.outputs = [], []

        def addParameter(self, p):
            self.params.append(p)

        def addOutput(self, o):
            self.outputs.append(o)

        def parameterAsString(self, parameters, name, context):
            return parameters.get(name) or ""

        def parameterAsBool(self, parameters, name, context):
            return bool(parameters.get(name, False))

        def parameterAsDouble(self, parameters, name, context):
            return float(parameters.get(name, 0))

        def checkParameterValues(self, parameters, context):
            return True, ""

    class FileParam(Recorder):
        Folder, File = "folder", "file"

    class NumberParam(Recorder):
        Double = "double"

    class Project:
        _instance = None

        def __init__(self):
            self.layers = []

        @classmethod
        def instance(cls):
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

        def addMapLayer(self, layer):
            self.layers.append(layer)
            return layer

    class Layer:
        def __init__(self, source, name, *rest):
            self.source, self.layer_name, self.renderer_set = source, name, None
            self.valid = "invalid" not in str(source)

        def isValid(self):
            return self.valid

        def setRenderer(self, r):
            self.renderer_set = r

        def triggerRepaint(self):
            pass

    class QgsRasterLayer(Layer):
        stats = (0.1, 0.9)

        def dataProvider(self):
            lo, hi = QgsRasterLayer.stats
            provider = types.SimpleNamespace(bandStatistics=lambda *a: types.SimpleNamespace(minimumValue=lo, maximumValue=hi))
            return provider

        def extent(self):
            return "extent"

    class QgsColorRampShader:
        Interpolated = "interpolated"

        class ColorRampItem:
            def __init__(self, value, color, label=""):
                self.value, self.color, self.label = value, color, label

        def __init__(self, lo=0.0, hi=1.0):
            self.range = (lo, hi)
            self.type, self.items = None, []

        def setColorRampType(self, t):
            self.type = t

        def setColorRampItemList(self, items):
            self.items = items

    class QgsRasterShader:
        def setRasterShaderFunction(self, f):
            self.function = f

    class QgsFillSymbol:
        @staticmethod
        def createSimple(props):
            return types.SimpleNamespace(props=props)

    namespace = {
        "Qgis": types.SimpleNamespace(),  # no RasterBandStatistic: exercises the older-QGIS fallback
        "QgsCategorizedSymbolRenderer": type("QgsCategorizedSymbolRenderer", (Recorder,), {}),
        "QgsRendererCategory": type("QgsRendererCategory", (Recorder,), {}),
        "QgsColorRampShader": QgsColorRampShader,
        "QgsFillSymbol": QgsFillSymbol,
        "QgsProcessing": QgsProcessing,
        "QgsProcessingAlgorithm": QgsProcessingAlgorithm,
        "QgsProcessingException": QgsProcessingException,
        "QgsProcessingOutputFolder": type("QgsProcessingOutputFolder", (Recorder,), {}),
        "QgsProcessingParameterBoolean": type("QgsProcessingParameterBoolean", (Recorder,), {}),
        "QgsProcessingParameterFeatureSource": type("QgsProcessingParameterFeatureSource", (Recorder,), {}),
        "QgsProcessingParameterFile": FileParam,
        "QgsProcessingParameterNumber": NumberParam,
        "QgsProcessingParameterString": type("QgsProcessingParameterString", (Recorder,), {}),
        "QgsProject": Project,
        "QgsRasterBandStats": types.SimpleNamespace(Min=1, Max=2),
        "QgsRasterLayer": QgsRasterLayer,
        "QgsRasterShader": QgsRasterShader,
        "QgsSingleBandPseudoColorRenderer": type("QgsSingleBandPseudoColorRenderer", (Recorder,), {}),
        "QgsSingleSymbolRenderer": type("QgsSingleSymbolRenderer", (Recorder,), {}),
        "QgsVectorLayer": type("QgsVectorLayer", (Layer,), {}),
    }
    for k, v in namespace.items():
        setattr(core, k, v)

    qgis = types.ModuleType("qgis")
    pyqt = types.ModuleType("qgis.PyQt")
    qtcore = types.ModuleType("qgis.PyQt.QtCore")
    qtgui = types.ModuleType("qgis.PyQt.QtGui")
    qtcore.QCoreApplication = types.SimpleNamespace(translate=lambda ctx, text: text)
    qtgui.QColor = lambda name: types.SimpleNamespace(name=name)

    processing = types.ModuleType("processing")
    processing.calls = []
    processing.run = lambda alg, params, **kw: processing.calls.append((alg, params, kw)) or {}
    modules = {"qgis": qgis, "qgis.core": core, "qgis.PyQt": pyqt, "qgis.PyQt.QtCore": qtcore, "qgis.PyQt.QtGui": qtgui, "processing": processing}
    return modules


class FakeFeedback:
    def __init__(self, cancel=False):
        self.info, self.commands, self.errors, self.progress, self.cancel = [], [], [], [], cancel

    def pushInfo(self, text):
        self.info.append(text)

    def pushCommandInfo(self, text):
        self.commands.append(text)

    def reportError(self, text, fatal=False):
        self.errors.append((text, fatal))

    def setProgress(self, value):
        self.progress.append(value)

    def isCanceled(self):
        return self.cancel


class FakeProcess:
    def __init__(self, lines, code):
        self.stdout, self.code, self.terminated = io.StringIO("".join(f"{line}\n" for line in lines)), code, False

    def terminate(self):
        self.terminated = True

    def wait(self):
        return self.code


@pytest.fixture
def toolbox(monkeypatch):
    modules = build_fake_qgis()
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("gossan_toolbox_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.fake_project = modules["qgis.core"].QgsProject
    module.fake_project._instance = None
    module.fake_processing = modules["processing"]
    return module


@pytest.fixture
def cli(toolbox, monkeypatch):
    """Replace subprocess.Popen: records every launch, answers from ``scripted`` by gossan sub-command."""
    state = types.SimpleNamespace(launches=[], scripted={}, procs=[])

    def popen(args, **kwargs):
        sub = args[1]
        lines, code = state.scripted.get(sub, ([f"{sub} ok"], 0))
        state.launches.append((list(args), kwargs))
        proc = FakeProcess(lines, code)
        state.procs.append(proc)
        return proc

    monkeypatch.setattr(toolbox.subprocess, "Popen", popen)
    return state


def make_params(tmp_path, **kw):
    exe = tmp_path / "venv" / "gossan.exe"
    exe.parent.mkdir(exist_ok=True)
    exe.write_text("")
    base = tmp_path / "base"
    base.mkdir(exist_ok=True)
    return {"INPUT": "aoi-layer", "PROJECT_NAME": "demo", "BASE_DIR": str(base), "GOSSAN": str(exe), "RESOLUTION": 150.0, "REUSE": False, **kw}


# --- the script itself ----------------------------------------------------------------------


def test_script_does_not_import_gossan_and_only_uses_subprocess():
    source = SCRIPT.read_text(encoding="utf-8")
    imports = re.findall(r"^\s*(?:from|import)\s+([\w.]+)", source, flags=re.MULTILINE)
    assert not [m for m in imports if m.split(".")[0] in {"gossan", "geopandas", "rasterio", "sklearn", "pandas"}]
    assert "subprocess.Popen" in source and "QgsProject.instance()" in source and "addMapLayer" in source


def test_defines_one_algorithm_with_the_requested_parameters(toolbox):
    alg = toolbox.GossanTargetingAlgorithm()
    alg.initAlgorithm()
    by_name = {p.name(): p for p in alg.params}
    assert list(by_name) == ["INPUT", "PROJECT_NAME", "BASE_DIR", "GOSSAN", "RESOLUTION", "REUSE", "ANALYZE"]
    assert by_name["ANALYZE"].kwargs["defaultValue"] is True  # analysing and reporting is the default
    assert type(by_name["INPUT"]).__name__ == "QgsProcessingParameterFeatureSource" and by_name["INPUT"].args[2] == ["polygon"]  # polygon layers only
    assert type(by_name["PROJECT_NAME"]).__name__ == "QgsProcessingParameterString"
    assert by_name["BASE_DIR"].kwargs["behavior"] == "folder"
    assert by_name["GOSSAN"].kwargs["behavior"] == "file" and by_name["GOSSAN"].kwargs["optional"] is True
    assert by_name["RESOLUTION"].kwargs["defaultValue"] == 100.0
    assert alg.name() == "gossan_harvest_and_target" and alg.group() == "Gossan" and alg.groupId() == "gossan"
    assert "not a probability" in alg.shortHelpString() and "Access Only" in alg.shortHelpString()
    assert isinstance(alg.createInstance(), toolbox.GossanTargetingAlgorithm)


# --- input checks ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["", "  ", "a/b", "a:b", "..", 'q"x'])
def test_bad_project_names_are_refused(toolbox, tmp_path, name):
    ok, message = toolbox.GossanTargetingAlgorithm().checkParameterValues(make_params(tmp_path, PROJECT_NAME=name), None)
    assert not ok and message


def test_base_directory_executable_and_existing_project_are_checked(toolbox, tmp_path, monkeypatch):
    alg = toolbox.GossanTargetingAlgorithm()
    params = make_params(tmp_path)
    assert alg.checkParameterValues(params, None) == (True, "")

    assert not alg.checkParameterValues({**params, "BASE_DIR": str(tmp_path / "nope")}, None)[0]

    (Path(params["BASE_DIR"]) / "demo").mkdir()
    ok, message = alg.checkParameterValues(params, None)
    assert not ok and "already exists" in message and "Reuse" in message
    assert alg.checkParameterValues({**params, "REUSE": True}, None)[0]

    monkeypatch.delenv("GOSSAN_EXECUTABLE", raising=False)
    monkeypatch.setattr(toolbox.shutil, "which", lambda name: None)
    ok, message = alg.checkParameterValues({**params, "GOSSAN": "", "REUSE": True}, None)
    assert not ok and "Gossan was not found" in message and "GOSSAN_EXECUTABLE" in message


def test_finding_the_executable(toolbox, tmp_path, monkeypatch):
    exe = tmp_path / "gossan.exe"
    exe.write_text("")
    assert toolbox.find_gossan(str(exe)) == str(exe)
    monkeypatch.setenv("GOSSAN_EXECUTABLE", str(exe))
    assert toolbox.find_gossan(None) == str(exe)
    assert toolbox.find_gossan(str(tmp_path / "missing.exe")) == str(exe)  # a bad explicit path falls through
    monkeypatch.delenv("GOSSAN_EXECUTABLE")
    monkeypatch.setattr(toolbox.shutil, "which", lambda name: "/usr/bin/gossan")
    assert toolbox.find_gossan(None) == "/usr/bin/gossan"


def test_child_environment_drops_qgis_python_and_gdal_settings(toolbox):
    sep = toolbox.os.pathsep
    env = {
        "PYTHONHOME": "C:/QGIS/apps/Python312", "PYTHONPATH": "C:/QGIS/python", "QGIS_PREFIX_PATH": "C:/QGIS/apps/qgis",
        "GDAL_DATA": "C:/QGIS/share/gdal", "PROJ_LIB": "C:/QGIS/share/proj", "proj_data": "x", "OSGEO4W_ROOT": "C:/OSGeo4W",
        "PATH": sep.join(["/Program Files/QGIS 3.36/bin", "/Windows", "/OSGeo4W/bin", "/Users/me/venv/Scripts"]),  # no drive letters: ":" is the separator on Linux
        "HOME": "/home/me", "ANTHROPIC_API_KEY": "kept",
    }  # fmt: skip
    original = dict(env)
    clean = toolbox.clean_environment(env)
    for gone in ("PYTHONHOME", "PYTHONPATH", "QGIS_PREFIX_PATH", "GDAL_DATA", "PROJ_LIB", "proj_data", "OSGEO4W_ROOT"):
        assert gone not in clean
    assert clean["PATH"].split(sep) == ["/Windows", "/Users/me/venv/Scripts"]  # QGIS and OSGeo4W entries dropped
    assert clean["HOME"] == "/home/me" and clean["ANTHROPIC_API_KEY"] == "kept"  # everything else untouched
    assert clean["PYTHONUTF8"] == "1" and clean["PYTHONIOENCODING"] == "utf-8"
    assert env == original  # the caller's environment is not modified


# --- running Gossan ------------------------------------------------------------------------


def run_algorithm(toolbox, params, feedback=None):
    alg = toolbox.GossanTargetingAlgorithm()
    feedback = feedback or FakeFeedback()
    return alg, feedback, alg.processAlgorithm(params, None, feedback)


def test_runs_new_harvest_and_target_in_order_in_the_base_directory(toolbox, cli, tmp_path):
    params = make_params(tmp_path)
    _alg, feedback, result = run_algorithm(toolbox, params)
    exe, base = params["GOSSAN"], params["BASE_DIR"]
    launches = cli.launches
    assert [a[1] for a, _ in launches] == ["new", "harvest", "target"]
    assert all(a[0] == exe for a, _ in launches)
    assert launches[0][0][2:4] == ["demo", "--aoi"] and launches[0][0][4].endswith("demo_aoi.gpkg")
    assert launches[1][0] == [exe, "harvest", "demo"]
    assert launches[2][0] == [exe, "target", "demo", "--resolution", "150"]
    assert all(kw["cwd"] == base for _, kw in launches)  # the project folder is created inside the base directory
    assert result == {"PROJECT_DIR": str(Path(base) / "demo")}

    # The AOI layer was exported by QGIS itself, to a temporary file that is gone afterwards.
    (alg_name, export_params, _), = toolbox.fake_processing.calls
    assert alg_name == "native:savefeatures" and export_params["INPUT"] == "aoi-layer"
    assert not Path(export_params["OUTPUT"]).exists()
    assert feedback.progress == sorted(feedback.progress) and feedback.progress[-1] == 95


def test_children_run_isolated_from_the_qgis_environment(toolbox, cli, tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONHOME", "C:/QGIS/apps/Python312")
    monkeypatch.setenv("GDAL_DATA", "C:/QGIS/share/gdal")
    run_algorithm(toolbox, make_params(tmp_path))
    for _, kw in cli.launches:
        assert "PYTHONHOME" not in kw["env"] and "GDAL_DATA" not in kw["env"] and kw["env"]["PYTHONUTF8"] == "1"
        assert kw["stdin"] is toolbox.subprocess.DEVNULL and kw["stderr"] is toolbox.subprocess.STDOUT
    assert "PYTHONHOME" in toolbox.os.environ  # QGIS's own environment is left alone


def test_command_output_is_streamed_to_the_log(toolbox, cli, tmp_path):
    cli.scripted["target"] = (["Grid: 40 x 30 cells", "Targets: 5 polygon(s)"], 0)
    _, feedback, _ = run_algorithm(toolbox, make_params(tmp_path))
    assert "Grid: 40 x 30 cells" in feedback.info and "Targets: 5 polygon(s)" in feedback.info
    assert any(c.endswith("target demo --resolution 150") for c in feedback.commands)


def test_a_failed_new_stops_the_run_with_the_programs_message(toolbox, cli, tmp_path):
    cli.scripted["new"] = (["Error: demo already exists"], 1)
    with pytest.raises(toolbox.QgsProcessingException, match="(?s)gossan new failed.*already exists"):
        run_algorithm(toolbox, make_params(tmp_path))
    assert [a[1] for a, _ in cli.launches] == ["new"]


def test_a_partly_failed_harvest_is_reported_and_the_run_continues(toolbox, cli, tmp_path):
    cli.scripted["harvest"] = (["aris_index: HTTP Error 502"], 1)
    _alg, feedback, result = run_algorithm(toolbox, make_params(tmp_path))
    assert [a[1] for a, _ in cli.launches] == ["new", "harvest", "target"]
    assert any("harvest reported a problem" in text and fatal is False for text, fatal in feedback.errors)
    assert "PROJECT_DIR" in result


def test_a_failed_target_stops_the_run_and_shows_the_reason(toolbox, cli, tmp_path):
    cli.scripted["target"] = (["Error: 6 known occurrence(s) fall in the AOI", "at least 10 are needed"], 1)
    with pytest.raises(toolbox.QgsProcessingException) as err:
        run_algorithm(toolbox, make_params(tmp_path))
    assert "gossan target failed (exit code 1)" in str(err.value) and "at least 10 are needed" in str(err.value)


def test_reusing_a_project_skips_the_export_and_new(toolbox, cli, tmp_path):
    params = make_params(tmp_path, REUSE=True)
    project = Path(params["BASE_DIR"]) / "demo"
    project.mkdir()
    (project / "project.yaml").write_text("name: demo\n")
    _, feedback, _ = run_algorithm(toolbox, params)
    assert [a[1] for a, _ in cli.launches] == ["harvest", "target"]
    assert toolbox.fake_processing.calls == [] and any("AOI layer is ignored" in text for text in feedback.info)


def test_cancel_stops_the_running_command(toolbox, cli, tmp_path):
    with pytest.raises(toolbox.QgsProcessingException, match="Cancelled"):
        run_algorithm(toolbox, make_params(tmp_path), FakeFeedback(cancel=True))
    assert cli.procs and cli.procs[0].terminated


# --- loading the results --------------------------------------------------------------------


def finished(toolbox, tmp_path, *, with_targets=True, with_raster=True):
    target = tmp_path / "base" / "demo" / "target"
    target.mkdir(parents=True)
    if with_raster:
        (target / "prospectivity.tif").write_bytes(b"x")
    if with_targets:
        (target / "targets.gpkg").write_bytes(b"x")
    alg = toolbox.GossanTargetingAlgorithm()
    alg._to_load = {"prospectivity": str(target / "prospectivity.tif"), "targets": str(target / "targets.gpkg")}
    return alg


def test_results_are_added_to_the_project_raster_first_then_targets(toolbox, tmp_path):
    alg = finished(toolbox, tmp_path)
    assert alg.postProcessAlgorithm(None, FakeFeedback()) == {}
    layers = toolbox.fake_project.instance().layers
    assert [layer.layer_name for layer in layers] == ["Gossan prospectivity", "Gossan targets (top percentile)"]  # polygons end up on top
    assert layers[0].source.endswith("prospectivity.tif")
    assert layers[1].source.endswith("targets.gpkg|layername=targets_top")


def test_prospectivity_gets_a_pseudocolour_ramp_over_its_value_range(toolbox, tmp_path):
    alg = finished(toolbox, tmp_path)
    alg.postProcessAlgorithm(None, FakeFeedback())
    raster = toolbox.fake_project.instance().layers[0]
    renderer = raster.renderer_set
    assert type(renderer).__name__ == "QgsSingleBandPseudoColorRenderer" and renderer.args[1] == 1  # band 1
    ramp = renderer.args[2].function
    assert ramp.type == "interpolated" and ramp.range == (0.1, 0.9)  # from the layer's own statistics
    assert [round(i.value, 3) for i in ramp.items] == [0.1, 0.3, 0.5, 0.7, 0.9]
    assert [i.color.name for i in ramp.items] == list(toolbox.RAMP)  # viridis, low to high
    outline = toolbox.fake_project.instance().layers[1].renderer_set.args[0].props
    assert outline["color"] == "0,0,0,0" and outline["outline_color"].startswith("255,255,255")  # hollow, white outline


def test_a_flat_raster_falls_back_to_the_score_range(toolbox, tmp_path):
    toolbox.QgsRasterLayer.stats = (0.5, 0.5)
    alg = finished(toolbox, tmp_path)
    alg.postProcessAlgorithm(None, FakeFeedback())
    ramp = toolbox.fake_project.instance().layers[0].renderer_set.args[2].function
    assert ramp.range == (0.0, 1.0)


def test_no_target_polygons_means_only_the_raster_is_loaded(toolbox, tmp_path):
    alg = finished(toolbox, tmp_path, with_targets=False)
    feedback = FakeFeedback()
    alg.postProcessAlgorithm(None, feedback)
    assert [layer.layer_name for layer in toolbox.fake_project.instance().layers] == ["Gossan prospectivity"]
    assert any("No target polygons were produced" in text for text in feedback.info)


def test_unreadable_layers_are_reported_not_added(toolbox, tmp_path):
    alg = finished(toolbox, tmp_path)
    alg._to_load["prospectivity"] = str(tmp_path / "base" / "demo" / "target" / "invalid.tif")
    Path(alg._to_load["prospectivity"]).write_bytes(b"x")
    feedback = FakeFeedback()
    alg.postProcessAlgorithm(None, feedback)
    assert [layer.layer_name for layer in toolbox.fake_project.instance().layers] == ["Gossan targets (top percentile)"]
    assert any("Could not open" in text for text, _ in feedback.errors)


def test_post_process_with_nothing_to_load_is_harmless(toolbox):
    alg = toolbox.GossanTargetingAlgorithm()
    assert alg.postProcessAlgorithm(None, FakeFeedback()) == {}
    assert toolbox.fake_project.instance().layers == []


# --- the same script inside a real QGIS (skipped where QGIS is not installed) ---------------


def find_python_qgis() -> Path | None:
    import glob
    import os

    explicit = os.environ.get("GOSSAN_QGIS_PYTHON_BAT")
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    for pattern in (r"C:\Program Files\QGIS*\bin\python-qgis*.bat", r"C:\OSGeo4W*\bin\python-qgis*.bat"):
        found = sorted(glob.glob(pattern))
        if found:
            return Path(found[-1])
    return None


@pytest.mark.skipif(sys.platform != "win32" or find_python_qgis() is None, reason="needs a Windows QGIS install (python-qgis.bat)")
def test_runs_in_a_real_qgis_with_a_stand_in_for_the_cli(tmp_path):
    """Real QGIS API calls: parameters, AOI export, processing, layer loading and symbology.

    The Gossan program is replaced by a batch file that creates the folders and copies in small
    prepared outputs, so no network or modelling is involved.
    """
    import json
    import subprocess

    import geopandas as gpd
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds
    from shapely.geometry import box

    crs = "EPSG:32610"
    aoi = gpd.GeoDataFrame({"id": [1]}, geometry=[box(690000, 5617000, 694000, 5620000)], crs=crs)
    aoi_path = tmp_path / "aoi.gpkg"
    aoi.to_file(aoi_path, driver="GPKG")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    data = np.linspace(0.1, 0.9, 40 * 30, dtype=np.float32).reshape(30, 40)
    with rasterio.open(prepared / "prospectivity.tif", "w", driver="GTiff", width=40, height=30, count=1, dtype="float32", crs=crs,
                       transform=from_bounds(690000, 5617000, 694000, 5620000, 40, 30), nodata=-9999.0) as ds:  # fmt: skip
        ds.write(data, 1)
    gpd.GeoDataFrame({"rank": [1, 2, 3]}, geometry=[box(691000, 5618000, 691400, 5618400), box(692000, 5618500, 692300, 5618800), box(693000, 5617200, 693200, 5617400)], crs=crs).to_file(
        prepared / "targets.gpkg", layer="targets_top", driver="GPKG"
    )
    gpd.GeoDataFrame({"sample": [0, 1]}, geometry=[box(690500, 5617500, 692000, 5619000), box(692000, 5617500, 693500, 5619000)], crs=crs).to_file(
        prepared / "analysis.gpkg", layer="rgs_catchments", driver="GPKG"
    )
    gpd.GeoDataFrame(
        {"rank": [1, 2, 3], "ground_status": ["open", "held", "excluded"]},
        geometry=[box(691000, 5618000, 691400, 5618400), box(692000, 5618500, 692300, 5618800), box(693000, 5617200, 693200, 5617400)], crs=crs,
    ).to_file(prepared / "analysis.gpkg", layer="target_decisions", driver="GPKG")
    fake = tmp_path / "fake_gossan.bat"
    bs = chr(92)  # a backslash, spelled out so no escape sequence can turn it into a tab or newline
    out = f'"%2{bs}target{bs}"'
    nl = chr(13) + chr(10)  # batch files want CRLF line endings
    fake.write_text(
        nl.join(
            [
                "@echo off",
                'if "%1"=="new" ( mkdir "%2" & echo Created project %2 & exit /b 0 )',
                'if "%1"=="harvest" ( echo minfile: 12 features written & exit /b 0 )',
                (
                    f'if "%1"=="target" ( mkdir "%2{bs}target" & copy /y "{prepared}{bs}prospectivity.tif" {out} >nul & '
                    f'copy /y "{prepared}{bs}targets.gpkg" {out} >nul & echo Targets: 3 & exit /b 0 )'
                ),
                (
                    f'if "%1"=="run" ( mkdir "%2{bs}analysis" & copy /y "{prepared}{bs}analysis.gpkg" "%2{bs}analysis{bs}" >nul & '
                    "echo analyze: ok & exit /b 0 )"
                ),
                "exit /b 2",
                "",
            ]
        )
    )
    base = tmp_path / "base"
    base.mkdir()
    harness = Path(__file__).parent / "qgis_harness.py"
    proc = subprocess.run(
        [str(find_python_qgis()), str(harness), str(SCRIPT), str(aoi_path), str(base), str(fake), "analyze"],
        capture_output=True, text=True, timeout=600, errors="replace", check=False,
    )  # fmt: skip
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT:")), None)
    assert line, f"no result from QGIS:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    result = json.loads(line[len("RESULT:"):])

    assert result["check"] == [True, ""] and result["errors"] == []
    assert Path(result["results"]["PROJECT_DIR"]) == base / "qgis_demo"
    layers = {layer["name"]: layer for layer in result["layers"]}
    assert list(layers) == [
        "Gossan targets by ground status", "Gossan stream-sediment catchments",
        "Gossan targets (top percentile)", "Gossan prospectivity",
    ], result  # the layer tree lists the newest on top
    raster, targets = layers["Gossan prospectivity"], layers["Gossan targets (top percentile)"]
    assert raster["valid"] and raster["type"] == "QgsRasterLayer" and raster["crs"] == "EPSG:32610"
    assert raster["renderer"] == "singlebandpseudocolor" and raster["bands"] == 1
    assert raster["ramp"][0] == pytest.approx(0.1, abs=0.01) and raster["ramp"][-1] == pytest.approx(0.9, abs=0.01)
    assert raster["colours"] == ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"]
    assert targets["valid"] and targets["features"] == 3 and targets["renderer"] == "singleSymbol" and targets["outline"] == "#ffffff"
    catchments, decisions = layers["Gossan stream-sediment catchments"], layers["Gossan targets by ground status"]
    assert catchments["valid"] and catchments["features"] == 2 and catchments["renderer"] == "singleSymbol" and catchments["outline"] == "#898781"
    assert decisions["valid"] and decisions["features"] == 3 and decisions["renderer"] == "categorizedSymbol"
    assert decisions["attribute"] == "ground_status"
    assert decisions["categories"] == [["open", "#2a78d6"], ["partly held", "#1baf7a"], ["held", "#eb6834"], ["excluded", "#52514e"]]


# --- analysing and reporting from QGIS ---------------------------------------------------------


def test_analyze_runs_gossan_run_for_analyze_and_report_after_the_model(toolbox, cli, tmp_path):
    params = make_params(tmp_path, ANALYZE=True)
    alg, feedback, _ = run_algorithm(toolbox, params)
    exe = params["GOSSAN"]
    assert [a[1] for a, _ in cli.launches] == ["new", "harvest", "target", "run"]
    assert cli.launches[3][0] == [exe, "run", "demo", "--steps", "analyze,report"]
    assert feedback.progress == sorted(feedback.progress) and feedback.progress[-1] == 99
    assert alg._to_load["analysis"].endswith("analysis.gpkg") and "analysis" in alg._to_load["analysis"]
    assert not feedback.errors


def test_a_problem_in_the_analysis_is_reported_and_the_model_results_are_still_loaded(toolbox, cli, tmp_path):
    cli.scripted["run"] = (["analyze: skipped: no rgs layer", "report: ok"], 1)  # exit 1 = partial
    alg, feedback, result = run_algorithm(toolbox, make_params(tmp_path, ANALYZE=True))
    assert any("analysis or the report reported a problem" in text and fatal is False for text, fatal in feedback.errors)
    assert "PROJECT_DIR" in result and "prospectivity" in alg._to_load


def test_without_analyze_the_commands_are_unchanged(toolbox, cli, tmp_path):
    alg, _, _ = run_algorithm(toolbox, make_params(tmp_path, ANALYZE=False))
    assert [a[1] for a, _ in cli.launches] == ["new", "harvest", "target"] and "analysis" not in alg._to_load


def test_analysis_layers_are_added_catchments_below_the_status_coloured_targets(toolbox, tmp_path):
    alg = finished(toolbox, tmp_path)
    gpkg = tmp_path / "base" / "demo" / "analysis" / "analysis.gpkg"
    gpkg.parent.mkdir(parents=True)
    gpkg.write_bytes(b"x")
    alg._to_load["analysis"] = str(gpkg)
    alg.postProcessAlgorithm(None, FakeFeedback())
    layers = toolbox.fake_project.instance().layers
    assert [layer.layer_name for layer in layers] == [
        "Gossan prospectivity", "Gossan targets (top percentile)",
        "Gossan stream-sediment catchments", "Gossan targets by ground status",
    ]  # fmt: skip
    assert layers[2].source.endswith("analysis.gpkg|layername=rgs_catchments")
    decisions = layers[3]
    assert decisions.source.endswith("analysis.gpkg|layername=target_decisions")
    renderer = decisions.renderer_set
    assert type(renderer).__name__ == "QgsCategorizedSymbolRenderer" and renderer.args[0] == "ground_status"
    assert [c.args[0] for c in renderer.args[1]] == ["open", "partly held", "held", "excluded"]


def test_a_missing_analysis_layer_is_skipped_with_a_note(toolbox, tmp_path):
    alg = finished(toolbox, tmp_path)
    gpkg = tmp_path / "base" / "demo" / "analysis" / "invalid.gpkg"  # the fake treats 'invalid' as unreadable
    gpkg.parent.mkdir(parents=True)
    gpkg.write_bytes(b"x")
    alg._to_load["analysis"] = str(gpkg)
    feedback = FakeFeedback()
    alg.postProcessAlgorithm(None, feedback)
    assert [layer.layer_name for layer in toolbox.fake_project.instance().layers] == ["Gossan prospectivity", "Gossan targets (top percentile)"]
    assert sum("not added" in text for text in feedback.info) == 2
