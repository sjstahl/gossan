"""Gossan for the QGIS Processing Toolbox.

Add this file with the Processing Toolbox's "Add Script to Toolbox..." button (see README.md).
It appears as *Gossan > Gossan: harvest and target*.

The algorithm takes an AOI polygon layer, a project name and a base folder, then runs
``gossan new``, ``gossan harvest`` and ``gossan target`` as separate processes and loads the
resulting ``prospectivity.tif`` and ``targets.gpkg`` (layer ``targets_top``) into the current
project. Gossan is never imported here: it runs from its own Python environment, so its
packages (GeoPandas, rasterio, scikit-learn and others) cannot clash with the ones QGIS
ships. For the same reason the child processes are started with QGIS's Python and GDAL
environment variables removed (see ``clean_environment``).

Written for QGIS 3.x. Layers are added to the project in ``postProcessAlgorithm``, which QGIS
runs on the main thread after the algorithm finishes.
"""

from __future__ import annotations

import contextlib
import os
import queue
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from qgis.core import (
    Qgis,
    QgsColorRampShader,
    QgsFillSymbol,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingOutputFolder,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFile,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
    QgsProject,
    QgsRasterBandStats,
    QgsRasterLayer,
    QgsRasterShader,
    QgsSingleBandPseudoColorRenderer,
    QgsSingleSymbolRenderer,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QColor

EXECUTABLE_ENV = "GOSSAN_EXECUTABLE"
#: Environment variables that point a child process at QGIS's own Python or GDAL/PROJ data.
QGIS_ENV_VARS = (
    "PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP",
    "QGIS_PREFIX_PATH", "QGIS_PLUGINPATH", "QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH",
    "GDAL_DATA", "GDAL_DRIVER_PATH", "GDAL_PLUGINS", "PROJ_LIB", "PROJ_DATA", "GEOTIFF_CSV",
    "OSGEO4W_ROOT", "O4W_QT_PREFIX", "O4W_QT_BINARIES", "O4W_QT_PLUGINS", "O4W_QT_LIBRARIES", "O4W_QT_TRANSLATIONS", "O4W_QT_HEADERS",
)  # fmt: skip
#: PATH entries containing any of these are dropped, so QGIS's GDAL DLLs are not found first.
QGIS_PATH_MARKERS = ("qgis", "osgeo4w")
#: Pseudocolour ramp for the prospectivity raster (viridis, low to high).
RAMP = ("#440154", "#3b528b", "#21918c", "#5ec962", "#fde725")
TARGETS_LAYER = "targets_top"
PROJECT_NAME_FORBIDDEN = '/\\:*?"<>|'


def tr(text: str) -> str:
    return QCoreApplication.translate("GossanToolbox", text)


def _scoped(cls, enum_name: str, member: str):
    """``cls.<enum_name>.<member>`` on newer QGIS, ``cls.<member>`` on older 3.x."""
    ns = getattr(cls, enum_name, None)
    return getattr(ns if ns is not None and hasattr(ns, member) else cls, member)


def _min_max_stats():
    """The Min | Max statistics flag: ``Qgis.RasterBandStatistic`` on QGIS 3.36+, else ``QgsRasterBandStats``."""
    new = getattr(Qgis, "RasterBandStatistic", None)
    if new is not None and hasattr(new, "Min"):
        return new.Min | new.Max
    return _scoped(QgsRasterBandStats, "Stats", "Min") | _scoped(QgsRasterBandStats, "Stats", "Max")


# --- helpers that do not need QGIS (so they can be tested without it) --------------------


def clean_environment(env: dict[str, str]) -> dict[str, str]:
    """A copy of ``env`` with QGIS's Python/GDAL/PROJ settings and PATH entries removed."""
    out = {k: v for k, v in env.items() if k.upper() not in QGIS_ENV_VARS}
    path_key = next((k for k in out if k.upper() == "PATH"), None)
    if path_key:
        keep = [p for p in out[path_key].split(os.pathsep) if not any(m in p.lower() for m in QGIS_PATH_MARKERS)]
        out[path_key] = os.pathsep.join(keep)
    out["PYTHONIOENCODING"] = "utf-8"
    out["PYTHONUTF8"] = "1"
    return out


def validate_project_name(name: str) -> str | None:
    """A reason the name cannot be a project folder, or None if it can."""
    name = name.strip()
    if not name:
        return "Enter a project name."
    if name in {".", ".."} or any(c in name for c in PROJECT_NAME_FORBIDDEN):
        return f"'{name}' cannot be used as a folder name (avoid {PROJECT_NAME_FORBIDDEN})."
    return None


def find_gossan(explicit: str | None) -> str | None:
    """The Gossan executable: the one given, else $GOSSAN_EXECUTABLE, else ``gossan`` on PATH."""
    for candidate in (explicit, os.environ.get(EXECUTABLE_ENV)):
        if candidate and Path(candidate).is_file():
            return str(candidate)
    return shutil.which("gossan")


def run_cli(args: list[str], cwd: str, feedback) -> tuple[int, list[str]]:
    """Run ``args`` in ``cwd``, sending each output line to ``feedback``.

    Returns (exit code, output lines). Honors the Cancel button by terminating the process.
    """
    feedback.pushCommandInfo(" ".join(f'"{a}"' if " " in a else a for a in args))
    process = subprocess.Popen(
        args, cwd=cwd, env=clean_environment(dict(os.environ)),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )  # fmt: skip
    lines: queue.Queue = queue.Queue()

    def pump() -> None:
        for line in process.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=pump, daemon=True).start()
    output: list[str] = []
    while True:
        if feedback.isCanceled():
            process.terminate()
            raise QgsProcessingException(tr("Cancelled; the running Gossan command was stopped."))
        try:
            item = lines.get(timeout=0.2)
        except queue.Empty:
            continue
        if item is None:
            break
        text = item.rstrip()
        output.append(text)
        feedback.pushInfo(text)
    process.stdout.close()
    return process.wait(), output


# --- the algorithm ------------------------------------------------------------------------


class GossanTargetingAlgorithm(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    PROJECT_NAME = "PROJECT_NAME"
    BASE_DIR = "BASE_DIR"
    GOSSAN = "GOSSAN"
    RESOLUTION = "RESOLUTION"
    REUSE = "REUSE"
    PROJECT_DIR = "PROJECT_DIR"

    def __init__(self) -> None:
        super().__init__()
        self._to_load: dict[str, str] = {}

    def createInstance(self):
        return GossanTargetingAlgorithm()

    def name(self) -> str:
        return "gossan_harvest_and_target"

    def displayName(self) -> str:
        return tr("Gossan: harvest and target")

    def group(self) -> str:
        return tr("Gossan")

    def groupId(self) -> str:
        return "gossan"

    def shortHelpString(self) -> str:
        return tr(
            "Runs Gossan on an area of interest: creates a project folder, downloads public BC data "
            "(MINFILE, tenure, ARIS index, RGS), builds a prospectivity model, and loads the "
            "prospectivity raster and the top-percentile target polygons into QGIS.\n\n"
            "Gossan runs as separate command-line processes in its own Python environment: point "
            "'Gossan executable' at gossan (or gossan.exe) from that environment, or set GOSSAN_EXECUTABLE. "
            "The area must hold at least 10 known MINFILE occurrences. Harvesting needs internet "
            "access and can take several minutes.\n\n"
            "The prospectivity values are a relative ranking, not a probability of a deposit. "
            "Outputs include ARIS data, which is Access Only: keep them internal."
        )

    def initAlgorithm(self, config=None) -> None:
        polygon = _scoped(QgsProcessing, "SourceType", "TypeVectorPolygon")
        self.addParameter(QgsProcessingParameterFeatureSource(self.INPUT, tr("Area of interest (polygon layer)"), [polygon]))
        self.addParameter(QgsProcessingParameterString(self.PROJECT_NAME, tr("Project name (becomes the folder name)"), defaultValue="gossan_project"))
        self.addParameter(
            QgsProcessingParameterFile(self.BASE_DIR, tr("Base directory (the project folder is created inside it)"),
                                       behavior=_scoped(QgsProcessingParameterFile, "Behavior", "Folder"))
        )  # fmt: skip
        self.addParameter(
            QgsProcessingParameterFile(self.GOSSAN, tr("Gossan executable (blank: $GOSSAN_EXECUTABLE, then PATH)"),
                                       behavior=_scoped(QgsProcessingParameterFile, "Behavior", "File"), optional=True)
        )  # fmt: skip
        self.addParameter(
            QgsProcessingParameterNumber(self.RESOLUTION, tr("Grid resolution (metres)"),
                                         type=_scoped(QgsProcessingParameterNumber, "Type", "Double"),
                                         defaultValue=100.0, minValue=1.0)
        )  # fmt: skip
        self.addParameter(QgsProcessingParameterBoolean(self.REUSE, tr("Reuse the project folder if it already exists (the AOI layer is then ignored)"), defaultValue=False))
        self.addOutput(QgsProcessingOutputFolder(self.PROJECT_DIR, tr("Gossan project folder")))

    def checkParameterValues(self, parameters, context):
        problem = validate_project_name(self.parameterAsString(parameters, self.PROJECT_NAME, context))
        if problem:
            return False, problem
        base = self.parameterAsString(parameters, self.BASE_DIR, context)
        if not base or not Path(base).is_dir():
            return False, tr("Choose an existing base directory.")
        name = self.parameterAsString(parameters, self.PROJECT_NAME, context).strip()
        if (Path(base) / name).exists() and not self.parameterAsBool(parameters, self.REUSE, context):
            return False, tr(f"'{Path(base) / name}' already exists. Pick another name, or tick 'Reuse the project folder'.")
        exe = self.parameterAsString(parameters, self.GOSSAN, context) or None
        if not find_gossan(exe):
            return False, tr("Gossan was not found. Set 'Gossan executable' to the gossan program in its Python environment, or set GOSSAN_EXECUTABLE.")
        return super().checkParameterValues(parameters, context)

    def processAlgorithm(self, parameters, context, feedback):
        import processing  # QGIS's processing module, available inside QGIS

        name = self.parameterAsString(parameters, self.PROJECT_NAME, context).strip()
        base = Path(self.parameterAsString(parameters, self.BASE_DIR, context))
        reuse = self.parameterAsBool(parameters, self.REUSE, context)
        resolution = self.parameterAsDouble(parameters, self.RESOLUTION, context)
        exe = find_gossan(self.parameterAsString(parameters, self.GOSSAN, context) or None)
        if not exe:
            raise QgsProcessingException(tr("Gossan was not found (set 'Gossan executable' or GOSSAN_EXECUTABLE)."))
        project_dir = base / name
        cwd = str(base)

        if reuse and (project_dir / "project.yaml").is_file():
            feedback.pushInfo(tr(f"Reusing the existing project {project_dir}; the AOI layer is ignored."))
        else:
            feedback.pushInfo(tr("Exporting the area of interest..."))
            with tempfile.TemporaryDirectory(prefix="gossan_aoi_") as tmp:
                aoi = str(Path(tmp) / f"{name}_aoi.gpkg")
                processing.run(
                    "native:savefeatures", {"INPUT": parameters[self.INPUT], "OUTPUT": aoi},
                    context=context, feedback=feedback, is_child_algorithm=True,
                )  # fmt: skip
                feedback.setProgress(5)
                self._run(exe, ["new", name, "--aoi", aoi], cwd, feedback, "gossan new")
        feedback.setProgress(15)

        # Harvest can fail for one source and still leave enough to model (the command exits 1 then),
        # so a failure here is reported and the run goes on; `gossan target` stops with a clear message
        # if what is missing matters.
        code = self._run(exe, ["harvest", name], cwd, feedback, "gossan harvest", fatal=False)
        if code != 0:
            feedback.reportError(tr("gossan harvest reported a problem (see the log above); continuing with what was downloaded."), False)
        feedback.setProgress(60)

        self._run(exe, ["target", name, "--resolution", f"{resolution:g}"], cwd, feedback, "gossan target")
        feedback.setProgress(95)

        target = project_dir / "target"
        self._to_load = {
            "prospectivity": str(target / "prospectivity.tif"),
            "targets": str(target / "targets.gpkg"),
        }
        return {self.PROJECT_DIR: str(project_dir)}

    def _run(self, exe: str, args: list[str], cwd: str, feedback, label: str, fatal: bool = True) -> int:
        feedback.pushInfo(f"--- {label} ---")
        code, output = run_cli([exe, *args], cwd, feedback)
        if code != 0 and fatal:
            tail = "\n".join(output[-12:]) or tr("(no output)")
            raise QgsProcessingException(tr(f"{label} failed (exit code {code}):\n{tail}"))
        return code

    def postProcessAlgorithm(self, context, feedback):
        """Add the results to the open project (runs on the main thread)."""
        project = QgsProject.instance()
        raster_path = self._to_load.get("prospectivity")
        if raster_path and Path(raster_path).is_file():
            raster = QgsRasterLayer(raster_path, "Gossan prospectivity")
            if raster.isValid():
                apply_pseudocolor(raster)
                project.addMapLayer(raster)
            else:
                feedback.reportError(tr(f"Could not open {raster_path}."), False)
        targets_path = self._to_load.get("targets")
        if targets_path and Path(targets_path).is_file():
            targets = QgsVectorLayer(f"{targets_path}|layername={TARGETS_LAYER}", "Gossan targets (top percentile)", "ogr")
            if targets.isValid():
                targets.setRenderer(outline_renderer())
                project.addMapLayer(targets)  # added after the raster, so drawn above it
            else:
                feedback.reportError(tr(f"Could not open {targets_path}."), False)
        else:
            feedback.pushInfo(tr("No target polygons were produced, so no targets layer was added."))
        return {}


# --- symbology ----------------------------------------------------------------------------


def apply_pseudocolor(layer: QgsRasterLayer) -> None:
    """Singleband pseudocolour on band 1: a viridis ramp over the layer's min to max."""
    provider = layer.dataProvider()
    lo, hi = 0.0, 1.0
    with contextlib.suppress(Exception):  # statistics can fail on odd data; keep the 0 to 1 score range
        stats = provider.bandStatistics(1, _min_max_stats(), layer.extent(), 0)
        if stats.maximumValue > stats.minimumValue:
            lo, hi = float(stats.minimumValue), float(stats.maximumValue)
    step = (hi - lo) / (len(RAMP) - 1)
    items = [QgsColorRampShader.ColorRampItem(lo + i * step, QColor(colour), f"{lo + i * step:.2f}") for i, colour in enumerate(RAMP)]
    ramp = QgsColorRampShader(lo, hi)
    ramp.setColorRampType(_scoped(QgsColorRampShader, "Type", "Interpolated"))
    ramp.setColorRampItemList(items)
    shader = QgsRasterShader()
    shader.setRasterShaderFunction(ramp)
    layer.setRenderer(QgsSingleBandPseudoColorRenderer(provider, 1, shader))
    layer.triggerRepaint()


def outline_renderer() -> QgsSingleSymbolRenderer:
    """Hollow polygons with a white outline, so targets read over the pseudocolour raster."""
    symbol = QgsFillSymbol.createSimple({"color": "0,0,0,0", "outline_color": "255,255,255,255", "outline_width": "0.7"})
    return QgsSingleSymbolRenderer(symbol)
