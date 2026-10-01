"""Run qgis/gossan_toolbox.py inside a real (headless) QGIS, the way the Processing dialog does.

Used by tests/test_qgis_toolbox.py through QGIS's own Python (python-qgis.bat); not a pytest file.
Arguments: script path, AOI file, base directory, Gossan executable. Prints one JSON line.
"""

import importlib.util
import json
import os
import sys

from qgis.core import (
    QgsApplication,
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsProject,
    QgsRasterLayer,
    QgsVectorLayer,
)

QgsApplication.setPrefixPath(os.environ.get("QGIS_PREFIX_PATH", ""), True)
app = QgsApplication([], False)
app.initQgis()
sys.path.append(os.path.join(os.environ["QGIS_PREFIX_PATH"], "python", "plugins"))
from processing.core.Processing import Processing

Processing.initialize()

script, aoi, base, exe = sys.argv[1:5]
spec = importlib.util.spec_from_file_location("gossan_toolbox", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Feedback(QgsProcessingFeedback):
    def __init__(self):
        super().__init__()
        self.errors = []

    def reportError(self, error, fatalError=False):
        self.errors.append(error)


params = {"INPUT": aoi, "PROJECT_NAME": "qgis_demo", "BASE_DIR": base, "GOSSAN": exe, "RESOLUTION": 200.0, "REUSE": False}
alg = module.GossanTargetingAlgorithm().create()  # a fresh instance, as the dialog makes
context, feedback = QgsProcessingContext(), Feedback()
out = {"check": list(alg.checkParameterValues(params, context)), "layers": []}
assert alg.prepare(params, context, feedback)
out["results"] = alg.runPrepared(params, context, feedback)
out["post"] = alg.postProcess(context, feedback)
for node in QgsProject.instance().layerTreeRoot().findLayers():
    layer = node.layer()
    info = {"name": layer.name(), "type": type(layer).__name__, "valid": layer.isValid(), "crs": layer.crs().authid()}
    if isinstance(layer, QgsRasterLayer):
        shader = layer.renderer().shader().rasterShaderFunction()
        info.update(renderer=layer.renderer().type(), ramp=[round(i.value, 3) for i in shader.colorRampItemList()],
                    colours=[i.color.name() for i in shader.colorRampItemList()], bands=layer.bandCount())  # fmt: skip
    elif isinstance(layer, QgsVectorLayer):
        symbol = layer.renderer().symbol().symbolLayer(0)
        info.update(renderer=layer.renderer().type(), features=layer.featureCount(), outline=symbol.strokeColor().name())
    out["layers"].append(info)
out["errors"] = feedback.errors
print("RESULT:" + json.dumps(out, default=str))
app.exitQgis()
