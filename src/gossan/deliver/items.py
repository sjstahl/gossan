"""What can go in a client package, and what has to stay behind.

Every candidate layer, raster and report is matched to its licence in ``provenance.jsonl`` (an input by
the licence of its fetch or ingest; an output of Gossan by the licence it was written with, which already
carries "internal use only" when any input was Access Only). An item goes in only when its licence is on
record and does not restrict it. An item with no record is withheld, not guessed at: the ledger can be
incomplete, and handing a client data the Province did not release is the failure to avoid. The names of
ARIS products are withheld whatever the ledger says, as in ``gossan.report.licensing``.

Leads are Gossan's own business-development output about other people's ground, and are never packaged.
``include_internal`` lifts the licence test (never the leads rule) for a package kept inside the company;
the package then says so on every face.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pyogrio

from gossan.core.project import Project
from gossan.core.provenance import ProvenanceRecord

NEVER = ("leads",)
RESTRICTING = ("access only", "internal use only")
INPUT_OPERATIONS = {"fetch", "manual_ingest"}

#: data.gpkg layers that are context for the results, in drawing order (bottom first), with titles.
CONTEXT_LAYERS = {
    "bedrock_geology": "Bedrock geology",
    "mto_tenure": "Mineral tenure",
    "faults": "Faults",
    "minfile": "Known mineral occurrences (MINFILE)",
}  # fmt: skip
#: Layers that make the web map too heavy; they go in the QGIS package only.
NOT_ON_WEB_MAP = {"bedrock_geology"}
ASTER_RASTERS = ("aster_aloh", "aster_mgoh", "aster_ferric", "aster_gossan", "aster_ferrous")


@dataclass
class Item:
    key: str  # the name in the package
    kind: str  # "vector", "raster" or "report"
    title: str
    source: Path  # file (a layer of it for vectors: ``layer``)
    layer: str | None = None
    group: str = "results"  # "results" or "context"
    identifier: str = ""  # what provenance calls it (a dataset name, or a project-relative path)
    licence: str | None = None
    web: bool = True  # drawn on the web map
    withheld: str | None = None  # why it is not in the package

    @property
    def included(self) -> bool:
        return self.withheld is None


@dataclass
class Plan:
    items: list[Item] = field(default_factory=list)

    @property
    def included(self) -> list[Item]:
        return [i for i in self.items if i.included]

    @property
    def withheld(self) -> list[Item]:
        return [i for i in self.items if not i.included]


def restricts(licence: str | None, name: str = "") -> str | None:
    """Why ``licence`` (or the name) bars an item from a client package, or None if nothing does."""
    if "aris" in name.lower():
        return "ARIS product (Access Only)"
    if licence is None:
        return "no provenance record, so its licence cannot be confirmed"
    low = licence.lower()
    for words in RESTRICTING:
        if words in low:
            return f"licence restricts it ({licence})"
    return None


def _latest(records: list[ProvenanceRecord], match) -> ProvenanceRecord | None:
    found = [r for r in records if match(r)]
    return found[-1] if found else None


def input_licence(records: list[ProvenanceRecord], dataset: str) -> str | None:
    r = _latest(records, lambda r: r.dataset == dataset and r.operation in INPUT_OPERATIONS)
    return None if r is None else r.license


def output_licence(records: list[ProvenanceRecord], identifier: str) -> str | None:
    """Licence an output of Gossan was written with, by the path it was logged under."""
    r = _latest(records, lambda r: r.identifier == identifier or r.identifier.startswith(identifier + ":"))
    return None if r is None else r.license


def _layers(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    try:
        return {str(n) for n in pyogrio.list_layers(path)[:, 0]}
    except Exception:  # noqa: BLE001 (an unreadable file simply offers no layers)
        return set()


def candidates(project: Project) -> list[Item]:
    """Everything the package could hold, with its licence looked up and withheld items marked."""
    root = project.root
    records = project.provenance.read()
    items: list[Item] = []

    targets = root / "target" / "targets.gpkg"
    if "targets_top" in _layers(targets):
        items.append(Item("targets", "vector", "Prospectivity targets (random forest)", targets, "targets_top", identifier="target/targets.gpkg"))
    knowledge = root / "target" / "knowledge_targets.gpkg"
    if "knowledge_top" in _layers(knowledge):
        items.append(Item("knowledge_targets", "vector", "Targets from geological reasoning alone", knowledge, "knowledge_top", identifier="target/knowledge_targets.gpkg"))
    decisions = root / "analysis" / "analysis.gpkg"
    if "target_decisions" in _layers(decisions):
        items.append(Item("target_decisions", "vector", "Targets with ground status, access and history", decisions, "target_decisions", identifier="analysis/analysis.gpkg:target_decisions"))

    for name, title, file in (
        ("prospectivity", "Prospectivity score (random forest)", "target/prospectivity.tif"),
        ("knowledge", "Knowledge-driven score", "target/knowledge.tif"),
        ("uncertainty", "Spread between model runs", "target/uncertainty.tif"),
    ):
        if (root / file).is_file():
            items.append(Item(name, "raster", title, root / file, identifier=file))
    for name in ASTER_RASTERS:
        path = root / "rasters" / f"{name}.tif"
        if path.is_file():
            items.append(Item(name, "raster", f"ASTER index {name.removeprefix('aster_')}", path, identifier=f"rasters/{name}.tif", web=False))
    for path in sorted((root / "rasters").glob("*.tif")) if (root / "rasters").is_dir() else []:
        if path.stem in {"magnetics", "nrcan_magnetics"}:
            items.append(Item(path.stem, "raster", "Regional magnetic field", path, identifier=path.stem, web=False))

    present = _layers(project.data_gpkg)
    for name, title in CONTEXT_LAYERS.items():
        if name in present:
            items.append(Item(name, "vector", title, project.data_gpkg, name, group="context", identifier=name, web=name not in NOT_ON_WEB_MAP))

    for fmt, label in (("pdf", "PDF"), ("docx", "Word")):
        path = root / "outputs" / f"analysis_report.{fmt}"
        if path.is_file():
            items.append(Item(f"report_{fmt}", "report", f"Analysis report ({label})", path, identifier=f"outputs/{path.name}", web=False))

    for item in items:
        # Rasters and reports are logged under their path; context layers under their dataset name.
        item.licence = input_licence(records, item.identifier) if item.group == "context" else output_licence(records, item.identifier)
        if item.kind == "raster" and item.licence is None:
            item.licence = input_licence(records, item.identifier)  # an ingested grid is logged by its dataset name
        if item.kind == "report":
            marked = _latest(records, lambda r, i=item: r.identifier == i.identifier)
            if marked is not None and marked.parameters.get("access_only_marking"):
                item.licence = (item.licence or "") + " (marked Access Only)"
    return items


def plan(project: Project, *, include_internal: bool = False) -> Plan:
    """The candidates, each marked withheld or not."""
    items = candidates(project)
    for item in items:
        if item.key in NEVER:
            item.withheld = "Gossan's own business-development output; never packaged"
        elif not include_internal:
            item.withheld = restricts(item.licence, f"{item.key} {item.identifier}")
    return Plan(items)
