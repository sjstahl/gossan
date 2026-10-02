"""Fail-safe decision on whether deliverables carry the BC "Access Only" marking.

``core.licensing`` reads ``provenance.jsonl``. That ledger can be incomplete (a file copied in by
hand, a deleted or rewritten log), so this module also looks at the project itself: if anything
whose name contains ``aris`` (any case) is in the data directories, the project is treated as
holding ARIS data and gets the "Access Only" disclaimer and "INTERNAL DRAFT" watermark, whatever
the ledger says. A false positive costs an extra watermark; a false negative leaks Access Only data.

Searched: ``raw/``, ``rasters/``, ``digitize/`` (file and folder names, recursively) and the layer
names in ``data.gpkg``. Not searched: ``outputs/`` (made by Gossan) and ``provenance.jsonl``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pyogrio

from gossan.core.licensing import aris_evidence
from gossan.core.project import Project

DATA_DIRS = ("raw", "rasters", "digitize")
EVIDENCE_LIMIT = 6


@dataclass(frozen=True)
class Marking:
    marked: bool
    #: Short descriptions of what triggered the marking: provenance records first, then files.
    evidence: list[str] = field(default_factory=list)
    #: True when the files alone (not the ledger) triggered it.
    from_files_only: bool = False


def aris_files(project: Project, limit: int = EVIDENCE_LIMIT) -> list[str]:
    """Paths (relative to the project) and layer names matching ``*aris*``, case-insensitive."""
    found: list[str] = []
    for name in DATA_DIRS:
        base = project.root / name
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if "aris" in path.name.lower():
                found.append(f"file {path.relative_to(project.root).as_posix()}")
                if len(found) >= limit:
                    return found
    gpkg = project.data_gpkg
    if gpkg.is_file():
        try:
            layers = [str(n) for n in pyogrio.list_layers(gpkg)[:, 0]]
        except (OSError, pyogrio.errors.DataSourceError):  # an unreadable GeoPackage must not hide the files above
            layers = []
        found += [f"layer {n}" for n in layers if "aris" in n.lower()]
    return found[:limit]


def decide_marking(project: Project) -> Marking:
    """Marked if the provenance ledger OR the project's files show ARIS data."""
    from_ledger = aris_evidence(project.provenance.read(), limit=EVIDENCE_LIMIT)
    from_files = aris_files(project)
    evidence = (from_ledger + from_files)[:EVIDENCE_LIMIT]
    return Marking(marked=bool(evidence), evidence=evidence, from_files_only=bool(from_files) and not from_ledger)


def project_includes_aris(project: Project) -> bool:
    return decide_marking(project).marked


__all__ = ["Marking", "aris_files", "decide_marking", "project_includes_aris"]
