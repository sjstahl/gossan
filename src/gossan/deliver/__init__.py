"""Client deliverables: a self-contained package with a QGIS project and an interactive map (``gossan deliver``).

The package builder is imported on first use, not when ``gossan.deliver`` is imported: it needs the analysis
report, and the analysis report needs ``gossan.deliver.items`` (for the licence tests), so importing the builder
here eagerly made the two import each other and failed whenever the report modules were imported first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gossan.deliver.package import DeliverError, DeliverResult, build_deliverable

__all__ = ["DeliverError", "DeliverResult", "build_deliverable"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from gossan.deliver import package

        return getattr(package, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
