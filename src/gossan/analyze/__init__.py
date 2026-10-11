"""Analysis of harvested data: inventory, geochemistry, occurrences against geology.

Each step reads ``data.gpkg`` (and ``rasters/``), writes to ``<project>/analysis/`` (tables as CSV,
spatial results as layers of ``analysis.gpkg``, a JSON summary) and logs its outputs to provenance
with a licence derived from its inputs. Steps run in ``STEPS`` order because later ones use earlier
results (occurrences pick up the nearest RGS sample's scores when geochem has run).
"""

from gossan.analyze.common import AnalysisError
from gossan.analyze.geochem import analyze_geochem
from gossan.analyze.inventory import build_inventory
from gossan.analyze.occurrences import analyze_occurrences

STEPS = {
    "inventory": build_inventory,
    "geochem": analyze_geochem,
    "occurrences": analyze_occurrences,
}

__all__ = ["STEPS", "AnalysisError", "analyze_geochem", "analyze_occurrences", "build_inventory"]
