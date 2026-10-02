"""Which licence obligations a project carries, read from ``provenance.jsonl``.

The ARIS assessment-report index is published by BC as *Access Only*: reports, maps and tables made
from it are internal. Projects with no ARIS data (for example a Nevada project built from USGS and BLM
data) carry no such restriction, so deliverables for them must not be stamped as if they did. This
module decides, from the provenance log alone, whether ARIS data is involved.

A record counts as ARIS evidence when its dataset name is ``aris`` or starts with ``aris_`` (``aris_index``,
``aris_pdfs``, ``aris_text``, ``aris_maps``, ``aris_tables_*``), its source names ARIS, or its
licence says "Access Only". Outputs made by Gossan itself carry an accurate licence (see
``derived_license``), so they raise the flag only when ARIS data went into them.
"""

from __future__ import annotations

import re

from gossan.core.provenance import ProvenanceRecord

ARIS_SOURCE = re.compile(r"\bARIS\b")
ARIS_DATASET = re.compile(r"aris(_|$)")
ARIS_DERIVED_LICENSE = "Derived product including ARIS data (Access Only): internal use only"


def aris_evidence(records: list[ProvenanceRecord], limit: int = 6) -> list[str]:
    """Short descriptions of the records that list ARIS data (empty if none do)."""
    found: list[str] = []
    for r in records:
        reason = None
        if ARIS_DATASET.match(r.dataset.lower()):
            reason = f"dataset {r.dataset}"
        elif ARIS_SOURCE.search(r.source):
            reason = f"source {r.source}"
        elif "access only" in r.license.lower():
            reason = f"licence of {r.dataset} (Access Only)"
        if reason and reason not in found:
            found.append(reason)
    return found[:limit]


def uses_aris(records: list[ProvenanceRecord]) -> bool:
    return bool(aris_evidence(records, limit=1))


def derived_license(records: list[ProvenanceRecord], *, includes_aris: bool) -> str:
    """Licence text for a Gossan-made output.

    With ARIS data in it: the Access Only text. Without: a plain statement naming the licences of
    the harvested sources the project holds, so the record never claims a restriction that does not apply.
    """
    if includes_aris:
        return ARIS_DERIVED_LICENSE
    upstream = sorted(
        {r.license for r in records if r.operation in {"fetch", "manual_ingest"} and r.license and not r.license.lower().startswith("n/a")}
    )
    if not upstream:
        return "Derived product from the project's layers"
    return "Derived product from harvested layers (" + "; ".join(upstream) + ")"
