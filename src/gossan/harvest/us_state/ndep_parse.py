"""Read an NDEP Bureau of Mining Regulation and Reclamation (BMRR) notice PDF.

Two kinds are read: "Notice of Final Decision" (a Reclamation Permit has been decided) and "Notice of
Proposed Action" (an application is out for comment). Both name a reclamation permit, the permittee
and project, the county, the disturbance in acres, and the sections, township and range, in
fixed phrasing, for example:

    Reclamation Permit No. 0456 / Falcon Copper Corporation / Muncy Creek Drilling Project
    ... Reclamation Permit (0456) for an Exploration Project to Falcon Copper Corporation ...
    ... Sections 8, 9, and 16, Township 20 North, Range 66 East: Mount Diablo Baseline and Meridian,
    White Pine County, Nevada ... authorizes 11.6 acres of Phase I disturbance ...

Water Pollution Control permit notices and anything else are not reclamation notices and return None.
The parser reads text only; whoever calls it keeps the original PDF.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from gossan.harvest.us_state.plss import Township

PERMIT = re.compile(r"Reclamation\s+Permit\s+(?:No\.?\s*)?\(?(\d{4})\)?", re.IGNORECASE)
POSTED = re.compile(r"(?:Date\s+of\s+Posting|Web\s+Posting)\s*:\s*(\d{1,2}/\d{1,2}/\d{4})", re.IGNORECASE)
TRS = re.compile(
    r"Sections?\s+([\d,\sand&]+?)\s*,?\s*Township\s+(\d+)\s+(North|South)\s*,?\s*Range\s+(\d+)\s+(East|West)",
    re.IGNORECASE,
)
COUNTY = re.compile(r"([A-Z][A-Za-z]+(?:\s[A-Z][A-Za-z]+)?)\s+County,\s+Nevada")
KIND_WORDS = re.compile(r"for\s+(?:an?|the)\s+([A-Za-z\- ]{3,40}?)\s+to\s+[A-Z]")
DRILLING_WORDS = re.compile(r"explor|drill", re.IGNORECASE)


@dataclass
class NdepNotice:
    permit_no: str
    kind: str  # "decision" or "proposed"
    permittee: str
    project: str
    project_type: str  # "exploration" or "mine"
    county: str | None
    acres: float | None
    posted: date | None
    townships: list[Township] = field(default_factory=list)
    mentions_drilling: bool = False


def _acres(text: str) -> float | None:
    for pattern in (
        r"total\s+proposed\s+disturbed\s+acreage\s+is\s+([\d,]+(?:\.\d+)?)\s+acres",
        r"authorizes\s+([\d,]+(?:\.\d+)?)\s+acres",
        r"([\d,]+(?:\.\d+)?)\s+acres",
    ):
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return float(m.group(1).replace(",", ""))
    return None


def parse_notice(text: str) -> NdepNotice | None:
    """The notice in ``text``, or None if it is not a BMRR reclamation permit notice."""
    flat = " ".join(text.replace("­", "").split())
    upper = flat.upper()
    if "NOTICE OF FINAL DECISION" in upper:
        kind = "decision"
    elif "NOTICE OF PROPOSED ACTION" in upper or re.search(r"\bNOPA\b", upper):
        kind = "proposed"
    else:
        return None
    permit = PERMIT.search(flat)
    if not permit:
        return None

    # The permittee and project are the two lines under the permit heading.
    lines = [ln.strip() for ln in text.splitlines()]
    permittee = project = ""
    for i, ln in enumerate(lines):
        if PERMIT.search(ln) and len(ln) < 60:
            following = [x for x in lines[i + 1 : i + 6] if x]
            if len(following) >= 2:
                permittee, project = following[0], following[1]
            break
    if not permittee:  # fall back to the sentence "... to <Permittee> (Permittee)"
        m = re.search(r"to\s+(.{3,80}?)\s+\(Permittee\)", flat)
        permittee = m.group(1) if m else ""

    kind_phrase = KIND_WORDS.search(flat)
    phrase = kind_phrase.group(1) if kind_phrase else ""
    is_exploration = bool(DRILLING_WORDS.search(phrase)) or (not phrase and bool(DRILLING_WORDS.search(project)))

    posted_m = POSTED.search(flat)
    posted = datetime.strptime(posted_m.group(1), "%m/%d/%Y").replace(tzinfo=UTC).date() if posted_m else None
    county = COUNTY.search(flat)
    townships = [
        Township(
            township=int(m.group(2)),
            ns=m.group(3)[0].upper(),
            range=int(m.group(4)),
            ew=m.group(5)[0].upper(),
            sections=tuple(int(n) for n in re.findall(r"\d+", m.group(1))),
        )
        for m in TRS.finditer(flat)
    ]
    return NdepNotice(
        permit_no=permit.group(1),
        kind=kind,
        permittee=permittee,
        project=project,
        project_type="exploration" if is_exploration else "mine",
        county=county.group(1) if county else None,
        acres=_acres(flat),
        posted=posted,
        townships=townships,
        mentions_drilling=bool(re.search(r"\bdrill", flat, re.IGNORECASE)),
    )
