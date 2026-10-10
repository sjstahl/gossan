"""Drilling indicator for BC leads, from Notice of Work (NoW) applications that include drilling.

A tenure is flagged when a NoW with surface drilling (the ``drilling_notices`` layer, see
``gossan.harvest.bc_provincial.drilling_notices``) is on it. A notice is on a tenure when its
``TENURE_NUMBER`` list names the tenure, or its point lies within ``drilling_match_buffer_m`` of the
tenure polygon (a NoW point is a work location, which can sit just outside the claim it was filed under).

Each notice is classified as of the scoring date:

* ``active``: approved (or "No Permit Required"), permit not closed, and not past its authorisation end date;
* ``pending``: an application still in progress (received, referred, under review ...);
* ``lapsed``: approved but closed or past its authorisation end date;
* rejected and withdrawn applications are ignored.

The tenure's flag is its best notice (active over pending over lapsed). The flag is reported
separately from the score; the score only gets a small, configurable bonus (see ``LeadsConfig``).
This is a permit signal: it shows drilling is authorised or applied for, not that a rig is on site.
"""

from __future__ import annotations

import re
from datetime import date

import geopandas as gpd
import pandas as pd

FLAG_ORDER = ["active", "pending", "lapsed"]
NOT_ASSESSED = "not assessed"
NONE_FOUND = "none"
APPROVED = {"approved", "no permit required"}
IGNORED = {"rejected", "withdrawn"}
COLUMNS = [
    "drilling_flag", "drilling_active_count", "drilling_pending_count", "drilling_lapsed_count",
    "drilling_latest_filed", "drilling_operator", "drilling_notice_numbers",
]  # fmt: skip


def parse_bc_date(values: pd.Series) -> pd.Series:
    """Parse BC's date text, which ends in ``Z`` (for example ``2026-09-14Z``), as UTC timestamps."""
    text = values.astype("string").str.rstrip("Z")
    return pd.to_datetime(text, utc=True, errors="coerce", format="mixed")


def classify_notices(notices: gpd.GeoDataFrame, as_of: date) -> pd.Series:
    """``active``, ``pending``, ``lapsed`` or ``ignored`` for each notice, as of ``as_of``."""
    as_of_ts = pd.Timestamp(as_of, tz="UTC")
    status = notices["NOW_APPLICATION_STATUS_DESC"].fillna("").str.strip().str.lower()
    closed = notices["PERMIT_STATUS_CODE_DESC"].fillna("").str.strip().str.lower() == "closed"
    expired = parse_bc_date(notices["AMENDMENT_AUTH_END_DATE"]) < as_of_ts
    out = pd.Series("pending", index=notices.index)
    out[status.isin(APPROVED) & ~(closed | expired)] = "active"
    out[status.isin(APPROVED) & (closed | expired)] = "lapsed"
    out[status.isin(IGNORED)] = "ignored"
    return out


def _tenure_numbers(text: object) -> set[int]:
    return {int(n) for n in re.findall(r"\d+", text)} if isinstance(text, str) else set()


def drilling_summary(
    tenure: gpd.GeoDataFrame, notices: gpd.GeoDataFrame, buffer_m: float, as_of: date
) -> pd.DataFrame:
    """Per tenure (indexed by ``TENURE_NUMBER_ID``): the columns in ``COLUMNS``.

    ``tenure`` has one row per tenure; both frames are in the same projected CRS.
    """
    notices = notices.copy()
    notices["_status"] = classify_notices(notices, as_of)
    notices = notices[notices["_status"] != "ignored"]
    pairs: set[tuple[int, int]] = set()
    ids = {int(i) for i in tenure["TENURE_NUMBER_ID"]}
    for idx, text in notices["TENURE_NUMBER"].items():
        pairs |= {(t, idx) for t in _tenure_numbers(text) & ids}
    if len(notices) and len(tenure):
        reach = notices.geometry.buffer(buffer_m) if buffer_m > 0 else notices.geometry  # buffer(0) empties a point
        buffered = gpd.GeoDataFrame(geometry=reach, index=notices.index, crs=notices.crs)
        near = gpd.sjoin(tenure[["TENURE_NUMBER_ID", "geometry"]], buffered, how="inner", predicate="intersects")
        pairs |= {(int(t), int(i)) for t, i in zip(near["TENURE_NUMBER_ID"], near["index_right"], strict=True)}

    filed = parse_bc_date(notices["NOW_APPLICATION_SUBMITTED_DATE"])
    rows = {}
    for tenure_id in sorted({t for t, _ in pairs}):
        mine = notices.loc[sorted(i for t, i in pairs if t == tenure_id)]
        counts = mine["_status"].value_counts()
        flag = next(f for f in FLAG_ORDER if counts.get(f, 0))
        best = mine[mine["_status"] == flag]
        latest = filed[best.index].max()
        operator = best.assign(_filed=filed[best.index]).sort_values("_filed", ascending=False)
        names = operator["PERMITTEE_NAME"].where(operator["PERMITTEE_NAME"].notna(), operator["MINE_NAME"])
        rows[tenure_id] = {
            "drilling_flag": flag,
            "drilling_active_count": int(counts.get("active", 0)),
            "drilling_pending_count": int(counts.get("pending", 0)),
            "drilling_lapsed_count": int(counts.get("lapsed", 0)),
            "drilling_latest_filed": None if pd.isna(latest) else latest.strftime("%Y-%m-%d"),
            "drilling_operator": names.dropna().iloc[0] if names.notna().any() else "",
            "drilling_notice_numbers": "; ".join(sorted(mine["NOW_NUMBER"].dropna().astype(str))),
        }
    return pd.DataFrame.from_dict(rows, orient="index", columns=COLUMNS)
