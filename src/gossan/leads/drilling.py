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

Notices on **major mines** (producing mines whose open permits still include some drilling disturbance,
such as New Afton) are flagged and labelled ``Major Mine`` but score nothing by default
(``drilling_include_major_mines``): they say little about a new exploration program.

The tenure's flag is its best notice (active over pending over lapsed). The flag is reported
separately from the score; the score only gets a small, configurable bonus (see ``LeadsConfig``).
Nevada projects use the same flag from NDEP reclamation permit notices for exploration projects (see
``gossan.harvest.us_state.ndep_notices``). Those notices have no end date, so a decision is ``active`` for
``drilling_window_months`` (18) after it was posted and a proposed action ``pending`` for as long, then
both are ``lapsed``; they are whole PLSS sections, so a notice counts for any claim it touches, and mine
permits are ignored.

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
    "drilling_latest_filed", "drilling_operator", "drilling_notice_numbers", "drilling_mine_scale",
    "drilling_scored_flag",
]  # fmt: skip


def parse_bc_date(values: pd.Series) -> pd.Series:
    """Parse BC's date text, which ends in ``Z`` (for example ``2026-09-14Z``), as UTC timestamps."""
    text = values.astype("string").str.rstrip("Z")
    return pd.to_datetime(text, utc=True, errors="coerce", format="mixed")


def classify_notices(notices: gpd.GeoDataFrame, as_of: date, window_months: int = 18) -> pd.Series:
    """``active``, ``pending``, ``lapsed`` or ``ignored`` for each notice, as of ``as_of``.

    Nevada notices (``NOTICE_SOURCE`` set) carry no authorisation end date, so a decision counts as
    active for ``window_months`` after it was posted and a proposed action as pending for as long;
    after that both are lapsed.
    """
    as_of_ts = pd.Timestamp(as_of, tz="UTC")
    status = notices["NOW_APPLICATION_STATUS_DESC"].fillna("").str.strip().str.lower()
    closed = notices["PERMIT_STATUS_CODE_DESC"].fillna("").str.strip().str.lower() == "closed"
    expired = parse_bc_date(notices["AMENDMENT_AUTH_END_DATE"]) < as_of_ts
    out = pd.Series("pending", index=notices.index)
    out[status.isin(APPROVED) & ~(closed | expired)] = "active"
    out[status.isin(APPROVED) & (closed | expired)] = "lapsed"
    out[status.isin(IGNORED)] = "ignored"
    if "NOTICE_SOURCE" in notices.columns:
        nv = notices["NOTICE_SOURCE"].notna()
        decided = notices["DECISION_DATE"].notna()
        posted = parse_bc_date(notices["DECISION_DATE"].where(decided, notices["NOW_APPLICATION_SUBMITTED_DATE"]))
        recent = (posted >= as_of_ts - pd.DateOffset(months=window_months)).fillna(False)
        out[nv & decided & recent] = "active"
        out[nv & ~decided & recent] = "pending"
        out[nv & ~recent] = "lapsed"
    return out


def is_major_mine(notices: gpd.GeoDataFrame) -> pd.Series:
    """BC marks Notices of Work on major (producing) mines; Nevada notices carry no such mark."""
    kind = notices["MINE_TYPE"].fillna("").str.lower() if "MINE_TYPE" in notices.columns else pd.Series("", index=notices.index)
    flagged = notices["MAJOR_MINE_IND"].fillna(0).astype(float) == 1 if "MAJOR_MINE_IND" in notices.columns else False
    return (kind == "major mine") | flagged


def _best_flag(statuses: pd.Series) -> str:
    return next((f for f in FLAG_ORDER if (statuses == f).any()), NONE_FOUND)


def _tenure_numbers(text: object) -> set[int]:
    return {int(n) for n in re.findall(r"\d+", text)} if isinstance(text, str) else set()


def drilling_summary(
    tenure: gpd.GeoDataFrame,
    notices: gpd.GeoDataFrame,
    buffer_m: float,
    as_of: date,
    window_months: int = 18,
    include_major_mines: bool = False,
) -> pd.DataFrame:
    """Per tenure (indexed by ``TENURE_NUMBER_ID``): the columns in ``COLUMNS``.

    ``tenure`` has one row per tenure; both frames are in the same projected CRS. Notices on major
    mines are shown (``drilling_flag``, ``drilling_mine_scale``) but score nothing unless
    ``include_major_mines``: ``drilling_scored_flag`` is the best status among the notices that count.
    """
    notices = notices.copy()
    notices["_status"] = classify_notices(notices, as_of, window_months)
    notices["_major"] = is_major_mine(notices)
    notices = notices[notices["_status"] != "ignored"]
    if "PROJECT_TYPE" in notices.columns:  # Nevada mine permits are context, not a drilling signal
        notices = notices[notices["PROJECT_TYPE"].isna() | (notices["PROJECT_TYPE"] == "exploration")]
    if "TENURE_NUMBER" not in notices.columns:
        notices["TENURE_NUMBER"] = None
    pairs: set[tuple[int, int]] = set()
    ids = {int(i) for i in tenure["TENURE_NUMBER_ID"]}
    for idx, text in notices["TENURE_NUMBER"].items():
        pairs |= {(t, idx) for t in _tenure_numbers(text) & ids}
    if len(notices) and len(tenure):
        is_point = notices.geom_type == "Point"  # only points are buffered; Nevada notices are section polygons
        reach = notices.geometry.where(~is_point | (buffer_m <= 0), notices.geometry.buffer(buffer_m))
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
            "drilling_mine_scale": "Major Mine" if mine["_major"].all() else ("Regional Mine" if not mine["_major"].any() else "Major and regional"),
            "drilling_scored_flag": _best_flag((mine if include_major_mines else mine[~mine["_major"]])["_status"]),
        }
    return pd.DataFrame.from_dict(rows, orient="index", columns=COLUMNS)
