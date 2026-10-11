"""Tenure activity: what changed on the ground since the last look, and what to watch.

Mineral Titles Online publishes only *current* tenure, with no change history. Gossan reconstructs
history from its own record: every ``gossan harvest`` of ``mto_tenure`` saves the raw layer under
``raw/mto_tenure/`` (identical downloads are reused, not repeated; see ``gossan.core.rawstore``)
and logs when it looked. Those files are dated snapshots. This step compares consecutive snapshots,
and reads a watchlist from the latest one, which works from the first harvest.

**Changes between snapshots** (one row per tenure and kind of change; "after" is when the old state
was last seen, "by" when the new one was first seen, so the change happened in between):

* ``new``            the tenure was not listed before (newly staked or newly registered);
* ``gone``           it is no longer listed (lapsed, forfeited, converted or cancelled; the layer
                     does not say which);
* ``owner_changed``  the holders differ;
* ``transferred``    its ownership-transfer count rose;
* ``work_filed``     its statement-of-work count rose (assessment work was filed: an active holder);
* ``cash_in_lieu``   its cash-in-lieu count rose (the holder paid instead of working);
* ``extended``       its good-to date moved later;
* ``reduced`` / ``area_changed``  its area shrank (or its reduction count rose) / grew by more than
                     ``AREA_TOLERANCE``.

**Watchlist** (from the latest snapshot, as of ``today``; thresholds in ``TenureConfig``): ``expiring``
(good-to soon), ``recent_staking`` (issued within the last year), ``recently_updated`` (record
changed lately, from ``UPDATE_TIMESTAMP``) and ``churn`` (often transferred).

Caveats: a snapshot that lists no tenure at all while an earlier one did is treated as a failed
download and skipped, not as every claim lapsing. Tenures that move across the AOI edge can appear
as new or gone. The layer does not record *why* a tenure disappeared.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from gossan.analyze.common import (
    log_outputs,
    read_layer,
    write_json,
    write_layer,
    write_table,
)
from gossan.core.project import Project

DATASET = "mto_tenure"
STAMP = re.compile(r"_(\d{8}T\d{6}\d*)Z\.geojson$")
AREA_TOLERANCE = 0.01  # relative change in hectares that counts as a boundary change
CHANGE_TYPES = (
    "new", "gone", "owner_changed", "transferred", "work_filed", "cash_in_lieu", "extended", "reduced", "area_changed",
)  # fmt: skip
WATCH_FLAGS = ("expiring", "recent_staking", "recently_updated", "churn")
STALE_SNAPSHOT_DAYS = 30
EVENT_COLUMNS = [
    "tenure_number", "change", "detail", "owner", "claim_name", "tenure_type", "area_ha", "good_to",
    "after", "by", "snapshot_old", "snapshot_new",
]  # fmt: skip


@dataclass(frozen=True)
class Snapshot:
    """One state of the tenure layer, and when Gossan saw it."""

    file: Path
    first_seen: datetime
    last_seen: datetime


@dataclass
class TenureResult:
    snapshots: list[Snapshot]
    events: gpd.GeoDataFrame
    watch: gpd.GeoDataFrame
    summary: dict
    outputs: list[str] = field(default_factory=list)


# --- snapshots ---------------------------------------------------------------------------------


def _stamp(name: str) -> datetime | None:
    m = STAMP.search(name)
    if not m:
        return None
    text = m.group(1)
    return datetime.strptime(text[:15], "%Y%m%dT%H%M%S").replace(tzinfo=UTC)


def list_snapshots(project: Project) -> list[Snapshot]:
    """The distinct states of the raw tenure layer in the order Gossan saw them. A state seen again
    after a different one (A, B, A) is a new snapshot each time; consecutive sightings of the same
    file are one snapshot with a first and a last time."""
    folder = project.raw_dir / DATASET
    files = {p.name: p for p in sorted(folder.glob(f"{DATASET}_*.geojson"))} if folder.is_dir() else {}
    seen: dict[str, set[datetime]] = {}
    for name, path in files.items():
        stamp = _stamp(name)  # files from before provenance recorded sightings still have their name
        seen[name] = {stamp} if stamp else {datetime.fromtimestamp(path.stat().st_mtime, UTC)}
    for r in project.provenance.read():
        if r.dataset == DATASET and r.operation == "fetch":
            name = Path(str(r.parameters.get("file", ""))).name
            if name in seen:
                seen[name].add(r.retrieved_at)
    observations = sorted((t, name) for name, times in seen.items() for t in times)
    snapshots: list[Snapshot] = []
    for t, name in observations:
        if snapshots and snapshots[-1].file.name == name:
            snapshots[-1] = Snapshot(snapshots[-1].file, snapshots[-1].first_seen, t)
        else:
            snapshots.append(Snapshot(files[name], t, t))
    return snapshots


def _clean(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    for col in gdf.columns.drop(gdf.geometry.name):
        gdf[col] = gdf[col].map(lambda v: v.strip() if isinstance(v, str) else v)
    return gdf


def read_snapshot(path: Path, project: Project) -> gpd.GeoDataFrame | None:
    """A raw snapshot as per-tenure rows in the project CRS, restricted to the AOI (None if empty)."""
    import json

    if not json.loads(path.read_text(encoding="utf-8"))["features"]:
        return None
    gdf = _clean(gpd.read_file(path, engine="pyogrio")).set_crs("EPSG:4326", allow_override=True).to_crs(project.config.crs)
    gdf = gdf[gdf.intersects(project.read_aoi().union_all())]
    return per_tenure(gdf) if len(gdf) else None


def _count(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").fillna(0) if col in df else pd.Series(0.0, index=df.index)


def _dates(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_datetime(df[col], errors="coerce", utc=True) if col in df else pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")


def per_tenure(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """One row per tenure, with the fields change detection compares. A tenure with several holders
    has one row per holder in the layer; they are merged (holders sorted and joined)."""
    gdf = gdf.dropna(subset=["TENURE_NUMBER_ID"]).copy()
    gdf["number"] = pd.to_numeric(gdf["TENURE_NUMBER_ID"]).astype("int64")
    owner = gdf["OWNER_NAME"] if "OWNER_NAME" in gdf else pd.Series("", index=gdf.index)
    frame = pd.DataFrame(
        {
            "number": gdf["number"],
            "claim_name": gdf.get("CLAIM_NAME"),
            "tenure_type": gdf.get("TENURE_TYPE_DESCRIPTION"),
            "owner": owner.fillna(""),
            "issue": _dates(gdf, "ISSUE_DATE"),
            "good_to": _dates(gdf, "GOOD_TO_DATE"),
            "updated": _dates(gdf, "UPDATE_TIMESTAMP"),
            "area_ha": pd.to_numeric(gdf["AREA_IN_HECTARES"], errors="coerce") if "AREA_IN_HECTARES" in gdf else np.nan,
            "n_transfer": _count(gdf, "OWNERSHIP_TRANSFER_EVENT_COUNT"),
            "n_work": _count(gdf, "STATEMENT_OF_WORK_EVENT_COUNT"),
            "n_cash": _count(gdf, "CASH_IN_LIEU_EVENT_COUNT"),
            "n_reduction": _count(gdf, "REDUCTION_EVENT_COUNT"),
            "geometry": gdf.geometry,
        },
        index=gdf.index,
    )
    grouped = frame.groupby("number", sort=True)
    agg = grouped.agg(
        claim_name=("claim_name", "first"), tenure_type=("tenure_type", "first"),
        owner=("owner", lambda s: "; ".join(sorted({v for v in s if v}))),
        issue=("issue", "first"), good_to=("good_to", "max"), updated=("updated", "max"), area_ha=("area_ha", "first"),
        n_transfer=("n_transfer", "max"), n_work=("n_work", "max"), n_cash=("n_cash", "max"), n_reduction=("n_reduction", "max"),
        geometry=("geometry", "first"),
    )  # fmt: skip
    return gpd.GeoDataFrame(agg, geometry="geometry", crs=gdf.crs)


# --- comparing two snapshots --------------------------------------------------------------------


def _event(row: pd.Series, number: int, change: str, detail: str, old: Snapshot, new: Snapshot) -> dict:
    return {
        "tenure_number": number, "change": change, "detail": detail, "owner": row["owner"] or None,
        "claim_name": row["claim_name"], "tenure_type": row["tenure_type"],
        "area_ha": None if pd.isna(row["area_ha"]) else round(float(row["area_ha"]), 1),
        "good_to": None if pd.isna(row["good_to"]) else row["good_to"].date().isoformat(),
        "after": old.last_seen.date().isoformat(), "by": new.first_seen.date().isoformat(),
        "snapshot_old": old.file.name, "snapshot_new": new.file.name,
        "geometry": row["geometry"],
    }  # fmt: skip


def diff_snapshots(old_df: gpd.GeoDataFrame, new_df: gpd.GeoDataFrame, old: Snapshot, new: Snapshot) -> list[dict]:
    """Change rows between two per-tenure frames (see the module docstring for the kinds)."""
    events: list[dict] = []
    for number in sorted(set(new_df.index) - set(old_df.index)):
        events.append(_event(new_df.loc[number], number, "new", f"issued {_day(new_df.loc[number, 'issue'])}", old, new))
    for number in sorted(set(old_df.index) - set(new_df.index)):
        events.append(_event(old_df.loc[number], number, "gone", f"was good to {_day(old_df.loc[number, 'good_to'])}", old, new))
    for number in sorted(set(old_df.index) & set(new_df.index)):
        a, b = old_df.loc[number], new_df.loc[number]
        if a["owner"] != b["owner"]:
            events.append(_event(b, number, "owner_changed", f"{a['owner'] or '?'} -> {b['owner'] or '?'}", old, new))
        for change, col, label in (("transferred", "n_transfer", "transfers"), ("work_filed", "n_work", "work filings"), ("cash_in_lieu", "n_cash", "cash-in-lieu payments")):
            if b[col] > a[col]:
                events.append(_event(b, number, change, f"{int(a[col])} -> {int(b[col])} {label}", old, new))
        if pd.notna(a["good_to"]) and pd.notna(b["good_to"]) and b["good_to"] - a["good_to"] >= timedelta(days=1):
            events.append(_event(b, number, "extended", f"good to {_day(a['good_to'])} -> {_day(b['good_to'])}", old, new))
        if pd.notna(a["area_ha"]) and pd.notna(b["area_ha"]) and a["area_ha"] > 0:
            change_ratio = (b["area_ha"] - a["area_ha"]) / a["area_ha"]
            if change_ratio < -AREA_TOLERANCE or b["n_reduction"] > a["n_reduction"]:
                events.append(_event(b, number, "reduced", f"{a['area_ha']:.1f} -> {b['area_ha']:.1f} ha", old, new))
            elif change_ratio > AREA_TOLERANCE:
                events.append(_event(b, number, "area_changed", f"{a['area_ha']:.1f} -> {b['area_ha']:.1f} ha", old, new))
    return events


def _day(ts: object) -> str:
    return "?" if ts is None or pd.isna(ts) else pd.Timestamp(ts).date().isoformat()


# --- the watchlist -----------------------------------------------------------------------------


def build_watch(tenure: gpd.GeoDataFrame, today: date, cfg) -> gpd.GeoDataFrame:
    """Tenures worth a look in the latest layer, one row each with the flags that apply."""
    t = per_tenure(tenure)
    now = pd.Timestamp(today, tz="UTC")
    flags = pd.DataFrame(
        {
            "expiring": (t["good_to"] >= now) & (t["good_to"] <= now + pd.Timedelta(days=cfg.expiring_days)),
            "recent_staking": t["issue"] >= now - pd.Timedelta(days=cfg.recent_days),
            "recently_updated": t["updated"] >= now - pd.Timedelta(days=cfg.update_days),
            "churn": t["n_transfer"] >= cfg.churn_transfers,
        },
        index=t.index,
    )
    keep = flags.any(axis=1)
    out = t[keep].copy()
    out["flags"] = flags[keep].apply(lambda r: "; ".join(f for f in WATCH_FLAGS if r[f]), axis=1)
    out["days_to_good_to"] = (out["good_to"] - now).dt.days
    out["issued"] = out["issue"].dt.date
    out["updated_on"] = out["updated"].dt.date
    out["good_to_date"] = out["good_to"].dt.date
    out["transfers"] = out["n_transfer"].astype(int)
    out = out.reset_index().rename(columns={"number": "tenure_number"})
    cols = ["tenure_number", "claim_name", "tenure_type", "owner", "area_ha", "flags", "good_to_date", "days_to_good_to", "issued", "updated_on", "transfers"]
    out["_priority"] = flags[keep].sum(axis=1).to_numpy()
    out = out.sort_values(["_priority", "days_to_good_to"], ascending=[False, True])
    return gpd.GeoDataFrame(out[[*cols, "geometry"]].reset_index(drop=True), geometry="geometry", crs=tenure.crs)


# --- the step ------------------------------------------------------------------------------------


def build_tenure_changes(project: Project, today: date | None = None) -> TenureResult:
    """Compare the saved tenure snapshots and build the watchlist; writes to ``analysis/``.

    Without a ``mto_tenure`` layer there is nothing to compare: the result says so, and only the
    short summary file recording that is written.
    """
    today = today or datetime.now(UTC).date()
    cfg = project.config.tenure
    tenure = read_layer(project, DATASET)
    empty_events = gpd.GeoDataFrame({c: [] for c in EVENT_COLUMNS}, geometry=[], crs=project.config.crs)
    if tenure is None:
        summary = {"available": False, "reason": "no mto_tenure layer (run 'gossan harvest --source mto_tenure')"}
        outputs = [write_json(project, summary, "tenure_summary")]  # so reports can say why, not "run analyze"
        log_outputs(project, "tenure", outputs, [], {"available": False})
        return TenureResult([], empty_events, gpd.GeoDataFrame(geometry=[], crs=project.config.crs), summary, outputs)

    snapshots = list_snapshots(project)
    notes: list[str] = []
    states: list[tuple[Snapshot, gpd.GeoDataFrame]] = []
    for snap in snapshots:
        frame = read_snapshot(snap.file, project)
        if frame is None:
            if states:  # an empty answer after a populated one is a failed download, not a clean sweep
                notes.append(f"snapshot of {snap.first_seen:%Y-%m-%d %H:%M} lists no tenure in the AOI; ignored")
            continue
        states.append((snap, frame))
    rows: list[dict] = []
    for (old, old_df), (new, new_df) in pairwise(states):
        rows += diff_snapshots(old_df, new_df, old, new)
    events = gpd.GeoDataFrame(rows, geometry="geometry", crs=project.config.crs) if rows else empty_events
    if rows:
        events = events[[*EVENT_COLUMNS, "geometry"]]

    watch = build_watch(tenure, today, cfg)
    latest_seen = snapshots[-1].last_seen if snapshots else None
    new_by_owner = (
        per_tenure(tenure).assign(_recent=lambda d: d["issue"] >= pd.Timestamp(today, tz="UTC") - pd.Timedelta(days=cfg.recent_days))
        .query("_recent")
        .groupby("owner")
        .agg(tenures=("area_ha", "size"), hectares=("area_ha", "sum"))
        .sort_values("hectares", ascending=False)
    )  # fmt: skip
    summary = {
        "available": True,
        "as_of": today.isoformat(),
        "n_tenures": int(tenure["TENURE_NUMBER_ID"].nunique()),
        "n_snapshots": len(states),
        "snapshots": [{"file": s.file.name, "first_seen": s.first_seen.isoformat(), "last_seen": s.last_seen.isoformat()} for s, _ in states],
        "latest_snapshot_age_days": None if latest_seen is None else max((today - latest_seen.date()).days, 0),
        "events_by_type": {k: int(v) for k, v in events["change"].value_counts().items()} if rows else {},
        "n_events": len(rows),
        "watch_counts": {f: int(watch["flags"].str.contains(f).sum()) for f in WATCH_FLAGS} if len(watch) else dict.fromkeys(WATCH_FLAGS, 0),
        "n_watch": len(watch),
        "recent_staking_by_owner": [
            {"owner": o or "(unknown)", "tenures": int(r.tenures), "hectares": round(float(r.hectares), 0)} for o, r in new_by_owner.head(8).iterrows()
        ],
        "notes": notes,
    }
    outputs = [
        write_table(project, events.drop(columns="geometry"), "tenure_changes"),
        write_table(project, watch.drop(columns="geometry"), "tenure_watch"),
        write_table(
            project,
            pd.DataFrame([{"file": s.file.name, "first_seen": s.first_seen.isoformat(), "last_seen": s.last_seen.isoformat(), "tenures_in_aoi": len(f)} for s, f in states]),
            "tenure_snapshots",
        ),
        write_json(project, summary, "tenure_summary"),
    ]
    if rows:
        outputs.append(write_layer(project, events, "tenure_changes"))
    if len(watch):
        outputs.append(write_layer(project, watch, "tenure_watch"))
    log_outputs(project, "tenure", outputs, [DATASET], {"snapshots": len(states), "events": len(rows), "watch": len(watch), "as_of": today.isoformat()})
    return TenureResult([s for s, _ in states], events, watch, summary, outputs)
