"""Put the model's targets and the tenure watch beside the lead score (BC).

The lead score answers "whose data are ripe for compilation": a new holder, a long paper history. The
prospectivity model answers "where is the ground worth looking at". A property that scores on both is a
better call to make than one that scores on either, so each lead also gets:

* ``target_overlap_pct`` and ``target_overlap_ha``: how much of the tenure lies inside the top-percentile
  target polygons of ``gossan target``; ``n_targets``, ``best_target`` and ``best_target_rank``: which
  targets touch it and the best ranked of them;
* ``prospectivity_p90`` and ``prospectivity_pctile``: the 90th percentile of the model's score over the
  tenure (its hottest tenth of ground, so a small hot spot in a large claim is not averaged away) and
  the share of the AOI's cells scoring at or below that;
* ``watch_flags`` and ``latest_change``: the tenure watchlist flags (expiring, recently staked,
  recently updated, often transferred) and the most recent change between saved tenure snapshots, from
  ``gossan analyze``: a change of holder is a better "new holder" signal than the original issue date;
* ``combined_score`` = ``(1 - target_weight) * score + target_weight * prospectivity_pctile`` and its
  ``combined_rank``. The lead ``score`` and ``rank`` are unchanged.

How much to trust the merit half: the model's skill is local and modest (see its effort-matched AUC in
``target/cv_report.json``, quoted in the run's provenance). Where that AUC is near 0.5 the percentile
is close to noise and ``target_weight`` should be lowered; ``run_leads`` says so. The columns are added only
when ``gossan target`` has been run; the others only when ``gossan analyze`` has.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask

from gossan.core.project import Project

TARGETS_PATH = ("target", "targets.gpkg")
TARGETS_LAYER = "targets_top"
PROSPECTIVITY = ("target", "prospectivity.tif")
CV_REPORT = ("target", "cv_report.json")
#: AUC of the model against equally explored ground below which its ranking of ground adds little.
WEAK_AUC = 0.6

TARGET_COLUMNS = [
    "target_overlap_pct", "target_overlap_ha", "n_targets", "best_target", "best_target_rank",
    "prospectivity_p90", "prospectivity_pctile", "combined_score", "combined_rank",
]  # fmt: skip
WATCH_COLUMNS = ["watch_flags", "latest_change"]


@dataclass
class Linked:
    leads: gpd.GeoDataFrame
    columns: list[str]
    notes: list[str]
    model_auc: float | None = None


def model_skill(project: Project) -> float | None:
    """The effort-matched AUC of the last model run (plain AUC if it had no effort layers), if any."""
    path = project.root.joinpath(*CV_REPORT)
    if not path.is_file():
        return None
    cv = json.loads(path.read_text(encoding="utf-8"))
    for key in ("auc_effort_matched_mean", "auc_mean"):
        if cv.get(key) is not None:
            return float(cv[key])
    return None


def _zonal_p90(ds: rasterio.DatasetReader, geom) -> float:
    try:
        data, _ = mask(ds, [geom], crop=True, filled=False, all_touched=True)
    except ValueError:  # the geometry does not reach the raster
        return float("nan")
    values = data[0].compressed()
    values = values[np.isfinite(values)]
    return float(np.percentile(values, 90)) if len(values) else float("nan")


def _prospectivity(project: Project, leads: gpd.GeoDataFrame) -> pd.DataFrame | None:
    path = project.root.joinpath(*PROSPECTIVITY)
    if not path.is_file():
        return None
    with rasterio.open(path) as ds:
        everything = ds.read(1, masked=True).compressed()
        everything = np.sort(everything[np.isfinite(everything)])
        if not len(everything):
            return None
        p90 = np.array([_zonal_p90(ds, g) for g in leads.to_crs(ds.crs).geometry])
    pctile = np.where(np.isnan(p90), np.nan, 100.0 * np.searchsorted(everything, p90, side="right") / len(everything))
    return pd.DataFrame({"prospectivity_p90": np.round(p90, 4), "prospectivity_pctile": np.round(pctile, 1)}, index=leads.index)


def _targets(project: Project, leads: gpd.GeoDataFrame) -> pd.DataFrame | None:
    path = project.root.joinpath(*TARGETS_PATH)
    if not path.is_file():
        return None
    try:
        targets = gpd.read_file(path, layer=TARGETS_LAYER, engine="pyogrio").to_crs(leads.crs)
    except Exception:  # noqa: BLE001 (a damaged or older file: the leads still stand without it)
        return None
    rows = []
    for lead in leads.geometry.values:
        hit = targets[targets.intersects(lead)]
        shares = hit.geometry.intersection(lead).area if len(hit) else pd.Series(dtype=float)
        covered = float(shares.sum())  # targets do not overlap each other, so this is the union area
        best = hit.sort_values("rank").iloc[0] if len(hit) else None
        rows.append(
            {
                "target_overlap_pct": round(100.0 * min(covered / lead.area, 1.0), 1) if lead.area else 0.0,
                "target_overlap_ha": round(covered / 1e4, 1),
                "n_targets": int((shares > 0).sum()),
                "best_target": "" if best is None else str(best["target_id"]),
                "best_target_rank": pd.NA if best is None else int(best["rank"]),
            }
        )
    out = pd.DataFrame(rows, index=leads.index)
    out["best_target_rank"] = out["best_target_rank"].astype("Int64")
    return out


def _watch(project: Project, leads: gpd.GeoDataFrame) -> pd.DataFrame | None:
    folder = project.root / "analysis"
    watch = folder / "tenure_watch.csv"
    changes = folder / "tenure_changes.csv"
    if not watch.is_file() and not changes.is_file():
        return None
    keys = leads["lead_id"].astype(str)
    out = pd.DataFrame({"watch_flags": "", "latest_change": ""}, index=leads.index)
    if watch.is_file():
        w = pd.read_csv(watch, dtype={"tenure_number": str}).dropna(subset=["flags"])
        out["watch_flags"] = keys.map(w.drop_duplicates("tenure_number").set_index("tenure_number")["flags"]).fillna("")
    if changes.is_file():
        c = pd.read_csv(changes, dtype={"tenure_number": str})
        if len(c):
            c = c.sort_values("by").drop_duplicates("tenure_number", keep="last")
            text = c.set_index("tenure_number").apply(lambda r: f"{r['change']} (by {str(r['by'])[:10]})", axis=1)
            out["latest_change"] = keys.map(text).fillna("")
    return out


def link_evidence(project: Project, leads: gpd.GeoDataFrame, target_weight: float) -> Linked:
    """``leads`` with the target and watch columns that the project's files allow, ranked again by the
    combined score when the model has run."""
    if leads.empty:
        return Linked(leads, [], [])
    notes: list[str] = []
    columns: list[str] = []
    out = leads.copy()
    overlap, prospectivity = _targets(project, out), _prospectivity(project, out)
    if overlap is None and prospectivity is None:
        notes.append("no target/targets.gpkg or prospectivity.tif: run 'gossan target' for target overlap and the combined score")
    if overlap is not None:
        for col in overlap:
            out[col] = overlap[col]
        columns += list(overlap)
    if prospectivity is not None:
        for col in prospectivity:
            out[col] = prospectivity[col]
        columns += list(prospectivity)
        merit = out["prospectivity_pctile"].fillna(0.0)
        out["combined_score"] = ((1.0 - target_weight) * out["score"] + target_weight * merit).round(2)
        order = out.sort_values(["combined_score", "score", "lead_id"], ascending=[False, False, True]).index
        out["combined_rank"] = pd.Series(range(1, len(out) + 1), index=order).reindex(out.index).astype(int)
        columns += ["combined_score", "combined_rank"]
    watch = _watch(project, out)
    if watch is None:
        notes.append("no tenure analysis: run 'gossan analyze --steps tenure' for watchlist flags and recent changes")
    else:
        for col in watch:
            out[col] = watch[col]
        columns += list(watch)
    auc = model_skill(project) if prospectivity is not None else None
    if prospectivity is not None and (auc is None or auc < WEAK_AUC):
        shown = "unmeasured" if auc is None else f"{auc:.2f}"
        notes.append(f"the model's skill is weak or unmeasured (AUC {shown}); the prospectivity half of the combined score is a weak signal, consider lowering target_weight")
    return Linked(out, columns, notes, auc)
