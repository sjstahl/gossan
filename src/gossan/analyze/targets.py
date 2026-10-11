"""Target decision table: for each target from ``gossan target``, what stands for and against it.

Prospectivity says where the evidence looks like known occurrences. Whether a target is worth a
visit, a claim or a pitch depends on more: who holds the ground, whether it can be held at all,
how hard it is to reach, what has been done there, and what the geochemistry around it says. This
step gathers those facts per target polygon (``target/targets.gpkg``) from whatever the project
holds, and summarizes them in ``ground_status`` and ``notes``.

Columns (missing layers leave them empty, never zero):

* evidence: rank, mean score and spread (from ``gossan target``), dominant bedrock unit and its
  share, distance to the nearest fault, the best RGS pathfinder index and multivariate
  anomalies among samples within ``GEOCHEM_RADIUS_M``, known occurrences inside;
* ground: share under current mineral tenure (and holders, earliest expiry), share in parks,
  protected areas, conservancies, national parks, Indian reserves or treaty lands (``excluded``),
  in Mineral Titles reserves that restrict mineral tenure, and in granted crown-grant lots
  (private mineral title);
* access: distance to the nearest road (and its class) and transmission line;
* history: assessment reports, their latest year and reported spend, and historic and recent
  Notices of Work within ``HISTORY_RADIUS_M``.

``ground_status`` is "excluded" (more than half in excluded land), "held" (more than half under
tenure or crown grants), "partly held" (some tenure) or "open" (none recorded). Land status changes
and some designations are not harvested: confirm on Mineral Titles Online before acting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime

import geopandas as gpd
import numpy as np
import pandas as pd

from gossan.analyze.common import (
    AnalysisError,
    log_outputs,
    read_layer,
    write_json,
    write_layer,
    write_table,
)
from gossan.analyze.drill_results import classify_holes
from gossan.analyze.occurrences import unit_name
from gossan.core.project import Project

TARGETS_FILE = "target/targets.gpkg"
TARGETS_LAYER = "targets_top"
GEOCHEM_RADIUS_M = 2000.0
HISTORY_RADIUS_M = 1000.0
EXCLUDED_LAYERS = ("parks_protected", "conservancies", "national_parks", "indian_reserves", "treaty_lands")
MAJORITY = 0.5
#: A tenure whose earliest expiry is this close is flagged in the notes (ground may come open).
EXPIRY_SOON_DAYS = 180


@dataclass
class TargetDecisionResult:
    table: gpd.GeoDataFrame
    summary: dict
    outputs: list[str] = field(default_factory=list)


def _share(targets: gpd.GeoDataFrame, polys: gpd.GeoDataFrame | None) -> pd.Series | None:
    """Share of each target's area covered by ``polys`` (None without the layer)."""
    if polys is None:
        return None
    union = polys.geometry.union_all()
    return (targets.geometry.intersection(union).area / targets.geometry.area).round(3)


def _near(targets: gpd.GeoDataFrame, points: gpd.GeoDataFrame, radius: float) -> list[pd.DataFrame]:
    """For each target, the rows of ``points`` within ``radius`` of it."""
    zones = targets.geometry.buffer(radius)
    return [points[points.intersects(z)] for z in zones]


def _years(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def _drill_evidence(project: Project, t: gpd.GeoDataFrame) -> dict | None:
    """Per target: how many digitized drill holes lie on or beside it (placed by their own collar
    coordinates, within ``drill.near_m``; holes known only by their report's location within
    ``drill.report_radius_m``) and what they found. None if no drill tables were approved."""
    classified = classify_holes(project)
    if classified is None or classified.holes.empty:
        return None
    cfg = project.config.drill
    holes = classified.holes[classified.holes.geometry.notna()]
    exact = holes[holes["location_quality"] == "collar_coordinates"]
    report = holes[holes["location_quality"] == "report_location"]
    counts, texts, notes = [], [], []
    for geom in t.geometry:
        near = exact[exact.distance(geom) <= cfg.near_m]
        wide = report[report.distance(geom) <= cfg.report_radius_m]
        n_by = near["class"].value_counts()
        counts.append(len(near))
        parts = [f"{int(v)} {k.replace('_', ' ')}" for k, v in n_by.items()]
        texts.append(", ".join(parts) or None)
        note = None
        if len(near):
            mineral = near[near["class"] == "mineralized"]
            if len(mineral):
                best = mineral.sort_values("best_grade", ascending=False).iloc[0]
                note = f"drilled: {len(near)} hole(s), {len(mineral)} mineralized (best {best['best_element']} {best['best_grade']:g} {best['best_unit']} over {best['best_interval']}, ARIS {best['aris_number']})"
            elif (near["class"] == "barren").any():
                note = f"already drilled: {len(near)} hole(s), {int((near['class'] == 'barren').sum())} barren and none mineralized (a hole tests only its own line)"
            else:
                note = f"drilled: {len(near)} hole(s), inconclusive or without usable assays"
        elif len(wide):
            reports = ", ".join(sorted({str(int(n)) for n in wide["aris_number"].dropna()})[:3])
            note = f"ARIS report(s) {reports} describe drilling near it, but the holes could not be placed"
        notes.append(note)
    return {"n": counts, "text": texts, "notes": notes}


def _tenure_changes(project: Project) -> gpd.GeoDataFrame | None:
    """The ``tenure_changes`` layer from ``gossan analyze --steps tenure``, if it has any rows."""
    path = project.root / "analysis" / "analysis.gpkg"
    if not path.exists():
        return None
    try:
        gdf = gpd.read_file(path, layer="tenure_changes", engine="pyogrio")
    except Exception:  # noqa: BLE001 (no such layer: no tenure changes were found)
        return None
    return gdf.to_crs(project.config.crs) if len(gdf) else None


def build_decisions(project: Project) -> TargetDecisionResult:
    path = project.root / TARGETS_FILE
    if not path.exists():
        raise AnalysisError("no target/targets.gpkg (run 'gossan target' first)")
    targets = gpd.read_file(path, layer=TARGETS_LAYER, engine="pyogrio").to_crs(project.config.crs)
    if targets.empty:
        raise AnalysisError("target/targets.gpkg holds no targets")
    t = targets[["rank", "target_id", "area_ha", "mean_score", "max_score", "mean_spread", "n_known_occurrences", "geometry"]].copy()
    inputs: list[str] = []
    notes: list[list[str]] = [[] for _ in range(len(t))]

    # --- geology and structure ---------------------------------------------------------------
    bedrock = read_layer(project, "bedrock_geology")
    if bedrock is not None:
        inputs.append("bedrock_geology")
        bedrock = bedrock.assign(unit=bedrock.apply(unit_name, axis=1))
        units, shares = [], []
        for geom in t.geometry:
            pieces = bedrock.geometry.intersection(geom).area
            by_unit = pieces.groupby(bedrock["unit"]).sum()
            by_unit = by_unit[by_unit > 0]
            if by_unit.empty:
                units.append(None)
                shares.append(np.nan)
                continue
            units.append(by_unit.idxmax())
            shares.append(round(float(by_unit.max() / geom.area), 3))
        t["main_unit"], t["main_unit_share"] = units, shares
    faults = read_layer(project, "faults")
    if faults is not None:
        inputs.append("faults")
        union = faults.geometry.union_all()
        t["dist_fault_m"] = t.geometry.distance(union).round(0)

    # --- geochemistry ------------------------------------------------------------------------
    scores_path = project.root / "analysis" / "analysis.gpkg"
    scores = None
    if scores_path.exists():
        try:
            scores = gpd.read_file(scores_path, layer="rgs_scores", engine="pyogrio").to_crs(project.config.crs)
        except Exception:  # noqa: BLE001 (no geochem layer yet: leave the columns out)
            scores = None
    if scores is not None and len(scores):
        inputs.append("rgs")
        idx_cols = [c for c in scores if c.startswith("idx_")]
        best_idx, best_val, n_anom, best_elems = [], [], [], []
        for near in _near(t, scores, GEOCHEM_RADIUS_M):
            n_anom.append(int(near["multivariate_anomaly"].astype(bool).sum()) if len(near) else 0)
            if near.empty or not idx_cols:
                best_idx.append(None)
                best_val.append(np.nan)
                best_elems.append(None)
                continue
            stacked = near[idx_cols].max()
            col = stacked.idxmax()
            best_idx.append(col[4:])
            best_val.append(round(float(stacked.max()), 2))
            best_elems.append(near.loc[near[col].idxmax(), "top_elements"] or None)
        t["best_geochem_index"], t["best_geochem_value"] = best_idx, best_val
        t["n_geochem_anomalies"], t["best_sample_elements"] = n_anom, best_elems

    # --- ground: tenure, exclusions, reserves, crown grants ------------------------------------
    tenure = read_layer(project, "mto_tenure")
    if tenure is not None:
        inputs.append("mto_tenure")
        mineral = tenure
        if "TENURE_TYPE_DESCRIPTION" in tenure:
            mineral = tenure[tenure["TENURE_TYPE_DESCRIPTION"].astype(str).str.lower().eq("mineral")]
        t["tenure_share"] = _share(t, mineral) if len(mineral) else 0.0
        holders, expiry = [], []
        for geom in t.geometry:
            hit = mineral[mineral.intersects(geom)]
            names = sorted({str(n) for n in hit.get("OWNER_NAME", pd.Series(dtype=str)).dropna()})
            holders.append("; ".join(names[:3]) + (f" (+{len(names) - 3})" if len(names) > 3 else "") or None)
            dates = pd.to_datetime(hit.get("GOOD_TO_DATE", pd.Series(dtype=str)), errors="coerce").dropna()
            expiry.append(dates.min().date().isoformat() if len(dates) else None)
        t["tenure_holders"], t["earliest_good_to"] = holders, expiry
    excluded_layers = [(name, read_layer(project, name)) for name in EXCLUDED_LAYERS]
    present = [(n, g) for n, g in excluded_layers if g is not None]
    unions = {n: g.geometry.union_all() for n, g in present}
    if present:
        inputs += [n for n, _ in present]
        t["excluded_share"] = _share(t, gpd.GeoDataFrame(pd.concat([g[["geometry"]] for _, g in present]), crs=t.crs))
        for i, geom in enumerate(t.geometry):
            for name, g in present:
                hit = g[g.intersects(geom)]
                if len(hit):
                    label = next((c for c in ("PROTECTED_LANDS_NAME", "ENGLISH_NAME", "CLAB_ID") if c in hit), None)
                    what = ", ".join(sorted({str(v) for v in hit[label].dropna()})[:2]) if label else name
                    share = unions[name].intersection(geom).area / geom.area
                    notes[i].append(f"{share:.0%} in {name.replace('_', ' ')}: {what}")
    reserves = read_layer(project, "reserve_sites")
    if reserves is not None:
        inputs.append("reserve_sites")
        if "SITE_MINERAL_IND" in reserves:
            reserves = reserves[reserves["SITE_MINERAL_IND"].astype(str).str.upper().eq("Y")]
        t["mineral_reserve_share"] = _share(t, reserves) if len(reserves) else 0.0
    changes = _tenure_changes(project)
    if changes is not None:
        inputs.append("mto_tenure")
        summaries = []
        for i, geom in enumerate(t.geometry):
            hit = changes[changes.intersects(geom)]
            if hit.empty:
                summaries.append(None)
                continue
            counts = hit["change"].value_counts()
            summaries.append("; ".join(f"{k.replace('_', ' ')} {int(v)}" for k, v in counts.items()))
            notes[i].append(f"tenure activity on or beside it between snapshots ({summaries[-1]}; {hit['after'].min()} to {hit['by'].max()})")
        t["tenure_changes"] = summaries
    drilled = _drill_evidence(project, t)
    if drilled is not None:
        inputs.append("aris_pdfs")  # the drill tables are digitized ARIS reports
        t["n_drill_holes"], t["drill_result"] = drilled["n"], drilled["text"]
        for i, note in enumerate(drilled["notes"]):
            if note:
                notes[i].append(note)
    grants = read_layer(project, "crown_grants")
    if grants is not None:
        inputs.append("crown_grants")
        if "LOT_STATUS" in grants:
            status = grants["LOT_STATUS"].astype(str).str.upper()
            grants = grants[status.str.contains("GRANTED") & ~status.str.contains("NOT")]
        t["crown_grant_share"] = _share(t, grants) if len(grants) else 0.0

    # --- access ----------------------------------------------------------------------------------
    roads = read_layer(project, "roads")
    if roads is not None:
        inputs.append("roads")
        nearest = gpd.sjoin_nearest(t[["geometry"]], roads[[c for c in ("ROAD_CLASS",) if c in roads] + ["geometry"]], distance_col="d")
        nearest = nearest[~nearest.index.duplicated(keep="first")]
        t["dist_road_m"] = nearest["d"].round(0)
        if "ROAD_CLASS" in nearest:
            t["nearest_road_class"] = nearest["ROAD_CLASS"]
    power = read_layer(project, "transmission_lines")
    if power is not None:
        inputs.append("transmission_lines")
        t["dist_power_m"] = t.geometry.distance(power.geometry.union_all()).round(0)

    # --- history ---------------------------------------------------------------------------------
    aris = read_layer(project, "aris_index")
    if aris is not None:
        inputs.append("aris_index")
        n, latest, spend = [], [], []
        for near in _near(t, aris, HISTORY_RADIUS_M):
            n.append(len(near))
            years = _years(near.get("REPORT_YEAR", pd.Series(dtype=float))).dropna()
            latest.append(int(years.max()) if len(years) else None)
            dollars = pd.to_numeric(near.get("MERGE_SPENT_2002", pd.Series(dtype=float)), errors="coerce")
            spend.append(round(float(dollars.sum()), 0) if len(near) else 0.0)
        t["n_aris_reports"], t["latest_report_year"], t["reported_spend_2002_dollars"] = n, latest, spend
    for layer, col in (("historic_notices", "n_historic_notices"), ("drilling_notices", "n_drilling_notices")):
        g = read_layer(project, layer)
        if g is not None:
            inputs.append(layer)
            t[col] = [len(near) for near in _near(t, g, HISTORY_RADIUS_M)]
    minfile = read_layer(project, "minfile")
    if minfile is not None:
        inputs.append("minfile")
        names = []
        for geom in t.geometry:
            hit = minfile[minfile.within(geom.buffer(HISTORY_RADIUS_M))]
            label = hit.get("MINFILE_NAME1", pd.Series(dtype=str)).dropna().astype(str)
            names.append(", ".join(sorted(set(label))[:4]) or None)
        t["occurrences_within_1km"] = names

    # --- status and notes --------------------------------------------------------------------------
    def col(name: str) -> pd.Series:
        return t[name].fillna(0.0) if name in t else pd.Series(0.0, index=t.index)

    excluded, held = col("excluded_share"), col("tenure_share") + col("crown_grant_share")
    status = np.select(
        [excluded > MAJORITY, held > MAJORITY, held > 0],
        ["excluded", "held", "partly held"],
        default="open" if "tenure_share" in t else "unknown",
    )
    t["ground_status"] = status
    today = datetime.now(UTC).date()
    for i in range(len(t)):
        row = t.iloc[i]
        if col("mineral_reserve_share").iloc[i] > 0:
            notes[i].append(f"{row['mineral_reserve_share']:.0%} in a Mineral Titles reserve restricting mineral tenure")
        if col("crown_grant_share").iloc[i] > 0:
            notes[i].append(f"{row['crown_grant_share']:.0%} crown-granted (private mineral title)")
        if "earliest_good_to" in t and isinstance(row["earliest_good_to"], str):
            soon = (date.fromisoformat(row["earliest_good_to"]) - today).days <= EXPIRY_SOON_DAYS
            notes[i].append(
                f"tenure expires within {EXPIRY_SOON_DAYS // 30} months ({row['earliest_good_to']})"
                if soon
                else f"tenure good to {row['earliest_good_to']} at the earliest"
            )
        if "dist_road_m" in t and row["dist_road_m"] > 5000:
            notes[i].append(f"{row['dist_road_m'] / 1000:.1f} km from the nearest mapped road")
        if "n_aris_reports" in t and row["n_aris_reports"] == 0:
            notes[i].append("no assessment reports within 1 km")
    t["notes"] = ["; ".join(n) or None for n in notes]

    table = gpd.GeoDataFrame(t, geometry="geometry", crs=targets.crs)
    flat = table.drop(columns="geometry")
    summary = {
        "n_targets": len(table),
        "by_ground_status": table["ground_status"].value_counts().to_dict(),
        "open_targets": flat[flat["ground_status"] == "open"].head(10)[
            [c for c in ("rank", "target_id", "area_ha", "mean_score", "main_unit", "best_geochem_index", "dist_road_m") if c in flat]
        ].to_dict("records"),
        "layers_used": inputs,
    }
    outputs = [
        write_layer(project, table, "target_decisions"),
        write_table(project, flat, "target_decisions"),
        write_json(project, summary, "target_decisions_summary"),
    ]
    log_outputs(project, "targets", outputs, ["target", *inputs], {"geochem_radius_m": GEOCHEM_RADIUS_M, "history_radius_m": HISTORY_RADIUS_M})
    return TargetDecisionResult(table, summary, outputs)
