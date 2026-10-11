"""Known occurrences against geology and structure: where do deposits sit, compared with the AOI?

For each bedrock unit, geological period and rock class the AOI holds, it compares the share of
known occurrences inside with the share of AOI area (``ratio`` > 1: more occurrences than the
unit's size predicts). ``p_value`` is the Poisson chance of seeing at least that many occurrences
if they fell at random by area; with tens of occurrences, read it as "how surprising", not as a
test result. Fault proximity compares occurrence distances to the nearest fault (any type, and
thrusts) with the distances of a regular grid of points over the AOI.

Occurrences are grouped by the first letter of the BC deposit-profile code (``DEPOSIT_FAMILIES``)
and into a commodity class (metallic, industrial, specialty), because one statistic over gypsum
beds and copper veins together answers no useful question. Statistics are reported for all
occurrences and for each class with enough members.

Associations reflect where people have looked as well as where deposits are; the 1:250 000-scale
geology also smooths contacts. Treat the output as evidence for a geologist to weigh.

Also written: one row per occurrence with its unit, distances to faults, family, whether MINFILE
holds production or inventory records for it, and the nearest RGS sample's anomaly summary (when
the geochemistry step has run).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, poisson
from shapely.geometry import MultiLineString

from gossan.analyze.common import (
    AnalysisError,
    log_outputs,
    read_layer,
    write_json,
    write_layer,
    write_table,
)
from gossan.core.project import Project

#: BC Mineral Deposit Profiles, by the first letter of DEPOSIT_TYPE_CODE.
DEPOSIT_FAMILIES = {
    "A": "Organic",
    "B": "Residual and surficial",
    "C": "Placer",
    "D": "Continental sediments and volcanics",
    "E": "Sediment-hosted",
    "F": "Chemical sediment",
    "G": "Volcanogenic",
    "H": "Epithermal",
    "I": "Vein, breccia and stockwork",
    "J": "Manto",
    "K": "Skarn",
    "L": "Porphyry",
    "M": "Ultramafic and mafic",
    "N": "Carbonatite",
    "O": "Pegmatite",
    "P": "Metamorphic-hosted",
    "Q": "Gems and semi-precious stones",
    "R": "Industrial rocks",
    "S": "Other",
}
CLASSES = {
    "metallic": set("CEGHIJKLMP"),
    "industrial": set("AFQR"),
    "specialty": set("NO"),  # carbonatite and pegmatite: REE, Nb, Ta, Li
}
MIN_CLASS_SIZE = 5
DISTANCE_BANDS_M = [0, 500, 1000, 2000, 5000, np.inf]
BACKGROUND_SPACING_M = 250.0


@dataclass
class OccurrenceResult:
    occurrences: gpd.GeoDataFrame
    geology: pd.DataFrame
    faults: pd.DataFrame
    summary: dict
    outputs: list[str] = field(default_factory=list)


def deposit_family(code: object) -> str:
    if not isinstance(code, str) or not code.strip():
        return "Unclassified"
    return DEPOSIT_FAMILIES.get(code.strip()[0].upper(), "Unclassified")


def commodity_class(code: object) -> str:
    if not isinstance(code, str) or not code.strip():
        return "unclassified"
    letter = code.strip()[0].upper()
    return next((name for name, letters in CLASSES.items() if letter in letters), "unclassified")


def unit_name(row: pd.Series) -> str:
    """A readable unit label: stratigraphic name and period, or rock class when unnamed."""
    name = str(row.get("STRATIGRAPHIC_NAME") or "").strip()
    period = str(row.get("GEOLOGICAL_PERIOD") or "").strip()
    if not name or name.lower() in {"unnamed", "none", "nan"}:
        rock = str(row.get("ROCK_CLASS") or "").strip()
        name = f"Unnamed {rock}" if rock and rock.lower() not in {"none", "nan"} else "Unnamed"
    if period and period.lower() not in {"none", "nan", "age unknown"}:
        name = f"{name} ({period})"
    return name


def background_points(aoi, spacing: float) -> np.ndarray:
    minx, miny, maxx, maxy = aoi.bounds
    xs = np.arange(minx + spacing / 2, maxx, spacing)
    ys = np.arange(miny + spacing / 2, maxy, spacing)
    gx, gy = np.meshgrid(xs, ys)
    pts = gpd.GeoSeries(gpd.points_from_xy(gx.ravel(), gy.ravel()))
    return pts[pts.within(aoi)]


def enrichment(groups: pd.Series, areas: pd.Series, total_area: float, n_total: int) -> pd.DataFrame:
    """Observed vs area-expected counts per group (``groups``: group label of each occurrence)."""
    counts = groups.value_counts()
    df = pd.DataFrame({"area_km2": areas / 1e6})
    df["area_share"] = areas / total_area
    df["n_occurrences"] = counts.reindex(df.index).fillna(0).astype(int)
    df["occurrence_share"] = df["n_occurrences"] / n_total if n_total else 0.0
    df["expected"] = df["area_share"] * n_total
    df["ratio"] = np.where(df["expected"] > 0, df["n_occurrences"] / df["expected"], np.nan)
    df["p_value"] = [poisson.sf(n - 1, e) if n > 0 else 1.0 for n, e in zip(df["n_occurrences"], df["expected"], strict=True)]
    return df.sort_values(["ratio", "n_occurrences"], ascending=False)


def _distance_table(occ_d: np.ndarray, bg_d: np.ndarray, label: str, subset: str) -> list[dict]:
    rows = []
    for lo, hi in pairwise(DISTANCE_BANDS_M):
        occ_share = float(((occ_d >= lo) & (occ_d < hi)).mean()) if len(occ_d) else np.nan
        area_share = float(((bg_d >= lo) & (bg_d < hi)).mean())
        rows.append(
            {
                "faults": label,
                "occurrences": subset,
                "band": f"{lo:g}-{hi:g} m" if np.isfinite(hi) else f">{lo:g} m",
                "n_occurrences": int(((occ_d >= lo) & (occ_d < hi)).sum()),
                "occurrence_share": round(occ_share, 3),
                "area_share": round(area_share, 3),
                "ratio": round(occ_share / area_share, 2) if area_share > 0 else np.nan,
            }
        )
    return rows


def analyze_occurrences(project: Project) -> OccurrenceResult:
    minfile = read_layer(project, "minfile", required=True)
    aoi = project.read_aoi().union_all()
    occ = minfile[minfile.within(aoi)].copy().reset_index(drop=True)
    if occ.empty:
        raise AnalysisError("no MINFILE occurrences inside the AOI")
    code = occ["DEPOSIT_TYPE_CODE1"] if "DEPOSIT_TYPE_CODE1" in occ else pd.Series(None, index=occ.index)
    occ["deposit_family"] = code.map(deposit_family)
    occ["commodity_class"] = code.map(commodity_class)
    subsets = {"all": occ.index}
    for cls in CLASSES:
        members = occ.index[occ["commodity_class"] == cls]
        if len(members) >= MIN_CLASS_SIZE:
            subsets[cls] = members
    inputs = ["minfile"]
    summary: dict = {
        "n_occurrences": len(occ),
        "by_family": occ["deposit_family"].value_counts().to_dict(),
        "by_class": occ["commodity_class"].value_counts().to_dict(),
        "classes_analysed": list(subsets),
    }

    # --- geology ---------------------------------------------------------------------------
    geology_table = pd.DataFrame()
    bedrock = read_layer(project, "bedrock_geology")
    if bedrock is not None:
        inputs.append("bedrock_geology")
        geo = bedrock.copy()
        geo["unit"] = geo.apply(unit_name, axis=1)
        geo["geometry"] = geo.geometry.intersection(aoi)
        geo = geo[~geo.geometry.is_empty]
        joined = gpd.sjoin(occ[["geometry"]], geo[["unit", "GEOLOGICAL_PERIOD", "ROCK_CLASS", "geometry"]], how="left", predicate="within")
        joined = joined[~joined.index.duplicated(keep="first")]
        occ["unit"] = joined["unit"]
        occ["geological_period"] = joined["GEOLOGICAL_PERIOD"]
        occ["rock_class"] = joined["ROCK_CLASS"]
        total = geo.area.sum()
        tables = []
        levels = (("unit", "unit", "unit"), ("period", "GEOLOGICAL_PERIOD", "geological_period"), ("rock_class", "ROCK_CLASS", "rock_class"))
        for level, col, occ_col in levels:
            areas = geo.assign(_a=geo.area).groupby(geo[col].fillna("Unknown"))["_a"].sum()
            for subset, idx in subsets.items():
                t = enrichment(occ.loc[idx, occ_col].fillna("Unknown"), areas, total, len(idx))
                t.insert(0, "occurrences", subset)
                t.insert(0, "level", level)
                tables.append(t.reset_index(names="group"))
        geology_table = pd.concat(tables, ignore_index=True)
        top = geology_table[(geology_table["level"] == "unit") & (geology_table["n_occurrences"] >= 2)]
        summary["enriched_units"] = {
            subset: top[top["occurrences"] == subset]
            .nlargest(5, "ratio")[["group", "n_occurrences", "area_share", "ratio", "p_value"]]
            .round(3)
            .to_dict("records")
            for subset in subsets
        }

    # --- faults ----------------------------------------------------------------------------
    fault_table = pd.DataFrame()
    faults = read_layer(project, "faults")
    if faults is not None:
        inputs.append("faults")
        bg = background_points(aoi, BACKGROUND_SPACING_M)
        rows: list[dict] = []
        summary["fault_proximity"] = {}
        kinds = {"all": faults}
        if "FAULT_TYPE" in faults:
            thrusts = faults[faults["FAULT_TYPE"].astype(str).str.contains("thrust", case=False)]
            if len(thrusts):
                kinds["thrust"] = thrusts
        for label, lines in kinds.items():
            union = MultiLineString([g for geom in lines.geometry for g in getattr(geom, "geoms", [geom])])
            occ[f"dist_{label}_fault_m"] = occ.distance(union).round(0)
            bg_d = bg.distance(union).to_numpy()
            for subset, idx in subsets.items():
                occ_d = occ.loc[idx, f"dist_{label}_fault_m"].to_numpy()
                rows += _distance_table(occ_d, bg_d, label, subset)
                ks = ks_2samp(occ_d, bg_d)
                summary["fault_proximity"][f"{label}/{subset}"] = {
                    "median_occurrence_m": float(np.median(occ_d)),
                    "median_area_m": float(np.median(bg_d)),
                    "ks_p_value": float(ks.pvalue),
                }
        fault_table = pd.DataFrame(rows)

    # --- production, inventory, nearby geochemistry -----------------------------------------
    for layer, col in (("minfile_production", "has_production"), ("minfile_inventory", "n_inventory_records")):
        records = read_layer(project, layer)
        if records is not None and "MINFILE_NUMBER" in records:
            inputs.append(layer)
            counts = records["MINFILE_NUMBER"].value_counts()
            n = occ["MINFILE_NUMBER"].map(counts).fillna(0).astype(int)
            occ[col] = n > 0 if col == "has_production" else n
    scores_path = project.root / "analysis" / "analysis.gpkg"
    try:
        scores = gpd.read_file(scores_path, layer="rgs_scores", engine="pyogrio") if scores_path.exists() else None
    except Exception:  # noqa: BLE001 (an older or partial analysis.gpkg only means no geochem columns)
        scores = None
    if scores is not None and len(scores):
        inputs.append("rgs")
        nearest = gpd.sjoin_nearest(occ[["geometry"]], scores, how="left", distance_col="nearest_rgs_m")
        nearest = nearest[~nearest.index.duplicated(keep="first")]
        occ["nearest_rgs_m"] = nearest["nearest_rgs_m"].round(0)
        for col in ("MASTER_ID", "multivariate_anomaly", "top_elements"):
            if col in nearest:
                occ[f"nearest_rgs_{col.lower()}"] = nearest[col]

    keep = [
        c
        for c in (
            "MINFILE_NUMBER", "MINFILE_NAME1", "STATUS_DESCRIPTION", "COMMODITY_DESCRIPTION1",
            "COMMODITY_DESCRIPTION2", "DEPOSIT_TYPE_CODE1", "DEPOSIT_TYPE_DESCRIPTION1",
            "deposit_family", "commodity_class", "unit", "geological_period", "rock_class",
            "dist_all_fault_m", "dist_thrust_fault_m", "has_production", "n_inventory_records",
            "nearest_rgs_m", "nearest_rgs_master_id", "nearest_rgs_multivariate_anomaly",
            "nearest_rgs_top_elements", "MINFILE_SUMMARY_URL",
        )
        if c in occ
    ]  # fmt: skip
    occ_out = gpd.GeoDataFrame(occ[keep], geometry=occ.geometry.values, crs=occ.crs)

    outputs = [write_layer(project, occ_out, "occurrences")]
    if len(geology_table):
        outputs.append(write_table(project, geology_table.round(4), "occurrence_geology"))
    if len(fault_table):
        outputs.append(write_table(project, fault_table, "occurrence_faults"))
    outputs.append(write_json(project, summary, "occurrence_summary"))
    log_outputs(project, "occurrences", outputs, inputs, {"classes": list(subsets), "bands_m": DISTANCE_BANDS_M[1:-1]})
    return OccurrenceResult(occ_out, geology_table, fault_table, summary, outputs)
