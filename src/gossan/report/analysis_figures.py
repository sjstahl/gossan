"""Figures for the analysis report, drawn from the files ``gossan analyze`` and ``gossan target`` wrote.

Each function returns the PNG path, or None when its inputs are missing (the report then says which
command produces them). Colours follow the dataviz palette (three categorical slots, validated for
all-pairs separation: blue, orange, aqua; one blue sequential ramp; neutral greys for context), so
meaning never rests on hue alone: every mark class is also named in a legend, and the report
repeats the numbers in tables. Maps share ``report.assets``'s page furniture.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import pyogrio
from matplotlib.colors import LinearSegmentedColormap

from gossan.core.project import Project
from gossan.report.assets import AOI_COLOUR, _finish, layer_or_none

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
TINT = "#cde2fb"  # the lightest blue step: the "highlighted ground" fill
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
GROUND_COLOURS = {"open": BLUE, "partly held": AQUA, "held": ORANGE, "excluded": INK2, "unknown": MUTED}
ANALYSIS_GPKG = "analysis/analysis.gpkg"


def analysis_layer(project: Project, name: str) -> gpd.GeoDataFrame | None:
    gdf = layer_or_none(project.root / ANALYSIS_GPKG, name)
    return None if gdf is None else gdf.to_crs(project.config.crs)


def data_layer(project: Project, name: str) -> gpd.GeoDataFrame | None:
    gdf = layer_or_none(project.data_gpkg, name)
    return None if gdf is None else gdf.to_crs(project.config.crs)


def read_csv(project: Project, name: str) -> pd.DataFrame | None:
    path = project.root / "analysis" / f"{name}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if len(df) else None


def _extent(ax, project: Project) -> None:
    minx, miny, maxx, maxy = project.read_aoi().total_bounds
    pad = 0.04 * max(maxx - minx, maxy - miny)
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)


def _legend(ax, handles, ncol: int = 3) -> None:
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=ncol, fontsize=7, frameon=False)


def _clean_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=7)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def enriched_units(project: Project, limit: int = 3) -> tuple[list[str], str]:
    """Bedrock units where occurrences are over-represented (>= 2 occurrences, ratio > 1.5,
    p < 0.05), best first, and which occurrence class they were found for."""
    table = read_csv(project, "occurrence_geology")
    if table is None:
        return [], "all"
    for subset in ("metallic", "all"):
        t = table[(table["level"] == "unit") & (table["occurrences"] == subset)]
        t = t[(t["n_occurrences"] >= 2) & (t["ratio"] > 1.5) & (t["p_value"] < 0.05)]
        if len(t):
            return list(t.sort_values("ratio", ascending=False)["group"].head(limit)), subset
    return [], "all"


def geology_map(project: Project, dest: Path) -> Path | None:
    """Bedrock units (those enriched in occurrences tinted), faults and occurrences by class."""
    from gossan.analyze.occurrences import unit_name

    bedrock = data_layer(project, "bedrock_geology")
    if bedrock is None:
        return None
    aoi = project.read_aoi()
    bedrock = bedrock.assign(unit=bedrock.apply(unit_name, axis=1)).clip(aoi.union_all())
    names, subset = enriched_units(project)
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    bedrock.plot(ax=ax, facecolor="#f4f3ef", edgecolor="#c3c2b7", linewidth=0.4)
    handles = []
    if names:
        bedrock[bedrock["unit"].isin(names)].plot(ax=ax, facecolor=TINT, edgecolor="#6da7ec", linewidth=0.6)
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=TINT, edgecolor="#6da7ec", label=f"Units rich in {subset} occurrences"))
    faults = data_layer(project, "faults")
    if faults is not None:
        faults = faults.clip(aoi.union_all())
        thrusts = faults[faults.get("FAULT_TYPE", pd.Series("", index=faults.index)).astype(str).str.contains("thrust", case=False)]
        other = faults.drop(thrusts.index)
        if len(other):
            other.plot(ax=ax, color=MUTED, linewidth=0.6)
            handles.append(plt.Line2D([], [], color=MUTED, linewidth=0.8, label="Fault"))
        if len(thrusts):
            thrusts.plot(ax=ax, color=INK, linewidth=1.1)
            handles.append(plt.Line2D([], [], color=INK, linewidth=1.2, label="Thrust fault"))
    occ = analysis_layer(project, "occurrences")
    if occ is None:
        occ = data_layer(project, "minfile")
    if occ is not None:
        cls = occ["commodity_class"] if "commodity_class" in occ else pd.Series("unclassified", index=occ.index)
        for label, colour, marker in (("metallic", ORANGE, "o"), ("industrial", AQUA, "s"), ("other", MUTED, "^")):
            sel = occ[cls == label] if label != "other" else occ[~cls.isin(["metallic", "industrial"])]
            if len(sel):
                sel.plot(ax=ax, color=colour, marker=marker, markersize=26, edgecolor="white", linewidth=0.6, zorder=5)
                handles.append(plt.Line2D([], [], marker=marker, linestyle="", markerfacecolor=colour, markeredgecolor="white", label=f"{label.title()} occurrence ({len(sel)})"))
    aoi.boundary.plot(ax=ax, color=AOI_COLOUR, linewidth=1.4)
    _extent(ax, project)
    _legend(ax, handles, ncol=3)
    return _finish(fig, ax, dest, project.config.crs)


def enrichment_chart(project: Project, dest: Path) -> Path | None:
    """Occurrences per bedrock unit relative to what the unit's area predicts (bars; 1 = by area)."""
    table = read_csv(project, "occurrence_geology")
    if table is None:
        return None
    subset = "metallic" if ((table["level"] == "unit") & (table["occurrences"] == "metallic") & (table["n_occurrences"] >= 2)).any() else "all"
    t = table[(table["level"] == "unit") & (table["occurrences"] == subset) & (table["n_occurrences"] >= 2)]
    t = t.sort_values("ratio", ascending=False).head(8).iloc[::-1]
    if t.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(t) + 1.3))
    y = np.arange(len(t))
    ax.barh(y, t["ratio"], height=0.6, color=BLUE)
    ax.axvline(1.0, color=INK2, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(1.0, len(t) - 0.35, "  expected by area", fontsize=7, color=INK2, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels([g if len(g) <= 46 else g[:43] + "..." for g in t["group"]], fontsize=7, color=INK)
    for yi, (n, ratio, p) in enumerate(zip(t["n_occurrences"], t["ratio"], t["p_value"], strict=True)):
        ax.text(ratio, yi, f"  {ratio:.1f}x  (n={n}{', p<0.05' if p < 0.05 else ''})", va="center", fontsize=7, color=INK)
    ax.set_xlim(0, float(t["ratio"].max()) * 1.35)
    ax.set_xlabel(f"Ratio of {subset} occurrences to the number expected from the unit's area", fontsize=7, color=INK2)
    _clean_axes(ax)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=130)
    plt.close(fig)
    return dest


def fault_chart(project: Project, dest: Path) -> Path | None:
    """Share of occurrences vs share of the area in each distance band from the nearest fault."""
    table = read_csv(project, "occurrence_faults")
    if table is None:
        return None
    t = table[(table["faults"] == "all") & (table["occurrences"] == "all")]
    if t.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    x = np.arange(len(t))
    w = 0.38
    ax.bar(x - w / 2, t["occurrence_share"] * 100, width=w, color=BLUE, label="Known occurrences")
    ax.bar(x + w / 2, t["area_share"] * 100, width=w, color="#c3c2b7", label="Area of interest")
    ax.set_xticks(x)
    ax.set_xticklabels(t["band"], fontsize=7)
    ax.set_ylabel("Share (%)", fontsize=7, color=INK2)
    ax.set_xlabel("Distance to nearest mapped fault", fontsize=7, color=INK2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=7)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=130)
    plt.close(fig)
    return dest


def catchment_map(project: Project, dest: Path) -> Path | None:
    """Stream-sediment catchments shaded by their highest pathfinder index, multivariate
    anomalies outlined, known occurrences on top. Falls back to sample points without a DEM."""
    polys = analysis_layer(project, "rgs_catchments")
    points = analysis_layer(project, "rgs_scores")
    frame = polys if polys is not None else points
    if frame is None:
        return None
    idx = [c for c in frame.columns if c.startswith("idx_")]
    if not idx:
        return None
    value = frame[idx].max(axis=1)
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    handles = []
    vmin, vmax = float(value.quantile(0.05)), float(value.max())
    if polys is not None:
        polys.plot(ax=ax, column=value, cmap=SEQ, vmin=vmin, vmax=vmax, edgecolor="white", linewidth=0.15)
        anomalous = polys[polys["multivariate_anomaly"].astype(bool)] if "multivariate_anomaly" in polys else polys.iloc[0:0]
        if len(anomalous):
            anomalous.boundary.plot(ax=ax, color=INK, linewidth=0.9)
            handles.append(plt.Line2D([], [], color=INK, linewidth=1.2, label=f"Unusual element combination ({len(anomalous)})"))
    else:
        points.plot(ax=ax, column=value, cmap=SEQ, vmin=vmin, vmax=vmax, markersize=18, edgecolor="white", linewidth=0.3)
    sm = plt.cm.ScalarMappable(cmap=SEQ, norm=plt.Normalize(vmin, vmax))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Highest pathfinder index (mean normal score)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    occ = data_layer(project, "minfile")
    if occ is not None:
        occ.plot(ax=ax, color=ORANGE, edgecolor="white", markersize=22, linewidth=0.6, zorder=5)
        handles.append(plt.Line2D([], [], marker="o", linestyle="", markerfacecolor=ORANGE, markeredgecolor="white", label=f"Known occurrence ({len(occ)})"))
    project.read_aoi().boundary.plot(ax=ax, color=AOI_COLOUR, linewidth=1.3)
    _extent(ax, project)
    if handles:
        _legend(ax, handles, ncol=2)
    ax.set_title("Stream-sediment catchments" if polys is not None else "Stream-sediment samples", fontsize=9)
    return _finish(fig, ax, dest, project.config.crs)


def ground_map(project: Project, dest: Path, label_top: int = 10) -> Path | None:
    """Targets at their centroids, sized by area and coloured by ground status, over tenure."""
    dec = analysis_layer(project, "target_decisions")
    if dec is None or "ground_status" not in dec:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    handles = []
    tenure = data_layer(project, "mto_tenure")
    if tenure is not None:
        tenure.plot(ax=ax, facecolor="#f1f0ec", edgecolor="#c3c2b7", linewidth=0.4)
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="#f1f0ec", edgecolor="#c3c2b7", label="Mineral tenure"))
    for name in ("parks_protected", "conservancies", "national_parks", "indian_reserves", "treaty_lands"):
        g = data_layer(project, name)
        if g is not None:
            g.plot(ax=ax, facecolor="none", edgecolor=INK2, hatch="////", linewidth=0.3)
    if any(data_layer(project, n) is not None for n in ("parks_protected", "conservancies", "national_parks", "indian_reserves", "treaty_lands")):
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=INK2, hatch="////", label="Parks, reserves, treaty land"))
    project.read_aoi().boundary.plot(ax=ax, color=AOI_COLOUR, linewidth=1.3)
    cent = dec.geometry.representative_point()
    size = 18 + 14 * np.sqrt(dec["area_ha"].clip(lower=1).to_numpy())
    for status, colour in GROUND_COLOURS.items():
        sel = (dec["ground_status"] == status).to_numpy()
        if sel.any():
            ax.scatter(cent.x[sel], cent.y[sel], s=np.minimum(size[sel], 240), color=colour, edgecolor="white", linewidth=0.6, zorder=5)
            handles.append(plt.Line2D([], [], marker="o", linestyle="", markerfacecolor=colour, markeredgecolor="white", label=f"Target, {status} ({int(sel.sum())})"))
    for r in dec.nsmallest(label_top, "rank").itertuples():
        pt = r.geometry.representative_point()
        ax.annotate(str(r.rank), (pt.x, pt.y), xytext=(4, 4), textcoords="offset points", fontsize=6.5, color=INK, zorder=6)
    _extent(ax, project)
    _legend(ax, handles, ncol=3)
    ax.set_title(f"Targets by ground status (top {label_top} numbered by rank)", fontsize=9)
    return _finish(fig, ax, dest, project.config.crs)


def table_exists(project: Project, layer: str) -> bool:
    path = project.root / ANALYSIS_GPKG
    return path.exists() and layer in {str(n) for n in pyogrio.list_layers(path)[:, 0]}
