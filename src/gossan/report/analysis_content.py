"""The analysis report as format-neutral blocks, built from what ``gossan analyze`` and
``gossan target`` wrote. ``analysis_render`` turns the same blocks into a PDF and a Word file, so
the two never disagree.

Nothing is recomputed here: the report quotes the project's own output files. A section whose
inputs are missing says which command produces them. Findings are phrased from the numbers with
thresholds named in this module, and every conclusion that rests on few samples says so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from gossan.core.licensing import access_only_datasets
from gossan.core.project import Project
from gossan.report.analysis_figures import (
    analysis_layer,
    catchment_map,
    enrichment_chart,
    fault_chart,
    geology_map,
    ground_map,
    read_csv,
)
from gossan.report.assets import finite_range, raster_map
from gossan.report.builder import data_sources
from gossan.report.licensing import decide_marking

#: Fewer positives than this and cross-validation and enrichment tests are shaky; the report says so.
FEW_OCCURRENCES = 30
#: Effort-matched AUC bands used to word the model's skill.
SKILL_BANDS = ((0.55, "no clear skill"), (0.65, "modest skill"), (0.75, "moderate skill"), (1.01, "good skill"))
MAX_TARGET_ROWS = 15
MAX_SAMPLE_ROWS = 10


@dataclass
class Heading:
    text: str
    level: int = 1


@dataclass
class Para:
    text: str
    small: bool = False


@dataclass
class Bullets:
    items: list[str]


@dataclass
class Table:
    rows: list[list[str]]  # first row is the header
    widths: list[float]  # relative column widths
    note: str | None = None


@dataclass
class Figure:
    path: Path
    caption: str
    width: float = 1.0  # share of the text width


@dataclass
class Note:
    text: str
    kind: str = "info"  # "info" or "warn"


@dataclass
class PageBreak:
    pass


Block = Heading | Para | Bullets | Table | Figure | Note | PageBreak


@dataclass
class AnalysisReport:
    title: str
    subtitle: str
    marked: bool
    restricted: list[str]
    blocks: list[Block] = field(default_factory=list)
    #: Datasets the report's numbers come from (for the provenance record's licence).
    inputs: list[str] = field(default_factory=list)
    sections: list[str] = field(default_factory=list)


# --- reading the outputs --------------------------------------------------------------------


def _json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _num(v: object, fmt: str = "{:,.0f}", empty: str = "") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
        return empty
    return fmt.format(v)


def _text(v: object) -> str:
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA else str(v)


def pval(p: float) -> str:
    """A p-value as printed: three decimals, or "<0.001"."""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def skill_word(auc: float) -> str:
    return next(word for limit, word in SKILL_BANDS if auc < limit)


def _need(command: str, what: str) -> list[Block]:
    return [Para(f"{what} is not available yet. Run `{command}` to produce it.")]


@dataclass
class Loaded:
    inventory: pd.DataFrame | None
    geochem: dict | None
    elements: pd.DataFrame | None
    scores: gpd.GeoDataFrame | None
    occ_summary: dict | None
    occ_geology: pd.DataFrame | None
    occ_faults: pd.DataFrame | None
    decisions: pd.DataFrame | None
    decisions_summary: dict | None
    target_grid: dict | None
    target_cv: dict | None
    importance: pd.DataFrame | None
    stale: list[str]


def load(project: Project) -> Loaded:
    a = project.root / "analysis"
    t = project.root / "target"
    cv = _json(t / "cv_report.json")
    importance = pd.read_csv(t / "feature_importance.csv") if (t / "feature_importance.csv").exists() else None
    decisions = read_csv(project, "target_decisions")
    stale = []
    if project.data_gpkg.exists():
        newest_data = project.data_gpkg.stat().st_mtime
        for name in ("inventory.csv", "geochem_summary.json", "occurrence_summary.json", "target_decisions.csv"):
            path = a / name
            if path.exists() and path.stat().st_mtime < newest_data:
                stale.append(name)
    if (t / "targets.gpkg").exists() and (a / "target_decisions.csv").exists() and (a / "target_decisions.csv").stat().st_mtime < (t / "targets.gpkg").stat().st_mtime:
        stale.append("target_decisions.csv (older than target/targets.gpkg)")
    return Loaded(
        read_csv(project, "inventory"),
        _json(a / "geochem_summary.json"),
        read_csv(project, "geochem_elements"),
        analysis_layer(project, "rgs_scores"),
        _json(a / "occurrence_summary.json"),
        read_csv(project, "occurrence_geology"),
        read_csv(project, "occurrence_faults"),
        decisions,
        _json(a / "target_decisions_summary.json"),
        _json(t / "grid.json"),
        cv if cv and cv.get("folds") else None,
        importance,
        stale,
    )


# --- sections -----------------------------------------------------------------------------------


def _findings(project: Project, L: Loaded) -> list[str]:
    out: list[str] = []
    aoi_km2 = project.read_aoi().union_all().area / 1e6
    if L.inventory is not None:
        vec = int((L.inventory["kind"] == "vector").sum())
        ras = int((L.inventory["kind"] == "raster").sum())
        out.append(f"{len(L.inventory)} datasets ({vec} vector layers, {ras} raster) cover the {aoi_km2:,.0f} km² area of interest.")
    s = L.occ_summary
    if s:
        by = s.get("by_class", {})
        out.append(
            f"{s['n_occurrences']} known MINFILE occurrences fall inside the area ("
            + ", ".join(f"{v} {k}" for k, v in by.items() if v) + ")."
        )
        enriched = (s.get("enriched_units") or {})
        subset = "metallic" if enriched.get("metallic") else "all"
        top = [u for u in enriched.get(subset, []) if u["p_value"] < 0.05]
        if top:
            u = top[0]
            out.append(
                f"{subset.title()} occurrences concentrate in {u['group']}: {u['n_occurrences']} occurrences on {u['area_share']:.1%} of the area "
                f"({u['ratio']:.1f} times the number expected by area, p {'= ' if u['p_value'] >= 0.001 else ''}{pval(u['p_value'])})."
            )
        else:
            out.append("No bedrock unit holds clearly more occurrences than its area predicts.")
        fp = (s.get("fault_proximity") or {}).get("all/all")
        if fp:
            word = "closer to" if fp["median_occurrence_m"] < fp["median_area_m"] else "no closer to"
            out.append(
                f"Occurrences lie {word} mapped faults than the area as a whole (median {fp['median_occurrence_m']:,.0f} m against "
                f"{fp['median_area_m']:,.0f} m; Kolmogorov-Smirnov p {'= ' if fp['ks_p_value'] >= 0.001 else ''}{pval(fp['ks_p_value'])})."
            )
    g = L.geochem
    if g:
        c = g.get("catchments")
        out.append(
            f"{g['n_samples']} stream-sediment samples were scored on {len(g['elements_continuous'])} continuous and "
            f"{len(g['elements_censored'])} mostly-below-detection elements; {g['n_multivariate_anomalies']} have an unusual overall element combination"
            + (f", each mapped to the {c['median_catchment_km2']:.1f} km² (median) catchment it drains." if isinstance(c, dict) else ".")
        )
    d = L.decisions_summary
    if d:
        status = d["by_ground_status"]
        out.append(f"The model's {d['n_targets']} targets: " + ", ".join(f"{v} {k}" for k, v in status.items()) + ".")
    cv = L.target_cv
    if cv and cv.get("auc_effort_matched_mean") is not None:
        eff = cv["auc_effort_matched_mean"]
        out.append(
            f"Spatial cross-validation: AUC {cv['auc_mean']:.2f}, but {eff:.2f} against equally explored ground, which is {skill_word(eff)} "
            "once exploration effort is accounted for."
        )
    elif cv:
        out.append(f"Spatial cross-validation: AUC {cv['auc_mean']:.2f} (effort-matched AUC not available).")
    return out


def _caveats(project: Project, L: Loaded) -> list[str]:
    out = [
        "Maps and scores are machine-generated aids. A geologist must review them before they inform any decision.",
        "Known occurrences are where people looked and found something. Statistics built on them reflect exploration history as well as geology.",
    ]
    s = L.occ_summary
    if s:
        n_metal = (s.get("by_class") or {}).get("metallic", 0)
        if 0 < n_metal < FEW_OCCURRENCES:
            out.append(f"Only {n_metal} metallic occurrences are available, so unit enrichment, fault statistics and cross-validation are uncertain; treat them as leads, not results.")
    if L.target_cv and L.target_cv.get("auc_effort_matched_mean") is not None and L.target_cv["auc_effort_matched_mean"] < SKILL_BANDS[1][0]:
        out.append("The model ranks known occurrences above equally explored ground only weakly: use its targets to prioritise a review, not to replace it.")
    if L.decisions is not None:
        out.append("Land status comes from the layers listed in the data sources and may be incomplete or out of date. Confirm tenure and designations on Mineral Titles Online before acting.")
    if L.stale:
        out.append("These analysis files are older than the data they describe; re-run `gossan analyze`: " + ", ".join(L.stale) + ".")
    return out


def _sec_summary(project: Project, L: Loaded) -> list[Block]:
    blocks: list[Block] = [Heading("Summary")]
    findings = _findings(project, L)
    if findings:
        blocks.append(Bullets(findings))
    else:
        blocks += _need("gossan analyze", "The analysis")
    blocks.append(Heading("Read this first", 2))
    blocks.append(Bullets(_caveats(project, L)))
    return blocks


def _sec_inventory(project: Project, L: Loaded) -> list[Block]:
    blocks: list[Block] = [Heading("Data compiled")]
    if L.inventory is None:
        return blocks + _need("gossan analyze --steps inventory", "The data inventory")
    blocks.append(Para("Everything harvested for the area, with how much of it each dataset covers and under which terms it may be used."))
    rows = [["Dataset", "Features", "Coverage", "Licence"]]
    for r in L.inventory.itertuples():
        lic = _text(r.license)
        short = "Access Only" if "access only" in lic.lower() else "OGL-BC" if "British Columbia" in lic else "OGL-Canada" if "Canada" in lic else lic
        rows.append([r.dataset, _num(r.n_features), _text(r.coverage), short])
    blocks.append(Table(rows, [2.2, 0.9, 1.9, 1.3], "Access Only: published by the Province without the right to reproduce or redistribute; reports using it are internal."))
    return blocks


def _sec_geology(project: Project, L: Loaded, assets: Path) -> list[Block]:
    blocks: list[Block] = [Heading("Geology and known occurrences")]
    s = L.occ_summary
    if s is None:
        return blocks + _need("gossan analyze --steps occurrences", "The occurrence analysis")
    blocks.append(
        Para(
            "Known occurrences are split into metallic and industrial-mineral groups, because a single statistic over gypsum beds and copper veins answers no useful question. "
            "For each bedrock unit the table compares the share of occurrences with the share of the area; a ratio above 1 means more occurrences than the unit's size predicts. "
            "The p-value is the Poisson chance of at least that many if occurrences fell at random by area."
        )
    )
    fig = geology_map(project, assets / "geology_map.png")
    if fig:
        blocks.append(Figure(fig, "Figure 1. Bedrock geology, faults and known occurrences. Tinted units hold more occurrences than their area predicts.", 0.95))
    if L.occ_geology is not None:
        for subset in ("metallic", "industrial"):
            t = L.occ_geology[(L.occ_geology["level"] == "unit") & (L.occ_geology["occurrences"] == subset) & (L.occ_geology["n_occurrences"] >= 1)]
            if t.empty:
                continue
            t = t.sort_values(["n_occurrences", "ratio"], ascending=False).head(8)
            rows = [["Bedrock unit", "Occurrences", "Share of area", "Times expected", "p-value"]]
            for r in t.itertuples():
                rows.append([r.group, str(r.n_occurrences), f"{r.area_share:.1%}", f"{r.ratio:.1f}", pval(r.p_value)])
            blocks.append(Heading(f"{subset.title()} occurrences by bedrock unit", 2))
            blocks.append(Table(rows, [3.2, 0.9, 0.9, 0.9, 0.7], f"Top {len(t)} units by number of occurrences. With few occurrences, a single extra one can change a ratio a lot."))
        chart = enrichment_chart(project, assets / "enrichment.png")
        if chart:
            blocks.append(Figure(chart, "Figure 2. Occurrences per bedrock unit relative to the number the unit's area predicts.", 0.95))
    fig = fault_chart(project, assets / "faults.png")
    if fig:
        fp = (s.get("fault_proximity") or {}).get("all/all", {})
        blocks.append(Heading("Faults", 2))
        blocks.append(Figure(fig, "Figure 3. Share of known occurrences and of the area in each distance band from the nearest mapped fault."
                             + (f" Median distance: {fp['median_occurrence_m']:,.0f} m for occurrences, {fp['median_area_m']:,.0f} m for the area." if fp else ""), 0.9))
    fam = s.get("by_family")
    if fam:
        blocks.append(Heading("Deposit types in MINFILE", 2))
        blocks.append(Table([["BC deposit-profile family", "Occurrences"], *[[k, str(v)] for k, v in fam.items()]], [3.6, 1.0]))
    return blocks


def _sec_geochem(project: Project, L: Loaded, assets: Path) -> list[Block]:
    blocks: list[Block] = [Heading("Stream-sediment geochemistry")]
    g = L.geochem
    if g is None or L.scores is None:
        return blocks + _need("gossan analyze --steps geochem", "The geochemistry analysis")
    c = g.get("catchments")
    blocks.append(
        Para(
            "Each Regional Geochemical Survey sample is scored element by element against the other samples (normal scores, so elements are comparable), "
            "corrected for the iron and manganese oxides that adsorb metals, and combined into pathfinder indices for common deposit models. "
            "A robust multivariate distance flags samples whose whole element combination is unusual even when no single element is extreme."
        )
    )
    if isinstance(c, dict):
        blocks.append(
            Para(
                f"A stream-sediment sample reflects everything upstream of it, so each sample was given its catchment by routing water over the {c['resolution_m']:.0f} m terrain model "
                f"(median catchment {c['median_catchment_km2']:.1f} km², samples moved a median {c['median_snap_m']:.0f} m onto the stream, {c['n_truncated']} catchments possibly cut off at the edge of the terrain data). "
                "Anomalies below are therefore anomalies of an upstream area."
            )
        )
    else:
        blocks.append(Note(f"Samples are treated as points: {c or 'no terrain model was harvested'}. Run `gossan harvest --source dem` to follow anomalies upstream.", "warn"))
    fig = catchment_map(project, assets / "catchments.png")
    if fig:
        blocks.append(Figure(fig, "Figure 4. Catchments shaded by their highest pathfinder index. Black outlines mark catchments with an unusual overall element combination. White ground drains past no sample.", 0.95))
    # top samples
    s = L.scores
    idx_cols = [col for col in s.columns if col.startswith("idx_")]
    if idx_cols:
        filled = s[idx_cols].fillna(-np.inf)  # a sample with no index at all sorts last, not an error
        s = s.assign(best_index=filled.max(axis=1), best_model=filled.idxmax(axis=1).str[4:].str.replace("_", " "))
        top = s[np.isfinite(s["best_index"])].sort_values("best_index", ascending=False).head(MAX_SAMPLE_ROWS)
        rows = [["Sample", "Best-fitting model", "Index", "Catchment km²", "Main rock unit", "Elevated elements (normal score)"]]
        for r in top.itertuples():
            rows.append([
                _text(getattr(r, "MASTER_ID", "")), r.best_model, f"{r.best_index:.2f}",
                _num(getattr(r, "catchment_area_km2", np.nan), "{:.1f}"),
                _text(getattr(r, "main_catchment_unit", "")), _text(getattr(r, "top_elements", "")),
            ])
        blocks.append(Heading("Samples to follow up", 2))
        blocks.append(Table(rows, [0.9, 1.1, 0.5, 0.7, 1.7, 2.2], f"Top {len(top)} by highest pathfinder index. An index of 2 means the model's elements average in about the top 2.5% of samples."))
    # elements
    if L.elements is not None:
        tier = L.elements["tier"].value_counts()
        cens = ", ".join(L.elements.loc[L.elements["tier"] == "censored", "element"])
        drop = ", ".join(L.elements.loc[L.elements["tier"] == "dropped", "element"])
        blocks.append(Heading("Elements used", 2))
        blocks.append(Bullets([
            f"{int(tier.get('continuous', 0))} continuous elements (analysed in most samples, mostly above detection) feed the multivariate analysis and the corrected scores.",
            f"{int(tier.get('censored', 0))} elements are mostly below detection ({cens}). A detection is the signal, so they are scored and used in the indices but kept out of the multivariate analysis.",
            f"{int(tier.get('dropped', 0))} elements were not usable ({drop or 'none'}): too few samples analysed, or nothing above the detection floor.",
        ]))
    r2 = g.get("catchment_r2_adjusted")
    if r2:
        top_r2 = sorted(r2.items(), key=lambda kv: -kv[1])[:6]
        blocks.append(Para(
            "How much of each element's variation follows iron, manganese, catchment size and the rocks drained: "
            + ", ".join(f"{el} {v:.0%}" for el, v in top_r2) + ". "
            "A high figure means that element mostly maps lithology rather than mineralization. Scores here are corrected for iron and manganese only; "
            "correcting for lithology and dilution is available but removes real signal where the host rocks themselves carry the deposits.", small=True))
    loads = g.get("pca_top_loadings")
    if loads:
        rows = [["Component", "Variance", "High (positive)", "High (negative)"]]
        for i, (pc, v) in enumerate(list(loads.items())[:4]):
            rows.append([pc, f"{g['pca_explained_variance'][i]:.0%}", ", ".join(v["positive"]), ", ".join(v["negative"])])
        blocks.append(Heading("Main patterns in the data", 2))
        blocks.append(Table(rows, [0.9, 0.8, 2.3, 2.3], "Principal components of the centred-log-ratio transformed concentrations. Elements at opposite ends vary against each other."))
    return blocks


def _sec_targets(project: Project, L: Loaded, assets: Path) -> list[Block]:
    blocks: list[Block] = [Heading("Targets")]
    grid, cv = L.target_grid, L.target_cv
    if grid is None:
        return blocks + _need("gossan target", "The prospectivity model")
    cls = grid.get("occurrence_class", "all")
    blocks.append(
        Para(
            f"A random forest trained by positive-unlabeled bagging ranks every {grid['resolution_m']:g} m cell by how much it resembles the {grid['n_known_occurrences_in_aoi']} known "
            f"{'' if cls == 'all' else cls + ' '}occurrences in the supplied datasets. Occurrences are the positives; all other cells are unlabeled, not barren. "
            "The score is a relative ranking, not a probability of a deposit."
        )
    )
    min_ha, smooth = grid.get("min_target_ha"), grid.get("smooth_m")
    if min_ha is not None:
        blocks.append(
            Para(
                "Targets are connected cells in the top percentile of the score"
                + (f" smoothed over {smooth:g} m" if smooth else "")
                + f". Polygons under {min_ha:g} ha are dropped ({grid.get('n_targets_dropped_small', 0)} here): the smallest unit "
                "anyone can stake is a BC cell claim of roughly 16 to 21 ha, so a single high-scoring cell is not a target by itself.",
                small=True,
            )
        )
    pm = raster_map(project, project.root / "target" / "prospectivity.tif", assets / "prospectivity.png", title="Prospectivity score", cmap="viridis",
                    colourbar_label="Relative score (0 to 1)", overlay_targets=True, overlay_occurrences=True)
    rng = finite_range(project.root / "target" / "prospectivity.tif")
    if pm:
        blocks.append(Figure(pm, "Figure 5. Prospectivity score with top-percentile targets outlined in white and known occurrences in red."
                             + (f" Score range {rng[0]:.2f} to {rng[1]:.2f}." if rng else ""), 0.95))
    if L.importance is not None:
        from gossan.report.pdf import (
            friendly_feature,  # shared names for the model's features
        )

        total = float(L.importance["importance"].sum()) or 1.0
        rows = [["Dataset", "Share of model importance"]]
        for r in L.importance.head(10).itertuples():
            rows.append([friendly_feature(r.feature), f"{100 * r.importance / total:.1f}%"])
        blocks.append(Heading("What drives the model", 2))
        blocks.append(Table(rows, [3.8, 1.6], "How much the forest relied on each dataset, not whether the relationship is geologically meaningful."))
    blocks.append(Heading("How well does it find known occurrences?", 2))
    if cv is None:
        blocks.append(Para("Cross-validation was not run or had too few occurrences."))
    else:
        eff = cv.get("auc_effort_matched_mean")
        rows = [["Measure", "Mean ± sd over folds", "Reads as"]]
        rows.append(["AUC", f"{cv['auc_mean']:.2f} ± {cv['auc_std']:.2f}", "chance a held-out occurrence outranks a held-out unlabeled cell (0.5 = chance)"])
        if eff is not None:
            rows.append(["Effort-matched AUC", f"{eff:.2f} ± {cv['auc_effort_matched_std']:.2f}", f"same, against equally explored ground: {skill_word(eff)}"])
        rows.append(["Top-5% capture", f"{cv['capture_top5_mean']:.0%} ± {cv['capture_top5_std']:.0%}", "share of held-out occurrences in the top 5% of cells (chance: 5%)"])
        if cv.get("capture_top5_smoothed_mean") is not None:
            rows.append(["Top-5% capture, smoothed", f"{cv['capture_top5_smoothed_mean']:.0%} ± {cv['capture_top5_smoothed_std']:.0%}", f"same, choosing cells on the score smoothed over {grid.get('smooth_m', 0):g} m"])
        gap = cv.get("buffer_m")
        gap_text = (
            f" A {gap / 1000:.1f} km gap separates each test block from the training cells so neighbours cannot leak."
            if gap
            else " No gap separates test blocks from training cells, so scores near block edges are optimistic."
        )
        blocks.append(Table(rows, [1.5, 1.4, 3.6], f"{cv['n_folds_scored']} spatial folds; whole blocks of ground are held out so the model is tested on places it has not seen.{gap_text}"))
        blocks.append(Para(
            "Occurrences sit where people explored, so a model can score well just by predicting exploration. The effort-matched figure weights each unlabeled cell by how heavily "
            "it was explored, and is the fairer test of whether the map finds deposits rather than camps.", small=True))
    return blocks


def _sec_decisions(project: Project, L: Loaded, assets: Path) -> list[Block]:
    blocks: list[Block] = [Heading("Target decision table")]
    d = L.decisions
    if d is None:
        return blocks + _need("gossan analyze --steps targets", "The target decision table")
    blocks.append(Para(
        "Prospectivity says where the evidence looks like known occurrences. Whether a target is worth a visit or a claim also depends on who holds the ground, "
        "whether it can be held at all, how hard it is to reach and what was done there. Ground status is: excluded (over half in parks, reserves or treaty land), held "
        "(over half under tenure or crown grant), partly held, or open (no tenure recorded)."))
    fig = ground_map(project, assets / "ground.png")
    if fig:
        blocks.append(Figure(fig, "Figure 6. Targets at their centroids, sized by area and coloured by ground status.", 0.95))
    rows = [["#", "Target", "ha", "Score", "Ground", "Host unit", "Geochemistry", "Road km", "Notes"]]
    for r in d.sort_values("rank").head(MAX_TARGET_ROWS).itertuples():
        geo = ""
        if _text(getattr(r, "best_geochem_index", "")):
            geo = f"{str(r.best_geochem_index).replace('_', ' ')} {r.best_geochem_value:.1f}"
        rows.append([
            str(r.rank), r.target_id, _num(r.area_ha), f"{r.mean_score:.2f}", _text(r.ground_status), _text(getattr(r, "main_unit", "")),
            geo, _num(getattr(r, "dist_road_m", np.nan) / 1000 if hasattr(r, "dist_road_m") else np.nan, "{:.1f}"), _text(getattr(r, "notes", "")),
        ])
    blocks.append(Table(rows, [0.3, 0.55, 0.5, 0.5, 0.7, 1.5, 1.1, 0.5, 2.0], f"Top {min(len(d), MAX_TARGET_ROWS)} of {len(d)} targets by rank. The full table is analysis/target_decisions.csv (and the target_decisions layer of analysis.gpkg, for QGIS)."))
    open_t = d[d["ground_status"] == "open"]
    text = f"{len(open_t)} of {len(d)} targets are on ground with no tenure recorded."
    if (d["area_ha"] < 20).any():
        text += " Polygons of a few hectares are single high-scoring cells; weigh larger, contiguous targets and those near known geochemical anomalies more heavily."
    blocks.append(Para(text, small=True))
    return blocks


def _sec_method(project: Project, L: Loaded) -> list[Block]:
    cfg = project.config
    t, g = cfg.target, cfg.geochem
    return [
        Heading("Method and limits"),
        Bullets([
            "Harvest: public BC data are downloaded from DataBC (WFS) and NRCan, kept unmodified in raw/, and logged with source, date and licence in provenance.jsonl.",
            (
                "Geochemistry: element selection by detection rate; normal scores; iron-manganese regression; pathfinder indices (mean corrected score of a deposit model's elements); "
                "compositional principal components; minimum-covariance-determinant distance for multivariate anomalies."
            ),
            f"Catchments: depression-filled D8 flow routing over the terrain model at {g.catchment_resolution_m:g} m; samples snapped to the nearest stream cell within {g.snap_m:g} m.",
            (
                f"Model: {t.n_bags} bags of {t.n_trees}-tree forests, {t.unlabeled_ratio:g} unlabeled cells per positive, {t.folds}-fold spatial block cross-validation. "
                f"Exploration effort mode: {t.effort}; positives: {t.occurrence_class} occurrences; seed {t.seed}."
            ),
            (
                "Model features describe the ground (faults, contacts, intrusions, unit age, terrain, geochemistry, host-unit favourability) rather than where exploration happened, "
                "so the map can point at ground nobody has tested."
            ),
        ]),
        Para("The tables are in analysis/ and target/ as CSV, GeoPackage and GeoTIFF for use in QGIS. Re-run `gossan harvest`, `gossan analyze` and `gossan target` to refresh them.", small=True),
    ]


def _sec_sources(project: Project, notices: list[str]) -> list[Block]:
    blocks: list[Block] = [Heading("Data sources and licences")]
    sources = data_sources(project.provenance.read())
    if not sources:
        blocks.append(Para("No external datasets have been retrieved for this project."))
    else:
        rows = [["Dataset", "Source", "Retrieved", "Licence"]]
        for r in sources:
            rows.append([r.dataset, r.source, f"{r.retrieved_at:%Y-%m-%d}", r.license])
        blocks.append(Table(rows, [1.3, 2.4, 0.8, 2.2]))
    if notices:
        blocks.append(Heading("Notices", 2))
        blocks.append(Bullets(notices))
    return blocks


#: Sections that start on a new page; the short ones (data, method, sources) follow on.
NEW_PAGE = {"geology", "geochemistry", "targets", "decisions", "method"}

NOTICES = {
    "Open Government Licence - British Columbia": "Contains information licensed under the Open Government Licence - British Columbia.",
    "Open Government Licence - Canada": "Contains information licensed under the Open Government Licence - Canada.",
}


def build_content(project: Project, assets: Path) -> AnalysisReport:
    """Assemble the report's blocks; figures are written under ``assets``."""
    L = load(project)
    records = project.provenance.read()
    marking = decide_marking(project)  # also looks at files, in case the ledger is incomplete
    restricted = access_only_datasets(records) or (marking.evidence if marking.marked else [])
    cfg = project.config

    report = AnalysisReport(
        title=f"{cfg.name}: analysis report",
        subtitle=" | ".join(filter(None, [f"Client: {cfg.client}" if cfg.client else "", f"CRS {cfg.crs}", f"Generated {datetime.now(UTC):%Y-%m-%d}", "Draft for geologist review"])),
        marked=marking.marked or bool(restricted),
        restricted=restricted,
    )
    licences = " | ".join(r.license for r in records)
    notices = [text for key, text in NOTICES.items() if key in licences]
    if restricted:
        notices.append(f"Access Only data (BC Data Catalogue) are used: {', '.join(restricted)}. No reproduction or redistribution without the Province's written permission.")
    builders = {
        "summary": lambda: _sec_summary(project, L),
        "data": lambda: _sec_inventory(project, L),
        "geology": lambda: _sec_geology(project, L, assets),
        "geochemistry": lambda: _sec_geochem(project, L, assets),
        "targets": lambda: _sec_targets(project, L, assets),
        "decisions": lambda: _sec_decisions(project, L, assets),
        "method": lambda: _sec_method(project, L),
        "sources": lambda: _sec_sources(project, notices),
    }
    for name, build in builders.items():
        if name in NEW_PAGE:
            report.blocks.append(PageBreak())
        report.blocks += build()
        report.sections.append(name)
    report.inputs = _inputs(records)
    return report


def _inputs(records: list) -> list[str]:
    """Every dataset the project fetched. The report quotes analysis outputs made from them, so
    it takes the strictest licence among them rather than guessing which fed which number."""
    return sorted({r.dataset for r in records if r.operation in {"fetch", "manual_ingest"}})
