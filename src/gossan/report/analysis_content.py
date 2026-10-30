"""The analysis report as format-neutral blocks, built from what ``gossan analyze`` and
``gossan target`` wrote. ``analysis_render`` turns the same blocks into a PDF and a Word file, so
the two never disagree.

Nothing is recomputed here: the report quotes the project's own output files. A section whose
inputs are missing says which command produces them. Findings are phrased from the numbers with
thresholds named in this module, and every conclusion that rests on few samples says so.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from gossan.analyze.targets import build_decisions
from gossan.core.licensing import access_only_datasets, is_open_dataset
from gossan.core.project import Project
from gossan.deliver.items import output_licence, restricts
from gossan.report.analysis_figures import (
    analysis_layer,
    catchment_map,
    drill_map,
    enrichment_chart,
    fault_chart,
    geology_map,
    ground_map,
    magnetics_map,
    read_csv,
    spectral_map,
    tenure_map,
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
    tenure: dict | None = None
    tenure_events: pd.DataFrame | None = None
    tenure_watch: pd.DataFrame | None = None
    drill: dict | None = None
    priorities: pd.DataFrame | None = None
    priorities_summary: dict | None = None
    spectral: dict | None = None
    knowledge: dict | None = None


def load(project: Project, *, open_only: bool = False) -> Loaded:
    a = project.root / "analysis"
    t = project.root / "target"
    records = project.provenance.read()
    if open_only and (t / "targets.gpkg").exists():
        if restricts(output_licence(records, "analysis/analysis.gpkg:target_decisions")) is not None:
            build_decisions(project, open_only=True)
            records = project.provenance.read()
    cv = _json(t / "cv_report.json")
    importance = pd.read_csv(t / "feature_importance.csv") if (t / "feature_importance.csv").exists() else None
    decisions = read_csv(project, "target_decisions")
    inventory = read_csv(project, "inventory")
    if open_only and inventory is not None:
        inventory = inventory[inventory["dataset"].map(lambda d: is_open_dataset(records, str(d)))].reset_index(drop=True)
        inventory = inventory if len(inventory) else None
    stale = []
    if project.data_gpkg.exists():
        newest_data = project.data_gpkg.stat().st_mtime
        for name in ("inventory.csv", "geochem_summary.json", "occurrence_summary.json", "tenure_summary.json", "target_decisions.csv"):
            path = a / name
            if path.exists() and path.stat().st_mtime < newest_data:
                stale.append(name)
    if (t / "targets.gpkg").exists() and (a / "target_decisions.csv").exists() and (a / "target_decisions.csv").stat().st_mtime < (t / "targets.gpkg").stat().st_mtime:
        stale.append("target_decisions.csv (older than target/targets.gpkg)")
    return Loaded(
        inventory,
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
        _json(a / "tenure_summary.json"),
        read_csv(project, "tenure_changes"),
        read_csv(project, "tenure_watch"),
        None if open_only else _json(a / "drill_summary.json"),
        None if open_only else read_csv(project, "digitize_priorities"),
        None if open_only else _json(a / "digitize_priorities_summary.json"),
        _json(a / "spectral_summary.json"),
        _json(t / "knowledge_report.json"),
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
    t = L.tenure
    if t and t.get("available"):
        wc = t["watch_counts"]
        out.append(
            f"Tenure: {wc['expiring']} of {t['n_tenures']} tenures reach their good-to date within 180 days and {wc['recent_staking']} were staked in the last year"
            + (f"; {t['n_events']} change(s) between {t['n_snapshots']} saved snapshots." if t["n_snapshots"] >= 2 else "; only one snapshot is saved, so changes cannot be compared yet.")
        )
    dr = L.drill
    if dr and dr.get("available"):
        c = dr["by_class"]
        out.append(
            f"Drilling: {dr['n_holes']} digitized hole(s) from {dr['n_reports']} report(s): {c['mineralized']} mineralized, {c['barren']} barren, "
            f"{c['inconclusive'] + c['no_assays']} inconclusive or without usable assays."
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
    mag = magnetics_map(project, assets / "magnetics.png")
    if mag:
        blocks.append(
            Figure(
                mag,
                "Figure 1b. Regional magnetic field and its tilt angle, which is about +90 degrees over the top of a magnetic source, 0 over its edges, and shows "
                "weak and strong sources alike. Compiled data (100 to 200 m cells): they show where magnetic ground changes over a few hundred metres, not detail "
                "a claim-scale survey would give.",
                0.95,
            )
        )
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


def _sec_spectral(project: Project, L: Loaded, assets: Path) -> list[Block]:
    """ASTER band-ratio alteration indices. Absent from the report unless ``gossan spectral`` was run."""
    s = L.spectral
    if s is None:
        return []
    blocks: list[Block] = [Heading("Satellite alteration indices (ASTER)")]
    if not s.get("available"):
        return blocks + [Para(f"Not available: {s.get('reason', 'no scenes')}.")]
    blocks.append(
        Para(
            f"Band ratios from {s['n_scenes']} ASTER scene(s) at {s['resolution_m']:.0f} m, on at-sensor radiance assuming {s['gain']} gain, "
            "after removing vegetation, snow and cells with no data. They respond to exposed rock and soil only: where forest or snow covers the ground, "
            "there is no value, and a grey cell is unknown, not unaltered. Each ratio is a published proxy for a mineral group, not a mineral map, and the values "
            "are not corrected for the atmosphere or the sun angle."
        )
    )
    bare = s.get("median_bare_pct")
    if bare is not None:
        blocks.append(Para(f"A typical scene leaves {bare:.0f}% of its cells as bare ground after masking."))
    fig = spectral_map(project, assets / "spectral_map.png")
    if fig:
        blocks.append(Figure(fig, "Figure 1. ASTER alteration indices on bare ground. Darker is a stronger response.", 0.95))
    rows = []
    for ix in s.get("indices", []):
        auc = ix.get("auc")
        rows.append([ix["label"], ix["formula"], f"{ix['valid_pct']:.0f}%", str(ix["n_occurrences"]), "too few" if auc is None else f"{auc:.2f}"])
    if rows:
        blocks.append(
            Table(
                [["Index", "Formula", "AOI with a value", "Occurrences with a value", "Occurrence AUC"], *rows],
                [2.0, 1.4, 1.0, 1.1, 0.9],
                "AUC is the chance that a metallic occurrence scores above a random bare-ground cell (0.5 is no separation). Occurrences cluster in a few "
                "districts and sit where there is exposure, so read it as a screen, not a validation.",
            )
        )
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
    blocks += _knowledge_blocks(L)
    return blocks


def _knowledge_blocks(L: Loaded) -> list[Block]:
    """The knowledge-driven score beside the forest: its layers, its checks and where the two disagree."""
    k = L.knowledge
    if not k:
        return []
    blocks: list[Block] = [Heading("A second opinion: the knowledge-driven score", 2)]
    if not k.get("available"):
        return blocks + [Para(f"Not available: {k.get('reason', 'no evidence layers')}.")]
    host = f"; host units: {', '.join(k['host_units'])}" if k.get("host_units") else ""
    blocks.append(
        Para(
            "The forest learns from the known occurrences, so it can only repeat what they share. This score is geological reasoning written down: each line of evidence "
            f"is turned into a 0 to 1 favourability by a stated rule and the lines are averaged with set weights (deposit model: {k['deposit_model'].replace('_', ' ')}{host}). "
            "It never looks at the occurrences, so they are an independent check on it. Weights are starting points from general exploration geology, not tuned on this area."
        )
    )
    rows = [["Evidence", "Weight", "Covers", "AUC alone"]]
    for layer in k["layers"]:
        auc = layer.get("auc")
        rows.append([layer["label"], f"{layer['weight_used']:g}", f"{layer['coverage_pct']:.0f}%", "n/a" if auc is None else f"{auc:.2f}"])
    blocks.append(Table(rows, [2.6, 0.8, 0.8, 1.0], "AUC alone: how well that rule by itself ranks the known occurrences above other cells. 0.5 means it does not bear out here; below 0.5 the occurrences sit where the rule says they should not, and its weight is worth a second look."))
    c, cv = k.get("checks") or {}, L.target_cv or {}

    def fmt(mean, sd):
        return "n/a" if mean is None else f"{mean:.2f} ± {sd:.2f}"

    compare = [["Held-out measure", "Random forest", "Knowledge score"]]
    compare.append(["AUC", fmt(cv.get("auc_mean"), cv.get("auc_std")), fmt(c.get("auc_fold_mean"), c.get("auc_fold_std"))])
    if cv.get("auc_effort_matched_mean") is not None or c.get("auc_effort_matched_fold_mean") is not None:
        compare.append(["Effort-matched AUC", fmt(cv.get("auc_effort_matched_mean"), cv.get("auc_effort_matched_std")), fmt(c.get("auc_effort_matched_fold_mean"), c.get("auc_effort_matched_fold_std"))])
    cap_f = "n/a" if cv.get("capture_top5_mean") is None else f"{cv['capture_top5_mean']:.0%}"
    cap_k = "n/a" if c.get("capture_top5_fold_mean") is None else f"{c['capture_top5_fold_mean']:.0%}"
    compare.append(["Top-5% capture", cap_f, cap_k])
    blocks.append(Table(compare, [2.0, 1.6, 1.6], "Same spatial folds for both. The knowledge score is not fitted, so its figures are its skill on those blocks."))
    rho = k.get("rank_correlation_with_forest")
    agree = f"Their rank correlation is {rho:.2f}; " if rho is not None else ""
    support = ""
    if L.decisions is not None and "knowledge_support" in L.decisions:
        counts = L.decisions["knowledge_support"].value_counts().to_dict()
        support = " Of the forest's targets, " + ", ".join(f"{counts.get(k_, 0)} {k_}" for k_ in ("supported", "partly", "forest only")) + "."
    blocks.append(
        Para(
            f"{agree}the top {100 - k['top_percentile']:g}% of ground overlaps by {k['top_overlap_with_forest']:.0%} (shared cells over cells in either).{support} "
            f"The knowledge score alone has {k['n_knowledge_targets']} target polygon(s) (target/knowledge_targets.gpkg); the ones that miss every forest target are where geology "
            "looks good but the data the forest learned from are thin."
        )
    )
    if k.get("note"):
        blocks.append(Para(k["note"].capitalize() + ".", small=True))
    return blocks


def _sec_tenure(project: Project, L: Loaded, assets: Path) -> list[Block]:
    blocks: list[Block] = [Heading("Tenure activity")]
    s = L.tenure
    if s is None:
        return blocks + _need("gossan analyze --steps tenure", "The tenure analysis")
    if not s.get("available"):
        return blocks + [Para(f"Not available: {s.get('reason', 'no tenure data')}.")]
    blocks.append(
        Para(
            "Mineral Titles Online publishes only current tenure, so Gossan keeps its own history: every harvest saves the tenure layer, and consecutive saved "
            "snapshots are compared. The watchlist below is read from the latest snapshot and works from the first harvest."
        )
    )
    n_snap, age = s["n_snapshots"], s.get("latest_snapshot_age_days")
    when = ", ".join(f"{x['first_seen'][:10]}" for x in s["snapshots"][-3:])
    blocks.append(Para(f"{s['n_tenures']} tenures in the area; {n_snap} saved snapshot(s)" + (f" (latest: {when.split(', ')[-1]}, {age} day(s) before this report)." if n_snap else ".")))
    if n_snap < 2:
        blocks.append(Note("Only one snapshot exists, so there is nothing to compare yet. Re-run `gossan run <project> --steps harvest,analyze,report` on a schedule (weekly is enough) and changes appear in the next report.", "warn"))
    elif age is not None and age > 30:
        blocks.append(Note(f"The latest snapshot is {age} days old: re-harvest to see current tenure.", "warn"))
    for note in s.get("notes", []):
        blocks.append(Note(note, "warn"))
    fig = tenure_map(project, assets / "tenure.png")
    if fig:
        blocks.append(Figure(fig, "Figure 6. Mineral tenure: staked in the last year (blue), good-to date soon (orange), and changed between snapshots (black outline).", 0.95))

    ev = L.tenure_events
    if n_snap >= 2:
        blocks.append(Heading("Changes between snapshots", 2))
        if ev is None:
            blocks.append(Para(f"No tenure changed between the {n_snap} snapshots ({s['snapshots'][0]['first_seen'][:10]} to {s['snapshots'][-1]['last_seen'][:10]})."))
        else:
            counts = ev["change"].value_counts()
            blocks.append(Table([["Kind of change", "Tenures"], *[[k.replace("_", " "), str(int(v))] for k, v in counts.items()]], [3.0, 1.0],
                                "'gone' means no longer listed (lapsed, forfeited, converted or cancelled; the layer does not say which)."))
            rows = [["Tenure", "Change", "Detail", "Holder", "Happened between"]]
            for r in ev.sort_values("by", ascending=False).head(15).itertuples():
                rows.append([str(r.tenure_number), r.change.replace("_", " "), _text(r.detail), _text(r.owner), f"{r.after} and {r.by}"])
            blocks.append(Table(rows, [0.8, 1.1, 2.4, 2.0, 1.5], f"Most recent {min(len(ev), 15)} of {len(ev)} changes; the full list is analysis/tenure_changes.csv."))

    wc = s["watch_counts"]
    blocks.append(Heading("Watchlist", 2))
    blocks.append(Bullets([
        f"{wc['expiring']} tenure(s) with a good-to date in the next 180 days: the holder must file work or pay to keep them, or the ground may open.",
        f"{wc['recent_staking']} staked in the last year; {wc['recently_updated']} with a record updated in the last 90 days; {wc['churn']} that have changed hands twice or more.",
    ]))
    w = L.tenure_watch
    if w is not None:
        rows = [["Tenure", "Claim", "Holder", "ha", "Flags", "Good to", "Days"]]
        for r in w.head(15).itertuples():
            rows.append([str(r.tenure_number), _text(r.claim_name), _text(r.owner), _num(r.area_ha), r.flags.replace("_", " "), _text(r.good_to_date), _num(r.days_to_good_to)])
        blocks.append(Table(rows, [0.7, 1.4, 1.9, 0.5, 1.7, 0.9, 0.5], f"Top {min(len(w), 15)} of {len(w)} by number of flags, then soonest expiry. The full list is analysis/tenure_watch.csv."))
    if s.get("recent_staking_by_owner"):
        blocks.append(Heading("Recent staking by holder", 2))
        blocks.append(Table([["Holder", "Tenures", "Hectares"], *[[o["owner"], str(o["tenures"]), f"{o['hectares']:,.0f}"] for o in s["recent_staking_by_owner"]]], [3.4, 0.9, 1.1],
                            "Tenures issued in the last year, largest holders first."))
    return blocks


def _digitize_next(L: Loaded) -> list[Block]:
    """The drilling reports worth digitizing next (from ``gossan analyze --steps priorities``)."""
    s, t = L.priorities_summary, L.priorities
    if not s or not s.get("available") or t is None or not len(s["top"]):
        return []
    near = f"{s['n_tier1']} of the reports not yet digitized lie within {s['radius_m'] / 1000:g} km of a {s['reference']}"
    blocks: list[Block] = [
        Heading("Reports worth digitizing next", 2),
        Para(f"{s['n_drilling_reports']} of the {s['n_reports']} assessment reports in the area describe drilling; {s['n_digitized']} have been digitized. {near}. "
             "A report's location is a point on the property, not the holes, so distance is a guide to relevance rather than a measurement."),
    ]
    col = next((c for c in t.columns if c.startswith("distance_to_nearest_")), None)
    rows = [["#", "ARIS", "Year", "Report", "Spend (2002 $)", "Distance (km)"]]
    for r in t[t["priority"].notna()].head(8).itertuples():
        dist = getattr(r, col) if col else None
        rows.append([str(int(r.priority)), str(r.aris_number), _num(r.year, "{:.0f}"), _text(r.title)[:70], _num(r.spend_2002_dollars), _num(None if dist is None or pd.isna(dist) else dist / 1000, "{:.1f}")])
    blocks.append(Table(rows, [0.3, 0.7, 0.6, 3.6, 1.0, 0.9], "Ranked by closeness to a target, then by reported spend. Download in this order with `gossan harvest <project> --limit-pdfs N --pdf-order priority`, then `gossan digitize` and `gossan review`."))
    return blocks


def _sec_drilling(project: Project, L: Loaded, assets: Path) -> list[Block]:
    blocks: list[Block] = [Heading("Drilling evidence")]
    d = L.drill
    if d is None:
        return blocks + _need("gossan analyze --steps drilling", "The drilling analysis")
    if not d.get("available"):
        return blocks + [Para(f"No digitized drill holes yet: {d.get('reason', '')}.")] + _digitize_next(L)
    c, loc = d["by_class"], d["by_location"]
    blocks.append(
        Para(
            f"{d['n_holes']} drill hole(s) from {d['n_reports']} assessment report(s) were digitized and approved by a reviewer. Each is graded in one unit per element and called "
            "mineralized (an interval reaches the screening grade for any element), barren (sampled over enough metres and nowhere near any screening grade), or inconclusive. "
            "A barren hole tests only its own line: ore bodies are narrow and holes miss them, so barren is evidence about that spot, not proof the ground is empty."
        )
    )
    fig = drill_map(project, assets / "drilling.png")
    if fig:
        blocks.append(Figure(fig, "Figure 8. Digitized drill holes by result, with the targets. Open symbols are holes known only by their report's location.", 0.95))
    rows = [["Result", "Holes"], *[[k.replace("_", " "), str(v)] for k, v in c.items() if v]]
    blocks.append(Table(rows, [2.0, 0.8]))
    blocks.append(Table(
        [["Position known from", "Holes"], ["collar coordinates", str(loc["collar_coordinates"])], ["the report's location only", str(loc["report_location"])], ["neither (not on the map)", str(loc["unplaced"])]],
        [2.6, 0.8], "Collar coordinates are read as UTM in the project's zone and the datum set under georef:; a collar that lands outside the area is not placed (local mine grid, wrong zone or datum?).",
    ))
    left = d["left_out_of_grades"]
    if left["no_unit"] or left["unknown_unit"]:
        blocks.append(Note(f"{left['no_unit']} assay value(s) have no unit and {left['unknown_unit']} an unknown one: they were left out of the grades rather than guessed, because a bare 'Au' column could be ppb or g/t. Add the unit to the extracted table and re-approve it.", "warn"))
    if d["mineralized_holes"]:
        blocks.append(Heading("Mineralized holes", 2))
        rows = [["Hole", "Element", "Best interval grade", "Interval"]]
        rows += [[h["hole"], h["element"], f"{h['grade']:g} {h['unit']}", h["interval"]] for h in d["mineralized_holes"]]
        blocks.append(Table(rows, [2.6, 0.8, 1.6, 1.4], "Screening grades are far below economic grades (see drill: in project.yaml): they show that mineralization is present."))
    if L.decisions is not None and "n_drill_holes" in L.decisions:
        tested = L.decisions[L.decisions["n_drill_holes"].fillna(0) > 0]
        blocks.append(Para(f"{len(tested)} of {len(L.decisions)} targets already have a digitized drill hole on or beside them; the decision table says what they found."))
    return blocks + _digitize_next(L)


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
        blocks.append(Figure(fig, "Figure 7. Targets at their centroids, sized by area and coloured by ground status.", 0.95))
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
                f"Exploration effort mode: {t.effort}; positives: {(L.target_grid or {}).get('occurrence_class', t.occurrence_class)} occurrences; seed {t.seed}."
            ),
            (
                "Model features describe the ground (faults, contacts, intrusions, unit age, terrain, geochemistry, host-unit favourability) rather than where exploration happened, "
                "so the map can point at ground nobody has tested."
            ),
        ]),
        Para("The tables are in analysis/ and target/ as CSV, GeoPackage and GeoTIFF for use in QGIS. Re-run `gossan harvest`, `gossan analyze` and `gossan target` to refresh them.", small=True),
    ]


def _sec_sources(project: Project, notices: list[str], *, open_only: bool = False) -> list[Block]:
    blocks: list[Block] = [Heading("Data sources and licences")]
    records = project.provenance.read()
    sources = data_sources(records)
    if open_only:
        sources = [r for r in sources if is_open_dataset(records, r.dataset)]
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
NEW_PAGE = {"geology", "geochemistry", "targets", "tenure", "decisions", "method"}  # drilling follows tenure on its page

NOTICES = {
    "Open Government Licence - British Columbia": "Contains information licensed under the Open Government Licence - British Columbia.",
    "Open Government Licence - Canada": "Contains information licensed under the Open Government Licence - Canada.",
}


def number_figures(report: AnalysisReport) -> None:
    """Number the figures 1, 2, 3, ... in the order they appear, whatever their captions started with,
    so adding or dropping a figure never leaves a gap or a repeat."""
    n = 0
    for block in report.blocks:
        if isinstance(block, Figure):
            n += 1
            block.caption = f"Figure {n}. " + re.sub(r"^Figure \d+[a-z]?\. ", "", block.caption)


def build_content(project: Project, assets: Path, *, open_only: bool = False) -> AnalysisReport:
    """Assemble the report's blocks; figures are written under ``assets``."""
    L = load(project, open_only=open_only)
    records = project.provenance.read()
    marking = decide_marking(project)  # also looks at files, in case the ledger is incomplete
    restricted = [] if open_only else (access_only_datasets(records) or (marking.evidence if marking.marked else []))
    cfg = project.config

    report = AnalysisReport(
        title=f"{cfg.name}: analysis report",
        subtitle=" | ".join(filter(None, [f"Client: {cfg.client}" if cfg.client else "", f"CRS {cfg.crs}", f"Generated {datetime.now(UTC):%Y-%m-%d}", "Draft for geologist review"])),
        marked=False if open_only else (marking.marked or bool(restricted)),
        restricted=restricted,
    )
    licences = " | ".join(r.license for r in records if not open_only or is_open_dataset(records, r.dataset))
    notices = [text for key, text in NOTICES.items() if key in licences]
    if restricted:
        notices.append(f"Access Only data (BC Data Catalogue) are used: {', '.join(restricted)}. No reproduction or redistribution without the Province's written permission.")
    builders = {
        "summary": lambda: _sec_summary(project, L),
        "data": lambda: _sec_inventory(project, L),
        "geology": lambda: _sec_geology(project, L, assets),
        "geochemistry": lambda: _sec_geochem(project, L, assets),
        "spectral": lambda: _sec_spectral(project, L, assets),
        "targets": lambda: _sec_targets(project, L, assets),
        "tenure": lambda: _sec_tenure(project, L, assets),
        "drilling": lambda: [] if open_only else _sec_drilling(project, L, assets),
        "decisions": lambda: _sec_decisions(project, L, assets),
        "method": lambda: _sec_method(project, L),
        "sources": lambda: _sec_sources(project, notices, open_only=open_only),
    }
    for name, build in builders.items():
        if name in NEW_PAGE:
            report.blocks.append(PageBreak())
        blocks = build()
        if not blocks:  # an optional section with nothing to say
            if name in NEW_PAGE:
                report.blocks.pop()
            continue
        report.blocks += blocks
        report.sections.append(name)
    report.inputs = [d for d in _inputs(records) if not open_only or is_open_dataset(records, d)]
    number_figures(report)
    return report


def _inputs(records: list) -> list[str]:
    """Every dataset the project fetched. The report quotes analysis outputs made from them, so
    it takes the strictest licence among them rather than guessing which fed which number."""
    return sorted({r.dataset for r in records if r.operation in {"fetch", "manual_ingest"}})
