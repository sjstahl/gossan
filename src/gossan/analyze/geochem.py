"""Multi-element analysis of the RGS stream-sediment samples.

What it does, per sample:

1. **Picks one analysis per element.** RGS reports many elements by several methods (AAS, INAA,
   ICP-MS, ...). For each element the method with the most values is used. Below-detection
   results are published as zero or as the detection limit (the column's "floor"), so elements
   fall in three tiers:

   * *continuous*: at least ``MIN_COVERAGE`` of samples analysed, ``MIN_DISTINCT`` distinct values
     and no more than ``MAX_CENSORED`` at the floor. Used everywhere, including PCA.
   * *censored*: analysed in enough samples but mostly at the floor (typical for Au, Ag, Sb, Cd).
     A value above the floor is exactly the anomaly of interest, so these are scored and used in
     the indices, but left out of PCA, which needs continuous data.
   * *dropped*: too few samples analysed, or nothing above the floor. Listed with the reason.
2. **Converts to ppm.** Zeros become half the smallest positive value.
3. **Normal scores** (``<el>_z``): each element's values ranked across samples and mapped to the
   standard normal (ties, including every value at the floor, share a rank). They are comparable
   between elements, unaffected by coarse lab steps and robust to outliers; 2 is about the top
   2.5%. Concentrations stay in ``<el>_ppm``.
4. **Background correction.** A sample can look anomalous for reasons that are not mineralization:
   Fe and Mn oxide coatings adsorb metals (*scavenging*); a large catchment dilutes whatever a
   small source sheds (*dilution*); and some rocks are simply richer in an element than others
   (*lithology*: black shale is naturally high in Zn, Mo and U). Each element's log value is
   regressed on log Fe and log Mn and, when the DEM gives each sample its catchment
   (``gossan.analyze.catchments``) and ``geochem.dilution_correction`` / ``lithology_levelling``
   are on, on log catchment area and the share of each bedrock unit in the catchment. The
   residual, what is left after the expected background, is normal-scored (``<el>_zc``;
   continuous elements only, censored ones keep their uncorrected score). Deposit indices and
   ``top_elements`` (elements scoring >= 2) use these corrected scores. Samples whose catchment
   could not be delineated get the scavenging correction only.

   Dilution and lithology corrections are off by default: where deposits are hosted by
   particular units, levelling those units away removes real signal (see ``GeochemConfig``).
   Either way, ``catchment_r2_adjusted`` in the summary reports how much of each element's
   variation catchment size and lithology explain: how much of the "anomaly" pattern simply
   follows the rocks.
5. **Deposit-model indices** (``idx_<model>``): the mean corrected z of a model's pathfinder
   elements (``DEPOSIT_INDICES``). They are starting points for review, not classifications.
6. **Compositional PCA.** Concentrations are compositional (parts of a whole), so they are
   centred-log-ratio transformed before PCA. The leading components usually separate lithology
   from mineralization; read the loadings table to name them.
7. **Multivariate anomaly**: robust (minimum covariance determinant) Mahalanobis distance on the
   leading components. ``multivariate_anomaly`` flags the ``ANOMALY_TOP_FRACTION`` most distant
   samples: those whose overall element *combination* is unusual, even if no single element is
   extreme. (Geochemical data are too heavy-tailed to trust chi-square p-values here.)

With a DEM, each sample's catchment is written as ``rgs_catchments`` (subcatchment polygons:
the ground draining to that sample before the next one downstream) carrying the sample's scores,
which is how anomalies should be mapped and followed up. Survey vintages and labs differ (see
``rgs_survey_areas``); levelling between surveys is not attempted here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA

from gossan.analyze.catchments import Catchments, catchment_polygons, delineate
from gossan.analyze.common import (
    AnalysisError,
    log_outputs,
    read_layer,
    write_json,
    write_layer,
    write_table,
)
from gossan.analyze.occurrences import unit_name
from gossan.core.project import Project

COLUMN = re.compile(r"^(?P<element>[A-Z]+)_(?P<method>[A-Z]+)_(?P<unit>PPM|PPB|PERCENTAGE)$")
EXCLUDED_ELEMENTS = {"WATER"}  # water chemistry columns (e.g. WATER_URANIUM_PPB) are not sediment
TO_PPM = {"PPM": 1.0, "PPB": 1e-3, "PERCENTAGE": 1e4}
MIN_COVERAGE = 0.5
MIN_DISTINCT = 5
MAX_CENSORED = 0.5
#: A censored element needs at least this many samples above its floor to be scored.
MIN_DETECTIONS = 3
#: Samples with fewer than this share of the selected elements are left out of PCA/anomaly scoring.
MIN_SAMPLE_COMPLETENESS = 0.7
MAX_COMPONENTS = 6
ANOMALY_TOP_FRACTION = 0.05
Z_REPORT = 2.0
SCAVENGERS = ("IRON", "MANGANESE")

#: Pathfinder suites. Edit to suit the deposit models of the area.
DEPOSIT_INDICES: dict[str, list[str]] = {
    "sediment_hosted_zn_pb": ["ZINC", "LEAD", "CADMIUM", "SILVER", "BARIUM"],
    "polymetallic_vein": ["LEAD", "ZINC", "SILVER", "ANTIMONY", "COPPER", "ARSENIC"],
    "cu_porphyry_skarn": ["COPPER", "MOLYBDENUM", "BISMUTH", "TUNGSTEN", "GOLD"],
    "gold_vein": ["GOLD", "ARSENIC", "ANTIMONY", "MERCURY", "SILVER"],
    "rare_earth_carbonatite": ["CERIUM", "LANTHANUM", "SAMARIUM", "THORIUM", "URANIUM", "TANTALUM"],
}
ID_COLUMNS = ("MASTER_ID", "MATERIAL", "STREAM_ORDER", "ELEVATION")


@dataclass
class ElementChoice:
    element: str
    column: str | None
    method: str | None
    n: int = 0
    coverage: float = 0.0
    censored: float = 0.0
    distinct: int = 0
    tier: str = "dropped"  # "continuous", "censored" or "dropped"
    dropped: str | None = None


@dataclass
class GeochemResult:
    samples: gpd.GeoDataFrame
    elements: pd.DataFrame
    loadings: pd.DataFrame
    summary: dict
    outputs: list[str] = field(default_factory=list)


def _censored_share(v: pd.Series) -> float:
    """Share of values at the floor: zeros when the column has any (they stand for "below
    detection", so the smallest positive value is a real detection), else its smallest value."""
    positive = v[v > 0]
    if positive.empty:
        return 1.0
    if (v <= 0).any():
        return float((v <= 0).mean())
    return float((v <= positive.min()).mean())


def choose_columns(df: pd.DataFrame) -> list[ElementChoice]:
    """One usable analytical column per element (see the module docstring)."""
    by_element: dict[str, list[tuple[str, str]]] = {}
    for col in df.columns:
        m = COLUMN.match(str(col))
        if m and m["element"] not in EXCLUDED_ELEMENTS:
            by_element.setdefault(m["element"], []).append((col, m["method"]))
    choices = []
    n = len(df)
    for element in sorted(by_element):
        stats = []
        for col, method in by_element[element]:
            v = pd.to_numeric(df[col], errors="coerce").dropna()
            stats.append(
                ElementChoice(
                    element, col, method, len(v), len(v) / n if n else 0.0,
                    _censored_share(v) if len(v) else 1.0, int(v.nunique()),
                )
            )
        stats.sort(key=lambda s: (s.coverage, s.distinct), reverse=True)
        usable = [
            s for s in stats
            if s.coverage >= MIN_COVERAGE and s.distinct >= MIN_DISTINCT and s.censored <= MAX_CENSORED
        ]  # fmt: skip
        if usable:
            usable[0].tier = "continuous"
            choices.append(usable[0])
            continue
        analysed = sorted(
            (s for s in stats if s.coverage >= MIN_COVERAGE and s.distinct >= 2), key=lambda s: s.censored
        )
        if analysed and round((1 - analysed[0].censored) * analysed[0].n) >= MIN_DETECTIONS:
            analysed[0].tier = "censored"
            choices.append(analysed[0])
            continue
        best = stats[0]
        if best.coverage < MIN_COVERAGE:
            reason = f"analysed in {best.coverage:.0%} of samples (need {MIN_COVERAGE:.0%})"
        else:
            reason = f"{best.censored:.0%} of values at the detection floor ({best.column})"
        best.dropped = reason
        choices.append(best)
    return choices


def to_ppm(df: pd.DataFrame, choices: list[ElementChoice]) -> pd.DataFrame:
    """Selected columns in ppm, zeros replaced by half the smallest positive value; NaN kept."""
    out = {}
    for c in choices:
        if c.dropped or c.column is None:
            continue
        unit = COLUMN.match(c.column)["unit"]  # type: ignore[index]
        v = pd.to_numeric(df[c.column], errors="coerce") * TO_PPM[unit]
        floor = v[v > 0].min()
        out[c.element] = v.where(~(v <= 0), floor / 2.0)
    return pd.DataFrame(out, index=df.index)


def normal_scores(x: pd.DataFrame) -> pd.DataFrame:
    """Each column ranked (ties averaged, NaN kept) and mapped to standard-normal quantiles."""
    out = {}
    for col in x.columns:
        v = x[col].to_numpy(dtype=float)
        ok = np.isfinite(v)
        z = np.full(len(v), np.nan)
        if ok.any():
            z[ok] = norm.ppf(rankdata(v[ok]) / (ok.sum() + 1))
        out[col] = z
    return pd.DataFrame(out, index=x.index)


def scavenging_residuals(logs: pd.DataFrame) -> pd.DataFrame:
    """Each element's log value minus its least-squares fit on log Fe and log Mn (NaN-safe)."""
    present = [s for s in SCAVENGERS if s in logs]
    if len(present) < len(SCAVENGERS):
        return logs.copy()
    out = logs.copy()
    for el in logs.columns:
        if el in present:
            continue
        rows = logs[[el, *present]].dropna()
        if len(rows) < 10:
            continue
        A = np.column_stack([np.ones(len(rows)), rows[present].to_numpy()])
        coef, *_ = np.linalg.lstsq(A, rows[el].to_numpy(), rcond=None)
        fitted = coef[0] + logs[present].to_numpy() @ coef[1:]
        out[el] = logs[el] - fitted
    return out


def background_residuals(logs: pd.DataFrame, covariates: pd.DataFrame | None) -> pd.DataFrame:
    """Scavenging residuals, extended with ``covariates`` (one row per sample, NaN where unknown)
    as extra predictors for the samples that have them. An element keeps its scavenging-only
    residual where the extended fit lacks rows (fewer than three per predictor)."""
    base = scavenging_residuals(logs)
    if covariates is None or covariates.empty:
        return base
    present = [s for s in SCAVENGERS if s in logs]
    predictors = pd.concat([logs[present], covariates], axis=1) if len(present) == len(SCAVENGERS) else covariates
    out = base.copy()
    for el in logs.columns:
        if el in present:
            continue
        rows = pd.concat([logs[el], predictors], axis=1).dropna()
        if len(rows) < max(10, 3 * (predictors.shape[1] + 1)):
            continue
        A = np.column_stack([np.ones(len(rows)), rows[predictors.columns].to_numpy()])
        coef, *_ = np.linalg.lstsq(A, rows[el].to_numpy(), rcond=None)
        known = predictors.notna().all(axis=1) & logs[el].notna()
        fitted = coef[0] + predictors[known].to_numpy() @ coef[1:]
        out.loc[known, el] = logs.loc[known, el] - fitted
    return out


def catchment_covariates(table: pd.DataFrame, unit_names: list[str] | None, shares: np.ndarray | None, min_unit_share: float) -> pd.DataFrame:
    """Predictors from catchments: log10 catchment area and, with geology, the share of each
    common bedrock unit (rare units pooled; the most common one left out as the reference, since
    the shares sum to one)."""
    cov = pd.DataFrame({"log_catchment_area": np.log10(table["catchment_area_km2"].clip(lower=1e-3))}, index=table.index)
    if unit_names is None or shares is None:
        return cov
    df = pd.DataFrame(shares, columns=unit_names, index=table.index)
    mean = df.mean()
    common = [u for u in unit_names if mean[u] >= min_unit_share]
    if len(common) < 2:
        return cov
    rest = [u for u in unit_names if u not in common]
    if rest:
        df["(other units)"] = df[rest].sum(axis=1)
        common.append("(other units)")
    reference = df[common].mean().idxmax()
    keep = [u for u in common if u != reference]
    return cov.join(df[keep].add_prefix("unit: "))


def _top(z: pd.DataFrame) -> list[str]:
    tops = []
    for _, row in z.iterrows():
        hits = row[row >= Z_REPORT].sort_values(ascending=False)
        tops.append(", ".join(f"{el.title()} {v:.1f}" for el, v in hits.items()))
    return tops


def score_samples(
    rgs: gpd.GeoDataFrame, covariates: pd.DataFrame | None = None, catchment: pd.DataFrame | None = None
) -> GeochemResult:
    """Score every RGS sample (see the module docstring); writes nothing.

    ``covariates`` (aligned with ``rgs``) are extra background predictors (``catchment_covariates``);
    ``catchment`` columns (areas, snapping, main unit) are copied onto the scored samples.
    Raises ``AnalysisError`` if fewer than three elements are usable as continuous data.
    """
    choices = choose_columns(rgs)
    ppm = to_ppm(rgs, choices)
    continuous = [c.element for c in choices if c.tier == "continuous"]
    if len(continuous) < 3:
        raise AnalysisError(f"only {len(continuous)} continuous RGS element(s); need 3 for multivariate analysis")
    logs = np.log10(ppm)
    z = normal_scores(logs)
    zc = z.copy()
    residuals = background_residuals(logs[continuous], covariates)
    zc[continuous] = normal_scores(residuals)
    n_pred = 2 + (0 if covariates is None else covariates.shape[1])
    background_r2 = {}
    for el in continuous:
        v, r = logs[el].dropna(), residuals[el].dropna()
        if len(v) > n_pred + 1 and v.var() > 0:
            r2 = 1 - r.var() / v.var()
            background_r2[el.title()] = round(float(1 - (1 - r2) * (len(v) - 1) / (len(v) - n_pred - 1)), 3)

    # Compositional PCA on the continuous elements of complete-enough samples, gaps filled with medians.
    cont = ppm[continuous]
    complete = cont.notna().mean(axis=1) >= MIN_SAMPLE_COMPLETENESS
    filled = np.log(cont[complete].fillna(cont.median()))
    clr = filled.sub(filled.mean(axis=1), axis=0)
    k = int(min(MAX_COMPONENTS, clr.shape[1] - 1, max(complete.sum() - 1, 1)))
    pca = PCA(n_components=k).fit(clr)
    scores = pd.DataFrame(pca.transform(clr), index=clr.index, columns=[f"PC{i + 1}" for i in range(k)])
    loadings = pd.DataFrame(pca.components_.T, index=clr.columns, columns=scores.columns)
    loadings.loc["explained_variance"] = pca.explained_variance_ratio_
    mcd = MinCovDet(random_state=0).fit(scores.to_numpy())
    d2 = pd.Series(mcd.mahalanobis(scores.to_numpy()), index=scores.index)
    cutoff = d2.quantile(1 - ANOMALY_TOP_FRACTION)

    cols: dict[str, pd.Series] = {}
    for el in ppm.columns:
        cols[f"{el.lower()}_ppm"] = ppm[el]
        cols[f"{el.lower()}_z"] = z[el].round(3)
        cols[f"{el.lower()}_zc"] = zc[el].round(3)
    for name, suite in DEPOSIT_INDICES.items():
        available = [e for e in suite if e in zc]
        idx = zc[available].mean(axis=1).round(3) if len(available) >= 2 else pd.Series(np.nan, index=zc.index)
        cols[f"idx_{name}"] = idx
    cols.update({c: scores[c].round(4) for c in scores})
    cols["anomaly_d2"] = d2.round(3)
    cols["anomaly_rank"] = d2.rank(ascending=False)
    cols["multivariate_anomaly"] = (d2 >= cutoff).reindex(rgs.index, fill_value=False).astype(bool)
    cols["top_elements"] = pd.Series(_top(zc), index=rgs.index)
    ids = rgs[[c for c in ID_COLUMNS if c in rgs]]
    extra = [catchment.reindex(rgs.index)] if catchment is not None else []
    out = pd.concat([ids, *extra, pd.DataFrame(cols).reindex(rgs.index)], axis=1)
    samples = gpd.GeoDataFrame(out, geometry=rgs.geometry.values, crs=rgs.crs)

    elements = pd.DataFrame(
        [
            {
                "element": c.element.title(),
                "column": c.column,
                "method": c.method,
                "n_samples": c.n,
                "coverage": round(c.coverage, 3),
                "share_at_floor": round(c.censored, 3),
                "distinct_values": c.distinct,
                "tier": c.tier,
                "dropped_reason": c.dropped,
                **(
                    {
                        "median_ppm": ppm[c.element].median(),
                        "p95_ppm": ppm[c.element].quantile(0.95),
                        "p98_ppm": ppm[c.element].quantile(0.98),
                        "max_ppm": ppm[c.element].max(),
                    }
                    if c.element in ppm
                    else {}
                ),
            }
            for c in choices
        ]
    )
    index_cols = [c for c in samples if c.startswith("idx_")]
    summary = {
        "n_samples": len(samples),
        "n_samples_scored": int(complete.sum()),
        "elements_continuous": [e.title() for e in continuous],
        "elements_censored": [c.element.title() for c in choices if c.tier == "censored"],
        "elements_dropped": {c.element.title(): c.dropped for c in choices if c.dropped},
        "scavenging_correction": all(s in ppm for s in SCAVENGERS),
        #: Adjusted R2 of each element's background model: the share of its variation explained by
        #: scavenging (and catchment size and lithology), which the corrected scores remove.
        "background_r2_adjusted": background_r2,
        "background_predictors": [
            *([f"log {s.title()}" for s in SCAVENGERS] if all(s in ppm for s in SCAVENGERS) else []),
            *([] if covariates is None else list(covariates.columns)),
        ],
        "pca_explained_variance": [round(float(v), 3) for v in pca.explained_variance_ratio_],
        "pca_top_loadings": {
            pc: {
                "positive": [e.title() for e in loadings[pc].drop("explained_variance").nlargest(4).index],
                "negative": [e.title() for e in loadings[pc].drop("explained_variance").nsmallest(4).index],
            }
            for pc in scores.columns
        },
        "n_multivariate_anomalies": int(samples["multivariate_anomaly"].sum()),
        "top_samples_by_index": {
            c: samples.nlargest(5, c)[[*(x for x in ("MASTER_ID",) if x in samples), c, "top_elements"]].to_dict("records")
            for c in index_cols
            if samples[c].notna().any()
        },
    }
    return GeochemResult(samples, elements, loadings, summary)


def _explained(rgs: gpd.GeoDataFrame, covariates: pd.DataFrame) -> dict[str, float]:
    """Adjusted R2 of each continuous element on Fe, Mn, catchment area and catchment units: a
    diagnostic of how strongly the geochemistry follows the rocks and the drainage."""
    choices = [c for c in choose_columns(rgs) if c.tier == "continuous"]
    logs = np.log10(to_ppm(rgs, choices))
    residuals = background_residuals(logs, covariates)
    p = 2 + covariates.shape[1]
    out = {}
    for el in logs.columns:
        v, r = logs[el].dropna(), residuals[el].dropna()
        if len(v) > p + 1 and v.var() > 0 and el not in SCAVENGERS:
            r2 = 1 - r.var() / v.var()
            out[el.title()] = round(float(1 - (1 - r2) * (len(v) - 1) / (len(v) - p - 1)), 3)
    return out


def sample_catchments(project: Project, rgs: gpd.GeoDataFrame) -> tuple[Catchments | None, str]:
    """The samples' catchments over ``rasters/dem.tif`` (with bedrock units when harvested), or
    None and the reason."""
    dem = project.root / "rasters" / "dem.tif"
    if not dem.exists():
        return None, "no rasters/dem.tif (run 'gossan harvest --source dem'): samples treated as points"
    cfg = project.config.geochem
    bedrock = read_layer(project, "bedrock_geology")
    if bedrock is not None:
        bedrock = bedrock.assign(unit=bedrock.apply(unit_name, axis=1))
    c = delineate(
        dem, rgs.geometry, resolution=cfg.catchment_resolution_m, snap_m=cfg.snap_m,
        stream_km2=cfg.stream_km2, units=bedrock,
    )  # fmt: skip
    return c, ""


def _catchment_columns(c: Catchments) -> pd.DataFrame:
    cols = c.samples[["catchment_area_km2", "subcatchment_area_km2", "snap_dist_m", "catchment_truncated"]].copy()
    if c.unit_names is not None and c.unit_shares is not None:
        shares = np.nan_to_num(c.unit_shares, nan=-1.0)
        best = shares.argmax(axis=1)
        cols["main_catchment_unit"] = [c.unit_names[b] if shares[i, b] >= 0 else None for i, b in enumerate(best)]
        cols["main_catchment_unit_share"] = np.where(shares.max(axis=1) >= 0, shares.max(axis=1).round(3), np.nan)
    return cols


@dataclass
class ProjectScores:
    result: GeochemResult
    catchments: Catchments | None
    #: Datasets the scores depend on (for licensing).
    inputs: list[str]


def score_project(project: Project, rgs: gpd.GeoDataFrame) -> ProjectScores:
    """Score ``rgs`` (the project's samples, in the project CRS) with catchment corrections when a
    DEM is available; writes nothing. Used by ``analyze_geochem`` and the target features."""
    inputs = ["rgs"]
    catch, why_not = sample_catchments(project, rgs)
    covariates = info = None
    if catch is not None:
        inputs.append("dem")
        if catch.unit_names is not None:
            inputs.append("bedrock_geology")
        info = _catchment_columns(catch).set_index(rgs.index)
        gcfg = project.config.geochem
        everything = catchment_covariates(info, catch.unit_names, catch.unit_shares, gcfg.min_unit_share).set_index(rgs.index)
        chosen = [
            c for c in everything.columns
            if (c == "log_catchment_area" and gcfg.dilution_correction)
            or (c.startswith("unit: ") and gcfg.lithology_levelling)
        ]  # fmt: skip
        covariates = everything[chosen] if chosen else None
    result = score_samples(rgs, covariates, info)
    if catch is not None:
        result.summary["catchment_r2_adjusted"] = _explained(rgs, everything)
    result.summary["catchments"] = (
        why_not
        if catch is None
        else {
            "resolution_m": catch.resolution,
            "median_catchment_km2": float(np.nanmedian(info["catchment_area_km2"])),
            "median_snap_m": float(np.nanmedian(info["snap_dist_m"])),
            "n_truncated": int(info["catchment_truncated"].sum()),
        }
    )
    return ProjectScores(result, catch, inputs)


def analyze_geochem(project: Project) -> GeochemResult:
    """Score the project's RGS samples (catchment-aware when there is a DEM) and write the results
    to ``analysis/``."""
    scored = score_project(project, read_layer(project, "rgs", required=True))
    result, catch, inputs = scored.result, scored.catchments, scored.inputs
    samples, elements, loadings, summary = result.samples, result.elements, result.loadings, result.summary
    info = samples[[c for c in ("catchment_area_km2", "subcatchment_area_km2") if c in samples]]
    outputs = []
    if catch is not None:
        polys = catchment_polygons(catch)
        keep = [c for c in samples.columns if c in ("MASTER_ID", "multivariate_anomaly", "top_elements", "anomaly_d2") or c.startswith("idx_")]
        polys = polys.join(samples[keep].reset_index(drop=True), on="sample")
        polys = polys.join(info.reset_index(drop=True), on="sample")
        outputs.append(write_layer(project, polys.to_crs(project.config.crs), "rgs_catchments"))
    outputs += [
        write_layer(project, samples, "rgs_scores"),
        write_table(project, elements, "geochem_elements"),
        write_table(project, loadings.reset_index(names="element"), "geochem_pca_loadings"),
        write_json(project, summary, "geochem_summary"),
    ]
    log_outputs(
        project, "geochem", outputs, inputs,
        {
            "elements_continuous": summary["elements_continuous"],
            "elements_censored": summary["elements_censored"],
            "n_components": len(summary["pca_explained_variance"]),
            "anomaly_top_fraction": ANOMALY_TOP_FRACTION,
        },
    )  # fmt: skip
    result.outputs = outputs
    return result
