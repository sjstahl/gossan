from pathlib import Path

def patch(path_str: str, old: str, new: str) -> None:
    p = Path(path_str)
    text = p.read_text(encoding="utf-8")
    norm = text.replace("\r\n", "\n")
    assert old in norm, f"Could not find target block in {path_str}"
    p.write_text(norm.replace(old, new, 1), encoding="utf-8", newline="\n")
    print(f"Updated {path_str}")

# 1. src/gossan/core/licensing.py
patch(
    "src/gossan/core/licensing.py",
    """def input_licenses(records: list[ProvenanceRecord], datasets: list[str]) -> dict[str, str]:
    \"\"\"Licence of each named dataset, from its latest fetch or manual-ingest record (datasets
    with no such record are left out).\"\"\"
    out: dict[str, str] = {}
    for r in records:
        if r.dataset in datasets and r.operation in {"fetch", "manual_ingest"}:
            out[r.dataset] = r.license
    return out""",
    """def input_licenses(records: list[ProvenanceRecord], datasets: list[str]) -> dict[str, str]:
    \"\"\"Licence of each named dataset, from its latest fetch or manual-ingest record (datasets
    with no such record are left out).\"\"\"
    out: dict[str, str] = {}
    for r in records:
        if r.dataset in datasets and r.operation in {"fetch", "manual_ingest"}:
            out[r.dataset] = r.license
    return out


def is_open_dataset(records: list[ProvenanceRecord], dataset: str) -> bool:
    \"\"\"True when ``dataset`` has a fetch or manual-ingest record whose licence is neither Access Only
    nor internal-only, and its name is not an ARIS dataset.\"\"\"
    if ARIS_DATASET.match(dataset.lower()) or "aris" in dataset.lower():
        return False
    lic = input_licenses(records, [dataset]).get(dataset)
    if lic is None:
        return False
    low = lic.lower()
    return "access only" not in low and "internal use only" not in low""",
)

# 2. src/gossan/analyze/common.py
patch(
    "src/gossan/analyze/common.py",
    """from gossan.core.licensing import derived_license_from
from gossan.core.project import Project""",
    """from gossan.core.licensing import derived_license_from, is_open_dataset
from gossan.core.project import Project""",
)

patch(
    "src/gossan/analyze/common.py",
    """def read_layer(project: Project, name: str, *, required: bool = False) -> gpd.GeoDataFrame | None:
    \"\"\"A ``data.gpkg`` layer in the project CRS; None (or ``AnalysisError`` if ``required``) when
    the layer is absent or empty.\"\"\"
    if name in layer_names(project):""",
    """def read_layer(
    project: Project, name: str, *, required: bool = False, open_only: bool = False
) -> gpd.GeoDataFrame | None:
    \"\"\"A ``data.gpkg`` layer in the project CRS; None (or ``AnalysisError`` if ``required``) when
    the layer is absent, empty, or (with ``open_only``) carries no open licence.\"\"\"
    if open_only and not is_open_dataset(project.provenance.read(), name):
        if required:
            raise AnalysisError(f"layer {name} has a restricted or missing licence (cannot use with --open-only)")
        return None
    if name in layer_names(project):""",
)

# 3. src/gossan/analyze/targets.py
patch(
    "src/gossan/analyze/targets.py",
    """from gossan.analyze.drill_results import classify_holes
from gossan.analyze.occurrences import unit_name
from gossan.core.project import Project""",
    """from gossan.analyze.drill_results import classify_holes
from gossan.analyze.occurrences import unit_name
from gossan.core.licensing import is_open_dataset
from gossan.core.project import Project""",
)

patch(
    "src/gossan/analyze/targets.py",
    """def build_decisions(project: Project) -> TargetDecisionResult:
    path = project.root / TARGETS_FILE
    if not path.exists():
        raise AnalysisError("no target/targets.gpkg (run 'gossan target' first)")
    targets = gpd.read_file(path, layer=TARGETS_LAYER, engine="pyogrio").to_crs(project.config.crs)
    if targets.empty:
        raise AnalysisError("target/targets.gpkg holds no targets")
    extra = [c for c in ("knowledge_pctile", "knowledge_support") if c in targets]  # from the knowledge-driven score
    t = targets[["rank", "target_id", "area_ha", "mean_score", "max_score", "mean_spread", "n_known_occurrences", *extra, "geometry"]].copy()
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
            if near.empty or not idx_cols or near[idx_cols].isna().all().all():  # samples with no usable index (too few elements measured)
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
    minfile = read_layer(project, "minfile")""",
    """def build_decisions(project: Project, *, open_only: bool = False) -> TargetDecisionResult:
    path = project.root / TARGETS_FILE
    if not path.exists():
        raise AnalysisError("no target/targets.gpkg (run 'gossan target' first)")
    targets = gpd.read_file(path, layer=TARGETS_LAYER, engine="pyogrio").to_crs(project.config.crs)
    if targets.empty:
        raise AnalysisError("target/targets.gpkg holds no targets")
    records = project.provenance.read()
    extra = [c for c in ("knowledge_pctile", "knowledge_support") if c in targets]  # from the knowledge-driven score
    t = targets[["rank", "target_id", "area_ha", "mean_score", "max_score", "mean_spread", "n_known_occurrences", *extra, "geometry"]].copy()
    inputs: list[str] = []
    notes: list[list[str]] = [[] for _ in range(len(t))]

    # --- geology and structure ---------------------------------------------------------------
    bedrock = read_layer(project, "bedrock_geology", open_only=open_only)
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
    faults = read_layer(project, "faults", open_only=open_only)
    if faults is not None:
        inputs.append("faults")
        union = faults.geometry.union_all()
        t["dist_fault_m"] = t.geometry.distance(union).round(0)

    # --- geochemistry ------------------------------------------------------------------------
    scores_path = project.root / "analysis" / "analysis.gpkg"
    scores = None
    if scores_path.exists() and (not open_only or is_open_dataset(records, "rgs")):
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
            if near.empty or not idx_cols or near[idx_cols].isna().all().all():  # samples with no usable index (too few elements measured)
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
    tenure = read_layer(project, "mto_tenure", open_only=open_only)
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
    excluded_layers = [(name, read_layer(project, name, open_only=open_only)) for name in EXCLUDED_LAYERS]
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
    reserves = read_layer(project, "reserve_sites", open_only=open_only)
    if reserves is not None:
        inputs.append("reserve_sites")
        if "SITE_MINERAL_IND" in reserves:
            reserves = reserves[reserves["SITE_MINERAL_IND"].astype(str).str.upper().eq("Y")]
        t["mineral_reserve_share"] = _share(t, reserves) if len(reserves) else 0.0
    changes = _tenure_changes(project) if not open_only or is_open_dataset(records, "mto_tenure") else None
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
    drilled = None if open_only else _drill_evidence(project, t)
    if drilled is not None:
        inputs.append("aris_pdfs")  # the drill tables are digitized ARIS reports
        t["n_drill_holes"], t["drill_result"] = drilled["n"], drilled["text"]
        for i, note in enumerate(drilled["notes"]):
            if note:
                notes[i].append(note)
    grants = read_layer(project, "crown_grants", open_only=open_only)
    if grants is not None:
        inputs.append("crown_grants")
        if "LOT_STATUS" in grants:
            status = grants["LOT_STATUS"].astype(str).str.upper()
            grants = grants[status.str.contains("GRANTED") & ~status.str.contains("NOT")]
        t["crown_grant_share"] = _share(t, grants) if len(grants) else 0.0

    # --- access ----------------------------------------------------------------------------------
    roads = read_layer(project, "roads", open_only=open_only)
    if roads is not None:
        inputs.append("roads")
        nearest = gpd.sjoin_nearest(t[["geometry"]], roads[[c for c in ("ROAD_CLASS",) if c in roads] + ["geometry"]], distance_col="d")
        nearest = nearest[~nearest.index.duplicated(keep="first")]
        t["dist_road_m"] = nearest["d"].round(0)
        if "ROAD_CLASS" in nearest:
            t["nearest_road_class"] = nearest["ROAD_CLASS"]
    power = read_layer(project, "transmission_lines", open_only=open_only)
    if power is not None:
        inputs.append("transmission_lines")
        t["dist_power_m"] = t.geometry.distance(power.geometry.union_all()).round(0)

    # --- history ---------------------------------------------------------------------------------
    aris = read_layer(project, "aris_index", open_only=open_only)
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
        g = read_layer(project, layer, open_only=open_only)
        if g is not None:
            inputs.append(layer)
            t[col] = [len(near) for near in _near(t, g, HISTORY_RADIUS_M)]
    minfile = read_layer(project, "minfile", open_only=open_only)""",
)

patch(
    "src/gossan/analyze/targets.py",
    """    log_outputs(project, "targets", outputs, ["target", *inputs], {"geochem_radius_m": GEOCHEM_RADIUS_M, "history_radius_m": HISTORY_RADIUS_M})""",
    """    log_outputs(project, "targets", outputs, ["target", *inputs], {"geochem_radius_m": GEOCHEM_RADIUS_M, "history_radius_m": HISTORY_RADIUS_M, "open_only": open_only})""",
)

# 4. src/gossan/report/analysis_content.py
patch(
    "src/gossan/report/analysis_content.py",
    """from gossan.core.licensing import access_only_datasets
from gossan.core.project import Project""",
    """from gossan.analyze.targets import build_decisions
from gossan.core.licensing import access_only_datasets, is_open_dataset
from gossan.core.project import Project
from gossan.deliver.items import output_licence, restricts""",
)

patch(
    "src/gossan/report/analysis_content.py",
    """def load(project: Project) -> Loaded:
    a = project.root / "analysis"
    t = project.root / "target"
    cv = _json(t / "cv_report.json")
    importance = pd.read_csv(t / "feature_importance.csv") if (t / "feature_importance.csv").exists() else None
    decisions = read_csv(project, "target_decisions")""",
    """def load(project: Project, *, open_only: bool = False) -> Loaded:
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
        inventory = inventory if len(inventory) else None""",
)

patch(
    "src/gossan/report/analysis_content.py",
    """    return Loaded(
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
        _json(a / "tenure_summary.json"),
        read_csv(project, "tenure_changes"),
        read_csv(project, "tenure_watch"),
        _json(a / "drill_summary.json"),
        read_csv(project, "digitize_priorities"),
        _json(a / "digitize_priorities_summary.json"),
        _json(a / "spectral_summary.json"),
        _json(t / "knowledge_report.json"),
    )""",
    """    return Loaded(
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
    )""",
)

patch(
    "src/gossan/report/analysis_content.py",
    """def _sec_sources(project: Project, notices: list[str]) -> list[Block]:
    blocks: list[Block] = [Heading("Data sources and licences")]
    sources = data_sources(project.provenance.read())""",
    """def _sec_sources(project: Project, notices: list[str], *, open_only: bool = False) -> list[Block]:
    blocks: list[Block] = [Heading("Data sources and licences")]
    records = project.provenance.read()
    sources = data_sources(records)
    if open_only:
        sources = [r for r in sources if is_open_dataset(records, r.dataset)]""",
)

patch(
    "src/gossan/report/analysis_content.py",
    """def build_content(project: Project, assets: Path) -> AnalysisReport:
    \"\"\"Assemble the report's blocks; figures are written under ``assets``.\"\"\"
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
        "spectral": lambda: _sec_spectral(project, L, assets),
        "targets": lambda: _sec_targets(project, L, assets),
        "tenure": lambda: _sec_tenure(project, L, assets),
        "drilling": lambda: _sec_drilling(project, L, assets),
        "decisions": lambda: _sec_decisions(project, L, assets),
        "method": lambda: _sec_method(project, L),
        "sources": lambda: _sec_sources(project, notices),
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
    report.inputs = _inputs(records)""",
    """def build_content(project: Project, assets: Path, *, open_only: bool = False) -> AnalysisReport:
    \"\"\"Assemble the report's blocks; figures are written under ``assets``.\"\"\"
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
    report.inputs = [d for d in _inputs(records) if not open_only or is_open_dataset(records, d)]""",
)

# 5. src/gossan/report/analysis_render.py
patch(
    "src/gossan/report/analysis_render.py",
    """def build_analysis_report(project: Project, fmt: str = "both") -> list[Path]:
    \"\"\"Write ``outputs/analysis_report.pdf`` and/or ``.docx`` (``fmt``: pdf, docx or both) and log
    each to provenance. Figures are drawn under ``outputs/report_assets/analysis/``.\"\"\"
    if fmt not in {"pdf", "docx", "both"}:
        raise ValueError("fmt must be pdf, docx or both")
    outputs = project.outputs_dir
    assets = outputs / ASSETS_DIR / "analysis"
    assets.mkdir(parents=True, exist_ok=True)
    report = build_content(project, assets)""",
    """def build_analysis_report(project: Project, fmt: str = "both", *, open_only: bool = False) -> list[Path]:
    \"\"\"Write ``outputs/analysis_report.pdf`` and/or ``.docx`` (``fmt``: pdf, docx or both) and log
    each to provenance. Figures are drawn under ``outputs/report_assets/analysis/``.\"\"\"
    if fmt not in {"pdf", "docx", "both"}:
        raise ValueError("fmt must be pdf, docx or both")
    outputs = project.outputs_dir
    assets = outputs / ASSETS_DIR / "analysis"
    assets.mkdir(parents=True, exist_ok=True)
    report = build_content(project, assets, open_only=open_only)""",
)

patch(
    "src/gossan/report/analysis_render.py",
    """            parameters={
                "access_only_marking": report.marked,
                "access_only_datasets": report.restricted,
                "sections": report.sections,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                **extra,
            },""",
    """            parameters={
                "access_only_marking": report.marked,
                "access_only_datasets": report.restricted,
                "open_only": open_only,
                "sections": report.sections,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                **extra,
            },""",
)

# 6. src/gossan/deliver/package.py
patch(
    "src/gossan/deliver/package.py",
    """from gossan.deliver import webmap
from gossan.deliver.items import Item, Plan, plan""",
    """from gossan.analyze.targets import build_decisions
from gossan.deliver import webmap
from gossan.deliver.items import Item, Plan, _layers, output_licence, plan, restricts
from gossan.report.analysis_render import build_analysis_report""",
)

patch(
    "src/gossan/deliver/package.py",
    """def build_deliverable(project: Project, out: Path | None = None, *, include_internal: bool = False, make_zip: bool = False) -> DeliverResult:
    \"\"\"Build the package. Raises ``DeliverError`` when there is nothing to deliver or the folder is not ours to replace.\"\"\"
    the_plan: Plan = plan(project, include_internal=include_internal)""",
    """def _rebuild_open_outputs(project: Project) -> None:
    \"\"\"Rebuild the decision table and any existing analysis report from open-licensed layers only.\"\"\"
    root = project.root
    if "targets_top" in _layers(root / "target" / "targets.gpkg"):
        if restricts(output_licence(project.provenance.read(), "analysis/analysis.gpkg:target_decisions")) is not None or not (root / "analysis" / "analysis.gpkg").is_file():
            build_decisions(project, open_only=True)
    has_pdf = (project.outputs_dir / "analysis_report.pdf").is_file()
    has_docx = (project.outputs_dir / "analysis_report.docx").is_file()
    if has_pdf or has_docx:
        records = project.provenance.read()
        pdf_restricted = has_pdf and restricts(output_licence(records, "outputs/analysis_report.pdf")) is not None
        docx_restricted = has_docx and restricts(output_licence(records, "outputs/analysis_report.docx")) is not None
        if pdf_restricted or docx_restricted:
            fmt = "both" if (has_pdf and has_docx) else ("pdf" if has_pdf else "docx")
            build_analysis_report(project, fmt, open_only=True)


def build_deliverable(
    project: Project, out: Path | None = None, *, include_internal: bool = False, make_zip: bool = False, open_only: bool = False
) -> DeliverResult:
    \"\"\"Build the package. Raises ``DeliverError`` when there is nothing to deliver or the folder is not ours to replace.\"\"\"
    if open_only and include_internal:
        raise DeliverError("cannot combine --open-only and --include-internal")
    if open_only:
        _rebuild_open_outputs(project)
    the_plan: Plan = plan(project, include_internal=include_internal)""",
)

# 7. src/gossan/cli.py
patch(
    "src/gossan/cli.py",
    """    lead: Annotated[
        str | None,
        typer.Option(help="Feature this lead (tenure number from 'gossan leads run') in the PDF."),
    ] = None,
) -> None:""",
    """    lead: Annotated[
        str | None,
        typer.Option(help="Feature this lead (tenure number from 'gossan leads run') in the PDF."),
    ] = None,
    open_only: Annotated[
        bool,
        typer.Option("--open-only", help="For --kind analysis: build from open-licensed layers only, omitting ARIS and other Access Only inputs so the report can go in a client package."),
    ] = False,
) -> None:""",
)

patch(
    "src/gossan/cli.py",
    """        if kind == "analysis":
            for path in build_analysis_report(proj, fmt):
                typer.echo(f"Wrote {path}")
            return""",
    """        if kind == "analysis":
            for path in build_analysis_report(proj, fmt, open_only=open_only):
                typer.echo(f"Wrote {path}")
            return""",
)

patch(
    "src/gossan/cli.py",
    """    include_internal: Annotated[
        bool, typer.Option("--include-internal", help="Also package Access Only and other restricted items (for use inside the company only; the package is marked INTERNAL).")
    ] = False,
    zip_: Annotated[bool, typer.Option("--zip", help="Also write <folder>.zip.")] = False,
    skip_qa: Annotated[bool, typer.Option("--skip-qa", help="Build even if 'gossan qa' finds errors (invalid claim geometry, data outside the AOI, a layer with no CRS).")] = False,
) -> None:""",
    """    include_internal: Annotated[
        bool, typer.Option("--include-internal", help="Also package Access Only and other restricted items (for use inside the company only; the package is marked INTERNAL).")
    ] = False,
    open_only: Annotated[
        bool, typer.Option("--open-only", help="Rebuild the decision table and analysis report from open-licensed layers only (omitting ARIS and other Access Only inputs) so they can go in a client package.")
    ] = False,
    zip_: Annotated[bool, typer.Option("--zip", help="Also write <folder>.zip.")] = False,
    skip_qa: Annotated[bool, typer.Option("--skip-qa", help="Build even if 'gossan qa' finds errors (invalid claim geometry, data outside the AOI, a layer with no CRS).")] = False,
) -> None:""",
)

patch(
    "src/gossan/cli.py",
    """    try:
        result = build_deliverable(proj, out, include_internal=include_internal, make_zip=zip_)""",
    """    try:
        result = build_deliverable(proj, out, include_internal=include_internal, make_zip=zip_, open_only=open_only)""",
)

# 8. tests/test_deliver.py
patch(
    "tests/test_deliver.py",
    """def test_a_report_without_restricted_data_goes_with_the_package(project):""",
    """def test_open_only_rebuilds_decisions_and_report_from_open_layers_for_client_delivery(project):
    from gossan.harvest.bc_provincial.wfs import ACCESS_ONLY_BC
    from gossan.qa.checks import QaResult, check_package

    aris = gpd.GeoDataFrame(
        {"ARIS_NUMBER": [12345], "REPORT_YEAR": [2020], "MERGE_SPENT_2002": [50000.0]},
        geometry=[Point(693200, 5618500)],
        crs=CRS,
    )
    aris.to_file(project.data_gpkg, layer="aris_index", driver="GPKG", engine="pyogrio")
    project.provenance.log(dataset="aris_index", source="BC Data Catalogue", identifier="aris_index", url=None, license=ACCESS_ONLY_BC, operation="fetch")
    STEPS["targets"](project)
    build_analysis_report(project, "pdf")

    blocked = build_deliverable(project)
    assert {i.key for i in blocked.withheld} == {"target_decisions", "report_pdf"}

    clean = build_deliverable(project, open_only=True)
    assert not clean.withheld and not clean.internal
    assert "target_decisions" in gpkg_layers(clean.folder)
    assert (clean.folder / "demo_analysis_report.pdf").is_file()
    decisions_df = gpd.read_file(clean.folder / "data" / "deliverable.gpkg", layer="target_decisions", engine="pyogrio")
    assert "n_aris_reports" not in decisions_df.columns and "ground_status" in decisions_df.columns
    qa_res = QaResult()
    check_package(project, qa_res)
    assert qa_res.passed


def test_a_report_without_restricted_data_goes_with_the_package(project):""",
)
