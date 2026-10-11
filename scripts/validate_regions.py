"""Run Gossan on several British Columbia areas and tabulate how it behaves: validation, not a test.

For each region (a lon/lat box) this makes a project under ``--workdir``, harvests the BC sources, runs the
prospectivity model with spatial cross-validation (and a metallic-only variant), runs the analysis steps,
and records per region: area and cell count, what each harvest source returned and how long it took, how
long each step took and the peak memory, the model's cross-validation figures, the knowledge-driven score's
checks and the figures that say how far the model can be trusted. Results go to
``<workdir>/validation_<region>.json`` and a table to ``<workdir>/validation_report.md``.

It needs the network (DataBC and NRCan services) and takes tens of minutes per region. Re-runs reuse a
project's harvested layers; ``--fresh`` starts the project over. Not part of the test suite.

    python scripts/validate_regions.py --workdir C:\\gossan-validation --regions boundary,stewart
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import sys
import time
import traceback
from ctypes import wintypes
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

from gossan.analyze import STEPS as ANALYSIS_STEPS
from gossan.analyze import AnalysisError
from gossan.core.project import Project
from gossan.harvest import default_sources
from gossan.pipeline import Reporter, harvest_sources
from gossan.target.run import TargetError, run_target

#: name -> (west, south, east, north) in degrees. Different terranes, each roughly 5,000 to 12,000 km2.
REGIONS: dict[str, tuple[float, float, float, float]] = {
    "boundary": (-118.9, 49.0, -117.6, 49.7),  # Greenwood and Rossland: skarn, vein and porphyry Cu-Au
    "kootenay_wide": (-116.6, 49.3, -114.9, 50.5),  # East Kootenay and Purcells: sediment-hosted Zn-Pb, veins
    "quesnel": (-122.4, 52.3, -121.1, 53.1),  # Cariboo and Quesnel Trough: alkalic porphyry Cu-Au
    "stewart": (-130.5, 55.7, -129.3, 56.6),  # Golden Triangle: porphyry Cu-Au and epithermal Au-Ag
}


def peak_memory_mb() -> float | None:
    """Peak working set of this process in MB (Windows; None elsewhere)."""
    if sys.platform != "win32":
        return None

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t)] + [
            (name, ctypes.c_size_t) for name in ("WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")
        ]  # fmt: skip

    c = Counters()
    c.cb = ctypes.sizeof(c)
    kernel, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    kernel.GetCurrentProcess.restype = wintypes.HANDLE  # a 64-bit handle: the default int return would truncate it
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    ok = psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return round(c.PeakWorkingSetSize / 1e6) if ok else None


class Timer:
    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}

    @contextlib.contextmanager
    def step(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] = round(time.perf_counter() - started, 1)


class Quiet(Reporter):
    def __init__(self) -> None:
        self.lines: list[str] = []

    def say(self, level: str, text: str) -> None:
        self.lines.append(f"{level}: {text}")


def make_project(workdir: Path, name: str, box_deg: tuple[float, float, float, float], fresh: bool) -> Project:
    from typer.testing import CliRunner

    from gossan.cli import app

    root = workdir / name
    if fresh and root.exists():
        import shutil

        shutil.rmtree(root)
    if not root.exists():
        aoi = workdir / f"{name}_aoi.gpkg"
        gpd.GeoDataFrame({"id": [1]}, geometry=[box(*box_deg)], crs=4326).to_file(aoi, driver="GPKG")
        with contextlib.chdir(workdir):  # `gossan new` makes the project folder in the current directory
            result = CliRunner().invoke(app, ["new", name, "--aoi", str(aoi)])
        if result.exit_code != 0:
            raise RuntimeError(f"gossan new failed: {result.output}")
    return Project(root)


def cv_figures(cv: dict | None) -> dict | None:
    if not cv or not cv.get("folds"):
        return None
    keys = ("n_folds_scored", "auc_mean", "auc_std", "auc_effort_matched_mean", "auc_effort_matched_std", "capture_top5_mean", "capture_top5_std", "auc_vs_barren_mean", "n_barren_cells", "block_size_m", "buffer_m")
    return {k: cv.get(k) for k in keys}


def run_model(project: Project, label: str, timer: Timer, **overrides) -> dict:
    cfg = project.config.target.model_copy(update=overrides)
    out: dict = {"label": label, "overrides": overrides}
    try:
        with timer.step(f"target_{label}"):
            result = run_target(project, cfg, cross_validate=True)
    except TargetError as e:
        out["error"] = str(e)
        return out
    out.update(
        n_cells=result.grid.n_cells, resolution_m=result.grid.resolution, n_positives=len(result.stack.positives), n_occurrences_in_layer=result.stack.n_occurrences_total,
        features=result.stack.model_names, features_skipped=result.stack.skipped, cv=cv_figures(result.cv), n_targets=result.n_targets,
        knowledge=None if result.knowledge is None else {k: v for k, v in result.knowledge.items() if k in ("available", "layers", "checks", "rank_correlation_with_forest", "top_overlap_with_forest", "n_knowledge_targets", "reason")},
        top_importances=sorted(result.importances.items(), key=lambda kv: -kv[1])[:6],
    )  # fmt: skip
    return out


def validate(workdir: Path, name: str, box_deg: tuple[float, float, float, float], fresh: bool, skip_harvest: bool) -> dict:
    timer = Timer()
    record: dict = {"region": name, "box_deg": box_deg}
    project = make_project(workdir, name, box_deg, fresh)
    record["area_km2"] = round(float(project.read_aoi().area.sum()) / 1e6)
    record["crs"] = project.config.crs
    reporter = Quiet()
    if not skip_harvest:
        with timer.step("harvest"):
            harvested = harvest_sources(project, default_sources(project.config.jurisdiction), reporter)
        record["harvest"] = {"counts": harvested.counts, "failures": harvested.failures}
    record["peak_mb_after_harvest"] = peak_memory_mb()
    record["models"] = [run_model(project, "default", timer)]
    record["models"].append(run_model(project, "metallic", timer, occurrence_class="metallic"))
    record["peak_mb_after_models"] = peak_memory_mb()
    analysis: dict[str, str] = {}
    with timer.step("analyze"):
        for step, func in ANALYSIS_STEPS.items():
            try:
                func(project)
                analysis[step] = "ok"
            except AnalysisError as e:
                analysis[step] = f"skipped: {e}"
            except Exception as e:  # noqa: BLE001 (a failing step is a finding, not a reason to stop)
                analysis[step] = f"FAILED {type(e).__name__}: {e}"
    record["analysis"] = analysis
    record["seconds"] = timer.seconds
    record["peak_mb"] = peak_memory_mb()
    record["pipeline_notes"] = reporter.lines[:40]
    return record


def fmt(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def table(records: list[dict]) -> str:
    lines = [
        "| Region | km2 | Cells | Occurrences (metallic) | Features | CV AUC | Effort-matched AUC | Top-5% capture | Knowledge AUC | Seconds (harvest / models / analyze) | Peak MB |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        d = next((m for m in r["models"] if m["label"] == "default" and "error" not in m), None)
        m = next((m for m in r["models"] if m["label"] == "metallic" and "error" not in m), None)
        cv = (d or {}).get("cv") or {}
        kn = ((d or {}).get("knowledge") or {}).get("checks") or {}
        s = r["seconds"]
        lines.append(
            f"| {r['region']} | {r['area_km2']:,} | {(d or {}).get('n_cells', 'n/a')} | {(d or {}).get('n_positives', 'n/a')} ({(m or {}).get('n_positives', 'n/a')}) | {len((d or {}).get('features', []))} | "
            f"{fmt(cv.get('auc_mean'))} ± {fmt(cv.get('auc_std'))} | {fmt(cv.get('auc_effort_matched_mean'))} | {fmt(cv.get('capture_top5_mean'))} | {fmt(kn.get('auc_fold_mean'))} | "
            f"{s.get('harvest', 'n/a')} / {round(s.get('target_default', 0) + s.get('target_metallic', 0))} / {s.get('analyze', 'n/a')} | {r.get('peak_mb')} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--regions", default=",".join(REGIONS), help="Comma-separated names from REGIONS.")
    parser.add_argument("--fresh", action="store_true", help="Delete and recreate each project.")
    parser.add_argument("--skip-harvest", action="store_true", help="Use the layers already in each project.")
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    records = []
    for name in (n.strip() for n in args.regions.split(",") if n.strip()):
        if name not in REGIONS:
            print(f"unknown region {name!r}; choose from {', '.join(REGIONS)}", file=sys.stderr)
            return 2
        print(f"== {name}", flush=True)
        try:
            record = validate(args.workdir, name, REGIONS[name], args.fresh, args.skip_harvest)
        except Exception as e:  # noqa: BLE001
            record = {"region": name, "area_km2": 0, "models": [], "seconds": {}, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}
        (args.workdir / f"validation_{name}.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        records.append(record)
        print(table([r for r in records if "error" not in r]) if any("error" not in r for r in records) else record.get("error"), flush=True)
    good = [r for r in records if "error" not in r]
    (args.workdir / "validation_report.md").write_text(table(good) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
