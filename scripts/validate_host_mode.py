"""Does the empirical host layer help the knowledge score? A validation run, not a test.

For each region this copies a project already made by ``scripts/validate_regions.py`` (harvested layers and rasters,
not the raw downloads or earlier outputs, so the original is untouched), then runs the prospectivity model twice on the
same layers with the same settings and seed: with ``knowledge.host_mode: named`` (no host units named, so the host layer
is absent: the knowledge score as validated in October 2026) and with ``host_mode: empirical`` (the host layer learned
from the occurrences and relearned inside every spatial fold). The forest is the same in both, which the script checks.
It records the knowledge score's held-out figures per fold, so the two runs can be compared fold by fold, and the host
layer's own held-out AUC. Results go to ``<workdir>/host_mode_<region>.json`` and a table to ``host_mode_report.md``.

It needs no network. Time is the cost: each region runs the model with cross-validation twice.

    python scripts/validate_host_mode.py --source C:\\gossan-validation --workdir C:\\gossan-hostmode
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
from validate_regions import REGIONS, Timer, run_model

from gossan.core.project import Project

SKIP = {"raw", "target", "analysis", "outputs"}
METRICS = ("auc", "auc_effort_matched", "capture_top5")


def copy_project(source: Path, dest: Path) -> Project:
    """Copy the project without its raw downloads and earlier outputs."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for item in source.iterdir():
        if item.name in SKIP:
            continue
        if item.is_dir():
            shutil.copytree(item, dest / item.name)
        else:
            shutil.copy2(item, dest / item.name)
    return Project(dest)


def paired(base: list[float], emp: list[float]) -> dict | None:
    """Mean and spread of the fold-by-fold difference (empirical minus named), and how many folds improved."""
    n = min(len(base), len(emp))
    if n < 2:
        return None
    diff = np.array(emp[:n]) - np.array(base[:n])
    return {"mean": float(diff.mean()), "std": float(diff.std(ddof=1)), "n_folds": n, "n_better": int((diff > 0).sum())}


def compare(runs: dict[str, dict]) -> dict:
    base, emp = runs["named"], runs["empirical"]
    out: dict = {}
    bk, ek = (base.get("knowledge") or {}).get("checks") or {}, (emp.get("knowledge") or {}).get("checks") or {}
    for metric in METRICS:
        out[metric] = {
            "named": bk.get(f"{metric}_fold_mean"),
            "empirical": ek.get(f"{metric}_fold_mean"),
            "paired": paired(bk.get(f"{metric}_by_fold") or [], ek.get(f"{metric}_by_fold") or []),
        }
    out["host_layer_auc_held_out"] = ek.get("host_layer_auc_fold_mean")
    host = next((layer for layer in (emp.get("knowledge") or {}).get("layers", []) if layer["layer"] == "host"), None)
    out["host_layer_auc_in_sample"] = None if host is None else host.get("auc")
    out["host_layer_present"] = host is not None
    fb, fe = (base.get("cv") or {}), (emp.get("cv") or {})
    out["forest_identical"] = bool(fb) and all(fb.get(k) == fe.get(k) for k in ("auc_mean", "auc_effort_matched_mean", "capture_top5_mean"))
    out["forest"] = {k: fb.get(k) for k in ("auc_mean", "auc_effort_matched_mean", "capture_top5_mean")}
    return out


def one_region(source: Path, workdir: Path, name: str) -> dict:
    record: dict = {"region": name, "box_deg": REGIONS[name]}
    project = copy_project(source / name, workdir / name)
    record["crs"] = project.config.crs
    record["occurrence_class_setting"] = project.config.target.occurrence_class
    timer = Timer()
    runs = {}
    for mode in ("named", "empirical"):
        knowledge = project.config.target.knowledge.model_copy(update={"host_mode": mode})
        runs[mode] = run_model(project, mode, timer, knowledge=knowledge)
        if "error" in runs[mode]:
            record["error"] = f"{mode}: {runs[mode]['error']}"
    record["runs"] = runs
    if "error" not in record:
        record["comparison"] = compare(runs)
        record["n_positives"] = runs["named"]["n_positives"]
        record["n_cells"] = runs["named"]["n_cells"]
    record["seconds"] = timer.seconds
    return record


def f(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def table(records: list[dict]) -> str:
    lines = [
        "| Region | Occurrences | Knowledge AUC, named / empirical (paired diff, folds better) | Effort-matched AUC, named / empirical (paired diff, folds better) | Top-5% capture, named / empirical | Host layer AUC, held out / in-sample | Forest unchanged |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in records:
        c = r["comparison"]

        def cell(m: str, pct: bool = False, c: dict = c) -> str:
            row = c[m]
            p = row["paired"]
            tail = "" if p is None else f" ({p['mean']:+.3f} ± {p['std']:.3f}, {p['n_better']}/{p['n_folds']})"
            fmt = (lambda v: "n/a" if v is None else f"{v:.0%}") if pct else f
            return f"{fmt(row['named'])} / {fmt(row['empirical'])}" + ("" if pct else tail)

        lines.append(
            f"| {r['region']} | {r['n_positives']} | {cell('auc')} | {cell('auc_effort_matched')} | {cell('capture_top5', True)} | "
            f"{f(c['host_layer_auc_held_out'])} / {f(c['host_layer_auc_in_sample'])} | {'yes' if c['forest_identical'] else 'NO'} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, required=True, help="Folder holding the validate_regions projects.")
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--regions", default=",".join(REGIONS))
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    records = []
    for name in (n.strip() for n in args.regions.split(",") if n.strip()):
        if name not in REGIONS:
            print(f"unknown region {name!r}; choose from {', '.join(REGIONS)}", file=sys.stderr)
            return 2
        print(f"== {name}", flush=True)
        try:
            record = one_region(args.source, args.workdir, name)
        except Exception as e:  # noqa: BLE001 (a failing region is a finding, not a reason to stop)
            record = {"region": name, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}
        (args.workdir / f"host_mode_{name}.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        records.append(record)
        good = [r for r in records if "error" not in r]
        print(table(good) if good else record.get("error"), flush=True)
    good = [r for r in records if "error" not in r]
    (args.workdir / "host_mode_report.md").write_text(table(good) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
