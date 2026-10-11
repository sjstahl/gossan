"""The whole job in one go: harvest, model, analyze, report.

``gossan run`` calls ``run_pipeline``. The steps depend on each other in a fixed order, which is the
reason for a single command: ``target`` needs the harvested layers, ``analyze``'s decision table
needs the targets, and the report quotes both, so running them out of order silently gives an
older or emptier result.

``STEPS`` is that order. A step that cannot run does not stop the others, because every later step
degrades gracefully (a report says which command would supply a missing part), so a long run is
not lost to one unavailable data source:

* ``ok``       the step did everything asked of it;
* ``partial``  it did some of it (some sources unreachable, some analysis steps without input);
* ``skipped``  it could not run for a reason that is not an error (for example, too few known
               occurrences to train a model); the reason is given;
* ``failed``   it raised an unexpected error.

The commands print through a ``Reporter`` so the pipeline itself never writes to the terminal and can
be tested, or scheduled, without a console.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from gossan.analyze import STEPS as ANALYSIS_STEPS
from gossan.analyze import AnalysisError
from gossan.core.config import TargetConfig
from gossan.core.project import Project
from gossan.harvest import CONNECTORS, default_sources
from gossan.harvest.national.nrcan_geophysics import ManualDownloadRequired
from gossan.harvest.satellite.stac import SatelliteAuthError
from gossan.report.analysis_render import build_analysis_report
from gossan.target.run import TargetError, TargetResult, run_target

STEPS = ("harvest", "target", "analyze", "report")
OK, PARTIAL, SKIPPED, FAILED = "ok", "partial", "skipped", "failed"


class Reporter:
    """Where the pipeline's messages go. The base class discards them."""

    def say(self, level: str, text: str) -> None:
        """``level``: info (normal output), note or warn (yellow, to stderr), error (red, to stderr),
        hint (plain, to stderr)."""

    def step(self, name: str) -> None:
        """A pipeline step is starting."""

    def target_result(self, result: TargetResult) -> None:
        """The prospectivity model finished."""

    def analysis_result(self, name: str, result: object) -> None:
        """One analysis step finished."""


@dataclass
class StepOutcome:
    step: str
    status: str
    detail: str = ""
    seconds: float = 0.0


@dataclass
class HarvestResult:
    counts: dict[str, int] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)


# --- the steps ---------------------------------------------------------------------------------


def parse_names(text: str | None, allowed: list[str] | tuple[str, ...], what: str) -> list[str]:
    """Comma-separated ``text`` as a list of names in the order ``allowed`` defines; raises
    ``ValueError`` naming anything unknown. None or empty means all of ``allowed``."""
    names = [s.strip() for s in text.split(",") if s.strip()] if text else list(allowed)
    unknown = [n for n in names if n not in allowed]
    if unknown:
        raise ValueError(f"unknown {what}(s): {', '.join(unknown)}")
    return names


def harvest_sources(proj: Project, names: list[str], reporter: Reporter, *, limit_scenes: int = 0) -> HarvestResult:
    """Run each named connector; one that fails does not stop the others. Messages go to ``reporter``
    exactly as ``gossan harvest`` prints them."""
    result = HarvestResult()
    for n in names:
        other = getattr(CONNECTORS[n], "jurisdiction", "any")
        if other not in ("any", proj.config.jurisdiction):
            reporter.say("note", f"Note: {n} is a {other} source and this is a {proj.config.jurisdiction} project.")
        connector = CONNECTORS[n](proj)
        if hasattr(connector, "limit_scenes"):
            connector.limit_scenes = limit_scenes
        try:
            count = connector.run()
        except (ManualDownloadRequired, SatelliteAuthError) as e:
            reporter.say("warn", f"{n}: {e}")
            result.failures[n] = str(e).splitlines()[0] if str(e) else type(e).__name__
        except OSError as e:  # network failures (URLError is an OSError)
            reporter.say("error", f"Error: {n}: {e}")
            reporter.say("hint", "Fallback: download the data manually and use 'gossan ingest'.")
            result.failures[n] = str(e)
        else:
            result.counts[n] = count
            reporter.say("info", f"{n}: {count} {connector.output_description}")
            if hasattr(connector, "report"):
                reporter.say("info", f"  {connector.report.text(n)}")
    return result


def run_analysis_steps(proj: Project, names: list[str], reporter: Reporter) -> tuple[list[str], dict[str, str]]:
    """Run the named ``gossan analyze`` steps in their fixed order; a step whose input is missing is
    skipped with its reason. Returns (completed step names, {skipped step: reason})."""
    done: list[str] = []
    skipped: dict[str, str] = {}
    for name in [n for n in ANALYSIS_STEPS if n in names]:
        try:
            result = ANALYSIS_STEPS[name](proj)
        except AnalysisError as e:
            reporter.say("warn", f"{name}: skipped: {e}")
            skipped[name] = str(e)
            continue
        done.append(name)
        reporter.analysis_result(name, result)
    return done, skipped


def _harvest_step(proj: Project, sources: list[str] | None, reporter: Reporter) -> StepOutcome:
    names = sources or default_sources(proj.config.jurisdiction)
    result = harvest_sources(proj, names, reporter)
    if not result.failures:
        return StepOutcome("harvest", OK, f"{len(result.counts)} source(s)")
    status = PARTIAL if result.counts else FAILED
    return StepOutcome(
        "harvest", status, f"{len(result.counts)} of {len(names)} source(s); unavailable: {', '.join(result.failures)}"
    )


def _target_step(proj: Project, cfg: TargetConfig, cross_validate: bool, reporter: Reporter) -> StepOutcome:
    try:
        result = run_target(proj, cfg, cross_validate=cross_validate)
    except TargetError as e:
        reporter.say("warn", f"target: skipped: {e}")
        return StepOutcome("target", SKIPPED, str(e))
    reporter.target_result(result)
    detail = f"{result.n_targets} target(s)"
    cv = result.cv or {}
    if cv.get("auc_effort_matched_mean") is not None:
        detail += f"; effort-matched AUC {cv['auc_effort_matched_mean']:.2f}"
    elif cv.get("auc_mean") is not None:  # no exploration-effort layers to match against
        detail += f"; AUC {cv['auc_mean']:.2f}"
    return StepOutcome("target", OK, detail)


def _analyze_step(proj: Project, reporter: Reporter) -> StepOutcome:
    done, skipped = run_analysis_steps(proj, list(ANALYSIS_STEPS), reporter)
    detail = f"{len(done)} of {len(ANALYSIS_STEPS)} step(s)"
    if skipped:
        detail += "; skipped: " + ", ".join(skipped)
        return StepOutcome("analyze", PARTIAL if done else SKIPPED, detail)
    return StepOutcome("analyze", OK, detail)


def _report_step(proj: Project, report_format: str, reporter: Reporter) -> StepOutcome:
    try:
        paths = build_analysis_report(proj, report_format)
    except Exception as e:  # noqa: BLE001 (a late failure must not hide what the earlier steps did)
        reporter.say("error", f"Error: report: {type(e).__name__}: {e}")
        return StepOutcome("report", FAILED, f"{type(e).__name__}: {e}")
    for path in paths:
        reporter.say("info", f"Wrote {path}")
    return StepOutcome("report", OK, ", ".join(p.name for p in paths))


def run_pipeline(
    proj: Project,
    steps: list[str] | None = None,
    *,
    sources: list[str] | None = None,
    target_cfg: TargetConfig | None = None,
    cross_validate: bool = True,
    report_format: str = "both",
    reporter: Reporter | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> list[StepOutcome]:
    """Run ``steps`` (default: all of ``STEPS``, always in that order) and return one outcome each.

    ``sources`` limits harvest to those connectors (default: the jurisdiction's defaults);
    ``target_cfg`` defaults to the project's ``target:`` settings. Logs a ``run_pipeline`` record
    to provenance with every step's status and duration.
    """
    reporter = reporter or Reporter()
    wanted = [s for s in STEPS if s in (steps or STEPS)]
    cfg = target_cfg or proj.config.target
    outcomes: list[StepOutcome] = []
    for name in wanted:
        reporter.step(name)
        started = clock()
        if name == "harvest":
            outcome = _harvest_step(proj, sources, reporter)
        elif name == "target":
            outcome = _target_step(proj, cfg, cross_validate, reporter)
        elif name == "analyze":
            outcome = _analyze_step(proj, reporter)
        else:
            outcome = _report_step(proj, report_format, reporter)
        outcome.seconds = clock() - started
        outcomes.append(outcome)
    proj.provenance.log(
        dataset="pipeline_run",
        source="gossan",
        identifier=proj.config.name,
        url=None,
        license="N/A (run log)",
        operation="run_pipeline",
        parameters={
            "steps": [{"step": o.step, "status": o.status, "detail": o.detail, "seconds": round(o.seconds, 1)} for o in outcomes],
            "sources": sources,
            "cross_validate": cross_validate,
            "report_format": report_format,
        },
    )
    return outcomes


def summary_lines(outcomes: list[StepOutcome]) -> list[str]:
    """A fixed-width table of the run: step, status, time and what happened."""
    lines = [f"{'Step':<9} {'Status':<8} {'Time':>7}  Detail"]
    for o in outcomes:
        lines.append(f"{o.step:<9} {o.status:<8} {o.seconds:>6.0f}s  {o.detail}")
    return lines


def all_ok(outcomes: list[StepOutcome]) -> bool:
    return all(o.status == OK for o in outcomes)


def project_exists(path: Path) -> bool:
    return (path / "project.yaml").is_file()
