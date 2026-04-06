"""MLflow-based evaluators for Valravn success criteria SC-002 through SC-006."""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import mlflow

from valravn.models.report import FindingsReport

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "valravn-evaluation"

SUITES = {
    "anomaly-detection",
    "citation-coverage",
    "evidence-integrity",
    "self-correction",
    "report-completeness",
    "all",
}


# ---------------------------------------------------------------------------
# Individual evaluators
# ---------------------------------------------------------------------------


def _eval_anomaly_detection(report: FindingsReport) -> bool:
    """SC-002: report contains at least one Anomaly entry."""
    return len(report.anomalies) > 0


def _eval_citation_coverage(report: FindingsReport) -> bool:
    """SC-004: every Conclusion cites at least one ToolInvocationRecord.

    The model layer already enforces non-empty supporting_invocation_ids via a
    field_validator, so any Conclusion that made it into the report is guaranteed
    to be cited.  A report with *no* conclusions passes vacuously (nothing to
    violate the invariant).
    """
    for conclusion in report.conclusions:
        if not conclusion.supporting_invocation_ids:
            return False
    return True


def _eval_evidence_integrity(report: FindingsReport) -> bool:
    """SC-005: no evidence file was modified — each path must exist and be read-only."""
    for ref in report.evidence_refs:
        p = Path(ref)
        if not p.exists():
            print(
                f"[SC-005 WARNING] Evidence path not found at evaluation time: {p}",
                file=sys.stderr,
            )
            return False
        mode = p.stat().st_mode
        # Writable by owner, group, or other → integrity violation
        if mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            return False
    return True


def _eval_self_correction(report: FindingsReport) -> bool:
    """SC-003: when tool failures occurred, at least one retry with modified args
    must have been attempted (i.e. a SelfCorrectionEvent recorded).

    If there were no tool failures the criterion is vacuously satisfied.
    """
    if not report.tool_failures:
        return True
    return len(report.self_corrections) > 0


def _eval_report_completeness(report: FindingsReport) -> bool:
    """SC-006: report contains prompt, evidence_refs, generated_at_utc, and at
    least one populated section (conclusions or anomalies).
    """
    if not report.prompt:
        return False
    if not report.evidence_refs:
        return False
    if report.generated_at_utc is None:
        return False
    tz = report.generated_at_utc.tzinfo
    if tz is None or report.generated_at_utc.utcoffset().total_seconds() != 0:
        return False
    if not report.conclusions and not report.anomalies:
        return False
    return True


# ---------------------------------------------------------------------------
# Suite dispatcher
# ---------------------------------------------------------------------------

_SUITE_MAP: dict[str, tuple[str, callable]] = {
    "anomaly-detection": ("sc_002_anomaly_detection", _eval_anomaly_detection),
    "citation-coverage": ("sc_004_citation_coverage", _eval_citation_coverage),
    "evidence-integrity": ("sc_005_evidence_integrity", _eval_evidence_integrity),
    "self-correction": ("sc_003_self_correction", _eval_self_correction),
    "report-completeness": ("sc_006_report_completeness", _eval_report_completeness),
}


def evaluate_report(report_path: Path, suite: str = "all") -> dict[str, bool]:
    """Load report JSON, run requested suite, log to MLflow.

    Parameters
    ----------
    report_path:
        Path to a ``FindingsReport`` JSON file produced by Valravn.
    suite:
        One of ``anomaly-detection``, ``citation-coverage``,
        ``evidence-integrity``, ``self-correction``, ``report-completeness``,
        or ``all``.

    Returns
    -------
    dict[str, bool]
        Mapping of metric name → pass/fail for every evaluator that ran.
    """
    if suite not in SUITES:
        raise ValueError(f"Unknown suite {suite!r}. Choose from: {sorted(SUITES)}")

    report_data = json.loads(report_path.read_text())
    report = FindingsReport.model_validate(report_data)

    suites_to_run: dict[str, tuple[str, callable]]
    if suite == "all":
        suites_to_run = _SUITE_MAP
    else:
        suites_to_run = {suite: _SUITE_MAP[suite]}

    results: dict[str, bool] = {}

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"eval_{report_path.stem}"):
        mlflow.log_param("report_path", str(report_path))
        mlflow.log_param("suite", suite)

        for suite_key, (metric_name, evaluator_fn) in suites_to_run.items():
            passed = evaluator_fn(report)
            results[metric_name] = passed
            mlflow.log_metric(metric_name, 1.0 if passed else 0.0)

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Valravn SC evaluators against a FindingsReport JSON."
    )
    parser.add_argument(
        "--suite",
        default="all",
        choices=sorted(SUITES),
        help="Evaluation suite to run (default: all)",
    )
    parser.add_argument(
        "--report",
        required=True,
        type=Path,
        metavar="REPORT_JSON",
        help="Path to the FindingsReport JSON file",
    )
    args = parser.parse_args()

    results = evaluate_report(args.report, args.suite)
    for k, v in results.items():
        print(f"{'PASS' if v else 'FAIL'} {k}")
