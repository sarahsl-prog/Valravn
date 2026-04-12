"""Unit tests for MLflow-based evaluators (SC-002 through SC-006) and golden dataset."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from valravn.models.records import Anomaly, AnomalyResponseAction
from valravn.models.report import (
    Conclusion,
    FindingsReport,
    SelfCorrectionEvent,
    ToolFailureRecord,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_report(**overrides) -> FindingsReport:
    """Return a minimal valid FindingsReport, applying any overrides."""
    defaults = dict(
        task_id="test-id",
        prompt="test prompt",
        evidence_refs=["/mnt/evidence/test.raw"],
        conclusions=[],
        anomalies=[],
        tool_failures=[],
        self_corrections=[],
        investigation_plan_path=Path("/tmp/plan.json"),
    )
    defaults.update(overrides)
    return FindingsReport(**defaults)


def _save_report(report: FindingsReport, tmp_path: Path) -> Path:
    """Serialise *report* to a JSON file inside *tmp_path* and return the path."""
    p = tmp_path / "report.json"
    p.write_text(report.model_dump_json())
    return p


# Patch target — mlflow calls inside the evaluators module.
_MLFLOW_MODULE = "valravn.evaluation.evaluators.mlflow"


# ---------------------------------------------------------------------------
# SC-004 citation coverage
# ---------------------------------------------------------------------------


def test_citation_coverage_passes(tmp_path):
    """A report whose conclusions all cite at least one invocation passes SC-004."""
    conclusion = Conclusion(
        statement="Lateral movement detected",
        supporting_invocation_ids=["inv-abc"],
        confidence="high",
    )
    report = _minimal_report(conclusions=[conclusion])
    report_path = _save_report(report, tmp_path)

    with patch(_MLFLOW_MODULE) as mock_mlflow:
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        from valravn.evaluation.evaluators import evaluate_report

        results = evaluate_report(report_path, suite="citation-coverage")

    assert results["sc_004_citation_coverage"] is True


def test_citation_coverage_no_conclusions_passes(tmp_path):
    """SC-004 is vacuously satisfied when there are no conclusions at all.

    Empty supporting_invocation_ids is rejected at the model layer (field_validator),
    so the only way a valid report has no citations is if it has no conclusions.
    """
    report = _minimal_report(conclusions=[])
    report_path = _save_report(report, tmp_path)

    with patch(_MLFLOW_MODULE) as mock_mlflow:
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        from valravn.evaluation.evaluators import evaluate_report

        results = evaluate_report(report_path, suite="citation-coverage")

    assert results["sc_004_citation_coverage"] is True


# ---------------------------------------------------------------------------
# SC-006 report completeness
# ---------------------------------------------------------------------------


def test_report_completeness_passes(tmp_path):
    """A fully populated report with at least one conclusion passes SC-006."""
    conclusion = Conclusion(
        statement="Persistence mechanism found",
        supporting_invocation_ids=["inv-1"],
        confidence="medium",
    )
    report = _minimal_report(
        prompt="Investigate memory image",
        evidence_refs=["/mnt/evidence/mem.lime"],
        conclusions=[conclusion],
    )
    report_path = _save_report(report, tmp_path)

    with patch(_MLFLOW_MODULE) as mock_mlflow:
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        from valravn.evaluation.evaluators import evaluate_report

        results = evaluate_report(report_path, suite="report-completeness")

    assert results["sc_006_report_completeness"] is True


def test_report_completeness_fails_no_sections(tmp_path):
    """A report with no conclusions and no anomalies fails SC-006."""
    report = _minimal_report(conclusions=[], anomalies=[])
    report_path = _save_report(report, tmp_path)

    with patch(_MLFLOW_MODULE) as mock_mlflow:
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        from valravn.evaluation.evaluators import evaluate_report

        results = evaluate_report(report_path, suite="report-completeness")

    assert results["sc_006_report_completeness"] is False


# ---------------------------------------------------------------------------
# SC-005 evidence integrity
# ---------------------------------------------------------------------------


def test_evidence_integrity_passes(tmp_path):
    """An evidence file that is read-only passes SC-005."""
    evidence = tmp_path / "evidence.raw"
    evidence.write_bytes(b"\x00" * 16)
    os.chmod(evidence, 0o444)

    report = _minimal_report(evidence_refs=[str(evidence)])
    report_path = _save_report(report, tmp_path)

    with patch(_MLFLOW_MODULE) as mock_mlflow:
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        from valravn.evaluation.evaluators import evaluate_report

        results = evaluate_report(report_path, suite="evidence-integrity")

    assert results["sc_005_evidence_integrity"] is True


def test_evidence_integrity_fails_writable(tmp_path):
    """An evidence file that is writable fails SC-005."""
    evidence = tmp_path / "evidence.raw"
    evidence.write_bytes(b"\x00" * 16)
    os.chmod(evidence, 0o644)  # owner-writable

    report = _minimal_report(evidence_refs=[str(evidence)])
    report_path = _save_report(report, tmp_path)

    with patch(_MLFLOW_MODULE) as mock_mlflow:
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        from valravn.evaluation.evaluators import evaluate_report

        results = evaluate_report(report_path, suite="evidence-integrity")

    assert results["sc_005_evidence_integrity"] is False


# ---------------------------------------------------------------------------
# SC-002 anomaly detection
# ---------------------------------------------------------------------------


def test_anomaly_detection_passes(tmp_path):
    """A report containing at least one Anomaly passes SC-002."""
    anomaly = Anomaly(
        description="Suspicious process",
        source_invocation_ids=["inv-x"],
        forensic_significance="possible rootkit",
        response_action=AnomalyResponseAction.ADDED_FOLLOW_UP,
    )
    report = _minimal_report(anomalies=[anomaly])
    report_path = _save_report(report, tmp_path)

    with patch(_MLFLOW_MODULE) as mock_mlflow:
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        from valravn.evaluation.evaluators import evaluate_report

        results = evaluate_report(report_path, suite="anomaly-detection")

    assert results["sc_002_anomaly_detection"] is True


def test_anomaly_detection_fails_empty(tmp_path):
    """A report with no anomalies fails SC-002."""
    report = _minimal_report(anomalies=[])
    report_path = _save_report(report, tmp_path)

    with patch(_MLFLOW_MODULE) as mock_mlflow:
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        from valravn.evaluation.evaluators import evaluate_report

        results = evaluate_report(report_path, suite="anomaly-detection")

    assert results["sc_002_anomaly_detection"] is False


# ---------------------------------------------------------------------------
# SC-003 self-correction
# ---------------------------------------------------------------------------


def test_self_correction_passes_when_retry_recorded(tmp_path):
    """When a tool failure occurred and a self-correction was recorded, SC-003 passes."""
    failure = ToolFailureRecord(
        step_id="s1",
        invocation_ids=["inv-1"],
        final_error="permission denied",
        diagnostic_context="tried without sudo",
    )
    correction = SelfCorrectionEvent(
        step_id="s1",
        attempt_number=2,
        original_cmd=["vol.py", "pslist"],
        corrected_cmd=["sudo", "vol.py", "pslist"],
        correction_rationale="Added sudo to address permission error",
    )
    report = _minimal_report(tool_failures=[failure], self_corrections=[correction])
    report_path = _save_report(report, tmp_path)

    with patch(_MLFLOW_MODULE) as mock_mlflow:
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        from valravn.evaluation.evaluators import evaluate_report

        results = evaluate_report(report_path, suite="self-correction")

    assert results["sc_003_self_correction"] is True


def test_self_correction_fails_when_no_retry(tmp_path):
    """When a tool failure occurred but no self-correction was recorded, SC-003 fails."""
    failure = ToolFailureRecord(
        step_id="s1",
        invocation_ids=["inv-1"],
        final_error="timeout",
        diagnostic_context="took too long",
    )
    report = _minimal_report(tool_failures=[failure], self_corrections=[])
    report_path = _save_report(report, tmp_path)

    with patch(_MLFLOW_MODULE) as mock_mlflow:
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        from valravn.evaluation.evaluators import evaluate_report

        results = evaluate_report(report_path, suite="self-correction")

    assert results["sc_003_self_correction"] is False


# ---------------------------------------------------------------------------
# Golden dataset
# ---------------------------------------------------------------------------


def test_add_to_golden_appends_jsonl(tmp_path):
    """add_to_golden writes a valid JSON line to the golden dataset file."""
    report = _minimal_report(prompt="golden dataset test")
    report_path = _save_report(report, tmp_path)

    # Override GOLDEN_DATASET path to a temp location so we don't pollute the repo
    golden_path = tmp_path / "datasets" / "golden.jsonl"

    import valravn.evaluation.datasets as datasets_module

    original = datasets_module.GOLDEN_DATASET
    datasets_module.GOLDEN_DATASET = golden_path
    try:
        from valravn.evaluation.datasets import add_to_golden

        add_to_golden(report_path)
    finally:
        datasets_module.GOLDEN_DATASET = original

    assert golden_path.exists(), "golden.jsonl was not created"
    lines = golden_path.read_text().splitlines()
    assert len(lines) == 1, "Expected exactly one JSONL line"
    parsed = json.loads(lines[0])
    assert parsed["prompt"] == "golden dataset test"
    assert parsed["task_id"] == "test-id"


def test_add_to_golden_appends_multiple(tmp_path):
    """Calling add_to_golden twice appends two lines to the dataset."""
    report1 = _minimal_report(task_id="id-1", prompt="first")
    report2 = _minimal_report(task_id="id-2", prompt="second")
    path1 = tmp_path / "r1.json"
    path2 = tmp_path / "r2.json"
    path1.write_text(report1.model_dump_json())
    path2.write_text(report2.model_dump_json())

    golden_path = tmp_path / "datasets" / "golden.jsonl"

    import valravn.evaluation.datasets as datasets_module

    original = datasets_module.GOLDEN_DATASET
    datasets_module.GOLDEN_DATASET = golden_path
    try:
        from valravn.evaluation.datasets import add_to_golden

        add_to_golden(path1)
        add_to_golden(path2)
    finally:
        datasets_module.GOLDEN_DATASET = original

    lines = golden_path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["task_id"] == "id-1"
    assert json.loads(lines[1])["task_id"] == "id-2"


def test_evaluate_report_unknown_suite_raises(tmp_path):
    """evaluate_report raises ValueError for an unrecognised suite name."""
    import pytest

    from valravn.evaluation.evaluators import evaluate_report

    report = _minimal_report()
    report_path = tmp_path / "report.json"
    report_path.write_text(report.model_dump_json())

    with pytest.raises(ValueError, match="Unknown suite"):
        evaluate_report(report_path, suite="nonexistent-suite")


# ---------------------------------------------------------------------------
# A-07: SC-005 hash comparison
# ---------------------------------------------------------------------------


def test_evidence_integrity_detects_modified_readonly_file_via_hash(tmp_path):
    """A-07: SC-005 must fail when evidence_hashes differ even if file is read-only.

    This is the case that permission-checking alone cannot detect: a read-only
    file that was modified in-place (e.g. by temporarily changing permissions,
    writing new content, then restoring read-only). Hash comparison is required.
    """
    import hashlib

    evidence = tmp_path / "evidence.raw"
    evidence.write_bytes(b"original content")
    original_hash = hashlib.sha256(b"original content").hexdigest()

    # Simulate a tamper-then-restore: file is now read-only but content changed
    os.chmod(evidence, 0o644)
    evidence.write_bytes(b"tampered content")
    os.chmod(evidence, 0o444)  # restore read-only — permission check would pass!

    # Report stores the PRE-tamper hash
    report = _minimal_report(
        evidence_refs=[str(evidence)],
        evidence_hashes={str(evidence): original_hash},
    )
    report_path = _save_report(report, tmp_path)

    with patch(_MLFLOW_MODULE) as mock_mlflow:
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        from valravn.evaluation.evaluators import evaluate_report

        results = evaluate_report(report_path, suite="evidence-integrity")

    assert results["sc_005_evidence_integrity"] is False, (
        "SC-005 must fail when hash mismatch detected (file tampered then locked)"
    )


def test_evidence_integrity_passes_when_hash_matches(tmp_path):
    """A-07: SC-005 must pass when stored hash matches current file hash."""
    import hashlib

    evidence = tmp_path / "evidence.raw"
    evidence.write_bytes(b"untouched content")
    os.chmod(evidence, 0o444)
    current_hash = hashlib.sha256(b"untouched content").hexdigest()

    report = _minimal_report(
        evidence_refs=[str(evidence)],
        evidence_hashes={str(evidence): current_hash},
    )
    report_path = _save_report(report, tmp_path)

    with patch(_MLFLOW_MODULE) as mock_mlflow:
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        from valravn.evaluation.evaluators import evaluate_report

        results = evaluate_report(report_path, suite="evidence-integrity")

    assert results["sc_005_evidence_integrity"] is True
