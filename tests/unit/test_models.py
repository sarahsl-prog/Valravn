import os
from datetime import datetime, timezone
from datetime import timezone as tz2
from pathlib import Path

import pytest
from pydantic import ValidationError

from valravn.models.records import Anomaly, AnomalyResponseAction, ToolInvocationRecord
from valravn.models.report import (
    Conclusion,
    FindingsReport,
    ToolFailureRecord,
)
from valravn.models.task import (
    InvestigationPlan,
    InvestigationTask,
    PlannedStep,
    StepStatus,
)


def test_planned_step_gets_id():
    step = PlannedStep(
        skill_domain="sleuthkit", tool_cmd=["fls", "-r", "/mnt/img"], rationale="list files"
    )
    assert step.id  # auto-generated UUID


def test_planned_step_default_status():
    step = PlannedStep(skill_domain="sleuthkit", tool_cmd=["fls"], rationale="r")
    assert step.status == StepStatus.PENDING


def test_planned_step_original_tool_cmd_immutable():
    """A-08: original_tool_cmd is set from tool_cmd at creation and survives overwrites."""
    original = ["fls", "-r", "/mnt/img"]
    step = PlannedStep(skill_domain="sleuthkit", tool_cmd=original, rationale="r")

    assert step.original_tool_cmd == original

    # Overwrite tool_cmd (as a correction would do)
    step.tool_cmd = ["fls", "-r", "/mnt/img", "--fixed-flag"]

    # original_tool_cmd must remain unchanged
    assert step.original_tool_cmd == original
    assert step.tool_cmd == ["fls", "-r", "/mnt/img", "--fixed-flag"]


def test_investigation_plan_timestamps():
    plan = InvestigationPlan(task_id="abc")
    assert plan.created_at_utc.tzinfo == timezone.utc
    assert plan.last_updated_utc.tzinfo == timezone.utc


def test_investigation_task_rejects_empty_prompt(read_only_evidence):
    with pytest.raises(ValidationError, match="prompt must not be empty"):
        InvestigationTask(prompt="  ", evidence_refs=[str(read_only_evidence)])


def test_investigation_task_rejects_writable_evidence(tmp_path):
    writable = tmp_path / "img.raw"
    writable.write_bytes(b"x")
    with pytest.raises(ValidationError, match="writable"):
        InvestigationTask(prompt="find files", evidence_refs=[str(writable)])


def test_investigation_task_rejects_missing_evidence():
    with pytest.raises(ValidationError, match="does not exist"):
        InvestigationTask(prompt="find files", evidence_refs=["/nonexistent/path.raw"])


def test_investigation_task_accepts_read_only_evidence(read_only_evidence):
    task = InvestigationTask(prompt="find files", evidence_refs=[str(read_only_evidence)])
    assert task.id
    assert task.created_at_utc.tzinfo == timezone.utc


def test_investigation_task_accepts_multiple_evidence(tmp_path):
    """InvestigationTask accepts multiple evidence paths."""
    ev1 = tmp_path / "ev1.raw"
    ev2 = tmp_path / "ev2.raw"
    ev3 = tmp_path / "ev3.raw"
    for ev in (ev1, ev2, ev3):
        ev.write_bytes(b"x")
        os.chmod(ev, 0o444)

    task = InvestigationTask(
        prompt="test",
        evidence_refs=[str(ev1), str(ev2), str(ev3)]
    )
    assert len(task.evidence_refs) == 3
    assert task.id


def test_investigation_task_rejects_multiple_mixed_evidence(tmp_path):
    """InvestigationTask rejects when any evidence is writable."""
    ev1 = tmp_path / "ev1.raw"
    ev2 = tmp_path / "ev2.raw"
    ev1.write_bytes(b"x")
    os.chmod(ev1, 0o444)
    ev2.write_bytes(b"x")
    # ev2 is writable (default)

    with pytest.raises(ValidationError, match="writable"):
        InvestigationTask(prompt="test", evidence_refs=[str(ev1), str(ev2)])


# --- records.py tests ---


def test_tool_invocation_record_success_flag(tmp_path):
    rec = ToolInvocationRecord(
        step_id="s1",
        attempt_number=1,
        cmd=["fls", "-r", "/mnt/img"],
        exit_code=0,
        stdout_path=tmp_path / "out.stdout",
        stderr_path=tmp_path / "out.stderr",
        started_at_utc=datetime.now(tz2.utc),
        completed_at_utc=datetime.now(tz2.utc),
        duration_seconds=1.2,
        had_output=True,
    )
    assert rec.success is True
    assert rec.id


def test_tool_invocation_record_failure_flag(tmp_path):
    rec = ToolInvocationRecord(
        step_id="s1",
        attempt_number=1,
        cmd=["fls"],
        exit_code=1,
        stdout_path=tmp_path / "out.stdout",
        stderr_path=tmp_path / "out.stderr",
        started_at_utc=datetime.now(tz2.utc),
        completed_at_utc=datetime.now(tz2.utc),
        duration_seconds=0.1,
        had_output=False,
    )
    assert rec.success is False


def test_anomaly_requires_source_invocations():
    with pytest.raises(ValidationError, match="at least one source"):
        Anomaly(
            description="conflict",
            source_invocation_ids=[],
            forensic_significance="significant",
            response_action=AnomalyResponseAction.NO_FOLLOW_UP,
        )


def test_anomaly_valid():
    a = Anomaly(
        description="timestamp predates OS install",
        source_invocation_ids=["inv-1"],
        forensic_significance="indicates backdating",
        response_action=AnomalyResponseAction.ADDED_FOLLOW_UP,
    )
    assert a.id
    assert a.detected_at_utc.tzinfo == timezone.utc


# --- report.py tests ---


def test_conclusion_requires_citations():
    with pytest.raises(ValidationError, match="must cite"):
        Conclusion(
            statement="malware present",
            supporting_invocation_ids=[],
            confidence="high",
        )


def test_findings_report_utc_timestamp():
    report = FindingsReport(
        task_id="t1",
        prompt="find files",
        evidence_refs=["/mnt/img"],
        conclusions=[],
        anomalies=[],
        tool_failures=[],
        self_corrections=[],
        investigation_plan_path=Path("/tmp/plan.json"),
    )
    assert report.generated_at_utc.tzinfo == timezone.utc


def test_findings_report_exit_code_clean():
    report = FindingsReport(
        task_id="t1",
        prompt="find files",
        evidence_refs=["/mnt/img"],
        conclusions=[],
        anomalies=[],
        tool_failures=[],
        self_corrections=[],
        investigation_plan_path=Path("/tmp/plan.json"),
    )
    assert report.exit_code == 0


def test_findings_report_exit_code_with_failures():
    failure = ToolFailureRecord(
        step_id="s1",
        invocation_ids=["inv-1"],
        final_error="No such file",
        diagnostic_context="tried 3 times",
    )
    report = FindingsReport(
        task_id="t1",
        prompt="find files",
        evidence_refs=["/mnt/img"],
        conclusions=[],
        anomalies=[],
        tool_failures=[failure],
        self_corrections=[],
        investigation_plan_path=Path("/tmp/plan.json"),
    )
    assert report.exit_code == 1
