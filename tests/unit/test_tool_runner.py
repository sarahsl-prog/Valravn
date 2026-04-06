# Re-import subprocess for timeout test
import subprocess
from datetime import timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from valravn.models.report import SelfCorrectionEvent, ToolFailureRecord
from valravn.models.task import InvestigationPlan, InvestigationTask, PlannedStep
from valravn.nodes.tool_runner import _CorrectionSpec, run_forensic_tool


def _state(read_only_evidence, output_dir, tool_cmd):
    task = InvestigationTask(prompt="test", evidence_refs=[str(read_only_evidence)])
    step = PlannedStep(skill_domain="sleuthkit", tool_cmd=tool_cmd, rationale="r")
    plan = InvestigationPlan(task_id=task.id, steps=[step])
    return {
        "task": task, "plan": plan, "invocations": [],
        "anomalies": [], "report": None,
        "current_step_id": step.id, "skill_cache": {},
        "messages": [], "_output_dir": str(output_dir),
        # max_attempts=1: single-shot for these basic unit tests (no retry path)
        "_retry_config": {"max_attempts": 1, "retry_delay_seconds": 0.0},
    }


def test_run_tool_captures_stdout(read_only_evidence, output_dir):
    state = _state(read_only_evidence, output_dir, ["echo", "hello world"])
    result = run_forensic_tool(state)
    rec = result["invocations"][-1]
    assert Path(rec.stdout_path).read_text() == "hello world\n"
    assert rec.exit_code == 0
    assert rec.success is True


def test_run_tool_captures_stderr(read_only_evidence, output_dir):
    state = _state(read_only_evidence, output_dir, ["bash", "-c", "echo err >&2; exit 1"])
    result = run_forensic_tool(state)
    rec = result["invocations"][-1]
    assert "err" in Path(rec.stderr_path).read_text()
    assert rec.exit_code == 1


def test_run_tool_stdout_not_in_evidence(read_only_evidence, output_dir):
    state = _state(read_only_evidence, output_dir, ["echo", "data"])
    result = run_forensic_tool(state)
    rec = result["invocations"][-1]
    assert str(output_dir) in str(rec.stdout_path)
    assert str(read_only_evidence.parent) not in str(rec.stdout_path)


def test_run_tool_timestamps_utc(read_only_evidence, output_dir):
    state = _state(read_only_evidence, output_dir, ["echo", "x"])
    result = run_forensic_tool(state)
    rec = result["invocations"][-1]
    assert rec.started_at_utc.tzinfo == timezone.utc
    assert rec.completed_at_utc.tzinfo == timezone.utc


def test_fr007_rejects_output_under_evidence(read_only_evidence, tmp_path_factory):
    # output_dir is set to a subdirectory of the evidence directory — must be rejected
    evidence_dir = read_only_evidence.parent
    bad_output = evidence_dir / "subdir"
    bad_output.mkdir(exist_ok=True)
    state = _state(read_only_evidence, bad_output, ["echo", "x"])
    with pytest.raises(ValueError, match="evidence directory"):
        run_forensic_tool(state)


def test_record_json_persisted(read_only_evidence, output_dir):
    state = _state(read_only_evidence, output_dir, ["echo", "record"])
    result = run_forensic_tool(state)
    inv_id = result["_last_invocation_id"]
    record_file = output_dir / "analysis" / f"{inv_id}.record.json"
    assert record_file.exists(), "ToolInvocationRecord JSON was not written to disk"
    import json
    data = json.loads(record_file.read_text())
    assert data["id"] == inv_id
    assert data["attempt_number"] == 1


def test_run_tool_invalid_step_id_raises(read_only_evidence, output_dir):
    task = InvestigationTask(prompt="test", evidence_refs=[str(read_only_evidence)])
    step = PlannedStep(skill_domain="sleuthkit", tool_cmd=["echo", "x"], rationale="r")
    plan = InvestigationPlan(task_id=task.id, steps=[step])
    state = {
        "task": task, "plan": plan, "invocations": [],
        "anomalies": [], "report": None,
        "current_step_id": "nonexistent-step-id", "skill_cache": {},
        "messages": [], "_output_dir": str(output_dir),
        "_retry_config": {"max_attempts": 3, "retry_delay_seconds": 0.0},
    }
    with pytest.raises(ValueError, match="not found in plan"):
        run_forensic_tool(state)


# ---------------------------------------------------------------------------
# US3 — Retry loop and self-correction tests
# ---------------------------------------------------------------------------

def _make_proc(returncode: int, stdout: str = "", stderr: str = ""):
    """Build a mock CompletedProcess-like object."""
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    return proc


def _state_with_retries(read_only_evidence, output_dir, tool_cmd, max_attempts=3):
    task = InvestigationTask(prompt="test", evidence_refs=[str(read_only_evidence)])
    step = PlannedStep(skill_domain="sleuthkit", tool_cmd=tool_cmd, rationale="r")
    plan = InvestigationPlan(task_id=task.id, steps=[step])
    return {
        "task": task, "plan": plan, "invocations": [],
        "anomalies": [], "report": None,
        "current_step_id": step.id, "skill_cache": {},
        "messages": [], "_output_dir": str(output_dir),
        "_retry_config": {"max_attempts": max_attempts, "retry_delay_seconds": 0.0},
        "_self_corrections": [],
    }


def test_retry_on_failure_succeeds_second_attempt(read_only_evidence, output_dir):
    """First subprocess call fails (exit 1, no output); second succeeds (exit 0, output)."""
    fail_proc = _make_proc(returncode=1, stdout="", stderr="bad arg")
    ok_proc = _make_proc(returncode=0, stdout="result data", stderr="")

    correction = _CorrectionSpec(corrected_cmd=["fixed", "cmd"], rationale="fixed the flag")
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = correction

    with patch("subprocess.run", side_effect=[fail_proc, ok_proc]), \
         patch("valravn.nodes.tool_runner._get_correction_llm", return_value=mock_llm):
        state = _state_with_retries(read_only_evidence, output_dir, ["original", "cmd"])
        result = run_forensic_tool(state)

    assert result["_step_succeeded"] is True
    assert result["_step_exhausted"] is False
    assert result["_tool_failure"] is None
    assert len(result["invocations"]) == 2
    assert result["invocations"][0].attempt_number == 1
    assert result["invocations"][1].attempt_number == 2
    assert len(result["_self_corrections"]) == 1


def test_exhaustion_creates_tool_failure_record(read_only_evidence, output_dir):
    """All three attempts fail; node returns _step_exhausted=True and a ToolFailureRecord."""
    fail_proc = _make_proc(returncode=1, stdout="", stderr="error")
    correction = _CorrectionSpec(corrected_cmd=["still", "broken"], rationale="attempt fix")
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = correction

    with patch("subprocess.run", return_value=fail_proc), \
         patch("valravn.nodes.tool_runner._get_correction_llm", return_value=mock_llm):
        state = _state_with_retries(read_only_evidence, output_dir, ["bad", "cmd"], max_attempts=3)
        result = run_forensic_tool(state)

    assert result["_step_exhausted"] is True
    assert result["_step_succeeded"] is False
    assert isinstance(result["_tool_failure"], ToolFailureRecord)

    failure = result["_tool_failure"]
    # All three invocation IDs should be recorded
    assert len(failure.invocation_ids) == 3
    assert len(result["invocations"]) == 3
    # Two self-correction events (after attempt 1 and attempt 2)
    assert len(result["_self_corrections"]) == 2
    # PlannedStep.status must be EXHAUSTED (T027)
    from valravn.models.task import StepStatus
    step = result["plan"].steps[0]
    assert step.status == StepStatus.EXHAUSTED


def test_self_correction_event_fields(read_only_evidence, output_dir):
    """SelfCorrectionEvent captures original_cmd, corrected_cmd, and correction_rationale."""
    fail_proc = _make_proc(returncode=1, stdout="", stderr="missing flag")
    ok_proc = _make_proc(returncode=0, stdout="success output", stderr="")

    original_cmd = ["vol.py", "-f", "mem.raw", "wrong.plugin"]
    corrected_cmd = ["vol.py", "-f", "mem.raw", "windows.pslist"]
    rationale = "Plugin name was incorrect"

    correction = _CorrectionSpec(corrected_cmd=corrected_cmd, rationale=rationale)
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = correction

    with patch("subprocess.run", side_effect=[fail_proc, ok_proc]), \
         patch("valravn.nodes.tool_runner._get_correction_llm", return_value=mock_llm):
        state = _state_with_retries(read_only_evidence, output_dir, original_cmd)
        result = run_forensic_tool(state)

    assert len(result["_self_corrections"]) == 1
    event = result["_self_corrections"][0]
    assert isinstance(event, SelfCorrectionEvent)
    assert event.original_cmd == original_cmd
    assert event.corrected_cmd == corrected_cmd
    assert event.correction_rationale == rationale
    assert event.attempt_number == 1


def test_exhaustion_exit_code_one(read_only_evidence, output_dir):
    """_tool_failure is not None when exhausted — graph uses this to set exit code 1."""
    fail_proc = _make_proc(returncode=2, stdout="", stderr="fatal error")
    correction = _CorrectionSpec(corrected_cmd=["retry", "cmd"], rationale="trying again")
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = correction

    with patch("subprocess.run", return_value=fail_proc), \
         patch("valravn.nodes.tool_runner._get_correction_llm", return_value=mock_llm):
        state = _state_with_retries(read_only_evidence, output_dir, ["fail", "cmd"], max_attempts=2)
        result = run_forensic_tool(state)

    # The presence of _tool_failure is what drives exit_code=1 in FindingsReport
    assert result["_tool_failure"] is not None
    assert isinstance(result["_tool_failure"], ToolFailureRecord)
    assert result["_tool_failure"].step_id == state["current_step_id"]
    assert "exit_code=2" in result["_tool_failure"].final_error


def test_tool_timeout_sets_exit_code_minus_one(read_only_evidence, output_dir):
    """Tool timeout (after 1hr) results in exit_code=-1."""
    # The timeout handling is in _run_single_attempt, verify by mocking subprocess.run
    timeout_error = subprocess.TimeoutExpired(
        cmd=["sleep", "10"],
        timeout=3600,
    )

    with patch("subprocess.run", side_effect=timeout_error):
        state = _state(read_only_evidence, output_dir, ["sleep", "10"])
        result = run_forensic_tool(state)

    rec = result["invocations"][-1]
    assert rec.exit_code == -1
    assert rec.success is False
    assert "timed out" in rec.stderr_path.read_text()
