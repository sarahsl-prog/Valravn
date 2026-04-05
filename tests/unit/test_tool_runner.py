from datetime import timezone
from pathlib import Path

import pytest

from valravn.models.task import InvestigationPlan, InvestigationTask, PlannedStep
from valravn.nodes.tool_runner import run_forensic_tool


def _state(read_only_evidence, output_dir, tool_cmd):
    task = InvestigationTask(prompt="test", evidence_refs=[str(read_only_evidence)])
    step = PlannedStep(skill_domain="sleuthkit", tool_cmd=tool_cmd, rationale="r")
    plan = InvestigationPlan(task_id=task.id, steps=[step])
    return {
        "task": task, "plan": plan, "invocations": [],
        "anomalies": [], "report": None,
        "current_step_id": step.id, "skill_cache": {},
        "messages": [], "_output_dir": str(output_dir),
        "_retry_config": {"max_attempts": 3, "retry_delay_seconds": 0.0},
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
