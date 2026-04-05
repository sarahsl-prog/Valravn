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
