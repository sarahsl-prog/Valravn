from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from valravn.models.task import InvestigationPlan, InvestigationTask, PlannedStep, StepStatus
from valravn.nodes.plan import plan_investigation, update_plan


def _base_state(read_only_evidence, output_dir):
    task = InvestigationTask(
        prompt="identify network connections",
        evidence_refs=[str(read_only_evidence)],
    )
    plan = InvestigationPlan(task_id=task.id)
    return {
        "task": task,
        "plan": plan,
        "invocations": [],
        "anomalies": [],
        "report": None,
        "current_step_id": None,
        "skill_cache": {},
        "messages": [],
        "_output_dir": str(output_dir),
    }


def test_plan_investigation_populates_steps(read_only_evidence, output_dir):
    mock_response = MagicMock()
    mock_response.steps = [
        MagicMock(
            skill_domain="memory-analysis",
            tool_cmd=["python3", "/opt/volatility3-2.20.0/vol.py", "-f", "/mnt/mem.lime", "windows.netstat"],
            rationale="list network connections",
        )
    ]

    state = _base_state(read_only_evidence, output_dir)

    with patch("valravn.nodes.plan._get_llm") as mock_llm_fn:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_llm_fn.return_value = mock_llm

        result = plan_investigation(state)

    assert len(result["plan"].steps) == 1
    assert result["plan"].steps[0].skill_domain == "memory-analysis"
    assert result["current_step_id"] == result["plan"].steps[0].id


def test_plan_investigation_writes_json(read_only_evidence, output_dir):
    mock_response = MagicMock()
    mock_response.steps = [
        MagicMock(skill_domain="sleuthkit", tool_cmd=["fls", "-r"], rationale="list files")
    ]
    state = _base_state(read_only_evidence, output_dir)

    with patch("valravn.nodes.plan._get_llm") as mock_llm_fn:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_llm_fn.return_value = mock_llm

        plan_investigation(state)

    plan_file = output_dir / "analysis" / "investigation_plan.json"
    assert plan_file.exists()


def test_update_plan_marks_step_completed(read_only_evidence, output_dir):
    task = InvestigationTask(prompt="test", evidence_refs=[str(read_only_evidence)])
    step = PlannedStep(skill_domain="sleuthkit", tool_cmd=["fls"], rationale="r")
    plan = InvestigationPlan(task_id=task.id, steps=[step])
    state = {
        "task": task, "plan": plan, "invocations": [], "anomalies": [],
        "report": None, "current_step_id": step.id, "skill_cache": {},
        "messages": [], "_step_succeeded": True, "_output_dir": str(output_dir),
        "_step_exhausted": False, "_tool_failure": None, "_tool_failures": [],
    }
    result = update_plan(state)
    assert result["plan"].steps[0].status == StepStatus.COMPLETED


def test_update_plan_marks_step_exhausted(read_only_evidence, output_dir):
    task = InvestigationTask(prompt="test", evidence_refs=[str(read_only_evidence)])
    step = PlannedStep(skill_domain="sleuthkit", tool_cmd=["fls"], rationale="r")
    plan = InvestigationPlan(task_id=task.id, steps=[step])
    from valravn.models.report import ToolFailureRecord
    failure = ToolFailureRecord(step_id=step.id, invocation_ids=["inv-1"],
                                final_error="err", diagnostic_context="ctx")
    state = {
        "task": task, "plan": plan, "invocations": [], "anomalies": [],
        "report": None, "current_step_id": step.id, "skill_cache": {},
        "messages": [], "_step_succeeded": False, "_output_dir": str(output_dir),
        "_step_exhausted": True, "_tool_failure": failure, "_tool_failures": [],
    }
    result = update_plan(state)
    assert result["plan"].steps[0].status == StepStatus.EXHAUSTED
    assert len(result["_tool_failures"]) == 1
