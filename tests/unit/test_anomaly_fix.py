"""Tests for Task 3: Fix Anomaly Response Action and Follow-Up Logic."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from valravn.models.records import AnomalyResponseAction
from valravn.models.task import InvestigationPlan, InvestigationTask, PlannedStep
from valravn.nodes.anomaly import check_anomalies, record_anomaly

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_state(
    read_only_evidence: Path,
    output_dir: Path,
    detected_anomaly_data: dict | None = None,
) -> dict:
    task = InvestigationTask(
        prompt="analyse evidence",
        evidence_refs=[str(read_only_evidence)],
    )
    step = PlannedStep(
        skill_domain="sleuthkit",
        tool_cmd=["fls", "-r", str(read_only_evidence)],
        rationale="list files",
    )
    plan = InvestigationPlan(task_id=task.id, steps=[step])

    return {
        "task": task,
        "plan": plan,
        "invocations": [],
        "anomalies": [],
        "report": None,
        "current_step_id": step.id,
        "skill_cache": {},
        "messages": [],
        "_output_dir": str(output_dir),
        "_last_invocation_id": "inv-test-001",
        "_detected_anomaly_data": detected_anomaly_data,
    }


# ---------------------------------------------------------------------------
# Test 1: LLM response includes response_action in _detected_anomaly_data
# ---------------------------------------------------------------------------


@patch("valravn.nodes.anomaly._get_anomaly_llm")
def test_anomaly_response_action_from_llm(mock_llm_fn, tmp_path):
    """_detected_anomaly_data must contain response_action from LLM result."""
    stdout_file = tmp_path / "stdout.txt"
    stdout_file.write_text("suspicious output\n")

    from datetime import datetime, timezone

    from valravn.models.records import ToolInvocationRecord

    now = datetime.now(timezone.utc)
    invocation = ToolInvocationRecord(
        step_id="step-001",
        attempt_number=1,
        cmd=["fls", "-r", "/evidence.raw"],
        exit_code=0,
        stdout_path=stdout_file,
        stderr_path=stdout_file,
        started_at_utc=now,
        completed_at_utc=now,
        duration_seconds=0.5,
        had_output=True,
    )

    import json
    # Mock LLM returns response_action = "no_follow_up_warranted"
    mock_response = MagicMock(content=json.dumps({
        "anomaly_detected": True,
        "description": "Suspicious absence of expected files",
        "forensic_significance": "Files may have been wiped",
        "category": "unexpected_absence",
        "response_action": "no_follow_up_warranted",
    }))

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response
    mock_llm_fn.return_value = mock_llm

    state = {
        "invocations": [invocation],
        "task": None,
        "plan": None,
        "anomalies": [],
        "_output_dir": str(tmp_path),
        "_last_invocation_id": "",
        "_detected_anomaly_data": None,
    }

    result = check_anomalies(state)

    assert result["_pending_anomalies"] is True
    assert result["_detected_anomaly_data"] is not None
    assert "response_action" in result["_detected_anomaly_data"]
    assert result["_detected_anomaly_data"]["response_action"] == "no_follow_up_warranted"


# ---------------------------------------------------------------------------
# Test 2: record_anomaly with NO_FOLLOW_UP creates no follow-up steps
# ---------------------------------------------------------------------------


def test_record_anomaly_no_follow_up_when_action_says_so(read_only_evidence, output_dir):
    """When response_action is no_follow_up_warranted, _follow_up_steps must be empty."""
    detected_data = {
        "anomaly_detected": True,
        "description": "Minor timestamp skew",
        "forensic_significance": "Low significance",
        "category": "timestamp_contradiction",
        "response_action": "no_follow_up_warranted",
    }

    state = _make_state(read_only_evidence, output_dir, detected_anomaly_data=detected_data)

    result = record_anomaly(state)

    assert result["_follow_up_steps"] == []
    assert len(result["anomalies"]) == 1
    anomaly = result["anomalies"][0]
    assert anomaly.response_action == AnomalyResponseAction.NO_FOLLOW_UP


# ---------------------------------------------------------------------------
# Test 3: record_anomaly creates context-aware follow-up (not strings)
# ---------------------------------------------------------------------------


def test_record_anomaly_context_aware_follow_up(read_only_evidence, output_dir):
    """timestamp_contradiction category must produce plaso-timeline follow-up, not strings."""
    detected_data = {
        "anomaly_detected": True,
        "description": "Timestamp contradiction in $STANDARD_INFORMATION vs $FILE_NAME",
        "forensic_significance": "Possible timestomping",
        "category": "timestamp_contradiction",
        "response_action": "added_follow_up_steps",
    }

    state = _make_state(read_only_evidence, output_dir, detected_anomaly_data=detected_data)

    result = record_anomaly(state)

    assert len(result["_follow_up_steps"]) > 0

    follow_up: PlannedStep = result["_follow_up_steps"][0]
    # Should NOT use strings
    assert follow_up.tool_cmd[0] != "strings"
    # Should use the plaso-timeline mapping
    assert follow_up.skill_domain == "plaso-timeline"
    assert "log2timeline.py" in follow_up.tool_cmd


# ---------------------------------------------------------------------------
# Additional tests for edge cases
# ---------------------------------------------------------------------------


def test_record_anomaly_invalid_response_action_defaults_to_no_follow_up(
    read_only_evidence, output_dir
):
    """An unrecognised response_action string must default to NO_FOLLOW_UP."""
    detected_data = {
        "anomaly_detected": True,
        "description": "Some anomaly",
        "forensic_significance": "moderate",
        "category": "cross_tool_conflict",
        "response_action": "completely_invalid_value",
    }

    state = _make_state(read_only_evidence, output_dir, detected_anomaly_data=detected_data)

    result = record_anomaly(state)

    assert result["_follow_up_steps"] == []
    assert result["anomalies"][0].response_action == AnomalyResponseAction.NO_FOLLOW_UP


def test_record_anomaly_unknown_category_falls_back_to_strings(read_only_evidence, output_dir):
    """An anomaly category not in _FOLLOW_UP_COMMANDS must fall back to strings command."""
    detected_data = {
        "anomaly_detected": True,
        "description": "Novel anomaly type",
        "forensic_significance": "unknown impact",
        "category": "some_unknown_category",
        "response_action": "added_follow_up_steps",
    }

    state = _make_state(read_only_evidence, output_dir, detected_anomaly_data=detected_data)

    result = record_anomaly(state)

    assert len(result["_follow_up_steps"]) > 0
    follow_up: PlannedStep = result["_follow_up_steps"][0]
    assert follow_up.tool_cmd[0] == "strings"
    assert "-n" in follow_up.tool_cmd
    assert "20" in follow_up.tool_cmd


def test_record_anomaly_orphaned_relationship_uses_volatility3(read_only_evidence, output_dir):
    """orphaned_relationship category must produce a python3 vol.py memory-analysis follow-up."""
    detected_data = {
        "anomaly_detected": True,
        "description": "Process with no valid parent",
        "forensic_significance": "Potential process injection",
        "category": "orphaned_relationship",
        "response_action": "added_follow_up_steps",
    }

    state = _make_state(read_only_evidence, output_dir, detected_anomaly_data=detected_data)

    result = record_anomaly(state)

    assert len(result["_follow_up_steps"]) > 0
    follow_up: PlannedStep = result["_follow_up_steps"][0]
    assert follow_up.skill_domain == "memory-analysis"
    assert follow_up.tool_cmd[0] == "python3"
    assert any("vol.py" in part for part in follow_up.tool_cmd)
