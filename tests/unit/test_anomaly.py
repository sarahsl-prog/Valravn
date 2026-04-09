from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from valravn.models.records import AnomalyResponseAction, ToolInvocationRecord
from valravn.models.task import InvestigationPlan, InvestigationTask, PlannedStep
from valravn.nodes.anomaly import check_anomalies, record_anomaly

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_invocation(stdout_path: Path, cmd: list[str] | None = None) -> ToolInvocationRecord:
    now = datetime.now(timezone.utc)
    return ToolInvocationRecord(
        step_id="step-001",
        attempt_number=1,
        cmd=cmd or ["fls", "-r", "/evidence.raw"],
        exit_code=0,
        stdout_path=stdout_path,
        stderr_path=stdout_path,  # reuse for simplicity
        started_at_utc=now,
        completed_at_utc=now,
        duration_seconds=0.5,
        had_output=True,
    )


def _base_state(
    read_only_evidence: Path,
    output_dir: Path,
    invocations: list | None = None,
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
        "invocations": invocations or [],
        "anomalies": [],
        "report": None,
        "current_step_id": step.id,
        "skill_cache": {},
        "messages": [],
        "_output_dir": str(output_dir),
        "_last_invocation_id": "inv-abc-123",
        "_detected_anomaly_data": detected_anomaly_data,
    }


# ---------------------------------------------------------------------------
# check_anomalies tests
# ---------------------------------------------------------------------------


def test_check_anomalies_no_invocations(read_only_evidence, output_dir):
    """Empty invocations list must return _pending_anomalies: False immediately."""
    state = _base_state(read_only_evidence, output_dir, invocations=[])

    result = check_anomalies(state)

    assert result["_pending_anomalies"] is False
    assert result["_detected_anomaly_data"] is None


def test_check_anomalies_none_detected(read_only_evidence, output_dir, tmp_path):
    """When LLM reports no anomaly, node returns _pending_anomalies: False."""
    stdout_file = tmp_path / "stdout.txt"
    stdout_file.write_text("normal fls output\nd/d 5: Users\n")

    invocation = _make_invocation(stdout_file)

    import json

    mock_response = MagicMock(
        content=json.dumps(
            {
                "anomaly_detected": False,
                "description": "",
                "forensic_significance": "",
                "category": "",
                "response_action": "no_follow_up_warranted",
            }
        )
    )

    state = _base_state(read_only_evidence, output_dir, invocations=[invocation])

    with patch("valravn.nodes.anomaly._get_anomaly_llm") as mock_llm_fn:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_llm_fn.return_value = mock_llm

        result = check_anomalies(state)

    assert result["_pending_anomalies"] is False
    assert result["_detected_anomaly_data"] is None


def test_check_anomalies_detected(read_only_evidence, output_dir, tmp_path):
    """When LLM reports a critical anomaly, node returns _pending_anomalies: True with data."""
    stdout_file = tmp_path / "stdout.txt"
    stdout_file.write_text("suspicious: process pid=1234 ppid=0 name=cmd.exe\n")

    invocation = _make_invocation(
        stdout_file, cmd=["python3", "/opt/volatility3-2.20.0/vol.py", "windows.pstree"]
    )

    # Use integrity_failure category which bypasses trust filtering
    anomaly_dump = {
        "anomaly_detected": True,
        "description": "MFT hash mismatch detected",
        "forensic_significance": "Evidence of anti-forensic tampering",
        "category": "integrity_failure",
    }

    import json

    mock_response = MagicMock(
        content=json.dumps({**anomaly_dump, "response_action": "no_follow_up_warranted"})
    )

    state = _base_state(read_only_evidence, output_dir, invocations=[invocation])

    with patch("valravn.nodes.anomaly._get_anomaly_llm") as mock_llm_fn:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_llm_fn.return_value = mock_llm

        result = check_anomalies(state)

    assert result["_pending_anomalies"] is True
    assert result["_detected_anomaly_data"] is not None
    assert result["_detected_anomaly_data"]["category"] == "integrity_failure"


def test_check_anomalies_filtered_by_low_trust(read_only_evidence, output_dir, tmp_path):
    """Low trust filters non-critical anomalies at start of investigation."""
    stdout_file = tmp_path / "stdout.txt"
    stdout_file.write_text("suspicious: process pid=1234 ppid=0 name=cmd.exe\n")

    invocation = _make_invocation(
        stdout_file, cmd=["python3", "/opt/volatility3-2.20.0/vol.py", "windows.pstree"]
    )

    # Non-critical anomaly should be filtered when trust is 0
    anomaly_dump = {
        "anomaly_detected": True,
        "description": "cmd.exe with no parent process",
        "forensic_significance": "Possible hollow process or direct kernel injection",
        "category": "orphaned_relationship",
    }

    import json

    mock_response = MagicMock(
        content=json.dumps({**anomaly_dump, "response_action": "no_follow_up_warranted"})
    )

    state = _base_state(read_only_evidence, output_dir, invocations=[invocation])

    with patch("valravn.nodes.anomaly._get_anomaly_llm") as mock_llm_fn:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_llm_fn.return_value = mock_llm

        result = check_anomalies(state)

    # Should be filtered due to low trust and non-critical category
    assert result["_pending_anomalies"] is False
    assert result["_detected_anomaly_data"] is None


# ---------------------------------------------------------------------------
# record_anomaly tests
# ---------------------------------------------------------------------------


def test_record_anomaly_persists_json(read_only_evidence, output_dir):
    """record_anomaly must write analysis/anomalies.json with the new anomaly."""
    detected_data = {
        "anomaly_detected": True,
        "description": "Hash mismatch on $MFT",
        "forensic_significance": "Evidence of anti-forensic tampering",
        "category": "integrity_failure",
        "response_action": "added_follow_up_steps",
    }

    state = _base_state(
        read_only_evidence,
        output_dir,
        detected_anomaly_data=detected_data,
    )

    record_anomaly(state)

    anomalies_path = Path(output_dir) / "analysis" / "anomalies.json"
    assert anomalies_path.exists(), "anomalies.json was not created"

    written = json.loads(anomalies_path.read_text())
    assert isinstance(written, list)
    assert len(written) == 1
    assert "integrity_failure" in written[0]["description"]
    assert written[0]["response_action"] == AnomalyResponseAction.ADDED_FOLLOW_UP


def test_record_anomaly_adds_follow_up_step(read_only_evidence, output_dir):
    """record_anomaly must return a non-empty _follow_up_steps list."""
    detected_data = {
        "anomaly_detected": True,
        "description": "Timestamp contradiction in $STANDARD_INFORMATION vs $FILE_NAME",
        "forensic_significance": "Possible timestomping",
        "category": "timestamp_contradiction",
        "response_action": "added_follow_up_steps",
    }

    state = _base_state(
        read_only_evidence,
        output_dir,
        detected_anomaly_data=detected_data,
    )

    result = record_anomaly(state)

    assert "_follow_up_steps" in result
    assert len(result["_follow_up_steps"]) > 0

    follow_up: PlannedStep = result["_follow_up_steps"][0]
    # timestamp_contradiction maps to log2timeline.py, not strings
    assert "log2timeline.py" in follow_up.tool_cmd
    assert follow_up.skill_domain == "plaso-timeline"
    assert "Follow-up investigation of anomaly" in follow_up.rationale


def test_record_anomaly_appends_to_existing_anomalies(read_only_evidence, output_dir):
    """record_anomaly must append to an existing anomalies list in state."""
    from valravn.models.records import Anomaly

    existing_anomaly = Anomaly(
        description="pre-existing anomaly",
        forensic_significance="low",
        source_invocation_ids=["inv-000"],
        response_action=AnomalyResponseAction.NO_FOLLOW_UP,
    )

    detected_data = {
        "anomaly_detected": True,
        "description": "New cross-tool conflict",
        "forensic_significance": "Volatility and fls disagree on MFT entry count",
        "category": "cross_tool_conflict",
    }

    state = _base_state(
        read_only_evidence,
        output_dir,
        detected_anomaly_data=detected_data,
    )
    state["anomalies"] = [existing_anomaly]

    result = record_anomaly(state)

    assert len(result["anomalies"]) == 2
    descriptions = [a.description for a in result["anomalies"]]
    assert any("pre-existing" in d for d in descriptions)
    assert any("cross_tool_conflict" in d for d in descriptions)

    # JSON should contain both
    anomalies_path = Path(output_dir) / "analysis" / "anomalies.json"
    written = json.loads(anomalies_path.read_text())
    assert len(written) == 2


def test_record_anomaly_clears_pending_flag(read_only_evidence, output_dir):
    """record_anomaly must always return _pending_anomalies: False."""
    detected_data = {
        "anomaly_detected": True,
        "description": "Unexpected absence of prefetch files",
        "forensic_significance": "Prefetch disabled or wiped",
        "category": "unexpected_absence",
    }

    state = _base_state(
        read_only_evidence,
        output_dir,
        detected_anomaly_data=detected_data,
    )

    result = record_anomaly(state)

    assert result["_pending_anomalies"] is False


def test_record_anomaly_generates_follow_up_with_strings_cmd(read_only_evidence, output_dir):
    """Follow-up step uses strings -n 20 for unknown/fallback category."""
    detected_data = {
        "anomaly_detected": True,
        "description": "Suspicious process",
        "forensic_significance": "potential malware",
        "category": "unknown_category",
        "response_action": "added_follow_up_steps",
    }

    state = _base_state(
        read_only_evidence,
        output_dir,
        detected_anomaly_data=detected_data,
    )

    result = record_anomaly(state)

    follow_up = result["_follow_up_steps"][0]
    assert "strings" in " ".join(follow_up.tool_cmd)
    assert "-n" in " ".join(follow_up.tool_cmd)
    assert "20" in " ".join(follow_up.tool_cmd)
