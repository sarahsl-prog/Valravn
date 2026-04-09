from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from valravn.models.records import Anomaly, AnomalyResponseAction, ToolInvocationRecord
from valravn.models.task import InvestigationTask
from valravn.nodes.conclusions import synthesize_conclusions

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_invocation(stdout_path: Path, inv_id: str = "inv-001") -> ToolInvocationRecord:
    now = datetime.now(timezone.utc)
    inv = ToolInvocationRecord(
        step_id="step-001",
        attempt_number=1,
        cmd=["fls", "-r", "/evidence.raw"],
        exit_code=0,
        stdout_path=stdout_path,
        stderr_path=stdout_path,
        started_at_utc=now,
        completed_at_utc=now,
        duration_seconds=0.5,
        had_output=True,
    )
    inv.id = inv_id
    return inv


def _make_llm_response(statements: list[str]) -> MagicMock:
    """Build a mock LLM response whose .content is valid JSON for _ConclusionsOutput."""
    conclusions = [
        {
            "statement": stmt,
            "supporting_invocation_ids": [f"inv-{i:03d}"],
            "confidence": "high",
        }
        for i, stmt in enumerate(statements)
    ]
    return MagicMock(content=json.dumps({"conclusions": conclusions}))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_synthesize_conclusions_empty_when_no_invocations(read_only_evidence):
    """When no invocations exist, return empty list and do NOT call the LLM."""
    task = InvestigationTask(
        prompt="test", evidence_refs=[str(read_only_evidence)]
    )
    state = {
        "invocations": [],
        "task": task,
        "anomalies": [],
    }

    with patch("valravn.nodes.conclusions._get_conclusions_llm") as mock_llm_fn:
        result = synthesize_conclusions(state)

    mock_llm_fn.assert_not_called()
    assert result == {"_conclusions": []}


def test_synthesize_conclusions_produces_conclusion_dicts(read_only_evidence, tmp_path):
    """With invocations, the LLM is called and conclusions are returned as dicts."""
    stdout_file = tmp_path / "stdout.txt"
    stdout_file.write_text("d/d 5:\tUsers\nd/d 6:\tWindows\n")

    inv = _make_invocation(stdout_file, inv_id="inv-abc")

    task = InvestigationTask(
        prompt="investigate malware", evidence_refs=[str(read_only_evidence)]
    )

    state = {
        "invocations": [inv],
        "task": task,
        "anomalies": [],
    }

    mock_response = _make_llm_response(["Malware artifacts found in Users directory"])

    with patch("valravn.nodes.conclusions._get_conclusions_llm") as mock_llm_fn:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_llm_fn.return_value = mock_llm

        result = synthesize_conclusions(state)

    mock_llm_fn.assert_called_once()
    mock_llm.invoke.assert_called_once()

    assert "_conclusions" in result
    conclusions = result["_conclusions"]
    assert isinstance(conclusions, list)
    assert len(conclusions) == 1

    c = conclusions[0]
    assert "statement" in c
    assert "supporting_invocation_ids" in c
    assert "confidence" in c
    assert c["statement"] == "Malware artifacts found in Users directory"
    assert c["confidence"] == "high"


def test_synthesize_conclusions_includes_anomalies_in_prompt(read_only_evidence, tmp_path):
    """Anomaly descriptions are included in the LLM prompt."""
    stdout_file = tmp_path / "out.txt"
    stdout_file.write_text("some output")

    inv = _make_invocation(stdout_file)
    task = InvestigationTask(
        prompt="test investigation", evidence_refs=[str(read_only_evidence)]
    )

    anomaly = Anomaly(
        description="Timestamp contradiction in $STANDARD_INFORMATION",
        forensic_significance="Possible timestomping",
        source_invocation_ids=["inv-001"],
        response_action=AnomalyResponseAction.ADDED_FOLLOW_UP,
    )

    state = {
        "invocations": [inv],
        "task": task,
        "anomalies": [anomaly],
    }

    mock_response = _make_llm_response([])
    captured_messages = []

    def capture_invoke(messages):
        captured_messages.extend(messages)
        return mock_response

    with patch("valravn.nodes.conclusions._get_conclusions_llm") as mock_llm_fn:
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = capture_invoke
        mock_llm_fn.return_value = mock_llm

        synthesize_conclusions(state)

    assert len(captured_messages) == 2
    human_msg = captured_messages[1]
    assert "Timestamp contradiction" in human_msg.content
    assert "timestomping" in human_msg.content


def test_synthesize_conclusions_truncates_long_stdout(read_only_evidence, tmp_path):
    """stdout is truncated to MAX_STDOUT_CHARS (10,000) per invocation."""
    stdout_file = tmp_path / "big.txt"
    # Write more than 10,000 chars
    stdout_file.write_text("A" * 20_000)

    inv = _make_invocation(stdout_file)
    task = InvestigationTask(
        prompt="check big output", evidence_refs=[str(read_only_evidence)]
    )

    state = {
        "invocations": [inv],
        "task": task,
        "anomalies": [],
    }

    mock_response = _make_llm_response([])
    captured_messages = []

    def capture_invoke(messages):
        captured_messages.extend(messages)
        return mock_response

    with patch("valravn.nodes.conclusions._get_conclusions_llm") as mock_llm_fn:
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = capture_invoke
        mock_llm_fn.return_value = mock_llm

        synthesize_conclusions(state)

    human_content = captured_messages[1].content
    # The 20,000-char string should be truncated, so AAAA... block <= 10,000 chars
    assert human_content.count("A") <= 10_000


def test_synthesize_conclusions_handles_missing_stdout(read_only_evidence, tmp_path):
    """Node handles missing stdout files gracefully (no exception)."""
    missing_path = tmp_path / "nonexistent.txt"

    inv = _make_invocation(missing_path)
    task = InvestigationTask(
        prompt="test", evidence_refs=[str(read_only_evidence)]
    )

    state = {
        "invocations": [inv],
        "task": task,
        "anomalies": [],
    }

    mock_response = _make_llm_response([])

    with patch("valravn.nodes.conclusions._get_conclusions_llm") as mock_llm_fn:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_llm_fn.return_value = mock_llm

        result = synthesize_conclusions(state)

    assert "_conclusions" in result
    assert isinstance(result["_conclusions"], list)
