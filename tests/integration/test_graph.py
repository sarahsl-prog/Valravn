"""US1 end-to-end integration test: plan → skill → tool → report.

Runs the full LangGraph pipeline with:
  - LLM mocked to return a single 'strings' step on the stub evidence fixture
  - Skill loader mocked via SKILL_PATHS to point to a tmp skill file
  - check_anomalies LLM mocked to return no anomaly detected
  - Real subprocess.run for the tool step (strings is available on SIFT/Ubuntu)
  - Real write_findings_report producing a Markdown report

Note: LangGraph strips keys not present in AgentState TypedDict before passing
state to nodes. The ``_output_dir`` key set in ``graph.run()`` is therefore
unavailable inside nodes, which fall back to ``Path(".")`` (CWD).  The test
changes CWD to ``tmp_path`` so that all output lands there, making assertions
straightforward without modifying production code.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from valravn.config import AppConfig, OutputConfig, RetryConfig
from valravn.models.task import InvestigationTask
from valravn.nodes.anomaly import _AnomalyCheckResult
from valravn.nodes.plan import _PlanSpec, _StepSpec

# Absolute path to the committed stub fixture
_STUB = Path(__file__).parent.parent / "fixtures" / "evidence" / "memory.lime.stub"


@pytest.mark.integration
def test_us1_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Full graph run on a synthetic evidence stub using 'strings'."""
    # Confirm fixture exists and is read-only before the run
    assert _STUB.exists(), f"Stub fixture missing: {_STUB}"
    stub_mode_before = _STUB.stat().st_mode
    stub_mtime_before = _STUB.stat().st_mtime
    assert not os.access(_STUB, os.W_OK), "Stub fixture must not be writable before run"

    evidence_path = _STUB

    # Write a dummy skill file so SKILL_PATHS resolves correctly
    skill_file = tmp_path / "sleuthkit_skill.md"
    skill_file.write_text("# Sleuth Kit Skill\nUse fls, icat, strings.")

    # Change CWD to tmp_path so that node fallback Path(".") writes here
    monkeypatch.chdir(tmp_path)

    # Build task/config objects
    task = InvestigationTask(
        prompt="Investigate suspicious memory image for malware indicators",
        evidence_refs=[str(evidence_path)],
    )
    app_cfg = AppConfig(retry=RetryConfig(max_attempts=1, retry_delay_seconds=0.0))
    out_cfg = OutputConfig(output_dir=tmp_path / "output")
    out_cfg.ensure_dirs()

    # Mock the planning LLM to return one planned step: strings on the stub
    mock_plan_llm = MagicMock()
    mock_plan_llm.invoke.return_value = _PlanSpec(
        steps=[
            _StepSpec(
                skill_domain="sleuthkit",
                tool_cmd=["strings", str(evidence_path)],
                rationale="Extract printable strings from memory fixture",
            )
        ]
    )

    # Mock the anomaly LLM to return no anomaly (avoids API call and follow-up loops)
    mock_anomaly_llm = MagicMock()
    mock_anomaly_llm.invoke.return_value = _AnomalyCheckResult(anomaly_detected=False)

    # Patch SKILL_PATHS so the skill loader finds the domain without reading ~/.claude
    patched_skill_paths = {"sleuthkit": skill_file}

    with (
        patch("valravn.nodes.plan._get_llm", return_value=mock_plan_llm),
        patch("valravn.nodes.anomaly._get_anomaly_llm", return_value=mock_anomaly_llm),
        patch("valravn.nodes.skill_loader.SKILL_PATHS", patched_skill_paths),
    ):
        import valravn.graph as graph_mod

        exit_code = graph_mod.run(task, app_cfg, out_cfg)

    # Exit code must be 0 or 1 — never an exception
    assert exit_code in (0, 1), f"Unexpected exit code: {exit_code}"

    # Nodes fall back to CWD (tmp_path) when _output_dir is unavailable in AgentState.
    # Assert that a Markdown report exists under reports/ relative to CWD.
    reports_dir = tmp_path / "reports"
    md_files = list(reports_dir.glob("*.md"))
    assert md_files, f"No .md report found in {reports_dir}"

    # Evidence stub must be unmodified: same mtime and permissions
    stub_mode_after = _STUB.stat().st_mode
    stub_mtime_after = _STUB.stat().st_mtime
    assert stub_mode_after == stub_mode_before, "Stub fixture permissions were changed"
    assert stub_mtime_after == stub_mtime_before, "Stub fixture was modified (mtime changed)"
    assert not os.access(_STUB, os.W_OK), "Stub fixture must remain non-writable after run"
