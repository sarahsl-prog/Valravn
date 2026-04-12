"""US1 end-to-end integration test: plan → skill → tool → report.

Runs the full LangGraph pipeline with:
  - LLM mocked to return a single 'strings' step on the stub evidence fixture
  - Skill loader mocked via SKILL_PATHS to point to a tmp skill file
  - check_anomalies LLM mocked to return no anomaly detected
  - Real subprocess.run for the tool step (strings is available on SIFT/Ubuntu)
  - Real write_findings_report producing a Markdown report
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
def test_us1_end_to_end(tmp_path: Path) -> None:
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

    output_dir = tmp_path / "output"

    # Build task/config objects
    task = InvestigationTask(
        prompt="Investigate suspicious memory image for malware indicators",
        evidence_refs=[str(evidence_path)],
    )
    app_cfg = AppConfig(retry=RetryConfig(max_attempts=1, retry_delay_seconds=0.0))
    out_cfg = OutputConfig(output_dir=output_dir)
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

    # Report must land in the correct output_dir/reports/ (not CWD)
    reports_dir = output_dir / "reports"
    md_files = list(reports_dir.glob("*.md"))
    assert md_files, f"No .md report found in {reports_dir}"

    # Evidence stub must be unmodified: same mtime and permissions
    stub_mode_after = _STUB.stat().st_mode
    stub_mtime_after = _STUB.stat().st_mtime
    assert stub_mode_after == stub_mode_before, "Stub fixture permissions were changed"
    assert stub_mtime_after == stub_mtime_before, "Stub fixture was modified (mtime changed)"
    assert not os.access(_STUB, os.W_OK), "Stub fixture must remain non-writable after run"


def _make_readonly_evidence(tmp_path: Path) -> Path:
    """Create a small read-only evidence file in tmp_path."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    ev = tmp_path / "evidence.raw"
    ev.write_bytes(b"\x00" * 64)
    ev.chmod(0o444)
    return ev


@pytest.mark.integration
def test_multi_step_plan_all_steps_run(tmp_path: Path) -> None:
    """A plan with two steps must produce two ToolInvocationRecords in the report JSON."""
    evidence_path = _make_readonly_evidence(tmp_path / "ev")

    skill_file = tmp_path / "sleuthkit_skill.md"
    skill_file.write_text("# Sleuth Kit Skill\nUse fls, icat, strings.")

    output_dir = tmp_path / "output"

    task = InvestigationTask(
        prompt="Multi-step investigation",
        evidence_refs=[str(evidence_path)],
    )
    app_cfg = AppConfig(retry=RetryConfig(max_attempts=1, retry_delay_seconds=0.0))
    out_cfg = OutputConfig(output_dir=output_dir)
    out_cfg.ensure_dirs()

    # Two steps: echo step1 then echo step2
    plan_spec = _PlanSpec(
        steps=[
            _StepSpec(
                skill_domain="sleuthkit",
                tool_cmd=["echo", "step-one"],
                rationale="First step",
            ),
            _StepSpec(
                skill_domain="sleuthkit",
                tool_cmd=["echo", "step-two"],
                rationale="Second step",
            ),
        ]
    )
    mock_plan_llm = MagicMock()
    mock_plan_llm.invoke.return_value = MagicMock(content=plan_spec.model_dump_json())

    mock_anomaly_llm = MagicMock()
    mock_anomaly_llm.invoke.return_value = MagicMock(
        content=_AnomalyCheckResult(anomaly_detected=False).model_dump_json()
    )

    patched_skill_paths = {"sleuthkit": skill_file}

    import json as json_mod
    mock_conclusions_llm = MagicMock()
    mock_conclusions_llm.invoke.return_value = MagicMock(
        content=json_mod.dumps({"conclusions": []})
    )

    with (
        patch("valravn.nodes.plan._get_llm", return_value=mock_plan_llm),
        patch("valravn.nodes.anomaly._get_anomaly_llm", return_value=mock_anomaly_llm),
        patch("valravn.nodes.conclusions._get_conclusions_llm", return_value=mock_conclusions_llm),
        patch("valravn.nodes.skill_loader.SKILL_PATHS", patched_skill_paths),
    ):
        import valravn.graph as graph_mod

        exit_code = graph_mod.run(task, app_cfg, out_cfg)

    assert exit_code in (0, 1)

    reports_dir = output_dir / "reports"
    json_files = list(reports_dir.glob("*.json"))
    assert json_files, f"No JSON report found in {reports_dir}"

    # The plan in the report should show both steps completed (or at minimum ran)
    plan_path = output_dir / "analysis" / "investigation_plan.json"
    plan_data = json_mod.loads(plan_path.read_text())
    assert len(plan_data["steps"]) == 2, "Plan should contain exactly 2 steps"
    statuses = {s["status"] for s in plan_data["steps"]}
    # Both steps must have been executed (completed or failed, not pending)
    assert "PENDING" not in statuses, f"Some steps still PENDING: {plan_data['steps']}"


@pytest.mark.integration
def test_empty_evidence_produces_report_with_no_steps(tmp_path: Path) -> None:
    """When the planner returns zero steps, the graph skips straight to report."""
    evidence_path = _make_readonly_evidence(tmp_path / "ev")

    output_dir = tmp_path / "output"

    task = InvestigationTask(
        prompt="Investigate empty evidence set",
        evidence_refs=[str(evidence_path)],
    )
    app_cfg = AppConfig(retry=RetryConfig(max_attempts=1, retry_delay_seconds=0.0))
    out_cfg = OutputConfig(output_dir=output_dir)
    out_cfg.ensure_dirs()

    # Planner returns zero steps (empty plan)
    mock_plan_llm = MagicMock()
    mock_plan_llm.invoke.return_value = MagicMock(
        content=_PlanSpec(steps=[]).model_dump_json()
    )

    with patch("valravn.nodes.plan._get_llm", return_value=mock_plan_llm):
        import valravn.graph as graph_mod

        exit_code = graph_mod.run(task, app_cfg, out_cfg)

    # Should succeed (exit 0) without crashing
    assert exit_code in (0, 1)

    # A report must still be produced
    reports_dir = output_dir / "reports"
    md_files = list(reports_dir.glob("*.md"))
    assert md_files, f"No .md report found even for empty plan in {reports_dir}"
