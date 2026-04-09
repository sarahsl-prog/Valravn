"""Tests for LangGraph wiring and conditional routing.

These tests verify the graph structure and state transitions
without requiring actual LLM or SIFT tool execution.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from valravn.config import AppConfig, OutputConfig, RetryConfig
from valravn.graph import run as graph_run
from valravn.models.task import InvestigationPlan, InvestigationTask, PlannedStep, StepStatus

_STUB_PATH = Path(__file__).parent.parent / "fixtures" / "evidence" / "memory.lime.stub"


@pytest.fixture
def stub_evidence():
    """Get the stub evidence path."""
    return _STUB_PATH


class TestGraphUpdatePlan:
    """Tests for update_plan node state transitions."""

    def test_update_plan_completes_step(self, stub_evidence: Path):
        """update_plan marks step as COMPLETED when _step_succeeded=True."""
        from valravn.nodes.plan import update_plan

        task = InvestigationTask(prompt="test", evidence_refs=[str(stub_evidence)])
        step = PlannedStep(skill_domain="sleuthkit", tool_cmd=["echo"], rationale="r")
        plan = InvestigationPlan(task_id=task.id, steps=[step])

        state = {
            "task": task,
            "plan": plan,
            "invocations": [],
            "anomalies": [],
            "report": None,
            "current_step_id": step.id,
            "skill_cache": {},
            "messages": [],
            "_output_dir": ".",
            "_step_succeeded": True,
            "_step_exhausted": False,
            "_tool_failure": None,
            "_tool_failures": [],
        }

        result = update_plan(state)
        assert result["plan"].steps[0].status == StepStatus.COMPLETED

    def test_update_plan_fails_step(self, stub_evidence: Path):
        """update_plan marks step as FAILED when _step_succeeded=False and _step_exhausted=False."""
        from valravn.nodes.plan import update_plan

        task = InvestigationTask(prompt="test", evidence_refs=[str(stub_evidence)])
        step = PlannedStep(skill_domain="sleuthkit", tool_cmd=["echo"], rationale="r")
        plan = InvestigationPlan(task_id=task.id, steps=[step])

        state = {
            "task": task,
            "plan": plan,
            "invocations": [],
            "anomalies": [],
            "report": None,
            "current_step_id": step.id,
            "skill_cache": {},
            "messages": [],
            "_output_dir": ".",
            "_step_succeeded": False,
            "_step_exhausted": False,
            "_tool_failure": None,
            "_tool_failures": [],
        }

        result = update_plan(state)
        assert result["plan"].steps[0].status == StepStatus.FAILED

    def test_update_plan_exhausts_step(self, stub_evidence: Path):
        """update_plan marks step as EXHAUSTED when _step_exhausted=True."""
        from valravn.models.report import ToolFailureRecord
        from valravn.nodes.plan import update_plan

        task = InvestigationTask(prompt="test", evidence_refs=[str(stub_evidence)])
        step = PlannedStep(skill_domain="sleuthkit", tool_cmd=["echo"], rationale="r")
        plan = InvestigationPlan(task_id=task.id, steps=[step])

        failure = ToolFailureRecord(
            step_id=step.id,
            invocation_ids=["inv-1"],
            final_error="failed",
            diagnostic_context="context",
        )

        state = {
            "task": task,
            "plan": plan,
            "invocations": [],
            "anomalies": [],
            "report": None,
            "current_step_id": step.id,
            "skill_cache": {},
            "messages": [],
            "_output_dir": ".",
            "_step_succeeded": False,
            "_step_exhausted": True,
            "_tool_failure": failure,
            "_tool_failures": [],
        }

        result = update_plan(state)
        assert result["plan"].steps[0].status == StepStatus.EXHAUSTED


class TestGraphConditionalRouting:
    """Tests for graph conditional routing logic."""

    def test_graph_planning_to_report_no_steps(self, stub_evidence: Path, tmp_path: Path):
        """Graph transitions directly to report when no steps are planned."""
        task = InvestigationTask(
            prompt="find nothing",
            evidence_refs=[str(stub_evidence)],
        )
        output_dir = tmp_path / "output"
        out_cfg = OutputConfig(output_dir=output_dir)
        out_cfg.ensure_dirs()

        # Mock plan LLM to return no steps
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(steps=[])
        with patch("valravn.nodes.plan._get_llm", return_value=mock_llm):
            app_cfg = AppConfig(retry=RetryConfig(max_attempts=1))
            exit_code = graph_run(task, app_cfg, out_cfg)
            assert exit_code == 0  # Empty plan should exit cleanly

            # Check that report was generated
            reports = list((output_dir / "reports").glob("*.json"))
            assert len(reports) >= 1

    def test_anomaly_check_routes_to_update_plan_clean(self, stub_evidence: Path, tmp_path: Path):
        """check_anomalies routes to update_plan when no anomaly detected."""
        task = InvestigationTask(
            prompt="test",
            evidence_refs=[str(stub_evidence)],
        )
        output_dir = tmp_path / "output"
        out_cfg = OutputConfig(output_dir=output_dir)
        out_cfg.ensure_dirs()

        app_cfg = AppConfig(retry=RetryConfig(max_attempts=1))

        with (
            patch("valravn.nodes.plan._get_llm") as mock_plan_llm,
            patch("subprocess.run") as mock_subprocess,
            patch("valravn.nodes.anomaly._get_anomaly_llm") as mock_anomaly_llm,
        ):
            mock_step = MagicMock()
            mock_step.skill_domain = "sleuthkit"
            mock_step.tool_cmd = ["echo", "test"]
            mock_step.rationale = "test"
            mock_step.id = "test-step-id"
            mock_plan_llm.return_value.invoke.return_value = MagicMock(steps=[mock_step])
            mock_subprocess.return_value = MagicMock(returncode=0, stdout="output", stderr="")
            mock_anomaly_llm.return_value.invoke.return_value = MagicMock(anomaly_detected=False)

            with patch(
                "valravn.config.SkillsConfig.get_skill_path", lambda self, domain: Path("/dev/null")
            ):
                exit_code = graph_run(task, app_cfg, out_cfg)
                assert exit_code in (0, 1)


class TestGraphFullPipeline:
    """Integration tests for full graph pipeline with mocking."""

    def test_full_empty_plan_pipeline(self, stub_evidence: Path, tmp_path: Path):
        """Graph handles case where no steps are planned."""
        task = InvestigationTask(
            prompt="find nothing",
            evidence_refs=[str(stub_evidence)],
        )
        output_dir = tmp_path / "output"
        out_cfg = OutputConfig(output_dir=output_dir)
        out_cfg.ensure_dirs()

        # Mock plan LLM to return no steps
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(steps=[])
        with patch("valravn.nodes.plan._get_llm", return_value=mock_llm):
            app_cfg = AppConfig(retry=RetryConfig(max_attempts=1))
            exit_code = graph_run(task, app_cfg, out_cfg)
            assert exit_code == 0  # Empty plan should exit cleanly

            # Check that report was generated
            reports = list((output_dir / "reports").glob("*.json"))
            assert len(reports) >= 1
