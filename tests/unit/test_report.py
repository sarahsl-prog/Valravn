from datetime import datetime, timezone

import pytest

from valravn.models.records import ToolInvocationRecord
from valravn.models.task import InvestigationPlan, InvestigationTask, PlannedStep
from valravn.nodes.report import write_findings_report


def _state_with_invocation(read_only_evidence, output_dir):
    task = InvestigationTask(
        prompt="find connections",
        evidence_refs=[str(read_only_evidence)],
    )
    step = PlannedStep(
        skill_domain="memory-analysis",
        tool_cmd=["vol.py", "netstat"],
        rationale="r",
    )
    plan = InvestigationPlan(task_id=task.id, steps=[step])
    inv = ToolInvocationRecord(
        id="inv-1",
        step_id=step.id,
        attempt_number=1,
        cmd=["vol.py", "netstat"],
        exit_code=0,
        stdout_path=output_dir / "inv-1.stdout",
        stderr_path=output_dir / "inv-1.stderr",
        started_at_utc=datetime.now(timezone.utc),
        completed_at_utc=datetime.now(timezone.utc),
        duration_seconds=1.0,
        had_output=True,
    )
    return {
        "task": task,
        "plan": plan,
        "invocations": [inv],
        "anomalies": [],
        "report": None,
        "current_step_id": None,
        "skill_cache": {},
        "messages": [],
        "_output_dir": str(output_dir),
        "_conclusions": [
            {
                "statement": "TCP connection to 10.0.0.1:445",
                "supporting_invocation_ids": ["inv-1"],
                "confidence": "high",
            }
        ],
        "_tool_failures": [],
        "_self_corrections": [],
    }


def test_report_written_to_reports_dir(read_only_evidence, output_dir):
    state = _state_with_invocation(read_only_evidence, output_dir)
    write_findings_report(state)
    reports = list((output_dir / "reports").glob("*.md"))
    assert len(reports) == 1


def test_report_json_also_written(read_only_evidence, output_dir):
    state = _state_with_invocation(read_only_evidence, output_dir)
    write_findings_report(state)
    jsons = list((output_dir / "reports").glob("*.json"))
    assert len(jsons) == 1


def test_report_timestamp_utc(read_only_evidence, output_dir):
    state = _state_with_invocation(read_only_evidence, output_dir)
    result = write_findings_report(state)
    assert result["report"].generated_at_utc.tzinfo == timezone.utc


def test_report_conclusion_cites_invocation(read_only_evidence, output_dir):
    state = _state_with_invocation(read_only_evidence, output_dir)
    result = write_findings_report(state)
    assert result["report"].conclusions[0].supporting_invocation_ids == ["inv-1"]


def test_conclusion_without_citation_raises():
    from valravn.models.report import Conclusion

    with pytest.raises(ValueError, match="cite"):
        Conclusion(statement="test", supporting_invocation_ids=[], confidence="high")
