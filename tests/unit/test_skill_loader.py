from unittest.mock import patch

import pytest

from valravn.models.task import InvestigationPlan, InvestigationTask, PlannedStep
from valravn.nodes.skill_loader import SkillNotFoundError, load_skill


def _make_state(skill_domain: str, skill_content: str | None, read_only_evidence) -> dict:
    task = InvestigationTask(
        prompt="test",
        evidence_refs=[str(read_only_evidence)],
    )
    step = PlannedStep(skill_domain=skill_domain, tool_cmd=["fls"], rationale="r")
    plan = InvestigationPlan(task_id=task.id, steps=[step])
    return {
        "task": task,
        "plan": plan,
        "invocations": [],
        "anomalies": [],
        "report": None,
        "current_step_id": step.id,
        "skill_cache": {} if skill_content is None else {skill_domain: skill_content},
        "messages": [],
    }


def test_load_skill_reads_file(tmp_path, read_only_evidence):
    skill_file = tmp_path / "sleuthkit" / "SKILL.md"
    skill_file.parent.mkdir()
    skill_file.write_text("# Sleuth Kit Skill\nUse fls to list files.")

    state = _make_state("sleuthkit", None, read_only_evidence)

    with patch.dict("valravn.nodes.skill_loader.SKILL_PATHS", {"sleuthkit": skill_file}):
        result = load_skill(state)

    assert result["skill_cache"]["sleuthkit"] == "# Sleuth Kit Skill\nUse fls to list files."


def test_load_skill_uses_cache(read_only_evidence):
    state = _make_state("sleuthkit", "cached content", read_only_evidence)
    # File does not exist — should use cache without error
    result = load_skill(state)
    assert result["skill_cache"]["sleuthkit"] == "cached content"


def test_load_skill_unknown_domain(read_only_evidence):
    state = _make_state("unknown-domain", None, read_only_evidence)
    with pytest.raises(SkillNotFoundError):
        load_skill(state)
