from unittest.mock import patch

import pytest

from valravn.config import SkillsConfig
from valravn.models.task import InvestigationPlan, InvestigationTask, PlannedStep
from valravn.nodes.skill_loader import SkillNotFoundError, load_skill


def _make_state(
    skill_domain: str,
    skill_content: str | None,
    read_only_evidence,
    skills_config: SkillsConfig | None = None,
) -> dict:
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
        "_skills_config": skills_config,
    }


def test_load_skill_reads_file(tmp_path, read_only_evidence):
    skill_file = tmp_path / "sleuthkit" / "SKILL.md"
    skill_file.parent.mkdir()
    skill_file.write_text("# Sleuth Kit Skill\nUse fls to list files.")

    skills_config = SkillsConfig(base_path=tmp_path)
    state = _make_state("sleuthkit", None, read_only_evidence, skills_config)

    result = load_skill(state)

    assert result["skill_cache"]["sleuthkit"] == "# Sleuth Kit Skill\nUse fls to list files."


def test_load_skill_uses_cache(read_only_evidence):
    state = _make_state("sleuthkit", "cached content", read_only_evidence)
    # File does not exist — should use cache without error
    result = load_skill(state)
    assert result["skill_cache"]["sleuthkit"] == "cached content"


def test_load_skill_unknown_domain(read_only_evidence):
    # Test with a valid base_path but unknown domain
    skills_config = SkillsConfig()
    state = _make_state("unknown-domain", None, read_only_evidence, skills_config)
    with pytest.raises(SkillNotFoundError):
        load_skill(state)


def test_load_skill_missing_file(read_only_evidence, tmp_path):
    """Test that missing skill file raises error when not in cache."""
    skills_config = SkillsConfig(base_path=tmp_path)
    state = _make_state("sleuthkit", None, read_only_evidence, skills_config)

    with pytest.raises(SkillNotFoundError, match="Skill file not found"):
        load_skill(state)


def test_load_skill_uses_default_config(read_only_evidence, tmp_path):
    """Test that default config is used when _skills_config is None."""
    skill_file = tmp_path / "sleuthkit" / "SKILL.md"
    skill_file.parent.mkdir()
    skill_file.write_text("# Default Config Skill")

    # Patch the get_skill_path method to use our temp path
    with patch.object(
        SkillsConfig, "get_skill_path", lambda self, domain: tmp_path / "sleuthkit" / "SKILL.md"
    ):
        state = _make_state("sleuthkit", None, read_only_evidence, None)
        result = load_skill(state)

    assert result["skill_cache"]["sleuthkit"] == "# Default Config Skill"
