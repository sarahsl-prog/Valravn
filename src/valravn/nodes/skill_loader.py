from __future__ import annotations

from pathlib import Path

from loguru import logger

_SKILLS_BASE = Path.home() / ".claude" / "skills"

SKILL_PATHS: dict[str, Path] = {
    "memory-analysis": _SKILLS_BASE / "memory-analysis" / "SKILL.md",
    "sleuthkit": _SKILLS_BASE / "sleuthkit" / "SKILL.md",
    "windows-artifacts": _SKILLS_BASE / "windows-artifacts" / "SKILL.md",
    "plaso-timeline": _SKILLS_BASE / "plaso-timeline" / "SKILL.md",
    "yara-hunting": _SKILLS_BASE / "yara-hunting" / "SKILL.md",
}


class SkillNotFoundError(Exception):
    pass


def load_skill(state: dict) -> dict:
    """LangGraph node: load SKILL.md for the current step's domain into skill_cache."""
    step_id = state["current_step_id"]
    step = next((s for s in state["plan"].steps if s.id == step_id), None)
    if step is None:
        raise KeyError(step_id)
    logger.info("Node: load_skill | domain={} step={}", step.skill_domain, step_id[:8])
    domain = step.skill_domain

    cache: dict[str, str] = dict(state.get("skill_cache") or {})

    if domain in cache:
        return {"skill_cache": cache}

    if domain not in SKILL_PATHS:
        raise SkillNotFoundError(
            f"No skill file registered for domain '{domain}'. "
            f"Add it to SKILL_PATHS in nodes/skill_loader.py."
        )

    skill_path = SKILL_PATHS[domain]
    if not skill_path.exists():
        raise SkillNotFoundError(f"Skill file not found: {skill_path}")

    cache[domain] = skill_path.read_text()
    return {"skill_cache": cache}
