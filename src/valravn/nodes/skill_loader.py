from __future__ import annotations

from pathlib import Path

from loguru import logger

from valravn.config import SkillsConfig


class SkillNotFoundError(Exception):
    pass


def load_skill(state: dict) -> dict:
    """LangGraph node: load SKILL.md for the current step's domain into skill_cache."""
    step_id = state["current_step_id"]
    step = next(s for s in state["plan"].steps if s.id == step_id)
    logger.info("Node: load_skill | domain={} step={}", step.skill_domain, step_id[:8])
    domain = step.skill_domain

    cache: dict[str, str] = dict(state.get("skill_cache") or {})

    if domain in cache:
        return {"skill_cache": cache}

    # Get skills config from state or use defaults
    skills_config = state.get("_skills_config")
    if skills_config is None:
        skills_config = SkillsConfig()

    skill_path = skills_config.get_skill_path(domain)
    if skill_path is None:
        raise SkillNotFoundError(
            f"No skill file registered for domain '{domain}'. Add it to SkillsConfig.known_domains."
        )

    if not skill_path.exists():
        raise SkillNotFoundError(f"Skill file not found: {skill_path}")

    cache[domain] = skill_path.read_text()
    return {"skill_cache": cache}
