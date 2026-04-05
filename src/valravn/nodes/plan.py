from __future__ import annotations

import json
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from valravn.models.task import (
    InvestigationPlan,
    PlannedStep,
    StepStatus,
)


class _StepSpec(BaseModel):
    skill_domain: str
    tool_cmd: list[str]
    rationale: str


class _PlanSpec(BaseModel):
    steps: list[_StepSpec]


_SYSTEM_PROMPT = """\
You are an expert DFIR analyst on a SANS SIFT Ubuntu workstation.
Given an investigation prompt and evidence paths, return an ordered list of forensic
tool invocations to execute.

Rules:
- Use ONLY tools available on SIFT (Volatility 3, fls, icat, log2timeline.py, yara, etc.)
- Each step must target ONE specific forensic question
- skill_domain must be one of: memory-analysis, sleuthkit, windows-artifacts, plaso-timeline, yara-hunting
- tool_cmd must be the exact subprocess argv (list of strings)
- Do NOT include evidence paths as output destinations
"""


def _get_llm() -> object:
    llm = ChatAnthropic(model="claude-opus-4-6", temperature=0)
    return llm.with_structured_output(_PlanSpec)


def plan_investigation(state: dict) -> dict:
    """LangGraph node: derive initial investigation plan from prompt."""
    task = state["task"]
    output_dir = Path(state.get("_output_dir", "."))

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Investigation prompt: {task.prompt}\n"
                f"Evidence paths: {', '.join(task.evidence_refs)}"
            )
        ),
    ]

    plan_spec: _PlanSpec = _get_llm().invoke(messages)

    steps = [
        PlannedStep(
            skill_domain=s.skill_domain,
            tool_cmd=s.tool_cmd,
            rationale=s.rationale,
        )
        for s in plan_spec.steps
    ]

    plan = InvestigationPlan(task_id=task.id, steps=steps)
    _persist_plan(plan, output_dir)

    first_step_id = steps[0].id if steps else None

    return {
        "plan": plan,
        "current_step_id": first_step_id,
    }


def update_plan(state: dict) -> dict:
    """LangGraph node: mark current step complete/failed/exhausted; advance pointer."""
    plan: InvestigationPlan = state["plan"]
    step_id: str = state["current_step_id"]
    succeeded: bool = state.get("_step_succeeded", False)
    exhausted: bool = state.get("_step_exhausted", False)
    output_dir = Path(state.get("_output_dir", "."))

    if exhausted:
        plan.mark_step(step_id, StepStatus.EXHAUSTED)
    elif succeeded:
        plan.mark_step(step_id, StepStatus.COMPLETED)
    else:
        plan.mark_step(step_id, StepStatus.FAILED)

    # Collect tool failure if present
    tool_failures = list(state.get("_tool_failures") or [])
    if state.get("_tool_failure") is not None:
        tool_failures.append(state["_tool_failure"])

    # Append follow-up steps if any (T016)
    follow_up_steps = list(state.get("_follow_up_steps") or [])
    if follow_up_steps:
        plan.add_steps(follow_up_steps)

    next_step = plan.next_pending_step()
    next_id = next_step.id if next_step else None

    _persist_plan(plan, output_dir)

    return {
        "plan": plan,
        "current_step_id": next_id,
        "_step_succeeded": False,
        "_step_exhausted": False,
        "_pending_anomalies": False,
        "_tool_failure": None,
        "_tool_failures": tool_failures,
        "_follow_up_steps": [],
    }


def _persist_plan(plan: InvestigationPlan, output_dir: Path) -> None:
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    plan_path = analysis_dir / "investigation_plan.json"
    plan_path.write_text(
        json.dumps(plan.model_dump(mode="json"), indent=2, default=str)
    )
