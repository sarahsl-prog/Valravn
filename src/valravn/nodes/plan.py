from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel

from valravn.core.llm_factory import get_llm

from valravn.models.task import InvestigationPlan, PlannedStep, StepStatus


class _StepSpec(BaseModel):
    skill_domain: str
    tool_cmd: list[str]
    rationale: str


class _PlanSpec(BaseModel):
    steps: list[_StepSpec]


_SYSTEM_PROMPT = """\
You are an expert DFIR analyst on a SANS SIFT Ubuntu workstation.
Given an investigation prompt and evidence paths, produce a JSON object with a "steps"
array describing the forensic tool invocations to execute.

Rules:
- Use ONLY tools available on SIFT (Volatility 3, fls, icat, log2timeline.py, yara, etc.)
- Each step must address one focused forensic objective
- skill_domain must be one of: memory-analysis, sleuthkit, windows-artifacts,
  plaso-timeline, yara-hunting
- tool_cmd must be the exact subprocess argv (list of strings)
- Do NOT include evidence paths as output destinations
- ALL tool output files (--storage-file, -w, --output, etc.) MUST use the
  analysis_dir provided in the human message — never /tmp/, never hardcode paths

CRITICAL TOOL SYNTAX — follow exactly:

log2timeline.py (Plaso 20240308, GIFT PPA):
  Flags MUST come before the source path. Source path is the LAST argument.
  CORRECT:   ["log2timeline.py", "--storage-file", "./analysis/case.plaso", "--parsers", "win10", "--timezone", "UTC", "/mnt/ewf/ewf1"]
  WRONG:     ["log2timeline.py", "/mnt/ewf/ewf1", "--storage-file", "./analysis/case.plaso"]

fls (Sleuth Kit):
  CORRECT:   ["fls", "-r", "-m", "/", "/mnt/ewf/ewf1"]

icat (Sleuth Kit):
  CORRECT:   ["icat", "/mnt/ewf/ewf1", "<inode>"]

vol.py (Volatility 3):
  CORRECT:   ["python3", "/opt/volatility3-2.20.0/vol.py", "-f", "<memory_image>", "windows.pslist.PsList"]

yara:
  CORRECT:   ["/usr/local/bin/yara", "-r", "<rules.yar>", "<target_path>"]

Respond with a JSON object matching exactly this structure:
{
  "steps": [
    {
      "skill_domain": "<one of the domains above>",
      "tool_cmd": ["<executable>", "<arg1>", "<arg2>"],
      "rationale": "<why this step answers the investigation prompt>"
    }
  ]
}
"""


def _get_llm():
    """Get LLM for investigation planning with structured output."""
    return get_llm(module="plan", output_schema=_PlanSpec)


def plan_investigation(state: dict) -> dict:
    """LangGraph node: derive initial investigation plan from prompt."""
    task = state["task"]
    logger.info("Node: plan_investigation | prompt={!r}", task.prompt[:80])
    output_dir = Path(state.get("_output_dir", "."))

    analysis_dir = output_dir / "analysis"

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Investigation prompt: {task.prompt}\n"
                f"Evidence paths: {', '.join(task.evidence_refs)}\n"
                f"Analysis output directory: {analysis_dir}\n"
                f"Use {analysis_dir}/<filename> for ALL tool output files "
                f"(--storage-file, bodyfile, etc.)."
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
    import json

    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    plan_path = analysis_dir / "investigation_plan.json"
    plan_path.write_text(
        json.dumps(plan.model_dump(mode="json"), indent=2, default=str)
    )
