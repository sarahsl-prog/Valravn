from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from valravn.training.self_guide import POLARITY_REWARD_MAP, SelfGuidanceSignal

_SYSTEM_PROMPT = """\
You are an expert DFIR investigation analyst assessing the current progress of a forensic
investigation. Review the recent tool invocations and determine how well the investigation
is progressing toward its objectives.

Provide:
  - assessment : a concise single-sentence evaluation of the current investigation progress
  - polarity   : one of "positive", "neutral", or "negative"
                 positive  — investigation is making clear forward progress
                 neutral   — results are ambiguous or inconclusive
                 negative  — investigation is stalled, failing, or going off-track
"""


class _AssessmentResult(BaseModel):
    assessment: str
    polarity: str  # "positive", "neutral", "negative"


def _get_assessor_llm() -> object:
    from langchain_anthropic import ChatAnthropic

    llm = ChatAnthropic(model="claude-opus-4-6", temperature=0)
    return llm.with_structured_output(_AssessmentResult)


def assess_progress(state: dict) -> dict:
    """LangGraph node: self-assess investigation progress before each tool execution."""
    plan = state.get("plan")
    current_step_id = state.get("current_step_id")

    # If no current step, nothing to assess — return existing assessments unchanged
    if current_step_id is None:
        return {"_self_assessments": list(state.get("_self_assessments") or [])}

    # Build a compact history from the last 5 invocations
    invocations: list = list(state.get("invocations") or [])
    recent = invocations[-5:] if len(invocations) > 5 else invocations

    history_lines: list[str] = []
    for inv in recent:
        outcome = "success" if getattr(inv, "success", False) else "failure"
        history_lines.append(
            f"  - step={inv.step_id} attempt={inv.attempt_number} "
            f"cmd={inv.cmd} exit_code={inv.exit_code} outcome={outcome}"
        )

    history_text = "\n".join(history_lines) if history_lines else "  (no prior invocations)"

    # Find the current step to surface its description
    current_step = None
    if plan is not None:
        current_step = next(
            (s for s in plan.steps if s.id == current_step_id), None
        )

    step_description = (
        getattr(current_step, "description", current_step_id)
        if current_step is not None
        else current_step_id
    )

    human_content = (
        f"## Current step\n{step_description} (id={current_step_id})\n\n"
        f"## Recent tool invocations (last 5)\n{history_text}\n\n"
        "Assess the current investigation progress."
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]

    result: _AssessmentResult = _get_assessor_llm().invoke(messages)

    signal = SelfGuidanceSignal(
        assessment=result.assessment,
        polarity=result.polarity,
        scalar_reward=POLARITY_REWARD_MAP.get(result.polarity, 0.0),
    )

    existing: list[dict] = list(state.get("_self_assessments") or [])
    updated = existing + [signal.model_dump()]

    return {"_self_assessments": updated}
