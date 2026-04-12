from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel

from valravn.core.llm_factory import get_llm
from valravn.training.self_guide import POLARITY_REWARD_MAP, SelfGuidanceSignal

_SYSTEM_PROMPT = """\
You are an expert DFIR investigation analyst assessing the current progress of a forensic
investigation. Review the recent tool invocations and determine how well the investigation
is progressing toward its objectives.

Respond with valid JSON only — no markdown, no prose outside the JSON object.
Output exactly this structure:
{"assessment": "<single sentence>", "polarity": "<positive|neutral|negative>"}

polarity values:
  positive  — investigation is making clear forward progress
  neutral   — results are ambiguous or inconclusive
  negative  — investigation is stalled, failing, or going off-track
"""


class _AssessmentResult(BaseModel):
    assessment: str
    polarity: str  # "positive", "neutral", "negative"


def _parse_assessment(text: str) -> _AssessmentResult:
    """Parse LLM output into _AssessmentResult.

    Handles three formats models may emit:
    1. Clean JSON: {"assessment": "...", "polarity": "..."}
    2. Markdown-fenced JSON: ```json\\n{...}\\n```
    3. Plain key: value lines (fallback for non-compliant models)
    """
    stripped = text.strip()

    # Strip markdown code fences
    fenced = re.sub(r"^```[a-zA-Z]*\n?", "", stripped)
    fenced = re.sub(r"\n?```$", "", fenced).strip()

    # Attempt JSON parse
    for candidate in (fenced, stripped):
        try:
            data = json.loads(candidate)
            polarity = str(data.get("polarity", "neutral")).lower()
            if polarity not in ("positive", "neutral", "negative"):
                polarity = "neutral"
            return _AssessmentResult(
                assessment=str(data.get("assessment", candidate)),
                polarity=polarity,
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    # Fallback: parse "key: value" lines
    assessment_m = re.search(
        r"(?i)\*{0,2}assessment\*{0,2}[:\s]+(.+?)(?:\n|$)", text, re.DOTALL
    )
    polarity_m = re.search(r"(?i)\*{0,2}polarity\*{0,2}[:\s]+(\w+)", text)

    assessment = assessment_m.group(1).strip() if assessment_m else text.strip()
    polarity = polarity_m.group(1).strip().lower() if polarity_m else "neutral"
    if polarity not in ("positive", "neutral", "negative"):
        polarity = "neutral"

    return _AssessmentResult(assessment=assessment, polarity=polarity)


def assess_progress(state: dict) -> dict:
    """LangGraph node: self-assess investigation progress before each tool execution."""
    plan = state.get("plan")
    current_step_id = state.get("current_step_id")
    logger.info("Node: assess_progress | step={}", (current_step_id or "none")[:8])

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
        "Assess the current investigation progress. Respond with JSON only."
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]

    # Self-assessment is non-critical (observability only). If the LLM call
    # fails, log the error and continue rather than crashing the graph.
    try:
        llm = get_llm(module="self_assess")
        response = llm.invoke(messages)
        result = _parse_assessment(response.content)
    except Exception:
        logger.warning("Self-assessment LLM failed for step={}; skipping", current_step_id[:8])
        return {"_self_assessments": list(state.get("_self_assessments") or [])}

    signal = SelfGuidanceSignal(
        assessment=result.assessment,
        polarity=result.polarity,
        scalar_reward=POLARITY_REWARD_MAP.get(result.polarity, 0.0),
    )

    existing: list[dict] = list(state.get("_self_assessments") or [])
    updated = existing + [signal.model_dump()]

    return {"_self_assessments": updated}
