from __future__ import annotations

import json
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from valravn.models.records import Anomaly, AnomalyResponseAction
from valravn.models.task import PlannedStep

_SYSTEM_PROMPT = """\
You are an expert DFIR analyst reviewing forensic tool output on a SANS SIFT workstation.
Analyse the tool output below for anomalies in these categories:
  1. timestamp_contradiction  — timestamps that are implausible or conflict with each other
  2. orphaned_relationship    — process/object with no valid parent or owning entity
  3. cross_tool_conflict      — findings that contradict results from another tool
  4. unexpected_absence       — expected artifacts are entirely missing from output
  5. integrity_failure        — hash mismatches, corrupted records, or truncated data

Return structured output indicating whether an anomaly was detected and, if so, a
concise description, its forensic significance, and which category applies.
"""


class _AnomalyCheckResult(BaseModel):
    anomaly_detected: bool
    description: str = ""
    forensic_significance: str = ""
    category: str = ""  # one of the 5 categories listed above


def _get_anomaly_llm() -> object:
    llm = ChatAnthropic(model="claude-opus-4-6", temperature=0)
    return llm.with_structured_output(_AnomalyCheckResult)


def check_anomalies(state: dict) -> dict:
    """LangGraph node: inspect the most recent tool output for forensic anomalies."""
    invocations = state.get("invocations") or []
    if not invocations:
        return {"_pending_anomalies": False, "_detected_anomaly_data": None}

    invocation = invocations[-1]

    stdout_path = Path(invocation.stdout_path)
    tool_output = stdout_path.read_text(errors="replace") if stdout_path.exists() else ""

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Tool command: {' '.join(str(c) for c in invocation.cmd)}\n\n"
                f"Tool output:\n{tool_output}"
            )
        ),
    ]

    result: _AnomalyCheckResult = _get_anomaly_llm().invoke(messages)

    if result.anomaly_detected:
        return {
            "_pending_anomalies": True,
            "_detected_anomaly_data": result.model_dump(),
        }

    return {"_pending_anomalies": False, "_detected_anomaly_data": None}


def record_anomaly(state: dict) -> dict:
    """LangGraph node: persist a detected anomaly and queue a follow-up step."""
    data: dict = state.get("_detected_anomaly_data") or {}
    last_inv_id: str = state.get("_last_invocation_id") or ""
    output_dir = Path(state.get("_output_dir", "."))

    # Embed the category in description so it is preserved without adding a new field.
    category = data.get("category", "")
    description = data.get("description", "")
    if category and category not in description:
        description = f"[{category}] {description}"

    anomaly = Anomaly(
        description=description,
        forensic_significance=data.get("forensic_significance", ""),
        source_invocation_ids=[last_inv_id] if last_inv_id else ["unknown"],
        response_action=AnomalyResponseAction.ADDED_FOLLOW_UP,
    )

    updated_anomalies: list[Anomaly] = list(state.get("anomalies") or []) + [anomaly]

    # Persist all anomalies to disk.
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    anomalies_path = analysis_dir / "anomalies.json"
    anomalies_path.write_text(
        json.dumps(
            [a.model_dump(mode="json") for a in updated_anomalies],
            indent=2,
            default=str,
        )
    )

    # Build a follow-up step when the response action warrants it.
    follow_up_steps: list[PlannedStep] = []
    if anomaly.response_action == AnomalyResponseAction.ADDED_FOLLOW_UP:
        plan = state.get("plan")
        current_step_id = state.get("current_step_id")
        current_step: PlannedStep | None = None
        if plan is not None and current_step_id:
            for s in plan.steps:
                if s.id == current_step_id:
                    current_step = s
                    break

        skill_domain = current_step.skill_domain if current_step else "sleuthkit"
        evidence_refs = state.get("task").evidence_refs if state.get("task") else []
        evidence_path = evidence_refs[0] if evidence_refs else "/evidence"

        follow_up = PlannedStep(
            skill_domain=skill_domain,
            tool_cmd=["strings", "-n", "20", evidence_path],
            rationale=f"Follow-up investigation of anomaly: {anomaly.description}",
        )
        follow_up_steps = [follow_up]

    return {
        "anomalies": updated_anomalies,
        "_follow_up_steps": follow_up_steps,
        "_pending_anomalies": False,
    }
