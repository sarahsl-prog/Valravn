from __future__ import annotations

import json

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from valravn.core.llm_factory import get_llm
from valravn.core.parsing import parse_llm_json

from valravn.models.records import Anomaly, AnomalyResponseAction
from valravn.models.task import PlannedStep, StepStatus

_SYSTEM_PROMPT = """\
You are an expert DFIR analyst reviewing forensic tool output on a SANS SIFT workstation.
Analyse the tool output below for anomalies in these categories:
  1. timestamp_contradiction  — timestamps that are implausible or conflict with each other
  2. orphaned_relationship    — process/object with no valid parent or owning entity
  3. cross_tool_conflict      — findings that contradict results from another tool
  4. unexpected_absence       — expected artifacts are entirely missing from output
  5. integrity_failure        — hash mismatches, corrupted records, or truncated data

Respond with valid JSON only — no markdown, no prose outside the JSON object.
Output exactly this structure:
{
  "anomaly_detected": <true|false>,
  "description": "<concise description, or empty string>",
  "forensic_significance": "<significance, or empty string>",
  "category": "<one of the 5 categories above, or empty string>",
  "response_action": "<added_follow_up_steps|no_follow_up_warranted|investigation_cannot_proceed>"
}
"""


class _AnomalyCheckResult(BaseModel):
    anomaly_detected: bool
    description: str = ""
    forensic_significance: str = ""
    category: str = ""  # one of the 5 categories listed above
    response_action: str = "no_follow_up_warranted"  # LLM decides


_FOLLOW_UP_COMMANDS: dict[str, dict] = {
    "timestamp_contradiction": {
        "skill_domain": "plaso-timeline",
        "tool_cmd_template": ["log2timeline.py", "--parsers", "mft,usnjrnl", "{evidence}"],
    },
    "orphaned_relationship": {
        "skill_domain": "memory-analysis",
        "tool_cmd_template": ["vol3", "-f", "{evidence}", "pstree"],
    },
    "cross_tool_conflict": {
        "skill_domain": "sleuthkit",
        "tool_cmd_template": ["fls", "-r", "-m", "/", "{evidence}"],
    },
    "unexpected_absence": {
        "skill_domain": "yara-hunting",
        "tool_cmd_template": ["yara", "-r", "/opt/yara-rules/", "{evidence}"],
    },
    "integrity_failure": {
        "skill_domain": "sleuthkit",
        "tool_cmd_template": ["img_stat", "{evidence}"],
    },
}


def _get_anomaly_llm():
    """Get LLM for anomaly detection."""
    return get_llm(module="anomaly")


def check_anomalies(state: dict) -> dict:
    """LangGraph node: inspect the most recent tool output for forensic anomalies."""
    invocations = state.get("invocations") or []
    if not invocations:
        return {"_pending_anomalies": False, "_detected_anomaly_data": None}

    invocation = invocations[-1]

    MAX_TOOL_OUTPUT_BYTES = 50_000
    stdout_path = Path(invocation.stdout_path)
    raw = stdout_path.read_text(errors="replace") if stdout_path.exists() else ""
    tool_output = raw[:MAX_TOOL_OUTPUT_BYTES]

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Tool command: {' '.join(str(c) for c in invocation.cmd)}\n\n"
                f"Tool output:\n{tool_output}"
            )
        ),
    ]

    # Anomaly checking is non-critical. If the LLM call fails, log and
    # continue the investigation rather than crashing the graph.
    try:
        response = _get_anomaly_llm().invoke(messages)
        result: _AnomalyCheckResult = parse_llm_json(response.content, _AnomalyCheckResult)
    except Exception:
        from loguru import logger
        logger.warning("Anomaly-check LLM failed; skipping anomaly detection for this step")
        return {"_pending_anomalies": False, "_detected_anomaly_data": None}

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

    # Read response_action from data; default to NO_FOLLOW_UP on invalid values.
    raw_action = data.get("response_action", "no_follow_up_warranted")
    try:
        response_action = AnomalyResponseAction(raw_action)
    except ValueError:
        response_action = AnomalyResponseAction.NO_FOLLOW_UP

    anomaly = Anomaly(
        description=description,
        forensic_significance=data.get("forensic_significance", ""),
        source_invocation_ids=[last_inv_id] if last_inv_id else ["unknown"],
        response_action=response_action,
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
        evidence_refs = state.get("task").evidence_refs if state.get("task") else []
        evidence_path = evidence_refs[0] if evidence_refs else "/evidence"

        # Count existing anomaly follow-up steps to prevent runaway depth
        follow_up_count = sum(
            1 for s in plan.steps
            if s.rationale.startswith("Follow-up investigation of anomaly")
            and s.status == StepStatus.PENDING
        ) if plan is not None else 0

        if follow_up_count < 3:
            # Use context-aware follow-up command based on anomaly category.
            if category in _FOLLOW_UP_COMMANDS:
                cmd_spec = _FOLLOW_UP_COMMANDS[category]
                skill_domain = cmd_spec["skill_domain"]
                tool_cmd = [
                    evidence_path if part == "{evidence}" else part
                    for part in cmd_spec["tool_cmd_template"]
                ]
            else:
                # Fall back to strings for unknown category.
                current_step: PlannedStep | None = None
                if plan is not None and current_step_id:
                    for s in plan.steps:
                        if s.id == current_step_id:
                            current_step = s
                            break
                skill_domain = current_step.skill_domain if current_step else "sleuthkit"
                tool_cmd = ["strings", "-n", "20", evidence_path]

            follow_up = PlannedStep(
                skill_domain=skill_domain,
                tool_cmd=tool_cmd,
                rationale=f"Follow-up investigation of anomaly: {anomaly.description}",
            )
            follow_up_steps = [follow_up]
        else:
            follow_up_steps = []

    return {
        "anomalies": updated_anomalies,
        "_follow_up_steps": follow_up_steps,
        "_pending_anomalies": False,
    }
