from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from valravn.models.records import Anomaly, ToolInvocationRecord
from valravn.models.report import FindingsReport, SelfCorrectionEvent, ToolFailureRecord
from valravn.models.task import InvestigationPlan, InvestigationTask, PlannedStep


class AgentState(TypedDict):
    # --- Public domain state ---
    task: InvestigationTask
    plan: InvestigationPlan
    invocations: list[ToolInvocationRecord]
    anomalies: list[Anomaly]
    report: FindingsReport | None
    current_step_id: str | None
    skill_cache: dict[str, str]  # domain -> SKILL.md content
    messages: Annotated[list[BaseMessage], add_messages]

    # --- Private / ephemeral inter-node signals ---
    # These are set by graph.run() and updated by nodes between graph steps.
    # They must live in the TypedDict so LangGraph does not strip them.
    _output_dir: str
    _retry_config: dict[str, Any]
    _step_succeeded: bool
    _step_exhausted: bool
    _pending_anomalies: bool
    _last_invocation_id: str | None
    _detected_anomaly_data: dict[str, Any] | None
    _tool_failure: ToolFailureRecord | None
    _tool_failures: list[ToolFailureRecord]
    _self_corrections: list[SelfCorrectionEvent]
    _self_assessments: list[dict]
    _conclusions: list[Any]
    _follow_up_steps: list[PlannedStep]
    _investigation_halted: bool
    _evidence_hashes: dict[str, str]
