from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from valravn.models.records import Anomaly, ToolInvocationRecord
from valravn.models.report import FindingsReport
from valravn.models.task import InvestigationPlan, InvestigationTask


class AgentState(TypedDict):
    task: InvestigationTask
    plan: InvestigationPlan
    invocations: list[ToolInvocationRecord]
    anomalies: list[Anomaly]
    report: FindingsReport | None
    current_step_id: str | None
    skill_cache: dict[str, str]  # domain -> SKILL.md content
    messages: Annotated[list[BaseMessage], add_messages]
