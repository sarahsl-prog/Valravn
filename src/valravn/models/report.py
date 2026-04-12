from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Conclusion(BaseModel):
    statement: str
    supporting_invocation_ids: list[str]
    confidence: Literal["high", "medium", "low"]

    @field_validator("supporting_invocation_ids")
    @classmethod
    def must_cite(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("Conclusions must cite at least one tool invocation")
        return v


class ToolFailureRecord(BaseModel):
    step_id: str
    invocation_ids: list[str]
    final_error: str
    diagnostic_context: str


class SelfCorrectionEvent(BaseModel):
    step_id: str
    attempt_number: int
    original_cmd: list[str]
    corrected_cmd: list[str]
    correction_rationale: str


class FindingsReport(BaseModel):
    task_id: str
    prompt: str
    evidence_refs: list[str]
    generated_at_utc: datetime = None  # type: ignore[assignment]
    conclusions: list[Conclusion]
    anomalies: list  # list[Anomaly] — avoid circular import
    tool_failures: list[ToolFailureRecord]
    self_corrections: list[SelfCorrectionEvent]
    investigation_plan_path: Path
    evidence_hashes: dict[str, str] = Field(default_factory=dict)

    def model_post_init(self, __context: object) -> None:
        if self.generated_at_utc is None:
            self.generated_at_utc = datetime.now(timezone.utc)

    @property
    def exit_code(self) -> int:
        return 1 if self.tool_failures else 0
