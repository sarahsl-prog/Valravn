from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, field_validator


class AnomalyResponseAction(str, Enum):
    ADDED_FOLLOW_UP = "added_follow_up_steps"
    NO_FOLLOW_UP = "no_follow_up_warranted"
    INVESTIGATION_HALT = "investigation_cannot_proceed"


class ToolInvocationRecord(BaseModel):
    id: str = ""
    step_id: str
    attempt_number: int
    cmd: list[str]
    exit_code: int
    stdout_path: Path
    stderr_path: Path
    started_at_utc: datetime
    completed_at_utc: datetime
    duration_seconds: float
    had_output: bool  # True if stdout was non-empty

    def model_post_init(self, __context: object) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())

    @property
    def success(self) -> bool:
        return self.exit_code == 0


class Anomaly(BaseModel):
    id: str = ""
    description: str
    source_invocation_ids: list[str]
    forensic_significance: str
    response_action: AnomalyResponseAction
    follow_up_step_ids: list[str] = []
    detected_at_utc: datetime = None  # type: ignore[assignment]

    def model_post_init(self, __context: object) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())
        if self.detected_at_utc is None:
            self.detected_at_utc = datetime.now(timezone.utc)

    @field_validator("source_invocation_ids")
    @classmethod
    def require_source(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("at least one source invocation ID required")
        return v
