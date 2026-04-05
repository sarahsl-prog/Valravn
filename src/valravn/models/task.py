from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, field_validator, model_validator


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    EXHAUSTED = "exhausted"
    SKIPPED = "skipped"


class PlannedStep(BaseModel):
    id: str = ""
    skill_domain: str
    tool_cmd: list[str]
    rationale: str
    status: StepStatus = StepStatus.PENDING
    depends_on: list[str] = []
    invocation_ids: list[str] = []

    def model_post_init(self, __context: object) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())


class InvestigationPlan(BaseModel):
    task_id: str
    steps: list[PlannedStep] = []
    created_at_utc: datetime = None  # type: ignore[assignment]
    last_updated_utc: datetime = None  # type: ignore[assignment]

    def model_post_init(self, __context: object) -> None:
        now = datetime.now(timezone.utc)
        if self.created_at_utc is None:
            self.created_at_utc = now
        if self.last_updated_utc is None:
            self.last_updated_utc = now

    def next_pending_step(self) -> PlannedStep | None:
        return next((s for s in self.steps if s.status == StepStatus.PENDING), None)

    def mark_step(self, step_id: str, status: StepStatus) -> None:
        for step in self.steps:
            if step.id == step_id:
                step.status = status
                self.last_updated_utc = datetime.now(timezone.utc)
                return
        raise KeyError(f"Step {step_id} not found")

    def add_steps(self, steps: list[PlannedStep]) -> None:
        self.steps.extend(steps)
        self.last_updated_utc = datetime.now(timezone.utc)


class InvestigationTask(BaseModel):
    id: str = ""
    prompt: str
    evidence_refs: list[str]
    created_at_utc: datetime = None  # type: ignore[assignment]

    def model_post_init(self, __context: object) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())
        if self.created_at_utc is None:
            self.created_at_utc = datetime.now(timezone.utc)

    @field_validator("prompt")
    @classmethod
    def prompt_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt must not be empty")
        return v

    @field_validator("evidence_refs")
    @classmethod
    def refs_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("at least one evidence reference required")
        return v

    @model_validator(mode="after")
    def evidence_integrity(self) -> "InvestigationTask":
        for ref in self.evidence_refs:
            p = Path(ref)
            if not p.exists():
                raise ValueError(f"Evidence path does not exist: {ref}")
            if os.access(p, os.W_OK):
                raise ValueError(
                    f"Evidence path is writable: {ref}. Mount evidence read-only."
                )
        return self
