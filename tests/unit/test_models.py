import os
import uuid
from datetime import timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from valravn.models.task import (
    InvestigationTask,
    InvestigationPlan,
    PlannedStep,
    StepStatus,
)


def test_planned_step_gets_id():
    step = PlannedStep(skill_domain="sleuthkit", tool_cmd=["fls", "-r", "/mnt/img"], rationale="list files")
    assert step.id  # auto-generated UUID


def test_planned_step_default_status():
    step = PlannedStep(skill_domain="sleuthkit", tool_cmd=["fls"], rationale="r")
    assert step.status == StepStatus.PENDING


def test_investigation_plan_timestamps():
    plan = InvestigationPlan(task_id="abc")
    assert plan.created_at_utc.tzinfo == timezone.utc
    assert plan.last_updated_utc.tzinfo == timezone.utc


def test_investigation_task_rejects_empty_prompt(read_only_evidence):
    with pytest.raises(ValidationError, match="prompt must not be empty"):
        InvestigationTask(prompt="  ", evidence_refs=[str(read_only_evidence)])


def test_investigation_task_rejects_writable_evidence(tmp_path):
    writable = tmp_path / "img.raw"
    writable.write_bytes(b"x")
    with pytest.raises(ValidationError, match="writable"):
        InvestigationTask(prompt="find files", evidence_refs=[str(writable)])


def test_investigation_task_rejects_missing_evidence():
    with pytest.raises(ValidationError, match="does not exist"):
        InvestigationTask(prompt="find files", evidence_refs=["/nonexistent/path.raw"])


def test_investigation_task_accepts_read_only_evidence(read_only_evidence):
    task = InvestigationTask(prompt="find files", evidence_refs=[str(read_only_evidence)])
    assert task.id
    assert task.created_at_utc.tzinfo == timezone.utc
