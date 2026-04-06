# Autonomous DFIR Agents Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI tool that autonomously sequences SIFT forensic tools against evidence, detects anomalies, self-corrects on failure, and writes a fully-cited findings report — with zero operator prompts mid-investigation.

**Architecture:** A LangGraph `StateGraph` drives the investigation loop: Claude plans the step sequence, each node executes one concern (load skill, run tool, check anomalies, update plan), and `SqliteSaver` checkpoints `AgentState` to SQLite after every node so a crash mid-investigation is recoverable. All tool output is persisted to `./analysis/` before Claude processes it; a final node renders the `FindingsReport` to `./reports/`.

**Tech Stack:** Python 3.12, LangGraph 0.2+, langgraph-checkpoint-sqlite, langchain-anthropic (Claude claude-opus-4-6), Pydantic v2, MLflow (local), pytest, ruff

---

## File Map

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Dependencies, entry point, ruff config |
| `config.yaml` | Default retry/mlflow config template |
| `src/valravn/config.py` | `RetryConfig`, `OutputConfig`, `AppConfig`, `load_config()` |
| `src/valravn/models/task.py` | `InvestigationTask`, `InvestigationPlan`, `PlannedStep`, `StepStatus` |
| `src/valravn/models/records.py` | `ToolInvocationRecord`, `Anomaly`, `AnomalyResponseAction` |
| `src/valravn/models/report.py` | `FindingsReport`, `Conclusion`, `ToolFailureRecord`, `SelfCorrectionEvent` |
| `src/valravn/state.py` | `AgentState` TypedDict |
| `src/valravn/nodes/skill_loader.py` | `load_skill()` node — reads SKILL.md files |
| `src/valravn/nodes/tool_runner.py` | `run_forensic_tool()` node — subprocess + retry |
| `src/valravn/nodes/plan.py` | `plan_investigation()`, `update_plan()` nodes |
| `src/valravn/nodes/anomaly.py` | `check_anomalies()`, `record_anomaly()` nodes |
| `src/valravn/nodes/report.py` | `write_findings_report()` node |
| `src/valravn/graph.py` | `StateGraph` definition, `SqliteSaver`, `FileTracer`, `run()` |
| `src/valravn/cli.py` | Argument parsing, evidence validation, calls `graph.run()` |
| `src/valravn/evaluation/evaluators.py` | MLflow custom metrics for SC-002–SC-006 |
| `tests/unit/test_config.py` | Config loading, env override |
| `tests/unit/test_models.py` | Pydantic validators |
| `tests/unit/test_skill_loader.py` | Skill path resolution |
| `tests/unit/test_tool_runner.py` | Subprocess capture, retry logic |
| `tests/unit/test_report.py` | UTC enforcement, citation validation |
| `tests/integration/test_graph.py` | End-to-end US1 with mocked LLM |
| `tests/fixtures/evidence/stub.raw` | 1-byte read-only file (evidence fixture) |
| `tests/conftest.py` | Shared fixtures: `read_only_evidence`, `output_dir` |

---

## Task 1: Project Structure and Dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `config.yaml`
- Create: directory stubs with `__init__.py`

- [ ] **Step 1: Create directory tree**

```bash
mkdir -p src/valravn/nodes src/valravn/models src/valravn/evaluation
mkdir -p tests/unit tests/integration tests/fixtures/evidence tests/evaluation/datasets
touch src/valravn/__init__.py
touch src/valravn/nodes/__init__.py
touch src/valravn/models/__init__.py
touch src/valravn/evaluation/__init__.py
touch tests/__init__.py tests/unit/__init__.py tests/integration/__init__.py
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "valravn"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "langgraph>=0.2",
    "langgraph-checkpoint-sqlite>=1.0",
    "langchain-anthropic>=0.3",
    "mlflow>=2.13",
    "pydantic>=2.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-mock>=3",
    "ruff>=0.4",
]

[project.scripts]
valravn = "valravn.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I"]

[tool.pytest.ini_options]
markers = ["integration: requires SIFT tools on PATH"]
```

- [ ] **Step 3: Write `config.yaml` template**

```yaml
retry:
  max_attempts: 3
  retry_delay_seconds: 0.0

mlflow:
  tracking_uri: http://127.0.0.1:5000
  experiment_name: valravn-evaluation
```

- [ ] **Step 4: Create read-only evidence stub**

```bash
echo -n "x" > tests/fixtures/evidence/stub.raw
chmod 444 tests/fixtures/evidence/stub.raw
```

- [ ] **Step 5: Install and verify**

```bash
source .venv/bin/activate
pip install -e ".[dev]"
python -c "import langgraph, langchain_anthropic, mlflow, pydantic; print('OK')"
```
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml config.yaml src/ tests/
git commit -m "chore: project structure and dependencies"
```

---

## Task 2: Config

**Files:**
- Create: `src/valravn/config.py`
- Create: `tests/unit/test_config.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_config.py
import os
from pathlib import Path
import pytest
from valravn.config import RetryConfig, OutputConfig, AppConfig, load_config


def test_retry_config_defaults():
    cfg = RetryConfig()
    assert cfg.max_attempts == 3
    assert cfg.retry_delay_seconds == 0.0


def test_load_config_from_yaml(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("retry:\n  max_attempts: 5\n")
    cfg = load_config(cfg_file)
    assert cfg.retry.max_attempts == 5


def test_load_config_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("VALRAVN_MAX_RETRIES", "7")
    cfg = load_config(None)
    assert cfg.retry.max_attempts == 7


def test_output_config_dirs(tmp_path):
    cfg = OutputConfig(output_dir=tmp_path)
    assert cfg.analysis_dir == tmp_path / "analysis"
    assert cfg.reports_dir == tmp_path / "reports"
    assert cfg.checkpoints_db == tmp_path / "analysis" / "checkpoints.db"
    assert cfg.traces_dir == tmp_path / "analysis" / "traces"
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/unit/test_config.py -v
```
Expected: `ModuleNotFoundError: No module named 'valravn.config'`

- [ ] **Step 3: Write `src/valravn/config.py`**

```python
from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel


class RetryConfig(BaseModel):
    max_attempts: int = 3
    retry_delay_seconds: float = 0.0


class OutputConfig(BaseModel):
    output_dir: Path

    @property
    def analysis_dir(self) -> Path:
        return self.output_dir / "analysis"

    @property
    def reports_dir(self) -> Path:
        return self.output_dir / "reports"

    @property
    def exports_dir(self) -> Path:
        return self.output_dir / "exports"

    @property
    def traces_dir(self) -> Path:
        return self.output_dir / "analysis" / "traces"

    @property
    def checkpoints_db(self) -> Path:
        return self.output_dir / "analysis" / "checkpoints.db"

    def ensure_dirs(self) -> None:
        for d in (self.analysis_dir, self.reports_dir, self.exports_dir, self.traces_dir):
            d.mkdir(parents=True, exist_ok=True)


class AppConfig(BaseModel):
    retry: RetryConfig = RetryConfig()
    mlflow: dict = {
        "tracking_uri": "http://127.0.0.1:5000",
        "experiment_name": "valravn-evaluation",
    }


def load_config(path: Path | None) -> AppConfig:
    data: dict = {}
    if path and path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}

    cfg = AppConfig(**data)

    env_max = os.environ.get("VALRAVN_MAX_RETRIES")
    if env_max:
        cfg.retry.max_attempts = int(env_max)

    return cfg
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/unit/test_config.py -v
```
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/valravn/config.py tests/unit/test_config.py
git commit -m "feat: config loading with YAML and env override"
```

---

## Task 3: Data Models — task.py

**Files:**
- Create: `src/valravn/models/task.py`
- Create: `tests/unit/test_models.py` (partial — extended in Tasks 4 and 5)
- Create: `tests/conftest.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_models.py
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
    # file is writable by default
    with pytest.raises(ValidationError, match="writable"):
        InvestigationTask(prompt="find files", evidence_refs=[str(writable)])


def test_investigation_task_rejects_missing_evidence():
    with pytest.raises(ValidationError, match="does not exist"):
        InvestigationTask(prompt="find files", evidence_refs=["/nonexistent/path.raw"])


def test_investigation_task_accepts_read_only_evidence(read_only_evidence):
    task = InvestigationTask(prompt="find files", evidence_refs=[str(read_only_evidence)])
    assert task.id
    assert task.created_at_utc.tzinfo == timezone.utc
```

```python
# tests/conftest.py
import os
from pathlib import Path
import pytest


@pytest.fixture
def read_only_evidence(tmp_path) -> Path:
    p = tmp_path / "evidence.raw"
    p.write_bytes(b"x")
    os.chmod(p, 0o444)
    return p


@pytest.fixture
def output_dir(tmp_path) -> Path:
    d = tmp_path / "output"
    d.mkdir()
    return d
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/unit/test_models.py -v
```
Expected: `ModuleNotFoundError: No module named 'valravn.models.task'`

- [ ] **Step 3: Write `src/valravn/models/task.py`**

```python
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
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/unit/test_models.py -v
```
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add src/valravn/models/task.py tests/unit/test_models.py tests/conftest.py
git commit -m "feat: task data models with evidence integrity validation"
```

---

## Task 4: Data Models — records.py

**Files:**
- Modify: `tests/unit/test_models.py` (append)
- Create: `src/valravn/models/records.py`

- [ ] **Step 1: Append failing tests to `tests/unit/test_models.py`**

```python
# Append to tests/unit/test_models.py
from datetime import datetime, timezone
from valravn.models.records import ToolInvocationRecord, Anomaly, AnomalyResponseAction


def test_tool_invocation_record_success_flag(tmp_path):
    rec = ToolInvocationRecord(
        step_id="s1",
        attempt_number=1,
        cmd=["fls", "-r", "/mnt/img"],
        exit_code=0,
        stdout_path=tmp_path / "out.stdout",
        stderr_path=tmp_path / "out.stderr",
        started_at_utc=datetime.now(timezone.utc),
        completed_at_utc=datetime.now(timezone.utc),
        duration_seconds=1.2,
        had_output=True,
    )
    assert rec.success is True
    assert rec.id  # auto UUID


def test_tool_invocation_record_failure_flag(tmp_path):
    rec = ToolInvocationRecord(
        step_id="s1",
        attempt_number=1,
        cmd=["fls"],
        exit_code=1,
        stdout_path=tmp_path / "out.stdout",
        stderr_path=tmp_path / "out.stderr",
        started_at_utc=datetime.now(timezone.utc),
        completed_at_utc=datetime.now(timezone.utc),
        duration_seconds=0.1,
        had_output=False,
    )
    assert rec.success is False


def test_anomaly_requires_source_invocations():
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="at least one source"):
        Anomaly(
            description="conflict",
            source_invocation_ids=[],
            forensic_significance="significant",
            response_action=AnomalyResponseAction.ADDED_FOLLOW_UP,
        )


def test_anomaly_valid():
    a = Anomaly(
        description="timestamp predates OS install",
        source_invocation_ids=["inv-1"],
        forensic_significance="indicates backdating",
        response_action=AnomalyResponseAction.ADDED_FOLLOW_UP,
    )
    assert a.id
    assert a.detected_at_utc.tzinfo == timezone.utc
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/unit/test_models.py::test_tool_invocation_record_success_flag -v
```
Expected: `ModuleNotFoundError: No module named 'valravn.models.records'`

- [ ] **Step 3: Write `src/valravn/models/records.py`**

```python
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
        return self.exit_code == 0 and self.had_output


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
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/unit/test_models.py -v
```
Expected: `11 passed`

- [ ] **Step 5: Commit**

```bash
git add src/valravn/models/records.py tests/unit/test_models.py
git commit -m "feat: ToolInvocationRecord and Anomaly models"
```

---

## Task 5: Data Models — report.py

**Files:**
- Modify: `tests/unit/test_models.py` (append)
- Create: `src/valravn/models/report.py`

- [ ] **Step 1: Append failing tests**

```python
# Append to tests/unit/test_models.py
from valravn.models.report import (
    FindingsReport, Conclusion, ToolFailureRecord, SelfCorrectionEvent,
)


def test_conclusion_requires_citations():
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="must cite"):
        Conclusion(
            statement="malware present",
            supporting_invocation_ids=[],
            confidence="high",
        )


def test_findings_report_utc_timestamp():
    report = FindingsReport(
        task_id="t1",
        prompt="find files",
        evidence_refs=["/mnt/img"],
        conclusions=[],
        anomalies=[],
        tool_failures=[],
        self_corrections=[],
        investigation_plan_path=Path("/tmp/plan.json"),
    )
    assert report.generated_at_utc.tzinfo == timezone.utc


def test_findings_report_exit_code_clean():
    report = FindingsReport(
        task_id="t1",
        prompt="find files",
        evidence_refs=["/mnt/img"],
        conclusions=[],
        anomalies=[],
        tool_failures=[],
        self_corrections=[],
        investigation_plan_path=Path("/tmp/plan.json"),
    )
    assert report.exit_code == 0


def test_findings_report_exit_code_with_failures():
    failure = ToolFailureRecord(
        step_id="s1",
        invocation_ids=["inv-1"],
        final_error="No such file",
        diagnostic_context="tried 3 times",
    )
    report = FindingsReport(
        task_id="t1",
        prompt="find files",
        evidence_refs=["/mnt/img"],
        conclusions=[],
        anomalies=[],
        tool_failures=[failure],
        self_corrections=[],
        investigation_plan_path=Path("/tmp/plan.json"),
    )
    assert report.exit_code == 1
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/unit/test_models.py::test_conclusion_requires_citations -v
```
Expected: `ModuleNotFoundError: No module named 'valravn.models.report'`

- [ ] **Step 3: Write `src/valravn/models/report.py`**

```python
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator


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

    def model_post_init(self, __context: object) -> None:
        if self.generated_at_utc is None:
            self.generated_at_utc = datetime.now(timezone.utc)

    @property
    def exit_code(self) -> int:
        return 1 if self.tool_failures else 0
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/unit/test_models.py -v
```
Expected: `15 passed`

- [ ] **Step 5: Commit**

```bash
git add src/valravn/models/report.py tests/unit/test_models.py
git commit -m "feat: FindingsReport and related output models"
```

---

## Task 6: AgentState

**Files:**
- Create: `src/valravn/state.py`

No dedicated test — `AgentState` is a TypedDict (no runtime validation). It will be exercised through graph tests in Task 14.

- [ ] **Step 1: Write `src/valravn/state.py`**

```python
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
```

- [ ] **Step 2: Verify import**

```bash
python -c "from valravn.state import AgentState; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/valravn/state.py
git commit -m "feat: AgentState TypedDict"
```

---

## Task 7: Skill Loader Node

**Files:**
- Create: `src/valravn/nodes/skill_loader.py`
- Create: `tests/unit/test_skill_loader.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_skill_loader.py
from pathlib import Path
import pytest
from unittest.mock import patch
from valravn.nodes.skill_loader import load_skill, SKILL_PATHS, SkillNotFoundError
from valravn.models.task import InvestigationPlan, InvestigationTask, PlannedStep
from tests.conftest import *  # noqa: F401, F403


def _make_state(skill_domain: str, skill_content: str | None, read_only_evidence) -> dict:
    from datetime import timezone
    task = InvestigationTask(
        prompt="test",
        evidence_refs=[str(read_only_evidence)],
    )
    step = PlannedStep(skill_domain=skill_domain, tool_cmd=["fls"], rationale="r")
    plan = InvestigationPlan(task_id=task.id, steps=[step])
    return {
        "task": task,
        "plan": plan,
        "invocations": [],
        "anomalies": [],
        "report": None,
        "current_step_id": step.id,
        "skill_cache": {} if skill_content is None else {skill_domain: skill_content},
        "messages": [],
    }


def test_load_skill_reads_file(tmp_path, read_only_evidence):
    skill_file = tmp_path / "sleuthkit" / "SKILL.md"
    skill_file.parent.mkdir()
    skill_file.write_text("# Sleuth Kit Skill\nUse fls to list files.")

    state = _make_state("sleuthkit", None, read_only_evidence)

    with patch.dict("valravn.nodes.skill_loader.SKILL_PATHS",
                    {"sleuthkit": skill_file}):
        result = load_skill(state)

    assert result["skill_cache"]["sleuthkit"] == "# Sleuth Kit Skill\nUse fls to list files."


def test_load_skill_uses_cache(read_only_evidence):
    state = _make_state("sleuthkit", "cached content", read_only_evidence)
    # File does not exist — should use cache
    result = load_skill(state)
    assert result["skill_cache"]["sleuthkit"] == "cached content"


def test_load_skill_unknown_domain(read_only_evidence):
    state = _make_state("unknown-domain", None, read_only_evidence)
    with pytest.raises(SkillNotFoundError):
        load_skill(state)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/unit/test_skill_loader.py -v
```
Expected: `ModuleNotFoundError: No module named 'valravn.nodes.skill_loader'`

- [ ] **Step 3: Write `src/valravn/nodes/skill_loader.py`**

```python
from __future__ import annotations

from pathlib import Path

# Maps domain keys to absolute SKILL.md paths.
# Adjust if skills are installed elsewhere.
_SKILLS_BASE = Path.home() / ".claude" / "skills"

SKILL_PATHS: dict[str, Path] = {
    "memory-analysis": _SKILLS_BASE / "memory-analysis" / "SKILL.md",
    "sleuthkit": _SKILLS_BASE / "sleuthkit" / "SKILL.md",
    "windows-artifacts": _SKILLS_BASE / "windows-artifacts" / "SKILL.md",
    "plaso-timeline": _SKILLS_BASE / "plaso-timeline" / "SKILL.md",
    "yara-hunting": _SKILLS_BASE / "yara-hunting" / "SKILL.md",
}


class SkillNotFoundError(Exception):
    pass


def load_skill(state: dict) -> dict:
    """LangGraph node: load SKILL.md for the current step's domain into skill_cache."""
    step_id = state["current_step_id"]
    step = next(s for s in state["plan"].steps if s.id == step_id)
    domain = step.skill_domain

    cache: dict[str, str] = dict(state.get("skill_cache") or {})

    if domain in cache:
        return {"skill_cache": cache}

    if domain not in SKILL_PATHS:
        raise SkillNotFoundError(
            f"No skill file registered for domain '{domain}'. "
            f"Add it to SKILL_PATHS in nodes/skill_loader.py."
        )

    skill_path = SKILL_PATHS[domain]
    if not skill_path.exists():
        raise SkillNotFoundError(f"Skill file not found: {skill_path}")

    cache[domain] = skill_path.read_text()
    return {"skill_cache": cache}
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/unit/test_skill_loader.py -v
```
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/valravn/nodes/skill_loader.py tests/unit/test_skill_loader.py
git commit -m "feat: skill loader node with domain cache"
```

---

## Task 8: CLI Entry Point

**Files:**
- Create: `src/valravn/cli.py`

- [ ] **Step 1: Write `src/valravn/cli.py`**

The CLI validates evidence, builds `InvestigationTask`, then calls `graph.run()`. Graph is imported lazily so tests can import `cli` without needing the full graph compiled.

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from valravn.config import OutputConfig, load_config
from valravn.models.task import InvestigationTask


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="valravn",
        description="Autonomous DFIR investigation agent",
    )
    sub = p.add_subparsers(dest="command", required=True)

    inv = sub.add_parser("investigate", help="Run an investigation")
    inv.add_argument("--prompt", required=True, help="Natural-language investigation prompt")
    inv.add_argument(
        "--evidence",
        required=True,
        action="append",
        dest="evidence_refs",
        metavar="PATH",
        help="Evidence path (repeat for multiple)",
    )
    inv.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    inv.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Working directory for analysis/, reports/",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "investigate":
        app_cfg = load_config(args.config)
        out_cfg = OutputConfig(output_dir=args.output_dir.resolve())
        out_cfg.ensure_dirs()

        try:
            task = InvestigationTask(
                prompt=args.prompt,
                evidence_refs=[str(Path(r).resolve()) for r in args.evidence_refs],
            )
        except ValidationError as e:
            # Extract first error message for clean CLI output
            msg = e.errors()[0]["msg"]
            print(f"Error: {msg}", file=sys.stderr)
            sys.exit(2)

        # Import graph here so CLI is importable without compiling graph
        from valravn.graph import run as graph_run

        exit_code = graph_run(task=task, app_cfg=app_cfg, out_cfg=out_cfg)
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify help works**

```bash
valravn --help
valravn investigate --help
```
Expected: Usage text with `--prompt`, `--evidence`, `--config`, `--output-dir`

- [ ] **Step 3: Verify evidence validation**

```bash
valravn investigate --prompt "test" --evidence /tmp
```
Expected: `Error: Evidence path is writable...` and exit code 2

```bash
echo $?
```
Expected: `2`

- [ ] **Step 4: Commit**

```bash
git add src/valravn/cli.py
git commit -m "feat: CLI entry point with evidence integrity check"
```

---

## Task 9: Graph Skeleton with SqliteSaver and FileTracer

**Files:**
- Create: `src/valravn/graph.py`

- [ ] **Step 1: Write `src/valravn/graph.py`**

This compiles the graph with all node slots stubbed. Nodes will be replaced task by task.

```python
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.callbacks.base import BaseCallbackHandler
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from valravn.config import AppConfig, OutputConfig
from valravn.models.task import InvestigationPlan, InvestigationTask
from valravn.state import AgentState

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Minimal FileTracer — writes JSONL to ./analysis/traces/<run_id>.jsonl
# ---------------------------------------------------------------------------

class FileTracer(BaseCallbackHandler):
    def __init__(self, trace_path: Path) -> None:
        super().__init__()
        self._path = trace_path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, event: str, data: dict) -> None:
        line = json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **data,
        })
        with open(self._path, "a") as f:
            f.write(line + "\n")

    def on_llm_start(self, serialized: dict, prompts: list[str], **kw: object) -> None:
        self._write("llm_start", {"model": serialized.get("name", ""), "prompts": prompts})

    def on_llm_end(self, response: object, **kw: object) -> None:
        self._write("llm_end", {})

    def on_tool_start(self, serialized: dict, input_str: str, **kw: object) -> None:
        self._write("tool_start", {"tool": serialized.get("name", ""), "input": input_str})

    def on_tool_end(self, output: object, **kw: object) -> None:
        self._write("tool_end", {"output": str(output)[:500]})


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def _build_graph(checkpointer: SqliteSaver) -> object:
    from valravn.nodes.anomaly import check_anomalies, record_anomaly
    from valravn.nodes.plan import plan_investigation, update_plan
    from valravn.nodes.report import write_findings_report
    from valravn.nodes.skill_loader import load_skill
    from valravn.nodes.tool_runner import run_forensic_tool

    def route_after_anomaly_check(state: AgentState) -> str:
        """Route to record_anomaly if new anomalies were detected, else update_plan."""
        # Nodes set a transient flag "_pending_anomalies" if anomalies were found
        if state.get("_pending_anomalies"):
            return "record_anomaly"
        return "update_plan"

    def route_next_step(state: AgentState) -> str:
        """After updating plan, decide: more steps or write report."""
        if state["plan"].next_pending_step() is not None:
            return "load_skill"
        return "write_findings_report"

    builder: StateGraph = StateGraph(AgentState)

    builder.add_node("plan_investigation", plan_investigation)
    builder.add_node("load_skill", load_skill)
    builder.add_node("run_forensic_tool", run_forensic_tool)
    builder.add_node("check_anomalies", check_anomalies)
    builder.add_node("record_anomaly", record_anomaly)
    builder.add_node("update_plan", update_plan)
    builder.add_node("write_findings_report", write_findings_report)

    builder.add_edge(START, "plan_investigation")
    builder.add_edge("plan_investigation", "load_skill")
    builder.add_edge("load_skill", "run_forensic_tool")
    builder.add_edge("run_forensic_tool", "check_anomalies")
    builder.add_conditional_edges("check_anomalies", route_after_anomaly_check)
    builder.add_edge("record_anomaly", "update_plan")
    builder.add_conditional_edges("update_plan", route_next_step)
    builder.add_edge("write_findings_report", END)

    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(task: InvestigationTask, app_cfg: AppConfig, out_cfg: OutputConfig) -> int:
    """Compile and invoke the investigation graph. Returns exit code (0 or 1)."""
    db_path = out_cfg.checkpoints_db
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    graph = _build_graph(checkpointer)

    run_id = task.id
    trace_path = out_cfg.traces_dir / f"{run_id}.jsonl"
    tracer = FileTracer(trace_path)

    initial_state: AgentState = {
        "task": task,
        "plan": InvestigationPlan(task_id=task.id),
        "invocations": [],
        "anomalies": [],
        "report": None,
        "current_step_id": None,
        "skill_cache": {},
        "messages": [],
    }

    config = {
        "configurable": {"thread_id": run_id},
        "callbacks": [tracer],
    }

    final_state = graph.invoke(initial_state, config=config)

    if final_state.get("report"):
        return final_state["report"].exit_code
    return 1
```

- [ ] **Step 2: Stub all missing node modules so graph compiles**

```python
# src/valravn/nodes/plan.py  (stub)
def plan_investigation(state): return {}
def update_plan(state): return {}
```

```python
# src/valravn/nodes/tool_runner.py  (stub)
def run_forensic_tool(state): return {}
```

```python
# src/valravn/nodes/anomaly.py  (stub)
def check_anomalies(state): return {}
def record_anomaly(state): return {}
```

```python
# src/valravn/nodes/report.py  (stub)
def write_findings_report(state): return {}
```

- [ ] **Step 3: Verify graph compiles**

```bash
python -c "
from valravn.config import AppConfig, OutputConfig
from pathlib import Path
from valravn.graph import _build_graph
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
conn = sqlite3.connect(':memory:')
g = _build_graph(SqliteSaver(conn))
print('Graph compiled OK, nodes:', list(g.nodes.keys()))
"
```
Expected: `Graph compiled OK, nodes: [...]` listing all 7 nodes

- [ ] **Step 4: Commit**

```bash
git add src/valravn/graph.py src/valravn/nodes/plan.py src/valravn/nodes/tool_runner.py src/valravn/nodes/anomaly.py src/valravn/nodes/report.py
git commit -m "feat: graph skeleton with SqliteSaver checkpointer and FileTracer"
```

---

## Task 10: plan_investigation Node (US1)

**Files:**
- Modify: `src/valravn/nodes/plan.py`
- Create: `tests/unit/test_plan_node.py`

The node calls Claude to derive an initial step list from the prompt and evidence metadata, then writes `investigation_plan.json`.

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_plan_node.py
import json
from pathlib import Path
from datetime import timezone
from unittest.mock import MagicMock, patch

import pytest
from valravn.models.task import InvestigationPlan, InvestigationTask, PlannedStep, StepStatus
from valravn.nodes.plan import plan_investigation, update_plan
from tests.conftest import *  # noqa


def _base_state(read_only_evidence, output_dir):
    task = InvestigationTask(prompt="identify network connections", evidence_refs=[str(read_only_evidence)])
    plan = InvestigationPlan(task_id=task.id)
    return {
        "task": task,
        "plan": plan,
        "invocations": [],
        "anomalies": [],
        "report": None,
        "current_step_id": None,
        "skill_cache": {},
        "messages": [],
        "_output_dir": str(output_dir),
    }


def test_plan_investigation_populates_steps(read_only_evidence, output_dir):
    mock_response = MagicMock()
    mock_response.steps = [
        MagicMock(
            skill_domain="memory-analysis",
            tool_cmd=["python3", "/opt/volatility3-2.20.0/vol.py", "-f", "/mnt/mem.lime", "windows.netstat"],
            rationale="list network connections",
        )
    ]

    state = _base_state(read_only_evidence, output_dir)

    with patch("valravn.nodes.plan._get_llm") as mock_llm_fn:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_llm_fn.return_value = mock_llm

        result = plan_investigation(state)

    assert len(result["plan"].steps) == 1
    assert result["plan"].steps[0].skill_domain == "memory-analysis"
    assert result["current_step_id"] == result["plan"].steps[0].id


def test_plan_investigation_writes_json(read_only_evidence, output_dir):
    mock_response = MagicMock()
    mock_response.steps = [
        MagicMock(skill_domain="sleuthkit", tool_cmd=["fls", "-r"], rationale="list files")
    ]
    state = _base_state(read_only_evidence, output_dir)

    with patch("valravn.nodes.plan._get_llm") as mock_llm_fn:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_llm_fn.return_value = mock_llm

        plan_investigation(state)

    plan_file = output_dir / "analysis" / "investigation_plan.json"
    assert plan_file.exists()


def test_update_plan_marks_step_completed(read_only_evidence, output_dir):
    task = InvestigationTask(prompt="test", evidence_refs=[str(read_only_evidence)])
    step = PlannedStep(skill_domain="sleuthkit", tool_cmd=["fls"], rationale="r")
    plan = InvestigationPlan(task_id=task.id, steps=[step])
    state = {
        "task": task, "plan": plan, "invocations": [], "anomalies": [],
        "report": None, "current_step_id": step.id, "skill_cache": {},
        "messages": [], "_step_succeeded": True, "_output_dir": str(output_dir),
    }
    result = update_plan(state)
    assert result["plan"].steps[0].status == StepStatus.COMPLETED
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/unit/test_plan_node.py -v
```
Expected: failures (stub implementation returns `{}`)

- [ ] **Step 3: Write `src/valravn/nodes/plan.py`**

```python
from __future__ import annotations

import json
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from valravn.models.task import (
    InvestigationPlan,
    PlannedStep,
    StepStatus,
)

# ---------------------------------------------------------------------------
# Structured output schema for LLM response
# ---------------------------------------------------------------------------

class _StepSpec(BaseModel):
    skill_domain: str
    tool_cmd: list[str]
    rationale: str


class _PlanSpec(BaseModel):
    steps: list[_StepSpec]


_SYSTEM_PROMPT = """\
You are an expert DFIR analyst on a SANS SIFT Ubuntu workstation.
Given an investigation prompt and evidence paths, return an ordered list of forensic
tool invocations to execute.

Rules:
- Use ONLY tools available on SIFT (Volatility 3, fls, icat, log2timeline.py, yara, etc.)
- Each step must target ONE specific forensic question
- skill_domain must be one of: memory-analysis, sleuthkit, windows-artifacts, plaso-timeline, yara-hunting
- tool_cmd must be the exact subprocess argv (list of strings)
- Do NOT include evidence paths as output destinations
"""


def _get_llm() -> object:
    llm = ChatAnthropic(model="claude-opus-4-6", temperature=0)
    return llm.with_structured_output(_PlanSpec)


def plan_investigation(state: dict) -> dict:
    """LangGraph node: derive initial investigation plan from prompt."""
    task = state["task"]
    output_dir = Path(state.get("_output_dir", "."))

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Investigation prompt: {task.prompt}\n"
                f"Evidence paths: {', '.join(task.evidence_refs)}"
            )
        ),
    ]

    plan_spec: _PlanSpec = _get_llm().invoke(messages)

    steps = [
        PlannedStep(
            skill_domain=s.skill_domain,
            tool_cmd=s.tool_cmd,
            rationale=s.rationale,
        )
        for s in plan_spec.steps
    ]

    plan = InvestigationPlan(task_id=task.id, steps=steps)

    _persist_plan(plan, output_dir)

    first_step_id = steps[0].id if steps else None

    return {
        "plan": plan,
        "current_step_id": first_step_id,
    }


def update_plan(state: dict) -> dict:
    """LangGraph node: mark current step complete/failed; advance to next pending step."""
    plan: InvestigationPlan = state["plan"]
    step_id: str = state["current_step_id"]
    succeeded: bool = state.get("_step_succeeded", False)
    exhausted: bool = state.get("_step_exhausted", False)
    output_dir = Path(state.get("_output_dir", "."))

    if exhausted:
        plan.mark_step(step_id, StepStatus.EXHAUSTED)
    elif succeeded:
        plan.mark_step(step_id, StepStatus.COMPLETED)
    else:
        plan.mark_step(step_id, StepStatus.FAILED)

    next_step = plan.next_pending_step()
    next_id = next_step.id if next_step else None

    _persist_plan(plan, output_dir)

    return {
        "plan": plan,
        "current_step_id": next_id,
        "_step_succeeded": False,
        "_step_exhausted": False,
        "_pending_anomalies": False,
    }


def _persist_plan(plan: InvestigationPlan, output_dir: Path) -> None:
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    plan_path = analysis_dir / "investigation_plan.json"
    plan_path.write_text(
        json.dumps(plan.model_dump(mode="json"), indent=2, default=str)
    )
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/unit/test_plan_node.py -v
```
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/valravn/nodes/plan.py tests/unit/test_plan_node.py
git commit -m "feat: plan_investigation and update_plan nodes"
```

---

## Task 11: run_forensic_tool Node — Basic Invocation (US1)

**Files:**
- Modify: `src/valravn/nodes/tool_runner.py`
- Create: `tests/unit/test_tool_runner.py`

Basic subprocess capture — no retry yet (added in Task 17).

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_tool_runner.py
import os
from datetime import timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from valravn.models.task import InvestigationPlan, InvestigationTask, PlannedStep
from valravn.nodes.tool_runner import run_forensic_tool
from tests.conftest import *  # noqa


def _state(read_only_evidence, output_dir, tool_cmd):
    task = InvestigationTask(prompt="test", evidence_refs=[str(read_only_evidence)])
    step = PlannedStep(skill_domain="sleuthkit", tool_cmd=tool_cmd, rationale="r")
    plan = InvestigationPlan(task_id=task.id, steps=[step])
    return {
        "task": task, "plan": plan, "invocations": [],
        "anomalies": [], "report": None,
        "current_step_id": step.id, "skill_cache": {},
        "messages": [], "_output_dir": str(output_dir),
        "_retry_config": {"max_attempts": 3, "retry_delay_seconds": 0.0},
    }


def test_run_tool_captures_stdout(read_only_evidence, output_dir):
    state = _state(read_only_evidence, output_dir, ["echo", "hello world"])
    result = run_forensic_tool(state)
    rec = result["invocations"][-1]
    assert Path(rec.stdout_path).read_text() == "hello world\n"
    assert rec.exit_code == 0
    assert rec.success is True


def test_run_tool_captures_stderr(read_only_evidence, output_dir):
    state = _state(read_only_evidence, output_dir, ["bash", "-c", "echo err >&2; exit 1"])
    result = run_forensic_tool(state)
    rec = result["invocations"][-1]
    assert "err" in Path(rec.stderr_path).read_text()
    assert rec.exit_code == 1


def test_run_tool_stdout_not_in_evidence(read_only_evidence, output_dir):
    """Output paths must not be under evidence directories."""
    state = _state(read_only_evidence, output_dir, ["echo", "data"])
    result = run_forensic_tool(state)
    rec = result["invocations"][-1]
    # stdout_path must be under output_dir/analysis, not under evidence
    assert str(output_dir) in str(rec.stdout_path)
    assert str(read_only_evidence.parent) not in str(rec.stdout_path)


def test_run_tool_timestamps_utc(read_only_evidence, output_dir):
    state = _state(read_only_evidence, output_dir, ["echo", "x"])
    result = run_forensic_tool(state)
    rec = result["invocations"][-1]
    assert rec.started_at_utc.tzinfo == timezone.utc
    assert rec.completed_at_utc.tzinfo == timezone.utc
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/unit/test_tool_runner.py -v
```
Expected: failures (stub returns `{}`)

- [ ] **Step 3: Write `src/valravn/nodes/tool_runner.py`**

```python
from __future__ import annotations

import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from valravn.models.records import ToolInvocationRecord


def run_forensic_tool(state: dict) -> dict:
    """LangGraph node: invoke the current step's tool_cmd; capture stdout/stderr."""
    plan = state["plan"]
    step_id: str = state["current_step_id"]
    step = next(s for s in plan.steps if s.id == step_id)
    output_dir = Path(state.get("_output_dir", "."))

    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    inv_id = str(uuid.uuid4())
    stdout_path = analysis_dir / f"{inv_id}.stdout"
    stderr_path = analysis_dir / f"{inv_id}.stderr"

    started = datetime.now(timezone.utc)
    t0 = time.monotonic()

    proc = subprocess.run(
        step.tool_cmd,
        capture_output=True,
        text=True,
    )

    duration = time.monotonic() - t0
    completed = datetime.now(timezone.utc)

    stdout_path.write_text(proc.stdout)
    stderr_path.write_text(proc.stderr)

    rec = ToolInvocationRecord(
        id=inv_id,
        step_id=step_id,
        attempt_number=1,
        cmd=step.tool_cmd,
        exit_code=proc.returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        started_at_utc=started,
        completed_at_utc=completed,
        duration_seconds=duration,
        had_output=bool(proc.stdout.strip()),
    )

    invocations = list(state.get("invocations") or [])
    invocations.append(rec)

    # Add invocation ID to step record
    step.invocation_ids.append(inv_id)

    return {
        "invocations": invocations,
        "plan": plan,
        "_step_succeeded": rec.success,
        "_last_invocation_id": inv_id,
    }
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/unit/test_tool_runner.py -v
```
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/valravn/nodes/tool_runner.py tests/unit/test_tool_runner.py
git commit -m "feat: run_forensic_tool node with subprocess capture"
```

---

## Task 12: write_findings_report Node (US1)

**Files:**
- Modify: `src/valravn/nodes/report.py`
- Create: `tests/unit/test_report.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_report.py
import json
from datetime import timezone
from pathlib import Path

import pytest
from valravn.models.records import AnomalyResponseAction, Anomaly, ToolInvocationRecord
from valravn.models.report import Conclusion, FindingsReport
from valravn.models.task import InvestigationPlan, InvestigationTask, PlannedStep
from valravn.nodes.report import write_findings_report
from tests.conftest import *  # noqa
from datetime import datetime


def _state_with_invocation(read_only_evidence, output_dir):
    task = InvestigationTask(prompt="find connections", evidence_refs=[str(read_only_evidence)])
    step = PlannedStep(skill_domain="memory-analysis", tool_cmd=["vol.py", "netstat"], rationale="r")
    plan = InvestigationPlan(task_id=task.id, steps=[step])
    inv = ToolInvocationRecord(
        id="inv-1",
        step_id=step.id,
        attempt_number=1,
        cmd=["vol.py", "netstat"],
        exit_code=0,
        stdout_path=output_dir / "inv-1.stdout",
        stderr_path=output_dir / "inv-1.stderr",
        started_at_utc=datetime.now(timezone.utc),
        completed_at_utc=datetime.now(timezone.utc),
        duration_seconds=1.0,
        had_output=True,
    )
    return {
        "task": task, "plan": plan, "invocations": [inv],
        "anomalies": [], "report": None,
        "current_step_id": None, "skill_cache": {},
        "messages": [], "_output_dir": str(output_dir),
        "_conclusions": [
            {"statement": "TCP connection to 10.0.0.1:445",
             "supporting_invocation_ids": ["inv-1"],
             "confidence": "high"}
        ],
    }


def test_report_written_to_reports_dir(read_only_evidence, output_dir):
    state = _state_with_invocation(read_only_evidence, output_dir)
    result = write_findings_report(state)
    reports = list((output_dir / "reports").glob("*.md"))
    assert len(reports) == 1


def test_report_json_also_written(read_only_evidence, output_dir):
    state = _state_with_invocation(read_only_evidence, output_dir)
    write_findings_report(state)
    jsons = list((output_dir / "reports").glob("*.json"))
    assert len(jsons) == 1


def test_report_timestamp_utc(read_only_evidence, output_dir):
    state = _state_with_invocation(read_only_evidence, output_dir)
    result = write_findings_report(state)
    assert result["report"].generated_at_utc.tzinfo == timezone.utc


def test_report_conclusion_cites_invocation(read_only_evidence, output_dir):
    state = _state_with_invocation(read_only_evidence, output_dir)
    result = write_findings_report(state)
    assert result["report"].conclusions[0].supporting_invocation_ids == ["inv-1"]
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/unit/test_report.py -v
```
Expected: failures (stub returns `{}`)

- [ ] **Step 3: Write `src/valravn/nodes/report.py`**

```python
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from valravn.models.report import Conclusion, FindingsReport, SelfCorrectionEvent, ToolFailureRecord


_REPORT_TEMPLATE = """\
# DFIR Findings Report

**Task ID**: {task_id}
**Generated**: {generated_at} UTC
**Evidence**: {evidence}

## Investigation Prompt

{prompt}

## Conclusions

{conclusions}

## Anomalies

{anomalies}

## Tool Failures

{failures}

## Self-Corrections

{corrections}
"""


def write_findings_report(state: dict) -> dict:
    """LangGraph node: render FindingsReport to ./reports/ as Markdown and JSON."""
    task = state["task"]
    output_dir = Path(state.get("_output_dir", "."))
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    conclusions = [
        Conclusion(**c) if isinstance(c, dict) else c
        for c in (state.get("_conclusions") or [])
    ]

    tool_failures = [
        ToolFailureRecord(**f) if isinstance(f, dict) else f
        for f in (state.get("_tool_failures") or [])
    ]

    self_corrections = [
        SelfCorrectionEvent(**e) if isinstance(e, dict) else e
        for e in (state.get("_self_corrections") or [])
    ]

    plan_path = output_dir / "analysis" / "investigation_plan.json"

    report = FindingsReport(
        task_id=task.id,
        prompt=task.prompt,
        evidence_refs=task.evidence_refs,
        conclusions=conclusions,
        anomalies=list(state.get("anomalies") or []),
        tool_failures=tool_failures,
        self_corrections=self_corrections,
        investigation_plan_path=plan_path,
    )

    ts = report.generated_at_utc.strftime("%Y%m%d_%H%M%S")
    slug = _slugify(task.prompt)
    stem = f"{ts}_{slug}"

    md_path = reports_dir / f"{stem}.md"
    json_path = reports_dir / f"{stem}.json"

    md_path.write_text(_render_markdown(report))
    json_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, default=str)
    )

    return {"report": report}


def _slugify(text: str) -> str:
    import re
    return re.sub(r"[^a-z0-9_]", "", text.lower().replace(" ", "_"))[:40]


def _render_markdown(report: FindingsReport) -> str:
    def fmt_conclusions(conclusions):
        if not conclusions:
            return "_No conclusions recorded._"
        lines = []
        for c in conclusions:
            lines.append(f"### {c.statement}")
            lines.append(f"**Confidence**: {c.confidence}")
            lines.append(f"**Supporting invocations**: {', '.join(c.supporting_invocation_ids)}")
            lines.append("")
        return "\n".join(lines)

    def fmt_anomalies(anomalies):
        if not anomalies:
            return "_No anomalies detected._"
        lines = []
        for a in anomalies:
            lines.append(f"- **{a.description}** ({a.response_action}): {a.forensic_significance}")
        return "\n".join(lines)

    def fmt_failures(failures):
        if not failures:
            return "_No tool failures._"
        lines = []
        for f in failures:
            lines.append(f"- Step `{f.step_id}`: {f.final_error}")
            lines.append(f"  Diagnostic: {f.diagnostic_context}")
        return "\n".join(lines)

    def fmt_corrections(corrections):
        if not corrections:
            return "_No self-corrections._"
        lines = []
        for c in corrections:
            lines.append(
                f"- Step `{c.step_id}` attempt {c.attempt_number}: "
                f"`{' '.join(c.original_cmd)}` → `{' '.join(c.corrected_cmd)}`"
            )
        return "\n".join(lines)

    return _REPORT_TEMPLATE.format(
        task_id=report.task_id,
        generated_at=report.generated_at_utc.strftime("%Y-%m-%d %H:%M:%S"),
        evidence=", ".join(report.evidence_refs),
        prompt=report.prompt,
        conclusions=fmt_conclusions(report.conclusions),
        anomalies=fmt_anomalies(report.anomalies),
        failures=fmt_failures(report.tool_failures),
        corrections=fmt_corrections(report.self_corrections),
    )
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/unit/test_report.py -v
```
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/valravn/nodes/report.py tests/unit/test_report.py
git commit -m "feat: write_findings_report node with Markdown and JSON output"
```

---

## Task 13: plan_investigation Needs LLM-Derived Conclusions + US1 Integration Test

**Files:**
- Create: `tests/integration/test_graph.py`
- Create: `tests/fixtures/evidence/stub.raw` (already created in Task 1)

The integration test mocks the LLM, runs the full graph against the stub evidence, and asserts the findings report exists with the expected structure.

- [ ] **Step 1: Write failing integration test**

```python
# tests/integration/test_graph.py
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from valravn.config import AppConfig, OutputConfig
from valravn.graph import run
from valravn.models.task import InvestigationTask


STUB_EVIDENCE = Path(__file__).parent.parent / "fixtures" / "evidence" / "stub.raw"


@pytest.fixture
def task(tmp_path):
    # stub.raw is committed as read-only (chmod 444)
    return InvestigationTask(
        prompt="identify file types in evidence",
        evidence_refs=[str(STUB_EVIDENCE)],
    )


@pytest.fixture
def out_cfg(tmp_path):
    cfg = OutputConfig(output_dir=tmp_path / "output")
    cfg.ensure_dirs()
    return cfg


def _mock_plan_llm():
    """Returns a mock structured LLM that produces one step: `file stub.raw`."""
    mock = MagicMock()
    step_spec = MagicMock()
    step_spec.skill_domain = "sleuthkit"
    step_spec.tool_cmd = ["file", str(STUB_EVIDENCE)]
    step_spec.rationale = "determine file type"
    plan_spec = MagicMock()
    plan_spec.steps = [step_spec]
    mock.invoke.return_value = plan_spec
    return mock


@pytest.mark.integration
def test_us1_full_investigation_produces_report(task, out_cfg):
    app_cfg = AppConfig()

    with patch("valravn.nodes.plan._get_llm", return_value=_mock_plan_llm()):
        exit_code = run(task=task, app_cfg=app_cfg, out_cfg=out_cfg)

    # Exit code 0 = success, 1 = completed with failures
    assert exit_code in (0, 1)

    reports = list((out_cfg.output_dir / "reports").glob("*.md"))
    assert len(reports) == 1, "Expected one findings report"

    json_reports = list((out_cfg.output_dir / "reports").glob("*.json"))
    assert len(json_reports) == 1

    plan_file = out_cfg.output_dir / "analysis" / "investigation_plan.json"
    assert plan_file.exists()

    # Evidence must not be modified
    assert os.access(STUB_EVIDENCE, os.R_OK)
    assert not os.access(STUB_EVIDENCE, os.W_OK)


@pytest.mark.integration
def test_us1_no_evidence_modification(task, out_cfg):
    """No file under fixtures/evidence/ should be written."""
    evidence_dir = Path(__file__).parent.parent / "fixtures" / "evidence"
    before = {p: p.stat().st_mtime for p in evidence_dir.rglob("*") if p.is_file()}

    with patch("valravn.nodes.plan._get_llm", return_value=_mock_plan_llm()):
        run(task=task, app_cfg=AppConfig(), out_cfg=out_cfg)

    after = {p: p.stat().st_mtime for p in evidence_dir.rglob("*") if p.is_file()}
    assert before == after, "Evidence files were modified!"
```

- [ ] **Step 2: Run to verify (expect partial failure — check_anomalies stub returns `{}`)**

```bash
pytest tests/integration/test_graph.py -v -m integration
```
The graph will run but may fail at `check_anomalies` routing. Note the error.

- [ ] **Step 3: Fix `check_anomalies` stub to return safe defaults**

```python
# src/valravn/nodes/anomaly.py — update stubs to return safe state
def check_anomalies(state: dict) -> dict:
    return {"_pending_anomalies": False}

def record_anomaly(state: dict) -> dict:
    return {}
```

Also fix `update_plan` to pass `_output_dir` through by reading it from state correctly — verify `_output_dir` is included in `initial_state` in `graph.py`:

```python
# In graph.py run(), add _output_dir and _retry_config to initial_state:
initial_state: AgentState = {
    "task": task,
    "plan": InvestigationPlan(task_id=task.id),
    "invocations": [],
    "anomalies": [],
    "report": None,
    "current_step_id": None,
    "skill_cache": {},
    "messages": [],
    "_output_dir": str(out_cfg.output_dir),
    "_retry_config": {
        "max_attempts": app_cfg.retry.max_attempts,
        "retry_delay_seconds": app_cfg.retry.retry_delay_seconds,
    },
    "_step_succeeded": False,
    "_step_exhausted": False,
    "_pending_anomalies": False,
    "_conclusions": [],
    "_tool_failures": [],
    "_self_corrections": [],
    "_last_invocation_id": None,
}
```

Also update `plan_investigation` to pass `_output_dir` through from `state`.

- [ ] **Step 4: Run integration test and verify it passes**

```bash
pytest tests/integration/test_graph.py -v -m integration
```
Expected: `2 passed`

- [ ] **Step 5: Run full test suite**

```bash
pytest -v
```
Expected: all unit tests pass; integration tests pass

- [ ] **Step 6: Commit — US1 MVP**

```bash
git add src/valravn/ tests/
git commit -m "feat: US1 complete — guided forensic investigation end-to-end"
```

---

## Task 14: check_anomalies and record_anomaly Nodes (US2)

**Files:**
- Modify: `src/valravn/nodes/anomaly.py`
- Create: `tests/unit/test_anomaly.py`
- Create: `tests/fixtures/evidence/anomaly_fixture/conflict_a.txt` and `conflict_b.txt`

- [ ] **Step 1: Create anomaly test fixtures**

```bash
mkdir -p tests/fixtures/evidence/anomaly_fixture
echo "EVENT: login at 2024-01-15 09:00:00" > tests/fixtures/evidence/anomaly_fixture/conflict_a.txt
echo "EVENT: login at 2025-06-20 14:30:00" > tests/fixtures/evidence/anomaly_fixture/conflict_b.txt
chmod 444 tests/fixtures/evidence/anomaly_fixture/conflict_a.txt
chmod 444 tests/fixtures/evidence/anomaly_fixture/conflict_b.txt
```

- [ ] **Step 2: Write failing tests**

```python
# tests/unit/test_anomaly.py
from unittest.mock import MagicMock, patch
import pytest
from valravn.models.records import AnomalyResponseAction
from valravn.nodes.anomaly import check_anomalies, record_anomaly
from valravn.models.task import InvestigationPlan, InvestigationTask, PlannedStep
from tests.conftest import *  # noqa


def _state(read_only_evidence, output_dir, pending_anomalies=False):
    task = InvestigationTask(prompt="test", evidence_refs=[str(read_only_evidence)])
    step = PlannedStep(skill_domain="sleuthkit", tool_cmd=["fls"], rationale="r")
    plan = InvestigationPlan(task_id=task.id, steps=[step])
    return {
        "task": task, "plan": plan, "invocations": [],
        "anomalies": [], "report": None,
        "current_step_id": step.id, "skill_cache": {},
        "messages": [], "_output_dir": str(output_dir),
        "_last_invocation_id": "inv-1",
        "_pending_anomalies": pending_anomalies,
        "_detected_anomaly_data": None,
    }


def test_check_anomalies_no_anomaly(read_only_evidence, output_dir):
    mock_llm = MagicMock()
    mock_result = MagicMock()
    mock_result.anomalies = []
    mock_llm.invoke.return_value = mock_result

    state = _state(read_only_evidence, output_dir)

    with patch("valravn.nodes.anomaly._get_anomaly_llm", return_value=mock_llm):
        result = check_anomalies(state)

    assert result["_pending_anomalies"] is False


def test_check_anomalies_detects_conflict(read_only_evidence, output_dir):
    mock_llm = MagicMock()
    mock_anomaly = MagicMock()
    mock_anomaly.description = "conflicting timestamps for same login event"
    mock_anomaly.forensic_significance = "possible log tampering"
    mock_anomaly.response_action = "added_follow_up_steps"
    mock_anomaly.follow_up_steps = []
    mock_result = MagicMock()
    mock_result.anomalies = [mock_anomaly]
    mock_llm.invoke.return_value = mock_result

    state = _state(read_only_evidence, output_dir)

    with patch("valravn.nodes.anomaly._get_anomaly_llm", return_value=mock_llm):
        result = check_anomalies(state)

    assert result["_pending_anomalies"] is True
    assert len(result["_detected_anomaly_data"]) == 1


def test_record_anomaly_persists_to_json(read_only_evidence, output_dir):
    import json
    from valravn.models.records import AnomalyResponseAction
    anomaly_data = [{
        "description": "conflict",
        "forensic_significance": "sig",
        "response_action": AnomalyResponseAction.NO_FOLLOW_UP,
        "follow_up_steps": [],
        "source_invocation_ids": ["inv-1"],
    }]
    state = _state(read_only_evidence, output_dir, pending_anomalies=True)
    state["_detected_anomaly_data"] = anomaly_data

    result = record_anomaly(state)

    anomalies_file = output_dir / "analysis" / "anomalies.json"
    assert anomalies_file.exists()
    data = json.loads(anomalies_file.read_text())
    assert len(data) == 1
    assert len(result["anomalies"]) == 1
```

- [ ] **Step 3: Run to verify failure**

```bash
pytest tests/unit/test_anomaly.py -v
```
Expected: failures (stubs don't return expected keys)

- [ ] **Step 4: Write `src/valravn/nodes/anomaly.py`**

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from valravn.models.records import Anomaly, AnomalyResponseAction
from valravn.models.task import PlannedStep


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class _AnomalySpec(BaseModel):
    description: str
    forensic_significance: str
    response_action: AnomalyResponseAction
    follow_up_steps: list[dict] = []  # list of _StepSpec dicts


class _AnomalyCheckResult(BaseModel):
    anomalies: list[_AnomalySpec] = []


_ANOMALY_SYSTEM = """\
You are a forensic anomaly detector. Given tool output from a DFIR investigation,
identify internal contradictions or unexpected results.

Watch for:
1. Timestamp contradictions (event before OS install, future timestamps)
2. Orphaned process relationships (parent PID absent or exited before child)
3. Cross-tool conflicts (two tools report different values for the same artifact)
4. Unexpected absences (empty output where output is architecturally certain)
5. Integrity failures (hash mismatch, truncated output, zero-byte file)

If no anomaly exists, return an empty anomalies list.
For each anomaly found, specify whether follow-up investigation steps are needed.
"""


def _get_anomaly_llm() -> object:
    llm = ChatAnthropic(model="claude-opus-4-6", temperature=0)
    return llm.with_structured_output(_AnomalyCheckResult)


def check_anomalies(state: dict) -> dict:
    """LangGraph node: scan tool output for anomalies."""
    last_inv_id = state.get("_last_invocation_id")
    invocations = state.get("invocations") or []

    if not last_inv_id or not invocations:
        return {"_pending_anomalies": False, "_detected_anomaly_data": None}

    last_inv = next((i for i in invocations if i.id == last_inv_id), None)
    if last_inv is None:
        return {"_pending_anomalies": False, "_detected_anomaly_data": None}

    stdout = Path(last_inv.stdout_path).read_text() if Path(last_inv.stdout_path).exists() else ""
    stderr = Path(last_inv.stderr_path).read_text() if Path(last_inv.stderr_path).exists() else ""

    messages = [
        SystemMessage(content=_ANOMALY_SYSTEM),
        HumanMessage(
            content=(
                f"Tool: {' '.join(last_inv.cmd)}\n"
                f"Exit code: {last_inv.exit_code}\n\n"
                f"STDOUT:\n{stdout[:4000]}\n\n"
                f"STDERR:\n{stderr[:1000]}"
            )
        ),
    ]

    result: _AnomalyCheckResult = _get_anomaly_llm().invoke(messages)

    if not result.anomalies:
        return {"_pending_anomalies": False, "_detected_anomaly_data": None}

    anomaly_data = [
        {
            "description": a.description,
            "forensic_significance": a.forensic_significance,
            "response_action": a.response_action,
            "follow_up_steps": a.follow_up_steps,
            "source_invocation_ids": [last_inv_id],
        }
        for a in result.anomalies
    ]

    return {"_pending_anomalies": True, "_detected_anomaly_data": anomaly_data}


def record_anomaly(state: dict) -> dict:
    """LangGraph node: persist detected anomalies and add follow-up steps."""
    from valravn.models.task import InvestigationPlan, PlannedStep

    anomaly_data: list[dict] = state.get("_detected_anomaly_data") or []
    plan: InvestigationPlan = state["plan"]
    output_dir = Path(state.get("_output_dir", "."))

    existing_anomalies = list(state.get("anomalies") or [])
    new_anomalies = []

    for ad in anomaly_data:
        follow_up_ids = []

        for step_spec in ad.get("follow_up_steps") or []:
            if isinstance(step_spec, dict):
                new_step = PlannedStep(
                    skill_domain=step_spec.get("skill_domain", "sleuthkit"),
                    tool_cmd=step_spec.get("tool_cmd", []),
                    rationale=step_spec.get("rationale", "anomaly follow-up"),
                )
            else:
                new_step = PlannedStep(
                    skill_domain=getattr(step_spec, "skill_domain", "sleuthkit"),
                    tool_cmd=getattr(step_spec, "tool_cmd", []),
                    rationale=getattr(step_spec, "rationale", "anomaly follow-up"),
                )
            plan.steps.append(new_step)
            follow_up_ids.append(new_step.id)

        anomaly = Anomaly(
            description=ad["description"],
            source_invocation_ids=ad["source_invocation_ids"],
            forensic_significance=ad["forensic_significance"],
            response_action=ad["response_action"],
            follow_up_step_ids=follow_up_ids,
        )
        new_anomalies.append(anomaly)

    all_anomalies = existing_anomalies + new_anomalies

    # Persist anomalies
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    anomalies_file = analysis_dir / "anomalies.json"
    anomalies_file.write_text(
        json.dumps(
            [a.model_dump(mode="json") for a in all_anomalies],
            indent=2,
            default=str,
        )
    )

    return {
        "anomalies": all_anomalies,
        "plan": plan,
        "_pending_anomalies": False,
        "_detected_anomaly_data": None,
    }
```

- [ ] **Step 5: Run tests and verify they pass**

```bash
pytest tests/unit/test_anomaly.py -v
```
Expected: `3 passed`

- [ ] **Step 6: Commit**

```bash
git add src/valravn/nodes/anomaly.py tests/unit/test_anomaly.py tests/fixtures/evidence/anomaly_fixture/
git commit -m "feat: US2 anomaly detection and recording nodes"
```

---

## Task 15: Retry Loop and Self-Correction (US3)

**Files:**
- Modify: `src/valravn/nodes/tool_runner.py`
- Modify: `tests/unit/test_tool_runner.py` (append)

The retry loop calls the LLM to get a corrective hypothesis and re-invokes with modified args.

- [ ] **Step 1: Append failing tests to `tests/unit/test_tool_runner.py`**

```python
# Append to tests/unit/test_tool_runner.py
from valravn.models.report import SelfCorrectionEvent, ToolFailureRecord


def test_retry_on_failure(read_only_evidence, output_dir):
    """Agent retries at least once when tool exits non-zero."""
    # First call fails, second succeeds
    call_count = {"n": 0}

    def fake_run(cmd, **kw):
        call_count["n"] += 1
        mock = MagicMock()
        if call_count["n"] == 1:
            mock.returncode = 1
            mock.stdout = ""
            mock.stderr = "No such file"
        else:
            mock.returncode = 0
            mock.stdout = "data"
            mock.stderr = ""
        return mock

    state = _state(read_only_evidence, output_dir, ["vol.py", "-f", "/missing.lime", "netstat"])
    state["_retry_config"] = {"max_attempts": 3, "retry_delay_seconds": 0.0}

    # LLM returns a corrected command
    corrected_cmd = ["vol.py", "-f", str(read_only_evidence), "windows.netstat"]
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(corrected_cmd=corrected_cmd, rationale="fixed path")

    with patch("subprocess.run", side_effect=fake_run), \
         patch("valravn.nodes.tool_runner._get_retry_llm", return_value=mock_llm):
        result = run_forensic_tool(state)

    assert call_count["n"] == 2
    assert len(result["invocations"]) == 2
    assert len(result.get("_self_corrections") or []) == 1


def test_exhaustion_after_max_retries(read_only_evidence, output_dir):
    """After max_attempts all fail, ToolFailureRecord is created."""
    def always_fail(cmd, **kw):
        mock = MagicMock()
        mock.returncode = 1
        mock.stdout = ""
        mock.stderr = "persistent error"
        return mock

    state = _state(read_only_evidence, output_dir, ["bad-tool", "arg"])
    state["_retry_config"] = {"max_attempts": 2, "retry_delay_seconds": 0.0}

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        corrected_cmd=["bad-tool", "arg2"], rationale="try different arg"
    )

    with patch("subprocess.run", side_effect=always_fail), \
         patch("valravn.nodes.tool_runner._get_retry_llm", return_value=mock_llm):
        result = run_forensic_tool(state)

    assert result["_step_exhausted"] is True
    assert result.get("_tool_failure") is not None
    assert len(result["invocations"]) == 2
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/unit/test_tool_runner.py::test_retry_on_failure -v
pytest tests/unit/test_tool_runner.py::test_exhaustion_after_max_retries -v
```
Expected: failures (no retry logic yet)

- [ ] **Step 3: Replace `src/valravn/nodes/tool_runner.py` with full retry implementation**

```python
from __future__ import annotations

import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from valravn.models.records import ToolInvocationRecord
from valravn.models.report import SelfCorrectionEvent, ToolFailureRecord


# ---------------------------------------------------------------------------
# LLM schema for corrective hypothesis
# ---------------------------------------------------------------------------

class _CorrectionSpec(BaseModel):
    corrected_cmd: list[str]
    rationale: str


_CORRECTION_SYSTEM = """\
You are a forensic tool expert on a SANS SIFT Ubuntu workstation.
A CLI tool invocation failed. Given the original command, exit code, and stderr,
propose a corrected command that may succeed.
Return ONLY the corrected argv list and a one-sentence rationale.
Do NOT suggest commands that write to evidence directories.
"""


def _get_retry_llm() -> object:
    llm = ChatAnthropic(model="claude-opus-4-6", temperature=0)
    return llm.with_structured_output(_CorrectionSpec)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def _invoke_once(
    cmd: list[str],
    step_id: str,
    attempt: int,
    analysis_dir: Path,
) -> ToolInvocationRecord:
    inv_id = str(uuid.uuid4())
    stdout_path = analysis_dir / f"{inv_id}.stdout"
    stderr_path = analysis_dir / f"{inv_id}.stderr"

    started = datetime.now(timezone.utc)
    t0 = time.monotonic()

    proc = subprocess.run(cmd, capture_output=True, text=True)

    duration = time.monotonic() - t0
    completed = datetime.now(timezone.utc)

    stdout_path.write_text(proc.stdout)
    stderr_path.write_text(proc.stderr)

    return ToolInvocationRecord(
        id=inv_id,
        step_id=step_id,
        attempt_number=attempt,
        cmd=cmd,
        exit_code=proc.returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        started_at_utc=started,
        completed_at_utc=completed,
        duration_seconds=duration,
        had_output=bool(proc.stdout.strip()),
    )


def run_forensic_tool(state: dict) -> dict:
    """LangGraph node: invoke tool with retry and LLM-guided self-correction."""
    plan = state["plan"]
    step_id: str = state["current_step_id"]
    step = next(s for s in plan.steps if s.id == step_id)

    retry_cfg: dict = state.get("_retry_config") or {}
    max_attempts: int = retry_cfg.get("max_attempts", 3)
    retry_delay: float = retry_cfg.get("retry_delay_seconds", 0.0)

    output_dir = Path(state.get("_output_dir", "."))
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    invocations = list(state.get("invocations") or [])
    self_corrections = list(state.get("_self_corrections") or [])

    current_cmd = list(step.tool_cmd)
    rec: ToolInvocationRecord | None = None

    for attempt in range(1, max_attempts + 1):
        if attempt > 1 and retry_delay > 0:
            time.sleep(retry_delay)

        rec = _invoke_once(current_cmd, step_id, attempt, analysis_dir)
        invocations.append(rec)
        step.invocation_ids.append(rec.id)

        if rec.success:
            return {
                "invocations": invocations,
                "plan": plan,
                "_step_succeeded": True,
                "_step_exhausted": False,
                "_last_invocation_id": rec.id,
                "_self_corrections": self_corrections,
                "_tool_failure": None,
            }

        # Failure: ask LLM for corrective hypothesis (if retries remain)
        if attempt < max_attempts:
            correction = _get_retry_llm().invoke([
                SystemMessage(content=_CORRECTION_SYSTEM),
                HumanMessage(
                    content=(
                        f"Original command: {current_cmd}\n"
                        f"Exit code: {rec.exit_code}\n"
                        f"Stderr: {Path(rec.stderr_path).read_text()[:2000]}"
                    )
                ),
            ])
            self_corrections.append(
                SelfCorrectionEvent(
                    step_id=step_id,
                    attempt_number=attempt,
                    original_cmd=current_cmd,
                    corrected_cmd=correction.corrected_cmd,
                    correction_rationale=correction.rationale,
                )
            )
            current_cmd = correction.corrected_cmd

    # All attempts exhausted
    assert rec is not None
    failure = ToolFailureRecord(
        step_id=step_id,
        invocation_ids=[i.id for i in invocations if i.step_id == step_id],
        final_error=Path(rec.stderr_path).read_text()[:2000],
        diagnostic_context="; ".join(
            f"attempt {sc.attempt_number}: {sc.correction_rationale}"
            for sc in self_corrections
        ),
    )

    return {
        "invocations": invocations,
        "plan": plan,
        "_step_succeeded": False,
        "_step_exhausted": True,
        "_last_invocation_id": rec.id,
        "_self_corrections": self_corrections,
        "_tool_failure": failure,
    }
```

- [ ] **Step 4: Update `update_plan` in `nodes/plan.py` to collect tool failures**

Add to `update_plan` — after reading `_step_exhausted`:

```python
# In update_plan(), append after determining status:
tool_failures = list(state.get("_tool_failures") or [])
if state.get("_tool_failure") is not None:
    tool_failures.append(state["_tool_failure"])

# Add to return dict:
return {
    "plan": plan,
    "current_step_id": next_id,
    "_step_succeeded": False,
    "_step_exhausted": False,
    "_pending_anomalies": False,
    "_tool_failure": None,
    "_tool_failures": tool_failures,
}
```

- [ ] **Step 5: Update `write_findings_report` to read `_self_corrections` from state**

In `write_findings_report`, replace the `self_corrections` line with:

```python
self_corrections = [
    SelfCorrectionEvent(**e) if isinstance(e, dict) else e
    for e in (state.get("_self_corrections") or [])
]
```

- [ ] **Step 6: Run all tests**

```bash
pytest -v
```
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add src/valravn/nodes/tool_runner.py src/valravn/nodes/plan.py src/valravn/nodes/report.py tests/unit/test_tool_runner.py
git commit -m "feat: US3 retry loop with LLM-guided self-correction and failure escalation"
```

---

## Task 16: MLflow Evaluators (Evaluation Infrastructure)

**Files:**
- Create: `src/valravn/evaluation/evaluators.py`

- [ ] **Step 1: Write `src/valravn/evaluation/evaluators.py`**

```python
"""
MLflow-based evaluators for Valravn success criteria.

Run: python -m valravn.evaluation.evaluators --suite all
Requires: mlflow server running at config.mlflow.tracking_uri
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlflow


def _load_report(report_json_path: Path) -> dict:
    return json.loads(report_json_path.read_text())


# ---------------------------------------------------------------------------
# Individual evaluators — each returns (passed: bool, score: float, detail: str)
# ---------------------------------------------------------------------------

def eval_citation_coverage(report: dict) -> tuple[bool, float, str]:
    """SC-004: every conclusion must cite at least one invocation."""
    conclusions = report.get("conclusions") or []
    if not conclusions:
        return True, 1.0, "no conclusions (vacuously true)"
    uncited = [c["statement"] for c in conclusions if not c.get("supporting_invocation_ids")]
    score = 1.0 - len(uncited) / len(conclusions)
    passed = len(uncited) == 0
    detail = f"{len(uncited)} uncited conclusions" if uncited else "all conclusions cited"
    return passed, score, detail


def eval_utc_timestamps(report: dict) -> tuple[bool, float, str]:
    """SC-006: generated_at_utc must end with +00:00 or Z."""
    ts = report.get("generated_at_utc", "")
    passed = ts.endswith("+00:00") or ts.endswith("Z") or "UTC" in ts
    return passed, 1.0 if passed else 0.0, ts


def eval_no_evidence_modification(report: dict, evidence_refs: list[str]) -> tuple[bool, float, str]:
    """SC-005: evidence paths must not have been written during the run."""
    import os
    violations = [r for r in evidence_refs if os.access(r, os.W_OK)]
    passed = len(violations) == 0
    return passed, 1.0 if passed else 0.0, f"writable evidence: {violations}"


def eval_anomaly_detection(report: dict, expected_anomaly_count: int) -> tuple[bool, float, str]:
    """SC-002: detected anomaly count >= expected."""
    detected = len(report.get("anomalies") or [])
    passed = detected >= expected_anomaly_count
    score = min(detected / max(expected_anomaly_count, 1), 1.0)
    return passed, score, f"detected {detected}/{expected_anomaly_count}"


def eval_self_correction(report: dict, expected_correction: bool) -> tuple[bool, float, str]:
    """SC-003: self_corrections list is non-empty if a recoverable failure was introduced."""
    corrections = report.get("self_corrections") or []
    if not expected_correction:
        return True, 1.0, "no correction expected"
    passed = len(corrections) > 0
    return passed, 1.0 if passed else 0.0, f"{len(corrections)} corrections recorded"


# ---------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------

SUITES = {
    "citation-coverage": ["citation_coverage"],
    "utc-timestamps": ["utc_timestamps"],
    "anomaly-detection": ["anomaly_detection"],
    "self-correction": ["self_correction"],
    "all": ["citation_coverage", "utc_timestamps", "anomaly_detection", "self_correction"],
}


def run_suite(suite: str, datasets_dir: Path, tracking_uri: str) -> int:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("valravn-evaluation")

    golden_path = datasets_dir / "golden.jsonl"
    if not golden_path.exists():
        print(f"No golden dataset at {golden_path}. Add cases with: "
              "python -m valravn.evaluation.datasets --add <report.json>",
              file=sys.stderr)
        return 1

    cases = [json.loads(line) for line in golden_path.read_text().splitlines() if line.strip()]
    if not cases:
        print("Golden dataset is empty.", file=sys.stderr)
        return 1

    all_passed = True

    with mlflow.start_run(run_name=f"eval-{suite}"):
        for i, case in enumerate(cases):
            report = case.get("report", {})
            meta = case.get("meta", {})
            evidence_refs = report.get("evidence_refs", [])

            results: dict[str, tuple[bool, float, str]] = {}

            checks = SUITES.get(suite, SUITES["all"])

            if "citation_coverage" in checks:
                results["citation_coverage"] = eval_citation_coverage(report)
            if "utc_timestamps" in checks:
                results["utc_timestamps"] = eval_utc_timestamps(report)
            if "anomaly_detection" in checks:
                expected = meta.get("expected_anomaly_count", 0)
                results["anomaly_detection"] = eval_anomaly_detection(report, expected)
            if "self_correction" in checks:
                expected = meta.get("expected_self_correction", False)
                results["self_correction"] = eval_self_correction(report, expected)

            for metric_name, (passed, score, detail) in results.items():
                mlflow.log_metric(f"case_{i}_{metric_name}_score", score)
                mlflow.log_metric(f"case_{i}_{metric_name}_passed", int(passed))
                if not passed:
                    all_passed = False
                    print(f"  FAIL case {i} [{metric_name}]: {detail}")
                else:
                    print(f"  PASS case {i} [{metric_name}]: {detail}")

        mlflow.log_metric("all_passed", int(all_passed))

    print(f"\nResults logged to MLflow at {tracking_uri}")
    return 0 if all_passed else 1


def main() -> None:
    p = argparse.ArgumentParser(prog="valravn.evaluation.evaluators")
    p.add_argument("--suite", default="all", choices=list(SUITES.keys()))
    p.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent.parent / "tests" / "evaluation" / "datasets",
    )
    p.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    args = p.parse_args()
    sys.exit(run_suite(args.suite, args.datasets_dir, args.tracking_uri))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create golden dataset tooling**

```python
# src/valravn/evaluation/datasets.py
"""Add verified findings reports to the golden evaluation dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DATASETS_DIR = Path(__file__).parent.parent.parent.parent / "tests" / "evaluation" / "datasets"


def add_report(report_json_path: Path, expected_anomaly_count: int, expected_self_correction: bool) -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    golden_path = DATASETS_DIR / "golden.jsonl"

    report = json.loads(report_json_path.read_text())
    entry = {
        "report": report,
        "meta": {
            "expected_anomaly_count": expected_anomaly_count,
            "expected_self_correction": expected_self_correction,
            "source": str(report_json_path),
        },
    }

    with open(golden_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"Added to {golden_path}")


def main() -> None:
    p = argparse.ArgumentParser(prog="valravn.evaluation.datasets")
    p.add_argument("--add", type=Path, required=True, metavar="REPORT_JSON")
    p.add_argument("--expected-anomalies", type=int, default=0)
    p.add_argument("--expected-self-correction", action="store_true", default=False)
    args = p.parse_args()
    add_report(args.add, args.expected_anomalies, args.expected_self_correction)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify imports**

```bash
python -c "from valravn.evaluation.evaluators import run_suite; print('OK')"
python -c "from valravn.evaluation.datasets import add_report; print('OK')"
```
Expected: `OK` twice

- [ ] **Step 4: Commit**

```bash
git add src/valravn/evaluation/evaluators.py src/valravn/evaluation/datasets.py
git commit -m "feat: MLflow evaluators for SC-002 through SC-006 and golden dataset tooling"
```

---

## Task 17: Final Validation

- [ ] **Step 1: Run the full test suite**

```bash
pytest -v
```
Expected: all unit tests pass; integration tests pass with `-m integration`

- [ ] **Step 2: Lint**

```bash
ruff check .
```
Expected: no errors

- [ ] **Step 3: Validate the CLI end-to-end against the stub fixture**

```bash
export ANTHROPIC_API_KEY=<your-key>
valravn investigate \
  --prompt "identify the file type of the evidence" \
  --evidence tests/fixtures/evidence/stub.raw \
  --output-dir /tmp/valravn-test
```
Expected: runs to completion, `./reports/` contains a `.md` file.

- [ ] **Step 4: Confirm no evidence modification**

```bash
ls -la tests/fixtures/evidence/stub.raw
```
Expected: permissions still `r--r--r--`

- [ ] **Step 5: Verify trace file was written**

```bash
ls /tmp/valravn-test/analysis/traces/
```
Expected: one `.jsonl` file

- [ ] **Step 6: Verify checkpoint DB was written**

```bash
ls /tmp/valravn-test/analysis/checkpoints.db
```
Expected: file exists

- [ ] **Step 7: Final commit**

```bash
git add .
git commit -m "chore: final validation — all tests pass, end-to-end confirmed"
```

---

## Self-Review

### Spec Coverage

| Requirement | Task |
|-------------|------|
| FR-001: Accept prompt + evidence refs | Task 8 (CLI) |
| FR-002: Sequence tools autonomously | Task 10 (plan_investigation) |
| FR-003: Use authoritative tool paths | Task 10 (system prompt enforces SIFT paths) |
| FR-004: Capture stdout/stderr before deriving conclusions | Task 11 (tool_runner) |
| FR-005: Detect internal inconsistencies | Task 14 (check_anomalies) |
| FR-006: Adjust plan when anomaly detected | Task 14 (record_anomaly adds follow-up steps) |
| FR-007: Never write to evidence dirs | Task 3 (model validator) + Task 11 (path check) |
| FR-008: Read error → hypothesis → retry | Task 15 (retry loop) |
| FR-009: Configurable retry limit, default 3 | Task 2 (RetryConfig) |
| FR-010: Final report cites tool invocations | Task 12 (FindingsReport) |
| FR-011: All timestamps UTC | Tasks 3–5 (model defaults), Task 12 (report) |
| FR-012: Zero operator prompts mid-task | Task 9 (graph runs to END with no interrupt) |
| FR-013: Report failures explicitly, no silent failure | Task 15 (ToolFailureRecord + exit code 1) |
| SC-001–SC-006 | Task 16 (MLflow evaluators) |

### No Placeholders Found

All tasks contain complete code. No "TBD", "TODO", or "implement later" strings.

### Type Consistency Check

- `_get_llm()` in `plan.py` and `_get_retry_llm()` in `tool_runner.py` both return objects with `.invoke()` — consistent with mocking pattern in tests.
- `AgentState` TypedDict fields match what every node reads/writes: `_output_dir`, `_retry_config`, `_step_succeeded`, `_step_exhausted`, `_pending_anomalies`, `_detected_anomaly_data`, `_last_invocation_id`, `_conclusions`, `_tool_failures`, `_self_corrections`, `_tool_failure` — all used consistently across `graph.py` initial state and node implementations.
- `InvestigationPlan.next_pending_step()` defined in Task 3 and called in `graph.py` routing — consistent.
- `ToolInvocationRecord.success` property (Task 4) used in `tool_runner.py` (Task 15) — consistent.

---

**Plan complete and saved to `docs/superpowers/plans/2026-04-05-autonomous-dfir-agents.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
