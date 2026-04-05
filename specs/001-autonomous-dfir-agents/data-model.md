# Data Model: Autonomous DFIR Agents

**Branch**: `001-autonomous-dfir-agents` | **Date**: 2026-04-05

All entities extracted from `spec.md` Key Entities section plus fields derived from
functional requirements. Implemented as Pydantic v2 models in `src/valravn/models/`.

The top-level runtime container is `AgentState` (a `TypedDict` in `src/valravn/state.py`),
which is the LangGraph graph state passed between nodes. All entities below are fields
within or serialised from `AgentState`. LangGraph's checkpointer handles state persistence
between nodes — hand-rolled JSON persistence is limited to `./analysis/` audit artefacts.

---

## Entity: AgentState

**Source**: LangGraph graph state; contains all entities below as fields
**Location**: `src/valravn/state.py`

```python
class AgentState(TypedDict):
    task: InvestigationTask
    plan: InvestigationPlan
    invocations: list[ToolInvocationRecord]
    anomalies: list[Anomaly]
    report: FindingsReport | None
    current_step_id: str | None        # ID of the step currently executing
    skill_cache: dict[str, str]        # domain → loaded SKILL.md content
    messages: list[BaseMessage]        # LangGraph message history (for LLM nodes)
```

Passed between every LangGraph node. LangGraph's `SqliteSaver` checkpointer persists
state to `./analysis/checkpoints.db` after every node execution — survives process
crashes mid-investigation. `FileCallbackHandler` traces the full node/LLM/tool history
to `./analysis/traces/<run-id>.jsonl` — no cloud service required.

---

## Entity: InvestigationTask

**Source**: spec.md Key Entities; FR-001
**Location**: `src/valravn/models/task.py`

```python
class InvestigationTask(BaseModel):
    id: str                        # UUID4, generated at invocation
    prompt: str                    # Natural-language investigation prompt (FR-001)
    evidence_refs: list[str]       # Absolute paths to evidence mount points (FR-001)
    created_at_utc: datetime       # UTC, set at CLI invocation (FR-011)
    config: RetryConfig            # Retry and output settings (FR-009)
    output_dir: Path               # Resolved case working directory
```

**Validation rules**:
- `prompt` must be non-empty
- Each path in `evidence_refs` must exist and must NOT be writable by the process (enforces FR-007)
- `output_dir` must not overlap with any `evidence_refs` path or its ancestors

---

## Entity: InvestigationPlan

**Source**: spec.md Key Entities; FR-002, FR-006
**Location**: `src/valravn/models/task.py`

```python
class InvestigationPlan(BaseModel):
    task_id: str
    steps: list[PlannedStep]       # Ordered; agent may append during execution
    created_at_utc: datetime
    last_updated_utc: datetime
```

**Persistence**: Held in `AgentState` and checkpointed to `./analysis/checkpoints.db`
by `SqliteSaver` after every node execution. Also written to
`./analysis/investigation_plan.json` after every mutation for human-readable audit.

---

## Entity: PlannedStep

**Source**: derived from FR-002, FR-003, FR-006
**Location**: `src/valravn/models/task.py`

```python
class StepStatus(str, Enum):
    PENDING    = "pending"
    RUNNING    = "running"
    COMPLETED  = "completed"
    FAILED     = "failed"      # All retries exhausted
    EXHAUSTED  = "exhausted"   # Retries exhausted, escalated
    SKIPPED    = "skipped"     # Anomaly response: step not needed

class PlannedStep(BaseModel):
    id: str                        # UUID4
    skill_domain: str              # e.g., "memory-analysis", "sleuthkit"
    tool_cmd: list[str]            # Exact subprocess argv (FR-003)
    rationale: str                 # Why this step is in the plan
    status: StepStatus = StepStatus.PENDING
    depends_on: list[str] = []     # step IDs that must complete first
    invocation_ids: list[str] = [] # ToolInvocationRecord IDs produced by this step
```

---

## Entity: ToolInvocationRecord

**Source**: spec.md Key Entities; FR-004, FR-008, FR-013
**Location**: `src/valravn/models/records.py`

```python
class ToolInvocationRecord(BaseModel):
    id: str                        # UUID4
    step_id: str                   # Parent PlannedStep.id
    attempt_number: int            # 1-indexed (FR-008: retry tracking)
    cmd: list[str]                 # Exact argv executed
    exit_code: int
    stdout_path: Path              # Absolute path under ./analysis/ (FR-004)
    stderr_path: Path              # Absolute path under ./analysis/
    started_at_utc: datetime       # FR-011
    completed_at_utc: datetime     # FR-011
    duration_seconds: float
    success: bool                  # exit_code == 0 and stdout non-empty
```

**Persistence**: One JSON sidecar per invocation: `./analysis/<id>.json`.
Stdout/stderr written to `./analysis/<id>.stdout` and `./analysis/<id>.stderr`.

---

## Entity: Anomaly

**Source**: spec.md Key Entities; FR-005, FR-006
**Location**: `src/valravn/models/records.py`

```python
class AnomalyResponseAction(str, Enum):
    ADDED_FOLLOW_UP    = "added_follow_up_steps"
    NO_FOLLOW_UP       = "no_follow_up_warranted"
    INVESTIGATION_HALT = "investigation_cannot_proceed"

class Anomaly(BaseModel):
    id: str                           # UUID4
    description: str                  # Human-readable contradiction statement
    source_invocation_ids: list[str]  # ToolInvocationRecord IDs that produced the anomaly
    forensic_significance: str        # Why this matters for the investigation
    response_action: AnomalyResponseAction
    follow_up_step_ids: list[str]     # PlannedStep IDs added in response (if any)
    detected_at_utc: datetime         # FR-011
```

**Persistence**: Appended to `./analysis/anomalies.json`.

---

## Entity: FindingsReport

**Source**: spec.md Key Entities; FR-010, FR-013
**Location**: `src/valravn/models/report.py`

```python
class Conclusion(BaseModel):
    statement: str                    # The forensic finding
    supporting_invocation_ids: list[str]  # Must be non-empty (SC-004)
    confidence: Literal["high", "medium", "low"]

class ToolFailureRecord(BaseModel):
    step_id: str
    invocation_ids: list[str]         # All attempts
    final_error: str                  # stderr of last attempt
    diagnostic_context: str           # Agent's corrective hypothesis history

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
    generated_at_utc: datetime        # FR-011
    conclusions: list[Conclusion]     # SC-004: every conclusion cites invocations
    anomalies: list[Anomaly]
    tool_failures: list[ToolFailureRecord]   # FR-013
    self_corrections: list[SelfCorrectionEvent]
    investigation_plan_path: Path     # Link to ./analysis/investigation_plan.json
```

**Output**: Rendered as Markdown to `./reports/<YYYYMMDD_HHMMSS>_<prompt-slug>.md`.
JSON form also written to `./reports/<YYYYMMDD_HHMMSS>_<prompt-slug>.json` for
machine-readable consumption.

---

## Entity: RetryConfig

**Source**: FR-009; research.md R-005
**Location**: `src/valravn/config.py`

```python
class RetryConfig(BaseModel):
    max_attempts: int = 3             # FR-009: configurable, default 3
    retry_delay_seconds: float = 0.0  # 0 for deterministic CLI tools
```

Loaded from `config.yaml` (case dir or repo root), with `VALRAVN_MAX_RETRIES` env override.

---

## State Transitions

```
InvestigationTask created
  → InvestigationPlan generated (initial steps from prompt analysis)
    → PlannedStep: pending → running
      → ToolInvocationRecord created (stdout/stderr captured to ./analysis/)
        → success: PlannedStep → completed
        → failure: retry loop
          → ToolInvocationRecord (attempt 2) ...
          → max_attempts reached: PlannedStep → exhausted
                                  ToolFailureRecord added to FindingsReport
      → Anomaly detected during output processing
        → Anomaly persisted to anomalies.json
        → PlannedStep status update (follow-up added OR step skipped)
  → All steps terminal: FindingsReport generated → ./reports/
```

---

## Validation Rules Summary

| Entity | Rule | Enforcement |
|--------|------|-------------|
| InvestigationTask | evidence_refs paths not writable | FR-007; checked at startup |
| InvestigationTask | output_dir not ancestor of evidence | FR-007; checked at startup |
| ToolInvocationRecord | stdout_path under ./analysis/ | FR-004; OutputConfig validator |
| FindingsReport | every Conclusion has ≥1 supporting_invocation_ids | SC-004; model validator |
| FindingsReport | generated_at_utc is UTC | FR-011; `datetime.timezone.utc` enforced |
| Anomaly | source_invocation_ids non-empty | FR-005; model validator |
