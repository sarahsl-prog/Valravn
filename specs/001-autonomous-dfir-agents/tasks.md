# Tasks: Autonomous DFIR Agents

**Input**: Design documents from `/specs/001-autonomous-dfir-agents/`
**Branch**: `001-autonomous-dfir-agents`

**Organization**: Tasks are grouped by user story to enable independent implementation
and testing of each story. No test tasks are included — the spec does not request TDD.
Integration test scaffolding is included within each user story phase as it is required
to validate the independent test criterion.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no shared dependencies)
- **[Story]**: User story this task belongs to (US1, US2, US3)

---

## Phase 1: Setup

**Purpose**: Project initialization, dependency declaration, tooling configuration.

- [ ] T001 Create project structure per plan.md: `src/valravn/`, `src/valravn/nodes/`, `src/valravn/models/`, `src/valravn/evaluation/`, `tests/unit/`, `tests/integration/`, `tests/fixtures/evidence/`, `tests/evaluation/datasets/`, with `__init__.py` files
- [ ] T002 Create `pyproject.toml` with all dependencies: `langgraph`, `langgraph-checkpoint-sqlite`, `langchain-anthropic`, `mlflow`, `pydantic>=2`, `pytest`, `pytest-mock`, `ruff`; configure package entry point `valravn=src/valravn/cli:main`
- [ ] T003 [P] Configure `ruff` lint + format rules in `pyproject.toml`
- [ ] T004 [P] Create `config.yaml` template at repo root with `retry.max_attempts: 3`, `retry.retry_delay_seconds: 0`, `mlflow.tracking_uri: http://127.0.0.1:5000`, `mlflow.experiment_name: valravn-evaluation`

**Checkpoint**: `pip install -e ".[dev]"` succeeds; `ruff check .` passes on empty stubs

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Data models, config, agent state, graph skeleton, and CLI entry point that
every user story depends on.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T005 [P] Implement `RetryConfig` and `OutputConfig` Pydantic models with YAML file loading and `VALRAVN_MAX_RETRIES` env override in `src/valravn/config.py`
- [ ] T006 [P] Implement `InvestigationTask`, `InvestigationPlan`, `PlannedStep`, and `StepStatus` enum in `src/valravn/models/task.py`; include validation that `evidence_refs` paths are not writable (FR-007)
- [ ] T007 [P] Implement `ToolInvocationRecord` and `Anomaly` (with `AnomalyResponseAction` enum) Pydantic models in `src/valravn/models/records.py`
- [ ] T008 [P] Implement `FindingsReport`, `Conclusion`, `ToolFailureRecord`, and `SelfCorrectionEvent` Pydantic models in `src/valravn/models/report.py`; add model validator enforcing `Conclusion.supporting_invocation_ids` is non-empty (SC-004)
- [ ] T009 Implement `AgentState` TypedDict in `src/valravn/state.py` with fields: `task`, `plan`, `invocations`, `anomalies`, `report`, `current_step_id`, `skill_cache`, `messages` (depends on T006–T008)
- [ ] T010 [P] Implement `load_skill` node in `src/valravn/nodes/skill_loader.py`: resolve domain key to `~/.claude/skills/<domain>/SKILL.md`, read and return contents; raise `SkillNotFoundError` for unknown domains
- [ ] T011 Implement CLI entry point in `src/valravn/cli.py`: parse `--prompt`, `--evidence` (one or more), `--config`, `--output-dir`; validate evidence paths are not writable (exit code 2 on failure); construct `InvestigationTask`; call `graph.run()`
- [ ] T012 Implement `LangGraph StateGraph` skeleton in `src/valravn/graph.py`: instantiate `SqliteSaver` checkpointer writing to `<output_dir>/analysis/checkpoints.db`; attach `FileCallbackHandler` writing JSONL traces to `<output_dir>/analysis/traces/<run_id>.jsonl`; define all node slots (implementations added per story); compile graph

**Checkpoint**: `valravn investigate --help` executes without error; evidence path validation rejects writable paths with exit code 2

---

## Phase 3: User Story 1 — Guided Forensic Investigation (Priority: P1) 🎯 MVP

**Goal**: Agent accepts a prompt + evidence reference, independently sequences forensic
tool invocations via skill routing, and delivers a complete cited findings report with
zero operator intervention.

**Independent Test**: Run `valravn investigate --prompt "Identify active network connections at time of acquisition" --evidence <synthetic-fixture>` against the memory fixture in `tests/fixtures/evidence/`. Verify `./reports/` contains a Markdown report citing at least one tool invocation; exit code is 0 or 1; no file in `tests/fixtures/evidence/` is modified.

- [ ] T013 [P] [US1] Implement `plan_investigation` node in `src/valravn/nodes/plan.py`: send prompt + evidence metadata to Claude via `langchain-anthropic`; parse response into initial `list[PlannedStep]`; write initial `investigation_plan.json` to `<output_dir>/analysis/`
- [ ] T014 [P] [US1] Create minimal synthetic memory fixture in `tests/fixtures/evidence/memory.lime.stub` (a small valid file that produces known `strings`/`file` output — does not need to be a real memory image for US1 smoke test)
- [ ] T015 [US1] Implement `run_forensic_tool` node (basic, no retry yet) in `src/valravn/nodes/tool_runner.py`: invoke `subprocess.run()` with the step's `tool_cmd`; capture stdout to `<output_dir>/analysis/<uuid>.stdout`, stderr to `<output_dir>/analysis/<uuid>.stderr`; persist `ToolInvocationRecord` JSON; validate output paths are not under evidence directories (FR-007)
- [ ] T016 [US1] Implement `update_plan` node in `src/valravn/nodes/plan.py`: mark current step `completed` or `failed`; append follow-up steps if any; persist updated `investigation_plan.json`
- [ ] T017 [US1] Implement `write_findings_report` node in `src/valravn/nodes/report.py`: render `FindingsReport` to `<output_dir>/reports/<timestamp>_<slug>.md` and `.json`; all timestamps UTC (FR-011); every `Conclusion` must cite at least one `ToolInvocationRecord` (SC-004)
- [ ] T018 [US1] Wire US1 graph edges in `src/valravn/graph.py`: `START → plan_investigation → load_skill → run_forensic_tool → update_plan → [loop back to load_skill if steps remain, else] → write_findings_report → END`; add conditional edge routing on `StepStatus`
- [ ] T019 [US1] Add `tests/integration/test_graph.py` with a US1 end-to-end test: invoke graph against synthetic fixture, assert `./reports/` file exists, assert no evidence fixture was modified, assert exit code

**Checkpoint**: US1 independent test passes end-to-end; `./reports/<slug>.md` exists and cites tool output; no evidence fixture modified

---

## Phase 4: User Story 2 — Anomaly Recognition and Escalation (Priority: P2)

**Goal**: Agent detects internal contradictions in tool output, records them as structured
anomalies, and adjusts the investigation path in response.

**Independent Test**: Run the agent against a synthetic evidence fixture with a known injected contradiction (conflicting timestamp in `tests/fixtures/evidence/anomaly_fixture/`). Verify `./analysis/anomalies.json` contains at least one entry describing the contradiction; the findings report lists the anomaly; investigation path changed (follow-up step added or step skipped with reason).

- [ ] T020 [P] [US2] Create synthetic anomaly fixture in `tests/fixtures/evidence/anomaly_fixture/` that produces output with a detectable contradiction (e.g., two text files with conflicting timestamps for the same event)
- [ ] T021 [US2] Implement `check_anomalies` node in `src/valravn/nodes/anomaly.py`: pass tool output to Claude with anomaly-category system prompt (timestamp contradictions, orphaned relationships, cross-tool conflicts, unexpected absences, integrity failures); return list of detected anomalies or empty list
- [ ] T022 [US2] Implement `record_anomaly` node in `src/valravn/nodes/anomaly.py`: append each `Anomaly` to `AgentState.anomalies`; persist to `<output_dir>/analysis/anomalies.json`; add follow-up `PlannedStep` entries to plan if `response_action == ADDED_FOLLOW_UP` (FR-006)
- [ ] T023 [US2] Add conditional edge in `src/valravn/graph.py` after `run_forensic_tool`: route to `check_anomalies` → branch on anomaly detected (`record_anomaly → update_plan`) vs not detected (`update_plan` directly)
- [ ] T024 [US2] Integrate `AgentState.anomalies` into `FindingsReport` in `src/valravn/nodes/report.py`; ensure anomaly entries appear in the Markdown report with contradiction description and response action

**Checkpoint**: Agent correctly records the injected contradiction in `anomalies.json` and the findings report; investigation plan shows a follow-up step or explicit skip reason

---

## Phase 5: User Story 3 — Self-Correction on Tool Failure (Priority: P3)

**Goal**: On tool failure the agent reads the error, formulates a corrective hypothesis,
retries with a modified approach up to `max_attempts`, and escalates with full diagnostic
context if all retries are exhausted.

**Independent Test**: Point the agent at a non-existent evidence path as a deliberate failure. Verify: at least one retry with modified arguments is attempted; `./analysis/` contains `ToolInvocationRecord` entries for each attempt; the findings report contains a `ToolFailureRecord` with all diagnostic context; exit code is 1 (not 0, not silent).

- [ ] T025 [US3] Extend `run_forensic_tool` in `src/valravn/nodes/tool_runner.py` with the full retry loop: on non-zero exit or empty stdout — capture stderr, build corrective context dict (`{exit_code, stderr, attempt, hypothesis_prompt}`), return to agent; agent calls `run_forensic_tool` again with modified `tool_cmd`; repeat up to `RetryConfig.max_attempts`
- [ ] T026 [US3] Implement `SelfCorrectionEvent` recording in `src/valravn/nodes/tool_runner.py`: for each retry, persist a `SelfCorrectionEvent` (original cmd, corrected cmd, rationale) to `AgentState`
- [ ] T027 [US3] Implement retry-exhaustion path in `src/valravn/nodes/tool_runner.py`: when `max_attempts` reached, create `ToolFailureRecord` with all attempt `ToolInvocationRecord` IDs and final stderr; set `PlannedStep.status = EXHAUSTED`; never halt silently (FR-013)
- [ ] T028 [US3] Add exhaustion conditional edge in `src/valravn/graph.py`: `EXHAUSTED` step → determine if investigation can continue (other steps remain) → route to `update_plan` (continue) or `write_findings_report` (cannot proceed); exit code 1 if any failures
- [ ] T029 [US3] Integrate `ToolFailureRecord` and `SelfCorrectionEvent` lists from `AgentState` into `FindingsReport` in `src/valravn/nodes/report.py`

**Checkpoint**: Deliberate tool failure triggers at least one retry; `ToolFailureRecord` appears in findings report with full diagnostic context; exit code is 1; investigation does not halt silently

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Evaluation infrastructure, unit test coverage, and final validation.

- [ ] T030 [P] Implement MLflow custom evaluators for SC-002 through SC-006 in `src/valravn/evaluation/evaluators.py`: log pass/fail metrics per run to MLflow experiment `valravn-evaluation`; expose `--suite` CLI flag via `python -m valravn.evaluation.evaluators`
- [ ] T031 [P] Implement golden dataset tooling in `src/valravn/evaluation/datasets.py`: `--add <report.json>` appends a verified findings report to `tests/evaluation/datasets/golden.jsonl`; used by evaluators as ground truth
- [ ] T032 [P] Unit tests for retry logic and failure escalation in `tests/unit/test_tool_runner.py` (mock `subprocess.run`)
- [ ] T033 [P] Unit tests for skill loader path resolution and error handling in `tests/unit/test_skill_loader.py`
- [ ] T034 [P] Unit tests for anomaly recording and JSON persistence in `tests/unit/test_anomaly.py`
- [ ] T035 [P] Unit tests for report generation: UTC enforcement, citation validation, Markdown structure in `tests/unit/test_report.py`
- [ ] T036 Run `quickstart.md` end-to-end validation: follow setup steps on a clean `.venv`, run agent against synthetic fixture, start MLflow server and verify run appears at `http://127.0.0.1:5000`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Phase 1 — **blocks all user stories**
- **US1 (Phase 3)**: Depends on Phase 2 — no dependency on US2 or US3
- **US2 (Phase 4)**: Depends on Phase 2 and Phase 3 (extends `run_forensic_tool` and `write_findings_report`)
- **US3 (Phase 5)**: Depends on Phase 2 and Phase 3 (extends `run_forensic_tool` and `write_findings_report`)
- **Polish (Phase 6)**: Depends on all user story phases complete

### User Story Dependencies

- **US1 (P1)**: Starts immediately after Foundational — no story dependencies
- **US2 (P2)**: Extends `run_forensic_tool` and `graph.py` edges from US1 — start after US1 checkpoint
- **US3 (P3)**: Extends `run_forensic_tool` retry logic from US1 — start after US1 checkpoint; can run in parallel with US2 (different methods within `tool_runner.py`, different graph edges)

### Within Each User Story

- Models/state before nodes
- Nodes before graph edges
- Graph edges before integration test
- Fixtures can be created in parallel with node implementation

### Parallel Opportunities

- T005–T008 (all data models) — fully parallel, different files
- T013 and T014 (plan node + fixture creation) — parallel within US1
- T020 and T021 (fixture + check_anomalies node) — parallel within US2
- T030–T035 (all Polish tasks) — fully parallel, different files

---

## Parallel Example: Phase 2 (Foundational)

```
Launch simultaneously:
  T005 — src/valravn/config.py
  T006 — src/valravn/models/task.py
  T007 — src/valravn/models/records.py
  T008 — src/valravn/models/report.py
  T010 — src/valravn/nodes/skill_loader.py

Then sequentially:
  T009 — src/valravn/state.py  (needs T006–T008)
  T011 — src/valravn/cli.py    (needs T009)
  T012 — src/valravn/graph.py  (needs T009–T011)
```

## Parallel Example: User Story 1

```
Launch simultaneously:
  T013 — src/valravn/nodes/plan.py (plan_investigation)
  T014 — tests/fixtures/evidence/  (synthetic fixture)

Then sequentially:
  T015 — src/valravn/nodes/tool_runner.py
  T016 — src/valravn/nodes/plan.py (update_plan)
  T017 — src/valravn/nodes/report.py
  T018 — src/valravn/graph.py (wire edges)
  T019 — tests/integration/test_graph.py
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational — **STOP: verify CLI validation works**
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: run independent test for US1
5. Findings report exists, cites tool output, no evidence modified → MVP ready

### Incremental Delivery

1. Setup + Foundational → foundation ready
2. US1 complete → agent can investigate and report (MVP)
3. US2 complete → agent recognises contradictions and escalates
4. US3 complete → agent self-corrects on tool failure
5. Polish → evaluation infrastructure, unit coverage, quickstart validated

### Parallel Team Strategy

After Phase 2 checkpoint:
- **Developer A**: US2 (anomaly detection)
- **Developer B**: US3 (retry/self-correction)

US2 and US3 extend different methods in `tool_runner.py` and add different conditional
edges in `graph.py` — coordinate on those two files to avoid conflicts.

---

## Notes

- [P] tasks touch different files with no shared state — safe to run in parallel
- Evidence fixtures in `tests/fixtures/evidence/` must never be an output target
- All output paths must resolve under `<output_dir>/analysis/`, `/exports/`, or `/reports/`
- Constitution gates (evidence integrity, UTC, skill routing) are enforced at the model
  validation layer (Pydantic) and graph edge layer (conditional edges) — not ad-hoc checks
- Commit after each story checkpoint
