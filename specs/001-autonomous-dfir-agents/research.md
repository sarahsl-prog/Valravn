# Phase 0 Research: Autonomous DFIR Agents

**Branch**: `001-autonomous-dfir-agents` | **Date**: 2026-04-05

All NEEDS CLARIFICATION items from the plan's Technical Context are resolved here.
Each section follows: Decision → Rationale → Alternatives Considered.

---

## R-001: Agent Framework Selection

**Decision**: LangGraph (langgraph + langchain-anthropic) for the agent graph, with
LangSmith for run tracing, dataset curation, and evaluation/improvement cycles.

**Rationale**:
- **LangGraph** models the investigation cycle naturally as a stateful graph: nodes for
  skill loading, tool execution, anomaly checking, and report generation; edges that
  branch on anomaly detection or retry exhaustion. The `InvestigationPlan` maps directly
  to LangGraph's `StateGraph` state, which is persisted automatically via LangGraph
  checkpointers — no custom JSON persistence needed.
- **Deep agent support**: LangGraph's `create_react_agent` and `langgraph.prebuilt`
  patterns give interrupt/resume, human-in-the-loop hooks (useful for future escalation
  flows), and streaming out of the box. The graph's explicit node/edge structure makes
  constitution constraint enforcement (e.g., block `run_forensic_tool` unless `load_skill`
  node has fired for that domain) straightforward as conditional edges.
- **MLflow** (local server, `mlflow server --host 127.0.0.1`) handles experiment
  tracking, metric logging, and evaluation — fully self-hosted, no cloud dependency.
  This is a hard requirement: Valravn runs on air-gapped SIFT workstations where
  cloud services cannot be assumed reachable. MLflow stores all runs, metrics, and
  artefacts to a local directory (`./mlruns/` or a configured path).
- **LangGraph file-based tracing** (`langchain_core.tracers.file.FileCallbackHandler`)
  writes structured JSONL traces to `./analysis/traces/` — one file per run. This
  replaces LangSmith's automatic cloud tracing with a local equivalent that is
  compatible with air-gapped operation and remains forensically auditable.
- **MLflow custom metrics** assert the spec's success criteria per run:
  SC-002 (anomaly detection rate), SC-003 (self-correction rate), SC-004 (every
  conclusion cites a tool invocation), SC-005 (no evidence directory modified), SC-006
  (UTC timestamps). Evaluation runs are logged as MLflow experiments under the
  `valravn-evaluation` experiment.
- `langchain-anthropic` wraps the Anthropic SDK so Claude models are still used;
  no model lock-in.

**LangGraph graph topology** (high level):

```
[START]
  → plan_investigation          # LLM derives initial PlannedStep list from prompt
  → load_skill                  # Loads SKILL.md for next step's domain
  → run_forensic_tool           # subprocess invocation with retry loop
  → check_anomalies             # LLM scans tool output; branches on anomaly
    ↘ record_anomaly            # If anomaly: persist + add follow-up steps
  → update_plan                 # Mark step complete/failed; append follow-ups
  → [loop back to load_skill if steps remain]
  → write_findings_report
[END]
```

**Observability + evaluation stack (air-gap safe)**:
- **Tracing**: `FileCallbackHandler` writes JSONL to `./analysis/traces/<run-id>.jsonl`
  — full node/LLM/tool call history, no network required
- **Experiment tracking**: MLflow local server at `http://127.0.0.1:5000`; artefacts
  stored to `./mlruns/` under the case working directory
- **Evaluation datasets**: golden test cases stored as local JSONL files under
  `tests/evaluation/datasets/`; evaluated via `mlflow.evaluate()` with custom metrics
- **No `LANGCHAIN_API_KEY` or `LANGCHAIN_TRACING_V2` required**

**Alternatives Considered**:
- **Raw Anthropic SDK (tool_use loop)**: Full control but requires hand-writing state
  management, checkpointing, streaming, and tracing — all solved problems in LangGraph.
  Would also require building evaluation infrastructure from scratch. Rejected in favour
  of LangGraph.
- **CrewAI**: Multi-agent crew coordination. Out of scope for v1 single-agent design. Rejected.
- **AutoGen**: Microsoft ecosystem; no LangSmith integration; heavier dependency surface.
  Rejected.
- **Direct REST calls**: No tooling benefit. Rejected.

---

## R-002: Skill Routing Mechanism

**Decision**: Implement `load_skill(domain: str) -> str` as an agent-callable tool. The
tool reads the appropriate SKILL.md file from `~/.claude/skills/<domain>/SKILL.md` and
returns its contents to the model. The system prompt explicitly orders the model to call
`load_skill` before the first `run_forensic_tool` call for any given domain in a session.

**Rationale**:
- Constitution Principle III is an architectural constraint, not a documentation note. Making
  skill loading a callable tool means the agent's message history records exactly when each
  skill was consulted — this is auditable.
- Skill files evolve independently. Loading at runtime means the agent always uses the
  current version; embedding skill content in the system prompt would freeze it at invocation
  time.
- The model's context includes the skill content exactly when it needs it (immediately before
  tool execution), which produces better parameter selection than a front-loaded context dump.

**Domain-to-path mapping** (from constitution):

| Domain key | Skill path |
|-----------|-----------|
| `memory-analysis` | `~/.claude/skills/memory-analysis/SKILL.md` |
| `sleuthkit` | `~/.claude/skills/sleuthkit/SKILL.md` |
| `windows-artifacts` | `~/.claude/skills/windows-artifacts/SKILL.md` |
| `plaso-timeline` | `~/.claude/skills/plaso-timeline/SKILL.md` |
| `yara-hunting` | `~/.claude/skills/yara-hunting/SKILL.md` |

**Alternatives Considered**:
- **Embed all skills in system prompt**: Blows up context on every call regardless of which
  domains are needed. Stale on skill updates. Rejected.
- **Pre-load at startup based on evidence type**: Requires evidence-type detection before the
  agent even runs, which is premature. The model is better placed to determine which domains
  are needed after reading the prompt. Rejected.
- **Return skill summary not full content**: Loses the specific invocation conventions and
  caveats that are the whole point of the skill file. Rejected.

---

## R-003: Investigation Plan State Design

**Decision**: `AgentState` (LangGraph `TypedDict`) is the single source of truth during
execution, persisted automatically between every node by LangGraph's `SqliteSaver`
checkpointer to `./analysis/checkpoints.db`. `InvestigationPlan` lives as a field inside
`AgentState`. A human-readable audit copy is also written to
`./analysis/investigation_plan.json` after each plan mutation.

**Rationale**:
- `SqliteSaver` survives process crashes mid-investigation — critical for long-running
  DFIR workflows on a workstation that may be interrupted. `MemorySaver` (the in-memory
  alternative) loses all state on crash, which is unacceptable when a tool invocation
  may have been running for several minutes.
- SQLite here is the LangGraph checkpointer, not a hand-rolled ORM. No schema migrations,
  no application-level query code — LangGraph manages the DB entirely. The only dependency
  is `langgraph-checkpoint-sqlite` (ships with LangGraph).
- The checkpoint DB is local to the case working directory (`./analysis/checkpoints.db`),
  so it is scoped per investigation and never touches evidence directories.
- The plan must be dynamic: steps are added, reordered, or marked skipped as anomalies
  are detected (FR-006). LangGraph state updates are O(1) appends that trigger an
  automatic checkpoint write.
- JSON audit copy to `./analysis/` satisfies FR-004 (human-readable, directly ingestible
  by other analysis tools) without duplicating the checkpointer's role.

**State machine for PlannedStep**:
```
pending → running → completed
                 → failed → (retry) → running
                                    → exhausted
         → skipped  (anomaly response: not needed)
```

**Alternatives Considered**:
- **`MemorySaver` (in-memory checkpointer)**: Zero setup but loses all state on process
  crash. Unacceptable for long-running forensic investigations. Rejected.
- **Hand-rolled SQLite ORM**: More control but adds schema migration burden and
  application-level query code. `SqliteSaver` via LangGraph is the right abstraction —
  it is SQLite under the hood with none of the ORM overhead. Rejected.
- **Agent maintains plan purely in-context (no persistence)**: Loses crash resilience and
  the audit trail required for forensic reproducibility. Rejected.

---

## R-004: Anomaly Detection Approach

**Decision**: LLM-guided anomaly recognition for v1. The agent's system prompt defines
anomaly categories and the model identifies contradictions as it processes tool output.
When an anomaly is identified, the model calls `record_anomaly(description, source_invocation_ids,
response_action)`, which persists the anomaly to `./analysis/anomalies.json` and echoes it
into the findings report.

**Anomaly categories the system prompt trains the model to watch for**:
1. Timestamp contradictions (event before OS install date, future timestamps)
2. Orphaned process relationships (parent PID does not exist or exited before child)
3. Cross-tool conflicts (two tools report different values for the same artifact)
4. Unexpected absences (tool returns empty output where output is architecturally certain)
5. Integrity failures (hash mismatch, truncated output, zero-byte file)

**Rationale**:
- The LLM is better at recognising semantic contradictions (e.g., "this timestamp is before
  the OS was installed") than a rules engine, which would require encoding every possible
  contradiction type as a pattern.
- The `record_anomaly` tool call creates a structured, citeable record. The model cannot
  silently note an anomaly — it must produce a tool call, which is logged.
- For v1 with a 90% detection success criterion (SC-002), LLM-guided detection on known
  injected contradictions is achievable without a dedicated ML pipeline.

**Alternatives Considered**:
- **Rules-based engine (regex, threshold checks)**: Would catch well-defined numeric
  contradictions but misses semantic anomalies. Requires exhaustive rule authoring per
  tool/domain. Deferred to v2 as an augmentation, not replacement. Rejected as sole approach.
- **Separate anomaly-detection model pass**: Doubles LLM calls, increases latency. The main
  investigation model already reads all tool output — a second pass is redundant. Rejected.

---

## R-005: Retry Loop Design

**Decision**: Python retry loop in `tool_runner.py` with a `RetryConfig` dataclass. Default
`max_attempts=3`. On each failure: parse stderr → generate corrective hypothesis (returned
to agent as the failure context) → agent adjusts arguments → retry. The corrective hypothesis
is model-generated (the agent calls `run_forensic_tool` again with modified args after
receiving the failure result).

**Retry escalation path**:
```
Attempt 1: original invocation
  → failure: return {exit_code, stderr, hypothesis_prompt} to agent
Attempt 2: agent-modified invocation (different args or approach)
  → failure: return {exit_code, stderr, hypothesis_prompt} to agent
Attempt 3: agent-modified invocation (alternative tool or sub-command)
  → failure: call record_tool_failure() — escalate to findings report
```

**Configuration** (loaded from `config.yaml` at case directory or repo root, with env
override `VALRAVN_MAX_RETRIES`):
```yaml
retry:
  max_attempts: 3
  retry_delay_seconds: 0
```

**Rationale**:
- The agent (LLM) generates the corrective hypothesis because it has the full investigation
  context: it knows what the tool was trying to find and can infer why the invocation failed.
  A purely mechanical retry (same args) would just fail again.
- `retry_delay_seconds: 0` default is intentional — forensic tools are deterministic; a
  brief delay does not fix argument errors. Delay can be set for tools where a race condition
  exists (e.g., a mount settling).
- Configurable limit satisfies FR-009. Hard-coding would violate the requirement.

**Alternatives Considered**:
- **Exponential backoff**: Appropriate for rate-limited APIs, not for local CLI tools that
  fail for structural reasons (wrong path, wrong format). Rejected as default.
- **Retry in tool_runner without agent involvement**: Would retry with identical args,
  which cannot self-correct parameter errors. Rejected.

---

## R-006: Testing Strategy

**Decision**: Two-tier test suite.

**Unit tests** (`tests/unit/`) — mock subprocess, no SIFT tools required:
- Retry logic: verify attempt count, failure escalation, exit on success
- Skill loader: verify correct path construction, file-not-found handling
- Anomaly recording: verify JSON persistence, report injection
- Report generation: verify UTC timestamps, citation linkage, Markdown structure

**Integration tests** (`tests/integration/`) — real SIFT tools, synthetic evidence:
- `tests/fixtures/evidence/` contains minimal, purpose-built synthetic artifacts:
  - A 1 MB raw disk image with known file structure (Sleuth Kit tests)
  - A minimal Windows event log XML fixture (EZ Tools tests)
  - A tiny synthetic memory sample for Volatility smoke tests (if feasible)
- Fixtures are committed as read-only binary blobs; tests assert on known expected output
- Integration tests are gated behind a pytest marker `@pytest.mark.integration` and are
  skipped in CI environments where SIFT tools are not present

**Evidence integrity in tests**: `tests/fixtures/evidence/` is never an output target. Tests
that write output use `tmp_path` (pytest fixture) or `./analysis/` under a temp case dir.

**Alternatives Considered**:
- **Contract tests against live SIFT tools for every test run**: Slow and environment-
  dependent. Mock at the subprocess boundary for unit tests, real tools only for integration.
- **Test with actual case evidence**: Prohibited by Principle I and chain-of-custody rules.
  Synthetic fixtures only.
