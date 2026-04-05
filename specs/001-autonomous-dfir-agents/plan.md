# Implementation Plan: Autonomous DFIR Agents

**Branch**: `001-autonomous-dfir-agents` | **Date**: 2026-04-05 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-autonomous-dfir-agents/spec.md`

## Summary

Build an autonomous DFIR investigation agent for the SANS SIFT Workstation. The agent accepts
a natural-language investigation prompt and an evidence reference, autonomously sequences
appropriate forensic tool invocations (guided by per-domain skill files), detects anomalies
in tool output, self-corrects on failure, and delivers a fully cited findings report — with no
operator interaction mid-task. Implemented as a Python CLI using LangGraph (stateful agent
graph) with Claude via `langchain-anthropic`; MLflow (local server) for evaluation and
experiment tracking; fully air-gap safe — no cloud services required.

## Technical Context

**Language/Version**: Python 3.12 (`.venv` already initialised in repo root)
**Primary Dependencies**: `langgraph` (agent state graph); `langchain-anthropic` (Claude
model binding); `mlflow` (local experiment tracking + evaluation — air-gap safe); SIFT
native tools (Volatility 3, Sleuth Kit, EZ Tools, Plaso, YARA, bulk_extractor) invoked
via `subprocess`; `pydantic` (data model validation); `pytest` (testing)
**Storage**: File-based only — `./analysis/` (raw tool output + plan state), `./exports/`
(intermediate artefacts), `./reports/` (final findings report)
**Testing**: `pytest` with mock subprocess fixtures for unit tests; real SIFT tools against
synthetic evidence fixtures for integration tests
**Target Platform**: Linux x86-64, SANS SIFT Ubuntu (Python 3.12 system runtime)
**Project Type**: CLI tool (`valravn investigate --prompt "..." --evidence /path/to/evidence`)
**Performance Goals**: Deliver a complete findings report within 10 minutes for a well-scoped
investigation prompt (derived from User Story 1 acceptance criteria)
**Constraints**: Evidence directories are strictly read-only; all output in UTC; zero mid-task
operator prompts; per-tool retry limit configurable, defaulting to 3 attempts (FR-009)
**Scale/Scope**: Single agent per run, single local workstation, one evidence reference per
invocation (v1 scope per Assumptions)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design (see below).*

### Pre-Design Check (2026-04-05)

| Principle | Requirement | Satisfaction |
|-----------|------------|--------------|
| I — Evidence Integrity | No write to `/cases/`, `/mnt/`, `/media/`, `evidence/` | FR-007 prohibits this; subprocess invocations use read-only flags; enforced at invocation layer ✅ |
| II — Deterministic Execution | All conclusions cite raw tool output; no hallucination; failure reported | FR-004, FR-008, FR-013 directly satisfy; tool output persisted before any conclusion is derived ✅ |
| III — Skill-First Routing | Consult skill file before any forensic utility | Implemented as a `load_skill(domain)` tool call that the agent is architecturally required to invoke before the corresponding `run_tool` call; system prompt enforces ordering ✅ |
| IV — Output Discipline | Output only to `./analysis/`, `./exports/`, `./reports/`; all timestamps UTC | FR-004, FR-010, FR-011 satisfy; output paths validated at runtime ✅ |
| V — Autonomous Execution | Zero mid-task operator prompts; document autonomous decisions | FR-012 satisfies; agent loop runs to completion with no human-in-the-loop pause ✅ |

**Result**: PASS — no violations. Proceed to Phase 0.

### Post-Design Re-Check

See bottom of this document (updated after Phase 1 completes).

## Project Structure

### Documentation (this feature)

```text
specs/001-autonomous-dfir-agents/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── cli-contract.md
└── tasks.md             # Phase 2 output (/speckit-tasks — NOT created here)
```

### Source Code (repository root)

```text
src/
├── valravn/
│   ├── __init__.py
│   ├── cli.py               # Entry point: argument parsing, task construction
│   ├── graph.py             # LangGraph StateGraph definition (nodes + edges)
│   ├── state.py             # AgentState TypedDict — shared graph state
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── plan.py          # plan_investigation node
│   │   ├── skill_loader.py  # load_skill node — reads SKILL.md files
│   │   ├── tool_runner.py   # run_forensic_tool node — subprocess + retry
│   │   ├── anomaly.py       # check_anomalies + record_anomaly nodes
│   │   └── report.py        # write_findings_report node
│   ├── models/
│   │   ├── __init__.py
│   │   ├── task.py          # InvestigationTask, InvestigationPlan, PlannedStep
│   │   ├── records.py       # ToolInvocationRecord, Anomaly
│   │   └── report.py        # FindingsReport
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluators.py    # MLflow custom metrics for SC-002–SC-006
│   └── config.py            # RetryConfig, OutputConfig, loaded from config.yaml

tests/
├── unit/
│   ├── test_tool_runner.py  # Retry logic, failure escalation
│   ├── test_skill_loader.py # Skill file loading
│   ├── test_anomaly.py      # Anomaly recording
│   └── test_report.py       # Report generation
├── integration/
│   └── test_graph.py        # End-to-end graph run with synthetic evidence fixture
└── fixtures/
    └── evidence/            # Minimal synthetic evidence files (read-only)
```

**Structure Decision**: Single-project layout (Option 1). No frontend, no external API — all
interaction is CLI. `src/valravn/` keeps the importable package separate from tests and specs.

## Complexity Tracking

> No constitution violations detected. Table left intentionally empty.

---

## Post-Design Constitution Re-Check

*Populated after Phase 1 design completes.*

| Principle | Design Decision | Status |
|-----------|----------------|--------|
| I — Evidence Integrity | `tool_runner.py` validates output paths before every subprocess call; evidence paths are never passed as output targets | ✅ |
| II — Deterministic Execution | Every `ToolInvocationRecord` persists stdout/stderr to `./analysis/` before `graph.py` processes the result; `FileCallbackHandler` traces every node to `./analysis/traces/` | ✅ |
| III — Skill-First Routing | `load_skill` is a required graph node with a conditional edge that blocks `run_forensic_tool` unless `load_skill` has fired for that domain in the current state | ✅ |
| IV — Output Discipline | `OutputConfig` validates all paths at startup; `FindingsReport` enforces UTC via `datetime.timezone.utc` | ✅ |
| V — Autonomous Execution | LangGraph graph runs to `END` node without `interrupt_before`/`interrupt_after` in production mode; no `input()` anywhere in the call stack | ✅ |

**Result**: PASS — design is constitution-compliant.
