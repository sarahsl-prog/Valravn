# Valravn — Code Analysis Report

**Date:** 2026-04-11  
**Scope:** `src/valravn/`, `tests/`, `docs/`, `CLAUDE.md`, `README.md`  
**Method:** Full static analysis across all source, test, and documentation files

---

## Executive Summary

| Category | Issues Found | High | Medium | Low |
|----------|-------------|------|--------|-----|
| Logic & Code Errors | 11 | 3 | 4 | 4 |
| Test Gaps | 10 | 2 | 5 | 3 |
| Documentation Mismatches | 8 | 2 | 3 | 3 |
| **Total** | **29** | **7** | **12** | **10** |

---

## Section A: Logic & Code Errors

---

### A-01 — `mutator.py:181` — LLM response assigned directly to `MutationSpec` without parsing

**Severity:** HIGH — runtime `AttributeError` on every call to `apply_mutation()`

**File:** `src/valravn/training/mutator.py`, line 181

**Code:**
```python
spec: MutationSpec = _get_mutator_llm().invoke(messages)
```

`_get_mutator_llm().invoke(messages)` returns a LangChain `AIMessage` object, not a `MutationSpec`. The type annotation is not enforced at runtime. Line 187 immediately calls `_check_mutation_safety(playbook, spec)`, which accesses `spec.operation` — an attribute that `AIMessage` does not have. This crashes every time `apply_mutation()` is called.

The docstring on `_get_mutator_llm()` (line 148–152) even acknowledges this: *"Note: Pydantic validation happens on MutationSpec instantiation which is handled separately from LLM structured output"* — but the separate handling was never implemented.

**Proposed fix:**
```python
try:
    response = _get_mutator_llm().invoke(messages)
    spec = parse_llm_json(response.content, MutationSpec)
except Exception as e:
    logger.error("LLM invocation failed during mutation: {}", e)
    raise InvalidMutationError(f"LLM failed to produce valid mutation spec: {e}") from e
```

Add `from valravn.core.parsing import parse_llm_json` to the imports.

---

### A-02 — `anomaly.py:158` — `INVESTIGATION_HALT` response action is silently ignored

**Severity:** HIGH — investigation continues when the LLM says it cannot proceed

**Files:** `src/valravn/nodes/anomaly.py` line 158, `src/valravn/models/records.py`

The anomaly detection prompt (line 32) instructs the LLM to return one of three `response_action` values, including `"investigation_cannot_proceed"`. `AnomalyResponseAction.INVESTIGATION_HALT` is defined in `records.py` and is correctly parsed from the LLM response (lines 129–133). However, `record_anomaly()` only acts on `ADDED_FOLLOW_UP` (line 158) and treats everything else as "no action needed". An `INVESTIGATION_HALT` verdict is stored in the anomaly record but never surfaces to the graph — the investigation loop continues regardless.

**Code:**
```python
# record_anomaly(), line 157-158
follow_up_steps: list[PlannedStep] = []
if anomaly.response_action == AnomalyResponseAction.ADDED_FOLLOW_UP:
    # ... build follow-up step
    # INVESTIGATION_HALT is never checked here
```

There is also no conditional edge in `graph.py` that routes to an early `write_findings_report` when the investigation should halt.

**Proposed fix:** Add an `_investigation_halted` boolean field to `AgentState`, set it to `True` in `record_anomaly()` when `response_action == INVESTIGATION_HALT`, and add a routing check in `update_plan` (or a new `route_after_record_anomaly`) that routes to `synthesize_conclusions` when `_investigation_halted` is `True`.

---

### A-03 — `anomaly.py` `_FOLLOW_UP_COMMANDS` — invalid tool commands for follow-up steps

**Severity:** HIGH — follow-up steps will crash immediately upon execution

**File:** `src/valravn/nodes/anomaly.py`, lines 46–66

Two entries in `_FOLLOW_UP_COMMANDS` contain invalid tool invocations:

**`timestamp_contradiction`** (line 48):
```python
"tool_cmd_template": ["log2timeline.py", "--parsers", "mft,usnjrnl", "{evidence}"],
```
Missing the required `--storage-file <output.plaso>` argument. Per the system prompt in `nodes/plan.py` (line 43), `log2timeline.py` requires `--storage-file` and flags must come *before* the source path. This command will fail immediately with a usage error.

**`orphaned_relationship`** (line 52):
```python
"tool_cmd_template": ["vol3", "-f", "{evidence}", "pstree"],
```
The executable `vol3` does not exist on SIFT. The correct invocation, as documented in both `nodes/plan.py` (line 56) and `nodes/tool_runner.py` (line 43), is `python3 /opt/volatility3-2.20.0/vol.py`. Additionally, `pstree` is not a valid Volatility 3 plugin name; the correct plugin is `windows.pstree.PsTree`.

**Proposed fix:**
```python
"timestamp_contradiction": {
    "skill_domain": "plaso-timeline",
    "tool_cmd_template": [
        "log2timeline.py", "--storage-file", "{analysis_dir}/followup_timeline.plaso",
        "--parsers", "mft,usnjrnl", "--timezone", "UTC", "{evidence}"
    ],
},
"orphaned_relationship": {
    "skill_domain": "memory-analysis",
    "tool_cmd_template": [
        "python3", "/opt/volatility3-2.20.0/vol.py",
        "-f", "{evidence}", "windows.pstree.PsTree"
    ],
},
```
The template substitution logic will also need to handle `{analysis_dir}` in addition to `{evidence}` (currently only `{evidence}` is substituted, lines 176–179).

---

### A-04 — `skill_loader.py:25` — unguarded `next()` raises `StopIteration` if step ID is not found

**Severity:** MEDIUM — crashes the graph with an unhelpful error

**File:** `src/valravn/nodes/skill_loader.py`, line 25

```python
step = next(s for s in state["plan"].steps if s.id == step_id)
```

If `step_id` is present but not in `plan.steps` (e.g., due to state corruption or a step being removed), this raises an unhandled `StopIteration`, which propagates as a bare exception through LangGraph with no context about which step was missing.

**Proposed fix:**
```python
step = next((s for s in state["plan"].steps if s.id == step_id), None)
if step is None:
    raise ValueError(f"Step {step_id!r} not found in plan (steps: {[s.id for s in state['plan'].steps]})")
```

---

### A-05 — `replay_buffer.py:87,104` — `archived_count` is incremented but never persisted

**Severity:** MEDIUM — metric counter silently resets to 0 on every restart

**File:** `src/valravn/training/replay_buffer.py`

`self.archived_count` is incremented in `_archive_entry()` (line 87) to track how many cases have been rejected to the archive. However, `save()` (lines 104–113) does not include `archived_count` in the payload, and `load()` (lines 115–127) does not restore it. After any save/load cycle, the counter restarts from 0.

**Proposed fix:** Add `"archived_count": self.archived_count` to the `payload` dict in `save()`, and add `instance.archived_count = payload.get("archived_count", 0)` in `load()`.

---

### A-06 — `feasibility.py:116` — string prefix matching on paths is unreliable

**Severity:** MEDIUM — evidence protection check has false positives and false negatives

**File:** `src/valravn/training/feasibility.py`, line 116

```python
if ev_path and arg.startswith(ev_path):
```

This string prefix check has two failure modes:

1. **False positive:** Evidence at `/mnt/evidence` would block commands targeting `/mnt/evidence2/file` because `/mnt/evidence2/file`.startswith(`/mnt/evidence`) is `True`.
2. **False negative:** Evidence at `/mnt/evidence/` (with trailing slash) would not match an argument of `/mnt/evidence` (without slash).

**Proposed fix:** Use `Path.is_relative_to()` (available in Python 3.9+, and the project requires 3.12+):
```python
try:
    if ev_path and Path(arg).resolve().is_relative_to(Path(ev_path).resolve()):
        return False, f"Destructive command '{executable}' targets evidence path {ev_path}"
except (ValueError, OSError):
    pass  # resolve() can fail on non-existent paths; skip check
```

---

### A-07 — `evaluators.py:53` — `_eval_evidence_integrity` docstring claims "modified" but code only checks permissions

**Severity:** MEDIUM — evaluator passes reports where evidence was modified then made read-only

**File:** `src/valravn/evaluation/evaluators.py`, lines 52–66

**Docstring:** `"SC-005: no evidence file was modified — each path must exist and be read-only."`

**Code behavior:** The evaluator checks `stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH` bits only. A scenario where evidence is modified (mtime/content changed) and then `chmod`'d back to read-only will pass SC-005. The evaluator tests the *current permission state*, not whether the file was changed during the investigation run.

**Proposed fix (two options):**

*Option 1 — Fix the docstring to match the code (minimal change):*
```python
"""SC-005: evidence files are unmodified — each path must exist and must not be writable."""
```

*Option 2 — Fix the code to match the docstring (stronger guarantee):*  
Record SHA-256 hashes of evidence files at investigation start (in `InvestigationTask` or `FindingsReport`) and compare them in the evaluator. This is a more significant change to the data model.

---

### A-08 — `tool_runner.py:259,289` — `PlannedStep` mutated in-place during retry loop

**Severity:** LOW — fragile shared-state pattern; correctness depends on mutation order

**File:** `src/valravn/nodes/tool_runner.py`, lines 259, 289

```python
step.invocation_ids.append(rec.id)   # line 259 — mutates list in-place
step.tool_cmd = correction.corrected_cmd  # line 289 — overwrites original command
```

The `PlannedStep` object in `plan.steps` is mutated directly. This means the corrected `tool_cmd` replaces the original for the lifetime of the run, and all retry attempts after the first correction use the LLM-corrected command. While intentional, this means `investigation_plan.json` on disk reflects corrected commands (written by `_persist_plan` in `update_plan`) rather than the original planned commands — which may be confusing during post-incident review.

**Proposed fix:** If audit fidelity is required, store original and corrected commands separately. At minimum, document the in-place mutation behavior in the node docstring.

---

### A-09 — `graph.py:62` — `state.get("_pending_anomalies")` without explicit default

**Severity:** LOW — works by accident; intent is unclear

**File:** `src/valravn/graph.py`, line 62

```python
if state.get("_pending_anomalies"):
```

If `_pending_anomalies` is not in `state`, `.get()` returns `None`, which is falsy. This works correctly in practice because `_pending_anomalies` is always initialized in `initial_state`. However, the intent is not explicit, and the pattern is inconsistent with how other nodes access this field (e.g., `state.get("_pending_anomalies", False)`).

**Proposed fix:**
```python
if state.get("_pending_anomalies", False):
```

---

### A-10 — `config.py:~75` — YAML model config silently ignored when env var is already set

**Severity:** LOW — user has no feedback when YAML override is skipped

**File:** `src/valravn/config.py`, approximately line 75

When `load_config()` populates env vars from YAML, it skips any module whose env var is already set (the `if env_key not in os.environ:` guard). This is correct behavior to allow env vars to override config files, but there is no log message when a skip occurs. A user who edits `config.yaml` to change the model but has a stale env var set will see no indication of why the YAML change has no effect.

**Proposed fix:**
```python
if env_key in os.environ:
    logger.info(
        "Config: {} already set via env ({}); ignoring YAML value {}",
        env_key, os.environ[env_key], provider_model
    )
else:
    os.environ[env_key] = ...
```

---

### A-11 — `anomaly.py:48` — `_FOLLOW_UP_COMMANDS` template substitution only handles `{evidence}`

**Severity:** LOW — cascades from A-03; related but distinct

**File:** `src/valravn/nodes/anomaly.py`, lines 176–179

```python
tool_cmd = [
    evidence_path if part == "{evidence}" else part
    for part in cmd_spec["tool_cmd_template"]
]
```

The substitution logic only replaces the literal string `"{evidence}"`. If A-03 is fixed by adding `{analysis_dir}` to the `log2timeline.py` template, this loop must be extended to also substitute `analysis_dir`. The current logic would leave `"{analysis_dir}"` as a literal string argument, causing an immediate tool failure.

**Proposed fix (aligned with A-03 fix):**
```python
analysis_dir = str(Path(state.get("_output_dir", ".")) / "analysis")
tool_cmd = [
    evidence_path if part == "{evidence}" else
    analysis_dir if part == "{analysis_dir}" else
    part
    for part in cmd_spec["tool_cmd_template"]
]
```

---

## Section B: Test Gaps

---

### B-01 — No test coverage for `core/llm_factory.py`

**Severity:** HIGH — the model fallback chain logic is completely untested

**File:** `src/valravn/core/llm_factory.py` — no corresponding test file

`_resolve_model_chain()` implements the priority logic (explicit argument → env var → DEFAULT_MODELS) and `get_llm()` builds the fallback chain. Neither is tested. Failure modes that go untested include:
- Module with no configured model raises `ValueError`
- Comma-separated env var produces correct chain order
- Fallback skips unavailable providers without crashing

**Proposed fix:** Create `tests/unit/test_llm_factory.py` with at minimum:
- `test_resolve_explicit_model` — explicit `provider_model` ignores env and defaults
- `test_resolve_env_var_chain` — comma-separated env var produces ordered list
- `test_resolve_default_model` — falls back to `DEFAULT_MODELS`
- `test_resolve_unknown_module_raises` — raises `ValueError`
- `test_get_llm_single_model` — returns a single model when chain has one entry
- `test_get_llm_skips_unavailable_fallback` — chain construction tolerates `ImportError`

---

### B-02 — No test coverage for `core/parsing.py`

**Severity:** HIGH — JSON extraction is a critical path used by every LLM node

**File:** `src/valravn/core/parsing.py` — no corresponding test file

`parse_llm_json()` is called by every node that uses an LLM. It handles both raw JSON and markdown-fenced JSON (` ```json ... ``` `). Untested scenarios include:
- Markdown fence stripping
- Invalid JSON raises a useful error
- LLM response with trailing prose outside the JSON object
- Pydantic validation failure on the parsed object

**Proposed fix:** Create `tests/unit/test_parsing.py` with tests for raw JSON, fenced JSON, invalid JSON, and validation failures.

---

### B-03 — No test coverage for `nodes/self_assess.py`

**Severity:** MEDIUM — assessment node and its fallback parsing logic are untested

**File:** `src/valravn/nodes/self_assess.py` — no corresponding test file

The `assess_progress` node runs before every tool execution and influences `_self_assessments`. It has a non-trivial fallback parsing path for models that don't comply with the structured output schema. None of this is tested.

**Proposed fix:** Create `tests/unit/test_self_assess.py` covering happy path, LLM failure (returns empty assessments), and the fallback parsing path.

---

### B-04 — `test_anomaly.py` and `test_anomaly_fix.py` have overlapping scenarios; `INVESTIGATION_HALT` is never tested

**Severity:** MEDIUM — duplicated effort and the most critical anomaly path is uncovered

**Files:** `tests/unit/test_anomaly.py`, `tests/unit/test_anomaly_fix.py`

Both files test `check_anomalies()` and `record_anomaly()` with overlapping happy-path scenarios. Neither file tests the `INVESTIGATION_HALT` response action (finding A-02) — the bug that allows the investigation to continue when the LLM says it cannot. Without a test, fixing A-02 has no regression protection.

**Proposed fix:**
1. Consolidate the two files into a single `test_anomaly.py` organized by function.
2. Add `test_record_anomaly_investigation_halt_sets_flag` to verify that when `response_action == "investigation_cannot_proceed"`, the returned state includes `_investigation_halted: True`.

---

### B-05 — `test_evaluators.py` does not expose the SC-005 semantic mismatch (finding A-07)

**Severity:** MEDIUM — the evaluator can be broken without any test failing

**File:** `tests/unit/test_evaluators.py`

The existing SC-005 test creates a read-only evidence file, which passes. It does not test the case where evidence was modified (mtime changed, content changed) but permissions were restored to read-only afterward. If A-07 is fixed to use hash comparison, the existing test also needs to be extended to verify the new behavior.

**Proposed fix:** Add a test that:
1. Creates evidence, records its hash in the report.
2. Modifies the file content.
3. Restores read-only permissions.
4. Calls `_eval_evidence_integrity()` — should fail.

---

### B-06 — `test_conclusions.py` missing edge cases for failed tool invocations and stdout boundary

**Severity:** MEDIUM — truncation boundary and failure handling are not validated precisely

**File:** `tests/unit/test_conclusions.py`

Two gaps:

1. **Truncation boundary:** The test asserts `human_content.count("A") <= 10_000` but not that it equals exactly `MAX_STDOUT_CHARS`. The evaluator could truncate to 1 byte and the test would still pass.

2. **Failed invocations:** There is no test for the case where `invocation.exit_code != 0`. It is unclear whether `synthesize_conclusions` should include or exclude the output of failed tool runs when building the LLM context.

**Proposed fix:**
- Change the truncation assertion to `== MAX_STDOUT_CHARS` (or whatever the constant is) after examining the actual constant value.
- Add a test where one invocation has `exit_code != 0` and verify whether its stdout is included or excluded in the LLM message.

---

### B-07 — `test_plan_node.py` does not verify that `investigation_plan.json` is written to disk

**Severity:** MEDIUM — file persistence side-effect of `plan_investigation` is untested

**File:** `tests/unit/test_plan_node.py`

The test asserts that `plan_investigation()` returns the correct `plan` and `current_step_id` in the state dict, but does not verify that `_persist_plan()` actually wrote `analysis/investigation_plan.json`. If `_persist_plan` is broken (e.g., wrong path, permission error), no test catches it.

**Proposed fix:** After calling `plan_investigation()`, assert:
```python
plan_file = tmp_path / "analysis" / "investigation_plan.json"
assert plan_file.exists()
data = json.loads(plan_file.read_text())
assert data["task_id"] == task.id
```

---

### B-08 — `test_rcl_loop.py` uses a `None`-returning mock for `reflect_on_trajectory`

**Severity:** LOW — the reflector's actual diagnostic output never flows through the loop in tests

**File:** `tests/unit/test_rcl_loop.py`

The mock for `reflect_on_trajectory` is configured to return `None` (or an empty mock). This means the test exercises the loop structure but not the data flow from reflection output to mutation input. The `apply_mutation()` function (which has A-01) would never be exercised with actual diagnostic text.

**Proposed fix:** Make the mock return a realistic `ReflectorOutput`-like object with a non-empty `diagnostic` string, so the data flow from reflection to mutation is tested end-to-end.

---

### B-09 — Missing shared fixtures in `conftest.py`

**Severity:** LOW — duplicated setup code across many test files

**File:** `tests/conftest.py` (24 lines; only defines `read_only_evidence` and `output_dir`)

The following fixture patterns are duplicated across 3 or more test files and should be promoted to `conftest.py`:

| Pattern | Duplicated in |
|---------|--------------|
| `mock_llm` (MagicMock with `invoke` returning `AIMessage`) | `test_anomaly.py`, `test_conclusions.py`, `test_plan_node.py`, `test_tool_runner.py` |
| `base_state` (minimal `AgentState` dict) | `test_anomaly.py`, `test_plan_node.py`, `test_tool_runner.py`, `test_conclusions.py` |
| `tool_invocation_record` fixture | `test_anomaly.py`, `test_conclusions.py` |

---

### B-10 — Integration tests missing multi-step routing and anomaly follow-up failure scenarios

**Severity:** LOW — full graph loop with >1 step has no integration coverage

**File:** `tests/integration/test_graph.py`

The existing integration test exercises a single-step plan. The following are not covered:

1. **Multi-step routing:** A plan with 2–3 steps to verify that `update_plan → load_skill` cycling works end-to-end.
2. **Anomaly-triggered follow-up that itself fails:** Confirms that the depth cap (3 follow-up steps) and the `EXHAUSTED` status path work together.
3. **Empty evidence file (zero bytes):** Verifies that `check_anomalies` handles `tool_output = ""` gracefully.

---

## Section C: Documentation Mismatches

---

### C-01 — `docs/architecture.md` — graph diagram omits `assess_progress` and `synthesize_conclusions`

**Severity:** HIGH — the published architecture does not match the running code

**File:** `docs/architecture.md`, lines 7–50

The ASCII diagram and Mermaid state diagram both show the flow as:

```
load_skill → run_forensic_tool → ... → update_plan → write_findings_report
```

The actual graph wiring in `graph.py` (lines 88–97) is:

```
load_skill → assess_progress → run_forensic_tool → ... → update_plan → synthesize_conclusions → write_findings_report
```

Two nodes are entirely absent from the documentation:

- **`assess_progress`** (runs between `load_skill` and `run_forensic_tool`) — evaluates the current step against accumulated state and populates `_self_assessments`
- **`synthesize_conclusions`** (runs between `update_plan`'s terminal exit and `write_findings_report`) — calls the LLM to derive forensic conclusions from all tool outputs

The State section (lines 162–175) also does not document `_self_assessments` or `_conclusions` fields that are used by these nodes.

**Proposed fix:** Update both diagrams and add node description sections for `assess_progress` and `synthesize_conclusions`. Add `_self_assessments` and `_conclusions` to the State table.

---

### C-02 — `docs/architecture.md:60` — `plan_investigation` described as calling Claude `claude-opus-4-6`

**Severity:** HIGH — contradicts the actual default model configuration

**File:** `docs/architecture.md`, line 60

> "Calls Claude (`claude-opus-4-6`) with a structured-output prompt..."

The actual code in `nodes/plan.py` calls `get_llm(module="plan")`. The `DEFAULT_MODELS` in `llm_factory.py` (line 29) resolves this to `"ollama:minimax-m2.5:cloud"`, not Claude. Claude is only used if the user explicitly overrides the model via `config.yaml` or the `VALRAVN_PLAN_MODEL` environment variable.

This mismatch (also present in C-05 below) means a user following the architecture document will have incorrect expectations about which model is running and what API key is required.

**Proposed fix:** Change the description to: *"Calls the configured LLM (default: `ollama:minimax-m2.5:cloud`; override via `VALRAVN_PLAN_MODEL` or `config.yaml`) with a structured-output prompt..."*

---

### C-03 — `docs/architecture.md:90` — `run_forensic_tool` described as using `subprocess.run`

**Severity:** MEDIUM — incorrect API used; streaming behavior not documented

**File:** `docs/architecture.md`, line 90

> "Executes `step.tool_cmd` via `subprocess.run` with a 1-hour timeout."

The actual implementation in `tool_runner.py` uses `subprocess.Popen` (via `_run_with_streaming()`) with real-time `os.read()` streaming of stderr to the terminal. Using `subprocess.run` would buffer all stderr until the process exits, preventing progress bars and live status output from `log2timeline.py` from being visible — which was explicitly addressed in a recent commit.

**Proposed fix:** Update to: *"Executes `step.tool_cmd` via `subprocess.Popen` with real-time stderr streaming to the terminal (for live progress from tools like `log2timeline.py`) and a configurable timeout."*

---

### C-04 — `docs/architecture.md:129` — `record_anomaly` described as always using `strings -n 20` fallback

**Severity:** MEDIUM — the actual follow-up logic is more sophisticated than documented

**File:** `docs/architecture.md`, line 129

> "...queues a follow-up `PlannedStep` (using `strings -n 20 <evidence>` as a safe default)."

The actual code uses a category-specific `_FOLLOW_UP_COMMANDS` lookup (lines 45–66 of `anomaly.py`) that dispatches to `log2timeline.py`, `vol.py`, `fls`, `yara`, or `img_stat` depending on the detected anomaly category. `strings -n 20` is only used as a last-resort fallback for unrecognized categories.

**Proposed fix:** Update the `record_anomaly` section to describe the category-based dispatch and list the five category-to-tool mappings.

---

### C-05 — `DEFAULT_MODELS` in `llm_factory.py` defaults to Ollama, but `README.md` and `CLAUDE.md` say Anthropic is the default

**Severity:** MEDIUM — users following setup docs will encounter unexpected behavior

**Files:** `src/valravn/core/llm_factory.py` lines 26–34, `CLAUDE.md` line 7, `README.md`

`DEFAULT_MODELS` sets every module to `"ollama:minimax-m2.5:cloud"`. `CLAUDE.md` lists only `langchain-anthropic` under Active Technologies and implies Anthropic/Claude is the runtime model. The README install instructions lead with `pip install -e ".[anthropic]"` as the primary path.

A fresh install following the docs will:
1. Install the Anthropic extra.
2. Set `ANTHROPIC_API_KEY`.
3. Run an investigation — and unexpectedly attempt to connect to a local Ollama instance on `localhost:11434`, not the Anthropic API.

**Proposed fix (two options):**

*Option 1:* Change `DEFAULT_MODELS` to use Anthropic Claude:
```python
DEFAULT_MODELS: dict[str, list[str]] = {
    "anomaly": ["anthropic:claude-haiku-4-5-20251001"],
    "conclusions": ["anthropic:claude-sonnet-4-6"],
    "plan": ["anthropic:claude-sonnet-4-6"],
    # ...
}
```

*Option 2:* Update `CLAUDE.md` and `README.md` to clarify that Ollama is the built-in default and document the steps to switch to Anthropic.

---

### C-06 — `CLAUDE.md` line 7 — states Python 3.12 but `.venv` is Python 3.13

**Severity:** LOW — documentation drift; not functionally breaking

**File:** `CLAUDE.md`, line 7

> "Python 3.12 (`.venv` already initialised in repo root)"

The `.venv` is Python 3.13. `pyproject.toml` correctly declares `requires-python = ">=3.12"`, so any version 3.12+ is intentionally supported.

**Proposed fix:** Change to `"Python 3.12+ (`.venv` already initialised in repo root)"` to reflect the flexible minimum requirement.

---

### C-07 — `CLAUDE.md` env vars section is incomplete

**Severity:** LOW — contributors and operators missing critical configuration options

**File:** `CLAUDE.md`, lines 34–39

The documented environment variables are:
```
ANTHROPIC_API_KEY       # required — Claude model via langchain-anthropic
VALRAVN_MAX_RETRIES     # optional override for retry.max_attempts (default: 3)
```

Variables present in the codebase but absent from this section:

| Variable | Used in | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | `llm_factory.py:174` | OpenAI model provider |
| `OLLAMA_BASE_URL` | `llm_factory.py:195` | Custom Ollama endpoint (default: `localhost:11434`) |
| `OPENROUTER_API_KEY` | `llm_factory.py:210` | OpenRouter provider |
| `OPENROUTER_BASE_URL` | — | Custom OpenRouter endpoint |
| `VALRAVN_{MODULE}_MODEL` | `llm_factory.py:52` | Per-module model override (e.g., `VALRAVN_PLAN_MODEL`) |
| `MLFLOW_TRACKING_URI` | `evaluators.py:15` | MLflow server address (default: `http://127.0.0.1:5000`) |

**Proposed fix:** Add the missing variables to the env vars section in `CLAUDE.md`.

---

### C-08 — `README.md` project structure lists `feasibility.py` twice with different descriptions

**Severity:** LOW — documentation noise

**File:** `README.md`, project structure section

`feasibility.py` appears in the training module listing with two separate entries:
- *"Feasibility rules registry"*
- *"Custom feasibility rules for replay buffer filtering"*

Both describe the same file. The file serves dual purposes (command safety validation via `FeasibilityMemory` and replay buffer case filtering via `register_feasibility_rule`), which is documented in the module docstring but represented as two separate entries in the README.

**Proposed fix:** Merge into a single entry: `"feasibility.py — command safety validation (FeasibilityMemory) and replay buffer feasibility rules registry"`.

---

## Appendix: Issue Index

| ID | File | Severity | One-line summary |
|----|------|----------|-----------------|
| A-01 | `training/mutator.py:181` | HIGH | LLM response not parsed; `AttributeError` on every call |
| A-02 | `nodes/anomaly.py:158` | HIGH | `INVESTIGATION_HALT` silently ignored; loop continues |
| A-03 | `nodes/anomaly.py:48,52` | HIGH | Follow-up commands for `timestamp_contradiction` and `orphaned_relationship` are invalid |
| A-04 | `nodes/skill_loader.py:25` | MEDIUM | Unguarded `next()` raises `StopIteration` on missing step |
| A-05 | `training/replay_buffer.py:87,104` | MEDIUM | `archived_count` not persisted in `save()`/`load()` |
| A-06 | `training/feasibility.py:116` | MEDIUM | `startswith()` path matching has false positives and false negatives |
| A-07 | `evaluation/evaluators.py:53` | MEDIUM | SC-005 checks writable bits, not whether evidence was modified |
| A-08 | `nodes/tool_runner.py:259,289` | LOW | `PlannedStep` mutated in-place during retry; no copy-on-write |
| A-09 | `graph.py:62` | LOW | `state.get("_pending_anomalies")` has no explicit default |
| A-10 | `config.py:~75` | LOW | Silent YAML skip when env var already set |
| A-11 | `nodes/anomaly.py:176` | LOW | Template substitution only handles `{evidence}`; cascades from A-03 |
| B-01 | `core/llm_factory.py` | HIGH | No tests for model chain resolution or fallback logic |
| B-02 | `core/parsing.py` | HIGH | No tests for JSON extraction (used by every LLM node) |
| B-03 | `nodes/self_assess.py` | MEDIUM | No tests for assessment node |
| B-04 | `test_anomaly.py` / `test_anomaly_fix.py` | MEDIUM | Duplicate scenarios; `INVESTIGATION_HALT` never tested |
| B-05 | `test_evaluators.py` | MEDIUM | SC-005 test doesn't cover modified-then-made-read-only evidence |
| B-06 | `test_conclusions.py` | MEDIUM | Truncation boundary imprecise; no test for failed invocations |
| B-07 | `test_plan_node.py` | MEDIUM | `investigation_plan.json` write not verified |
| B-08 | `test_rcl_loop.py` | LOW | Reflector mock returns `None`; no data flow through loop |
| B-09 | `tests/conftest.py` | LOW | Three fixture patterns duplicated across test files |
| B-10 | `tests/integration/test_graph.py` | LOW | Single-step only; multi-step routing untested |
| C-01 | `docs/architecture.md:7-50` | HIGH | Diagram missing `assess_progress` and `synthesize_conclusions` |
| C-02 | `docs/architecture.md:60` | HIGH | Claims Claude `claude-opus-4-6` is used; actual default is Ollama |
| C-03 | `docs/architecture.md:90` | MEDIUM | Says `subprocess.run`; actual code uses `subprocess.Popen` with streaming |
| C-04 | `docs/architecture.md:129` | MEDIUM | Describes `strings` fallback only; category-specific commands undocumented |
| C-05 | `llm_factory.py:26-34` vs `README.md`/`CLAUDE.md` | MEDIUM | Code defaults to Ollama; docs say Anthropic is the default |
| C-06 | `CLAUDE.md:7` | LOW | Says Python 3.12; `.venv` is 3.13 |
| C-07 | `CLAUDE.md:34-39` | LOW | Six env vars present in code are absent from docs |
| C-08 | `README.md` | LOW | `feasibility.py` listed twice in project structure |
