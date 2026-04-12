---
name: valravn-dev
description: >
  Orchestrates Valravn DFIR agent development tasks. Use this skill for feature requests,
  bug fixes, adding new graph nodes, improving evaluation metrics, refining DFIR prompts,
  or any multi-step development work on the Valravn project.
  Triggers for: "add a feature", "implement X", "fix Y", "improve Z", "evaluation is failing",
  "add a new node", "re-run", "update", "revise", "improve the planner", "redo", "again",
  "do it again", "retry", "continue from last", "based on last result".
  Sub-task triggers also routed here: "test it", "write tests for X", "prompt is hallucinating".
---

# Valravn Dev Orchestrator

## Execution Mode

- **Feature work (node + tests):** Agent team — `node-dev` + `test-writer` collaborate
- **Single-focus tasks:** Sub-agent — prompt engineering, evaluation runs, isolated fixes
- **Full feature with prompt work:** Hybrid — team for implementation, sub-agent for prompt polish

## Phase 0: Context Check

Before starting, check for prior workspace:
- `_workspace/` exists + user requested partial update → **partial re-run** (invoke only relevant agent)
- `_workspace/` exists + new feature/input → **new run** (rename `_workspace/` to `_workspace_prev/`, start fresh)
- `_workspace/` missing → **initial run**

## Phase 1: Task Analysis

Parse the user request into one or more focused tasks:

| Request type | Team or sub? | Agents involved |
|-------------|-------------|----------------|
| New node end-to-end | Team | node-dev + test-writer |
| Fix a bug in a node | Sub-agent | node-dev |
| Write tests only | Sub-agent | test-writer |
| Improve a prompt | Sub-agent | prompt-engineer |
| Improve prompt + add tests | Team | prompt-engineer + test-writer |
| Run evaluations | Direct (no agent) | `python -m valravn.evaluation.evaluators --suite all` |

If the request is ambiguous, ask the user to clarify before spawning agents.

## Phase 2: Team Mode — Feature Work

Use when implementing a new node or significant change that needs both code and tests.

```
TeamCreate(
  team_name="valravn-feature",
  members=["node-dev", "test-writer"]
)

TaskCreate tasks:
  1. node-dev: Implement <node_name> in src/valravn/nodes/<node_name>.py
     - Read relevant existing nodes for patterns
     - Wire into graph.py
     - Output: implementation file path
  2. test-writer: Write unit tests for <node_name>
     - Depends on task 1 (read the implementation)
     - Output: tests/unit/test_<node_name>.py
  3. Verify: run pytest tests/unit/ -v
```

Agent calls (all `model: "opus"`):
```python
# node-dev — implement first
Agent(
  subagent_type="general-purpose",
  model="opus",
  prompt="[node-dev role + valravn-node-dev skill + specific task]"
)

# test-writer — after implementation complete
Agent(
  subagent_type="general-purpose",
  model="opus",
  prompt="[test-writer role + valravn-test-writer skill + specific task]"
)
```

## Phase 3: Sub-Agent Mode — Focused Tasks

For single-focus work, dispatch one agent directly:

```python
Agent(
  subagent_type="general-purpose",
  model="opus",
  prompt="You are the Valravn [node-dev|test-writer|prompt-engineer] agent. [skill content]. Task: [specific task]."
)
```

## Phase 4: Verification

After all agent work completes:
1. Run `pytest tests/unit/ -v` — must pass
2. Run `ruff check .` — must pass
3. If evaluation-related changes: `python -m valravn.evaluation.evaluators --suite all`
4. Report results to user; if failures, diagnose and fix before claiming done

## Data Flow

- Agents write to `_workspace/{phase}_{agent}_{artifact}.{ext}`
- Final code goes to the actual source paths (`src/valravn/nodes/`, `tests/unit/`)
- Intermediate files stay in `_workspace/` for audit

## Error Handling

- If an agent fails or produces bad output: retry once with more explicit instructions
- If pytest fails after implementation: invoke test-writer sub-agent with the failure output
- If ruff fails: fix inline (minor style issues), or invoke node-dev with specific ruff errors
- Do not claim the task is complete until all checks pass

## Test Scenarios

**Happy path — new node:**
User: "Add a node that checks if yara rules matched anything in the tool output"
1. Phase 0: no prior workspace → initial run
2. Phase 1: new node = team mode
3. Phase 2: node-dev implements `check_yara_hits`, test-writer writes unit tests
4. Phase 4: pytest passes → done

**Partial re-run:**
User: "The tests for check_yara_hits are still failing"
1. Phase 0: `_workspace/` exists, partial re-run
2. Phase 1: test-only = sub-agent test-writer
3. Sub-agent reads current test file + failure output, fixes
4. Phase 4: pytest passes → done

**Evaluation:**
User: "Run the evaluation suite"
Direct: `python -m valravn.evaluation.evaluators --suite all`, report MLflow results
