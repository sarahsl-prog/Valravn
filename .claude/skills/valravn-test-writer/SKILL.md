---
name: valravn-test-writer
description: >
  Write, fix, or extend pytest tests for the Valravn DFIR agent.
  Use this skill whenever you are adding tests, fixing failing tests, improving coverage,
  working in tests/unit/ or tests/integration/, or working with conftest.py or fixtures.
  Also triggers for: "write tests for", "test the node", "fix failing test", "add coverage",
  "unit test", "integration test", "mock the LLM", "mock subprocess".
---

# Test Writing Skill

## Test Tiers

| Tier | Location | Mark | When |
|------|----------|------|------|
| Unit | `tests/unit/` | (none) | Always — no SIFT tools, no real LLM |
| Integration | `tests/integration/` | `@pytest.mark.integration` | SIFT workstation available |

Run commands:
```bash
pytest tests/unit/ -v                           # unit only
pytest tests/integration/ -v -m integration    # integration only
pytest                                          # all
```

## Unit Test Template

```python
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage
import pytest

from valravn.nodes.my_node import my_node
from valravn.models.task import InvestigationTask, InvestigationPlan, PlannedStep


@pytest.fixture
def base_state(tmp_path):
    task = InvestigationTask(
        prompt="Find persistence mechanisms",
        evidence_refs=[str(tmp_path / "disk.img")],
    )
    plan = InvestigationPlan(task_id=task.id, steps=[
        PlannedStep(skill_domain="sleuthkit", tool_cmd=["fls", "-r", str(tmp_path / "disk.img")], rationale="list files"),
    ])
    return {
        "task": task,
        "plan": plan,
        "invocations": [],
        "current_step_id": plan.steps[0].id,
        "_output_dir": str(tmp_path),
        "_retry_config": {"max_attempts": 1, "timeout_seconds": 5, "retry_delay_seconds": 0},
    }


@patch("valravn.nodes.my_node.get_llm")
def test_my_node_happy_path(mock_get_llm, base_state):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content='{"field": "value"}')
    mock_get_llm.return_value = mock_llm

    result = my_node(base_state)

    assert result["my_field"] == "value"
```

## Mocking Patterns

### Mock LLM
```python
@patch("valravn.nodes.plan.get_llm")
def test_plan(mock_get_llm, base_state):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content=json.dumps({
        "steps": [{"skill_domain": "sleuthkit", "tool_cmd": ["fls", "-r", "/img"], "rationale": "list"}]
    }))
    mock_get_llm.return_value = mock_llm
    # ...
```

### Mock subprocess (tool_runner tests)
```python
@patch("valravn.nodes.tool_runner.subprocess.Popen")
def test_tool_runner_success(mock_popen, base_state):
    mock_proc = MagicMock()
    mock_proc.stderr = MagicMock()
    mock_proc.stderr.fileno.return_value = -1  # won't be read
    mock_proc.returncode = 0
    mock_proc.wait.return_value = None
    mock_popen.return_value = mock_proc
    # Also patch os.read if needed for stderr streaming
```

### Stub subprocess with real tmp files (preferred for tool_runner)
Use `tmp_path` and write fake stdout/stderr files, then patch `_run_with_streaming` directly:
```python
@patch("valravn.nodes.tool_runner._run_with_streaming")
def test_tool_runner_success(mock_run, base_state, tmp_path):
    stdout_path = tmp_path / "analysis" / "fake.stdout"
    stdout_path.parent.mkdir(parents=True)
    stdout_path.write_text("MFT entry 0\n")
    mock_run.return_value = (0, True)  # (returncode, had_stdout)
```

## What Every New Node Needs

1. **Happy path** — verify the partial state dict keys and values
2. **Missing/empty input** — e.g., no steps in plan, empty evidence list
3. **LLM failure** — `mock_llm.invoke.side_effect = Exception("timeout")` — should not crash the graph
4. **Routing coverage** — if the node affects routing, test both branches

## Fixtures in conftest.py

Before adding a fixture, grep `tests/conftest.py`. Common reusable fixtures:
- `tmp_path` (built-in) — always use this for file I/O
- Add project-wide fixtures to `tests/conftest.py`
- Node-specific fixtures belong in `tests/unit/test_{node}.py`

## Integration Test Template

```python
import pytest
from pathlib import Path
from valravn.nodes.my_node import my_node

@pytest.mark.integration
def test_my_node_with_real_evidence(tmp_path):
    # Requires: SIFT workstation, real evidence at tests/fixtures/
    evidence = Path("tests/fixtures/sample_disk.img")
    if not evidence.exists():
        pytest.skip("SIFT fixtures not available")
    # ... real test
```

## Naming Conventions

- File: `test_{module_name}.py` (mirrors `src/valravn/{area}/{module_name}.py`)
- Function: `test_{node_or_function}_{condition_or_scenario}`
  - `test_run_forensic_tool_success`
  - `test_run_forensic_tool_timeout_sets_step_exhausted`
  - `test_plan_investigation_empty_steps_routes_to_conclusions`
