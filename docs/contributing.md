# Contributing to Valravn

## Development Setup

```bash
git clone https://github.com/sarahsl-prog/Valravn.git
cd Valravn

# Activate the virtual environment
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Verify the install
valravn --help
```

Required environment variable:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Running Tests

```bash
# Unit tests (no SIFT tools needed, all subprocess calls are mocked)
pytest tests/unit/ -v

# Integration tests (require SIFT tools on PATH)
pytest tests/integration/ -v -m integration

# All tests
pytest -v
```

Unit tests mock all subprocess and LLM calls. Integration tests use stub evidence files in `tests/fixtures/evidence/` and require live SIFT tools.

---

## Linting

```bash
ruff check .
```

Ruff is configured in `pyproject.toml` with `E`, `F`, and `I` rules and a 100-character line length. Fix import order and style issues before opening a PR.

---

## Project Constraints

### Air-gap

Valravn runs on air-gapped SIFT workstations. Do not introduce dependencies that make network requests at import time or by default. All tracing and evaluation must remain local.

### Evidence integrity

Code that touches evidence paths must never open them for writing. The `InvestigationTask` model validator and the `run_forensic_tool` node both enforce this. Any new code that accepts evidence paths must apply the same guard.

### Timestamps

All timestamps are UTC. Use `datetime.now(timezone.utc)` — never `datetime.utcnow()` (which returns a naive datetime).

### Pydantic v2

All data models use Pydantic v2. Use `model_dump(mode="json")` for serialisation and `model_validate(data)` for deserialisation.

### LangGraph nodes

Graph nodes are plain functions with the signature:

```python
def my_node(state: dict) -> dict:
    ...
    return {"field": new_value}
```

Nodes return a **partial** state dict. LangGraph merges it into the accumulated state. Do not return the full state object.

---

## Adding a Forensic Domain

1. Create `~/.claude/skills/<domain>/SKILL.md` with guidance on which tools to use and how.
2. Add an entry to `SKILL_PATHS` in `src/valravn/nodes/skill_loader.py`:
   ```python
   "my-domain": _SKILLS_BASE / "my-domain" / "SKILL.md",
   ```
3. Add unit tests covering the new domain key in `tests/unit/test_skill_loader.py`.

No other changes are needed — the planner, tool runner, and anomaly nodes are domain-agnostic.

---

## Adding an Evaluator

Evaluators live in `src/valravn/evaluation/evaluators.py`. Each evaluator is a function with the signature:

```python
def _eval_my_criterion(report: FindingsReport) -> bool:
    ...
```

Register it in `_SUITE_MAP`:

```python
"my-criterion": ("sc_NNN_my_criterion", _eval_my_criterion),
```

Add the suite key to `SUITES` and write a unit test in `tests/unit/test_evaluators.py`.

---

## Commit Style

Follow the existing commit style: lowercase imperative subject line, no trailing period. Reference the spec number where relevant (e.g. `fix: anomaly loop depth cap (T016)`).

---

## Branch Naming

Feature branches follow the pattern `NNN-short-description` (e.g. `001-autonomous-dfir-agents`). Documentation and fixup branches can use descriptive names (e.g. `doc-updates`).
