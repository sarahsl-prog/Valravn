# Valravn Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-04-05

## Active Technologies

- Python 3.12 (`.venv` already initialised in repo root) + `langgraph` + `langchain-anthropic` + `mlflow`; SIFT native tools via subprocess (001-autonomous-dfir-agents)

**Air-gap constraint**: No cloud services at runtime. All tracing, evaluation, and artifact
storage is local. Do not introduce dependencies that phone home by default.

## Project Structure

```text
src/valravn/
  cli.py, graph.py, state.py
  nodes/  models/  evaluation/
tests/
  unit/  integration/  fixtures/  evaluation/datasets/
```

## Commands

```bash
pytest                                               # all tests
pytest tests/unit/ -v                               # unit only (no SIFT tools)
pytest tests/integration/ -v -m integration         # integration (SIFT required)
mlflow server --host 127.0.0.1 --port 5000          # start local tracking server
python -m valravn.evaluation.evaluators --suite all # run all SC evaluators
ruff check .
```

## Environment Variables

```
ANTHROPIC_API_KEY       # required for Anthropic models — Claude via langchain-anthropic
OPENAI_API_KEY          # required for OpenAI models
OPENROUTER_API_KEY      # required for OpenRouter
OLLAMA_BASE_URL         # required for Ollama (default: http://localhost:11434)
VALRAVN_MAX_RETRIES     # optional override for retry.max_attempts (default: 3)
VALRAVN_{MODULE}_MODEL  # optional per-module model override (e.g., VALRAVN_PLAN_MODEL)
```

No `LANGCHAIN_API_KEY` or `LANGCHAIN_TRACING_V2` — LangSmith is not used.

## Code Style

Python 3.12: Follow standard conventions. Pydantic v2 for all data models. LangGraph
nodes are plain functions accepting `AgentState` and returning partial state dicts.

## Recent Changes

- 001-autonomous-dfir-agents: Evaluation replaced LangSmith → MLflow (local); air-gap safe
- v0.2.0 (2026-04-09): Added RCL training system (opt-in), checkpoint cleanup config, configurable skill paths, trust-based anomaly filtering, Ollama model fallback support

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
