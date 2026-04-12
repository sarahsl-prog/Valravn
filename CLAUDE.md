# Valravn Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-04-05

## Active Technologies

- Python 3.12+ (`.venv` already initialised in repo root — actual runtime is Python 3.13) + `langgraph` + `langchain-anthropic` + `mlflow`; SIFT native tools via subprocess (001-autonomous-dfir-agents)

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
# LLM providers (set at least one + configure VALRAVN_{MODULE}_MODEL to match)
ANTHROPIC_API_KEY       # required for anthropic: provider
OPENAI_API_KEY          # required for openai: provider
OPENROUTER_API_KEY      # required for openrouter: provider
OLLAMA_BASE_URL         # optional — Ollama server URL (default: http://localhost:11434)

# Per-module model selection (comma-separated for fallback chains)
# e.g. VALRAVN_PLAN_MODEL=anthropic:claude-sonnet-4-6,anthropic:claude-haiku-4-5-20251001
VALRAVN_PLAN_MODEL      # model for plan_investigation node
VALRAVN_ANOMALY_MODEL   # model for check_anomalies / record_anomaly nodes
VALRAVN_TOOL_MODEL      # model for run_forensic_tool self-correction
VALRAVN_CONCLUSIONS_MODEL  # model for synthesize_conclusions node
VALRAVN_REPORT_MODEL    # model for write_findings_report node
VALRAVN_REFLECTOR_MODEL # model for RCL reflector
VALRAVN_MUTATOR_MODEL   # model for RCL mutator

# Optional overrides
VALRAVN_MAX_RETRIES     # optional override for retry.max_attempts (default: 3)
MLFLOW_TRACKING_URI     # optional — MLflow server URI (default: local ./mlruns)
```

No `LANGCHAIN_API_KEY` or `LANGCHAIN_TRACING_V2` — LangSmith is not used.

## Code Style

Python 3.12+: Follow standard conventions. Pydantic v2 for all data models. LangGraph
nodes are plain functions accepting `AgentState` and returning partial state dicts.

## Recent Changes

- 001-autonomous-dfir-agents: Evaluation replaced LangSmith → MLflow (local); air-gap safe

<!-- MANUAL ADDITIONS START -->

## 하네스: Valravn DFIR Agent Development

**목표:** LangGraph 노드 구현, 테스트 작성, LLM 프롬프트 개선, 평가 실행을 전문 에이전트 팀으로 조율한다.

**트리거:** Valravn 기능 추가, 노드 구현, 버그 수정, 테스트 작성, 프롬프트 개선, 평가 실행 요청 시 `valravn-dev` 스킬을 사용하라. 단순 질문은 직접 응답 가능.

**변경 이력:**
| 날짜 | 변경 내용 | 대상 | 사유 |
|------|----------|------|------|
| 2026-04-11 | 초기 구성 | 전체 | 신규 구축 |

<!-- MANUAL ADDITIONS END -->
