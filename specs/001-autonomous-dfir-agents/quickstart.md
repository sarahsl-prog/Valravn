# Developer Quickstart: Autonomous DFIR Agents

**Branch**: `001-autonomous-dfir-agents` | **Date**: 2026-04-05

Valravn is designed for air-gapped SANS SIFT workstations. No cloud services are required
at runtime — all tracing, evaluation, and artifact storage is local.

---

## Prerequisites

- SANS SIFT Ubuntu Workstation (x86-64)
- Python 3.12 (`.venv` pre-initialised in repo root)
- `ANTHROPIC_API_KEY` — Claude model access via `langchain-anthropic`
- SIFT forensic tools available on PATH (Volatility 3, Sleuth Kit, EZ Tools, Plaso, YARA)

No external accounts or network access needed beyond the Anthropic API for the model itself.

---

## Setup

```bash
# Activate the virtual environment
source .venv/bin/activate

# Install dependencies (once pyproject.toml is created in implementation phase)
pip install -e ".[dev]"

# Set API key
export ANTHROPIC_API_KEY=<your-key>

# Start the local MLflow tracking server (leave running in a separate terminal)
mlflow server --host 127.0.0.1 --port 5000
```

---

## Run the Agent

```bash
# Minimal invocation (evidence must be read-only mounted)
valravn investigate \
  --prompt "Identify all processes with network connections at time of acquisition" \
  --evidence /mnt/cases/001/memory.lime

# With explicit output directory and config
valravn investigate \
  --prompt "Find evidence of lateral movement via SMB" \
  --evidence /mnt/cases/002/disk.e01 \
  --config config.yaml \
  --output-dir /cases/002/valravn-output
```

The agent runs autonomously via a LangGraph `StateGraph`. Watch stdout for node-level
progress. When complete:

```
./analysis/checkpoints.db          # SqliteSaver — full AgentState per node (crash-safe)
./analysis/traces/<run-id>.jsonl   # full node/LLM/tool trace
./analysis/investigation_plan.json # live plan evolution (human-readable audit copy)
./analysis/<uuid>.stdout/.stderr   # raw tool output
./reports/<timestamp>_<slug>.md    # findings report
```

Each run is also logged to the local MLflow server. View at `http://127.0.0.1:5000`.

---

## Run Tests

```bash
# Unit tests only (no SIFT tools required)
pytest tests/unit/ -v

# Integration tests (requires SIFT tools on PATH)
pytest tests/integration/ -v -m integration

# All tests
pytest -v
```

---

## Run Evaluations

Evaluation datasets live locally in `tests/evaluation/datasets/` as JSONL files.
The MLflow server must be running first.

```bash
# Evaluate anomaly detection rate (SC-002)
python -m valravn.evaluation.evaluators --suite anomaly-detection

# Evaluate citation coverage (SC-004: every conclusion cites a tool invocation)
python -m valravn.evaluation.evaluators --suite citation-coverage

# Run all evaluators
python -m valravn.evaluation.evaluators --suite all
```

Results appear at `http://127.0.0.1:5000` under the `valravn-evaluation` experiment.
Each evaluator logs pass/fail metrics for one spec success criterion (SC-002 through SC-006).

To add a golden test case from a completed investigation:
```bash
python -m valravn.evaluation.datasets --add ./reports/20260405_120000_identify_active_network.json
```

---

## Configuration

Create `config.yaml` in your case directory or repo root:

```yaml
retry:
  max_attempts: 3       # Attempts per tool invocation before escalation
  retry_delay_seconds: 0

mlflow:
  tracking_uri: http://127.0.0.1:5000
  experiment_name: valravn-evaluation

# Multi-provider LLM configuration (with fallback support)
models:
  plan:
    - ollama:kimi-k2.5:cloud
    - ollama:qwen3:14b
  # ... (other modules)

# RCL Training (opt-in)
training:
  enabled: false
  state_dir: ./training
  min_failure_trace_length: 100

# Skill paths
skills:
  base_path: ~/.claude/skills

# Checkpoint cleanup (set auto_cleanup: false for audit mode)
checkpoint_cleanup:
  retention_days: 7
  max_checkpoints_per_thread: 1000
  min_checkpoints_per_thread: 2
  auto_cleanup: true
  auto_vacuum: false
```

Override retry limit at runtime:
```bash
VALRAVN_MAX_RETRIES=5 valravn investigate --prompt "..." --evidence /mnt/...
```

---

## Project Layout

```
src/valravn/
├── cli.py                  # Entry point
├── graph.py                # LangGraph StateGraph — nodes + edges + conditional routing
├── state.py                # AgentState TypedDict
├── config.py               # AppConfig with all settings
├── checkpoint_cleanup.py   # SQLite checkpoint management
├── nodes/                  # One module per graph node
│   ├── plan.py
│   ├── skill_loader.py
│   ├── tool_runner.py
│   ├── anomaly.py          # Trust-based anomaly detection
│   ├── report.py
│   ├── conclusions.py      # Conclusion synthesis
│   └── self_assess.py      # Self-assessment for trust
├── models/                 # Pydantic data models
│   ├── task.py
│   ├── records.py
│   └── report.py
├── core/                   # Core utilities
│   └── llm_factory.py      # Multi-provider LLM factory
├── evaluation/             # MLflow-based evaluators
│   └── evaluators.py
└── training/               # RCL training system
    ├── playbook.py
    ├── replay_buffer.py
    ├── reflector.py
    ├── mutator.py
    ├── feasibility.py
    └── rcl_loop.py

tests/
├── unit/            # Mocked subprocess, no SIFT tools needed
├── integration/     # Real tools, synthetic evidence fixtures
├── evaluation/
│   └── datasets/    # Golden test cases as local JSONL files
└── fixtures/
    └── evidence/    # Read-only synthetic artifacts
```

---

## Adding a New Forensic Domain

1. Add the skill file to `~/.claude/skills/<domain>/SKILL.md` (or your configured base_path)
2. The domain will be automatically available — no code changes needed

To configure custom skill paths, set in `config.yaml`:
```yaml
skills:
  base_path: /path/to/skills
```

---

## Evidence Integrity Verification

Valravn refuses to start if any `--evidence` path is writable:

```
$ valravn investigate --prompt "..." --evidence /tmp/test-image.raw
Error: Evidence path /tmp/test-image.raw is writable. Mount evidence read-only before invoking Valravn.
Exit code: 2
```

To mount read-only:
```bash
ewfmount /cases/001/image.E01 /mnt/cases/001/  # EWF images
mount -o ro /dev/sdb1 /mnt/evidence/            # Raw partitions
```
