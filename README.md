# Valravn

**Autonomous DFIR investigation agent for SANS SIFT Workstations.**

Valravn orchestrates forensic tool execution via a [LangGraph](https://github.com/langchain-ai/langgraph) state machine, driven by Claude. Given an investigation prompt and one or more read-only evidence paths, it plans a sequence of forensic tool invocations, executes them, detects anomalies, self-corrects on failure, and produces a structured Markdown + JSON findings report — all locally, with no cloud services required at runtime.

---

## Requirements

| Requirement | Details |
|-------------|---------|
| Platform | SANS SIFT Ubuntu Workstation (x86-64) |
| Python | 3.12 |
| LLM Provider | Anthropic, OpenAI, Ollama, or OpenRouter via unified factory |
| SIFT tools | Volatility 3, Sleuth Kit (`fls`, `icat`, etc.), Plaso (`log2timeline.py`), YARA, EZ Tools |

**Model Support:**
- **Anthropic** (`ANTHROPIC_API_KEY`): Claude 3.x models (default)
- **OpenAI** (`OPENAI_API_KEY`): GPT-4o, GPT-4o-mini, etc.
- **Ollama** (`OLLAMA_BASE_URL` for local): Llama, Mistral, etc.
- **OpenRouter** (`OPENROUTER_API_KEY` + base URL): Any model via OpenRouter

No LangSmith, no external telemetry. All tracing and evaluation artifacts are stored locally.

---

## Installation

### Prerequisites

Valravn requires forensic tools from the SANS SIFT Workstation. If running on a standard Ubuntu system, install the tools:

**Sleuth Kit (file system analysis):**
```bash
sudo apt install sleuthkit
```

**Volatility 3 (memory forensics):**
```bash
# Download from https://github.com/volatilityfoundation/volatility3/releases
# or clone the repository
sudo mkdir -p /opt/volatility3-2.20.0
sudo pip install volatility3  # or download release tarball
# Create symlink if needed: sudo ln -s /path/to/vol.py /opt/volatility3-2.20.0/vol.py
```

**Plaso (timeline analysis):**
```bash
sudo add-apt-repository ppa:gift/stable
sudo apt install python3-plaso plaso-tools python3-pytsk3
```

**YARA (pattern matching):**
```bash
sudo apt install yara
# or from source for latest version
```

**EZ Tools (Windows artifacts):**
Download from https://ericzimmerman.github.io/ and install to `/opt/zimmermantools/`

**EWF tools (E01 image support):**
```bash
sudo apt install ewf-tools
```

### Valravn Installation

```bash
# Clone the repository
git clone https://github.com/sarahsl-prog/Valravn.git
cd Valravn

# Activate the virtual environment (pre-initialised in repo root)
source .venv/bin/activate

# Install with your chosen LLM provider(s):
pip install -e ".[anthropic]"          # Claude only (default)
pip install -e ".[openai]"             # OpenAI only
pip install -e ".[ollama]"             # Ollama only
pip install -e ".[all]"                # All providers

# Add dev tools alongside a provider:
pip install -e ".[anthropic,dev]"

# Export your Anthropic API key (or see Configuration below for other providers)
export ANTHROPIC_API_KEY=sk-ant-...
```

### Verify Tool Installation

```bash
# Check that all required forensic tools are available
valravn check-tools

# Detailed report with paths
valravn check-tools --verbose
```

---

## Quick Start

Evidence **must be read-only** before invoking Valravn. The agent refuses to start if any evidence path is writable.

```bash
# Mount evidence read-only first
ewfmount /cases/001/image.E01 /mnt/cases/001/   # EWF images
# or
mount -o ro /dev/sdb1 /mnt/evidence/             # raw partitions

# Run an investigation
valravn investigate \
  --prompt "Identify all processes with network connections at time of acquisition" \
  --evidence /mnt/cases/001/memory.lime

# Multiple evidence sources, custom output directory, explicit config
valravn investigate \
  --prompt "Find evidence of lateral movement via SMB" \
  --evidence /mnt/cases/002/disk.e01 \
  --evidence /mnt/cases/002/memory.lime \
  --config config.yaml \
  --output-dir /cases/002/valravn-output
```

### Output artifacts

```
<output-dir>/
  analysis/
    investigation_plan.json        # live plan (updated after each step)
    anomalies.json                 # detected anomalies
    checkpoints.db                 # SQLite — full AgentState per node (crash-safe)
    traces/<run-id>.jsonl          # JSONL trace of every LLM and tool event
    <uuid>.stdout / .stderr        # raw tool output per invocation
    <uuid>.record.json             # ToolInvocationRecord metadata
    abandoned_cases.jsonl          # (RCL training) rejected trajectories for later review
  reports/
    <timestamp>_<slug>.md           # Markdown findings report
    <timestamp>_<slug>.json        # Machine-readable FindingsReport
```

**Checkpoint Database:** The SQLite `checkpoints.db` enables crash recovery. Re-running with the same `thread_id` resumes from the last completed node. Configure automatic cleanup via `config.yaml` (see [docs/configuration.md](docs/configuration.md)).

---

## Configuration

Copy `config.yaml` to your output directory or repo root and adjust as needed:

```yaml
retry:
  max_attempts: 3         # tool invocation attempts before marking step exhausted
  retry_delay_seconds: 0  # delay between attempts (seconds)

mlflow:
  tracking_uri: http://127.0.0.1:5000
  experiment_name: valravn-evaluation

# Multi-provider LLM configuration (per-module model selection)
# Supports single model or fallback list (tried in order)
models:
  plan:
    - ollama:kimi-k2.5:cloud
    - ollama:qwen3:14b
  reflector:
    - ollama:kimi-k2.5:cloud
    - ollama:qwen3:14b
  mutator:
    - ollama:kimi-k2.5:cloud
    - ollama:qwen3:14b
  anomaly:
    - ollama:kimi-k2.5:cloud
    - ollama:qwen3:14b
  conclusions:
    - ollama:kimi-k2.5:cloud
    - ollama:qwen3:14b
  tool_runner:
    - ollama:kimi-k2.5:cloud
    - ollama:qwen3:14b

# RCL Training System (opt-in)
training:
  enabled: false              # Set to true to enable learning from failures
  state_dir: ./training       # Directory for training artifacts
  min_failure_trace_length: 100

# Skill Paths (forensic domain SKILL.md files)
skills:
  base_path: ~/.claude/skills

# Checkpoint Cleanup (set auto_cleanup: false for audit mode)
checkpoint_cleanup:
  retention_days: 7
  max_checkpoints_per_thread: 1000
  min_checkpoints_per_thread: 2
  auto_cleanup: true
  auto_vacuum: false
```

Override retry limit at runtime without editing the file:

```bash
VALRAVN_MAX_RETRIES=5 valravn investigate --prompt "..." --evidence /mnt/...
```

**Per-provider environment variables:**

```bash
# Anthropic (default)
export ANTHROPIC_API_KEY=sk-ant-...

# OpenAI
export OPENAI_API_KEY=sk-...

# Ollama (local)
export OLLAMA_BASE_URL=http://localhost:11434

# OpenRouter
export OPENROUTER_API_KEY=sk-or-...

# Per-module model override
export VALRAVN_PLAN_MODEL=openai:gpt-4o
export VALRAVN_MUTATOR_MODEL=anthropic:claude-3-opus-20240229
```

See [docs/configuration.md](docs/configuration.md) for the full configuration reference.

---

## Testing

```bash
# Unit tests — no SIFT tools required
pytest tests/unit/ -v --cov=src/valravn --cov-report=term-missing

# Integration tests — requires SIFT tools on PATH
pytest tests/integration/ -v -m integration --cov=src/valravn

# All tests with coverage
pytest -v --cov=src/valravn --cov-report=term-missing --cov-report=html

# Linting
ruff check .
```

### Test Coverage Status

| Module | Coverage |
|--------|----------|
| **models/** | 100% |
| **nodes/plan.py** | 94% |
| **nodes/tool_runner.py** | 94% |
| **nodes/anomaly.py** | 95% |
| **nodes/skill_loader.py** | 95% |
| **nodes/report.py** | 77% |
| **evaluation/evaluators.py** | 77% |
| **cli.py** | 0% (requires integration) |
| **graph.py** | 0% (requires integration) |
| **evaluation/datasets.py** | 0% (tests exist) |

**Overall: ~73% unit, ~66% with integration**  
*Coverage gaps in CLI/graph due to external LLM/SIFT tool dependencies.*

**Run coverage report:**
```bash
open htmlcov/index.html  # View detailed coverage HTML
```

---

## Evaluation (MLflow)

Valravn ships with MLflow-based evaluators for its success criteria. Start the local tracking server first:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Then run evaluators against a completed report:

```bash
# Single suite
python -m valravn.evaluation.evaluators \
  --suite anomaly-detection \
  --report reports/20260405_120000_identify_active_network.json

# All suites
python -m valravn.evaluation.evaluators \
  --suite all \
  --report reports/20260405_120000_identify_active_network.json
```

View results at `http://127.0.0.1:5000` under the `valravn-evaluation` experiment.

Available suites: `anomaly-detection`, `citation-coverage`, `evidence-integrity`, `self-correction`, `report-completeness`.

---

## Project Structure

```
src/valravn/
  cli.py                # CLI entry point (valravn investigate ...)
  graph.py              # LangGraph StateGraph — nodes, edges, conditional routing
  state.py              # AgentState TypedDict
  config.py             # AppConfig, OutputConfig, RetryConfig
  checkpoint_cleanup.py # SQLite checkpoint management (Q6)
  core/
    llm_factory.py      # Multi-provider LLM factory (Q4) — Anthropic, OpenAI, Ollama, OpenRouter
  nodes/
    plan.py             # LLM-driven investigation planner
    skill_loader.py     # Loads domain SKILL.md files from ~/.claude/skills/ (configurable)
    tool_runner.py      # Subprocess executor with retry and self-correction
    anomaly.py          # LLM-driven anomaly detector with trust-based filtering
    report.py           # Markdown + JSON report writer
    conclusions.py      # LLM-driven conclusion synthesis
    self_assess.py      # Self-assessment node for progress evaluation
  models/
    task.py             # InvestigationTask, InvestigationPlan, PlannedStep
    records.py          # ToolInvocationRecord, Anomaly
    report.py           # FindingsReport, Conclusion, ToolFailureRecord
  training/             # RCL training system (Q1-Q5)
    playbook.py         # SecurityPlaybook with protected entries (Q5)
    replay_buffer.py    # ReplayBuffer with archiving (Q1)
    reflector.py        # Trajectory reflection and attribution (BUG-002)
    mutator.py          # Playbook mutation with validation (BUG-003)
    feasibility.py      # Feasibility rules registry (Q2) + FeasibilityMemory
    rcl_loop.py         # Main RCL training orchestration
    feasibility.py      # Custom feasibility rules for replay buffer filtering
  evaluation/
    evaluators.py       # MLflow-backed SC evaluators
    datasets.py         # Golden test case management

tests/
  unit/               # All mocked — no SIFT tools needed
  integration/        # Real tools, synthetic evidence fixtures
  fixtures/evidence/  # Read-only stub evidence files
```

---

## Adding a Forensic Domain

1. Create a SKILL.md file at `<skills.base_path>/<domain>/SKILL.md` (default: `~/.claude/skills/<domain>/SKILL.md`).
2. The skill will be automatically available by domain name — no code changes needed.
3. Configure custom skill paths in `config.yaml` under the `skills:` section.

Supported domains out of the box: `memory-analysis`, `sleuthkit`, `windows-artifacts`, `plaso-timeline`, `yara-hunting`.

---

## Evidence Integrity

Valravn enforces chain-of-custody at startup. Any writable evidence path causes an immediate exit:

```
Error: Evidence path /tmp/test-image.raw is writable. Mount evidence read-only before invoking Valravn.
Exit code: 2
```

Output is never written to evidence directories. Analysis artifacts always go to `<output-dir>/analysis/` and `<output-dir>/reports/`.

---

## Architecture

See [docs/architecture.md](docs/architecture.md) for a detailed walkthrough of the LangGraph state machine, node responsibilities, and data flow.

---

## Contributing

See [docs/contributing.md](docs/contributing.md).

---

## License

MIT — see [LICENSE](LICENSE).
