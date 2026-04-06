# Valravn

**Autonomous DFIR investigation agent for SANS SIFT Workstations.**

Valravn orchestrates forensic tool execution via a [LangGraph](https://github.com/langchain-ai/langgraph) state machine, driven by Claude. Given an investigation prompt and one or more read-only evidence paths, it plans a sequence of forensic tool invocations, executes them, detects anomalies, self-corrects on failure, and produces a structured Markdown + JSON findings report — all locally, with no cloud services required at runtime.

---

## Requirements

| Requirement | Details |
|-------------|---------|
| Platform | SANS SIFT Ubuntu Workstation (x86-64) |
| Python | 3.12 |
| Claude API key | `ANTHROPIC_API_KEY` — used for planning, anomaly detection, and self-correction |
| SIFT tools | Volatility 3, Sleuth Kit (`fls`, `icat`, etc.), Plaso (`log2timeline.py`), YARA, EZ Tools |

No LangSmith, no external telemetry. All tracing and evaluation artifacts are stored locally.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/sarahsl-prog/Valravn.git
cd Valravn

# Activate the virtual environment (pre-initialised in repo root)
source .venv/bin/activate

# Install the package with development dependencies
pip install -e ".[dev]"

# Export your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...
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
  reports/
    <timestamp>_<slug>.md          # Markdown findings report
    <timestamp>_<slug>.json        # Machine-readable FindingsReport
```

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
```

Override retry limit at runtime without editing the file:

```bash
VALRAVN_MAX_RETRIES=5 valravn investigate --prompt "..." --evidence /mnt/...
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
  cli.py              # CLI entry point (valravn investigate ...)
  graph.py            # LangGraph StateGraph — nodes, edges, conditional routing
  state.py            # AgentState TypedDict
  config.py           # AppConfig, OutputConfig, RetryConfig
  nodes/
    plan.py           # LLM-driven investigation planner
    skill_loader.py   # Loads domain SKILL.md files from ~/.claude/skills/
    tool_runner.py    # Subprocess executor with retry and self-correction
    anomaly.py        # LLM-driven anomaly detector and follow-up generator
    report.py         # Markdown + JSON report writer
  models/
    task.py           # InvestigationTask, InvestigationPlan, PlannedStep
    records.py        # ToolInvocationRecord, Anomaly
    report.py         # FindingsReport, Conclusion, ToolFailureRecord
  evaluation/
    evaluators.py     # MLflow-backed SC evaluators
    datasets.py       # Golden test case management

tests/
  unit/               # All mocked — no SIFT tools needed
  integration/        # Real tools, synthetic evidence fixtures
  fixtures/evidence/  # Read-only stub evidence files
```

---

## Adding a Forensic Domain

1. Create `~/.claude/skills/<domain>/SKILL.md` with tool invocation guidance.
2. Add the domain key to `SKILL_PATHS` in `src/valravn/nodes/skill_loader.py`.
3. The `load_skill` graph node resolves paths dynamically — no other changes needed.

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
