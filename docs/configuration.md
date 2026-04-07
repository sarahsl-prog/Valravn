# Valravn — Configuration Reference

## config.yaml

Valravn loads configuration from a YAML file. Pass it explicitly with `--config`, or omit it to use built-in defaults.

```yaml
retry:
  max_attempts: 3          # integer ≥ 1; attempts per forensic tool step
  retry_delay_seconds: 0.0 # float ≥ 0; pause between attempts (seconds)

mlflow:
  tracking_uri: http://127.0.0.1:5000   # local MLflow server URI
  experiment_name: valravn-evaluation    # MLflow experiment name for evaluators
```

### `retry.max_attempts`

Controls how many times the `run_forensic_tool` node will attempt a single `PlannedStep` before marking it `EXHAUSTED`.

- On each failed attempt (before the last), Claude is asked for a corrected command. The correction is recorded as a `SelfCorrectionEvent`.
- On the final failed attempt, a `ToolFailureRecord` is written and the step is skipped.
- Default: `3`

### `retry.retry_delay_seconds`

Seconds to sleep between tool invocation attempts. Set to `0` (default) for no delay.

### `mlflow.tracking_uri`

URI of the local MLflow tracking server used by the evaluation module. The server must be running before executing evaluators:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

### `mlflow.experiment_name`

MLflow experiment name under which evaluation runs are logged. Defaults to `valravn-evaluation`.

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API key. Required if using Anthropic models. | — |
| `OPENAI_API_KEY` | OpenAI API key. Required if using OpenAI models. | — |
| `OLLAMA_BASE_URL` | Ollama server URL. Default: `http://localhost:11434`. | — |
| `OPENROUTER_API_KEY` | OpenRouter API key. Required if using OpenRouter. | — |
| `OPENROUTER_BASE_URL` | OpenRouter base URL. Default: `https://openrouter.ai/api/v1`. | — |
| `VALRAVN_{MODULE}_MODEL` | Override model for a specific module (e.g., `VALRAVN_PLAN_MODEL`). | `config.yaml` value |
| `VALRAVN_MAX_RETRIES` | Override `retry.max_attempts` without editing `config.yaml`. | `config.yaml` value |
| `MLFLOW_TRACKING_URI` | Override the MLflow tracking URI for the evaluation module. | `http://127.0.0.1:5000` |

Example:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
VALRAVN_MAX_RETRIES=5 valravn investigate --prompt "..." --evidence /mnt/...
```

---

## CLI Flags

```
valravn investigate
  --prompt TEXT        Investigation question in natural language  [required]
  --evidence PATH      Evidence file or directory (repeat for multiple)  [required]
  --config PATH        Path to config.yaml  [optional]
  --output-dir PATH    Root directory for analysis/ and reports/  [default: .]
```

### `--prompt`

A natural-language description of the investigation objective. The planner node sends this verbatim to Claude along with the evidence paths.

Examples:
- `"Identify all processes with active network connections at time of acquisition"`
- `"Find evidence of lateral movement via SMB shares"`
- `"Determine whether any user account was created in the 24 hours before the incident"`

### `--evidence`

Path to a read-only evidence file or directory. May be repeated:

```bash
valravn investigate \
  --prompt "..." \
  --evidence /mnt/cases/001/memory.lime \
  --evidence /mnt/cases/001/disk.e01
```

Valravn validates that every evidence path:
1. Exists on the filesystem.
2. Is **not writable** by the current user.

It will exit with code `2` if either check fails.

### `--output-dir`

Root directory for all output artifacts. Sub-directories `analysis/`, `reports/`, `exports/`, and `analysis/traces/` are created automatically.

The output directory must **not** be inside an evidence directory. Valravn enforces this at runtime.

---

## Output Directory Layout

```
<output-dir>/
  analysis/
    investigation_plan.json      # Current plan state (updated after each step)
    anomalies.json               # Detected anomalies list
    checkpoints.db               # SQLite — AgentState snapshots (crash recovery)
    traces/
      <run-id>.jsonl             # Full LLM + tool event trace (JSONL)
    <uuid>.stdout                # Raw stdout for one tool invocation
    <uuid>.stderr                # Raw stderr for one tool invocation
    <uuid>.record.json           # ToolInvocationRecord metadata
  reports/
    <timestamp>_<slug>.md        # Markdown findings report
    <timestamp>_<slug>.json      # Machine-readable FindingsReport
  exports/                       # Reserved for future export artefacts
```

---

## Skill Files

Valravn looks for domain skill files at:

```
~/.claude/skills/<domain>/SKILL.md
```

Registered domains and their paths:

| Domain key | Skill file |
|------------|------------|
| `memory-analysis` | `~/.claude/skills/memory-analysis/SKILL.md` |
| `sleuthkit` | `~/.claude/skills/sleuthkit/SKILL.md` |
| `windows-artifacts` | `~/.claude/skills/windows-artifacts/SKILL.md` |
| `plaso-timeline` | `~/.claude/skills/plaso-timeline/SKILL.md` |
| `yara-hunting` | `~/.claude/skills/yara-hunting/SKILL.md` |

To add a domain, create the SKILL.md file and add the key to `SKILL_PATHS` in `src/valravn/nodes/skill_loader.py`.

---

## Multi-Provider LLM Configuration

Valravn supports multiple LLM providers via a unified factory. Configure models per-module in `config.yaml`:

```yaml
models:
  plan: anthropic:claude-3-5-sonnet-20241022
  reflector: anthropic:claude-3-5-sonnet-20241022
  mutator: anthropic:claude-3-5-sonnet-20241022
  anomaly: anthropic:claude-3-5-sonnet-20241022
  conclusions: anthropic:claude-3-5-sonnet-20241022
  tool_runner: openai:gpt-4o  # Example: use OpenAI for tool corrections
```

**Provider format:** `provider:model_name`

| Provider | Format Example | Notes |
|----------|----------------|-------|
| Anthropic | `anthropic:claude-3-5-sonnet-20241022` | Requires `ANTHROPIC_API_KEY` |
| OpenAI | `openai:gpt-4o` | Requires `OPENAI_API_KEY` |
| Ollama | `ollama:llama3.2` | Local inference, requires Ollama server |
| OpenRouter | `openrouter:anthropic/claude-3-opus-20240229` | Requires `OPENROUTER_API_KEY` |

Override per-module via environment variables:

```bash
export VALRAVN_PLAN_MODEL=openai:gpt-4o
export VALRAVN_REFLECTOR_MODEL=openrouter:anthropic/claude-3-opus-20240229
```

---

## Checkpoint Database Cleanup

The SQLite checkpoint database enables crash recovery but can grow over time. Configure automatic cleanup:

```yaml
checkpoint_cleanup:
  retention_days: 7                    # Delete checkpoints older than N days
  max_checkpoints_per_thread: 100    # Keep at most N checkpoints per thread
  min_checkpoints_per_thread: 2      # Never delete below this threshold
  auto_cleanup: true                 # Run automatically after each investigation
  auto_vacuum: false                 # Reclaim disk space (slower)
```

**Why cleanup?** Each investigation thread writes a checkpoint after every node. With frequent investigations, this can accumulate gigabytes of state snapshots. The cleanup policy applies both time-based and count-based retention.

Manual cleanup example:

```python
from valravn.checkpoint_cleanup import cleanup_checkpoints

stats = cleanup_checkpoints(
    db_path="/path/to/checkpoints.db",
    retention_days=7,
    max_checkpoints=100
)
print(f"Deleted {stats['total_deleted']} checkpoints")
```
