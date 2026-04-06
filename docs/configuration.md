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
| `ANTHROPIC_API_KEY` | **Required.** Anthropic API key for Claude model access. | — |
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
