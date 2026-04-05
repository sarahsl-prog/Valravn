# CLI Contract: `valravn`

**Branch**: `001-autonomous-dfir-agents` | **Date**: 2026-04-05

Defines the stable interface contract between the operator and the Valravn agent. Any
change to flags, exit codes, or output paths is a breaking change requiring a contract
version bump.

---

## Command

```
valravn investigate --prompt TEXT --evidence PATH [--config PATH] [--output-dir PATH]
```

### Arguments

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--prompt` | string | Yes | — | Natural-language investigation prompt (FR-001) |
| `--evidence` | path | Yes | — | Absolute path to evidence mount point (read-only; FR-007) |
| `--config` | path | No | `./config.yaml` | Path to RetryConfig YAML |
| `--output-dir` | path | No | `./` | Working directory for `analysis/`, `exports/`, `reports/` |

Multiple `--evidence` flags may be supplied for investigations spanning more than one
acquisition (v1 single-agent scope; each path validated as read-only at startup).

### Constraints

- `--evidence` path must exist and must not be writable by the process. If the path is
  writable, the agent refuses to start with exit code 2 (see Exit Codes below).
- `--output-dir` must not be a descendant of any `--evidence` path.

---

## Output Artifacts

All output is written to subdirectories of `--output-dir` (default: current directory):

| Path | Content | Format |
|------|---------|--------|
| `./analysis/checkpoints.db` | LangGraph `SqliteSaver` checkpoint — full `AgentState` after every node | SQLite |
| `./analysis/investigation_plan.json` | Live investigation plan, updated each step (human-readable audit copy) | JSON |
| `./analysis/<uuid>.json` | ToolInvocationRecord per tool execution | JSON |
| `./analysis/<uuid>.stdout` | Raw stdout of tool invocation | Plain text |
| `./analysis/<uuid>.stderr` | Raw stderr of tool invocation | Plain text |
| `./analysis/anomalies.json` | All anomalies detected during investigation | JSON |
| `./reports/<timestamp>_<slug>.md` | Final findings report (human-readable) | Markdown |
| `./reports/<timestamp>_<slug>.json` | Final findings report (machine-readable) | JSON |

`<timestamp>` format: `YYYYMMDD_HHMMSS` UTC.
`<slug>` format: first 40 characters of the prompt, lowercased, spaces replaced with `_`,
non-alphanumeric characters removed.

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Investigation completed; findings report written to `./reports/` |
| 1 | Investigation completed but with tool failures that could not be self-corrected; findings report still written (FR-013) |
| 2 | Pre-flight validation failure (writable evidence path, missing prompt, bad config) — no investigation started |
| 3 | Unrecoverable agent error (API failure, context exhaustion) — partial `./analysis/` may exist |

---

## Standard Streams

- **stdout**: Human-readable progress lines (one per major step). Suitable for operator
  monitoring. Not a contract surface — format may change between versions.
- **stderr**: Errors and warnings only.
- **Findings report**: The contract artifact. Always in `./reports/`. stdout/stderr are
  supplementary.

---

## Environment Variables

| Variable | Effect |
|----------|--------|
| `VALRAVN_MAX_RETRIES` | Override `retry.max_attempts` from config (integer) |
| `ANTHROPIC_API_KEY` | Required; Anthropic SDK reads this automatically |

---

## Example Invocations

```bash
# Memory investigation
valravn investigate \
  --prompt "Identify active network connections at time of acquisition" \
  --evidence /mnt/cases/001/memory.lime \
  --output-dir /cases/001/analysis

# Disk + memory multi-evidence
valravn investigate \
  --prompt "Determine lateral movement from host WORKSTATION-04" \
  --evidence /mnt/cases/002/disk.e01 \
  --evidence /mnt/cases/002/memory.raw \
  --config /cases/002/valravn.yaml \
  --output-dir /cases/002/analysis
```

---

## Contract Version

**v1.0.0** — Initial contract. Stable from first implementation merge.

Breaking changes (flag removal, exit code redefinition, output path changes) require a
MAJOR version bump and operator notification.
