# Node Patterns Reference

## Existing Node Quick-Reference

| Node | File | Reads | Writes |
|------|------|-------|--------|
| `plan_investigation` | nodes/plan.py | task, _output_dir | plan, current_step_id |
| `load_skill` | nodes/skill_loader.py | current_step_id, plan, skill_cache | skill_cache, messages |
| `assess_progress` | nodes/self_assess.py | plan, current_step_id, invocations | _self_assessments, messages |
| `run_forensic_tool` | nodes/tool_runner.py | plan, current_step_id, task, _retry_config, _output_dir | invocations, plan, _step_succeeded, _step_exhausted, _last_invocation_id, _self_corrections, _tool_failure |
| `check_anomalies` | nodes/anomaly.py | invocations, _last_invocation_id, anomalies | anomalies, _pending_anomalies, _detected_anomaly_data |
| `record_anomaly` | nodes/anomaly.py | _detected_anomaly_data, anomalies | anomalies, _detected_anomaly_data, _pending_anomalies |
| `update_plan` | nodes/plan.py | plan, current_step_id, _step_succeeded, _step_exhausted, _tool_failure, _tool_failures, _follow_up_steps, _output_dir | plan, current_step_id, _step_succeeded, _step_exhausted, _pending_anomalies, _tool_failure, _tool_failures, _follow_up_steps |
| `synthesize_conclusions` | nodes/conclusions.py | task, invocations, anomalies, plan | _conclusions, messages |
| `write_findings_report` | nodes/report.py | task, plan, invocations, anomalies, _conclusions, _tool_failures, _self_corrections, _self_assessments, _output_dir | report |

## Retry Pattern

See `nodes/tool_runner.py` for the canonical retry loop with LLM-based self-correction. For nodes that call external tools:
- Read `_retry_config` from state for `max_attempts` / `timeout_seconds`
- On failure: call correction LLM, update command, retry
- On exhaustion: populate `ToolFailureRecord`, set `_step_exhausted=True`

## Follow-up Step Pattern

Nodes can inject additional investigation steps by appending to `_follow_up_steps`:
```python
return {
    "_follow_up_steps": [PlannedStep(skill_domain="memory-analysis", tool_cmd=[...], rationale="...")],
}
```
`update_plan` will call `plan.add_steps()` to absorb them.

## File Output Pattern

All node outputs (stdout, records, plan JSON) go under `output_dir / "analysis/"`:
```python
output_dir = Path(state.get("_output_dir", "."))
analysis_dir = output_dir / "analysis"
analysis_dir.mkdir(parents=True, exist_ok=True)
out_path = analysis_dir / f"{unique_id}.json"
out_path.write_text(data.model_dump_json(indent=2))
```

## Skill Cache Pattern

The `skill_cache` dict maps domain name → SKILL.md content string. `load_skill` populates it before `assess_progress` uses it. New nodes that need DFIR domain knowledge can read from `skill_cache`.

## Training Module

`src/valravn/training/` contains self-improvement components:
- `feasibility.py` — pre-execution safety check (blocks destructive/invalid commands)
- `playbook.py` — learned successful command sequences
- `reflector.py` — post-execution reflection on failures
- `rcl_loop.py` — reward-calibrated learning loop

New nodes that involve tool execution should call `FeasibilityMemory.check()` before executing.
