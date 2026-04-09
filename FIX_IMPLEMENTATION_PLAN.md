# Valravn Issue Fix Implementation Plan

**Created:** 2026-04-09  
**Target Version:** 0.2.0  
**Estimated Effort:** 3-4 days

---

## Priority Matrix

| Priority | Issue | Effort | Risk | Dependencies |
|----------|-------|--------|------|--------------|
| P0 | #2 Default model name | 10 min | Low | None |
| P0 | #9 Vol3 command typo | 10 min | Low | None |
| P1 | #1 Self-assessment decision | 4-8 hrs | Medium | None |
| P1 | #5 Feasibility type fix | 30 min | Low | None |
| P2 | #6 Output dir validation | 30 min | Low | None |
| P2 | #10 Checkpoint config | 2-3 hrs | Medium | None |
| P3 | #3 RCL integration | 6-8 hrs | High | #10 (config) |
| P3 | #4 Skill paths config | 1-2 hrs | Low | None |
| P4 | #7 Init exports | 30 min | Low | None |
| P4 | #8 Test docstring | 5 min | Low | None |

---

## Phase 1: Critical Fixes (P0) - Day 1 Morning

### Issue #2: Fix Default Model Names
**Files:** `src/valravn/core/llm_factory.py`

**Current Problem:**
```python
DEFAULT_MODELS = {
    "anomaly": ["ollama:minimax-m2.5:cloud"],  # Invalid model name
    ...
}
```

**Implementation Steps:**
1. Replace all instances of `ollama:minimax-m2.5:cloud` with `ollama:llama3.2`
2. Update `CLAUDE.md` to reflect the new default
3. Verify fallback chains work correctly

**Testing:**
- Unit test: Verify `get_default_model()` returns expected value
- Integration test: Mock Ollama unavailable, verify fallback to next model

**Code Change:**
```python
DEFAULT_MODELS: dict[str, list[str]] = {
    "anomaly": ["ollama:llama3.2"],
    "conclusions": ["ollama:llama3.2"],
    "plan": ["ollama:llama3.2"],
    "self_assess": ["ollama:llama3.2"],
    "tool_runner": ["ollama:llama3.2"],
    "reflector": ["ollama:llama3.2"],
    "mutator": ["ollama:llama3.2"],
}
```

---

### Issue #9: Fix Volatility Command Template
**Files:** `src/valravn/nodes/anomaly.py`

**Current Problem:**
```python
"orphaned_relationship": {
    "tool_cmd_template": ["vol3", "-f", "{evidence}", "pstree"],  # Wrong command
}
```

**Implementation Steps:**
1. Update command template to match SIFT installation
2. Verify other command templates are correct

**Testing:**
- Unit test: Verify command template substitution produces valid command
- Check other _FOLLOW_UP_COMMANDS entries for consistency

**Code Change:**
```python
"orphaned_relationship": {
    "skill_domain": "memory-analysis",
    "tool_cmd_template": [
        "python3", "/opt/volatility3-2.20.0/vol.py",
        "-f", "{evidence}", "windows.pstree.PsTree"
    ],
}
```

---

## Phase 2: Architecture Decision (P1) - Day 1 Afternoon

### Issue #1: Self-Assessment Integration Decision

**Decision Required:** Should self-assessment:
- **Option A:** Be removed entirely (reduces LLM calls, simplifies graph)
- **Option B:** Be integrated to influence execution (adds complexity, enables adaptive behavior)

**Recommendation:** Option B - integrate with anomaly detection

**Rationale:**
- The infrastructure exists and is tested
- Integration provides value: trust-based anomaly sensitivity
- Removal would discard working code

**Implementation (if Option B selected):**

**Files Modified:**
- `src/valravn/nodes/anomaly.py` - Add trust coefficient check
- `src/valravn/nodes/self_assess.py` - Export trust_coefficient function
- `src/valravn/graph.py` - Pass trust state

**Code Changes:**

1. Export trust_coefficient from self_guide.py:
```python
# In self_guide.py
__all__ = ["POLARITY_REWARD_MAP", "SelfGuidanceSignal", "trust_coefficient"]
```

2. Modify check_anomalies to use trust:
```python
# In anomaly.py
def check_anomalies(state: dict) -> dict:
    from valravn.training.self_guide import trust_coefficient
    from valravn.models.task import StepStatus
    
    invocations = state.get("invocations") or []
    if not invocations:
        return {"_pending_anomalies": False, "_detected_anomaly_data": None}

    # Calculate trust coefficient
    plan = state.get("plan")
    if plan and len(plan.steps) > 0:
        completed = sum(1 for s in plan.steps if s.status != StepStatus.PENDING)
        trust = trust_coefficient(completed, len(plan.steps))
    else:
        trust = 0.0

    # ... existing anomaly detection code ...
    
    if result.anomaly_detected:
        # Require higher confidence when trust is low
        if trust < 0.3 and result.forensic_significance not in ["critical", "high"]:
            logger.debug("Low trust ({}) filtered non-critical anomaly", trust)
            return {"_pending_anomalies": False, "_detected_anomaly_data": None}
        
        return {"_pending_anomalies": True, "_detected_anomaly_data": result.model_dump()}

    return {"_pending_anomalies": False, "_detected_anomaly_data": None}
```

**Testing:**
- Unit test: Low trust filters non-critical anomaly
- Unit test: High trust allows all anomalies
- Unit test: Critical anomalies bypass trust filter

**Alternative (if Option A selected - Remove):**

**Files Modified:**
- `src/valravn/graph.py` - Remove assess_progress node and edges
- `src/valravn/nodes/self_assess.py` - Delete file
- `src/valravn/training/self_guide.py` - Consider keeping for future use
- `src/valravn/state.py` - Remove `_self_assessments` field
- `tests/unit/` - Remove test_self_assess.py

**Code Changes:**
```python
# In graph.py - remove:
# builder.add_node("assess_progress", assess_progress)
# builder.add_edge("load_skill", "assess_progress")
# builder.add_edge("assess_progress", "run_forensic_tool")
# Replace with:
builder.add_edge("load_skill", "run_forensic_tool")
```

---

## Phase 3: Code Quality Fixes (P1-P2) - Day 2

### Issue #5: Fix Feasibility Check Parameter Types
**Files:** `src/valravn/training/feasibility.py`

**Implementation Steps:**
1. Update `_check_evidence_protection` signature to accept `list[str]`
2. Remove comma-join/split logic
3. Update all callers

**Code Changes:**
```python
class FeasibilityMemory:
    def _check_evidence_protection(
        self, cmd: list[str], evidence_refs: list[str]
    ) -> tuple[bool, str]:
        """Check that destructive commands do not target evidence paths."""
        if not cmd or not evidence_refs:
            return True, ""
        executable = Path(cmd[0]).name
        if executable not in self._DESTRUCTIVE_CMDS:
            return True, ""
        # evidence_refs is already a list
        for arg in cmd:
            for ev_path in evidence_refs:
                ev_path = ev_path.strip()
                if ev_path and arg.startswith(ev_path):
                    return False, (
                        f"Destructive command '{executable}' targets "
                        f"evidence path {ev_path}"
                    )
        return True, ""
    
    def check(
        self, cmd: list[str], evidence_refs: list[str], output_dir: str
    ) -> tuple[bool, list[str]]:
        """Validate command against all feasibility rules."""
        violations: list[str] = []
        
        for rule in self.rules:
            try:
                # Pass list directly, no comma-join needed
                result, msg = rule.check_fn(cmd, evidence_refs)
                if not result:
                    violations.append(
                        f"[{rule.rule_id}] {rule.description}: {msg}"
                    )
            except Exception as e:
                logger.warning(
                    "Feasibility rule {!r} raised exception: {}",
                    rule.rule_id, e
                )
        
        return len(violations) == 0, violations
```

**Testing:**
- Unit test: Verify list is passed correctly
- Unit test: Evidence protection still works
- Unit test: No comma-split behavior

---

### Issue #6: Add Output Directory Validation
**Files:** `src/valravn/graph.py`

**Implementation Steps:**
1. Add validation at start of `run()` function
2. Provide clear error message
3. Create output directories if they don't exist

**Code Changes:**
```python
def run(task: InvestigationTask, app_cfg: AppConfig, out_cfg: OutputConfig) -> int:
    """Compile and invoke the investigation graph. Returns exit code (0 or 1)."""
    # Validate output directory is writable
    output_dir = out_cfg.output_dir.resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        test_file = output_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
    except (PermissionError, OSError) as e:
        logger.error(
            "Output directory {} is not writable: {}",
            output_dir, e
        )
        raise ValueError(
            f"Output directory '{output_dir}' is not writable. "
            f"Ensure you have write permissions or specify a different path."
        ) from e
    
    db_path = out_cfg.checkpoints_db
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # ... rest of function
```

**Testing:**
- Unit test: Valid directory passes
- Unit test: Non-existent directory is created
- Unit test: Read-only directory raises ValueError
- Unit test: File (not directory) raises ValueError

---

### Issue #10: Implement Checkpoint Cleanup Configuration
**Files:**
- `src/valravn/config.py`
- `src/valravn/checkpoint_cleanup.py`
- `src/valravn/graph.py`

**Implementation Steps:**
1. Create CheckpointCleanupConfig model
2. Add to AppConfig
3. Modify checkpoint_cleanup.py to use config values
4. Call cleanup after investigation if auto_cleanup enabled

**Code Changes:**

1. In config.py:
```python
from typing import Optional

class CheckpointCleanupConfig(BaseModel):
    """Configuration for SQLite checkpoint database cleanup."""
    retention_days: int = 7
    max_checkpoints_per_thread: int = 1000
    min_checkpoints_per_thread: int = 2
    auto_cleanup: bool = True
    auto_vacuum: bool = False

class AppConfig(BaseModel):
    retry: RetryConfig = RetryConfig()
    mlflow: dict = {
        "tracking_uri": "http://127.0.0.1:5000",
        "experiment_name": "valravn-evaluation",
    }
    models: dict[str, str | list[str]] = {}
    checkpoint_cleanup: CheckpointCleanupConfig = CheckpointCleanupConfig()
```

2. In checkpoint_cleanup.py:
```python
def cleanup_checkpoints(
    db_path: Path,
    config: CheckpointCleanupConfig | None = None
) -> CleanupResult:
    """Clean up old checkpoints from the SQLite database."""
    if config is None:
        config = CheckpointCleanupConfig()
    
    # Use config values instead of hardcoded defaults
    retention_cutoff = datetime.now(timezone.utc) - timedelta(
        days=config.retention_days
    )
    # ... rest of implementation using config values
```

3. In graph.py after investigation:
```python
# After graph execution
if app_cfg.checkpoint_cleanup.auto_cleanup:
    from valravn.checkpoint_cleanup import cleanup_checkpoints
    cleanup_result = cleanup_checkpoints(
        db_path=out_cfg.checkpoints_db,
        config=app_cfg.checkpoint_cleanup
    )
    if cleanup_result.deleted_count > 0:
        logger.info(
            "Cleaned up {} old checkpoints",
            cleanup_result.deleted_count
        )
    
    if app_cfg.checkpoint_cleanup.auto_vacuum:
        # Vacuum the database
        conn = sqlite3.connect(str(out_cfg.checkpoints_db))
        conn.execute("VACUUM")
        conn.close()
```

**Testing:**
- Unit test: Config values are loaded from YAML
- Unit test: Cleanup respects retention_days
- Unit test: Cleanup respects max_checkpoints_per_thread
- Unit test: Cleanup never goes below min_checkpoints_per_thread
- Unit test: Auto-cleanup runs after investigation

---

## Phase 4: Feature Integration (P3) - Days 3-4

### Issue #3: Integrate RCL Training System
**Files:**
- `src/valravn/config.py` (add training config)
- `src/valravn/graph.py` (call RCL after investigation)
- `src/valravn/cli.py` (optional: add training command)

**Implementation Steps:**
1. Add training configuration section
2. Create trace builder functions
3. Call RCLTrainer after investigation completes
4. Add CLI command for training management

**Code Changes:**

1. In config.py:
```python
class TrainingConfig(BaseModel):
    """Configuration for RCL training system."""
    enabled: bool = False  # Opt-in by default
    state_dir: Path = Path("./training")
    min_failure_trace_length: int = 100  # Don't train on trivial failures

class AppConfig(BaseModel):
    # ... existing fields ...
    training: TrainingConfig = TrainingConfig()
```

2. In graph.py:
```python
def _build_success_trace(state: dict) -> str:
    """Build success trace from completed invocations."""
    invocations = state.get("invocations", [])
    lines = []
    for inv in invocations:
        if inv.exit_code == 0:
            lines.append(f"SUCCESS: {' '.join(inv.cmd)} -> exit_code={inv.exit_code}")
    return "\n".join(lines)

def _build_failure_trace(state: dict) -> str:
    """Build failure trace from failed steps."""
    failures = state.get("_tool_failures", [])
    lines = []
    for failure in failures:
        lines.append(
            f"FAILURE: step={failure.step_id} error={failure.final_error}"
        )
    return "\n".join(lines)

def run(task: InvestigationTask, app_cfg: AppConfig, out_cfg: OutputConfig) -> int:
    # ... existing code ...
    
    final_state = graph.invoke(initial_state, config=config)
    
    # RCL Integration
    if app_cfg.training.enabled:
        from valravn.training.rcl_loop import RCLTrainer
        
        trainer = RCLTrainer(app_cfg.training.state_dir)
        
        success_trace = _build_success_trace(final_state)
        failure_trace = _build_failure_trace(final_state)
        
        # Only process if we have meaningful data
        if (
            len(failure_trace) >= app_cfg.training.min_failure_trace_length
            or success_trace
        ):
            diagnostic = trainer.process_investigation_result(
                case_id=task.id,
                success_trace=success_trace,
                failure_trace=failure_trace,
                success=final_state.get("report")
                and not final_state["report"].tool_failures,
            )
            
            if diagnostic:
                logger.info(
                    "RCL diagnostic: attribution={} root_cause={}",
                    diagnostic.attribution,
                    diagnostic.root_cause
                )
    
    if final_state.get("report"):
        return final_state["report"].exit_code
    return 1
```

**Testing:**
- Unit test: RCL trainer instantiated with correct state_dir
- Unit test: Success/failure traces are built correctly
- Unit test: Training only runs when enabled in config
- Unit test: Diagnostic is logged when available
- Integration test: End-to-end with mock RCL components

---

### Issue #4: Make Skill Paths Configurable
**Files:**
- `src/valravn/config.py`
- `src/valravn/nodes/skill_loader.py`
- `src/valravn/state.py` (pass config to state)

**Implementation Steps:**
1. Add skills configuration to AppConfig
2. Modify skill_loader to use config
3. Pass config through state

**Code Changes:**

1. In config.py:
```python
class SkillsConfig(BaseModel):
    """Configuration for skill file paths."""
    base_path: Path = Path.home() / ".claude" / "skills"
    
    def get_skill_path(self, domain: str) -> Path | None:
        """Get the path to a skill file for a given domain."""
        known_domains = {
            "memory-analysis": "memory-analysis/SKILL.md",
            "sleuthkit": "sleuthkit/SKILL.md",
            "windows-artifacts": "windows-artifacts/SKILL.md",
            "plaso-timeline": "plaso-timeline/SKILL.md",
            "yara-hunting": "yara-hunting/SKILL.md",
        }
        if domain not in known_domains:
            return None
        return self.base_path / known_domains[domain]
```

2. In skill_loader.py:
```python
def load_skill(state: dict) -> dict:
    """LangGraph node: load SKILL.md for the current step's domain into skill_cache."""
    step_id = state["current_step_id"]
    step = next(s for s in state["plan"].steps if s.id == step_id)
    logger.info("Node: load_skill | domain={} step={}", step.skill_domain, step_id[:8])
    domain = step.skill_domain

    cache: dict[str, str] = dict(state.get("skill_cache") or {})

    if domain in cache:
        return {"skill_cache": cache}

    # Get config from state
    skills_config = state.get("_skills_config")
    if skills_config is None:
        from valravn.config import SkillsConfig
        skills_config = SkillsConfig()
    
    skill_path = skills_config.get_skill_path(domain)
    if skill_path is None:
        raise SkillNotFoundError(
            f"No skill file registered for domain '{domain}'. "
            f"Add it to SkillsConfig.known_domains."
        )
    
    if not skill_path.exists():
        raise SkillNotFoundError(f"Skill file not found: {skill_path}")

    cache[domain] = skill_path.read_text()
    return {"skill_cache": cache}
```

3. In state.py add:
```python
class AgentState(TypedDict):
    # ... existing fields ...
    _skills_config: SkillsConfig | None  # Pass skills config through state
```

4. In graph.py initial_state:
```python
initial_state: dict = {
    # ... existing fields ...
    "_skills_config": app_cfg.skills if hasattr(app_cfg, "skills") else None,
}
```

**Testing:**
- Unit test: Default skills path works
- Unit test: Custom base_path is respected
- Unit test: Unknown domain returns None
- Unit test: Skill loading fails gracefully when path doesn't exist

---

## Phase 5: Polish (P4) - Day 4 Afternoon

### Issue #7: Add __init__.py Exports
**Files:**
- `src/valravn/__init__.py`
- `src/valravn/models/__init__.py`
- `src/valravn/nodes/__init__.py`
- `src/valravn/evaluation/__init__.py`

**Implementation:**
```python
# src/valravn/models/__init__.py
from valravn.models.task import (
    InvestigationTask,
    InvestigationPlan,
    PlannedStep,
    StepStatus,
)
from valravn.models.records import (
    ToolInvocationRecord,
    Anomaly,
    AnomalyResponseAction,
)
from valravn.models.report import (
    FindingsReport,
    Conclusion,
    ToolFailureRecord,
    SelfCorrectionEvent,
)

__all__ = [
    # task.py
    "InvestigationTask",
    "InvestigationPlan",
    "PlannedStep",
    "StepStatus",
    # records.py
    "ToolInvocationRecord",
    "Anomaly",
    "AnomalyResponseAction",
    # report.py
    "FindingsReport",
    "Conclusion",
    "ToolFailureRecord",
    "SelfCorrectionEvent",
]
```

---

### Issue #8: Fix Test Docstring
**Files:** `tests/unit/test_models.py`

**Code Change:**
```python
def test_investigation_task_rejects_any_writable_evidence(tmp_path):
    """InvestigationTask rejects when any evidence path is writable."""
```

---

## Testing Strategy

### Unit Tests to Add/Modify

1. **test_llm_factory.py**
   - `test_default_models_are_valid_names()`
   - `test_default_models_use_expected_values()`

2. **test_anomaly.py**
   - `test_orphaned_relationship_uses_correct_volatility_command()`
   - `test_trust_coefficient_filters_non_critical_anomalies()`
   - `test_critical_anomalies_bypass_trust_filter()`

3. **test_feasibility.py**
   - `test_check_accepts_list_not_string()`
   - `test_evidence_protection_with_list_input()`

4. **test_graph.py**
   - `test_output_directory_validation()`
   - `test_nonexistent_output_dir_is_created()`
   - `test_unwritable_output_dir_raises_error()`

5. **test_config.py**
   - `test_checkpoint_cleanup_config_defaults()`
   - `test_checkpoint_cleanup_config_from_yaml()`
   - `test_training_config_defaults()`
   - `test_skills_config_defaults()`

6. **test_rcl_integration.py** (new file)
   - `test_rcl_trainer_not_called_when_disabled()`
   - `test_rcl_trainer_called_when_enabled()`
   - `test_success_trace_building()`
   - `test_failure_trace_building()`

### Integration Tests

1. **test_end_to_end_with_anomalies.py**
   - Full flow with anomaly detection and follow-ups
   - Verify trust coefficient affects behavior

2. **test_checkpoint_cleanup_integration.py**
   - Verify cleanup runs after investigation
   - Verify vacuum when enabled

### Manual Testing Checklist

- [ ] Run investigation with default config (Ollama)
- [ ] Verify anomaly follow-up uses correct vol.py command
- [ ] Verify output directory validation works
- [ ] Verify checkpoint cleanup runs
- [ ] Test with RCL enabled in config
- [ ] Test with custom skills path

---

## Rollback Plan

If issues arise:

1. **Phase 1 fixes** (model names, vol3 fix) are safe - minimal risk
2. **Self-assessment decision** - Can be deferred; current code works
3. **Type fixes** - Revert to original if issues; low impact
4. **RCL integration** - Can be disabled via config
5. **Config additions** - New fields have defaults; backward compatible

**Emergency rollback:**
```bash
git checkout -- src/valravn/core/llm_factory.py
# For other files as needed
```

---

## Design Decisions Needed

### Question 1: Self-Assessment Integration
**Options:**
- A) Remove entirely (simpler, fewer LLM calls)
- B) Integrate with anomaly detection (adds adaptive behavior)

**Recommendation:** B) Integrate with trust-based filtering

### Question 2: RCL Training Default
**Options:**
- A) Enabled by default (active learning)
- B) Disabled by default (opt-in feature)

**Recommendation:** B) Disabled by default - less surprising for users

### Question 3: Checkpoint Cleanup Default
**Options:**
- A) Auto-cleanup enabled (saves disk space)
- B) Auto-cleanup disabled (preserves history)

**Recommendation:** A) Enabled with conservative defaults (7 days, 1000 checkpoints)

---

## Definition of Done

- [ ] All 10 issues addressed
- [ ] Unit test coverage maintained or improved
- [ ] Integration tests pass
- [ ] Documentation updated (CLAUDE.md, README.md if needed)
- [ ] CHANGELOG.md entry added
- [ ] Version bumped to 0.2.0
- [ ] Manual testing completed
