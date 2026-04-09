from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.callbacks.base import BaseCallbackHandler
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from valravn.config import AppConfig, OutputConfig
from valravn.models.task import InvestigationPlan, InvestigationTask
from valravn.state import AgentState

# ---------------------------------------------------------------------------
# FileTracer — writes JSONL to ./analysis/traces/<run_id>.jsonl
# ---------------------------------------------------------------------------


class FileTracer(BaseCallbackHandler):
    def __init__(self, trace_path: Path) -> None:
        super().__init__()
        self._path = trace_path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, event: str, data: dict) -> None:
        line = json.dumps(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": event,
                **data,
            }
        )
        with open(self._path, "a") as f:
            f.write(line + "\n")

    def on_llm_start(self, serialized: dict, prompts: list[str], **kw: object) -> None:
        self._write("llm_start", {"model": serialized.get("name", ""), "prompts": prompts})

    def on_llm_end(self, response: object, **kw: object) -> None:
        self._write("llm_end", {})

    def on_tool_start(self, serialized: dict, input_str: str, **kw: object) -> None:
        self._write("tool_start", {"tool": serialized.get("name", ""), "input": input_str})

    def on_tool_end(self, output: object, **kw: object) -> None:
        self._write("tool_end", {"output": str(output)[:500]})


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def _build_graph(checkpointer: SqliteSaver) -> object:
    from valravn.nodes.anomaly import check_anomalies, record_anomaly
    from valravn.nodes.conclusions import synthesize_conclusions
    from valravn.nodes.plan import plan_investigation, update_plan
    from valravn.nodes.report import write_findings_report
    from valravn.nodes.self_assess import assess_progress
    from valravn.nodes.skill_loader import load_skill
    from valravn.nodes.tool_runner import run_forensic_tool

    def route_after_anomaly_check(state: AgentState) -> str:
        if state.get("_pending_anomalies"):
            return "record_anomaly"
        return "update_plan"

    def route_after_planning(state: AgentState) -> str:
        if state.get("current_step_id") is None:
            return "synthesize_conclusions"
        return "load_skill"

    def route_next_step(state: AgentState) -> str:
        if state["plan"].next_pending_step() is not None:
            return "load_skill"
        return "synthesize_conclusions"

    builder: StateGraph = StateGraph(AgentState)

    builder.add_node("plan_investigation", plan_investigation)
    builder.add_node("load_skill", load_skill)
    builder.add_node("assess_progress", assess_progress)
    builder.add_node("run_forensic_tool", run_forensic_tool)
    builder.add_node("check_anomalies", check_anomalies)
    builder.add_node("record_anomaly", record_anomaly)
    builder.add_node("update_plan", update_plan)
    builder.add_node("synthesize_conclusions", synthesize_conclusions)
    builder.add_node("write_findings_report", write_findings_report)

    builder.add_edge(START, "plan_investigation")
    builder.add_conditional_edges("plan_investigation", route_after_planning)
    builder.add_edge("load_skill", "assess_progress")
    builder.add_edge("assess_progress", "run_forensic_tool")
    builder.add_edge("run_forensic_tool", "check_anomalies")
    builder.add_conditional_edges("check_anomalies", route_after_anomaly_check)
    builder.add_edge("record_anomaly", "update_plan")
    builder.add_conditional_edges("update_plan", route_next_step)
    builder.add_edge("synthesize_conclusions", "write_findings_report")
    builder.add_edge("write_findings_report", END)

    return builder.compile(checkpointer=checkpointer)


def _build_success_trace(state: dict) -> str:
    """Build success trace from completed invocations."""
    invocations = state.get("invocations", [])
    lines = []
    for inv in invocations:
        if inv.exit_code == 0:
            lines.append(
                f"SUCCESS: {' '.join(str(c) for c in inv.cmd)} "
                f"-> exit_code={inv.exit_code} duration={inv.duration_seconds:.2f}s"
            )
    return "\n".join(lines)


def _build_failure_trace(state: dict) -> str:
    """Build failure trace from failed steps."""
    failures = state.get("_tool_failures", [])
    lines = []
    for failure in failures:
        lines.append(f"FAILURE: step={failure.step_id} error={failure.final_error}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run(task: InvestigationTask, app_cfg: AppConfig, out_cfg: OutputConfig) -> int:
    """Compile and invoke the investigation graph. Returns exit code (0 or 1)."""
    import sqlite3

    from loguru import logger

    # Validate output directory is writable
    output_dir = out_cfg.output_dir.resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        test_file = output_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
    except (PermissionError, OSError) as e:
        logger.error("Output directory {} is not writable: {}", output_dir, e)
        raise ValueError(
            f"Output directory '{output_dir}' is not writable. "
            f"Ensure you have write permissions or specify a different path."
        ) from e

    db_path = out_cfg.checkpoints_db
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    graph = _build_graph(checkpointer)

    run_id = task.id
    trace_path = out_cfg.traces_dir / f"{run_id}.jsonl"
    tracer = FileTracer(trace_path)

    initial_state: dict = {
        "task": task,
        "plan": InvestigationPlan(task_id=task.id),
        "invocations": [],
        "anomalies": [],
        "report": None,
        "current_step_id": None,
        "skill_cache": {},
        "messages": [],
        "_output_dir": str(out_cfg.output_dir),
        "_retry_config": {
            "max_attempts": app_cfg.retry.max_attempts,
            "retry_delay_seconds": app_cfg.retry.retry_delay_seconds,
            "timeout_seconds": app_cfg.retry.timeout_seconds,
        },
        "_step_succeeded": False,
        "_step_exhausted": False,
        "_pending_anomalies": False,
        "_conclusions": [],
        "_tool_failures": [],
        "_self_corrections": [],
        "_tool_failure": None,
        "_last_invocation_id": None,
        "_detected_anomaly_data": None,
        "_self_assessments": [],
        "_follow_up_steps": [],
        "_skills_config": app_cfg.skills,
    }

    config = {
        "configurable": {"thread_id": run_id},
        "callbacks": [tracer],
    }

    final_state = graph.invoke(initial_state, config=config)

    # RCL Integration
    if app_cfg.training.enabled:
        from valravn.training.rcl_loop import RCLTrainer

        trainer = RCLTrainer(app_cfg.training.state_dir)

        success_trace = _build_success_trace(final_state)
        failure_trace = _build_failure_trace(final_state)

        # Only process if we have meaningful data
        if len(failure_trace) >= app_cfg.training.min_failure_trace_length or success_trace:
            diagnostic = trainer.process_investigation_result(
                case_id=task.id,
                success_trace=success_trace,
                failure_trace=failure_trace,
                success=final_state.get("report") and not final_state["report"].tool_failures,
            )

            if diagnostic:
                logger.info(
                    "RCL diagnostic: attribution=%s root_cause=%s",
                    diagnostic.attribution,
                    diagnostic.root_cause,
                )

    # Checkpoint cleanup
    if app_cfg.checkpoint_cleanup.auto_cleanup:
        from valravn.checkpoint_cleanup import cleanup_checkpoints

        cleanup_result = cleanup_checkpoints(
            db_path=out_cfg.checkpoints_db, config=app_cfg.checkpoint_cleanup
        )
        if cleanup_result.deleted_count > 0:
            logger.info("Cleaned up %d old checkpoints", cleanup_result.deleted_count)

        if app_cfg.checkpoint_cleanup.auto_vacuum:
            import sqlite3

            conn = sqlite3.connect(str(out_cfg.checkpoints_db))
            conn.execute("VACUUM")
            conn.close()
            logger.debug("Vacuumed checkpoint database")

    if final_state.get("report"):
        return final_state["report"].exit_code
    return 1
