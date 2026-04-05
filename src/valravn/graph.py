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
        line = json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **data,
        })
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
    from valravn.nodes.plan import plan_investigation, update_plan
    from valravn.nodes.report import write_findings_report
    from valravn.nodes.skill_loader import load_skill
    from valravn.nodes.tool_runner import run_forensic_tool

    def route_after_anomaly_check(state: AgentState) -> str:
        if state.get("_pending_anomalies"):
            return "record_anomaly"
        return "update_plan"

    def route_next_step(state: AgentState) -> str:
        if state["plan"].next_pending_step() is not None:
            return "load_skill"
        return "write_findings_report"

    builder: StateGraph = StateGraph(AgentState)

    builder.add_node("plan_investigation", plan_investigation)
    builder.add_node("load_skill", load_skill)
    builder.add_node("run_forensic_tool", run_forensic_tool)
    builder.add_node("check_anomalies", check_anomalies)
    builder.add_node("record_anomaly", record_anomaly)
    builder.add_node("update_plan", update_plan)
    builder.add_node("write_findings_report", write_findings_report)

    builder.add_edge(START, "plan_investigation")
    builder.add_edge("plan_investigation", "load_skill")
    builder.add_edge("load_skill", "run_forensic_tool")
    builder.add_edge("run_forensic_tool", "check_anomalies")
    builder.add_conditional_edges("check_anomalies", route_after_anomaly_check)
    builder.add_edge("record_anomaly", "update_plan")
    builder.add_conditional_edges("update_plan", route_next_step)
    builder.add_edge("write_findings_report", END)

    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(task: InvestigationTask, app_cfg: AppConfig, out_cfg: OutputConfig) -> int:
    """Compile and invoke the investigation graph. Returns exit code (0 or 1)."""
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
    }

    config = {
        "configurable": {"thread_id": run_id},
        "callbacks": [tracer],
    }

    final_state = graph.invoke(initial_state, config=config)

    if final_state.get("report"):
        return final_state["report"].exit_code
    return 1
