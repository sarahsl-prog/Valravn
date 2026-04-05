from __future__ import annotations

import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from valravn.models.records import ToolInvocationRecord


def run_forensic_tool(state: dict) -> dict:
    """LangGraph node: invoke the current step's tool_cmd; capture stdout/stderr."""
    plan = state["plan"]
    step_id: str = state["current_step_id"]
    step = next(s for s in plan.steps if s.id == step_id)
    output_dir = Path(state.get("_output_dir", "."))

    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    inv_id = str(uuid.uuid4())
    stdout_path = analysis_dir / f"{inv_id}.stdout"
    stderr_path = analysis_dir / f"{inv_id}.stderr"

    started = datetime.now(timezone.utc)
    t0 = time.monotonic()

    proc = subprocess.run(
        step.tool_cmd,
        capture_output=True,
        text=True,
    )

    duration = time.monotonic() - t0
    completed = datetime.now(timezone.utc)

    stdout_path.write_text(proc.stdout)
    stderr_path.write_text(proc.stderr)

    rec = ToolInvocationRecord(
        id=inv_id,
        step_id=step_id,
        attempt_number=1,
        cmd=step.tool_cmd,
        exit_code=proc.returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        started_at_utc=started,
        completed_at_utc=completed,
        duration_seconds=duration,
        had_output=bool(proc.stdout.strip()),
    )

    invocations = list(state.get("invocations") or [])
    invocations.append(rec)
    step.invocation_ids.append(inv_id)

    return {
        "invocations": invocations,
        "plan": plan,
        "_step_succeeded": rec.success,
        "_last_invocation_id": inv_id,
    }
