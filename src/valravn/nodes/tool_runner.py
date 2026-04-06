from __future__ import annotations

import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from valravn.models.records import ToolInvocationRecord
from valravn.models.report import SelfCorrectionEvent, ToolFailureRecord


class _CorrectionSpec(BaseModel):
    corrected_cmd: list[str]
    rationale: str


def _get_correction_llm():
    from langchain_anthropic import ChatAnthropic

    llm = ChatAnthropic(model="claude-opus-4-6", temperature=0)
    return llm.with_structured_output(_CorrectionSpec)


def _request_correction(
    step_id: str,
    attempt_number: int,
    original_cmd: list[str],
    exit_code: int,
    stderr: str,
) -> _CorrectionSpec:
    prompt = (
        f"A forensic tool invocation failed on attempt {attempt_number}.\n\n"
        f"Original command: {original_cmd}\n"
        f"Exit code: {exit_code}\n"
        f"Stderr output:\n{stderr}\n\n"
        "Produce a corrected argv list that will resolve the failure. "
        "Return only the corrected command and your rationale."
    )
    llm = _get_correction_llm()
    return llm.invoke(prompt)


def _run_single_attempt(
    step,
    step_id: str,
    analysis_dir: Path,
    timeout_seconds: int = 3600,
) -> tuple[ToolInvocationRecord, subprocess.CompletedProcess]:
    inv_id = str(uuid.uuid4())
    stdout_path = analysis_dir / f"{inv_id}.stdout"
    stderr_path = analysis_dir / f"{inv_id}.stderr"

    started = datetime.now(timezone.utc)
    t0 = time.monotonic()

    try:
        proc = subprocess.run(
            step.tool_cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as e:
        # Build a synthetic failed process result
        proc = subprocess.CompletedProcess(
            args=step.tool_cmd,
            returncode=-1,
            stdout=e.output.decode() if e.output else "",
            stderr=f"Tool timed out after {timeout_seconds}s",
        )

    duration = time.monotonic() - t0
    completed = datetime.now(timezone.utc)

    stdout_path.write_text(proc.stdout)
    stderr_path.write_text(proc.stderr)

    rec = ToolInvocationRecord(
        id=inv_id,
        step_id=step_id,
        attempt_number=len(step.invocation_ids) + 1,
        cmd=step.tool_cmd,
        exit_code=proc.returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        started_at_utc=started,
        completed_at_utc=completed,
        duration_seconds=duration,
        had_output=bool(proc.stdout.strip()),
    )

    rec_path = analysis_dir / f"{inv_id}.record.json"
    rec_path.write_text(rec.model_dump_json(indent=2))

    return rec, proc


def run_forensic_tool(state: dict) -> dict:
    """LangGraph node: invoke the current step's tool_cmd with retry and self-correction."""
    plan = state["plan"]
    step_id: str = state["current_step_id"]
    step = next((s for s in plan.steps if s.id == step_id), None)
    if step is None:
        raise ValueError(f"Step {step_id!r} not found in plan")
    output_dir = Path(state.get("_output_dir", "."))

    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # FR-007: validate output paths before any attempt (check once with a probe path)
    probe_id = str(uuid.uuid4())
    probe_stdout = analysis_dir / f"{probe_id}.stdout"
    probe_stderr = analysis_dir / f"{probe_id}.stderr"
    evidence_refs = state["task"].evidence_refs
    for ev in evidence_refs:
        ev_path = Path(ev).resolve()
        ev_dir = ev_path if ev_path.is_dir() else ev_path.parent
        out_under_ev = probe_stdout.resolve().is_relative_to(ev_dir)
        err_under_ev = probe_stderr.resolve().is_relative_to(ev_dir)
        if out_under_ev or err_under_ev:
            raise ValueError(
                f"Output path {analysis_dir} is under evidence directory {ev_dir}. "
                "Set --output-dir to a non-evidence location."
            )

    retry_cfg = state.get("_retry_config") or {}
    max_attempts: int = retry_cfg.get("max_attempts", 3)
    timeout_seconds: int = retry_cfg.get("timeout_seconds", 3600)
    retry_delay: float = retry_cfg.get("retry_delay_seconds", 0.0)

    invocations: list[ToolInvocationRecord] = list(state.get("invocations") or [])
    self_corrections: list[SelfCorrectionEvent] = list(state.get("_self_corrections") or [])

    last_inv_id: str | None = None
    tool_failure: ToolFailureRecord | None = None
    step_succeeded = False
    step_exhausted = False

    for attempt in range(1, max_attempts + 1):
        rec, proc = _run_single_attempt(step, step_id, analysis_dir, timeout_seconds)
        invocations.append(rec)
        step.invocation_ids.append(rec.id)
        last_inv_id = rec.id

        if rec.success:
            step_succeeded = True
            break

        # Failed attempt
        is_last = attempt == max_attempts
        if not is_last:
            # Ask Claude for a corrected command
            correction = _request_correction(
                step_id=step_id,
                attempt_number=attempt,
                original_cmd=list(rec.cmd),
                exit_code=proc.returncode,
                stderr=proc.stderr,
            )
            event = SelfCorrectionEvent(
                step_id=step_id,
                attempt_number=attempt,
                original_cmd=list(rec.cmd),
                corrected_cmd=correction.corrected_cmd,
                correction_rationale=correction.rationale,
            )
            self_corrections.append(event)
            step.tool_cmd = correction.corrected_cmd
            if retry_delay > 0:
                time.sleep(retry_delay)
        else:
            # Exhausted all attempts
            step_exhausted = True
            first_stderr_line = (
                proc.stderr.splitlines()[0] if proc.stderr.strip() else "no error message"
            )
            tool_failure = ToolFailureRecord(
                step_id=step_id,
                invocation_ids=list(step.invocation_ids),
                final_error=f"exit_code={rec.exit_code}: {first_stderr_line}",
                diagnostic_context=proc.stderr,
            )

    return {
        "invocations": invocations,
        "plan": plan,
        "_step_succeeded": step_succeeded,
        "_step_exhausted": step_exhausted,
        "_last_invocation_id": last_inv_id,
        "_self_corrections": self_corrections,
        "_tool_failure": tool_failure,
    }
