from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from valravn.core.llm_factory import get_llm
from valravn.core.parsing import parse_llm_json

from valravn.models.records import ToolInvocationRecord
from valravn.models.report import SelfCorrectionEvent, ToolFailureRecord


class _CorrectionSpec(BaseModel):
    corrected_cmd: list[str]
    rationale: str


def _get_correction_llm():
    """Get LLM for tool command correction."""
    return get_llm(module="tool_runner")


_CORRECTION_CONTEXT = """\
You are correcting a forensic tool command on a SANS SIFT Ubuntu workstation.
Key tool syntax rules:

log2timeline.py (Plaso 20240308):
  ALL flags must come before the source path. Source path is the LAST argument.
  ONLY use flags from `log2timeline.py --help`. Do NOT invent flags.
  Non-existent flags (will error): --log2timeline-mode, --mode, --output, --format, --output-format
  CORRECT: log2timeline.py --storage-file ./analysis/out.plaso --parsers win10 --timezone UTC /mnt/ewf/ewf1
  WRONG:   log2timeline.py /mnt/ewf/ewf1 --storage-file ./analysis/out.plaso

fls: fls [-r] [-m /] <image_or_device>
icat: icat <image_or_device> <inode>
vol.py: python3 /opt/volatility3-2.20.0/vol.py -f <image> <plugin>
yara: /usr/local/bin/yara [-r] <rules.yar> <target>
"""


def _request_correction(
    step_id: str,
    attempt_number: int,
    original_cmd: list[str],
    exit_code: int,
    stderr: str,
    analysis_dir: Path | None = None,
) -> _CorrectionSpec:
    output_note = (
        f"ALL output files must go to: {analysis_dir}\n"
        if analysis_dir else ""
    )
    prompt = (
        f"{_CORRECTION_CONTEXT}\n"
        f"{output_note}"
        f"A forensic tool invocation failed on attempt {attempt_number}.\n\n"
        f"Original command: {original_cmd}\n"
        f"Exit code: {exit_code}\n"
        f"Stderr output:\n{stderr}\n\n"
        "Produce a corrected argv list that will resolve the failure.\n"
        "Respond with valid JSON only — no markdown, no prose outside the JSON object.\n"
        'Output exactly: {"corrected_cmd": ["<executable>", "<arg1>", ...], "rationale": "<why>"}'
    )
    llm = _get_correction_llm()
    response = llm.invoke(prompt)
    return parse_llm_json(response.content, _CorrectionSpec)


def _run_with_streaming(
    cmd: list[str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: int,
) -> tuple[int, bool]:
    """Run a subprocess with real-time stderr streaming to the terminal.

    Stdout is written directly to file (forensic output, often large).
    Stderr is streamed to both a file and the terminal so progress bars
    and status lines from tools like log2timeline.py are visible.

    Returns:
        (returncode, had_stdout) tuple.

    Raises:
        subprocess.TimeoutExpired: if the process exceeds timeout_seconds.
    """
    with open(stdout_path, "w") as stdout_f, open(stderr_path, "wb") as stderr_f:
        proc = subprocess.Popen(
            cmd,
            stdout=stdout_f,
            stderr=subprocess.PIPE,
        )

        # Stream stderr to file + terminal in real-time.
        # Use os.read() on the raw fd so we get whatever bytes are available
        # immediately — this keeps progress bars and \r-based status lines
        # responsive rather than buffering until a newline.
        fd = proc.stderr.fileno()
        deadline = time.monotonic() + timeout_seconds

        while True:
            if time.monotonic() > deadline:
                proc.kill()
                proc.wait()
                raise subprocess.TimeoutExpired(cmd, timeout_seconds)

            try:
                chunk = os.read(fd, 4096)
            except OSError:
                break
            if not chunk:
                break

            stderr_f.write(chunk)
            sys.stderr.buffer.write(chunk)
            sys.stderr.buffer.flush()

        proc.wait()

    had_stdout = stdout_path.stat().st_size > 0
    return proc.returncode, had_stdout


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
        returncode, had_stdout = _run_with_streaming(
            step.tool_cmd, stdout_path, stderr_path, timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        stderr_path.write_text(f"Tool timed out after {timeout_seconds}s")
        if not stdout_path.exists():
            stdout_path.write_text("")
        returncode = -1
        had_stdout = False

    duration = time.monotonic() - t0
    completed = datetime.now(timezone.utc)

    # Read back stderr for the CompletedProcess (used by correction logic)
    stderr_text = stderr_path.read_text(errors="replace") if stderr_path.exists() else ""
    proc = subprocess.CompletedProcess(
        args=step.tool_cmd,
        returncode=returncode,
        stdout="",  # stdout is on disk, not in memory
        stderr=stderr_text,
    )

    rec = ToolInvocationRecord(
        id=inv_id,
        step_id=step_id,
        attempt_number=len(step.invocation_ids) + 1,
        cmd=step.tool_cmd,
        exit_code=returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        started_at_utc=started,
        completed_at_utc=completed,
        duration_seconds=duration,
        had_output=had_stdout,
    )

    rec_path = analysis_dir / f"{inv_id}.record.json"
    rec_path.write_text(rec.model_dump_json(indent=2))

    return rec, proc


def run_forensic_tool(state: dict) -> dict:
    """LangGraph node: invoke the current step's tool_cmd with retry and self-correction."""
    plan = state["plan"]
    step_id: str = state["current_step_id"]
    step = next((s for s in plan.steps if s.id == step_id), None)
    logger.info("Node: run_forensic_tool | step={} cmd={}", step_id[:8], step.tool_cmd if step else "?")
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

    # Feasibility check — block unsafe commands before execution
    from valravn.training.feasibility import FeasibilityMemory

    feasibility = FeasibilityMemory()
    passed, violations = feasibility.check(
        cmd=step.tool_cmd,
        evidence_refs=[str(r) for r in evidence_refs],
        output_dir=str(output_dir),
    )
    if not passed:
        tool_failure = ToolFailureRecord(
            step_id=step_id,
            invocation_ids=[],
            final_error=f"Feasibility check blocked execution: {'; '.join(violations)}",
            diagnostic_context="\n".join(violations),
        )
        return {
            "invocations": invocations,
            "plan": plan,
            "_step_succeeded": False,
            "_step_exhausted": True,
            "_last_invocation_id": None,
            "_self_corrections": self_corrections,
            "_tool_failure": tool_failure,
        }

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
            # Ask LLM for a corrected command; if the LLM call fails
            # (timeout, bad JSON, network error), skip correction and
            # retry with the same command rather than crashing the graph.
            try:
                correction = _request_correction(
                    step_id=step_id,
                    attempt_number=attempt,
                    original_cmd=list(rec.cmd),
                    exit_code=proc.returncode,
                    stderr=proc.stderr,
                    analysis_dir=analysis_dir,
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
            except Exception:
                logger.warning(
                    "Self-correction LLM failed for step={} attempt={}; retrying with original cmd",
                    step_id[:8], attempt,
                )
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
