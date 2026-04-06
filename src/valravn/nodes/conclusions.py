from __future__ import annotations

from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

_SYSTEM_PROMPT = """\
You are an expert DFIR analyst. Given the investigation prompt, tool invocation outputs,
and any detected anomalies, synthesize forensic conclusions.

Each conclusion must:
- Make a specific, falsifiable statement about what was found (or not found)
- Cite at least one invocation ID from the evidence that supports it
- Assign a confidence level: "high" (directly observed), "medium" (inferred from evidence),
  or "low" (circumstantial or incomplete evidence)

If the evidence is insufficient for any conclusions, return an empty list.
Do NOT speculate beyond what the evidence supports.
"""

MAX_STDOUT_CHARS = 10_000


class _ConclusionSpec(BaseModel):
    statement: str
    supporting_invocation_ids: list[str]
    confidence: str  # "high", "medium", "low"


class _ConclusionsOutput(BaseModel):
    conclusions: list[_ConclusionSpec]


def _get_conclusions_llm() -> object:
    llm = ChatAnthropic(model="claude-opus-4-6", temperature=0)
    return llm.with_structured_output(_ConclusionsOutput)


def synthesize_conclusions(state: dict) -> dict:
    """LangGraph node: synthesize forensic conclusions from all tool outputs."""
    invocations = state.get("invocations") or []
    if not invocations:
        return {"_conclusions": []}

    task = state.get("task")
    prompt_text = task.prompt if task else "Unknown investigation"

    # Build per-invocation summaries
    invocation_summaries: list[str] = []
    for inv in invocations:
        stdout_path = Path(inv.stdout_path)
        raw = stdout_path.read_text(errors="replace") if stdout_path.exists() else ""
        output_snippet = raw[:MAX_STDOUT_CHARS]
        cmd_str = " ".join(str(c) for c in inv.cmd)
        summary = (
            f"Invocation ID: {inv.id}\n"
            f"Command: {cmd_str}\n"
            f"Exit code: {inv.exit_code}\n"
            f"Output (truncated to {MAX_STDOUT_CHARS} chars):\n{output_snippet}"
        )
        invocation_summaries.append(summary)

    # Include anomaly descriptions if any
    anomalies = state.get("anomalies") or []
    anomaly_section = ""
    if anomalies:
        anomaly_lines = [
            f"- [{a.description}] (significance: {a.forensic_significance})"
            for a in anomalies
        ]
        anomaly_section = "\n\nDetected anomalies:\n" + "\n".join(anomaly_lines)

    human_content = (
        f"Investigation prompt: {prompt_text}\n\n"
        f"Tool invocation outputs:\n\n"
        + "\n\n---\n\n".join(invocation_summaries)
        + anomaly_section
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]

    result: _ConclusionsOutput = _get_conclusions_llm().invoke(messages)

    conclusions = [c.model_dump() for c in result.conclusions]
    return {"_conclusions": conclusions}
