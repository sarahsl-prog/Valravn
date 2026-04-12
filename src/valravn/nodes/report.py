from __future__ import annotations

import json
import re
from pathlib import Path

from valravn.models.report import (
    Conclusion,
    FindingsReport,
    SelfCorrectionEvent,
    ToolFailureRecord,
)

_REPORT_TEMPLATE = """\
# DFIR Findings Report

**Task ID**: {task_id}
**Generated**: {generated_at} UTC
**Evidence**: {evidence}

## Investigation Prompt

{prompt}

## Conclusions

{conclusions}

## Anomalies

{anomalies}

## Tool Failures

{failures}

## Self-Corrections

{corrections}
"""


def write_findings_report(state: dict) -> dict:
    """LangGraph node: render FindingsReport to ./reports/ as Markdown and JSON."""
    task = state["task"]
    output_dir = Path(state.get("_output_dir", "."))
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    conclusions = [
        Conclusion(**c) if isinstance(c, dict) else c
        for c in (state.get("_conclusions") or [])
    ]
    tool_failures = [
        ToolFailureRecord(**f) if isinstance(f, dict) else f
        for f in (state.get("_tool_failures") or [])
    ]
    self_corrections = [
        SelfCorrectionEvent(**e) if isinstance(e, dict) else e
        for e in (state.get("_self_corrections") or [])
    ]

    plan_path = output_dir / "analysis" / "investigation_plan.json"

    report = FindingsReport(
        task_id=task.id,
        prompt=task.prompt,
        evidence_refs=task.evidence_refs,
        conclusions=conclusions,
        anomalies=list(state.get("anomalies") or []),
        tool_failures=tool_failures,
        self_corrections=self_corrections,
        investigation_plan_path=plan_path,
        evidence_hashes=dict(state.get("_evidence_hashes") or {}),
    )

    ts = report.generated_at_utc.strftime("%Y%m%d_%H%M%S")
    slug = _slugify(task.prompt)
    stem = f"{ts}_{slug}"

    md_path = reports_dir / f"{stem}.md"
    json_path = reports_dir / f"{stem}.json"

    md_path.write_text(_render_markdown(report))
    json_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, default=str)
    )

    return {"report": report}


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", text.lower().replace(" ", "_"))[:40]


def _render_markdown(report: FindingsReport) -> str:
    def fmt_conclusions(conclusions):
        if not conclusions:
            return "_No conclusions recorded._"
        lines = []
        for c in conclusions:
            lines.append(f"### {c.statement}")
            lines.append(f"**Confidence**: {c.confidence}")
            lines.append(
                f"**Supporting invocations**: {', '.join(c.supporting_invocation_ids)}"
            )
            lines.append("")
        return "\n".join(lines)

    def fmt_anomalies(anomalies):
        if not anomalies:
            return "_No anomalies detected._"
        lines = []
        for a in anomalies:
            lines.append(
                f"- **{a.description}** ({a.response_action}): {a.forensic_significance}"
            )
        return "\n".join(lines)

    def fmt_failures(failures):
        if not failures:
            return "_No tool failures._"
        lines = []
        for f in failures:
            lines.append(f"- Step `{f.step_id}`: {f.final_error}")
            lines.append(f"  Diagnostic: {f.diagnostic_context}")
        return "\n".join(lines)

    def fmt_corrections(corrections):
        if not corrections:
            return "_No self-corrections._"
        lines = []
        for c in corrections:
            lines.append(
                f"- Step `{c.step_id}` attempt {c.attempt_number}: "
                f"`{' '.join(c.original_cmd)}` → `{' '.join(c.corrected_cmd)}`\n"
                f"  Rationale: {c.correction_rationale}"
            )
        return "\n".join(lines)

    return _REPORT_TEMPLATE.format(
        task_id=report.task_id,
        generated_at=report.generated_at_utc.strftime("%Y-%m-%d %H:%M:%S"),
        evidence=", ".join(report.evidence_refs),
        prompt=report.prompt,
        conclusions=fmt_conclusions(report.conclusions),
        anomalies=fmt_anomalies(report.anomalies),
        failures=fmt_failures(report.tool_failures),
        corrections=fmt_corrections(report.self_corrections),
    )
