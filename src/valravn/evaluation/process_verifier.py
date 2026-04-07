"""Process Verification Evaluator based on Agentic-MME.

Evaluates agent procedures (S-axis / strategy) and evidence quality
(V-axis / evidence) without any LLM calls.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ProcessCheckpoint:
    description: str
    axis: str  # "strategy" or "evidence"
    passed: bool = False


# ---------------------------------------------------------------------------
# Checkpoint templates
# ---------------------------------------------------------------------------

_CHECKPOINT_TEMPLATES: dict[str, list[ProcessCheckpoint]] = {
    "memory_analysis": [
        ProcessCheckpoint("Agent ran process listing tool (pslist/pstree)", "strategy"),
        ProcessCheckpoint("Agent checked for code injection (malfind)", "strategy"),
        ProcessCheckpoint("Agent examined network connections (netscan)", "strategy"),
        ProcessCheckpoint("Process listing output contains PID and PPID data", "evidence"),
        ProcessCheckpoint("Injection scan output contains VAD/memory region data", "evidence"),
    ],
    "disk_forensics": [
        ProcessCheckpoint("Agent ran filesystem timeline tool (mftparser/fls)", "strategy"),
        ProcessCheckpoint("Agent checked for deleted files (unrm/recoverjpeg)", "strategy"),
        ProcessCheckpoint("Agent examined file metadata (stat/istat)", "strategy"),
        ProcessCheckpoint("Timeline output contains timestamps and inode data", "evidence"),
        ProcessCheckpoint("Filesystem output contains file path and size data", "evidence"),
    ],
    "windows_artifacts": [
        ProcessCheckpoint("Agent examined registry hives (printkey/hivelist)", "strategy"),
        ProcessCheckpoint("Agent reviewed Windows event logs (evtlogs/mftenum)", "strategy"),
        ProcessCheckpoint("Registry output contains key names and values", "evidence"),
    ],
}


# ---------------------------------------------------------------------------
# ProcessVerifier
# ---------------------------------------------------------------------------


class ProcessVerifier:
    """Verifies agent process quality against checkpoint templates."""

    def define_checkpoints(self, investigation_type: str) -> list[ProcessCheckpoint]:
        """Return fresh copies of checkpoints for the given investigation type."""
        template = _CHECKPOINT_TEMPLATES.get(investigation_type, [])
        return [copy.copy(cp) for cp in template]

    def verify_strategy(
        self, checkpoints: list[ProcessCheckpoint], tools_used: list[str]
    ) -> list[ProcessCheckpoint]:
        """Mark strategy checkpoints as passed via keyword match against tool names."""
        tools_lower = [t.lower() for t in tools_used]
        for cp in checkpoints:
            if cp.axis != "strategy":
                continue
            desc_words = set(cp.description.lower().split())
            for tool in tools_lower:
                if any(word in tool for word in desc_words) or any(
                    token in cp.description.lower() for token in tool.split("/")
                ):
                    cp.passed = True
                    break
            # Also check if any keyword from the description appears in any tool name
            if not cp.passed:
                for tool in tools_lower:
                    for word in desc_words:
                        if word in tool:
                            cp.passed = True
                            break
                    if cp.passed:
                        break
        return checkpoints

    def verify_evidence(
        self, checkpoints: list[ProcessCheckpoint], tool_outputs: list[str]
    ) -> list[ProcessCheckpoint]:
        """Mark evidence checkpoints as passed via keyword match against tool outputs."""
        outputs_lower = [o.lower() for o in tool_outputs]
        for cp in checkpoints:
            if cp.axis != "evidence":
                continue
            # Extract meaningful keywords from the description (skip common words)
            _STOP_WORDS = {"agent", "ran", "the", "and", "for", "output", "contains", "data"}
            desc_words = [
                w for w in cp.description.lower().split() if w not in _STOP_WORDS
            ]
            for output in outputs_lower:
                if any(word in output for word in desc_words):
                    cp.passed = True
                    break
        return checkpoints

    def compute_scores(
        self,
        strategy_results: list[ProcessCheckpoint],
        evidence_results: list[ProcessCheckpoint],
        agent_steps: int,
        reference_steps: int,
    ) -> dict:
        """Compute strategy, evidence, overthink penalty, and composite scores.

        Returns a dict with keys:
          strategy_score, evidence_score, overthink_penalty, composite_score
        """
        strategy_cps = [cp for cp in strategy_results if cp.axis == "strategy"]
        evidence_cps = [cp for cp in evidence_results if cp.axis == "evidence"]

        strategy_score = (
            sum(cp.passed for cp in strategy_cps) / len(strategy_cps)
            if strategy_cps
            else 0.0
        )
        evidence_score = (
            sum(cp.passed for cp in evidence_cps) / len(evidence_cps)
            if evidence_cps
            else 0.0
        )
        penalty = compute_overthink_penalty(agent_steps, reference_steps)
        composite_score = (
            (strategy_score + evidence_score) / 2.0
        ) * (1.0 - penalty)

        return {
            "strategy_score": strategy_score,
            "evidence_score": evidence_score,
            "overthink_penalty": penalty,
            "composite_score": composite_score,
        }


# ---------------------------------------------------------------------------
# Standalone penalty function
# ---------------------------------------------------------------------------


def compute_overthink_penalty(agent_steps: int, reference_steps: int) -> float:
    """Penalise excess steps: max(0, (agent_steps - reference_steps) / (reference_steps + 1))."""
    return max(0.0, (agent_steps - reference_steps) / (reference_steps + 1))
