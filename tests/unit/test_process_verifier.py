"""Tests for ProcessVerifier and compute_overthink_penalty (Agentic-MME)."""

from __future__ import annotations

import pytest

from valravn.evaluation.process_verifier import (
    ProcessCheckpoint,
    ProcessVerifier,
    compute_overthink_penalty,
)


# ---------------------------------------------------------------------------
# Checkpoint definition tests
# ---------------------------------------------------------------------------


def test_process_verifier_strategy_checkpoints():
    verifier = ProcessVerifier()
    checkpoints = verifier.define_checkpoints("memory_analysis")
    strategy_cps = [cp for cp in checkpoints if cp.axis == "strategy"]
    assert len(strategy_cps) > 0
    assert all(isinstance(cp, ProcessCheckpoint) for cp in strategy_cps)


def test_process_verifier_evidence_checkpoints():
    verifier = ProcessVerifier()
    checkpoints = verifier.define_checkpoints("memory_analysis")
    evidence_cps = [cp for cp in checkpoints if cp.axis == "evidence"]
    assert len(evidence_cps) > 0
    assert all(isinstance(cp, ProcessCheckpoint) for cp in evidence_cps)


# ---------------------------------------------------------------------------
# Overthink penalty tests
# ---------------------------------------------------------------------------


def test_overthink_penalty_exact_match():
    penalty = compute_overthink_penalty(agent_steps=5, reference_steps=5)
    assert penalty == 0.0


def test_overthink_penalty_excess_steps():
    penalty = compute_overthink_penalty(agent_steps=10, reference_steps=5)
    assert penalty > 0.0


def test_overthink_penalty_fewer_steps():
    penalty = compute_overthink_penalty(agent_steps=3, reference_steps=5)
    assert penalty == 0.0
