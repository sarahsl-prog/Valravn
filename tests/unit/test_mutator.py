from __future__ import annotations

from unittest.mock import MagicMock, patch

from valravn.training.mutator import MutationSpec, apply_mutation
from valravn.training.optimizer_state import OptimizerState
from valravn.training.playbook import SecurityPlaybook


def test_mutator_add_operation():
    """apply_mutation with ADD operation adds an entry to the playbook and logs the change."""
    playbook = SecurityPlaybook()
    optimizer_state = OptimizerState()

    mock_result = MutationSpec(
        operation="ADD",
        entry_id="rule-001",
        rule="Always verify evidence hash before analysis",
        rationale="Hash verification prevents working on corrupted or tampered evidence",
    )

    with patch("valravn.training.mutator._get_mutator_llm") as mock_llm_fn:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_result
        mock_llm_fn.return_value = mock_llm

        apply_mutation(
            playbook=playbook,
            optimizer_state=optimizer_state,
            iteration=1,
            diagnostic_text="actionable_gap: agent skipped hash verification",
        )

    assert "rule-001" in playbook.entries
    assert playbook.entries["rule-001"]["rule"] == "Always verify evidence hash before analysis"
    assert len(optimizer_state.change_ledger) == 1
    assert "ADD" in optimizer_state.change_ledger[0]
    assert "rule-001" in optimizer_state.change_ledger[0]


def test_mutator_noop_for_intractable():
    """apply_mutation with NOOP leaves the playbook unchanged and logs nothing."""
    playbook = SecurityPlaybook()
    playbook.add_entry("rule-001", "Verify hashes", "Integrity check", iteration=0)
    optimizer_state = OptimizerState()

    mock_result = MutationSpec(
        operation="NOOP",
        entry_id="",
        rule="",
        rationale="",
    )

    with patch("valravn.training.mutator._get_mutator_llm") as mock_llm_fn:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_result
        mock_llm_fn.return_value = mock_llm

        apply_mutation(
            playbook=playbook,
            optimizer_state=optimizer_state,
            iteration=2,
            diagnostic_text="intractable: evidence is encrypted, no actionable gap identified",
        )

    assert list(playbook.entries.keys()) == ["rule-001"]
    assert len(optimizer_state.change_ledger) == 0


def test_mutator_delete_operation():
    """apply_mutation with DELETE removes the entry from playbook and logs the change."""
    playbook = SecurityPlaybook()
    playbook.add_entry("rule-002", "Outdated rule", "No longer applicable", iteration=0)
    optimizer_state = OptimizerState()

    mock_result = MutationSpec(
        operation="DELETE",
        entry_id="rule-002",
        rule="",
        rationale="Rule is redundant given new tooling",
    )

    with patch("valravn.training.mutator._get_mutator_llm") as mock_llm_fn:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_result
        mock_llm_fn.return_value = mock_llm

        apply_mutation(
            playbook=playbook,
            optimizer_state=optimizer_state,
            iteration=3,
            diagnostic_text=(
                "execution_variance: rule-002 caused agent to over-investigate low-risk artifacts"
            ),
        )

    assert "rule-002" not in playbook.entries
    assert len(optimizer_state.change_ledger) == 1
    assert "DELETE" in optimizer_state.change_ledger[0]
    assert "rule-002" in optimizer_state.change_ledger[0]
