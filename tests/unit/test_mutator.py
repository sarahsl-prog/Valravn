from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from valravn.training.mutator import MutationSpec, apply_mutation
from valravn.training.optimizer_state import OptimizerState
from valravn.training.playbook import SecurityPlaybook


def _ai(spec: MutationSpec) -> AIMessage:
    """Wrap a MutationSpec as an AIMessage for mocking."""
    return AIMessage(content=spec.model_dump_json())


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
        mock_llm.invoke.return_value = _ai(mock_result)
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
        mock_llm.invoke.return_value = _ai(mock_result)
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
        mock_llm.invoke.return_value = _ai(mock_result)
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


def test_mutator_update_operation():
    """apply_mutation with UPDATE modifies existing entry."""
    playbook = SecurityPlaybook()
    playbook.add_entry("rule-001", "Old rule text", "Original rationale", iteration=0)
    optimizer_state = OptimizerState()

    mock_result = MutationSpec(
        operation="UPDATE",
        entry_id="rule-001",
        rule="Updated: Verify SHA-256 hashes before analysis",
        rationale="Specification of hash algorithm needed",
    )

    with patch("valravn.training.mutator._get_mutator_llm") as mock_llm_fn:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _ai(mock_result)
        mock_llm_fn.return_value = mock_llm

        apply_mutation(
            playbook=playbook,
            optimizer_state=optimizer_state,
            iteration=2,
            diagnostic_text="actionable_gap: hash algorithm not specified",
        )

    assert playbook.entries["rule-001"]["rule"] == "Updated: Verify SHA-256 hashes before analysis"
    assert playbook.entries["rule-001"]["rationale"] == "Specification of hash algorithm needed"
    assert len(optimizer_state.change_ledger) == 1


def test_mutation_spec_valid_operations():
    """BUG-003: Valid mutation specs should be accepted."""
    # All valid operation types
    for op in ["ADD", "UPDATE", "DELETE", "NOOP"]:
        spec = MutationSpec(
            operation=op,
            entry_id="test-rule",
            rule="Test rule",
            rationale="Test rationale",
        )
        assert spec.operation == op


def test_mutation_spec_rejects_invalid_operation():
    """BUG-003: Invalid operation raises ValidationError."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as exc_info:
        MutationSpec(
            operation="INVALID_OP",
            entry_id="test-rule",
            rule="Test",
        )
    
    assert "operation" in str(exc_info.value)
    assert "ADD" in str(exc_info.value) or "DELETE" in str(exc_info.value)


def test_mutation_spec_entry_id_validation():
    """BUG-003: Entry ID must be kebab-case (lowercase letters, hyphens, numbers)."""
    import pytest
    from pydantic import ValidationError

    # Valid entry IDs
    valid_ids = ["rule-hash-verify", "a", "abc-123", "my-rule-001"]
    for entry_id in valid_ids:
        spec = MutationSpec(
            operation="ADD",
            entry_id=entry_id,
            rule="Test rule",
        )
        assert spec.entry_id == entry_id

    # Invalid entry IDs
    invalid_ids = [
        "Rule-Uppercase",
        "rule_with_underscores",
        "rule space",
        "rule.dot",
        "123-starts-with-number",
        "",
    ]
    for entry_id in invalid_ids:
        with pytest.raises(ValidationError) as exc_info:
            MutationSpec(
                operation="ADD",
                entry_id=entry_id,
                rule="Test rule",
            )
        err = str(exc_info.value)
        assert "entry_id" in err or "entry_id must be kebab-case" in err


def test_mutation_spec_rule_safety():
    """BUG-003: Rule text cannot contain injection patterns (backticks, $, |, ;, newlines)."""
    import pytest
    from pydantic import ValidationError

    # Valid rule (no injection) — should not raise
    MutationSpec(
        operation="ADD",
        entry_id="rule-test",
        rule="Verify SHA256 hash of evidence files",
    )

    # Invalid rules with shell injection patterns
    injection_rules = [
        "Run `cat /etc/passwd`",  # Backticks
        "Execute $(whoami)",      # Command substitution
        "rm -rf /; echo done",    # Semicolon
        "cat file | grep secret",  # Pipe
        "Line1\nLine2",           # Newline
    ]
    
    for rule in injection_rules:
        with pytest.raises(ValidationError) as exc_info:
            MutationSpec(
                operation="ADD",
                entry_id="rule-test",
                rule=rule,
            )
        assert "rule" in str(exc_info.value)


def test_mutation_spec_length_limits():
    """BUG-003: Entry IDs, rules, and rationales have maximum lengths."""
    import pytest
    from pydantic import ValidationError

    # Entry ID too long (> 64 chars)
    with pytest.raises(ValidationError):
        MutationSpec(
            operation="ADD",
            entry_id="x" * 65,
            rule="Test rule",
        )

    # Rule too long (> 500 chars)
    with pytest.raises(ValidationError):
        MutationSpec(
            operation="ADD",
            entry_id="rule-test",
            rule="x" * 501,
        )

    # Rationale too long (> 500 chars) - only checked if provided
    with pytest.raises(ValidationError):
        MutationSpec(
            operation="ADD",
            entry_id="rule-test",
            rule="Test rule",
            rationale="x" * 501,
        )


def test_mutation_safety_playbook_size_limit():
    """BUG-003: Cannot ADD entry when playbook at max capacity."""
    import pytest

    from valravn.training.mutator import InvalidMutationError

    playbook = SecurityPlaybook()
    # Fill playbook to max capacity (1000 entries)
    for i in range(1000):
        playbook.add_entry(f"rule-{i:04d}", f"Rule {i}", "Test", iteration=0)
    
    optimizer_state = OptimizerState()

    # Try to add one more
    mock_result = MutationSpec(
        operation="ADD",
        entry_id="rule-new",
        rule="New rule",
        rationale="Should fail",
    )

    with patch("valravn.training.mutator._get_mutator_llm") as mock_llm_fn:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _ai(mock_result)
        mock_llm_fn.return_value = mock_llm

        with pytest.raises(InvalidMutationError) as exc_info:
            apply_mutation(
                playbook=playbook,
                optimizer_state=optimizer_state,
                iteration=1,
                diagnostic_text="actionable_gap: needs one more rule",
            )

        assert "max capacity" in str(exc_info.value) or "1000" in str(exc_info.value)


def test_mutation_safety_delete_nonexistent():
    """BUG-003: DELETE on non-existent entry raises InvalidMutationError."""
    import pytest

    from valravn.training.mutator import InvalidMutationError

    playbook = SecurityPlaybook()
    playbook.add_entry("rule-exists", "Existing rule", "Rationale", iteration=0)
    optimizer_state = OptimizerState()

    mock_result = MutationSpec(
        operation="DELETE",
        entry_id="rule-does-not-exist",
        rationale="Trying to delete non-existent rule",
    )

    with patch("valravn.training.mutator._get_mutator_llm") as mock_llm_fn:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _ai(mock_result)
        mock_llm_fn.return_value = mock_llm

        with pytest.raises(InvalidMutationError) as exc_info:
            apply_mutation(
                playbook=playbook,
                optimizer_state=optimizer_state,
                iteration=1,
                diagnostic_text="actionable_gap: rule is harmful",
            )
        
        assert "DELETE" in str(exc_info.value)
        assert "non-existent" in str(exc_info.value)


def test_mutation_safety_update_nonexistent():
    """BUG-003: UPDATE on non-existent entry raises InvalidMutationError."""
    import pytest

    from valravn.training.mutator import InvalidMutationError

    playbook = SecurityPlaybook()
    playbook.add_entry("rule-exists", "Existing rule", "Rationale", iteration=0)
    optimizer_state = OptimizerState()

    mock_result = MutationSpec(
        operation="UPDATE",
        entry_id="rule-does-not-exist",
        rule="Updated rule text",
        rationale="Trying to update non-existent rule",
    )

    with patch("valravn.training.mutator._get_mutator_llm") as mock_llm_fn:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _ai(mock_result)
        mock_llm_fn.return_value = mock_llm

        with pytest.raises(InvalidMutationError) as exc_info:
            apply_mutation(
                playbook=playbook,
                optimizer_state=optimizer_state,
                iteration=1,
                diagnostic_text="actionable_gap: rule needs update",
            )

        assert "UPDATE" in str(exc_info.value)
        assert "non-existent" in str(exc_info.value)


def test_apply_mutation_parses_llm_json():
    """A-01: apply_mutation must parse the LLM AIMessage response, not assign it directly."""
    from langchain_core.messages import AIMessage

    playbook = SecurityPlaybook()
    optimizer_state = OptimizerState()

    llm_json = '{"operation": "NOOP", "entry_id": "", "rule": "", "rationale": ""}'

    with patch("valravn.training.mutator._get_mutator_llm") as mock_factory:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content=llm_json)
        mock_factory.return_value = mock_llm

        # Should not raise AttributeError: 'AIMessage' object has no attribute 'operation'
        apply_mutation(playbook, optimizer_state, iteration=1, diagnostic_text="test diag")

    # NOOP means no entries added
    assert len(playbook.entries) == 0
