"""Tests for SecurityPlaybook protected entries (Q5)."""

from __future__ import annotations

import pytest

from valravn.training.playbook import SecurityPlaybook, ProtectedEntryError
from valravn.training.mutator import InvalidMutationError, apply_mutation


class TestProtectedEntries:
    """Testprotected list functionality for playbook entries."""

    def test_protected_cannot_be_deleted(self):
        """Protected entries should raise ProtectedEntryError on delete."""
        playbook = SecurityPlaybook()
        playbook.add_entry("critical-rule", "Never delete malware samples", "Forensics integrity")
        playbook.protect_entry("critical-rule")
        
        assert playbook.is_protected("critical-rule") is True
        
        with pytest.raises(ProtectedEntryError):
            playbook.delete_entry("critical-rule")
    
    def test_unprotected_entry_can_be_deleted(self):
        """Unprotected entries delete normally."""
        playbook = SecurityPlaybook()
        playbook.add_entry("optional-rule", "Optional guideline", "Rationale")
        
        assert playbook.is_protected("optional-rule") is False
        
        playbook.delete_entry("optional-rule")
        assert "optional-rule" not in playbook.entries
    
    def test_unprotect_entry_allows_deletion(self):
        """Unprotecting enables deletion."""
        playbook = SecurityPlaybook()
        playbook.add_entry("rule", "Content", "Rationale")
        playbook.protect_entry("rule")
        
        # Initially protected
        with pytest.raises(ProtectedEntryError):
            playbook.delete_entry("rule")
        
        # After unprotect
        playbook.unprotect_entry("rule")
        assert playbook.is_protected("rule") is False
        
        playbook.delete_entry("rule")
        assert "rule" not in playbook.entries
    
    def test_protect_nonexistent_entry_is_noop(self):
        """Protecting non-existent entry doesn't add to protected_id."""
        playbook = SecurityPlaybook()
        playbook.protect_entry("nonexistent")
        assert "nonexistent" not in playbook.protected_ids
    
    def test_protected_shows_in_prompt_section(self):
        """Protected entries marked in to_prompt_section output."""
        playbook = SecurityPlaybook()
        playbook.add_entry("protected-rule", "Critical", "Important")
        playbook.add_entry("normal-rule", "Optional", "Meh")
        playbook.protect_entry("protected-rule")
        
        output = playbook.to_prompt_section()
        assert "[PROTECTED]" in output
        assert "**protected-rule** [PROTECTED]" in output


class TestMutatorProtectedHandling:
    """Test mutator respects protected entries."""

    def test_mutator_blocked_on_protected_delete(self):
        """Mutator raises InvalidMutationError when trying to delete protected."""
        from unittest.mock import MagicMock, patch
        
        playbook = SecurityPlaybook()
        playbook.add_entry("core-rule", "Core security rule", "Critical")
        playbook.protect_entry("core-rule")
        optimizer_state = MagicMock()
        
        # Mock LLM to return DELETE on protected entry
        with patch("valravn.training.mutator._get_mutator_llm") as mock_get_llm:
            from valravn.training.mutator import MutationSpec
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MutationSpec(
                operation="DELETE",
                entry_id="core-rule",
                rule="",
                rationale="Try to delete",
            )
            mock_get_llm.return_value = mock_llm
            
            with pytest.raises(InvalidMutationError) as exc_info:
                apply_mutation(playbook, optimizer_state, 1, "test diagnostic")
            
            assert "DELETE blocked" in str(exc_info.value)
    
    def test_protected_entry_preserved_in_updates(self):
        """UPDATE can modify protected entries (DELETE is the blocked operation)."""
        playbook = SecurityPlaybook()
        playbook.add_entry("core-rule", "Original", "Rationale")
        playbook.protect_entry("core-rule")
        
        # UPDATE should still work on protected entries
        playbook.update_entry("core-rule", "Updated content", "New rationale")
        assert playbook.entries["core-rule"]["rule"] == "Updated content"
