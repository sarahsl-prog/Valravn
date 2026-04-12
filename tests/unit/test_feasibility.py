"""Tests for custom feasibility rules registry (Q2)."""

from __future__ import annotations

import pytest

from valravn.training.feasibility import (
    check_feasibility,
    clear_feasibility_rules,
    exclude_failure_pattern,
    get_feasibility_rules,
    has_feasibility_rules,
    max_duration_seconds,
    register_feasibility_rule,
    require_min_invocations,
    unregister_feasibility_rule,
)


class TestFeasibilityRegistry:
    """Test feasibility rules registry functionality."""

    def setup_method(self) -> None:
        """Clear registry before each test."""
        clear_feasibility_rules()

    def teardown_method(self) -> None:
        """Clear registry after each test."""
        clear_feasibility_rules()

    def test_empty_registry_returns_feasible(self):
        """No rules means any case is feasible."""
        case = {"case_id": "test-1", "invocation_count": 5}
        is_feasible, reason = check_feasibility(case)
        assert is_feasible is True
        assert reason == ""

    def test_register_single_rule(self):
        """Register and verify a rule is called."""
        rule_called = []

        @register_feasibility_rule
        def must_have_id(case: dict) -> bool:
            rule_called.append(True)
            return "case_id" in case

        assert has_feasibility_rules() is True
        check_feasibility({"case_id": "test"})
        assert rule_called == [True]

    def test_failing_rule_rejects_case(self):
        """If any rule returns False, case is rejected."""
        @register_feasibility_rule
        def reject_all(case: dict) -> bool:
            return False

        case = {"case_id": "test-1"}
        is_feasible, reason = check_feasibility(case)
        assert is_feasible is False
        assert "Feasibility rule" in reason
        assert "reject_all" in reason

    def test_multiple_rules_any_fail_rejects(self):
        """Multiple rules: any failure rejects."""
        @register_feasibility_rule
        def rule_one(case: dict) -> bool:
            return True

        @register_feasibility_rule
        def rule_two(case: dict) -> bool:
            return False  # This one rejects

        @register_feasibility_rule
        def rule_three(case: dict) -> bool:
            return True

        case = {"case_id": "test"}
        is_feasible, _ = check_feasibility(case)
        assert is_feasible is False

    def test_all_rules_pass_allows_case(self):
        """All rules must pass for feasibility."""
        @register_feasibility_rule
        def rule_one(case: dict) -> bool:
            return case.get("invocation_count", 0) > 0

        @register_feasibility_rule
        def rule_two(case: dict) -> bool:
            return "case_id" in case

        case = {"case_id": "test-1", "invocation_count": 5}
        is_feasible, _ = check_feasibility(case)
        assert is_feasible is True

    def test_unregister_rule(self):
        """Unregister removes a rule."""
        @register_feasibility_rule
        def reject_all(case: dict) -> bool:
            return False

        assert has_feasibility_rules() is True
        
        unregister_feasibility_rule(reject_all)
        
        assert has_feasibility_rules() is False
        is_feasible, _ = check_feasibility({"case_id": "test"})
        assert is_feasible is True


class TestFeasibilityRuleGenerators:
    """Test built-in rule generator functions."""

    def setup_method(self) -> None:
        """Clear registry before each test."""
        clear_feasibility_rules()

    def teardown_method(self) -> None:
        """Clear registry after each test."""
        clear_feasibility_rules()

    def test_require_min_invocations(self):
        """Test min invocations rule generator."""
        register_feasibility_rule(require_min_invocations(3))

        # Should pass
        assert check_feasibility({"invocation_count": 5})[0] is True

        # Should fail
        assert check_feasibility({"invocation_count": 2})[0] is False

        # Default of 0 fails
        assert check_feasibility({})[0] is False

    def test_exclude_failure_pattern(self):
        """Test exclude failure pattern rule generator."""
        register_feasibility_rule(exclude_failure_pattern("timeout"))

        # Should pass
        assert check_feasibility({"failure_reason": "permission denied"})[0] is True

        # Should fail (case insensitive)
        assert check_feasibility({"failure_reason": "Connection Timeout Error"})[0] is False

    def test_max_duration_seconds(self):
        """Test max duration rule generator."""
        register_feasibility_rule(max_duration_seconds(3600))  # 1 hour

        # Should pass
        assert check_feasibility({"duration_seconds": 300})[0] is True

        # Should fail
        assert check_feasibility({"duration_seconds": 7200})[0] is False

    def test_get_feasibility_rules(self):
        """Test getting registered rules."""
        @register_feasibility_rule
        def my_rule(case: dict) -> bool:
            return True

        rules = get_feasibility_rules()
        assert len(rules) == 1
        assert rules[0].__name__ == "my_rule"


class TestFeasibilityErrorHandling:
    """Test error handling in feasibility checks."""

    def setup_method(self) -> None:
        """Clear registry before each test."""
        clear_feasibility_rules()

    def teardown_method(self) -> None:
        """Clear registry after each test."""
        clear_feasibility_rules()

    def test_continues_on_rule_exception(self):
        """If a rule raises, continue checking others."""
        @register_feasibility_rule
        def explode(case: dict) -> bool:
            raise ValueError("boom")

        @register_feasibility_rule
        def reject_all(case: dict) -> bool:
            return False

        # Should fail from reject_all, not because of explosion
        is_feasible, reason = check_feasibility({"case_id": "test"})
        assert is_feasible is False
        assert "reject_all" in reason


class TestFeasibilityMemoryPathMatching:
    """A-06: FeasibilityMemory must use Path.is_relative_to() for evidence path checks."""

    def test_evidence_path_prefix_false_positive(self):
        """A-06: /mnt/evidence2 must NOT match /mnt/evidence evidence path."""
        from valravn.training.feasibility import FeasibilityMemory

        fm = FeasibilityMemory()
        # /mnt/evidence2 should NOT be blocked when evidence is /mnt/evidence
        passed, _ = fm._check_evidence_protection(
            cmd=["rm", "-rf", "/mnt/evidence2"],
            evidence_refs="/mnt/evidence",
        )
        assert passed is True, "/mnt/evidence2 must not match /mnt/evidence"

    def test_exact_evidence_path_is_blocked(self):
        """A-06: Exact evidence path must still be blocked for destructive commands."""
        from valravn.training.feasibility import FeasibilityMemory

        fm = FeasibilityMemory()
        passed, msg = fm._check_evidence_protection(
            cmd=["rm", "-rf", "/mnt/evidence"],
            evidence_refs="/mnt/evidence",
        )
        assert passed is False
        assert "/mnt/evidence" in msg

    def test_subdirectory_of_evidence_is_blocked(self):
        """A-06: A path under the evidence directory must be blocked."""
        from valravn.training.feasibility import FeasibilityMemory

        fm = FeasibilityMemory()
        passed, msg = fm._check_evidence_protection(
            cmd=["rm", "-rf", "/mnt/evidence/subdir"],
            evidence_refs="/mnt/evidence",
        )
        assert passed is False
        assert "/mnt/evidence" in msg

    def test_evidence_path_without_trailing_slash(self):
        """A-06: Evidence path without trailing slash must not match path-prefixed names."""
        from valravn.training.feasibility import FeasibilityMemory

        fm = FeasibilityMemory()
        # /mnt/evidenceXYZ should NOT match evidence path /mnt/evidence
        passed, _ = fm._check_evidence_protection(
            cmd=["rm", "-rf", "/mnt/evidenceXYZ"],
            evidence_refs="/mnt/evidence",
        )
        assert passed is True, "/mnt/evidenceXYZ must not match /mnt/evidence"
