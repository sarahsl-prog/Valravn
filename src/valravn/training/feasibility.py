"""Custom feasibility rules registry for ReplayBuffer.

Allows users to register custom constraints that determine whether
trajectories/cases are eligible for replay buffer storage.

Example usage:
    @register_feasibility_rule
    def must_have_some_invocations(case):
        return case.get("invocation_count", 0) > 0
    
    @register_feasibility_rule
    def exclude_network_errors(case):
        return "network" not in case.get("failure_reason", "")

    # Check feasibility
    is_feasible, reason = check_feasibility(case_data)

Also includes FeasibilityMemory class for command safety validation:
    fm = FeasibilityMemory()
    passed, violations = fm.check(cmd=["rm", "/evidence"], evidence_refs=[...])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_LOGGER = logging.getLogger(__name__)

# Registry for replay buffer feasibility rules
_custom_feasibility_rules: list[Callable[[dict[str, Any]], bool]] = []


# ============================================================================
# FeasibilityMemory for Command Safety Validation
# ============================================================================

@dataclass
class FeasibilityRule:
    """A symbolic constraint rule for command validation."""
    rule_id: str
    description: str
    check_fn: Callable[[list[str], str], tuple[bool, str]]


class FeasibilityMemory:
    """Symbolic constraint validation for unsafe commands.
    
    Prevents execution of destructive or unsafe commands by checking
    against a set of safety rules before tool execution.
    """
    
    def __init__(self) -> None:
        """Initialize FeasibilityMemory with default safety rules."""
        self.rules: list[FeasibilityRule] = []
        self._load_default_rules()
    
    def _load_default_rules(self) -> None:
        """Load the default safety constraint rules."""
        # Rule 1: No destructive commands
        self.rules.append(FeasibilityRule(
            rule_id="no_destructive",
            description="Commands must not be destructive",
            check_fn=self._check_destructive
        ))
        
        # Rule 2: No evidence path modification
        self.rules.append(FeasibilityRule(
            rule_id="evidence_protection",
            description="Commands must not modify evidence paths",
            check_fn=self._check_evidence_protection
        ))
        
        # Rule 3: Valid command structure
        self.rules.append(FeasibilityRule(
            rule_id="valid_command",
            description="Command must have valid structure",
            check_fn=self._check_valid_command
        ))
    
    def _check_destructive(self, cmd: list[str], evidence_refs: str) -> tuple[bool, str]:
        """Check command is not destructive."""
        if not cmd:
            return True, ""
        destructive = {"rm", "del", "remove", "delete", "destroy", "overwrite", "dd"}
        if cmd[0] in destructive or any(p in destructive for p in cmd):
            return False, f"Command includes destructive operation"
        return True, ""
    
    def _check_evidence_protection(self, cmd: list[str], evidence_refs: str) -> tuple[bool, str]:
        """Check evidence paths are protected from modification."""
        if not evidence_refs:
            return True, ""
        evidence_list = evidence_refs.split(",") if isinstance(evidence_refs, str) else []
        for arg in cmd:
            for ev_path in evidence_list:
                ev_path = ev_path.strip()
                if ev_path and arg.startswith(ev_path):
                    return False, f"Command references evidence path {ev_path}"
        return True, ""
    
    def _check_valid_command(self, cmd: list[str], evidence_refs: str) -> tuple[bool, str]:
        """Check command has valid structure."""
        if not cmd:
            return False, "Empty command"
        if not all(isinstance(s, str) for s in cmd):
            return False, "Command arguments must be strings"
        return True, ""
    
    def check(self, cmd: list[str], evidence_refs: list[str], output_dir: str) -> tuple[bool, list[str]]:
        """Validate command against all feasibility rules.
        
        Args:
            cmd: Command as list of strings
            evidence_refs: List of evidence path strings
            output_dir: Output directory path (for context)
            
        Returns:
            Tuple of (passed, violations). passed is True if all rules pass.
            violations is a list of error strings for failed rules.
        """
        violations: list[str] = []
        
        for rule in self.rules:
            try:
                result, msg = rule.check_fn(cmd, ",".join(evidence_refs))
                if not result:
                    violations.append(f"[{rule.rule_id}] {rule.description}: {msg}")
            except Exception as e:
                _LOGGER.warning("Feasibility rule %r raised exception: %s", rule.rule_id, e)
        
        return len(violations) == 0, violations
    
    def add_rule(self, rule: FeasibilityRule) -> None:
        """Add a custom FeasibilityRule."""
        self.rules.append(rule)
    
    @classmethod
    def load(cls, path: Path) -> "FeasibilityMemory":
        """Load FeasibilityMemory (creates new instance with default rules)."""
        return cls()


# ============================================================================
# Replay Buffer Feasibility Rules Registry
# ============================================================================

def register_feasibility_rule(
    func: Callable[[dict[str, Any]], bool],
) -> Callable[[dict[str, Any]], bool]:
    """Register a custom feasibility check function.

    The function should return True if the case/trajectory is feasible for
    replay buffer storage, False otherwise.

    Rules are evaluated in registration order. If ANY rule returns False,
    the trajectory is deemed infeasible.

    Example:
        @register_feasibility_rule
        def must_have_tools(case):
            return case.get("invocation_count", 0) > 0
    
        @register_feasibility_rule
        def max_duration_minutes(case):
            return case.get("duration_seconds", 0) <= 3600

    Args:
        func: Callable that takes a dict and returns bool.

    Returns:
        The registered function (for use as decorator).
    """
    _custom_feasibility_rules.append(func)
    _LOGGER.debug("Registered custom feasibility rule: %s", func.__name__)
    return func


def unregister_feasibility_rule(
    func: Callable[[dict[str, Any]], bool],
) -> None:
    """Remove a previously registered feasibility rule.

    Args:
        func: The function to remove.
    """
    if func in _custom_feasibility_rules:
        _custom_feasibility_rules.remove(func)
        _LOGGER.debug("Unregistered custom feasibility rule: %s", func.__name__)


def clear_feasibility_rules() -> None:
    """Clear all custom feasibility rules."""
    _custom_feasibility_rules.clear()
    _LOGGER.debug("Cleared all custom feasibility rules")


def check_feasibility(case: dict[str, Any]) -> tuple[bool, str]:
    """Check if a case/trajectory is feasible according to custom rules.

    Runs all registered feasibility rules in order. If any rule returns
    False, the trajectory is deemed infeasible.

    Args:
        case: The case dict to evaluate (e.g., {"case_id": "...", "invocation_count": 5, ...}).

    Returns:
        Tuple of (is_feasible, reason). If feasible, reason is empty.
        If not feasible, reason describes which rule rejected it.
    """
    if not _custom_feasibility_rules:
        return True, ""

    for rule in _custom_feasibility_rules:
        try:
            result = rule(case)
            if not result:
                reason = f"Feasibility rule '{rule.__name__}' rejected case"
                _LOGGER.debug(reason)
                return False, reason
        except Exception as e:
            _LOGGER.error("Feasibility rule '%s' raised exception: %s", rule.__name__, e)
            # Continue checking other rules
            continue

    return True, ""


def get_feasibility_rules() -> list[Callable[[dict[str, Any]], bool]]:
    """Get a copy of the current feasibility rules.

    Returns:
        List of registered rule functions.
    """
    return list(_custom_feasibility_rules)


def has_feasibility_rules() -> bool:
    """Check if any custom feasibility rules are registered."""
    return len(_custom_feasibility_rules) > 0


# Predefined rule generators for common patterns

def require_min_invocations(min_count: int) -> Callable[[dict[str, Any]], bool]:
    """Generate a feasibility rule requiring minimum number of invocations.
    
    Example:
        register_feasibility_rule(require_min_invocations(3))
    """
    def check(case: dict[str, Any]) -> bool:
        return case.get("invocation_count", 0) >= min_count
    check.__name__ = f"require_min_invocations({min_count})"
    return check


def exclude_failure_pattern(pattern: str) -> Callable[[dict[str, Any]], bool]:
    """Generate a feasibility rule excluding failures matching a pattern.
    
    Example:
        register_feasibility_rule(exclude_failure_pattern("timeout"))
    """
    def check(case: dict[str, Any]) -> bool:
        failure = case.get("failure_reason", "")
        return pattern.lower() not in failure.lower()
    check.__name__ = f"exclude_failure_pattern({pattern!r})"
    return check


def max_duration_seconds(max_seconds: float) -> Callable[[dict[str, Any]], bool]:
    """Generate a feasibility rule with maximum duration limit.
    
    Example:
        register_feasibility_rule(max_duration_seconds(3600))  # 1 hour max
    """
    def check(case: dict[str, Any]) -> bool:
        return case.get("duration_seconds", 0) <= max_seconds
    check.__name__ = f"max_duration_seconds({max_seconds})"
    return check
