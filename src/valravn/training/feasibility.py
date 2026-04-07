"""Custom feasibility rules registry for ReplayBuffer.

Allows users to register custom constraints that determine whether
trajectories are eligible for replay buffer storage.

Example usage:
    @register_feasibility_rule
    def must_have_some_invocations(case):
        return case.get("invocation_count", 0) > 0
    
    @register_feasibility_rule
    def exclude_network_errors(case):
        return "network" not in case.get("failure_reason", "")

    # Check feasibility
    is_feasible, reason = check_feasibility(case_data)
"""

from __future__ import annotations

import logging
from typing import Any, Callable

_LOGGER = logging.getLogger(__name__)

# Registry of custom feasibility check functions
# Each function receives the case dict and returns True if feasible
_custom_feasibility_rules: list[Callable[[dict[str, Any]], bool]] = []


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
