from __future__ import annotations

import json
from pathlib import Path
from typing import Callable


class FeasibilityRule:
    """A single named feasibility constraint with an associated check function."""

    def __init__(
        self,
        rule_id: str,
        description: str,
        check_fn: Callable[[list[str], list[str], str], tuple[bool, str]],
    ) -> None:
        # check_fn(cmd, evidence_refs, output_dir) -> (passed: bool, violation_msg: str)
        self.rule_id = rule_id
        self.description = description
        self.check_fn = check_fn


# ---------------------------------------------------------------------------
# Default rule implementations
# ---------------------------------------------------------------------------

_DESTRUCTIVE_COMMANDS = {"rm", "shred", "mkfs", "fdisk", "wipefs"}
_NETWORK_COMMANDS = {"curl", "wget", "nc", "ncat", "ssh", "scp", "rsync"}


def _make_f001() -> FeasibilityRule:
    """F001: Never write to evidence directories."""

    def check(cmd: list[str], evidence_refs: list[str], output_dir: str) -> tuple[bool, str]:
        if not cmd or not evidence_refs:
            return True, ""

        # Resolve evidence parent directories from refs
        evidence_dirs: list[Path] = []
        for ref in evidence_refs:
            p = Path(ref)
            # Both the file itself and its parent directory are protected
            evidence_dirs.append(p.parent if p.suffix else p)
            evidence_dirs.append(p)

        def _under_evidence(target_str: str) -> bool:
            target = Path(target_str).resolve() if target_str else None
            if target is None:
                return False
            for ev in evidence_dirs:
                ev_resolved = ev.resolve()
                try:
                    target.relative_to(ev_resolved)
                    return True
                except ValueError:
                    pass
                # Also check without resolving (handles non-existent paths)
                try:
                    Path(target_str).relative_to(str(ev))
                    return True
                except ValueError:
                    pass
            return False

        binary = cmd[0]

        # cp / mv: last argument is the destination
        if binary in {"cp", "mv"} and len(cmd) >= 3:
            destination = cmd[-1]
            if _under_evidence(destination):
                return (
                    False,
                    f"F001: {binary} destination '{destination}' is inside an evidence path",
                )

        # Redirection operators and flags that introduce an output path
        write_flags = {">", ">>", "-o", "--output", "-w", "--write"}
        for i, arg in enumerate(cmd):
            if arg in write_flags and i + 1 < len(cmd):
                target = cmd[i + 1]
                if _under_evidence(target):
                    return (
                        False,
                        f"F001: output argument '{target}' "
                        f"(after '{arg}') is inside an evidence path",
                    )

        return True, ""

    return FeasibilityRule(
        rule_id="F001",
        description="Never write to evidence directories",
        check_fn=check,
    )


def _make_f002() -> FeasibilityRule:
    """F002: Block destructive commands."""

    def check(cmd: list[str], evidence_refs: list[str], output_dir: str) -> tuple[bool, str]:
        if not cmd:
            return True, ""
        if cmd[0] in _DESTRUCTIVE_COMMANDS:
            return False, f"F002: destructive command '{cmd[0]}' is not permitted"
        return True, ""

    return FeasibilityRule(
        rule_id="F002",
        description="Block destructive commands (rm, shred, mkfs, fdisk, wipefs)",
        check_fn=check,
    )


def _make_f003() -> FeasibilityRule:
    """F003: Block network commands (air-gap constraint)."""

    def check(cmd: list[str], evidence_refs: list[str], output_dir: str) -> tuple[bool, str]:
        if not cmd:
            return True, ""
        if cmd[0] in _NETWORK_COMMANDS:
            return False, f"F003: network command '{cmd[0]}' is not permitted (air-gap policy)"
        return True, ""

    return FeasibilityRule(
        rule_id="F003",
        description="Block network commands (curl, wget, nc, ncat, ssh, scp, rsync) — air-gap",
        check_fn=check,
    )


# ---------------------------------------------------------------------------
# FeasibilityMemory
# ---------------------------------------------------------------------------


class FeasibilityMemory:
    """Holds a list of FeasibilityRules and runs them as a safety gate."""

    def __init__(self) -> None:
        self.rules: list[FeasibilityRule] = []
        self._init_default_rules()

    def _init_default_rules(self) -> None:
        self.rules.append(_make_f001())
        self.rules.append(_make_f002())
        self.rules.append(_make_f003())

    def add_rule(self, rule: FeasibilityRule) -> None:
        self.rules.append(rule)

    def check(
        self,
        cmd: list[str],
        evidence_refs: list[str],
        output_dir: str,
    ) -> tuple[bool, list[str]]:
        """Run all rules against the command.

        Returns (all_passed, list_of_violation_strings).
        """
        violations: list[str] = []
        for rule in self.rules:
            passed, message = rule.check_fn(cmd, evidence_refs, output_dir)
            if not passed:
                violations.append(message)
        return (len(violations) == 0, violations)

    def save(self, path: Path) -> None:
        """Persist rule metadata (rule_id, description) to JSON.

        check_fn callables are not serialisable, so only metadata is stored.
        On load, default rules are re-created from code.
        """
        data = {
            "rules": [
                {"rule_id": r.rule_id, "description": r.description}
                for r in self.rules
            ]
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FeasibilityMemory":
        """Restore a FeasibilityMemory.

        Default rules (F001-F003) are always re-initialised from code.
        Custom rule metadata present in the file is available via the saved JSON
        but check_fn logic must be re-registered at runtime.
        """
        # We always start with default rules; the file serves as a record of
        # what was saved (including any custom rule IDs) but cannot restore
        # arbitrary callables.
        instance = cls()
        return instance
