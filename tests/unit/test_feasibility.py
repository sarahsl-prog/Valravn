from __future__ import annotations

import json
from pathlib import Path

import pytest

from valravn.training.feasibility import FeasibilityMemory, FeasibilityRule


def test_feasibility_default_rules_exist():
    mem = FeasibilityMemory()
    assert len(mem.rules) > 0
    rule_ids = {r.rule_id for r in mem.rules}
    assert "F001" in rule_ids
    assert "F002" in rule_ids
    assert "F003" in rule_ids


def test_feasibility_passes_safe_command():
    mem = FeasibilityMemory()
    evidence_refs = ["/mnt/evidence/disk.img"]
    output_dir = "/tmp/output"
    # vol3 command writing to output_dir — should pass all rules
    cmd = ["vol3", "-f", "/mnt/evidence/disk.img", "-o", output_dir, "windows.pslist"]
    passed, violations = mem.check(cmd, evidence_refs, output_dir)
    assert passed is True
    assert violations == []


def test_feasibility_blocks_write_to_evidence():
    mem = FeasibilityMemory()
    evidence_refs = ["/mnt/evidence/disk.img"]
    output_dir = "/tmp/output"
    # cp with destination inside evidence directory
    cmd = ["cp", "/tmp/output/result.txt", "/mnt/evidence/result.txt"]
    passed, violations = mem.check(cmd, evidence_refs, output_dir)
    assert passed is False
    assert len(violations) > 0


def test_feasibility_blocks_destructive_commands():
    mem = FeasibilityMemory()
    evidence_refs = ["/mnt/evidence/disk.img"]
    output_dir = "/tmp/output"
    cmd = ["rm", "-rf", "/tmp/output/old_results"]
    passed, violations = mem.check(cmd, evidence_refs, output_dir)
    assert passed is False
    assert len(violations) > 0


def test_feasibility_add_custom_rule():
    mem = FeasibilityMemory()

    def block_yara_on_mnt(cmd: list[str], evidence_refs: list[str], output_dir: str) -> tuple[bool, str]:
        if cmd[0] == "yara" and any(arg.startswith("/mnt/") for arg in cmd[1:]):
            return False, "F999: yara scans on /mnt/ are not permitted"
        return True, ""

    custom_rule = FeasibilityRule(
        rule_id="F999",
        description="Block yara on /mnt/",
        check_fn=block_yara_on_mnt,
    )
    mem.add_rule(custom_rule)

    rule_ids = {r.rule_id for r in mem.rules}
    assert "F999" in rule_ids

    cmd = ["yara", "rules.yar", "/mnt/evidence/disk.img"]
    passed, violations = mem.check(cmd, ["/mnt/evidence/disk.img"], "/tmp/output")
    assert passed is False
    assert any("F999" in v for v in violations)


def test_feasibility_save_and_load_custom_rules(tmp_path: Path):
    mem = FeasibilityMemory()

    def custom_check(cmd: list[str], evidence_refs: list[str], output_dir: str) -> tuple[bool, str]:
        return True, ""

    mem.add_rule(FeasibilityRule(rule_id="F100", description="Custom rule", check_fn=custom_check))
    rule_count = len(mem.rules)

    save_path = tmp_path / "feasibility.json"
    mem.save(save_path)

    loaded = FeasibilityMemory.load(save_path)
    # load restores default rules; custom rule metadata is preserved in file
    # but check_fn cannot be serialised — so count should be >= default count
    assert len(loaded.rules) > 0
    # The saved file must be valid JSON
    data = json.loads(save_path.read_text())
    assert "rules" in data
    # The custom rule metadata should be in the saved file
    saved_ids = [r["rule_id"] for r in data["rules"]]
    assert "F100" in saved_ids
