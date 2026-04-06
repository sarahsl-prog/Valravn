from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from valravn.training.rcl_loop import RCLTrainer
from valravn.training.reflector import ReflectionDiagnostic


def test_rcl_loop_initializes_components(tmp_path: Path) -> None:
    """All three components are created as non-None instances on first init."""
    trainer = RCLTrainer(state_dir=tmp_path)
    assert trainer.playbook is not None
    assert trainer.optimizer_state is not None
    assert trainer.replay_buffer is not None


def test_rcl_loop_saves_state(tmp_path: Path) -> None:
    """save_state creates playbook.json, optimizer_state.json, replay_buffer.json."""
    trainer = RCLTrainer(state_dir=tmp_path)
    trainer.save_state()
    assert (tmp_path / "playbook.json").exists()
    assert (tmp_path / "optimizer_state.json").exists()
    assert (tmp_path / "replay_buffer.json").exists()


def test_rcl_loop_loads_existing_state(tmp_path: Path) -> None:
    """A second RCLTrainer created from the same state_dir loads the first's state."""
    trainer1 = RCLTrainer(state_dir=tmp_path)
    trainer1.playbook.add_entry("rule-001", "Always verify hashes", "Integrity check", iteration=0)
    trainer1.optimizer_state.record_change(0, "ADD rule-001")
    trainer1.replay_buffer.add_failure("case-abc", {"case_id": "case-abc", "summary": "test"})
    trainer1._iteration = 5
    trainer1.save_state()

    trainer2 = RCLTrainer(state_dir=tmp_path)
    assert "rule-001" in trainer2.playbook.entries
    assert len(trainer2.optimizer_state.change_ledger) == 1
    assert "case-abc" in trainer2.replay_buffer.buffer
    assert trainer2._iteration == 5


def test_rcl_loop_processes_case_result(tmp_path: Path) -> None:
    """process_investigation_result calls reflector/mutator on actionable_gap."""
    mock_diagnostic = ReflectionDiagnostic(
        attribution="actionable_gap",
        root_cause="Agent skipped hash verification step",
        coverage_gap="Playbook lacks mandatory hash-check rule",
    )

    trainer = RCLTrainer(state_dir=tmp_path)
    initial_version = trainer.playbook.version

    with (
        patch(
            "valravn.training.rcl_loop.reflect_on_trajectory",
            return_value=mock_diagnostic,
        ) as mock_reflect,
        patch("valravn.training.rcl_loop.apply_mutation") as mock_mutate,
    ):
        result = trainer.process_investigation_result(
            case_id="case-001",
            success_trace="Agent ran hash check -> analysis -> report",
            failure_trace="Agent skipped hash check -> wrong conclusion",
            success=False,
        )

    # Reflector was called
    mock_reflect.assert_called_once()
    # Mutator was called because attribution == "actionable_gap"
    mock_mutate.assert_called_once()
    # Playbook version was incremented
    assert trainer.playbook.version == initial_version + 1
    # Failed case was added to replay buffer
    assert "case-001" in trainer.replay_buffer.buffer
    # Diagnostic returned
    assert result is mock_diagnostic
    # Iteration was incremented
    assert trainer._iteration == 1
