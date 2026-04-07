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
    from valravn.training.replay_buffer import ReplayBuffer
    
    mock_diagnostic = ReflectionDiagnostic(
        attribution="actionable_gap",
        root_cause="Agent skipped hash verification step",
        coverage_gap="Playbook lacks mandatory hash-check rule",
    )

    trainer = RCLTrainer(state_dir=tmp_path)
    trainer.replay_buffer = ReplayBuffer(n_pass=3, n_reject=3)  # Prevent immediate rejection
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


def test_rcl_loop_records_consecutive_failures_correctly(tmp_path: Path) -> None:
    """BUG-001: Replay buffer correctly tracks failure outcomes via record_outcome.
    
    Regression test: Previously, only the first failure was recorded correctly due to
    conditional else-branch. Now always calls record_outcome, ensuring accurate counts.
    
    Flow: add_failure (fails=1) -> record_outcome (fails=2, passes reset to 0)
    """
    from unittest.mock import patch
    from valravn.training.replay_buffer import ReplayBuffer
    
    # Use n_reject=4 to allow testing multiple failures without immediate rejection
    trainer = RCLTrainer(state_dir=tmp_path)
    trainer.replay_buffer = ReplayBuffer(n_pass=3, n_reject=4)  # Allow 4 failures before rejection
    
    with patch("valravn.training.rcl_loop.reflect_on_trajectory") as mock_reflect:
        mock_reflect.return_value = None  # No reflection when traces not both provided
        
        # First failure: add_failure creates entry with fails=1, then record_outcome resets passes=0, fails=2
        trainer.process_investigation_result(
            case_id="case-flaky",
            success_trace="",  # Empty traces = no reflection, straight to replay buffer
            failure_trace="",
            success=False,
        )
    assert "case-flaky" in trainer.replay_buffer.buffer
    assert trainer.replay_buffer.buffer["case-flaky"]["fails"] == 2  # init(1) + record_outcome(+1)
    assert trainer.replay_buffer.buffer["case-flaky"]["passes"] == 0
    
    # Second failure: record_outcome increments fails to 3
    with patch("valravn.training.rcl_loop.reflect_on_trajectory") as mock_reflect:
        mock_reflect.return_value = None
        trainer.process_investigation_result(
            case_id="case-flaky",
            success_trace="",
            failure_trace="",
            success=False,
        )
    assert trainer.replay_buffer.buffer["case-flaky"]["fails"] == 3
    assert trainer.replay_buffer.buffer["case-flaky"]["passes"] == 0
    
    # Third failure: record_outcome increments fails to 4, which is >= n_reject=4, so case is removed
    with patch("valravn.training.rcl_loop.reflect_on_trajectory") as mock_reflect:
        mock_reflect.return_value = None
        trainer.process_investigation_result(
            case_id="case-flaky",
            success_trace="",
            failure_trace="",
            success=False,
        )
    # After reaching n_reject=4, case is removed (rejected from buffer)
    assert "case-flaky" not in trainer.replay_buffer.buffer
