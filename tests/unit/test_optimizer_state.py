from __future__ import annotations

import json

from valravn.training.optimizer_state import OptimizerState


def test_optimizer_state_defaults():
    state = OptimizerState()
    assert state.phase == "exploratory"
    assert state.change_ledger == []
    assert state.open_hypotheses == []


def test_optimizer_state_record_change():
    state = OptimizerState()
    state.record_change(iteration=1, action="Added rule for port blocking")
    assert len(state.change_ledger) == 1
    assert "1" in state.change_ledger[0]
    assert "Added rule for port blocking" in state.change_ledger[0]


def test_optimizer_state_phase_transition():
    state = OptimizerState()
    assert state.phase == "exploratory"
    state.transition_to_convergent()
    assert state.phase == "convergent"
    state.transition_to_exploratory()
    assert state.phase == "exploratory"


def test_optimizer_state_to_context_is_valid_json():
    state = OptimizerState()
    state.record_change(iteration=1, action="Test action")
    state.add_hypothesis("Maybe port 22 is the culprit")
    context_str = state.to_context()
    parsed = json.loads(context_str)
    assert "phase" in parsed
    assert "changes" in parsed
    assert "hypotheses" in parsed


def test_optimizer_state_ledger_trims_to_recent():
    state = OptimizerState()
    for i in range(15):
        state.record_change(iteration=i, action=f"Action {i}")
    assert len(state.change_ledger) == 15
    context_str = state.to_context()
    parsed = json.loads(context_str)
    # to_context should only include last 10
    assert len(parsed["changes"]) == 10
    # The last 10 should be actions 5-14
    assert "Action 14" in parsed["changes"][-1]
    assert "Action 5" in parsed["changes"][0]


def test_optimizer_state_save_and_load(tmp_path):
    state = OptimizerState(phase="convergent")
    state.record_change(iteration=1, action="Test action")
    state.add_hypothesis("Hypothesis A")
    save_path = tmp_path / "optimizer_state.json"
    state.save(save_path)

    loaded = OptimizerState.load(save_path)
    assert loaded.phase == "convergent"
    assert len(loaded.change_ledger) == 1
    assert "Hypothesis A" in loaded.open_hypotheses
