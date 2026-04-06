from __future__ import annotations

import pytest

from valravn.evaluation.reward_calibrator import (
    ActionTierClassifier,
    IterativeRewardCalibrator,
    RolloutRecord,
)


def test_tier_classifier_evidence_gather():
    clf = ActionTierClassifier()
    result = clf.classify(
        tool_name="vol3",
        exit_code=0,
        had_output=True,
    )
    assert result == "evidence_gather"


def test_tier_classifier_error():
    clf = ActionTierClassifier()
    result = clf.classify(
        tool_name="vol3",
        exit_code=1,
        had_output=False,
    )
    assert result == "error"


def test_tier_classifier_duplicate():
    clf = ActionTierClassifier()
    history = [{"cmd": ["vol3", "-f", "memory.dmp", "windows.pslist"]}]
    result = clf.classify(
        tool_name="vol3",
        exit_code=0,
        had_output=True,
        history=history,
        cmd=["vol3", "-f", "memory.dmp", "windows.pslist"],
    )
    assert result == "duplicate"


def test_calibrator_produces_rewards():
    calibrator = IterativeRewardCalibrator(alpha=1.0, threshold=0.05)
    rollouts = [
        RolloutRecord(tiers=["evidence_gather", "evidence_gather"], success=True, case_id="c1"),
        RolloutRecord(tiers=["evidence_gather", "error"], success=False, case_id="c2"),
        RolloutRecord(tiers=["evidence_gather", "evidence_gather"], success=True, case_id="c3"),
        RolloutRecord(tiers=["error", "error"], success=False, case_id="c4"),
    ]
    rewards = calibrator.calibrate(rollouts)

    assert isinstance(rewards, dict)
    assert "evidence_gather" in rewards
    assert "error" in rewards
    assert rewards["evidence_gather"] >= 0.0
    assert rewards["error"] <= 0.0


def test_calibrator_score_turn():
    calibrator = IterativeRewardCalibrator(alpha=1.0, threshold=0.05)
    rollouts = [
        RolloutRecord(tiers=["evidence_gather"], success=True, case_id="c1"),
        RolloutRecord(tiers=["error"], success=False, case_id="c2"),
    ]
    calibrator.calibrate(rollouts)

    score = calibrator.score_turn("evidence_gather")
    assert isinstance(score, float)

    unknown_score = calibrator.score_turn("nonexistent_tier")
    assert unknown_score == 0.0
