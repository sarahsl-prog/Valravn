"""IRC-style reward calibration for multi-turn agent training.

Data-driven reward design: no LLM calls, pure statistical computation.
Uses point-biserial correlation to derive tier rewards from rollout outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from scipy.stats import pointbiserialr


# ---------------------------------------------------------------------------
# Action tier classification
# ---------------------------------------------------------------------------

EXPECTED_SIGN: dict[str, float] = {
    "evidence_gather": 1.0,   # should correlate positively with success
    "error": -1.0,            # should correlate negatively with success
    "duplicate": -1.0,
    "self_correction": 1.0,
    "anomaly_detected": 1.0,
}


class ActionTierClassifier:
    """Classify a single tool invocation into a named action tier."""

    TIERS = {
        "evidence_gather": "Successful read-only forensic tool execution",
        "error": "Failed tool invocation",
        "duplicate": "Redundant call repeating a previous action",
        "self_correction": "Modified command after prior failure",
        "anomaly_detected": "Tool run that revealed an anomaly",
    }

    def classify(
        self,
        tool_name: str,
        exit_code: int,
        had_output: bool,
        history: list[dict] | None = None,
        cmd: list[str] | None = None,
    ) -> str:
        """Return the tier label for this tool invocation.

        Priority order:
          1. Non-zero exit code → "error"
          2. Command matches a previous history entry → "duplicate"
          3. Had output (or exit_code == 0) → "evidence_gather"
          4. Default → "evidence_gather"
        """
        if exit_code != 0:
            return "error"

        if history and cmd is not None:
            for entry in history:
                if entry.get("cmd") == cmd:
                    return "duplicate"

        return "evidence_gather"


# ---------------------------------------------------------------------------
# Rollout record
# ---------------------------------------------------------------------------


@dataclass
class RolloutRecord:
    """A single episode record used for reward calibration."""

    tiers: list[str]
    success: bool
    case_id: str


# ---------------------------------------------------------------------------
# Iterative reward calibrator
# ---------------------------------------------------------------------------


class IterativeRewardCalibrator:
    """Derive per-tier rewards from rollout data using point-biserial correlation.

    For each tier, computes the point-biserial correlation between tier presence
    (binary: appeared in this rollout at least once) and episode success.
    The reward is ``alpha * corr`` when ``|corr| > threshold``, else 0.

    Iterates up to ``iterations`` times, flipping the sign of any tier whose
    reward contradicts the expected alignment (e.g. error tier rewarded
    positively), until no mismatches remain or iterations are exhausted.
    """

    def __init__(self, alpha: float = 1.0, threshold: float = 0.05) -> None:
        self.alpha = alpha
        self.threshold = threshold
        self.tier_rewards: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calibrate(
        self,
        rollouts: list[RolloutRecord],
        iterations: int = 3,
    ) -> dict[str, float]:
        """Compute calibrated rewards from rollout data.

        Parameters
        ----------
        rollouts:
            List of completed episode records.
        iterations:
            Maximum number of alignment correction passes.

        Returns
        -------
        dict mapping tier name → reward value.
        """
        for _ in range(iterations):
            rewards = self._compute_rewards(rollouts)
            mismatches = self._check_alignment(rewards)
            self.tier_rewards = rewards
            if not mismatches:
                break
            # Flip sign for mismatched tiers and iterate
            for tier in mismatches:
                if tier in rewards:
                    rewards[tier] = -rewards[tier]
            self.tier_rewards = rewards

        return self.tier_rewards

    def score_turn(self, tier: str) -> float:
        """Return the calibrated reward for a tier (0.0 if unknown)."""
        return self.tier_rewards.get(tier, 0.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_rewards(self, rollouts: list[RolloutRecord]) -> dict[str, float]:
        """Compute point-biserial correlation rewards for all observed tiers."""
        # Collect all tier names present across rollouts
        all_tiers: set[str] = set()
        for rollout in rollouts:
            all_tiers.update(rollout.tiers)

        success_flags = [1 if r.success else 0 for r in rollouts]

        rewards: dict[str, float] = {}
        for tier in all_tiers:
            presence = [1 if tier in r.tiers else 0 for r in rollouts]
            # pointbiserialr requires at least two distinct values in each array
            if len(set(presence)) < 2 or len(set(success_flags)) < 2:
                rewards[tier] = 0.0
                continue
            corr, _ = pointbiserialr(success_flags, presence)
            rewards[tier] = self.alpha * corr if abs(corr) > self.threshold else 0.0

        return rewards

    def _check_alignment(self, rewards: dict[str, float]) -> list[str]:
        """Return tiers whose reward sign contradicts the expected alignment."""
        mismatches: list[str] = []
        for tier, reward in rewards.items():
            expected = EXPECTED_SIGN.get(tier)
            if expected is None or reward == 0.0:
                continue
            if expected > 0 and reward < 0:
                mismatches.append(tier)
            elif expected < 0 and reward > 0:
                mismatches.append(tier)
        return mismatches
