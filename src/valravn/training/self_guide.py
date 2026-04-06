from __future__ import annotations

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Reward map
# ---------------------------------------------------------------------------

POLARITY_REWARD_MAP: dict[str, float] = {
    "positive": 0.1,
    "neutral": 0.0,
    "negative": -0.1,
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class SelfGuidanceSignal(BaseModel):
    assessment: str
    polarity: str  # "positive", "neutral", "negative"
    scalar_reward: float  # +0.1, 0.0, -0.1


# ---------------------------------------------------------------------------
# Trust schedule
# ---------------------------------------------------------------------------

def trust_coefficient(step: int, total_steps: int) -> float:
    """Return a scalar in [0, 1] controlling self-assessment weight.

    Four-phase schedule:
      Phase I   (0–20%):   constant 0.0  — warm-up, no self-guidance
      Phase II  (20–40%):  linear ramp 0 → 1
      Phase III (40–80%):  constant 1.0  — full self-guidance
      Phase IV  (80–100%): linear ramp 1 → 0  — annealing
    """
    progress = step / total_steps  # normalised in [0, 1)

    if progress < 0.20:
        return 0.0
    elif progress < 0.40:
        # linear ramp: 0.0 at progress=0.20, 1.0 at progress=0.40
        return (progress - 0.20) / 0.20
    elif progress < 0.80:
        return 1.0
    else:
        # linear ramp: 1.0 at progress=0.80, 0.0 at progress=1.00
        return (1.0 - progress) / 0.20
