from __future__ import annotations

from valravn.training.self_guide import POLARITY_REWARD_MAP, SelfGuidanceSignal, trust_coefficient

# ---------------------------------------------------------------------------
# trust_coefficient schedule tests
# ---------------------------------------------------------------------------

def test_trust_schedule_warmup():
    """Phase I (0-20%): steps 0-19 of 100 should return 0.0."""
    for step in range(20):
        assert trust_coefficient(step, 100) == 0.0, f"step={step} expected 0.0"


def test_trust_schedule_activation():
    """Phase II (20-40%): step 30 of 100 (midpoint of ramp) should be ~0.5."""
    result = trust_coefficient(30, 100)
    assert abs(result - 0.5) < 1e-9, f"expected ~0.5, got {result}"


def test_trust_schedule_full_strength():
    """Phase III (40-80%): steps 40-79 of 100 should return 1.0."""
    for step in range(40, 80):
        assert trust_coefficient(step, 100) == 1.0, f"step={step} expected 1.0"


def test_trust_schedule_annealing():
    """Phase IV (80-100%): step 90 of 100 should be strictly between 0 and 1."""
    result = trust_coefficient(90, 100)
    assert 0.0 < result < 1.0, f"expected (0, 1), got {result}"


def test_trust_schedule_boundary():
    """Last step (step 99 of 100) should be >= 0.0 (no negative values)."""
    result = trust_coefficient(99, 100)
    assert result >= 0.0, f"expected >= 0.0, got {result}"


# ---------------------------------------------------------------------------
# SelfGuidanceSignal structure tests
# ---------------------------------------------------------------------------

def test_self_guidance_signal_structure():
    """SelfGuidanceSignal fields are set correctly from construction."""
    sig = SelfGuidanceSignal(
        assessment="Investigation is progressing well; key artifacts identified.",
        polarity="positive",
        scalar_reward=POLARITY_REWARD_MAP["positive"],
    )
    assert sig.assessment == "Investigation is progressing well; key artifacts identified."
    assert sig.polarity == "positive"
    assert sig.scalar_reward == 0.1

    sig_neg = SelfGuidanceSignal(
        assessment="No useful output; tool failed repeatedly.",
        polarity="negative",
        scalar_reward=POLARITY_REWARD_MAP["negative"],
    )
    assert sig_neg.polarity == "negative"
    assert sig_neg.scalar_reward == -0.1

    sig_neu = SelfGuidanceSignal(
        assessment="Tool ran but results are ambiguous.",
        polarity="neutral",
        scalar_reward=POLARITY_REWARD_MAP["neutral"],
    )
    assert sig_neu.polarity == "neutral"
    assert sig_neu.scalar_reward == 0.0
