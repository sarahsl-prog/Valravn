"""Unit tests for DeltaLogic belief revision testing (Task 13)."""

from __future__ import annotations

from valravn.evaluation.belief_revision import (
    BeliefRevisionTester,
    EditType,
    RevisionEpisode,
    classify_failure_mode,
)


# ---------------------------------------------------------------------------
# test_revision_episode_construction
# ---------------------------------------------------------------------------


def test_revision_episode_construction():
    """Create a RevisionEpisode manually and verify its edit_type."""
    episode = RevisionEpisode(
        original_premises=["Host contacted unknown IP.", "No prior alerts."],
        query="Is this traffic suspicious?",
        original_label="UNLIKELY",
        edit_type=EditType.SUPPORT_INSERTION,
        edited_premises=[
            "Host contacted unknown IP.",
            "No prior alerts.",
            "Threat intel confirms domain is known C2.",
        ],
        revised_label="LIKELY",
    )

    assert episode.edit_type == EditType.SUPPORT_INSERTION
    assert episode.original_label == "UNLIKELY"
    assert episode.revised_label == "LIKELY"
    assert len(episode.edited_premises) == 3


# ---------------------------------------------------------------------------
# test_default_security_episodes_exist
# ---------------------------------------------------------------------------


def test_default_security_episodes_exist():
    """build_security_episodes returns at least 4 episodes covering all EditTypes."""
    tester = BeliefRevisionTester()
    episodes = tester.build_security_episodes()

    assert len(episodes) >= 4

    present_types = {ep.edit_type for ep in episodes}
    assert EditType.SUPPORT_INSERTION in present_types
    assert EditType.DEFEATING_FACT in present_types
    assert EditType.SUPPORT_REMOVAL in present_types
    assert EditType.IRRELEVANT_ADDITION in present_types


# ---------------------------------------------------------------------------
# test_failure_mode_classification
# ---------------------------------------------------------------------------


def test_failure_mode_classification():
    """Verify inertia, over_flip, and 'none' cases from classify_failure_mode."""
    # Case 1: revised answer is correct — no failure
    result = classify_failure_mode(
        initial_correct=True,
        revised_correct=True,
        revised_answer="the answer is LIKELY based on the new evidence",
        original_label="UNLIKELY",
        revised_label="LIKELY",
        edit_type=EditType.SUPPORT_INSERTION,
    )
    assert result == "none"

    # Case 2: initial was already wrong — not attributable, report "none"
    result = classify_failure_mode(
        initial_correct=False,
        revised_correct=False,
        revised_answer="UNLIKELY",
        original_label="UNLIKELY",
        revised_label="LIKELY",
        edit_type=EditType.SUPPORT_INSERTION,
    )
    assert result == "none"

    # Case 3: irrelevant addition caused a flip — over_flip
    result = classify_failure_mode(
        initial_correct=True,
        revised_correct=False,
        revised_answer="LIKELY",
        original_label="MEDIUM",
        revised_label="MEDIUM",
        edit_type=EditType.IRRELEVANT_ADDITION,
    )
    assert result == "over_flip"

    # Case 4: model keeps original label in revised answer despite defeating fact — inertia
    result = classify_failure_mode(
        initial_correct=True,
        revised_correct=False,
        revised_answer="the answer remains YES as before",
        original_label="YES",
        revised_label="NO",
        edit_type=EditType.DEFEATING_FACT,
    )
    assert result == "inertia"
