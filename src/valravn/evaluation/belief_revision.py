"""DeltaLogic belief revision testing.

Tests whether an agent (or any reasoning system) correctly updates its
conclusions when evidence changes.  No LLM calls are made here — this module
provides pure data structures and classification logic.

Reference: DeltaLogic paper on belief revision evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# ---------------------------------------------------------------------------
# EditType
# ---------------------------------------------------------------------------


class EditType(Enum):
    """The kind of edit applied to the premise set in a revision episode."""

    SUPPORT_INSERTION = "support_insertion"
    """A new premise is added that supports a different conclusion."""

    DEFEATING_FACT = "defeating_fact"
    """A new premise is added that defeats (rebuts) the original conclusion."""

    SUPPORT_REMOVAL = "support_removal"
    """An existing supporting premise is removed, weakening the conclusion."""

    IRRELEVANT_ADDITION = "irrelevant_addition"
    """A new premise is added that is logically irrelevant to the query."""


# ---------------------------------------------------------------------------
# RevisionEpisode
# ---------------------------------------------------------------------------


@dataclass
class RevisionEpisode:
    """A single belief-revision test case.

    Attributes:
        original_premises: The premise set before the edit.
        query: The yes/no or categorical question posed to the agent.
        original_label: Expected answer given the original premises.
        edit_type: The category of change applied.
        edited_premises: The premise set after the edit.
        revised_label: Expected answer given the edited premises.
    """

    original_premises: list[str]
    query: str
    original_label: str
    edit_type: EditType
    edited_premises: list[str]
    revised_label: str


# ---------------------------------------------------------------------------
# classify_failure_mode
# ---------------------------------------------------------------------------

_UNCERTAINTY_WORDS = frozenset(
    {
        "uncertain",
        "unclear",
        "unknown",
        "cannot determine",
        "can't determine",
        "not sure",
        "unsure",
        "inconclusive",
        "ambiguous",
        "insufficient",
    }
)


def classify_failure_mode(
    initial_correct: bool,
    revised_correct: bool,
    revised_answer: str,
    original_label: str,
    revised_label: str,
    edit_type: EditType,
) -> str:
    """Classify why a model failed to revise its belief correctly.

    Returns one of:
        "none"        — no failure (revised is correct, or initial was already wrong)
        "over_flip"   — model changed answer in response to an irrelevant edit
        "inertia"     — model did not update despite compelling new evidence
        "abstention"  — model hedged instead of committing to the revised label

    Args:
        initial_correct: Whether the model answered correctly *before* the edit.
        revised_correct: Whether the model answered correctly *after* the edit.
        revised_answer:  The model's free-text answer after the edit (lowercase
                         comparison is used internally).
        original_label:  The ground-truth label before the edit.
        revised_label:   The ground-truth label after the edit.
        edit_type:       The kind of edit that was applied.
    """
    if revised_correct:
        return "none"

    if not initial_correct:
        return "none"

    # At this point: the model was correct before the edit but wrong after.
    if edit_type is EditType.IRRELEVANT_ADDITION:
        return "over_flip"

    answer_lower = revised_answer.lower()

    if original_label.lower() in answer_lower:
        return "inertia"

    for word in _UNCERTAINTY_WORDS:
        if word in answer_lower:
            return "abstention"

    return "inertia"


# ---------------------------------------------------------------------------
# BeliefRevisionTester
# ---------------------------------------------------------------------------


class BeliefRevisionTester:
    """Generates and manages belief-revision test episodes for security scenarios."""

    def build_security_episodes(self) -> list[RevisionEpisode]:
        """Return a canonical set of security-domain revision episodes.

        One episode is provided for each EditType:

        1. SUPPORT_INSERTION  — new threat-intel confirms C2 domain
                                UNLIKELY -> LIKELY
        2. DEFEATING_FACT     — a configuration control mitigates the
                                vulnerability under assessment
                                YES -> NO
        3. SUPPORT_REMOVAL    — scheduled backup explains the data transfer,
                                removing the suspicious premise
                                LIKELY -> UNLIKELY
        4. IRRELEVANT_ADDITION — unrelated IT maintenance activity is added;
                                answer should remain unchanged
                                MEDIUM -> MEDIUM
        """
        return [
            # 1. SUPPORT_INSERTION
            RevisionEpisode(
                original_premises=[
                    "Workstation WS-42 made outbound connections to 198.51.100.7.",
                    "The destination IP has no prior reputation entry.",
                    "Connection occurred at 03:14 local time.",
                ],
                query="Is the outbound connection likely malicious C2 traffic?",
                original_label="UNLIKELY",
                edit_type=EditType.SUPPORT_INSERTION,
                edited_premises=[
                    "Workstation WS-42 made outbound connections to 198.51.100.7.",
                    "The destination IP has no prior reputation entry.",
                    "Connection occurred at 03:14 local time.",
                    "Updated threat intel feed confirms 198.51.100.7 is a known C2 domain "
                    "associated with APT-29.",
                ],
                revised_label="LIKELY",
            ),
            # 2. DEFEATING_FACT
            RevisionEpisode(
                original_premises=[
                    "Service nginx/1.18 is running on port 443.",
                    "CVE-2023-44487 (HTTP/2 Rapid Reset) affects nginx < 1.25.3.",
                    "No patch has been applied to this host.",
                ],
                query="Is this host vulnerable to CVE-2023-44487?",
                original_label="YES",
                edit_type=EditType.DEFEATING_FACT,
                edited_premises=[
                    "Service nginx/1.18 is running on port 443.",
                    "CVE-2023-44487 (HTTP/2 Rapid Reset) affects nginx < 1.25.3.",
                    "No patch has been applied to this host.",
                    "WAF rule set WAF-2023-1045 is active and fully mitigates "
                    "CVE-2023-44487 for this deployment.",
                ],
                revised_label="NO",
            ),
            # 3. SUPPORT_REMOVAL
            RevisionEpisode(
                original_premises=[
                    "FileServer-01 transferred 120 GB of data to an external IP at 02:00.",
                    "No prior large transfers have been observed for this host.",
                    "The destination IP is not in the corporate allow-list.",
                ],
                query="Does this data transfer indicate a likely data exfiltration event?",
                original_label="LIKELY",
                edit_type=EditType.SUPPORT_REMOVAL,
                edited_premises=[
                    "FileServer-01 transferred 120 GB of data to an external IP at 02:00.",
                    "The destination IP is not in the corporate allow-list.",
                    # Removed: "No prior large transfers have been observed."
                    # Added supporting context that weakens the original conclusion:
                    "Backup logs confirm a scheduled off-site backup job ran at 02:00 "
                    "and the destination IP belongs to the contracted backup provider.",
                ],
                revised_label="UNLIKELY",
            ),
            # 4. IRRELEVANT_ADDITION
            RevisionEpisode(
                original_premises=[
                    "User account svc_deploy has elevated privileges on the build server.",
                    "The account was last used 48 hours ago.",
                    "No anomalous login attempts have been detected.",
                ],
                query="What is the current risk level of svc_deploy account compromise?",
                original_label="MEDIUM",
                edit_type=EditType.IRRELEVANT_ADDITION,
                edited_premises=[
                    "User account svc_deploy has elevated privileges on the build server.",
                    "The account was last used 48 hours ago.",
                    "No anomalous login attempts have been detected.",
                    # Irrelevant: unrelated to svc_deploy or the build server
                    "The office printer queue on floor 3 was restarted by IT at 09:00.",
                ],
                revised_label="MEDIUM",
            ),
        ]
