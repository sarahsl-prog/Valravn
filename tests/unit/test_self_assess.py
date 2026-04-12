"""Tests for nodes/self_assess.py — assess_progress node."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage


def _make_state(current_step_id=None, invocations=None, self_assessments=None):
    return {
        "plan": None,
        "current_step_id": current_step_id,
        "invocations": invocations or [],
        "_self_assessments": self_assessments or [],
    }


class TestParseAssessment:
    """Tests for _parse_assessment helper."""

    def test_plain_json(self):
        from valravn.nodes.self_assess import _parse_assessment

        result = _parse_assessment('{"assessment": "Good progress", "polarity": "positive"}')
        assert result.assessment == "Good progress"
        assert result.polarity == "positive"

    def test_markdown_fenced_json(self):
        from valravn.nodes.self_assess import _parse_assessment

        text = '```json\n{"assessment": "Stalled", "polarity": "negative"}\n```'
        result = _parse_assessment(text)
        assert result.polarity == "negative"

    def test_invalid_polarity_defaults_to_neutral(self):
        from valravn.nodes.self_assess import _parse_assessment

        result = _parse_assessment('{"assessment": "ok", "polarity": "unknown_value"}')
        assert result.polarity == "neutral"

    def test_fallback_key_value_lines(self):
        from valravn.nodes.self_assess import _parse_assessment

        text = "assessment: Investigation looks good\npolarity: positive"
        result = _parse_assessment(text)
        assert result.polarity == "positive"

    def test_completely_unparseable_defaults_to_neutral(self):
        from valravn.nodes.self_assess import _parse_assessment

        result = _parse_assessment("not structured at all")
        assert result.polarity == "neutral"
        assert result.assessment != ""


class TestAssessProgressNode:
    """Tests for the assess_progress LangGraph node."""

    def test_returns_existing_assessments_when_no_current_step(self):
        from valravn.nodes.self_assess import assess_progress

        state = _make_state(current_step_id=None, self_assessments=[{"existing": True}])
        result = assess_progress(state)
        assert result["_self_assessments"] == [{"existing": True}]

    @patch("valravn.nodes.self_assess.get_llm")
    def test_appends_new_assessment_on_success(self, mock_get_llm):
        from valravn.nodes.self_assess import assess_progress

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(
            content='{"assessment": "Making progress", "polarity": "positive"}'
        )
        mock_get_llm.return_value = mock_llm

        state = _make_state(current_step_id="step-abc123")
        result = assess_progress(state)

        assert len(result["_self_assessments"]) == 1
        entry = result["_self_assessments"][0]
        assert entry["polarity"] == "positive"
        assert entry["assessment"] == "Making progress"
        assert entry["scalar_reward"] > 0

    @patch("valravn.nodes.self_assess.get_llm")
    def test_llm_failure_returns_existing_assessments_without_crash(self, mock_get_llm):
        from valravn.nodes.self_assess import assess_progress

        mock_get_llm.side_effect = RuntimeError("LLM unavailable")

        existing = [{"assessment": "prior", "polarity": "neutral", "scalar_reward": 0.0}]
        state = _make_state(current_step_id="step-abc123", self_assessments=existing)
        result = assess_progress(state)

        # Should not raise — returns prior assessments unchanged
        assert result["_self_assessments"] == existing

    @patch("valravn.nodes.self_assess.get_llm")
    def test_negative_polarity_produces_negative_scalar_reward(self, mock_get_llm):
        from valravn.nodes.self_assess import assess_progress

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(
            content='{"assessment": "Stalled", "polarity": "negative"}'
        )
        mock_get_llm.return_value = mock_llm

        state = _make_state(current_step_id="step-xyz")
        result = assess_progress(state)

        assert result["_self_assessments"][0]["scalar_reward"] < 0

    @patch("valravn.nodes.self_assess.get_llm")
    def test_accumulates_across_multiple_calls(self, mock_get_llm):
        from valravn.nodes.self_assess import assess_progress

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(
            content='{"assessment": "OK", "polarity": "neutral"}'
        )
        mock_get_llm.return_value = mock_llm

        existing = [{"assessment": "first", "polarity": "positive", "scalar_reward": 1.0}]
        state = _make_state(current_step_id="step-001", self_assessments=existing)
        result = assess_progress(state)

        assert len(result["_self_assessments"]) == 2
