from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from valravn.training.reflector import ReflectionDiagnostic, reflect_on_trajectory


def test_reflector_produces_structured_diagnostic():
    """reflect_on_trajectory returns a ReflectionDiagnostic with actionable_gap attribution."""
    import json
    mock_result = ReflectionDiagnostic(
        attribution="actionable_gap",
        root_cause="Agent failed to run hash verification step before moving to analysis",
        coverage_gap="Playbook lacks a mandatory hash-check rule at investigation start",
    )

    with patch("valravn.training.reflector._get_reflector_llm") as mock_llm_fn:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=json.dumps(mock_result.model_dump()))
        mock_llm_fn.return_value = mock_llm

        result = reflect_on_trajectory(
            success_trace="Agent ran hash check -> analysis -> report",
            failure_trace="Agent skipped hash check -> analysis -> report (integrity flag missed)",
            playbook_context="## Security Playbook\n\n(empty)",
        )

    assert isinstance(result, ReflectionDiagnostic)
    assert result.attribution == "actionable_gap"
    assert result.root_cause == (
        "Agent failed to run hash verification step before moving to analysis"
    )
    assert "hash-check" in result.coverage_gap


def test_reflector_intractable_attribution():
    """reflect_on_trajectory with intractable attribution has empty coverage_gap."""
    import json
    mock_result = ReflectionDiagnostic(
        attribution="intractable",
        root_cause="Evidence is encrypted and cannot be processed without decryption key",
        coverage_gap="",
    )

    with patch("valravn.training.reflector._get_reflector_llm") as mock_llm_fn:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=json.dumps(mock_result.model_dump()))
        mock_llm_fn.return_value = mock_llm

        result = reflect_on_trajectory(
            success_trace="N/A — no comparable success trace available",
            failure_trace="Agent attempted all playbook steps but evidence is fully encrypted",
            playbook_context=(
                "## Security Playbook\n\n- **rule-001**: Always verify evidence integrity"
            ),
        )

    assert isinstance(result, ReflectionDiagnostic)
    assert result.attribution == "intractable"
    assert result.coverage_gap == ""


def test_reflection_diagnostic_valid_attributions():
    """All three valid attribution values should be accepted (case-insensitive, whitespace-tolerant)."""
    for raw in ["actionable_gap", "execution_variance", "intractable"]:
        diag = ReflectionDiagnostic(
            attribution=raw,
            root_cause="Test cause",
            coverage_gap="",
        )
        assert diag.attribution == raw

    # Variations with whitespace and mixed case
    diag_upper = ReflectionDiagnostic(attribution="ACTIONABLE_GAP", root_cause="Test", coverage_gap="")
    assert diag_upper.attribution == "actionable_gap"

    diag_spaced = ReflectionDiagnostic(attribution="  execution_variance  ", root_cause="Test", coverage_gap="")
    assert diag_spaced.attribution == "execution_variance"


def test_reflection_diagnostic_invalid_attribution_raises():
    """BUG-002: Invalid attributions raise ValidationError and log to MLflow."""
    from unittest.mock import patch

    from pydantic import ValidationError

    invalid_values = [
        "actionable_gaps",  # typo
        "ambiguous",  # hallucinated
        "",  # empty
        "execution variance",  # underscore replaced with space
        "InTrackTable",  # completely wrong
    ]

    for invalid in invalid_values:
        with patch("valravn.training.reflector._log_invalid_attribution") as mock_log:
            with pytest.raises(ValidationError) as exc_info:
                ReflectionDiagnostic(
                    attribution=invalid,
                    root_cause="Test cause",
                    coverage_gap="",
                )

            # Verify error message mentions allowed values (wrapped in Pydantic ValidationError)
            error_str = str(exc_info.value)
            assert "actionable_gap" in error_str
            assert invalid in error_str or invalid.strip().lower() in error_str.lower()

            # Verify logging function was called
            mock_log.assert_called_once()
            call_args = mock_log.call_args[0]
            assert call_args[0] == invalid  # raw value
            assert call_args[1] == (invalid.strip().lower() if invalid else "")  # cleaned
