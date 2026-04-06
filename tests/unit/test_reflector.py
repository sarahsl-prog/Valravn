from __future__ import annotations

from unittest.mock import MagicMock, patch

from valravn.training.reflector import ReflectionDiagnostic, reflect_on_trajectory


def test_reflector_produces_structured_diagnostic():
    """reflect_on_trajectory returns a ReflectionDiagnostic with actionable_gap attribution."""
    mock_result = ReflectionDiagnostic(
        attribution="actionable_gap",
        root_cause="Agent failed to run hash verification step before moving to analysis",
        coverage_gap="Playbook lacks a mandatory hash-check rule at investigation start",
    )

    with patch("valravn.training.reflector._get_reflector_llm") as mock_llm_fn:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_result
        mock_llm_fn.return_value = mock_llm

        result = reflect_on_trajectory(
            success_trace="Agent ran hash check -> analysis -> report",
            failure_trace="Agent skipped hash check -> analysis -> report (integrity flag missed)",
            playbook_context="## Security Playbook\n\n(empty)",
        )

    assert isinstance(result, ReflectionDiagnostic)
    assert result.attribution == "actionable_gap"
    assert result.root_cause == "Agent failed to run hash verification step before moving to analysis"
    assert "hash-check" in result.coverage_gap


def test_reflector_intractable_attribution():
    """reflect_on_trajectory with intractable attribution has empty coverage_gap."""
    mock_result = ReflectionDiagnostic(
        attribution="intractable",
        root_cause="Evidence is encrypted and cannot be processed without decryption key",
        coverage_gap="",
    )

    with patch("valravn.training.reflector._get_reflector_llm") as mock_llm_fn:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_result
        mock_llm_fn.return_value = mock_llm

        result = reflect_on_trajectory(
            success_trace="N/A — no comparable success trace available",
            failure_trace="Agent attempted all playbook steps but evidence is fully encrypted",
            playbook_context="## Security Playbook\n\n- **rule-001**: Always verify evidence integrity",
        )

    assert isinstance(result, ReflectionDiagnostic)
    assert result.attribution == "intractable"
    assert result.coverage_gap == ""
