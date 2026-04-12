from __future__ import annotations

import pytest


def test_get_llm_raises_when_no_model_configured(monkeypatch):
    """C-05: _resolve_model_chain must raise ValueError when no model is configured.

    No DEFAULT_MODELS fallback should exist — the user must explicitly configure
    a model via env var or config.yaml.
    """
    monkeypatch.delenv("VALRAVN_PLAN_MODEL", raising=False)

    from valravn.core.llm_factory import _resolve_model_chain

    with pytest.raises(ValueError, match="No model configured"):
        _resolve_model_chain(module="plan")


def test_get_llm_uses_env_var(monkeypatch):
    """_resolve_model_chain returns the env var value when set."""
    monkeypatch.setenv("VALRAVN_PLAN_MODEL", "anthropic:claude-haiku-4-5-20251001")

    from valravn.core.llm_factory import _resolve_model_chain

    chain = _resolve_model_chain(module="plan")
    assert chain == ["anthropic:claude-haiku-4-5-20251001"]


def test_get_llm_uses_comma_separated_chain(monkeypatch):
    """_resolve_model_chain splits comma-separated env var into a list."""
    monkeypatch.setenv(
        "VALRAVN_PLAN_MODEL",
        "anthropic:claude-sonnet-4-6,anthropic:claude-haiku-4-5-20251001",
    )

    from valravn.core.llm_factory import _resolve_model_chain

    chain = _resolve_model_chain(module="plan")
    assert chain == [
        "anthropic:claude-sonnet-4-6",
        "anthropic:claude-haiku-4-5-20251001",
    ]


def test_get_llm_explicit_provider_model(monkeypatch):
    """_resolve_model_chain with explicit provider_model ignores env var."""
    monkeypatch.setenv("VALRAVN_PLAN_MODEL", "openai:gpt-4o")

    from valravn.core.llm_factory import _resolve_model_chain

    chain = _resolve_model_chain(provider_model="anthropic:claude-sonnet-4-6")
    assert chain == ["anthropic:claude-sonnet-4-6"]
