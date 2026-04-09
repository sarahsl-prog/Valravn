"""Unified LLM factory with automatic fallback chains.

Supports:
- anthropic: Claude models via langchain-anthropic
- openai: OpenAI models via langchain-openai
- ollama: Local models via langchain-ollama
- openrouter: Any model via OpenRouter's OpenAI-compatible API

Each module can be configured with a list of models. If the primary model
fails (timeout, connection error, API error), the next model in the chain
is tried automatically via LangChain's with_fallbacks().

Usage:
    llm = get_llm(module="plan")                  # uses configured chain
    llm = get_llm("anthropic:claude-sonnet-4-6")  # explicit single model
"""

from __future__ import annotations

import os

from loguru import logger

# Default model chains per module — tried in order on failure.
# A single string is also valid (no fallback).
DEFAULT_MODELS: dict[str, list[str]] = {
    "anomaly": ["ollama:kimi-k2.5:cloud", "ollama:qwen3:14b"],
    "conclusions": ["ollama:kimi-k2.5:cloud", "ollama:qwen3:14b"],
    "plan": ["ollama:kimi-k2.5:cloud", "ollama:qwen3:14b"],
    "self_assess": ["ollama:kimi-k2.5:cloud", "ollama:qwen3:14b"],
    "tool_runner": ["ollama:kimi-k2.5:cloud", "ollama:qwen3:14b"],
    "reflector": ["ollama:kimi-k2.5:cloud", "ollama:qwen3:14b"],
    "mutator": ["ollama:kimi-k2.5:cloud", "ollama:qwen3:14b"],
}


def _resolve_model_chain(
    provider_model: str | None = None,
    module: str | None = None,
) -> list[str]:
    """Resolve the ordered list of models to try for a given call.

    Priority:
    1. Explicit provider_model argument (single model, no fallback)
    2. VALRAVN_{MODULE}_MODEL env var (comma-separated for chains)
    3. DEFAULT_MODELS[module] list
    """
    if provider_model is not None:
        return [provider_model]

    if module:
        env_var = f"VALRAVN_{module.upper()}_MODEL"
        env_val = os.getenv(env_var)
        if env_val:
            # Comma-separated list from env var or config injection
            return [m.strip() for m in env_val.split(",") if m.strip()]

        default = DEFAULT_MODELS.get(module)
        if default:
            return list(default)

    raise ValueError(
        f"No model configured for module {module!r}. "
        f"Set VALRAVN_{module.upper() if module else '?'}_MODEL env var, "
        f"add it to config.yaml, or pass provider_model directly."
    )


def get_llm(
    provider_model: str | None = None,
    module: str | None = None,
    temperature: float = 0,
) -> object:
    """Get an LLM client, optionally with automatic fallback.

    When multiple models are configured for a module (via config.yaml,
    env var, or DEFAULT_MODELS), returns a LangChain RunnableWithFallbacks
    that tries each model in order on failure.

    Args:
        provider_model: Explicit "provider:model" string (no fallback).
        module: Module name for config lookup (e.g., "plan", "reflector").
        temperature: Sampling temperature (0 = deterministic).

    Returns:
        LangChain chat model, optionally wrapped with fallbacks.
    """
    chain = _resolve_model_chain(provider_model, module)

    if len(chain) == 1:
        return _create_llm_for(chain[0], temperature)

    # Build primary + fallbacks
    primary = _create_llm_for(chain[0], temperature)
    fallbacks = []
    for model_spec in chain[1:]:
        try:
            fallbacks.append(_create_llm_for(model_spec, temperature))
        except (ImportError, ValueError) as exc:
            logger.warning("Skipping fallback {} (not available: {})", model_spec, exc)

    if not fallbacks:
        return primary

    logger.debug(
        "LLM chain for {}: {} → {}",
        module or "direct",
        chain[0],
        " → ".join(chain[1:]),
    )
    return primary.with_fallbacks(fallbacks)


def _create_llm_for(provider_model: str, temperature: float) -> object:
    """Create a single LLM client from a provider:model string."""
    if ":" not in provider_model:
        raise ValueError(
            f"Model must be in format 'provider:model', got: {provider_model!r}. "
            f"Examples: 'anthropic:claude-sonnet-4-6', 'openai:gpt-4o', 'ollama:llama3'"
        )

    provider, model = provider_model.split(":", 1)
    provider = provider.lower().strip()
    model = model.strip()

    logger.debug("Creating {} LLM: {}", provider, model)

    if provider == "anthropic":
        return _create_anthropic_llm(model, temperature)
    elif provider == "openai":
        return _create_openai_llm(model, temperature)
    elif provider == "ollama":
        return _create_ollama_llm(model, temperature)
    elif provider == "openrouter":
        return _create_openrouter_llm(model, temperature)
    else:
        raise ValueError(
            f"Unsupported provider: {provider!r}. Supported: anthropic, openai, ollama, openrouter"
        )


def _create_anthropic_llm(model: str, temperature: float) -> object:
    """Create Anthropic Claude LLM."""
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as e:
        raise ImportError(
            "langchain-anthropic package required for Anthropic models. "
            "Install: pip install langchain-anthropic"
        ) from e

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set")

    return ChatAnthropic(
        model=model,
        temperature=temperature,
        anthropic_api_key=api_key,
    )


def _create_openai_llm(model: str, temperature: float) -> object:
    """Create OpenAI LLM."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise ImportError(
            "langchain-openai package required for OpenAI models. "
            "Install: pip install langchain-openai"
        ) from e

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set")

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
    )


def _create_ollama_llm(model: str, temperature: float) -> object:
    """Create Ollama local LLM."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError as e:
        raise ImportError(
            "langchain-ollama package required for Ollama models. "
            "Install: pip install langchain-ollama"
        ) from e

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    return ChatOllama(model=model, temperature=temperature, base_url=base_url)


def _create_openrouter_llm(model: str, temperature: float) -> object:
    """Create OpenRouter LLM via OpenAI-compatible API."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise ImportError(
            "langchain-openai package required for OpenRouter models. "
            "Install: pip install langchain-openai"
        ) from e

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.warning("OPENROUTER_API_KEY not set")

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


def get_default_model(module: str) -> str:
    """Get the first (primary) model for a module.

    Args:
        module: Module name (e.g., "plan", "reflector").

    Returns:
        Provider:model string for the primary model.
    """
    chain = _resolve_model_chain(module=module)
    return chain[0]
