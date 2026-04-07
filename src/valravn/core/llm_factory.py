"""Unified LLM factory supporting multiple providers.

Supports:
- anthropic: Claude models via langchain-anthropic
- openai: OpenAI models via langchain-openai
- ollama: Local models via langchain-ollama
- openrouter: Any model via OpenRouter's OpenAI-compatible API

Usage:
    llm = get_llm("anthropic:claude-3-5-sonnet-20241022")
    structured_llm = get_llm("openai:gpt-4o", output_schema=MySchema)
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel

_LOGGER = logging.getLogger(__name__)

# Default model configurations per module
DEFAULT_MODELS = {
    "anomaly": "anthropic:claude-3-5-sonnet-20241022",
    "conclusions": "anthropic:claude-3-5-sonnet-20241022",
    "plan": "anthropic:claude-3-5-sonnet-20241022",
    "self_assess": "anthropic:claude-3-5-sonnet-20241022",
    "tool_runner": "anthropic:claude-3-5-sonnet-20241022",
    "reflector": "anthropic:claude-3-5-sonnet-20241022",
    "mutator": "anthropic:claude-3-5-sonnet-20241022",
}


def get_llm(
    provider_model: str | None = None,
    module: str | None = None,
    output_schema: type[BaseModel] | None = None,
    temperature: float = 0,
) -> object:
    """Get an LLM client for the specified provider and model.

    Args:
        provider_model: Provider and model in format "provider:model".
                       If None, uses VALRAVN_*_MODEL env var or default.
        module: Module name (e.g., "plan", "reflector") for env var lookup.
        output_schema: Optional Pydantic model for structured output.
        temperature: Sampling temperature (0 = deterministic).

    Returns:
        Configured LangChain chat model, optionally with structured output.

    Raises:
        ValueError: If provider is unsupported or required env vars missing.
        ImportError: If provider package not installed.

    Examples:
        >>> llm = get_llm("anthropic:claude-3-5-sonnet-20241022")
        >>> llm = get_llm("openai:gpt-4o", temperature=0.7)
        >>> llm = get_llm("ollama:llama3.2", output_schema=MySchema)
        >>> llm = get_llm("openrouter:anthropic/claude-3-opus")
    """
    # Resolve provider_model from env var if not provided
    if provider_model is None:
        if module:
            env_var = f"VALRAVN_{module.upper()}_MODEL"
            provider_model = os.getenv(env_var, DEFAULT_MODELS.get(module))
        if provider_model is None:
            raise ValueError(
                f"No model configured for module '{module}'. "
                f"Set {env_var} env var or pass provider_model directly."
            )

    # Parse provider:model format
    if ":" not in provider_model:
        raise ValueError(
            f"Model must be in format 'provider:model', got: {provider_model!r}. "
            f"Examples: 'anthropic:claude-3-opus', 'openai:gpt-4o', 'ollama:llama3'"
        )

    provider, model = provider_model.split(":", 1)
    provider = provider.lower().strip()
    model = model.strip()

    _LOGGER.debug("Initializing %s LLM with model %s", provider, model)

    # Create LLM based on provider
    if provider == "anthropic":
        llm = _create_anthropic_llm(model, temperature)
    elif provider == "openai":
        llm = _create_openai_llm(model, temperature)
    elif provider == "ollama":
        llm = _create_ollama_llm(model, temperature)
    elif provider == "openrouter":
        llm = _create_openrouter_llm(model, temperature)
    else:
        raise ValueError(
            f"Unsupported provider: {provider!r}. "
            f"Supported: anthropic, openai, ollama, openrouter"
        )

    # Wrap with structured output if schema provided
    if output_schema is not None:
        return llm.with_structured_output(output_schema)

    return llm


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
        _LOGGER.warning("ANTHROPIC_API_KEY not set")

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
        _LOGGER.warning("OPENAI_API_KEY not set")

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

    return ChatOllama(
        model=model,
        temperature=temperature,
        base_url=base_url,
    )


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
        _LOGGER.warning("OPENROUTER_API_KEY not set")

    # OpenRouter uses OpenAI-compatible endpoint
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


def get_default_model(module: str) -> str:
    """Get the default model configuration for a module.

    Args:
        module: Module name (e.g., "plan", "reflector").

    Returns:
        Provider:model string or env var override.
    """
    env_var = f"VALRAVN_{module.upper()}_MODEL"
    return os.getenv(env_var, DEFAULT_MODELS.get(module, "anthropic:claude-3-5-sonnet-20241022"))
