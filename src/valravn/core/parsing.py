"""Shared LLM output parsing utilities.

Provides tolerant JSON parsing for models (like Ollama proxies) that may wrap
their output in markdown fences despite being asked for plain JSON.
"""
from __future__ import annotations

import json
import re
from typing import TypeVar

from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def parse_llm_json(text: str, model_cls: type[T]) -> T:
    """Parse an LLM text response into a Pydantic model.

    Accepts:
    - Plain JSON: {"key": "value"}
    - Markdown-fenced JSON: ```json\\n{...}\\n```

    Raises:
        OutputParserException: if the text cannot be parsed or validated.
    """
    stripped = text.strip()

    # Strip markdown code fences
    fenced = re.sub(r"^```[a-zA-Z]*\n?", "", stripped)
    fenced = re.sub(r"\n?```$", "", fenced).strip()

    last_exc: Exception | None = None
    for candidate in (fenced, stripped):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
            return model_cls.model_validate(data)
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            last_exc = exc

    raise OutputParserException(
        f"Invalid json output: {text[:300]}",
        llm_output=text,
    ) from last_exc
