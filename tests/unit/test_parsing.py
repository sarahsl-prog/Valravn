"""Tests for core/parsing.py — parse_llm_json utility."""
from __future__ import annotations

import pytest
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel


class _Simple(BaseModel):
    name: str
    value: int


class TestParseLlmJson:
    def test_plain_json(self):
        from valravn.core.parsing import parse_llm_json

        result = parse_llm_json('{"name": "foo", "value": 42}', _Simple)
        assert result.name == "foo"
        assert result.value == 42

    def test_markdown_fenced_json(self):
        from valravn.core.parsing import parse_llm_json

        text = '```json\n{"name": "bar", "value": 7}\n```'
        result = parse_llm_json(text, _Simple)
        assert result.name == "bar"
        assert result.value == 7

    def test_markdown_fence_without_language(self):
        from valravn.core.parsing import parse_llm_json

        text = '```\n{"name": "baz", "value": 0}\n```'
        result = parse_llm_json(text, _Simple)
        assert result.name == "baz"

    def test_raises_on_invalid_json(self):
        from valravn.core.parsing import parse_llm_json

        with pytest.raises(OutputParserException):
            parse_llm_json("not json at all", _Simple)

    def test_raises_on_valid_json_but_wrong_schema(self):
        from valravn.core.parsing import parse_llm_json

        with pytest.raises(OutputParserException):
            parse_llm_json('{"wrong_field": true}', _Simple)

    def test_raises_on_empty_string(self):
        from valravn.core.parsing import parse_llm_json

        with pytest.raises(OutputParserException):
            parse_llm_json("", _Simple)

    def test_strips_surrounding_whitespace(self):
        from valravn.core.parsing import parse_llm_json

        text = '  \n{"name": "ws", "value": 1}\n  '
        result = parse_llm_json(text, _Simple)
        assert result.name == "ws"
