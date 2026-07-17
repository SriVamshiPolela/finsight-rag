import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from finsight.llm.anthropic_provider import AnthropicProvider
from finsight.llm.openai_provider import OpenAIProvider, _to_openai_tool

TOOLS = [
    {
        "name": "filing_qa",
        "description": "Answer a question about a filing.",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}, "question": {"type": "string"}},
            "required": ["ticker", "question"],
        },
    }
]


def block(type_, **kwargs):
    return SimpleNamespace(type=type_, **kwargs)


class TestAnthropicProvider:
    def test_requires_api_key(self):
        with pytest.raises(ValueError):
            AnthropicProvider(api_key="", model="claude-sonnet-5")

    def test_complete_extracts_text(self):
        with patch("finsight.llm.anthropic_provider.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = SimpleNamespace(
                content=[block("text", text="hello world")]
            )
            mock_cls.return_value = mock_client

            provider = AnthropicProvider(api_key="fake", model="claude-sonnet-5")
            response = provider.complete(system="sys", messages=[{"role": "user", "content": "hi"}])

        assert response.text == "hello world"
        assert response.tool_calls == []

    def test_complete_extracts_tool_calls_and_passes_tool_choice(self):
        with patch("finsight.llm.anthropic_provider.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = SimpleNamespace(
                content=[block("tool_use", name="filing_qa", input={"ticker": "AAPL", "question": "?"})]
            )
            mock_cls.return_value = mock_client

            provider = AnthropicProvider(api_key="fake", model="claude-sonnet-5")
            response = provider.complete(
                system="sys", messages=[{"role": "user", "content": "hi"}], tools=TOOLS, force_tool_choice=True
            )

        assert response.tool_calls[0].name == "filing_qa"
        assert response.tool_calls[0].input == {"ticker": "AAPL", "question": "?"}
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["tool_choice"] == {"type": "any"}
        assert call_kwargs["tools"] == TOOLS


class TestOpenAIProvider:
    def test_requires_api_key(self):
        with pytest.raises(ValueError):
            OpenAIProvider(api_key="", model="gpt-4o-mini")

    def test_complete_extracts_text_and_prepends_system_message(self):
        with patch("finsight.llm.openai_provider.openai.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="hello", tool_calls=None))]
            )
            mock_cls.return_value = mock_client

            provider = OpenAIProvider(api_key="fake", model="gpt-4o-mini")
            response = provider.complete(system="sys", messages=[{"role": "user", "content": "hi"}])

        assert response.text == "hello"
        sent_messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
        assert sent_messages[0] == {"role": "system", "content": "sys"}

    def test_complete_extracts_tool_calls(self):
        with patch("finsight.llm.openai_provider.openai.OpenAI") as mock_cls:
            mock_client = MagicMock()
            tc = SimpleNamespace(
                function=SimpleNamespace(name="filing_qa", arguments=json.dumps({"ticker": "AAPL", "question": "?"}))
            )
            mock_client.chat.completions.create.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[tc]))]
            )
            mock_cls.return_value = mock_client

            provider = OpenAIProvider(api_key="fake", model="gpt-4o-mini")
            response = provider.complete(
                system="sys", messages=[{"role": "user", "content": "hi"}], tools=TOOLS, force_tool_choice=True
            )

        assert response.tool_calls[0].name == "filing_qa"
        assert response.tool_calls[0].input == {"ticker": "AAPL", "question": "?"}
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["tool_choice"] == "required"

    def test_tool_schema_translation(self):
        openai_tool = _to_openai_tool(TOOLS[0])
        assert openai_tool["type"] == "function"
        assert openai_tool["function"]["name"] == "filing_qa"
        assert openai_tool["function"]["parameters"] == TOOLS[0]["input_schema"]
