"""Second LLM provider behind the same interface as Anthropic — demonstrates
the vendor-abstraction pattern: agents call LLMProvider.complete() and don't
know or care which vendor is behind it."""

from __future__ import annotations

import json
from typing import Any

import openai

from finsight.llm.base import LLMProvider, LLMResponse, ToolCall


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required to use the openai provider")
        self.model = model
        self._client = openai.OpenAI(api_key=api_key)

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        force_tool_choice: bool = False,
    ) -> LLMResponse:
        oai_messages = [{"role": "system", "content": system}, *messages]
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = [_to_openai_tool(t) for t in tools]
            if force_tool_choice:
                kwargs["tool_choice"] = "required"

        response = self._client.chat.completions.create(
            model=self.model, messages=oai_messages, **kwargs
        )
        choice = response.choices[0].message
        tool_calls = [
            ToolCall(name=tc.function.name, input=json.loads(tc.function.arguments))
            for tc in (choice.tool_calls or [])
        ]
        return LLMResponse(text=choice.content or "", tool_calls=tool_calls)


def _to_openai_tool(anthropic_tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": anthropic_tool["name"],
            "description": anthropic_tool["description"],
            "parameters": anthropic_tool["input_schema"],
        },
    }
