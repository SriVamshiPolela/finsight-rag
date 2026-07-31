from __future__ import annotations

from typing import Any

import anthropic

from finsight.llm.base import LLMProvider, LLMResponse, ToolCall

MAX_TOKENS = 1024


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required to use the anthropic provider")
        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        force_tool_choice: bool = False,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = tools
            if force_tool_choice:
                kwargs["tool_choice"] = {"type": "any"}

        response = self._client.messages.create(
            model=self.model, system=system, messages=messages, max_tokens=MAX_TOKENS, **kwargs
        )

        text = "\n".join(block.text for block in response.content if block.type == "text")
        tool_calls = [
            ToolCall(name=block.name, input=block.input)
            for block in response.content
            if block.type == "tool_use"
        ]
        return LLMResponse(text=text, tool_calls=tool_calls)
