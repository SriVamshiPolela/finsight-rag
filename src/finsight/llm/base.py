"""LLM provider interface — the vendor-abstraction layer. Anthropic is
primary; OpenAI is wired in behind the same interface to demonstrate the
swap without rewriting any agent code."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        force_tool_choice: bool = False,
    ) -> LLMResponse:
        """`tools` are Anthropic-style tool schemas (name/description/input_schema);
        each provider is responsible for translating to its own wire format.
        `force_tool_choice=True` requires the model to call one of the tools
        rather than replying with plain text — used by the router."""
