"""LangGraph wiring for the multi-agent router: a router node classifies the
query via Claude tool-calling (the LLM *proposes*), a conditional edge
dispatches to one of four specialist nodes, each of which retrieves context
and calls the LLM again to generate a grounded answer (the backend
*disposes* — retrieval and dispatch are deterministic Python, not left to
the model)."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from finsight.agents.comparison import answer_comparison
from finsight.agents.filing_qa import answer_filing_qa
from finsight.agents.risk_flag import answer_risk_flag
from finsight.agents.router_tools import ROUTER_AGENT_NAMES, ROUTER_SYSTEM_PROMPT, ROUTER_TOOLS
from finsight.agents.summarization import answer_summarization
from finsight.config import ROUTING_LOG_PATH
from finsight.llm.base import LLMProvider
from finsight.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    query: str
    agent: str
    args: dict[str, Any]
    answer: str
    citations: list[dict[str, Any]]


def log_routing_decision(log_path: Path, query: str, agent: str, args: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "query": query,
                    "agent": agent,
                    "args": args,
                }
            )
            + "\n"
        )


def _router_node(llm: LLMProvider, log_path: Path):
    def node(state: AgentState) -> dict[str, Any]:
        response = llm.complete(
            system=ROUTER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": state["query"]}],
            tools=ROUTER_TOOLS,
            force_tool_choice=True,
        )
        if not response.tool_calls:
            raise ValueError("Router did not select an agent")
        call = response.tool_calls[0]
        logger.info("Routed to %s with args %s", call.name, call.input)
        log_routing_decision(log_path, state["query"], call.name, call.input)
        return {"agent": call.name, "args": call.input}

    return node


def _filing_qa_node(llm: LLMProvider, retriever: Retriever):
    def node(state: AgentState) -> dict[str, Any]:
        result = answer_filing_qa(llm, retriever, **state["args"])
        return {"answer": result.answer, "citations": [asdict(c) for c in result.citations]}

    return node


def _comparison_node(llm: LLMProvider, retriever: Retriever):
    def node(state: AgentState) -> dict[str, Any]:
        result = answer_comparison(llm, retriever, **state["args"])
        return {"answer": result.answer, "citations": [asdict(c) for c in result.citations]}

    return node


def _risk_flag_node(llm: LLMProvider, retriever: Retriever):
    def node(state: AgentState) -> dict[str, Any]:
        result = answer_risk_flag(llm, retriever, **state["args"])
        return {"answer": result.answer, "citations": [asdict(c) for c in result.citations]}

    return node


def _summarization_node(llm: LLMProvider):
    def node(state: AgentState) -> dict[str, Any]:
        result = answer_summarization(llm, **state["args"])
        return {"answer": result.answer, "citations": [asdict(c) for c in result.citations]}

    return node


def build_graph(llm: LLMProvider, retriever: Retriever, log_path: Path = ROUTING_LOG_PATH):
    graph = StateGraph(AgentState)
    graph.add_node("router", _router_node(llm, log_path))
    graph.add_node("filing_qa", _filing_qa_node(llm, retriever))
    graph.add_node("comparison", _comparison_node(llm, retriever))
    graph.add_node("risk_flag", _risk_flag_node(llm, retriever))
    graph.add_node("summarization", _summarization_node(llm))

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", lambda state: state["agent"], {name: name for name in ROUTER_AGENT_NAMES})
    for name in ROUTER_AGENT_NAMES:
        graph.add_edge(name, END)

    return graph.compile()
