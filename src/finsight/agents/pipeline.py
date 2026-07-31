from __future__ import annotations

from finsight.agents.graph import build_graph
from finsight.llm.registry import get_llm_provider
from finsight.retrieval.retriever import Retriever


def run_query(query: str, llm_provider_name: str | None = None) -> dict:
    """Wires up the LLM provider, retriever, and router graph, then answers
    one query end-to-end. Requires a real API key for the chosen provider."""
    llm = get_llm_provider(llm_provider_name)
    retriever = Retriever.load()
    graph = build_graph(llm, retriever)
    return graph.invoke({"query": query})
