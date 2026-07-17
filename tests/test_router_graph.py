import json
from unittest.mock import MagicMock

import pytest

from finsight.agents.graph import build_graph, log_routing_decision
from finsight.llm.base import LLMResponse, ToolCall
from finsight.vectorstore.base import QueryResult

AAPL_CHUNK = {
    "ticker": "AAPL",
    "section_key": "item1a",
    "section_title": "Risk Factors",
    "filing_date": "2024-11-01",
    "source_url": "https://example.com/aapl.htm",
    "text": "Apple faces supply chain risk.",
}


def test_log_routing_decision_writes_expected_line(tmp_path):
    log_path = tmp_path / "routing_log.jsonl"
    log_routing_decision(log_path, "What are Apple's risks?", "risk_flag", {"ticker": "AAPL"})

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["query"] == "What are Apple's risks?"
    assert row["agent"] == "risk_flag"
    assert row["args"] == {"ticker": "AAPL"}
    assert "timestamp" in row


def test_graph_routes_to_filing_qa_and_logs_decision(tmp_path):
    log_path = tmp_path / "routing_log.jsonl"
    llm = MagicMock()
    llm.complete.side_effect = [
        LLMResponse(
            text="",
            tool_calls=[ToolCall(name="filing_qa", input={"ticker": "AAPL", "question": "What are the risks?"})],
        ),
        LLMResponse(text="Apple's main risk is supply chain disruption.", tool_calls=[]),
    ]
    retriever = MagicMock()
    retriever.search.return_value = [QueryResult(id="1", score=0.9, metadata=AAPL_CHUNK)]

    graph = build_graph(llm, retriever, log_path=log_path)
    result = graph.invoke({"query": "What are the risks?"})

    assert result["agent"] == "filing_qa"
    assert result["answer"] == "Apple's main risk is supply chain disruption."
    assert result["citations"][0]["ticker"] == "AAPL"
    assert log_path.exists()
    logged = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert logged["agent"] == "filing_qa"


def test_graph_routes_to_summarization_without_retriever_call(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "finsight.agents.summarization.get_section_chunks", lambda ticker, section_key: [AAPL_CHUNK]
    )
    log_path = tmp_path / "routing_log.jsonl"
    llm = MagicMock()
    llm.complete.side_effect = [
        LLMResponse(
            text="", tool_calls=[ToolCall(name="summarization", input={"ticker": "AAPL", "section_key": "item1a"})]
        ),
        LLMResponse(text="Executive summary.", tool_calls=[]),
    ]
    retriever = MagicMock()

    graph = build_graph(llm, retriever, log_path=log_path)
    result = graph.invoke({"query": "Summarize Apple's risk factors"})

    assert result["agent"] == "summarization"
    assert result["answer"] == "Executive summary."
    retriever.search.assert_not_called()


def test_router_node_raises_when_no_tool_call_returned(tmp_path):
    log_path = tmp_path / "routing_log.jsonl"
    llm = MagicMock()
    llm.complete.return_value = LLMResponse(text="I refuse to pick a tool.", tool_calls=[])
    retriever = MagicMock()

    graph = build_graph(llm, retriever, log_path=log_path)
    with pytest.raises(ValueError):
        graph.invoke({"query": "anything"})
