from unittest.mock import MagicMock

from finsight.agents.comparison import answer_comparison
from finsight.agents.filing_qa import answer_filing_qa
from finsight.agents.risk_flag import RISK_SECTION_KEY, answer_risk_flag
from finsight.agents.summarization import answer_summarization
from finsight.llm.base import LLMResponse
from finsight.vectorstore.base import QueryResult

AAPL_CHUNK = {
    "ticker": "AAPL",
    "section_key": "item1a",
    "section_title": "Risk Factors",
    "filing_date": "2024-11-01",
    "source_url": "https://example.com/aapl.htm",
    "text": "Apple faces supply chain risk.",
}
MSFT_CHUNK = {
    "ticker": "MSFT",
    "section_key": "item1a",
    "section_title": "Risk Factors",
    "filing_date": "2024-07-30",
    "source_url": "https://example.com/msft.htm",
    "text": "Microsoft faces cybersecurity risk.",
}


def make_llm(text="the answer"):
    llm = MagicMock()
    llm.complete.return_value = LLMResponse(text=text, tool_calls=[])
    return llm


def test_filing_qa_passes_ticker_filter_and_builds_citations():
    retriever = MagicMock()
    retriever.search.return_value = [QueryResult(id="1", score=0.9, metadata=AAPL_CHUNK)]
    llm = make_llm("Apple's risk is supply chain disruption.")

    result = answer_filing_qa(llm, retriever, ticker="AAPL", question="What are Apple's risks?")

    retriever.search.assert_called_once_with("What are Apple's risks?", ticker="AAPL")
    assert result.agent == "filing_qa"
    assert result.answer == "Apple's risk is supply chain disruption."
    assert result.citations[0].ticker == "AAPL"
    user_message = llm.complete.call_args.kwargs["messages"][0]["content"]
    assert "supply chain risk" in user_message


def test_comparison_searches_each_ticker_separately_and_merges_citations():
    retriever = MagicMock()
    retriever.search.side_effect = [
        [QueryResult(id="1", score=0.9, metadata=AAPL_CHUNK)],
        [QueryResult(id="2", score=0.8, metadata=MSFT_CHUNK)],
    ]
    llm = make_llm()

    result = answer_comparison(llm, retriever, tickers=["AAPL", "MSFT"], topic="cybersecurity risk")

    assert retriever.search.call_count == 2
    retriever.search.assert_any_call("cybersecurity risk", ticker="AAPL")
    retriever.search.assert_any_call("cybersecurity risk", ticker="MSFT")
    assert {c.ticker for c in result.citations} == {"AAPL", "MSFT"}


def test_risk_flag_filters_to_risk_section():
    retriever = MagicMock()
    retriever.search.return_value = [QueryResult(id="1", score=0.9, metadata=AAPL_CHUNK)]
    llm = make_llm()

    answer_risk_flag(llm, retriever, ticker="AAPL")

    kwargs = retriever.search.call_args.kwargs
    assert kwargs["ticker"] == "AAPL"
    assert kwargs["section_key"] == RISK_SECTION_KEY == "item1a"


def test_summarization_uses_full_section_not_similarity_search(monkeypatch):
    captured = {}

    def fake_get_section_chunks(ticker, section_key):
        captured["args"] = (ticker, section_key)
        return [AAPL_CHUNK]

    monkeypatch.setattr("finsight.agents.summarization.get_section_chunks", fake_get_section_chunks)
    llm = make_llm("Executive summary text.")

    result = answer_summarization(llm, ticker="AAPL", section_key="item1a")

    assert captured["args"] == ("AAPL", "item1a")
    assert result.agent == "summarization"
    assert result.answer == "Executive summary text."
