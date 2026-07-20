import numpy as np
import pytest

from finsight.embeddings.base import EmbeddingModel
from finsight.eval.agent_cases import AgentEvalCase
from finsight.eval.agent_retrieval_check import keyword_hit, retrieve_for_case, run_retrieval_check
from finsight.retrieval.retriever import Retriever
from finsight.vectorstore.faiss_store import FAISSVectorStore

DIM = 4


def unit(vec):
    arr = np.array(vec, dtype="float32")
    return arr / np.linalg.norm(arr)


class FakeEmbeddingModel(EmbeddingModel):
    name = "fake"
    dimension = DIM

    def __init__(self, vector):
        self._vector = vector

    def embed_passages(self, texts):
        raise NotImplementedError

    def embed_query(self, text):
        return self._vector


def make_retriever():
    store = FAISSVectorStore(dimension=DIM, index_dir=None)
    store.upsert(
        ["aapl-risk", "msft-risk"],
        np.stack([unit([1, 0, 0, 0]), unit([0.9, 0.1, 0, 0])]),
        [
            {"ticker": "AAPL", "section_key": "item1a", "text": "aapl supply chain risk"},
            {"ticker": "MSFT", "section_key": "item1a", "text": "msft cybersecurity risk"},
        ],
    )
    model = FakeEmbeddingModel(unit([1, 0, 0, 0]))
    return Retriever(model, store, default_top_k=5)


def test_retrieve_for_case_filing_qa():
    case = AgentEvalCase(id="x", agent="filing_qa", query="anything", args={"ticker": "AAPL"}, expected_answer="", keywords=())
    chunks = retrieve_for_case(make_retriever(), case)
    assert all(c["ticker"] == "AAPL" for c in chunks)


def test_retrieve_for_case_risk_flag_filters_to_risk_section():
    case = AgentEvalCase(id="x", agent="risk_flag", query="q", args={"ticker": "MSFT"}, expected_answer="", keywords=())
    chunks = retrieve_for_case(make_retriever(), case)
    assert all(c["ticker"] == "MSFT" and c["section_key"] == "item1a" for c in chunks)


def test_retrieve_for_case_comparison_merges_per_ticker():
    case = AgentEvalCase(
        id="x", agent="comparison", query="q", args={"tickers": ["AAPL", "MSFT"], "topic": "risk"},
        expected_answer="", keywords=(),
    )
    chunks = retrieve_for_case(make_retriever(), case)
    assert {c["ticker"] for c in chunks} == {"AAPL", "MSFT"}


def test_retrieve_for_case_summarization_uses_corpus_lookup(monkeypatch):
    monkeypatch.setattr(
        "finsight.eval.agent_retrieval_check.get_section_chunks",
        lambda ticker, section_key: [{"ticker": ticker, "section_key": section_key, "text": "full section"}],
    )
    case = AgentEvalCase(
        id="x", agent="summarization", query="q", args={"ticker": "AAPL", "section_key": "item1"},
        expected_answer="", keywords=(),
    )
    chunks = retrieve_for_case(make_retriever(), case)
    assert chunks == [{"ticker": "AAPL", "section_key": "item1", "text": "full section"}]


def test_retrieve_for_case_unknown_agent_raises():
    case = AgentEvalCase(id="x", agent="not-a-real-agent", query="q", args={}, expected_answer="", keywords=())
    with pytest.raises(ValueError):
        retrieve_for_case(make_retriever(), case)


def test_keyword_hit_case_insensitive_across_chunks():
    case = AgentEvalCase(id="x", agent="filing_qa", query="q", args={}, expected_answer="", keywords=("Supply Chain",))
    assert keyword_hit(case, [{"text": "risk of supply chain disruption"}]) is True
    assert keyword_hit(case, [{"text": "unrelated content"}]) is False


def test_run_retrieval_check_aggregates_hit_rate():
    hit_case = AgentEvalCase(id="hit", agent="filing_qa", query="q", args={"ticker": "AAPL"}, expected_answer="", keywords=("supply",))
    miss_case = AgentEvalCase(id="miss", agent="filing_qa", query="q", args={"ticker": "AAPL"}, expected_answer="", keywords=("nonexistent-term",))

    result = run_retrieval_check([hit_case, miss_case], retriever=make_retriever())

    assert result["hit_rate"] == 0.5
    assert result["per_case"][0]["keyword_hit"] is True
    assert result["per_case"][1]["keyword_hit"] is False
