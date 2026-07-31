import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from finsight.api.app import app
from finsight.llm.base import LLMResponse

MANIFEST_ENTRY = {
    "ticker": "AAPL",
    "sector": "Technology",
    "form": "10-K",
    "filing_date": "2024-11-01",
    "accession_number": "0000320193-24-000123",
    "source_url": "https://example.com/aapl.htm",
    "raw_path": "data/raw/AAPL_0000320193240001238.html",
    "sections_found": ["item1", "item1a"],
    "chunk_count": 212,
}


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    retriever.search.return_value = []
    return retriever


@pytest.fixture
def client_without_llm(monkeypatch, mock_retriever, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps([MANIFEST_ENTRY]), encoding="utf-8")
    monkeypatch.setattr("finsight.api.app.MANIFEST_PATH", manifest_path)
    monkeypatch.setattr("finsight.api.app.Retriever.load", classmethod(lambda cls: mock_retriever))
    monkeypatch.setattr(
        "finsight.api.app.get_llm_provider", lambda *a, **k: (_ for _ in ()).throw(ValueError("no key"))
    )
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_with_llm(monkeypatch, mock_retriever, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps([MANIFEST_ENTRY]), encoding="utf-8")
    monkeypatch.setattr("finsight.api.app.MANIFEST_PATH", manifest_path)
    monkeypatch.setattr("finsight.api.app.Retriever.load", classmethod(lambda cls: mock_retriever))

    fake_llm = MagicMock()
    fake_llm.complete.return_value = LLMResponse(text="an answer", tool_calls=[])
    monkeypatch.setattr("finsight.api.app.get_llm_provider", lambda *a, **k: fake_llm)

    fake_graph = MagicMock()
    fake_graph.invoke.return_value = {"agent": "filing_qa", "answer": "Apple's risk is X.", "citations": []}
    monkeypatch.setattr("finsight.api.app.build_graph", lambda llm, retriever: fake_graph)

    with TestClient(app) as c:
        yield c


def test_health_reports_llm_unconfigured(client_without_llm):
    resp = client_without_llm.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "llm_configured": False}


def test_health_reports_llm_configured(client_with_llm):
    resp = client_with_llm.get("/health")
    assert resp.status_code == 200
    assert resp.json()["llm_configured"] is True


def test_filings_lists_ingested_manifest(client_without_llm):
    resp = client_without_llm.get("/filings")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["ticker"] == "AAPL"
    assert body[0]["chunk_count"] == 212


def test_query_returns_503_without_llm(client_without_llm):
    resp = client_without_llm.post("/query", json={"query": "What are Apple's risks?"})
    assert resp.status_code == 503
    assert "ANTHROPIC_API_KEY" in resp.json()["detail"]


def test_compare_returns_503_without_llm(client_without_llm):
    resp = client_without_llm.post("/compare", json={"tickers": ["AAPL", "MSFT"], "topic": "risk"})
    assert resp.status_code == 503


def test_query_rejects_empty_query(client_without_llm):
    resp = client_without_llm.post("/query", json={"query": ""})
    assert resp.status_code == 422


def test_compare_rejects_single_ticker(client_without_llm):
    resp = client_without_llm.post("/compare", json={"tickers": ["AAPL"], "topic": "risk"})
    assert resp.status_code == 422


def test_query_returns_routed_answer_when_llm_available(client_with_llm):
    resp = client_with_llm.post("/query", json={"query": "What are Apple's risks?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["agent"] == "filing_qa"
    assert body["answer"] == "Apple's risk is X."


def test_compare_bypasses_router_and_calls_comparison_agent_directly(client_with_llm, monkeypatch):
    from finsight.agents.types import AgentResponse

    fake_response = AgentResponse(agent="comparison", answer="They differ.", citations=[], contexts=[])
    monkeypatch.setattr("finsight.api.app.answer_comparison", lambda llm, retriever, tickers, topic: fake_response)

    resp = client_with_llm.post("/compare", json={"tickers": ["AAPL", "MSFT"], "topic": "risk"})

    assert resp.status_code == 200
    assert resp.json()["answer"] == "They differ."


def test_cors_preflight_allows_cross_origin_post(client_without_llm):
    resp = client_without_llm.options(
        "/query",
        headers={
            "Origin": "http://localhost:5500",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "*"
    assert "POST" in resp.headers["access-control-allow-methods"]


def test_cors_header_present_on_actual_response(client_without_llm):
    resp = client_without_llm.get("/health", headers={"Origin": "http://localhost:5500"})
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "*"
