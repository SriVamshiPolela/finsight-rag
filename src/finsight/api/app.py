"""FastAPI service: /health, /filings, /query (full router), /compare
(dedicated comparison endpoint, bypasses routing since the client already
knows it wants a comparison).

The retriever loads at startup regardless (no API key needed - it's just
the Phase 2 FAISS index). The LLM provider only loads if a key is present;
if not, /health and /filings still work, but /query and /compare return a
503 explaining why, rather than crashing the whole app at import time.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from finsight.agents.comparison import answer_comparison
from finsight.agents.graph import build_graph
from finsight.api.schemas import AgentAnswer, CompareRequest, Citation, FilingSummary, HealthResponse, QueryRequest
from finsight.config import MANIFEST_PATH
from finsight.llm.registry import get_llm_provider
from finsight.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.retriever = Retriever.load()
    try:
        app.state.llm = get_llm_provider()
        app.state.graph = build_graph(app.state.llm, app.state.retriever)
    except ValueError as exc:
        logger.warning("LLM provider unavailable - /query and /compare will 503 until fixed: %s", exc)
        app.state.llm = None
        app.state.graph = None
    yield


app = FastAPI(title="FinSight RAG API", lifespan=lifespan)

# Public, read-only demo API with no auth or user data - open CORS so the
# static demo page (web/demo.html) can call it directly from a browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _require_llm(app: FastAPI) -> None:
    if app.state.llm is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No LLM provider configured (missing ANTHROPIC_API_KEY or OPENAI_API_KEY). "
                "/health and /filings still work without one."
            ),
        )


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    return HealthResponse(status="ok", llm_configured=request.app.state.llm is not None)


@app.get("/filings", response_model=list[FilingSummary])
def list_filings() -> list[FilingSummary]:
    if not MANIFEST_PATH.exists():
        raise HTTPException(status_code=503, detail="No filings ingested yet - run scripts/ingest.py first.")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return [
        FilingSummary(
            ticker=m["ticker"],
            sector=m["sector"],
            form=m["form"],
            filing_date=m["filing_date"],
            accession_number=m["accession_number"],
            source_url=m["source_url"],
            chunk_count=m["chunk_count"],
        )
        for m in manifest
    ]


@app.post("/query", response_model=AgentAnswer)
def query(body: QueryRequest, request: Request) -> AgentAnswer:
    _require_llm(request.app)
    result = request.app.state.graph.invoke({"query": body.query})
    return AgentAnswer(
        agent=result["agent"], answer=result["answer"], citations=[Citation(**c) for c in result["citations"]]
    )


@app.post("/compare", response_model=AgentAnswer)
def compare(body: CompareRequest, request: Request) -> AgentAnswer:
    _require_llm(request.app)
    result = answer_comparison(request.app.state.llm, request.app.state.retriever, tickers=body.tickers, topic=body.topic)
    return AgentAnswer(
        agent=result.agent,
        answer=result.answer,
        citations=[
            Citation(ticker=c.ticker, section_title=c.section_title, filing_date=c.filing_date, source_url=c.source_url)
            for c in result.citations
        ],
    )
