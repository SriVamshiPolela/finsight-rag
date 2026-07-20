"""Live, key-free half of the Phase 4 eval: for each labeled case, runs the
same retrieval each specialist agent would use (without the LLM generation
step) and checks whether the expected keywords actually show up in what got
retrieved. Doesn't need an LLM - only the Phase 2 FAISS index."""

from __future__ import annotations

from finsight.agents.risk_flag import RISK_SECTION_KEY
from finsight.eval.agent_cases import AgentEvalCase
from finsight.retrieval.corpus import get_section_chunks
from finsight.retrieval.retriever import Retriever

RISK_QUERY = "risk factors, red flags, going concern, litigation, regulatory risk"


def retrieve_for_case(retriever: Retriever, case: AgentEvalCase) -> list[dict]:
    if case.agent == "filing_qa":
        return [r.metadata for r in retriever.search(case.query, ticker=case.args["ticker"])]
    if case.agent == "risk_flag":
        return [
            r.metadata
            for r in retriever.search(RISK_QUERY, ticker=case.args["ticker"], section_key=RISK_SECTION_KEY)
        ]
    if case.agent == "comparison":
        chunks: list[dict] = []
        for ticker in case.args["tickers"]:
            chunks.extend(r.metadata for r in retriever.search(case.args["topic"], ticker=ticker))
        return chunks
    if case.agent == "summarization":
        return get_section_chunks(case.args["ticker"], case.args["section_key"])
    raise ValueError(f"Unknown agent '{case.agent}'")


def keyword_hit(case: AgentEvalCase, chunks: list[dict]) -> bool:
    combined = " ".join(c["text"] for c in chunks).lower()
    return any(kw.lower() in combined for kw in case.keywords)


def run_retrieval_check(cases: list[AgentEvalCase], retriever: Retriever | None = None) -> dict:
    retriever = retriever or Retriever.load()
    per_case = []
    for case in cases:
        chunks = retrieve_for_case(retriever, case)
        per_case.append(
            {
                "id": case.id,
                "agent": case.agent,
                "chunk_count": len(chunks),
                "keyword_hit": keyword_hit(case, chunks),
            }
        )

    hit_rate = sum(r["keyword_hit"] for r in per_case) / len(per_case) if per_case else 0.0
    return {"hit_rate": hit_rate, "per_case": per_case}
