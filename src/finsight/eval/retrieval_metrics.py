"""Retrieval-quality metrics for the embedding benchmark — the RAG analogue
of precision/recall used to compare embedding models before picking one."""

from __future__ import annotations

from dataclasses import dataclass

from finsight.eval.queries import EvalQuery
from finsight.vectorstore.base import QueryResult


def _is_relevant(result: QueryResult, q: EvalQuery) -> bool:
    return result.metadata.get("ticker") == q.ticker and result.metadata.get("section_key") == q.section_key


def _matches_keyword(result: QueryResult, q: EvalQuery) -> bool:
    text = result.metadata.get("text", "").lower()
    return any(kw.lower() in text for kw in q.keywords)


@dataclass(frozen=True)
class QueryEvalResult:
    query: str
    hit_at_k: int
    reciprocal_rank: float
    keyword_hit_at_k: int


def evaluate_query(q: EvalQuery, results: list[QueryResult]) -> QueryEvalResult:
    hit = 0
    rr = 0.0
    keyword_hit = 0
    for rank, r in enumerate(results, start=1):
        if _is_relevant(r, q):
            hit = 1
            rr = max(rr, 1.0 / rank)
        if _matches_keyword(r, q):
            keyword_hit = 1
    return QueryEvalResult(query=q.query, hit_at_k=hit, reciprocal_rank=rr, keyword_hit_at_k=keyword_hit)


def aggregate(results: list[QueryEvalResult]) -> dict[str, float]:
    n = len(results)
    if n == 0:
        return {"hit_rate_at_k": 0.0, "mrr": 0.0, "keyword_hit_rate_at_k": 0.0}
    return {
        "hit_rate_at_k": sum(r.hit_at_k for r in results) / n,
        "mrr": sum(r.reciprocal_rank for r in results) / n,
        "keyword_hit_rate_at_k": sum(r.keyword_hit_at_k for r in results) / n,
    }
