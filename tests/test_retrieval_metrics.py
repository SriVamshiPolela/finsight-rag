from finsight.eval.queries import EvalQuery
from finsight.eval.retrieval_metrics import aggregate, evaluate_query
from finsight.vectorstore.base import QueryResult

Q = EvalQuery(
    query="What are Acme's risk factors?",
    ticker="ACME",
    section_key="item1a",
    keywords=("supply chain", "competition"),
)


def result(ticker, section_key, text="", score=1.0, id_="x"):
    return QueryResult(id=id_, score=score, metadata={"ticker": ticker, "section_key": section_key, "text": text})


def test_hit_and_mrr_when_relevant_result_is_first():
    results = [result("ACME", "item1a"), result("OTHER", "item1")]
    ev = evaluate_query(Q, results)
    assert ev.hit_at_k == 1
    assert ev.reciprocal_rank == 1.0


def test_mrr_when_relevant_result_is_second():
    results = [result("OTHER", "item1"), result("ACME", "item1a")]
    ev = evaluate_query(Q, results)
    assert ev.hit_at_k == 1
    assert ev.reciprocal_rank == 0.5


def test_no_hit_when_nothing_matches_ticker_and_section():
    results = [result("OTHER", "item1a"), result("ACME", "item7")]
    ev = evaluate_query(Q, results)
    assert ev.hit_at_k == 0
    assert ev.reciprocal_rank == 0.0


def test_keyword_hit_independent_of_ticker_match():
    results = [result("OTHER", "item1", text="risks include supply chain disruption")]
    ev = evaluate_query(Q, results)
    assert ev.hit_at_k == 0
    assert ev.keyword_hit_at_k == 1


def test_aggregate_averages_across_queries():
    from finsight.eval.retrieval_metrics import QueryEvalResult

    results = [
        QueryEvalResult(query="q1", hit_at_k=1, reciprocal_rank=1.0, keyword_hit_at_k=1),
        QueryEvalResult(query="q2", hit_at_k=0, reciprocal_rank=0.0, keyword_hit_at_k=0),
    ]
    metrics = aggregate(results)
    assert metrics["hit_rate_at_k"] == 0.5
    assert metrics["mrr"] == 0.5
    assert metrics["keyword_hit_rate_at_k"] == 0.5


def test_aggregate_empty_results_returns_zeros():
    assert aggregate([]) == {"hit_rate_at_k": 0.0, "mrr": 0.0, "keyword_hit_rate_at_k": 0.0}
