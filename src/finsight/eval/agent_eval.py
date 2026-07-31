"""Phase 4 full evaluation: routes each labeled case through the real
router+agent graph once, then scores it on routing correctness plus the four
RAGAS-methodology metrics. One graph.invoke() per case (not two) - routing
and generation scoring share the same run so this doesn't double the LLM
call count. Needs a real LLM to run; code-complete and mock-tested until
one is available (see README)."""

from __future__ import annotations

from finsight.eval.agent_cases import AgentEvalCase
from finsight.eval.ragas_metrics import answer_relevancy, context_precision, context_recall, faithfulness
from finsight.llm.base import LLMProvider

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def evaluate_case(llm: LLMProvider, graph, case: AgentEvalCase) -> dict:
    result = graph.invoke({"query": case.query})
    answer = result.get("answer", "")
    contexts = result.get("contexts", [])

    return {
        "id": case.id,
        "expected_agent": case.agent,
        "actual_agent": result.get("agent"),
        "routing_correct": result.get("agent") == case.agent,
        "answer": answer,
        "faithfulness": faithfulness(llm, answer, contexts),
        "answer_relevancy": answer_relevancy(llm, case.query, answer),
        "context_precision": context_precision(llm, case.query, contexts, case.expected_answer),
        "context_recall": context_recall(llm, case.expected_answer, contexts),
    }


def run_full_eval(llm: LLMProvider, graph, cases: list[AgentEvalCase]) -> dict:
    per_case = [evaluate_case(llm, graph, case) for case in cases]
    n = len(per_case) or 1

    aggregate = {"routing_accuracy": sum(c["routing_correct"] for c in per_case) / n}
    for metric in METRIC_NAMES:
        aggregate[metric] = sum(c[metric] for c in per_case) / n

    return {"aggregate": aggregate, "per_case": per_case}
