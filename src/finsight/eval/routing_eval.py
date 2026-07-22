"""Routing accuracy: does the LangGraph router pick the right specialist
agent (and the right args) for each labeled eval case? Needs a real LLM to
run the router - code-complete and mock-tested, not yet run live."""

from __future__ import annotations

from finsight.eval.agent_cases import AgentEvalCase


def evaluate_routing(graph, cases: list[AgentEvalCase]) -> dict:
    per_case = []
    for case in cases:
        result = graph.invoke({"query": case.query})
        correct = result.get("agent") == case.agent
        per_case.append(
            {
                "id": case.id,
                "expected_agent": case.agent,
                "actual_agent": result.get("agent"),
                "correct": correct,
            }
        )

    accuracy = sum(r["correct"] for r in per_case) / len(per_case) if per_case else 0.0
    return {"accuracy": accuracy, "per_case": per_case}
