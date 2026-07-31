from unittest.mock import MagicMock

from finsight.eval.agent_cases import AgentEvalCase
from finsight.eval.agent_eval import run_full_eval

CASE_A = AgentEvalCase(
    id="a",
    agent="filing_qa",
    query="q1",
    args={"ticker": "AAPL", "question": "q1"},
    expected_answer="ref answer",
    keywords=("x",),
)
CASE_B = AgentEvalCase(
    id="b",
    agent="risk_flag",
    query="q2",
    args={"ticker": "MSFT"},
    expected_answer="ref answer 2",
    keywords=("y",),
)


class FakeGraph:
    def __init__(self, responses: dict[str, dict]):
        self.responses = responses

    def invoke(self, state):
        return self.responses[state["query"]]


def test_run_full_eval_flags_routing_correctness(monkeypatch):
    monkeypatch.setattr("finsight.eval.agent_eval.faithfulness", lambda *a, **k: 0.8)
    monkeypatch.setattr("finsight.eval.agent_eval.answer_relevancy", lambda *a, **k: 0.7)
    monkeypatch.setattr("finsight.eval.agent_eval.context_precision", lambda *a, **k: 0.6)
    monkeypatch.setattr("finsight.eval.agent_eval.context_recall", lambda *a, **k: 0.5)

    graph = FakeGraph(
        {
            "q1": {"agent": "filing_qa", "answer": "answer 1", "contexts": ["ctx1"]},
            "q2": {"agent": "comparison", "answer": "answer 2", "contexts": ["ctx2"]},  # wrong agent
        }
    )

    result = run_full_eval(MagicMock(), graph, [CASE_A, CASE_B])

    assert result["per_case"][0]["routing_correct"] is True
    assert result["per_case"][1]["routing_correct"] is False
    assert result["aggregate"]["routing_accuracy"] == 0.5


def test_run_full_eval_averages_metrics_across_cases(monkeypatch):
    monkeypatch.setattr("finsight.eval.agent_eval.faithfulness", lambda llm, answer, contexts: 1.0)
    monkeypatch.setattr("finsight.eval.agent_eval.answer_relevancy", lambda llm, question, answer: 1.0)
    monkeypatch.setattr("finsight.eval.agent_eval.context_precision", lambda llm, question, contexts, ref: 1.0)
    monkeypatch.setattr("finsight.eval.agent_eval.context_recall", lambda llm, ref, contexts: 1.0)

    graph = FakeGraph(
        {
            "q1": {"agent": "filing_qa", "answer": "a1", "contexts": []},
            "q2": {"agent": "risk_flag", "answer": "a2", "contexts": []},
        }
    )

    result = run_full_eval(MagicMock(), graph, [CASE_A, CASE_B])

    assert result["aggregate"]["faithfulness"] == 1.0
    assert result["aggregate"]["routing_accuracy"] == 1.0
    assert len(result["per_case"]) == 2


def test_run_full_eval_empty_cases_returns_zeroed_aggregate():
    result = run_full_eval(MagicMock(), FakeGraph({}), [])
    assert result["aggregate"]["routing_accuracy"] == 0.0
    assert result["per_case"] == []
