from finsight.eval.agent_cases import AgentEvalCase
from finsight.eval.routing_eval import evaluate_routing

CASE_A = AgentEvalCase(
    id="a", agent="filing_qa", query="q1", args={"ticker": "AAPL", "question": "q1"},
    expected_answer="ref", keywords=("x",),
)
CASE_B = AgentEvalCase(
    id="b", agent="summarization", query="q2", args={"ticker": "MSFT", "section_key": "item1"},
    expected_answer="ref2", keywords=("y",),
)


class FakeGraph:
    def __init__(self, responses):
        self.responses = responses

    def invoke(self, state):
        return self.responses[state["query"]]


def test_evaluate_routing_all_correct():
    graph = FakeGraph({"q1": {"agent": "filing_qa"}, "q2": {"agent": "summarization"}})
    result = evaluate_routing(graph, [CASE_A, CASE_B])
    assert result["accuracy"] == 1.0
    assert all(c["correct"] for c in result["per_case"])


def test_evaluate_routing_partial_correct():
    graph = FakeGraph({"q1": {"agent": "filing_qa"}, "q2": {"agent": "risk_flag"}})
    result = evaluate_routing(graph, [CASE_A, CASE_B])
    assert result["accuracy"] == 0.5
    assert result["per_case"][1]["expected_agent"] == "summarization"
    assert result["per_case"][1]["actual_agent"] == "risk_flag"
    assert result["per_case"][1]["correct"] is False


def test_evaluate_routing_empty_cases():
    result = evaluate_routing(FakeGraph({}), [])
    assert result["accuracy"] == 0.0
    assert result["per_case"] == []
