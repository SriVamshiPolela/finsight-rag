from collections import Counter

from finsight.agents.router_tools import ROUTER_AGENT_NAMES
from finsight.eval.agent_cases import AGENT_EVAL_CASES

EXPECTED_ARGS_BY_AGENT = {
    "filing_qa": {"ticker", "question"},
    "comparison": {"tickers", "topic"},
    "risk_flag": {"ticker"},
    "summarization": {"ticker", "section_key"},
}


def test_case_count_within_spec_range():
    assert 20 <= len(AGENT_EVAL_CASES) <= 30


def test_ids_are_unique():
    ids = [c.id for c in AGENT_EVAL_CASES]
    assert len(ids) == len(set(ids))


def test_all_agents_are_valid_router_targets():
    for case in AGENT_EVAL_CASES:
        assert case.agent in ROUTER_AGENT_NAMES


def test_each_agent_type_has_multiple_cases():
    counts = Counter(c.agent for c in AGENT_EVAL_CASES)
    for agent in ROUTER_AGENT_NAMES:
        assert counts[agent] >= 3, f"too few eval cases for {agent}"


def test_args_match_expected_schema_per_agent():
    for case in AGENT_EVAL_CASES:
        assert set(case.args.keys()) == EXPECTED_ARGS_BY_AGENT[case.agent], case.id


def test_every_case_has_expected_answer_and_keywords():
    for case in AGENT_EVAL_CASES:
        assert case.expected_answer.strip(), case.id
        assert len(case.keywords) > 0, case.id
