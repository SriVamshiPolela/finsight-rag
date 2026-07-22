from unittest.mock import MagicMock

import pytest

from finsight.eval.ragas_metrics import answer_relevancy, context_precision, context_recall, faithfulness
from finsight.llm.base import LLMResponse, ToolCall


def make_llm(score: float):
    llm = MagicMock()
    llm.complete.return_value = LLMResponse(text="", tool_calls=[ToolCall(name="submit_score", input={"score": score})])
    return llm


def test_faithfulness_returns_judge_score_and_forces_tool_choice():
    llm = make_llm(0.8)
    score = faithfulness(llm, answer="the answer", contexts=["context a", "context b"])
    assert score == 0.8
    kwargs = llm.complete.call_args.kwargs
    assert kwargs["force_tool_choice"] is True
    assert "context a" in kwargs["messages"][0]["content"]


def test_faithfulness_handles_empty_context():
    llm = make_llm(0.0)
    faithfulness(llm, answer="the answer", contexts=[])
    user_message = llm.complete.call_args.kwargs["messages"][0]["content"]
    assert "no context retrieved" in user_message


def test_answer_relevancy_includes_question_and_answer():
    llm = make_llm(0.9)
    score = answer_relevancy(llm, question="What are the risks?", answer="Supply chain risk.")
    assert score == 0.9
    user_message = llm.complete.call_args.kwargs["messages"][0]["content"]
    assert "What are the risks?" in user_message
    assert "Supply chain risk." in user_message


def test_context_precision_numbers_each_context():
    llm = make_llm(0.5)
    context_precision(llm, question="q", contexts=["ctx1", "ctx2"], reference_answer="ref")
    user_message = llm.complete.call_args.kwargs["messages"][0]["content"]
    assert "[1] ctx1" in user_message
    assert "[2] ctx2" in user_message


def test_context_recall_compares_reference_to_context():
    llm = make_llm(1.0)
    score = context_recall(llm, reference_answer="the reference", contexts=["supporting context"])
    assert score == 1.0
    user_message = llm.complete.call_args.kwargs["messages"][0]["content"]
    assert "the reference" in user_message
    assert "supporting context" in user_message


def test_raises_when_judge_returns_no_tool_call():
    llm = MagicMock()
    llm.complete.return_value = LLMResponse(text="I decline to score this.", tool_calls=[])
    with pytest.raises(ValueError):
        faithfulness(llm, answer="a", contexts=["c"])
