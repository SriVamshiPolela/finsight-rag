"""RAGAS-methodology metrics (faithfulness, answer relevancy, context
precision, context recall), computed via an LLM judge through our own
LLMProvider instead of the `ragas` package.

Why not the `ragas` package: it was evaluated first (see requirements
history), but ragas 0.2.x-0.4.x unconditionally import a deprecated
`langchain_community.chat_models.vertexai` shim at module load time. Current
`langchain-community` no longer ships that shim, and pinning an old enough
`langchain-community` to get it back drags `langchain-core` below what
LangGraph 1.x requires, breaking Phase 3. Reimplementing the four metrics
directly against the LLMProvider abstraction we already built avoids the
conflict entirely and reuses the same tool-calling infrastructure as the
router (a "submit_score" tool instead of parsing free-text scores).

TruLens was also considered as an alternative eval framework: it leans
toward live tracing/feedback-function instrumentation inside a running app,
which fits production observability better than the batch, dataset-style
eval this project needed to match the Phase 2 benchmarking pattern.
"""

from __future__ import annotations

from finsight.llm.base import LLMProvider

SCORE_TOOL = {
    "name": "submit_score",
    "description": "Submit a numeric score between 0.0 and 1.0",
    "input_schema": {
        "type": "object",
        "properties": {"score": {"type": "number", "minimum": 0, "maximum": 1}},
        "required": ["score"],
    },
}


def _judge_score(llm: LLMProvider, system: str, user_message: str) -> float:
    response = llm.complete(
        system=system, messages=[{"role": "user", "content": user_message}], tools=[SCORE_TOOL], force_tool_choice=True
    )
    if not response.tool_calls:
        raise ValueError("Judge did not return a score")
    return float(response.tool_calls[0].input["score"])


FAITHFULNESS_SYSTEM = (
    "You are grading whether an answer is faithful to the given context: every claim in the answer "
    "must be supported by the context, with no hallucinated or unsupported claims. Score from 0.0 "
    "(completely unfaithful) to 1.0 (fully faithful)."
)


def faithfulness(llm: LLMProvider, answer: str, contexts: list[str]) -> float:
    context_text = "\n\n".join(contexts) or "(no context retrieved)"
    user_message = f"Context:\n{context_text}\n\nAnswer:\n{answer}"
    return _judge_score(llm, FAITHFULNESS_SYSTEM, user_message)


ANSWER_RELEVANCY_SYSTEM = (
    "You are grading how relevant an answer is to the given question, penalizing answers that are "
    "incomplete, off-topic, or padded with unnecessary information. Score from 0.0 (irrelevant) to "
    "1.0 (fully relevant and on-topic)."
)


def answer_relevancy(llm: LLMProvider, question: str, answer: str) -> float:
    user_message = f"Question:\n{question}\n\nAnswer:\n{answer}"
    return _judge_score(llm, ANSWER_RELEVANCY_SYSTEM, user_message)


CONTEXT_PRECISION_SYSTEM = (
    "You are grading retrieval precision: given a question and a reference answer, what fraction of "
    "the numbered retrieved context chunks are actually relevant to answering the question? Score "
    "from 0.0 (none relevant) to 1.0 (all relevant)."
)


def context_precision(llm: LLMProvider, question: str, contexts: list[str], reference_answer: str) -> float:
    context_text = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts)) or "(no context retrieved)"
    user_message = (
        f"Question:\n{question}\n\nReference answer:\n{reference_answer}\n\nRetrieved contexts:\n{context_text}"
    )
    return _judge_score(llm, CONTEXT_PRECISION_SYSTEM, user_message)


CONTEXT_RECALL_SYSTEM = (
    "You are grading retrieval recall: given a reference answer, what fraction of the information in "
    "it is actually supported by or present in the retrieved context? Score from 0.0 (none of it) to "
    "1.0 (all of it)."
)


def context_recall(llm: LLMProvider, reference_answer: str, contexts: list[str]) -> float:
    context_text = "\n\n".join(contexts) or "(no context retrieved)"
    user_message = f"Reference answer:\n{reference_answer}\n\nRetrieved context:\n{context_text}"
    return _judge_score(llm, CONTEXT_RECALL_SYSTEM, user_message)
