from finsight.agents.types import AgentResponse, chunks_to_citations, format_context
from finsight.llm.base import LLMProvider
from finsight.retrieval.retriever import Retriever

SYSTEM_PROMPT = (
    "You are a financial analyst assistant answering questions about SEC filings. "
    "Answer only using the provided filing excerpts. If the excerpts don't contain "
    "the answer, say so explicitly rather than guessing or using outside knowledge."
)


def answer_filing_qa(llm: LLMProvider, retriever: Retriever, ticker: str, question: str) -> AgentResponse:
    results = retriever.search(question, ticker=ticker)
    chunks = [r.metadata for r in results]
    user_message = f"Filing excerpts:\n{format_context(chunks)}\n\nQuestion: {question}"

    response = llm.complete(system=SYSTEM_PROMPT, messages=[{"role": "user", "content": user_message}])
    return AgentResponse(agent="filing_qa", answer=response.text, citations=chunks_to_citations(chunks))
