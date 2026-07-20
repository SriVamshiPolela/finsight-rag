from finsight.agents.types import AgentResponse, chunks_to_citations, format_context
from finsight.llm.base import LLMProvider
from finsight.retrieval.retriever import Retriever

SYSTEM_PROMPT = (
    "You are a financial analyst assistant comparing SEC filing disclosures across "
    "companies or periods. Answer only using the provided filing excerpts, grouped "
    "by company. Call out specific differences and similarities. If the excerpts "
    "don't cover one of the companies for this topic, say so explicitly."
)


def answer_comparison(llm: LLMProvider, retriever: Retriever, tickers: list[str], topic: str) -> AgentResponse:
    chunks: list[dict] = []
    for ticker in tickers:
        results = retriever.search(topic, ticker=ticker)
        chunks.extend(r.metadata for r in results)

    user_message = f"Filing excerpts:\n{format_context(chunks)}\n\nCompare: {topic}"

    response = llm.complete(system=SYSTEM_PROMPT, messages=[{"role": "user", "content": user_message}])
    return AgentResponse(
        agent="comparison",
        answer=response.text,
        citations=chunks_to_citations(chunks),
        contexts=[c["text"] for c in chunks],
    )
