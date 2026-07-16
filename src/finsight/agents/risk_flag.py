from finsight.agents.types import AgentResponse, chunks_to_citations, format_context
from finsight.llm.base import LLMProvider
from finsight.retrieval.retriever import Retriever

SYSTEM_PROMPT = (
    "You are a financial analyst assistant surfacing risk factors and red flags from "
    "SEC filings. Summarize the key risks in the provided excerpts, highlighting any "
    "going-concern language, litigation, regulatory, or unusual disclosures. Answer "
    "only using the provided excerpts."
)

RISK_SECTION_KEY = "item1a"


def answer_risk_flag(llm: LLMProvider, retriever: Retriever, ticker: str) -> AgentResponse:
    results = retriever.search(
        "risk factors, red flags, going concern, litigation, regulatory risk",
        ticker=ticker,
        section_key=RISK_SECTION_KEY,
    )
    chunks = [r.metadata for r in results]
    user_message = f"Risk Factors excerpts for {ticker}:\n{format_context(chunks)}"

    response = llm.complete(system=SYSTEM_PROMPT, messages=[{"role": "user", "content": user_message}])
    return AgentResponse(agent="risk_flag", answer=response.text, citations=chunks_to_citations(chunks))
