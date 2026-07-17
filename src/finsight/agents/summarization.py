from finsight.agents.types import AgentResponse, chunks_to_citations, format_context
from finsight.llm.base import LLMProvider
from finsight.retrieval.corpus import get_section_chunks

SYSTEM_PROMPT = (
    "You are a financial analyst assistant writing an executive summary of a single "
    "SEC filing section. Summarize only using the provided excerpts, in 3-5 concise "
    "sentences suitable for a busy executive."
)


def answer_summarization(llm: LLMProvider, ticker: str, section_key: str) -> AgentResponse:
    # Full-section retrieval, not similarity search: summarization needs
    # complete coverage of the section, not just the chunks most similar to
    # some query — there is no query here, only a (ticker, section) target.
    chunks = get_section_chunks(ticker, section_key)
    user_message = f"Filing section excerpts:\n{format_context(chunks)}\n\nWrite the executive summary."

    response = llm.complete(system=SYSTEM_PROMPT, messages=[{"role": "user", "content": user_message}])
    return AgentResponse(agent="summarization", answer=response.text, citations=chunks_to_citations(chunks))
