from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Citation:
    ticker: str
    section_title: str
    filing_date: str
    source_url: str


@dataclass(frozen=True)
class AgentResponse:
    agent: str
    answer: str
    citations: list[Citation]
    # Raw retrieved chunk text, distinct from `citations` (deduped source
    # metadata) — needed by Phase 4's faithfulness/context-precision/recall
    # evaluation, which compares the answer against what was actually
    # retrieved, not just where it came from.
    contexts: list[str]


def format_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{c['ticker']} | {c['section_title']} | {c['filing_date']}]\n{c['text']}" for c in chunks
    )


def chunks_to_citations(chunks: list[dict]) -> list[Citation]:
    seen: set[tuple[str, str, str]] = set()
    citations = []
    for c in chunks:
        key = (c["ticker"], c["section_key"], c["filing_date"])
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            Citation(
                ticker=c["ticker"],
                section_title=c["section_title"],
                filing_date=c["filing_date"],
                source_url=c["source_url"],
            )
        )
    return citations
