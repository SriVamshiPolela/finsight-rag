"""Direct chunk-corpus access for cases where similarity search isn't the
right tool — e.g. summarization wants the *entire* section, not just the
chunks most similar to some query."""

from __future__ import annotations

from finsight.embeddings.pipeline import load_chunks


def get_section_chunks(
    ticker: str,
    section_key: str,
    filing_date: str | None = None,
    chunks: list[dict] | None = None,
) -> list[dict]:
    """Returns all chunks for a company's filing section, in document order.
    Defaults to the most recent filing_date available if none is given."""
    chunks = chunks if chunks is not None else load_chunks()
    matches = [c for c in chunks if c["ticker"] == ticker and c["section_key"] == section_key]

    if filing_date:
        matches = [c for c in matches if c["filing_date"] == filing_date]
    elif matches:
        latest = max(c["filing_date"] for c in matches)
        matches = [c for c in matches if c["filing_date"] == latest]

    return sorted(matches, key=lambda c: c["chunk_index"])
