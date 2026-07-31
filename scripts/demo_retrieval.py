"""Live, runnable proof of the Phase 3 data layer that doesn't need an LLM
API key: exercises the Retriever (similarity search + ticker/section
filtering) and the corpus full-section lookup used by each specialist
agent's retrieval step, against the real indexed corpus.

Usage: python scripts/demo_retrieval.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finsight.retrieval.corpus import get_section_chunks  # noqa: E402
from finsight.retrieval.retriever import Retriever  # noqa: E402

SEARCH_CASES = [
    # (label, query, ticker, section_key) -- mirrors what filing_qa/comparison/risk_flag pass in
    ("filing_qa", "What was Apple's revenue growth driven by?", "AAPL", None),
    ("comparison (AAPL leg)", "supply chain risk", "AAPL", "item1a"),
    ("comparison (MSFT leg)", "supply chain risk", "MSFT", "item1a"),
    ("risk_flag", "risk factors, red flags, going concern, litigation, regulatory risk", "JPM", "item1a"),
]


def main() -> None:
    print("Loading e5-small FAISS index...")
    retriever = Retriever.load()

    for label, query, ticker, section_key in SEARCH_CASES:
        print(f"\n=== {label}: '{query}' (ticker={ticker}, section={section_key}) ===")
        results = retriever.search(query, top_k=3, ticker=ticker, section_key=section_key)
        if not results:
            print("  (no results)")
        for r in results:
            m = r.metadata
            preview = m["text"][:160].replace("\n", " ")
            print(f"  score={r.score:.3f} {m['ticker']} {m['section_title']} ({m['filing_date']}): {preview}...")

    print("\n=== summarization full-section lookup: COST item1 (Business) ===")
    chunks = get_section_chunks("COST", "item1")
    total_chars = sum(len(c["text"]) for c in chunks)
    print(f"  {len(chunks)} chunks, {total_chars} chars, filing_date={chunks[0]['filing_date'] if chunks else 'n/a'}")
    if chunks:
        print(f"  first chunk preview: {chunks[0]['text'][:160].replace(chr(10), ' ')}...")


if __name__ == "__main__":
    main()
