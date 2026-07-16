"""Phase 3 live CLI: routes a natural-language question through the LangGraph
router to one of the 4 specialist agents and prints the grounded answer.

Requires ANTHROPIC_API_KEY (or OPENAI_API_KEY with LLM_PROVIDER=openai) in
.env — this is the one Phase 3 entry point that needs a real LLM key.

Usage: python scripts/run_query.py "What are Apple's main risk factors?"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finsight.agents.pipeline import run_query  # noqa: E402

if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "What are Apple's main risk factors?"
    result = run_query(query)

    print(f"[routed to: {result['agent']}]\n")
    print(result["answer"])
    print("\nCitations:")
    for c in result["citations"]:
        print(f"  - {c['ticker']} | {c['section_title']} | {c['filing_date']} | {c['source_url']}")
