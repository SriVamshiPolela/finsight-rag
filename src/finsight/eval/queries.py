"""Hand-labeled query set for the Phase 2 embedding benchmark.

Each query is labeled with the (ticker, section) it should retrieve from,
plus keywords the retrieved chunk text should plausibly contain. This is a
proxy for exact chunk-level relevance judgments — reasonable at this scale
since 10-K structure is standardized enough that a query about "risk
factors" should surface `item1a` chunks for that company specifically.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalQuery:
    query: str
    ticker: str
    section_key: str
    keywords: tuple[str, ...]


EVAL_QUERIES: list[EvalQuery] = [
    EvalQuery("What are Apple's main supply chain risk factors?", "AAPL", "item1a", ("supply chain",)),
    EvalQuery("What litigation risks does Apple disclose?", "AAPL", "item1a", ("litigat",)),
    EvalQuery("What is Apple's core business and product lineup?", "AAPL", "item1", ("iphone", "product")),
    EvalQuery("What did Microsoft's management discuss about revenue growth?", "MSFT", "item7", ("revenue",)),
    EvalQuery("What is Microsoft's business segment structure?", "MSFT", "item1", ("segment",)),
    EvalQuery("What risks does Microsoft disclose about cybersecurity?", "MSFT", "item1a", ("cybersecurity", "security")),
    EvalQuery("What did NVIDIA discuss about data center revenue growth?", "NVDA", "item7", ("data center", "revenue")),
    EvalQuery("What are NVIDIA's risks related to competition?", "NVDA", "item1a", ("competit",)),
    EvalQuery("What is JPMorgan's overview of its business segments?", "JPM", "item1", ("segment", "bank")),
    EvalQuery("What regulatory risks does JPMorgan disclose?", "JPM", "item1a", ("regulat",)),
    EvalQuery("What is Goldman Sachs' business overview?", "GS", "item1", ("goldman", "business")),
    EvalQuery("What quantitative market risk disclosures does Goldman Sachs make?", "GS", "item7a", ("market risk", "interest rate")),
    EvalQuery("What does Walmart say about competition in its risk factors?", "WMT", "item1a", ("competit",)),
    EvalQuery("What is Costco's core business and membership model?", "COST", "item1", ("membership", "warehouse")),
    EvalQuery("What did Target's management discuss about consumer demand?", "TGT", "item7", ("sales", "demand", "consumer")),
    EvalQuery("What risks does Target disclose about inventory management?", "TGT", "item1a", ("inventory",)),
]
