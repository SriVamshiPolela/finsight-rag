"""Phase 4 labeled eval set: 24 cases (6 per agent type), covering all 8
ingested companies. Each case's expected_answer and keywords are grounded in
real excerpts pulled live from the indexed corpus via scripts/demo_retrieval.py
-style queries (see PR description) - not guessed at.

Used for two things once an LLM key exists:
- Routing accuracy: does the router pick `agent` with the right `args`?
- RAGAS generation quality: does the agent's actual answer line up with
  `expected_answer` (ground_truth) and stay faithful to retrieved context?
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentEvalCase:
    id: str
    agent: str
    query: str
    args: dict
    expected_answer: str
    keywords: tuple[str, ...]


AGENT_EVAL_CASES: list[AgentEvalCase] = [
    # --- filing_qa ---
    AgentEvalCase(
        id="fqa-aapl-supply-chain",
        agent="filing_qa",
        query="What are Apple's main supply chain risks?",
        args={"ticker": "AAPL", "question": "What are Apple's main supply chain risks?"},
        expected_answer=(
            "Apple faces risks of supply shortages and price increases, partly because global or "
            "regional economic conditions can significantly affect its suppliers, contract "
            "manufacturers, logistics providers, and other channel partners."
        ),
        keywords=("supply", "shortage"),
    ),
    AgentEvalCase(
        id="fqa-msft-cybersecurity",
        agent="filing_qa",
        query="What did Microsoft say about cybersecurity risks?",
        args={"ticker": "MSFT", "question": "What did Microsoft say about cybersecurity risks?"},
        expected_answer=(
            "Microsoft discloses risk from AI systems being used in unintended or fraudulent ways, "
            "including unauthorized access through its cloud-based services, and notes its "
            "data-handling practices remain under scrutiny."
        ),
        keywords=("cyber", "unauthorized", "data"),
    ),
    AgentEvalCase(
        id="fqa-nvda-datacenter",
        agent="filing_qa",
        query="What did NVIDIA say about data center revenue growth?",
        args={"ticker": "NVDA", "question": "What did NVIDIA say about data center revenue growth?"},
        expected_answer=(
            "NVIDIA's Data Center platform accelerates compute-intensive workloads like AI, data "
            "processing, and scientific computing; in fiscal 2025 it launched the Blackwell "
            "architecture, a full data-center-scale infrastructure of GPUs, CPUs, DPUs, and networking."
        ),
        keywords=("data center", "blackwell"),
    ),
    AgentEvalCase(
        id="fqa-jpm-regulatory",
        agent="filing_qa",
        query="What regulatory risks does JPMorgan disclose?",
        args={"ticker": "JPM", "question": "What regulatory risks does JPMorgan disclose?"},
        expected_answer=(
            "JPMorgan discloses risk from extensive supervision and regulation, including that some "
            "internal risk models require regulatory review and approval under Basel III before use, "
            "and that regulatory implementation differs across jurisdictions."
        ),
        keywords=("regulat", "basel"),
    ),
    AgentEvalCase(
        id="fqa-gs-overview",
        agent="filing_qa",
        query="What is Goldman Sachs' business overview?",
        args={"ticker": "GS", "question": "What is Goldman Sachs' business overview?"},
        expected_answer=(
            "Goldman Sachs describes itself as a leading global financial institution delivering a "
            "broad range of financial services to a diversified client base including corporations, "
            "financial institutions, governments, and individuals."
        ),
        keywords=("global financial institution", "clients"),
    ),
    AgentEvalCase(
        id="fqa-wmt-competition",
        agent="filing_qa",
        query="What does Walmart say about competition?",
        args={"ticker": "WMT", "question": "What does Walmart say about competition?"},
        expected_answer=(
            "Walmart says it competes with brick-and-mortar, eCommerce, and omnichannel retailers "
            "across discount, department, grocery, drug, dollar, variety, specialty, and "
            "membership-only warehouse club formats."
        ),
        keywords=("compete", "retailers"),
    ),
    # --- comparison ---
    AgentEvalCase(
        id="cmp-aapl-msft-cybersecurity",
        agent="comparison",
        query="Compare Apple and Microsoft on cybersecurity and data security risks.",
        args={"tickers": ["AAPL", "MSFT"], "topic": "cybersecurity and data security risks"},
        expected_answer=(
            "Apple frames data/security risk partly through new online-safety and minor-protection "
            "laws increasing regulatory exposure, while Microsoft frames it through operational risk "
            "of outages, data loss, and disruption from inadequate operations infrastructure."
        ),
        keywords=("data security", "outages"),
    ),
    AgentEvalCase(
        id="cmp-retail-competition",
        agent="comparison",
        query="How do Walmart, Costco, and Target describe competition in retail?",
        args={"tickers": ["WMT", "COST", "TGT"], "topic": "competition in retail"},
        expected_answer=(
            "Walmart competes across many formats including warehouse clubs; Costco competes "
            "globally on price, merchandise quality/selection, location, and service; Target "
            "competes with omnichannel, department, and off-price general merchandise retailers."
        ),
        keywords=("walmart", "costco", "target"),
    ),
    AgentEvalCase(
        id="cmp-jpm-gs-regulatory",
        agent="comparison",
        query="Compare the regulatory risk disclosures of JPMorgan and Goldman Sachs.",
        args={"tickers": ["JPM", "GS"], "topic": "regulatory risk"},
        expected_answer=(
            "Both JPMorgan and Goldman Sachs cite extensive supervision and regulation as a "
            "material risk; Goldman Sachs additionally flags risk from local regulators finding it "
            "non-compliant with local laws in a particular market."
        ),
        keywords=("regulat",),
    ),
    AgentEvalCase(
        id="cmp-aapl-nvda-supply-chain",
        agent="comparison",
        query="Compare Apple and NVIDIA's supply chain risk disclosures.",
        args={"tickers": ["AAPL", "NVDA"], "topic": "supply chain risk"},
        expected_answer=(
            "Apple discloses that changes to its supply chain involve regulatory and operational "
            "risk, while NVIDIA specifically notes its supply chain is concentrated in Asia and "
            "exposed to export controls."
        ),
        keywords=("supply chain",),
    ),
    AgentEvalCase(
        id="cmp-nvda-msft-ai-strategy",
        agent="comparison",
        query="Compare how NVIDIA and Microsoft describe their AI strategy and investment.",
        args={"tickers": ["NVDA", "MSFT"], "topic": "artificial intelligence strategy and investment"},
        expected_answer=(
            "NVIDIA emphasizes its AI technology leadership and large developer ecosystem (over 7.5 "
            "million CUDA developers), while Microsoft emphasizes continued significant investment "
            "in research, development, and new AI platform services."
        ),
        keywords=("ai", "nvidia", "microsoft"),
    ),
    AgentEvalCase(
        id="cmp-wmt-tgt-demand",
        agent="comparison",
        query="Compare consumer demand trends discussed by Walmart and Target.",
        args={"tickers": ["WMT", "TGT"], "topic": "consumer demand trends"},
        expected_answer=(
            "Walmart notes customer demand and disposable income can be affected by health or "
            "economic crises, while Target notes it depends on higher-margin merchandise sales and "
            "flags flat or declining sales as a risk to earnings growth."
        ),
        keywords=("demand", "sales"),
    ),
    # --- risk_flag ---
    AgentEvalCase(
        id="risk-aapl",
        agent="risk_flag",
        query="What are Apple's key risk factors and red flags?",
        args={"ticker": "AAPL"},
        expected_answer=(
            "Apple's risk factors include new and changing laws/regulations as its products enter "
            "specialized applications like health and financial services, among other business, "
            "reputational, and financial-condition risks."
        ),
        keywords=("law", "regulat"),
    ),
    AgentEvalCase(
        id="risk-msft",
        agent="risk_flag",
        query="What are Microsoft's key risk factors and red flags?",
        args={"ticker": "MSFT"},
        expected_answer=(
            "Microsoft flags risk from acquisition integration success, data-handling practices "
            "under scrutiny, and potential mismanagement perceptions from regulatory activity or "
            "negative public reaction."
        ),
        keywords=("acquisit", "data"),
    ),
    AgentEvalCase(
        id="risk-nvda",
        agent="risk_flag",
        query="What are NVIDIA's key risk factors and red flags?",
        args={"ticker": "NVDA"},
        expected_answer=(
            "NVIDIA flags risk from modification or interruption of its business processes and "
            "information systems, and notes its operating results have fluctuated and may continue "
            "to fluctuate relative to analyst and investor expectations."
        ),
        keywords=("business processes", "fluctuat"),
    ),
    AgentEvalCase(
        id="risk-jpm",
        agent="risk_flag",
        query="What are JPMorgan's key risk factors and red flags?",
        args={"ticker": "JPM"},
        expected_answer=(
            "JPMorgan flags risk from local economic, political, regulatory, and social factors in "
            "countries where it operates, people risk from attracting/retaining qualified employees, "
            "and extensive supervision and regulation."
        ),
        keywords=("regulat", "employ"),
    ),
    AgentEvalCase(
        id="risk-gs",
        agent="risk_flag",
        query="What are Goldman Sachs' key risk factors and red flags?",
        args={"ticker": "GS"},
        expected_answer=(
            "Goldman Sachs flags risk from governmental and regulatory scrutiny or negative "
            "publicity, potential civil or criminal liability, and adverse conditions in global "
            "financial markets."
        ),
        keywords=("regulat", "liability"),
    ),
    AgentEvalCase(
        id="risk-tgt",
        agent="risk_flag",
        query="What are Target's key risk factors and red flags?",
        args={"ticker": "TGT"},
        expected_answer=(
            "Target flags risk of adverse perceptions of its business, consumer boycotts, "
            "litigation, investigations, and regulatory proceedings, any of which could hurt "
            "reputation, results of operations, and financial condition."
        ),
        keywords=("boycott", "litigat"),
    ),
    # --- summarization ---
    AgentEvalCase(
        id="sum-aapl-business",
        agent="summarization",
        query="Summarize Apple's Business section (Item 1).",
        args={"ticker": "AAPL", "section_key": "item1"},
        expected_answer=(
            "Apple designs, manufactures, and markets smartphones, personal computers, tablets, "
            "wearables, and accessories, and sells related services, including iPhone, Mac, iPad, "
            "and Wearables/Home/Accessories product lines."
        ),
        keywords=("iphone", "product"),
    ),
    AgentEvalCase(
        id="sum-msft-mda",
        agent="summarization",
        query="Summarize Microsoft's Management's Discussion and Analysis (Item 7).",
        args={"ticker": "MSFT", "section_key": "item7"},
        expected_answer=(
            "Microsoft's MD&A discusses accounting judgment involved in revenue recognition, "
            "including how it estimates standalone selling price for performance obligations that "
            "aren't sold separately, such as certain on-premises licenses."
        ),
        keywords=("revenue", "selling price"),
    ),
    AgentEvalCase(
        id="sum-nvda-risk",
        agent="summarization",
        query="Summarize NVIDIA's Risk Factors section (Item 1A).",
        args={"ticker": "NVDA", "section_key": "item1a"},
        expected_answer=(
            "NVIDIA's Risk Factors section covers risks that could harm its business, financial "
            "condition, or stock price, including dependency on third-party suppliers for "
            "manufacturing, assembly, and testing, which reduces its control over quantity and quality."
        ),
        keywords=("supplier", "risk"),
    ),
    AgentEvalCase(
        id="sum-cost-business",
        agent="summarization",
        query="Summarize Costco's Business section (Item 1).",
        args={"ticker": "COST", "section_key": "item1"},
        expected_answer=(
            "Costco Wholesale, founded in 1983 in Seattle, operates membership warehouses in the "
            "U.S., Puerto Rico, Canada, Mexico, Japan, the UK, Korea, Australia, and other markets, "
            "reporting on a 52/53-week fiscal year."
        ),
        keywords=("membership", "warehouse"),
    ),
    AgentEvalCase(
        id="sum-gs-risk",
        agent="summarization",
        query="Summarize Goldman Sachs' Risk Factors section (Item 1A).",
        args={"ticker": "GS", "section_key": "item1a"},
        expected_answer=(
            "Goldman Sachs' Risk Factors section covers substantial risks inherent in its "
            "businesses, including adverse conditions in global financial markets and governmental "
            "or regulatory scrutiny and negative publicity."
        ),
        keywords=("market", "regulat"),
    ),
    AgentEvalCase(
        id="sum-wmt-risk",
        agent="summarization",
        query="Summarize Walmart's Risk Factors section (Item 1A).",
        args={"ticker": "WMT", "section_key": "item1a"},
        expected_answer=(
            "Walmart's Risk Factors section covers information-security risk, including misuse or "
            "loss of information and denial-of-service incidents that could disable or degrade the "
            "digital platforms that support its business."
        ),
        keywords=("information", "security"),
    ),
]
