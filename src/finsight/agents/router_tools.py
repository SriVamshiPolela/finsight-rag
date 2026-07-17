"""Router tool schemas — narrow, explicit input contracts for each specialist
agent (Anthropic tool-use format; OpenAIProvider translates these). The
router LLM call *proposes* one of these; the backend graph *disposes* by
calling the matching, deterministic agent function with exactly these args."""

ROUTER_SYSTEM_PROMPT = (
    "You are a routing classifier for a financial-filings question-answering system. "
    "Given a user question, choose exactly one tool that best handles it and fill in "
    "its arguments from the question. Tickers must be real stock ticker symbols "
    "(e.g. AAPL, MSFT). Do not answer the question yourself."
)

ROUTER_TOOLS = [
    {
        "name": "filing_qa",
        "description": (
            "Answer a direct factual question about a single specific company's filing "
            "(e.g. 'What was Apple's R&D spend in FY2024?')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker, e.g. AAPL"},
                "question": {"type": "string", "description": "The user's question, verbatim"},
            },
            "required": ["ticker", "question"],
        },
    },
    {
        "name": "comparison",
        "description": (
            "Compare a metric or disclosure across two or more companies, or across "
            "multiple periods of the same company."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "description": "Two or more tickers to compare",
                },
                "topic": {"type": "string", "description": "The metric or disclosure to compare"},
            },
            "required": ["tickers", "topic"],
        },
    },
    {
        "name": "risk_flag",
        "description": (
            "Surface and summarize risk-factor language, going-concern notes, or "
            "unusual disclosures for a single company."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "Stock ticker, e.g. AAPL"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "summarization",
        "description": "Produce an executive summary of one section of a company's filing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker, e.g. AAPL"},
                "section_key": {
                    "type": "string",
                    "enum": ["item1", "item1a", "item7", "item7a", "item8"],
                    "description": (
                        "item1=Business, item1a=Risk Factors, item7=MD&A, "
                        "item7a=Market Risk, item8=Financial Statements"
                    ),
                },
            },
            "required": ["ticker", "section_key"],
        },
    },
]

ROUTER_AGENT_NAMES = [tool["name"] for tool in ROUTER_TOOLS]
