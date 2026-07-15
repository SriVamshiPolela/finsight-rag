"""Central config for Phase 1 ingestion: which companies, sectors, and filings to pull."""

from pathlib import Path

# ticker -> sector, chosen for comparison-query variety across 3 sectors
COMPANIES: dict[str, str] = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "JPM": "Financials",
    "GS": "Financials",
    "WMT": "Consumer Retail",
    "COST": "Consumer Retail",
    "TGT": "Consumer Retail",
}

FORM_TYPE = "10-K"
FILINGS_PER_COMPANY = 2  # most recent N, enables period-over-period comparison

# SEC requires a descriptive User-Agent with contact info on every request
SEC_USER_AGENT = "FinSight-RAG research project (vamshipolela@gmail.com)"
SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"

# SEC rate limit is 10 req/sec; stay well under it
REQUEST_DELAY_SECONDS = 0.25

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Standard 10-K Item headers we try to segment on for section-tagged metadata
TEN_K_SECTIONS = {
    "item1": "Business",
    "item1a": "Risk Factors",
    "item7": "Management's Discussion and Analysis",
    "item7a": "Quantitative and Qualitative Disclosures About Market Risk",
    "item8": "Financial Statements and Supplementary Data",
}
