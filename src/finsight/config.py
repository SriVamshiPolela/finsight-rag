"""Central config: Phase 1 ingestion targets + Phase 2 embeddings/vector-store settings."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

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
INDEXES_DIR = DATA_DIR / "indexes"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"

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

# --- Phase 2: embeddings & vector store ---

# Registry of embedding models to benchmark. Both are small enough to run on
# CPU in a few minutes; e5 requires "query: "/"passage: " prefixes per its
# model card, bge-v1.5 does not require instruction prefixes.
EMBEDDING_MODEL_SPECS = {
    "bge-small": {
        "hf_name": "BAAI/bge-small-en-v1.5",
        "dimension": 384,
        "query_prefix": "",
        "passage_prefix": "",
    },
    "e5-small": {
        "hf_name": "intfloat/e5-small-v2",
        "dimension": 384,
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
}

# "faiss" (dev, default) or "pinecone" (prod path, swappable via env var)
VECTOR_STORE = os.environ.get("VECTOR_STORE", "faiss")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "finsight-rag")
PINECONE_CLOUD = os.environ.get("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.environ.get("PINECONE_REGION", "us-east-1")

EVAL_TOP_K = 5
# sqlite, not the plain filesystem store — MLflow 3.x put the filesystem
# backend into maintenance mode with no further feature updates
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = "finsight-embedding-benchmark"

# --- Phase 3: LLM providers & multi-agent routing ---

# Picked in the Phase 2 benchmark (higher hit_rate@5 and MRR than bge-small)
PRODUCTION_EMBEDDING_MODEL = "e5-small"
# FAISSVectorStore has no native metadata filter, so the Retriever over-fetches
# this many nearest neighbors and filters by ticker/section client-side. Fine
# at ~3.6k chunks; would need real filtered search (or per-ticker indexes) at scale.
RETRIEVAL_CANDIDATE_POOL = 500
RETRIEVAL_TOP_K = 5

# "anthropic" (primary) or "openai" (vendor-abstraction swap demo)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

ROUTING_LOG_PATH = PROCESSED_DIR / "routing_log.jsonl"

# --- Phase 4: agent evaluation (routing accuracy + RAGAS-methodology metrics) ---

MLFLOW_AGENT_EVAL_EXPERIMENT_NAME = "finsight-agent-evaluation"
AGENT_EVAL_RESULTS_PATH = PROCESSED_DIR / "agent_eval_results.json"
RETRIEVAL_CHECK_RESULTS_PATH = PROCESSED_DIR / "agent_retrieval_check.json"
