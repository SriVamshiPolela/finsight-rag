# FinSight RAG

Multi-agent Retrieval-Augmented Generation system for financial document
intelligence over SEC filings (10-K/10-Q/8-K).

**Status:** Phase 1 of 6 complete (data ingestion & chunking). See
[finsight-rag-claude-code-prompt_1.md](finsight-rag-claude-code-prompt_1.md)
for the full build spec and phase plan.

## Phase 1 — Data ingestion & chunking (done)

Pulls real 10-K filings for 8 companies across 3 sectors straight from SEC
EDGAR's free public APIs (no API key required), parses them into standard
Item sections, and chunks them with metadata for retrieval.

| Sector | Tickers |
|---|---|
| Technology | AAPL, MSFT, NVDA |
| Financials | JPM, GS |
| Consumer Retail | WMT, COST, TGT |

Each company's 2 most recent 10-Ks are pulled (enables period-over-period
comparison queries), giving **14 filings and ~3,600 chunks** in the current
corpus.

**Storage tradeoff:** raw HTML and chunked JSONL are written to local disk
(`data/raw/`, `data/processed/`). That's fine at this scale; at production
scale this would move to S3 for the raw/chunk store with the manifest in a
real database, so ingestion workers don't need a shared filesystem.

### Run it

```bash
python -m venv .venv
./.venv/Scripts/activate   # or source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt   # or just the ingestion subset for Phase 1
python scripts/ingest.py
```

Output:
- `data/raw/<TICKER>_<accession>.html` — raw filing HTML
- `data/processed/chunks.jsonl` — one JSON object per chunk, with `ticker`,
  `sector`, `filing_type`, `filing_date`, `accession_number`, `section_key`,
  `section_title`, `source_url`, `chunk_id`, `text`
- `data/processed/manifest.json` — one entry per filing ingested

### Tests

```bash
python -m pytest tests/ -v
```

7 tests covering HTML→text conversion, entity decoding, section splitting
(including the no-headings-found fallback), chunk metadata correctness, and
EDGAR API response filtering (mocked, no live network dependency in CI).

## Coming next

- **Phase 2** — embedding benchmark (BGE vs E5) + FAISS index, MLflow-logged
- **Phase 3** — LangGraph router + 4 specialist agents
- **Phase 4** — RAGAS evaluation framework
- **Phase 5** — FastAPI + Docker + AWS deployment
- **Phase 6** — full docs, CI/CD, demo script
