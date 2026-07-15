# FinSight RAG

Multi-agent Retrieval-Augmented Generation system for financial document
intelligence over SEC filings (10-K/10-Q/8-K).

**Status:** Phase 2 of 6 complete (embeddings & vector store). See
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

## Phase 2 — Embeddings & vector store (done)

Benchmarks two open-source embedding models against a 16-query hand-labeled
eval set, indexes chunks into FAISS, and logs everything to MLflow so the
model choice is backed by numbers, not a guess.

**Models compared** (both small enough to run on CPU in a few minutes):

| Model | HF name | Dim | Notes |
|---|---|---|---|
| BGE-small | `BAAI/bge-small-en-v1.5` | 384 | No instruction prefix needed (v1.5) |
| E5-small | `intfloat/e5-small-v2` | 384 | Requires `"query: "`/`"passage: "` prefixes per its model card — implemented in [sentence_transformer.py](src/finsight/embeddings/sentence_transformer.py) |

An OpenAI model (`text-embedding-3-small`) was in the original plan but
dropped to keep this phase key-free — both candidates here are fully
open-source and run locally.

**Eval methodology:** 16 hand-labeled queries ([queries.py](src/finsight/eval/queries.py)),
each labeled with the expected `(ticker, section)` and keywords the answer
should contain. For each query, both models retrieve top-5 chunks from their
own FAISS index over the full ~3,600-chunk corpus; we measure whether the
correct ticker+section shows up (`hit_rate@5`), how high it ranks (`MRR`),
and whether the retrieved text contains the expected keywords regardless of
exact section match (`keyword_hit_rate@5`).

**Results** (MLflow experiment `finsight-embedding-benchmark`, tracked in
`mlflow.db`):

| Model | hit_rate@5 | MRR | keyword_hit_rate@5 |
|---|---|---|---|
| bge-small | 0.5625 | 0.5313 | **1.0000** |
| e5-small | **0.6250** | **0.5365** | 0.9375 |

**Pick for production: e5-small.** It retrieves the exact expected
ticker+section more often and ranks it slightly higher on average, which
matters more than keyword recall for the Filing Q&A and Comparison agents in
Phase 3, which depend on the retrieved chunk being from the *right* filing
section. bge-small's perfect keyword hit rate suggests it's better at
surfacing topically-relevant text even when it picks the "wrong" section —
worth revisiting if Phase 4's RAGAS faithfulness scores disagree with this
call. With only 16 queries the margin (10/16 vs 9/16 correct) is a single
query wide, so this is a soft pick, not a landslide — noted honestly rather
than overstated.

**Vector store:** FAISS (`IndexFlatIP` over L2-normalized vectors = cosine
similarity) is the only path actually exercised — exact search is fine at
~3,600 chunks. Pinecone is fully implemented behind the same `VectorStore`
interface ([pinecone_store.py](src/finsight/vectorstore/pinecone_store.py))
and unit-tested against a mocked client, but not run against a live index —
no Pinecone API key is provisioned yet. Swap via `VECTOR_STORE=pinecone` in
`.env` once one is.

### Run it

```bash
python scripts/run_embedding_benchmark.py
```

Output:
- `data/indexes/<model_key>/index.faiss` + `meta.jsonl` — persisted FAISS index per model
- `data/processed/embedding_benchmark.json` — comparison summary
- MLflow runs in `mlflow.db` (`mlflow ui --backend-store-uri sqlite:///mlflow.db` to view)

### Tests

22 new tests (29 total) covering: FAISS upsert/query/save/load round-trips,
retrieval metric correctness (hit@k, MRR, keyword matching), embedding
prefix logic (mocked model, no weight download in CI), the Pinecone wrapper
(mocked client — batching, index-creation-if-missing, response mapping), and
the FAISS/Pinecone factory dispatch.

## Coming next

- **Phase 3** — LangGraph router + 4 specialist agents
- **Phase 4** — RAGAS evaluation framework
- **Phase 5** — FastAPI + Docker + AWS deployment
- **Phase 6** — full docs, CI/CD, demo script
