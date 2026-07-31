# FinSight RAG — 75-Second Demo Script

Meant to be read aloud in an interview, roughly as-is. ~195 words, ~75-90
seconds at a natural pace. Each paragraph maps to a file/metric you can
pull up live if asked to go deeper — see the cheat sheet below.

---

> FinSight RAG answers questions about real SEC 10-K filings — I built it
> end-to-end: ingestion, embeddings, multi-agent orchestration, evaluation,
> and deployment.
>
> It pulls real 10-Ks for eight companies from SEC EDGAR's free API — no
> synthetic data. I benchmarked two embedding models before picking one for
> production, logged to MLflow rather than eyeballed.
>
> The core is a LangGraph router: a Claude tool-call classifies each
> question and dispatches to one of four specialist agents — Filing Q&A,
> Comparison, Risk-Flag, Summarization — each with a narrow, explicit
> schema. Same "LLM proposes, backend disposes" pattern I use elsewhere.
>
> For evaluation, I built a 24-case labeled set scored on four
> RAGAS-methodology metrics, logged to MLflow. Real number: 79% routing
> accuracy — and I actually investigated the misses. They're all cases
> where my own eval questions were ambiguously worded, not router bugs.
> That's documented directly in the README, not hidden.
>
> I had it live on AWS Lambda, verified end-to-end with real Claude calls,
> then intentionally tore it down — an unauthenticated public endpoint with
> a billed API key is a real cost risk, not hypothetical. The live showcase
> now lives on Hugging Face Spaces with real captured transcripts.
>
> Ninety-two tests, GitHub Actions running lint and tests on every push.

---

## Cheat sheet — if they ask you to go deeper

| Claim | Where to look |
|---|---|
| Real filings, no synthetic data | [manifest.json](data/processed/manifest.json) after running `scripts/ingest.py`, or hit `/filings` |
| Embedding benchmark, logged not eyeballed | [README § Phase 2](README.md#phase-2--embeddings--vector-store-done), `mlflow ui --backend-store-uri sqlite:///mlflow.db` |
| LangGraph router + 4 agents | [src/finsight/agents/graph.py](src/finsight/agents/graph.py), [router_tools.py](src/finsight/agents/router_tools.py) |
| "LLM proposes, backend disposes" | Same principle stated in [README § Phase 3](README.md#phase-3--multi-agent-orchestration-done-live) |
| 24-case eval set | [src/finsight/eval/agent_cases.py](src/finsight/eval/agent_cases.py) |
| RAGAS metrics, real numbers | [README § Phase 4 results table](README.md#results--live-run-24-cases-logged-to-mlflow-finsight-agent-evaluation) |
| Routing-miss investigation | [README § Phase 4](README.md#phase-4--evaluation-framework-done-live), "On the 5 routing misses" |
| AWS Lambda built + torn down | [README § Phase 5](README.md#phase-5--api--deployment-built-verified-live-on-aws-then-retired), 4 real bugs hit & fixed |
| Live showcase | https://huggingface.co/spaces/srivamshipolela/finsight-rag |
| 92 tests, CI | [.github/workflows/ci.yml](.github/workflows/ci.yml), [Actions tab](https://github.com/SriVamshiPolela/finsight-rag/actions) |
