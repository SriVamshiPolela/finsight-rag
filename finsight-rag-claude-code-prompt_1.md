# FinSight RAG — Build Prompt for Claude Code

Paste this whole document into Claude Code as your first message in a fresh repo/directory. It's written so Claude Code can execute it phase by phase without needing you to re-explain context each time.

---

## 1. What you're building

**FinSight RAG** — a multi-agent Retrieval-Augmented Generation system for financial document intelligence over SEC filings (10-K, 10-Q, 8-K). This is a portfolio flagship project, not a toy demo — it needs to hold up to a recruiter or hiring manager clicking through the GitHub repo and reading the README in under 5 minutes, and it needs to hold up to a technical interviewer asking "walk me through the architecture."

**Why this project, specifically:** it should let a reviewer directly map what they see in the repo to these resume lines:
- RAG pipelines using LangChain, FAISS, and Pinecone with semantic search and embedding layers
- Multi-agent orchestration / agentic AI architecture
- LLM fine-tuning, prompt engineering
- FastAPI deployment on AWS (SageMaker/Lambda/EC2)
- MLflow experiment tracking, reproducible workflows
- Model evaluation discipline (ROC-AUC/Precision/Recall equivalent for RAG: faithfulness, answer relevancy, context precision)
- CI/CD via GitHub Actions

Every phase below should produce something that's actually true when a reviewer reads the README — don't describe capabilities that aren't implemented.

## 2. Tech stack (use exactly this — it's chosen to match my existing resume claims)

- **Language:** Python 3.11+
- **Orchestration/agents:** LangChain + LangGraph for the multi-agent router/graph
- **Retrieval:** LlamaIndex for document parsing/indexing where it's cleaner than raw LangChain
- **Vector stores:** FAISS for local/dev, Pinecone for the "production" deployment path (make this swappable via an env var / interface, not hardcoded)
- **Embeddings:** benchmark at least 2 models (e.g. OpenAI `text-embedding-3-small` and an open-source alternative like BGE or E5) — log the comparison, don't just pick one silently
- **LLM:** Claude (Anthropic API) as primary; keep an abstraction layer so a second provider (OpenAI or a local HF model) can be swapped in — this demonstrates the "vendor abstraction" pattern
- **API layer:** FastAPI
- **Experiment tracking:** MLflow, logging retrieval configs, embedding model choice, and evaluation scores as runs
- **Evaluation:** RAGAS (faithfulness, answer relevancy, context precision/recall) and note TruLens as an alternative you considered
- **Data source:** SEC EDGAR full-text search API (free, no key needed) — pull real 10-K/10-Q filings, don't use synthetic data
- **Infra/deploy:** Docker for local run; deployment target is AWS (Lambda for the lightweight path, SageMaker endpoint or ECS/Fargate for a "real" deployment path) — actually deploy at least the Docker+FastAPI version somewhere reachable (Fly.io/Render/AWS free tier), don't leave it as "deployment instructions only" if you can help it
- **CI/CD:** GitHub Actions — lint, test, and (optionally) build the Docker image on push
- **Testing:** pytest, with real assertions against retrieval quality thresholds, not just "does it run"

## 3. Architecture — multi-agent design

Build a **Router Agent** that classifies the incoming question and dispatches to specialist agents. This mirrors a pattern I already use in a separate voice-assistant project, so keep the shared design principle: *the LLM proposes, the backend disposes* — agents suggest actions/tool calls, deterministic code executes them.

Specialist agents (minimum viable set — add more if it's cheap to do):
1. **Filing Q&A Agent** — answers direct questions against a specific filing ("What was Apple's R&D spend in FY2024?")
2. **Comparison Agent** — compares a metric or disclosure across multiple companies or multiple periods of the same company
3. **Risk/Red-Flag Agent** — surfaces and summarizes risk-factor language, going-concern notes, or unusual disclosures from the "Risk Factors" section
4. **Summarization Agent** — produces an executive summary of a filing section on request

Each agent should have a narrow, explicit tool schema (not "call the LLM and hope"). Log which agent handled each query so you can show routing accuracy as a metric.

## 4. Build phases

Work through these in order. At the end of each phase, stop, summarize what was built, and show me the output (test results, a sample query/response, an MLflow screenshot description, etc.) before moving to the next phase. Commit to git at the end of each phase with a clear message.

**Phase 1 — Data ingestion & chunking**
SEC EDGAR API → PDF/HTML filing parser → recursive/semantic text chunking → metadata tagging (company, ticker, filing type, date, section). Store raw + chunked docs (local disk or S3, your call — note the tradeoff in the README). Pull filings for 5–10 real companies across at least 2 sectors so comparison queries have something to compare.

**Phase 2 — Embeddings & vector store**
Benchmark the 2+ embedding models mentioned above using MLflow to log retrieval quality on a small hand-labeled query set. Index into FAISS (dev) and Pinecone (prod path). Document which one you'd pick for production and why.

**Phase 3 — Multi-agent orchestration**
Build the Router Agent + 4 specialist agents described in section 3. Wire them through LangGraph (or LangChain agents/tools if that's cleaner). Add routing-decision logging.

**Phase 4 — Evaluation framework**
Build a labeled eval set (20–30 question/expected-answer pairs across the 4 agent types). Run RAGAS metrics. Log results to MLflow. This is the section that will most impress a technical interviewer — treat it with the same rigor as a model-evaluation section on my resume (ROC-AUC/precision/recall equivalents).

**Phase 5 — API + deployment**
FastAPI service exposing `/query`, `/compare`, `/health`, and an endpoint to list ingested filings. Containerize with Docker. Deploy the containerized service somewhere actually reachable. Write the AWS SageMaker/Lambda deployment path as a documented alternative (diagram + steps) even if you don't fully deploy it there — but be honest in the README about what's actually live vs. what's "designed for."

**Phase 6 — Docs, tests, polish**
- README with: architecture diagram (mermaid is fine), setup instructions, a real example query and response, the evaluation results table, and an honest "what I'd do next at scale" section
- pytest coverage for ingestion, retrieval, each agent, and the API layer
- A 60–90 second demo script I can literally read out loud in an interview, referencing specific files/metrics in the repo
- GitHub Actions workflow running lint + tests on push

## 5. Working style instructions for Claude Code

- Ask me clarifying questions before Phase 1 if anything above is ambiguous (e.g., which companies to pull, Pinecone API key setup) — don't guess on things that are cheap to ask about.
- Prefer real, working code over stubs. If something can't be finished in a session, leave a clearly marked `TODO` with a one-line reason, not silent placeholder logic.
- Keep secrets out of git — use `.env` + `.env.example`.
- After each phase, give me a short honest status: what works, what's mocked/simplified, what's genuinely production-grade vs. portfolio-grade.
- Optimize for *defensibility in an interview* over feature count — I'd rather have 4 agents that genuinely work and that I can explain line-by-line than 8 that are shallow.

## 6. Definition of done

The project is done when I can:
1. Clone the repo and run it locally in under 10 minutes following the README
2. Hit a live deployed endpoint and get a real answer from a real filing
3. Point to an MLflow run and an evaluation results table with real numbers
4. Explain the router/agent architecture from memory using the README diagram
5. Point to the GitHub Actions tab and show green CI runs
