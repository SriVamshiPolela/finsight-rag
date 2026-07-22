# Bakes in the pre-built e5-small FAISS index and chunk corpus (Phase 1/2
# outputs) so the container is self-contained and reachable immediately -
# not "run the ingestion pipeline on first boot." At ~3.6k chunks this is a
# few MB; at real scale this would pull from S3/Pinecone at startup instead.
FROM python:3.12-slim

WORKDIR /app

# libgomp1: OpenMP runtime needed by torch/faiss-cpu CPU wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch first: the default PyPI wheel pulls ~1.5GB of unused NVIDIA
# CUDA libraries, and embeddings only ever run on CPU here.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Bake the embedding model weights in at build time so the first request
# doesn't pay a HuggingFace Hub download, and so this works even if the
# container's filesystem is read-only at runtime (e.g. ECS Fargate).
ENV HF_HOME=/opt/hf_cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/e5-small-v2')"

COPY src/ src/
COPY data/processed/chunks.jsonl data/processed/chunks.jsonl
COPY data/processed/manifest.json data/processed/manifest.json
COPY data/indexes/e5-small/ data/indexes/e5-small/

ENV PYTHONPATH=/app/src
ENV VECTOR_STORE=faiss
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "finsight.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
