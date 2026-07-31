from __future__ import annotations

import json
import logging

from finsight.config import CHUNKS_PATH, INDEXES_DIR
from finsight.embeddings.base import EmbeddingModel
from finsight.embeddings.registry import get_embedding_model
from finsight.vectorstore.factory import get_vector_store

logger = logging.getLogger(__name__)


def load_chunks() -> list[dict]:
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_index_for_model(
    model_key: str, chunks: list[dict] | None = None, model: EmbeddingModel | None = None
) -> dict:
    """Embeds every chunk with the given model and upserts into a vector
    store persisted at data/indexes/<model_key>/. Returns a summary dict."""
    chunks = chunks if chunks is not None else load_chunks()
    model = model if model is not None else get_embedding_model(model_key)

    logger.info("Embedding %d chunks with %s (%s)", len(chunks), model_key, model.dimension)
    vectors = model.embed_passages([c["text"] for c in chunks])

    store = get_vector_store(dimension=model.dimension, index_dir=INDEXES_DIR / model_key)
    ids = [c["chunk_id"] for c in chunks]
    store.upsert(ids=ids, vectors=vectors, metadatas=chunks)
    store.save()

    return {
        "model_key": model_key,
        "dimension": model.dimension,
        "chunk_count": len(chunks),
        "index_dir": str(INDEXES_DIR / model_key),
    }
