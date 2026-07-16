from __future__ import annotations

from finsight.config import INDEXES_DIR, PRODUCTION_EMBEDDING_MODEL, RETRIEVAL_CANDIDATE_POOL, RETRIEVAL_TOP_K
from finsight.embeddings.base import EmbeddingModel
from finsight.embeddings.registry import get_embedding_model
from finsight.vectorstore.base import QueryResult, VectorStore
from finsight.vectorstore.faiss_store import FAISSVectorStore


class Retriever:
    """Semantic search over the indexed chunk corpus, with client-side
    ticker/section filtering (see RETRIEVAL_CANDIDATE_POOL for why)."""

    def __init__(self, model: EmbeddingModel, store: VectorStore, default_top_k: int = RETRIEVAL_TOP_K):
        self.model = model
        self.store = store
        self.default_top_k = default_top_k

    @classmethod
    def load(cls, model_key: str = PRODUCTION_EMBEDDING_MODEL, top_k: int = RETRIEVAL_TOP_K) -> "Retriever":
        model = get_embedding_model(model_key)
        store = FAISSVectorStore.load(model.dimension, INDEXES_DIR / model_key)
        return cls(model, store, top_k)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        ticker: str | None = None,
        section_key: str | None = None,
    ) -> list[QueryResult]:
        top_k = top_k or self.default_top_k
        vector = self.model.embed_query(query)
        candidates = self.store.query(vector, top_k=RETRIEVAL_CANDIDATE_POOL)
        filtered = [c for c in candidates if _matches(c, ticker, section_key)]
        return filtered[:top_k]


def _matches(result: QueryResult, ticker: str | None, section_key: str | None) -> bool:
    if ticker and result.metadata.get("ticker") != ticker:
        return False
    if section_key and result.metadata.get("section_key") != section_key:
        return False
    return True
