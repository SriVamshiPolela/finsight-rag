from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from finsight.embeddings.base import EmbeddingModel


class SentenceTransformerEmbedding(EmbeddingModel):
    """Wraps any sentence-transformers model, applying model-specific
    query/passage prefixes (required by e.g. E5, no-op for e.g. BGE-v1.5)."""

    def __init__(
        self,
        name: str,
        hf_name: str,
        dimension: int,
        query_prefix: str = "",
        passage_prefix: str = "",
    ):
        self.name = name
        self.dimension = dimension
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self._model = SentenceTransformer(hf_name)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        prefixed = [self.passage_prefix + t for t in texts]
        return self._model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(prefixed) > 50,
        ).astype("float32")

    def embed_query(self, text: str) -> np.ndarray:
        vec = self._model.encode(
            self.query_prefix + text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.astype("float32")
