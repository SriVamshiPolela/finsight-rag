"""Embedding model interface. Query/passage are embedded separately because
several retrieval-tuned models (e.g. E5) require different prefixes for each."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingModel(ABC):
    name: str
    dimension: int

    @abstractmethod
    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """Returns an (N, dimension) float32 array of L2-normalized embeddings."""

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Returns a (dimension,) float32 L2-normalized embedding."""
