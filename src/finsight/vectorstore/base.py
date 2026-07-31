from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class QueryResult:
    id: str
    score: float
    metadata: dict[str, Any]


class VectorStore(ABC):
    """Common interface so the retrieval layer doesn't care whether it's
    talking to local FAISS (dev) or Pinecone (prod) — swap via config.VECTOR_STORE."""

    @abstractmethod
    def upsert(
        self, ids: list[str], vectors: np.ndarray, metadatas: list[dict[str, Any]]
    ) -> None: ...

    @abstractmethod
    def query(self, vector: np.ndarray, top_k: int) -> list[QueryResult]: ...

    @abstractmethod
    def save(self) -> None: ...
