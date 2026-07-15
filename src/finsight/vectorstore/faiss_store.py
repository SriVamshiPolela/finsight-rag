from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from finsight.vectorstore.base import QueryResult, VectorStore


class FAISSVectorStore(VectorStore):
    """Local, disk-persisted vector store for the dev path.

    Uses IndexFlatIP (exact inner-product search) over pre-normalized
    vectors, which is equivalent to cosine similarity. Fine for a few
    thousand chunks; would need an approximate index (IVF/HNSW) at real scale.
    """

    def __init__(self, dimension: int, index_dir: Path):
        self.dimension = dimension
        self.index_dir = index_dir
        self._index = faiss.IndexFlatIP(dimension)
        self._ids: list[str] = []
        self._metadatas: list[dict[str, Any]] = []

    @property
    def index_path(self) -> Path:
        return self.index_dir / "index.faiss"

    @property
    def meta_path(self) -> Path:
        return self.index_dir / "meta.jsonl"

    def upsert(
        self, ids: list[str], vectors: np.ndarray, metadatas: list[dict[str, Any]]
    ) -> None:
        if len(ids) != len(metadatas) or len(ids) != vectors.shape[0]:
            raise ValueError("ids, vectors, and metadatas must be the same length")
        self._index.add(vectors.astype("float32"))
        self._ids.extend(ids)
        self._metadatas.extend(metadatas)

    def query(self, vector: np.ndarray, top_k: int) -> list[QueryResult]:
        if self._index.ntotal == 0:
            return []
        scores, indices = self._index.search(
            vector.reshape(1, -1).astype("float32"), min(top_k, self._index.ntotal)
        )
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(
                QueryResult(id=self._ids[idx], score=float(score), metadata=self._metadatas[idx])
            )
        return results

    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))
        with self.meta_path.open("w", encoding="utf-8") as f:
            for chunk_id, meta in zip(self._ids, self._metadatas):
                f.write(json.dumps({"id": chunk_id, "metadata": meta}) + "\n")

    @classmethod
    def load(cls, dimension: int, index_dir: Path) -> "FAISSVectorStore":
        store = cls(dimension, index_dir)
        store._index = faiss.read_index(str(store.index_path))
        store._ids = []
        store._metadatas = []
        with store.meta_path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                store._ids.append(row["id"])
                store._metadatas.append(row["metadata"])
        return store
