"""Production vector-store path. Code-complete and unit-tested against a
mocked client, but not exercised against a live Pinecone index in this repo
(no API key provisioned yet) — see README for the documented tradeoff.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pinecone import Pinecone, ServerlessSpec

from finsight.vectorstore.base import QueryResult, VectorStore

_UPSERT_BATCH_SIZE = 100


class PineconeVectorStore(VectorStore):
    def __init__(
        self,
        api_key: str,
        index_name: str,
        dimension: int,
        cloud: str = "aws",
        region: str = "us-east-1",
        namespace: str = "",
    ):
        if not api_key:
            raise ValueError(
                "PINECONE_API_KEY is required to use the pinecone vector store"
            )
        self.namespace = namespace
        self._client = Pinecone(api_key=api_key)
        if index_name not in [idx["name"] for idx in self._client.list_indexes()]:
            self._client.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
        self._index = self._client.Index(index_name)

    def upsert(
        self, ids: list[str], vectors: np.ndarray, metadatas: list[dict[str, Any]]
    ) -> None:
        records = [
            {"id": chunk_id, "values": vector.tolist(), "metadata": metadata}
            for chunk_id, vector, metadata in zip(ids, vectors, metadatas)
        ]
        for i in range(0, len(records), _UPSERT_BATCH_SIZE):
            self._index.upsert(
                vectors=records[i : i + _UPSERT_BATCH_SIZE], namespace=self.namespace
            )

    def query(self, vector: np.ndarray, top_k: int) -> list[QueryResult]:
        response = self._index.query(
            vector=vector.tolist(),
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace,
        )
        return [
            QueryResult(id=match["id"], score=float(match["score"]), metadata=match.get("metadata", {}))
            for match in response["matches"]
        ]

    def save(self) -> None:
        # No-op: Pinecone is a managed, always-persisted service, unlike the
        # local FAISS path which needs an explicit flush to disk.
        pass
