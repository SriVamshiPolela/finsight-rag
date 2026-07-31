from __future__ import annotations

from pathlib import Path

from finsight import config
from finsight.vectorstore.base import VectorStore


def get_vector_store(dimension: int, index_dir: Path) -> VectorStore:
    """Returns FAISS or Pinecone based on config.VECTOR_STORE ("faiss" | "pinecone")."""
    if config.VECTOR_STORE == "faiss":
        from finsight.vectorstore.faiss_store import FAISSVectorStore

        return FAISSVectorStore(dimension=dimension, index_dir=index_dir)

    if config.VECTOR_STORE == "pinecone":
        from finsight.vectorstore.pinecone_store import PineconeVectorStore

        return PineconeVectorStore(
            api_key=config.PINECONE_API_KEY,
            index_name=config.PINECONE_INDEX_NAME,
            dimension=dimension,
            cloud=config.PINECONE_CLOUD,
            region=config.PINECONE_REGION,
        )

    raise ValueError(f"Unknown VECTOR_STORE '{config.VECTOR_STORE}', expected 'faiss' or 'pinecone'")
