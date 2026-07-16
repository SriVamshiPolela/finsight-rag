from unittest.mock import patch

import pytest

from finsight import config
from finsight.vectorstore.factory import get_vector_store


def test_defaults_to_faiss(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VECTOR_STORE", "faiss")
    store = get_vector_store(dimension=4, index_dir=tmp_path)
    from finsight.vectorstore.faiss_store import FAISSVectorStore

    assert isinstance(store, FAISSVectorStore)


def test_dispatches_to_pinecone_when_configured(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "VECTOR_STORE", "pinecone")
    monkeypatch.setattr(config, "PINECONE_API_KEY", "fake-key")

    with patch("finsight.vectorstore.pinecone_store.Pinecone") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.list_indexes.return_value = [{"name": config.PINECONE_INDEX_NAME}]

        store = get_vector_store(dimension=4, index_dir=tmp_path)

    from finsight.vectorstore.pinecone_store import PineconeVectorStore

    assert isinstance(store, PineconeVectorStore)


def test_unknown_store_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "VECTOR_STORE", "not-a-real-store")
    with pytest.raises(ValueError):
        get_vector_store(dimension=4, index_dir=tmp_path)
