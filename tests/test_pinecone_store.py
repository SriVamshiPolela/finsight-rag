from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from finsight.vectorstore.pinecone_store import PineconeVectorStore


def make_store(existing_indexes=None):
    with patch("finsight.vectorstore.pinecone_store.Pinecone") as mock_pinecone_cls:
        mock_client = MagicMock()
        mock_client.list_indexes.return_value = [{"name": n} for n in (existing_indexes or [])]
        mock_index = MagicMock()
        mock_client.Index.return_value = mock_index
        mock_pinecone_cls.return_value = mock_client

        store = PineconeVectorStore(api_key="fake-key", index_name="finsight-rag", dimension=4)
        return store, mock_client, mock_index


def test_requires_api_key():
    with pytest.raises(ValueError):
        PineconeVectorStore(api_key="", index_name="finsight-rag", dimension=4)


def test_creates_index_when_missing():
    store, mock_client, mock_index = make_store(existing_indexes=[])
    mock_client.create_index.assert_called_once()


def test_skips_create_when_index_already_exists():
    store, mock_client, mock_index = make_store(existing_indexes=["finsight-rag"])
    mock_client.create_index.assert_not_called()


def test_upsert_converts_vectors_and_batches():
    store, mock_client, mock_index = make_store(existing_indexes=["finsight-rag"])
    ids = [f"id{i}" for i in range(150)]
    vectors = np.zeros((150, 4), dtype="float32")
    metas = [{"ticker": "AAA"} for _ in range(150)]

    store.upsert(ids, vectors, metas)

    assert mock_index.upsert.call_count == 2  # batch size 100 -> 100 + 50
    first_batch = mock_index.upsert.call_args_list[0].kwargs["vectors"]
    assert first_batch[0]["id"] == "id0"
    assert first_batch[0]["values"] == [0.0, 0.0, 0.0, 0.0]


def test_query_maps_pinecone_response_to_query_result():
    store, mock_client, mock_index = make_store(existing_indexes=["finsight-rag"])
    mock_index.query.return_value = {
        "matches": [{"id": "a", "score": 0.9, "metadata": {"ticker": "AAA"}}]
    }

    results = store.query(np.zeros(4, dtype="float32"), top_k=1)

    assert len(results) == 1
    assert results[0].id == "a"
    assert results[0].score == 0.9
    assert results[0].metadata == {"ticker": "AAA"}
