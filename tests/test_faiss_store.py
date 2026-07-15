import numpy as np
import pytest

from finsight.vectorstore.faiss_store import FAISSVectorStore

DIM = 4


def unit(vec: list[float]) -> np.ndarray:
    arr = np.array(vec, dtype="float32")
    return arr / np.linalg.norm(arr)


def test_upsert_and_query_returns_closest_match_first():
    store = FAISSVectorStore(dimension=DIM, index_dir=None)
    ids = ["a", "b", "c"]
    vectors = np.stack([unit([1, 0, 0, 0]), unit([0, 1, 0, 0]), unit([0.9, 0.1, 0, 0])])
    metas = [{"ticker": "AAA"}, {"ticker": "BBB"}, {"ticker": "CCC"}]

    store.upsert(ids, vectors, metas)
    results = store.query(unit([1, 0, 0, 0]), top_k=2)

    assert [r.id for r in results] == ["a", "c"]
    assert results[0].score > results[1].score
    assert results[0].metadata == {"ticker": "AAA"}


def test_query_top_k_capped_at_available_vectors():
    store = FAISSVectorStore(dimension=DIM, index_dir=None)
    store.upsert(["a"], np.stack([unit([1, 0, 0, 0])]), [{"ticker": "AAA"}])

    results = store.query(unit([1, 0, 0, 0]), top_k=5)

    assert len(results) == 1


def test_query_on_empty_store_returns_empty_list():
    store = FAISSVectorStore(dimension=DIM, index_dir=None)
    assert store.query(unit([1, 0, 0, 0]), top_k=3) == []


def test_upsert_rejects_mismatched_lengths():
    store = FAISSVectorStore(dimension=DIM, index_dir=None)
    with pytest.raises(ValueError):
        store.upsert(["a", "b"], np.stack([unit([1, 0, 0, 0])]), [{"ticker": "AAA"}])


def test_save_and_load_round_trip(tmp_path):
    store = FAISSVectorStore(dimension=DIM, index_dir=tmp_path)
    ids = ["a", "b"]
    vectors = np.stack([unit([1, 0, 0, 0]), unit([0, 1, 0, 0])])
    metas = [{"ticker": "AAA", "text": "hello"}, {"ticker": "BBB", "text": "world"}]
    store.upsert(ids, vectors, metas)
    store.save()

    loaded = FAISSVectorStore.load(dimension=DIM, index_dir=tmp_path)
    results = loaded.query(unit([1, 0, 0, 0]), top_k=1)

    assert results[0].id == "a"
    assert results[0].metadata == {"ticker": "AAA", "text": "hello"}
