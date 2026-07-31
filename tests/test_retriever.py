import numpy as np

from finsight.embeddings.base import EmbeddingModel
from finsight.retrieval.retriever import Retriever
from finsight.vectorstore.faiss_store import FAISSVectorStore

DIM = 4


def unit(vec):
    arr = np.array(vec, dtype="float32")
    return arr / np.linalg.norm(arr)


class FakeEmbeddingModel(EmbeddingModel):
    """Returns whatever vector was registered for a given query string."""

    name = "fake"
    dimension = DIM

    def __init__(self, query_vectors: dict[str, np.ndarray]):
        self._query_vectors = query_vectors

    def embed_passages(self, texts):
        raise NotImplementedError

    def embed_query(self, text):
        return self._query_vectors[text]


def make_retriever(query="find risks"):
    store = FAISSVectorStore(dimension=DIM, index_dir=None)
    ids = ["aapl-risk", "aapl-biz", "msft-risk"]
    vectors = np.stack([unit([1, 0, 0, 0]), unit([0.9, 0.1, 0, 0]), unit([0.8, 0.2, 0, 0])])
    metas = [
        {"ticker": "AAPL", "section_key": "item1a", "text": "aapl risk text"},
        {"ticker": "AAPL", "section_key": "item1", "text": "aapl biz text"},
        {"ticker": "MSFT", "section_key": "item1a", "text": "msft risk text"},
    ]
    store.upsert(ids, vectors, metas)

    model = FakeEmbeddingModel({query: unit([1, 0, 0, 0])})
    return Retriever(model, store, default_top_k=5)


def test_search_without_filters_returns_all_ranked_by_similarity():
    retriever = make_retriever()
    results = retriever.search("find risks")
    assert [r.id for r in results] == ["aapl-risk", "aapl-biz", "msft-risk"]


def test_search_filters_by_ticker():
    retriever = make_retriever()
    results = retriever.search("find risks", ticker="AAPL")
    assert {r.id for r in results} == {"aapl-risk", "aapl-biz"}


def test_search_filters_by_ticker_and_section():
    retriever = make_retriever()
    results = retriever.search("find risks", ticker="AAPL", section_key="item1a")
    assert [r.id for r in results] == ["aapl-risk"]


def test_search_respects_top_k_after_filtering():
    retriever = make_retriever()
    results = retriever.search("find risks", ticker="AAPL", top_k=1)
    assert len(results) == 1
    assert results[0].id == "aapl-risk"
