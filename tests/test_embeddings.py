from unittest.mock import MagicMock, patch

import numpy as np

from finsight.embeddings.sentence_transformer import SentenceTransformerEmbedding


def make_embedding(**kwargs):
    with patch("finsight.embeddings.sentence_transformer.SentenceTransformer") as mock_cls:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros((2, 4), dtype="float32")
        mock_cls.return_value = mock_model
        emb = SentenceTransformerEmbedding(
            name=kwargs.get("name", "test-model"),
            hf_name="fake/model",
            dimension=4,
            query_prefix=kwargs.get("query_prefix", ""),
            passage_prefix=kwargs.get("passage_prefix", ""),
        )
        return emb, mock_model


def test_embed_passages_applies_passage_prefix():
    emb, mock_model = make_embedding(passage_prefix="passage: ")
    emb.embed_passages(["hello", "world"])

    called_texts = mock_model.encode.call_args[0][0]
    assert called_texts == ["passage: hello", "passage: world"]
    assert mock_model.encode.call_args.kwargs["normalize_embeddings"] is True


def test_embed_query_applies_query_prefix():
    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros(4, dtype="float32")
    with patch("finsight.embeddings.sentence_transformer.SentenceTransformer", return_value=mock_model):
        emb = SentenceTransformerEmbedding(
            name="test-model", hf_name="fake/model", dimension=4, query_prefix="query: "
        )
        emb.embed_query("what are the risks?")

    called_text = mock_model.encode.call_args[0][0]
    assert called_text == "query: what are the risks?"


def test_no_prefix_leaves_text_unchanged():
    emb, mock_model = make_embedding(passage_prefix="", query_prefix="")
    emb.embed_passages(["plain text"])
    assert mock_model.encode.call_args[0][0] == ["plain text"]
