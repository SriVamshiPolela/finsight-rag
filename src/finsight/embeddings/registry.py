from finsight.config import EMBEDDING_MODEL_SPECS
from finsight.embeddings.base import EmbeddingModel
from finsight.embeddings.sentence_transformer import SentenceTransformerEmbedding


def get_embedding_model(key: str) -> EmbeddingModel:
    try:
        spec = EMBEDDING_MODEL_SPECS[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown embedding model '{key}'. Known: {list(EMBEDDING_MODEL_SPECS)}"
        ) from exc
    return SentenceTransformerEmbedding(
        name=key,
        hf_name=spec["hf_name"],
        dimension=spec["dimension"],
        query_prefix=spec["query_prefix"],
        passage_prefix=spec["passage_prefix"],
    )
