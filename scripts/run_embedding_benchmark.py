"""Phase 2: embed the Phase 1 chunk corpus with each candidate model, index
into FAISS, run the hand-labeled eval query set, and log everything to MLflow
so the two models can be compared on retrieval quality, not vibes.

Usage: python scripts/run_embedding_benchmark.py
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mlflow  # noqa: E402

from finsight.config import EMBEDDING_MODEL_SPECS, EVAL_TOP_K, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, PROCESSED_DIR  # noqa: E402
from finsight.embeddings.pipeline import build_index_for_model, load_chunks  # noqa: E402
from finsight.embeddings.registry import get_embedding_model  # noqa: E402
from finsight.eval.queries import EVAL_QUERIES  # noqa: E402
from finsight.eval.retrieval_metrics import aggregate, evaluate_query  # noqa: E402
from finsight.vectorstore.faiss_store import FAISSVectorStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run() -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    chunks = load_chunks()
    all_results = {}

    for model_key in EMBEDDING_MODEL_SPECS:
        with mlflow.start_run(run_name=model_key):
            model = get_embedding_model(model_key)
            build_summary = build_index_for_model(model_key, chunks=chunks, model=model)
            store = FAISSVectorStore.load(model.dimension, Path(build_summary["index_dir"]))

            per_query = []
            for q in EVAL_QUERIES:
                query_vector = model.embed_query(q.query)
                results = store.query(query_vector, top_k=EVAL_TOP_K)
                per_query.append(evaluate_query(q, results))
            metrics = aggregate(per_query)

            mlflow.log_params(
                {
                    "model_key": model_key,
                    "hf_name": EMBEDDING_MODEL_SPECS[model_key]["hf_name"],
                    "dimension": model.dimension,
                    "chunk_count": build_summary["chunk_count"],
                    "eval_query_count": len(EVAL_QUERIES),
                    "top_k": EVAL_TOP_K,
                }
            )
            mlflow.log_metrics(metrics)

            logger.info("%s: %s", model_key, metrics)
            all_results[model_key] = {**build_summary, **metrics}

    results_path = PROCESSED_DIR / "embedding_benchmark.json"
    results_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    logger.info("Wrote comparison results to %s", results_path)
    return all_results


if __name__ == "__main__":
    summary = run()
    print(json.dumps(summary, indent=2))
