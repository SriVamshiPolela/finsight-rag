"""Phase 4 live CLI: runs the full 24-case labeled eval set through the
router+agents once each, scores routing accuracy plus the four RAGAS-
methodology metrics (faithfulness, answer relevancy, context precision,
context recall) via an LLM judge, and logs the results to MLflow.

Requires ANTHROPIC_API_KEY (or OPENAI_API_KEY with LLM_PROVIDER=openai) in
.env - same requirement as scripts/run_query.py.

Usage: python scripts/run_agent_eval.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mlflow  # noqa: E402

from finsight.agents.graph import build_graph  # noqa: E402
from finsight.config import AGENT_EVAL_RESULTS_PATH, MLFLOW_AGENT_EVAL_EXPERIMENT_NAME, MLFLOW_TRACKING_URI  # noqa: E402
from finsight.eval.agent_cases import AGENT_EVAL_CASES  # noqa: E402
from finsight.eval.agent_eval import run_full_eval  # noqa: E402
from finsight.llm.registry import get_llm_provider  # noqa: E402
from finsight.retrieval.retriever import Retriever  # noqa: E402


def run() -> dict:
    llm = get_llm_provider()
    retriever = Retriever.load()
    graph = build_graph(llm, retriever)

    result = run_full_eval(llm, graph, AGENT_EVAL_CASES)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_AGENT_EVAL_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="phase4-agent-eval"):
        mlflow.log_param("case_count", len(AGENT_EVAL_CASES))
        mlflow.log_metrics(result["aggregate"])

    AGENT_EVAL_RESULTS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run()["aggregate"], indent=2))
