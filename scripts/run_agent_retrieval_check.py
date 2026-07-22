"""Live, key-free half of Phase 4: checks that the retrieval step underlying
every one of the 24 labeled eval cases actually surfaces the expected
keywords, across all 4 agent types. No LLM involved - only the Phase 2 FAISS
index - so this runs for real right now, unlike scripts/run_agent_eval.py.

Usage: python scripts/run_agent_retrieval_check.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finsight.config import RETRIEVAL_CHECK_RESULTS_PATH  # noqa: E402
from finsight.eval.agent_cases import AGENT_EVAL_CASES  # noqa: E402
from finsight.eval.agent_retrieval_check import run_retrieval_check  # noqa: E402

if __name__ == "__main__":
    result = run_retrieval_check(AGENT_EVAL_CASES)
    RETRIEVAL_CHECK_RESULTS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
