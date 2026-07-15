"""CLI entry point for Phase 1 ingestion.

Usage: python scripts/ingest.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finsight.ingestion.pipeline import ingest_all  # noqa: E402

if __name__ == "__main__":
    summary = ingest_all()
    print(json.dumps(summary, indent=2))
