"""LongMemEval-S raw-data adapter for the comparison harness."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT_DIR / "benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from longmemeval_bench import compute_recall_at_k, load_dataset, session_to_text

RAW_DATA_PATH = Path("/tmp/longmemeval-data/longmemeval_s.json")


def load_raw_dataset(
    path: Path = RAW_DATA_PATH,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load the raw S split through the existing benchmark loader."""
    data = load_dataset(str(path))
    return data[:limit] if limit is not None else data


__all__ = [
    "RAW_DATA_PATH",
    "compute_recall_at_k",
    "load_raw_dataset",
    "session_to_text",
]
