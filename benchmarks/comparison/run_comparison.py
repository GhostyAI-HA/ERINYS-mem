#!/usr/bin/env python3
"""Run deterministic baseline comparisons."""

from __future__ import annotations

import argparse
import json

import baseline_bm25
import baseline_rag
import common


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _system_config(system: str) -> tuple[str, common.IndexFactory]:
    if system == "rag":
        return baseline_rag.MODE, baseline_rag.build_index
    if system == "bm25":
        return baseline_bm25.MODE, baseline_bm25.build_index
    raise ValueError(f"Unsupported system: {system}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ERINYS baseline comparison")
    parser.add_argument(
        "--bench",
        choices=["locomo", "convomem", "longmemeval"],
        required=True,
    )
    parser.add_argument("--system", choices=["rag", "bm25"], required=True)
    parser.add_argument("--limit", type=_positive_int, default=None)
    parser.add_argument("--top-k", type=_positive_int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode, index_factory = _system_config(args.system)
    cases = common.load_cases(args.bench, args.limit)
    run = common.run_baseline(args.system, args.bench, mode, args.top_k, cases, index_factory)
    result_path, summary_path, csv_path = common.write_outputs(run)
    print(json.dumps(run["summary"], indent=2, ensure_ascii=False))
    print(f"results_jsonl={result_path}")
    print(f"summary_json={summary_path}")
    print(f"comparison_csv={csv_path}")


if __name__ == "__main__":
    main()
