#!/usr/bin/env python3
"""Run the strictly bounded Mem0 smoke path or emit a cost-gate SKIP."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import common
from mem0_adapter import (
    CostGateBlocked,
    LLMCallMeter,
    Mem0System,
    Mem0Gate,
    credential_gate,
    estimate_text_tokens,
    settings_snapshot,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SMOKE_LIMITS = {"locomo": 5, "convomem": 2}
DEFAULT_TOP_K = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the bounded Mem0 smoke")
    parser.add_argument("--bench", choices=sorted(SMOKE_LIMITS), required=True)
    parser.add_argument("--limit", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _check_smoke_limit(args.bench, args.limit)
    cases = common.load_cases(args.bench, args.limit)
    try:
        summary, results = _run_cases(args.bench, cases)
    except CostGateBlocked as error:
        summary, results = _skip_summary(args.bench, cases, error)
    result_path, summary_path = _write_outputs(args.bench, summary, results)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"results_jsonl={result_path}")
    print(f"summary_json={summary_path}")


def _check_smoke_limit(benchmark: str, limit: int) -> None:
    # Cost cap: the smoke may run BELOW the bound (smaller probes are cheaper),
    # never above it.
    ceiling = SMOKE_LIMITS[benchmark]
    if limit < 1 or limit > ceiling:
        raise SystemExit(f"Mem0 smoke is bounded to 1..{ceiling} for {benchmark}")


def _run_cases(
    benchmark: str,
    cases: list[common.QueryCase],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    gate = credential_gate()
    meter = LLMCallMeter()
    results: list[dict[str, object]] = []
    for case in cases:
        results.append(_run_case(case, meter))
    summary = _completed_summary(benchmark, cases, results, gate, meter)
    return summary, results


def _run_case(case: common.QueryCase, meter: LLMCallMeter) -> dict[str, object]:
    storage_dir = _case_storage_dir(case)
    system = Mem0System(case.case_id, storage_dir, meter)
    started = time.perf_counter()
    system.ingest(case.documents)
    ingest_s = time.perf_counter() - started
    started = time.perf_counter()
    ranked_ids = system.search(case.question, DEFAULT_TOP_K)
    search_ms = (time.perf_counter() - started) * 1000.0
    ranked = _ranked_documents(case, ranked_ids)
    recall, details = common.evaluate_case(case, ranked, DEFAULT_TOP_K)
    system.close()
    return {
        "case_id": case.case_id,
        "category": case.category,
        "recall": recall,
        "retrieved_ids": ranked_ids,
        "details": details,
        "ingest_wall_s": round(ingest_s, 3),
        "latency_ms": round(search_ms, 3),
    }


def _ranked_documents(
    case: common.QueryCase,
    ranked_ids: list[str],
) -> list[common.RankedDocument]:
    documents = {document.doc_id: document for document in case.documents}
    return [
        common.RankedDocument(documents[doc_id], 0.0)
        for doc_id in ranked_ids
        if doc_id in documents
    ]


def _completed_summary(
    benchmark: str,
    cases: list[common.QueryCase],
    results: list[dict[str, object]],
    gate: Mem0Gate,
    meter: LLMCallMeter,
) -> dict[str, object]:
    recalls = [float(result["recall"]) for result in results]
    latencies = [float(result["latency_ms"]) for result in results]
    ingest = sum(float(result["ingest_wall_s"]) for result in results)
    return {
        "system": "mem0",
        "benchmark": benchmark,
        "mode": "mem0_oss_local",
        "top_k": DEFAULT_TOP_K,
        "status": "COMPLETED",
        "n": len(results),
        "overall": common._overall(recalls),
        "latency": common._latency(latencies),
        "ingest_wall_s": round(ingest, 3),
        "retrieval_llm_calls": meter.calls,
        "mem0_settings": settings_snapshot(RESULTS_DIR / "mem0_local"),
        "smoke": _smoke_stats(cases, meter),
        "gate": gate.__dict__,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def _skip_summary(
    benchmark: str,
    cases: list[common.QueryCase],
    error: CostGateBlocked,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    gate = error.decision
    return {
        "system": "mem0",
        "benchmark": benchmark,
        "mode": "mem0_oss_local",
        "top_k": DEFAULT_TOP_K,
        "status": "SKIP",
        "n": len(cases),
        "evaluated_n": 0,
        "overall": {"r_at_k": None, "perfect_pct": None, "zero_pct": None},
        "latency": {"p50_ms": None, "p95_ms": None},
        "ingest_wall_s": None,
        "retrieval_llm_calls": 0,
        "mem0_settings": settings_snapshot(RESULTS_DIR / "mem0_local"),
        "smoke": _smoke_stats(cases, LLMCallMeter()),
        "gate": gate.__dict__,
        "skip_reason": gate.reason,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }, []


def _smoke_stats(cases: list[common.QueryCase], meter: LLMCallMeter) -> dict[str, object]:
    documents = [document for case in cases for document in case.documents]
    tokens = sum(estimate_text_tokens(document.text) for document in documents)
    return {
        "requested_cases": len(cases),
        "prepared_documents": len(documents),
        "input_tokens_approx": tokens,
        "planned_llm_calls": len(documents),
        "actual_llm_calls": meter.calls,
        "actual_input_tokens": meter.input_tokens,
        "actual_output_tokens": meter.output_tokens,
        "call_basis": "one Mem0.add(infer=True) per source document",
        "token_basis": "character length divided by four when provider usage is absent",
    }


def _case_storage_dir(case: common.QueryCase) -> Path:
    digest = hashlib.sha256(case.case_id.encode()).hexdigest()[:16]
    return RESULTS_DIR / "mem0_local" / case.benchmark / digest


def _write_outputs(
    benchmark: str,
    summary: dict[str, object],
    results: list[dict[str, object]],
) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = RESULTS_DIR / f"results_mem0_{benchmark}_{timestamp}.jsonl"
    summary_path = RESULTS_DIR / f"summary_mem0_{benchmark}_{timestamp}.json"
    with result_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return result_path, summary_path


if __name__ == "__main__":
    main()
