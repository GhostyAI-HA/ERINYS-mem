#!/usr/bin/env python3
"""Create a data-derived Mem0 full-run cost gate report."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

import common
from mem0_adapter import estimate_text_tokens, settings_snapshot

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_PATH = Path(__file__).resolve().parent / "MEM0_COST_ESTIMATE.md"
BENCHMARKS = ("locomo", "convomem")


def main() -> None:
    reports = [_benchmark_report(benchmark) for benchmark in BENCHMARKS]
    OUTPUT_PATH.write_text(_render(reports), encoding="utf-8")
    print(f"cost_estimate={OUTPUT_PATH}")


def _benchmark_report(benchmark: str) -> dict[str, object]:
    summary = _latest_summary(benchmark)
    smoke = _mapping(summary.get("smoke"))
    full_cases = common.load_cases(benchmark, None)
    full_documents = sum(len(case.documents) for case in full_cases)
    full_tokens = sum(
        estimate_text_tokens(document.text)
        for case in full_cases
        for document in case.documents
    )
    smoke_documents = int(smoke.get("prepared_documents", 0))
    smoke_calls = int(smoke.get("planned_llm_calls", 0))
    return {
        "benchmark": benchmark,
        "status": summary.get("status", "unknown"),
        "smoke_summary": str(_summary_path(summary)),
        "smoke_cases": summary.get("n", 0),
        "smoke_documents": smoke_documents,
        "full_cases": len(full_cases),
        "full_documents": full_documents,
        "full_input_tokens_approx": full_tokens,
        "projected_llm_calls": _scale(smoke_calls, smoke_documents, full_documents),
        "projected_input_tokens": _scale(
            int(smoke.get("input_tokens_approx", 0)), smoke_documents, full_documents
        ),
        "projected_output_tokens": _projected_output(summary, full_documents, smoke_documents),
    }


def _latest_summary(benchmark: str) -> dict[str, object]:
    paths = sorted(RESULTS_DIR.glob(f"summary_mem0_{benchmark}_*.json"))
    if not paths:
        raise FileNotFoundError(f"No Mem0 summary for {benchmark}")
    path = paths[-1]
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Invalid Mem0 summary: {path}")
    value["_summary_path"] = str(path)
    return value


def _summary_path(summary: dict[str, object]) -> str:
    return str(summary.get("_summary_path", "unknown"))


def _mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _scale(value: int, source: int, target: int) -> int | None:
    if source <= 0:
        return None
    return round(value * target / source)


def _projected_output(summary: dict[str, object], full_docs: int, smoke_docs: int) -> int | None:
    smoke = _mapping(summary.get("smoke"))
    actual = int(smoke.get("actual_output_tokens", 0))
    return _scale(actual, smoke_docs, full_docs) if actual else None


def _render(reports: list[dict[str, object]]) -> str:
    settings = settings_snapshot(RESULTS_DIR / "mem0_local")
    rates = _pricing_rates()
    lines = [
        "# Mem0 cost estimate (W2)",
        "",
        "Status: `SKIP` when the smoke summary is gated; no Mem0 LLM call is made in that state.",
        "",
        f"Settings (secrets omitted): `{json.dumps(settings, ensure_ascii=False, sort_keys=True)}`",
        "",
        "The projection is derived from the smoke summary and the full cached benchmark loader.",
        "The adapter performs one `Memory.add(infer=True)` call per source document; input tokens use the recorded approximation when provider usage is unavailable.",
        "",
        "| Benchmark | Smoke status | Smoke docs | Full cases | Full docs | Projected LLM calls | Projected input tokens | Projected output tokens | USD |",
        "|:--|:--|--:|--:|--:|--:|--:|--:|:--|",
    ]
    lines.extend(_render_row(report, rates) for report in reports)
    lines.extend(_pricing_note(rates))
    return "\n".join(lines) + "\n"


def _render_row(report: dict[str, object], rates: tuple[float, float] | None) -> str:
    cells = [
        report["benchmark"], report["status"], report["smoke_documents"],
        report["full_cases"], report["full_documents"], report["projected_llm_calls"],
        report["projected_input_tokens"], _display(report["projected_output_tokens"]),
        _usd_estimate(report, rates),
    ]
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def _display(value: object) -> object:
    return "UNAVAILABLE" if value is None else value


def _pricing_rates() -> tuple[float, float] | None:
    input_rate = os.getenv("MEM0_INPUT_USD_PER_1M")
    output_rate = os.getenv("MEM0_OUTPUT_USD_PER_1M")
    if not input_rate or not output_rate:
        return None
    try:
        return float(input_rate), float(output_rate)
    except ValueError:
        return None


def _usd_estimate(
    report: Mapping[str, object],
    rates: tuple[float, float] | None,
) -> str:
    output_tokens = report["projected_output_tokens"]
    if rates is None or output_tokens is None:
        return "UNAVAILABLE"
    input_tokens = float(report["projected_input_tokens"] or 0)
    value = (input_tokens * rates[0] + float(output_tokens) * rates[1]) / 1_000_000
    return f"{value:.6f}"


def _pricing_note(rates: tuple[float, float] | None) -> list[str]:
    if rates is not None:
        return ["", "USD rates came from `MEM0_INPUT_USD_PER_1M` and `MEM0_OUTPUT_USD_PER_1M`; this remains a projection because the smoke was not executed."]
    return [
        "",
        "USD: `UNAVAILABLE` because the smoke was not executed and no provider pricing rates were supplied. Set `MEM0_INPUT_USD_PER_1M` and `MEM0_OUTPUT_USD_PER_1M` only after confirming the current model pricing.",
        "",
        "Full Mem0 execution remains blocked pending explicit approval and credentials.",
    ]


if __name__ == "__main__":
    main()
