#!/usr/bin/env python3
"""Render comparison.csv and ERINYS summaries as Markdown and LaTeX."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import common

ERINYS_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _read_comparison_rows() -> list[dict[str, str]]:
    if not common.COMPARISON_CSV.exists():
        return []
    with common.COMPARISON_CSV.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _latest_erinys_rows() -> list[dict[str, str]]:
    latest: dict[str, tuple[str, dict[str, object]]] = {}
    for path in ERINYS_RESULTS_DIR.glob("summary_erinys_*_*.json"):
        summary = _load_summary(path)
        if not _is_target_summary(summary):
            continue
        benchmark = str(summary["benchmark"]).lower()
        stamp = _timestamp(path)
        if benchmark not in latest or stamp > latest[benchmark][0]:
            latest[benchmark] = (stamp, summary)
    return [_erinys_row(item[1]) for item in sorted(latest.values())]


def _load_summary(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Summary is not a JSON object: {path}")
    return value


def _is_target_summary(summary: dict[str, object]) -> bool:
    benchmark = str(summary.get("benchmark", "")).lower()
    mode = str(summary.get("mode", ""))
    return benchmark in {"locomo", "convomem"} and mode == "enhanced_v2_boost"


def _timestamp(path: Path) -> str:
    parts = path.stem.rsplit("_", 2)
    return "_".join(parts[-2:]) if len(parts) >= 2 else path.stem


def _erinys_row(summary: dict[str, object]) -> dict[str, str]:
    overall = summary.get("overall", {})
    timing = summary.get("timing", {})
    benchmark = str(summary["benchmark"]).lower()
    return {
        "system": "erinys",
        "benchmark": benchmark,
        "mode": str(summary["mode"]),
        "n": _erinys_n(summary),
        "r_at_k": _erinys_recall(overall),
        "perfect_pct": _erinys_perfect(overall),
        "latency_p50_ms": _erinys_latency(timing),
        "latency_p95_ms": "",
        "ingest_wall_s": _erinys_ingest(timing),
        "retrieval_llm_calls": "0",
    }


def _erinys_n(summary: dict[str, object]) -> str:
    return str(summary.get("total_questions", summary.get("total_items", "")))


def _erinys_recall(overall: object) -> str:
    data = overall if isinstance(overall, dict) else {}
    value = data.get("R@k", data.get("R@10", data.get("R@5", "")))
    return _fmt(value)


def _erinys_perfect(overall: object) -> str:
    data = overall if isinstance(overall, dict) else {}
    value = data.get("perfect_pct", data.get("R@10", data.get("R@5", "")))
    return _fmt(value)


def _erinys_latency(timing: object) -> str:
    data = timing if isinstance(timing, dict) else {}
    return _fmt(data.get("avg_search_per_q_ms", ""))


def _erinys_ingest(timing: object) -> str:
    data = timing if isinstance(timing, dict) else {}
    return _fmt(data.get("ingest_total_s", ""))


def _fmt(value: object) -> str:
    if value == "":
        return ""
    return f"{float(value):.3f}"


def _all_rows() -> list[dict[str, str]]:
    rows = _read_comparison_rows()
    rows.extend(_latest_erinys_rows())
    return sorted(rows, key=lambda row: (row["benchmark"], row["system"], row["mode"]))


def _display(value: str) -> str:
    return value if value else "-"


def render_markdown(rows: list[dict[str, str]]) -> str:
    header = "| System | Benchmark | Mode | N | R@k % | Perfect % | p50 ms | p95 ms | Ingest s | LLM calls |"
    divider = "|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|"
    body = [_markdown_row(row) for row in rows]
    return "\n".join([header, divider, *body])


def _markdown_row(row: dict[str, str]) -> str:
    cells = [
        row["system"], row["benchmark"], row["mode"], row["n"],
        row["r_at_k"], row["perfect_pct"], row["latency_p50_ms"],
        row["latency_p95_ms"], row["ingest_wall_s"],
        row["retrieval_llm_calls"],
    ]
    return "| " + " | ".join(_display(str(item)) for item in cells) + " |"


def render_latex(rows: list[dict[str, str]]) -> str:
    lines = [
        "\\begin{tabular}{lllrrrrrrr}",
        "\\hline",
        "System & Benchmark & Mode & N & R@k & Perfect & p50 & p95 & Ingest & LLM \\\\",
        "\\hline",
    ]
    lines.extend(_latex_row(row) for row in rows)
    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines)


def _latex_row(row: dict[str, str]) -> str:
    cells = [
        row["system"], row["benchmark"], row["mode"], row["n"],
        row["r_at_k"], row["perfect_pct"], row["latency_p50_ms"],
        row["latency_p95_ms"], row["ingest_wall_s"],
        row["retrieval_llm_calls"],
    ]
    return " & ".join(_latex_escape(_display(str(item))) for item in cells) + " \\\\"


def _latex_escape(value: str) -> str:
    return value.replace("_", "\\_")


def main() -> None:
    rows = _all_rows()
    print(render_markdown(rows))
    print()
    print(render_latex(rows))


if __name__ == "__main__":
    main()
