#!/usr/bin/env python3
"""Render W2_COMPLETION.md from generated summaries and receipts."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUTPUT = ROOT / "W2_COMPLETION.md"


def main() -> None:
    lme = _latest_lme_summaries()
    mem0 = _latest_mem0_summaries()
    erinys = _erinys_summaries()
    OUTPUT.write_text(_render(lme, mem0, erinys), encoding="utf-8")
    print(f"completion={OUTPUT}")


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected object: {path}")
    value["_path"] = str(path.relative_to(ROOT))
    return value


def _latest_lme_summaries() -> list[dict[str, object]]:
    paths = sorted(RESULTS.glob("summary_*_longmemeval_*.json"))
    return [_load(path) for path in paths]


def _latest_mem0_summaries() -> list[dict[str, object]]:
    selected: dict[str, Path] = {}
    for path in sorted(RESULTS.glob("summary_mem0_*.json")):
        benchmark = path.name.split("_")[2]
        selected[benchmark] = path
    return [_load(selected[key]) for key in sorted(selected)]


def _erinys_summaries() -> list[dict[str, object]]:
    paths = sorted(
        (RESULTS / "longmemeval_raw_erinys").glob("top_k_*/summary_*.json"),
        key=lambda path: int(path.parent.name.removeprefix("top_k_")),
    )
    return [_load(path) for path in paths]


def _render(
    lme: list[dict[str, object]],
    mem0: list[dict[str, object]],
    erinys: list[dict[str, object]],
) -> str:
    lines = [
        "# W2 completion",
        "",
        "実装・検証は `benchmarks/comparison/**` に限定し、commit/pushは実行していない。数値は生成summary JSONから読み込んだ。",
        "",
        "## 実行コマンド",
        "",
    ]
    lines.extend(_commands())
    lines.extend(["", "## 生成ファイル", ""])
    lines.extend(_file_list(lme, mem0, erinys))
    lines.extend(["", f"## LongMemEval-S raw baseline (n={_common_n(lme)})", ""])
    lines.extend(_lme_table(lme))
    lines.extend(["", "## ERINYS raw再測定（baseline比較とは別物）", ""])
    lines.extend(_erinys_table(erinys))
    lines.extend(["", "## Mem0 smoke / cost gate", ""])
    lines.extend(_mem0_table(mem0))
    lines.extend(["", "## 未解決事項", ""])
    lines.extend(_unresolved(mem0))
    return "\n".join(lines) + "\n"


def _commands() -> list[str]:
    return [
        "- `.venv/bin/python benchmarks/comparison/rebuild_comparison_csv.py`",
        "- `.venv/bin/python -m py_compile benchmarks/comparison/*.py`",
        "- `.venv/bin/python -m unittest benchmarks/comparison/test_w2.py`",
        "- `.venv/bin/python benchmarks/comparison/make_table.py`",
        "- `.venv/bin/python benchmarks/comparison/run_comparison.py --bench longmemeval --system bm25 --top-k 5`",
        "- `.venv/bin/python benchmarks/comparison/run_comparison.py --bench longmemeval --system bm25 --top-k 10`",
        "- `.venv/bin/python benchmarks/comparison/run_comparison.py --bench longmemeval --system rag --top-k 5`",
        "- `.venv/bin/python benchmarks/comparison/run_comparison.py --bench longmemeval --system rag --top-k 10`",
        "- `.venv/bin/python benchmarks/longmemeval_bench.py /tmp/longmemeval-data/longmemeval_s.json --mode enhanced_v2_boost --top-k 5 --no-progress --output benchmarks/comparison/results/longmemeval_raw_erinys/top_k_5`",
        "- `.venv/bin/python benchmarks/longmemeval_bench.py /tmp/longmemeval-data/longmemeval_s.json --mode enhanced_v2_boost --top-k 10 --no-progress --output benchmarks/comparison/results/longmemeval_raw_erinys/top_k_10`",
        "- `.venv/bin/python benchmarks/comparison/run_mem0_smoke.py --bench locomo --limit 5`",
        "- `.venv/bin/python benchmarks/comparison/run_mem0_smoke.py --bench convomem --limit 2`",
        "- `.venv/bin/python benchmarks/comparison/mem0_cost_estimate.py`",
        "- `.venv/bin/python benchmarks/comparison/write_w2_completion.py`",
    ]


def _file_list(
    lme: list[dict[str, object]],
    mem0: list[dict[str, object]],
    erinys: list[dict[str, object]],
) -> list[str]:
    files = [
        "- `README.md`, `common.py`, `run_comparison.py`, `make_table.py`",
        "- `longmemeval_loader.py`, `rebuild_comparison_csv.py`",
        "- `mem0_adapter.py`, `run_mem0_smoke.py`, `mem0_cost_estimate.py`",
        "- `test_w2.py`, `MEM0_COST_ESTIMATE.md`, `W2_COMPLETION.md`",
        "- `results/comparison.csv`",
    ]
    files.extend(f"- `{item['_path']}`" for item in lme)
    files.extend(f"- `{item['_path']}`" for item in mem0)
    files.extend(f"- `{item['_path']}`" for item in erinys)
    return files


def _lme_table(rows: list[dict[str, object]]) -> list[str]:
    lines = [
        "| System | Top-k | N | R@k % | Perfect % | p50 ms | p95 ms |",
        "|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for row in rows:
        overall = _mapping(row.get("overall"))
        latency = _mapping(row.get("latency"))
        lines.append(
            f"| {row.get('system')} | {row.get('top_k')} | {row.get('n')} | "
            f"{overall.get('r_at_k')} | {overall.get('perfect_pct')} | "
            f"{latency.get('p50_ms')} | {latency.get('p95_ms')} |"
        )
    return lines


def _erinys_table(rows: list[dict[str, object]]) -> list[str]:
    lines = [
        "| Top-k directory | N | R@5 % | R@10 % |",
        "|:--|--:|--:|--:|",
    ]
    for row in rows:
        overall = _mapping(row.get("overall"))
        lines.append(
            f"| {Path(str(row['_path'])).parts[-2]} | {row.get('total_questions')} | "
            f"{overall.get('R@5')} | {overall.get('R@10')} |"
        )
    return lines


def _mem0_table(rows: list[dict[str, object]]) -> list[str]:
    lines = [
        "| Benchmark | Status | Requested n | Evaluated n | Actual LLM calls | Reason |",
        "|:--|:--|--:|--:|--:|:--|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('benchmark')} | {row.get('status')} | {row.get('n')} | "
            f"{row.get('evaluated_n', row.get('n'))} | {row.get('retrieval_llm_calls')} | "
            f"{row.get('skip_reason', '')} |"
        )
    return lines


def _unresolved(rows: list[dict[str, object]]) -> list[str]:
    skipped = [str(row.get("benchmark")) for row in rows if row.get("status") == "SKIP"]
    lines = [
        "- LongMemEvalのcleaned pathは未取得のまま。今回のbaselineとERINYS再測定はraw版で、`BENCHMARKS.md`の既存cleaned値とは比較していない。",
        "- Mem0 smokeは `OPENAI_API_KEY` 未取得のためSKIP。構築・LLM呼び出し・採点は未実行で、full計測も未実行。",
        "- Mem0のUSDは、実行済みprovider使用量と確認済み価格が無いため `UNAVAILABLE`。`MEM0_COST_ESTIMATE.md` にはデータ由来の呼び出し数／入力トークン概算を残した。",
        f"- Mem0 skipped benchmarks: {', '.join(skipped) if skipped else 'none'}.",
        "- ConvoMem full projectionは現キャッシュの5カテゴリ（changing_evidence欠損）に基づく。",
        "- RAGのembedding modelはW1から継承した共有engineで、各summaryの `requested_rag_model` と `embedding_model` を区別して記録した。",
    ]
    return lines


def _mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _common_n(rows: list[dict[str, object]]) -> str:
    values = sorted({str(row.get("n", "unknown")) for row in rows})
    return values[0] if len(values) == 1 else ", ".join(values)


if __name__ == "__main__":
    main()
