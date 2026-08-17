# W2 completion

実装・検証は `benchmarks/comparison/**` に限定し、commit/pushは実行していない。数値は生成summary JSONから読み込んだ。

## 実行コマンド

- `.venv/bin/python benchmarks/comparison/rebuild_comparison_csv.py`
- `.venv/bin/python -m py_compile benchmarks/comparison/*.py`
- `.venv/bin/python -m unittest benchmarks/comparison/test_w2.py`
- `.venv/bin/python benchmarks/comparison/make_table.py`
- `.venv/bin/python benchmarks/comparison/run_comparison.py --bench longmemeval --system bm25 --top-k 5`
- `.venv/bin/python benchmarks/comparison/run_comparison.py --bench longmemeval --system bm25 --top-k 10`
- `.venv/bin/python benchmarks/comparison/run_comparison.py --bench longmemeval --system rag --top-k 5`
- `.venv/bin/python benchmarks/comparison/run_comparison.py --bench longmemeval --system rag --top-k 10`
- `.venv/bin/python benchmarks/longmemeval_bench.py /tmp/longmemeval-data/longmemeval_s.json --mode enhanced_v2_boost --top-k 5 --no-progress --output benchmarks/comparison/results/longmemeval_raw_erinys/top_k_5`
- `.venv/bin/python benchmarks/longmemeval_bench.py /tmp/longmemeval-data/longmemeval_s.json --mode enhanced_v2_boost --top-k 10 --no-progress --output benchmarks/comparison/results/longmemeval_raw_erinys/top_k_10`
- `.venv/bin/python benchmarks/comparison/run_mem0_smoke.py --bench locomo --limit 5`
- `.venv/bin/python benchmarks/comparison/run_mem0_smoke.py --bench convomem --limit 2`
- `.venv/bin/python benchmarks/comparison/mem0_cost_estimate.py`
- `.venv/bin/python benchmarks/comparison/write_w2_completion.py`

## 生成ファイル

- `README.md`, `common.py`, `run_comparison.py`, `make_table.py`
- `longmemeval_loader.py`, `rebuild_comparison_csv.py`
- `mem0_adapter.py`, `run_mem0_smoke.py`, `mem0_cost_estimate.py`
- `test_w2.py`, `MEM0_COST_ESTIMATE.md`, `W2_COMPLETION.md`
- `results/comparison.csv`
- `results/summary_bm25_longmemeval_20260817_095207.json`
- `results/summary_bm25_longmemeval_20260817_095226.json`
- `results/summary_rag_longmemeval_20260817_095529.json`
- `results/summary_rag_longmemeval_20260817_095853.json`
- `results/summary_mem0_convomem_20260817_100049.json`
- `results/summary_mem0_locomo_20260817_100049.json`
- `results/longmemeval_raw_erinys/top_k_5/summary_erinys_enhanced_v2_boost_20260817_1000.json`
- `results/longmemeval_raw_erinys/top_k_10/summary_erinys_enhanced_v2_boost_20260817_1000.json`

## LongMemEval-S raw baseline (n=500)

| System | Top-k | N | R@k % | Perfect % | p50 ms | p95 ms |
|:--|--:|--:|--:|--:|--:|--:|
| bm25 | 5 | 500 | 96.8 | 96.8 | 0.165 | 0.315 |
| bm25 | 10 | 500 | 98.4 | 98.4 | 0.167 | 0.399 |
| rag | 5 | 500 | 92.2 | 92.2 | 4.151 | 9.927 |
| rag | 10 | 500 | 95.2 | 95.2 | 4.397 | 9.191 |

## ERINYS raw再測定（baseline比較とは別物）

| Top-k directory | N | R@5 % | R@10 % |
|:--|--:|--:|--:|
| top_k_5 | 500 | 99.4 | 100.0 |
| top_k_10 | 500 | 99.4 | 100.0 |

## Mem0 smoke / cost gate

| Benchmark | Status | Requested n | Evaluated n | Actual LLM calls | Reason |
|:--|:--|--:|--:|--:|:--|
| convomem | SKIP | 10 | 0 | 0 | missing_required_env:OPENAI_API_KEY |
| locomo | SKIP | 5 | 0 | 0 | missing_required_env:OPENAI_API_KEY |

## 未解決事項

- LongMemEvalのcleaned pathは未取得のまま。今回のbaselineとERINYS再測定はraw版で、`BENCHMARKS.md`の既存cleaned値とは比較していない。
- Mem0 smokeは `OPENAI_API_KEY` 未取得のためSKIP。構築・LLM呼び出し・採点は未実行で、full計測も未実行。
- Mem0のUSDは、実行済みprovider使用量と確認済み価格が無いため `UNAVAILABLE`。`MEM0_COST_ESTIMATE.md` にはデータ由来の呼び出し数／入力トークン概算を残した。
- Mem0 skipped benchmarks: convomem, locomo.
- ConvoMem full projectionは現キャッシュの5カテゴリ（changing_evidence欠損）に基づく。
- RAGのembedding modelはW1から継承した共有engineで、各summaryの `requested_rag_model` と `embedding_model` を区別して記録した。
