# Comparison harness (W1 + W2)

このディレクトリは、既存ベンチマークのローダ／採点を再利用した比較用ハーネスです。生成物は `results/` 以下に限定します。`FINAL_REPORT.md` と [W1_RESULTS_TABLE.md](W1_RESULTS_TABLE.md) は既存検証結果の入口です。

## `--limit` と `--top-k`

- `locomo`: `--limit N` は全体の問題数の上限です。`--limit 5` は5問です。
- `convomem`: `--limit N` はカテゴリごとの件数です。現キャッシュの5カテゴリでは `--limit 2` は合計10問になります。カテゴリごとの欠損は自動補完せず、キャッシュにあるカテゴリだけを使用します。
- `longmemeval`: `--limit N` はraw LongMemEval-Sの全体上限です。省略時は `/tmp/longmemeval-data/longmemeval_s.json` の500問です。
- `--top-k K` は検索結果と `r_at_k` のKです。各summary JSONと `comparison.csv` の `top_k` に実行値を記録します。

`comparison.csv` はW1の旧10列を、summary JSONの `top_k` から再構築して11列へ移行します。対応するsummaryが一意に見つからない行は `unknown` です。

## データセットと既知の問題

- LoCoMo: `/tmp/locomo-data/locomo10.json`
- ConvoMem: `/tmp/convomem_cache`（既存 `convomem_bench.py` のキャッシュローダ）
- LongMemEval-S raw: `/tmp/longmemeval-data/longmemeval_s.json`（500問、HF `xiaowu0162/longmemeval`）

`benchmarks/longmemeval_bench.py` の既定値および `BENCHMARKS.md` の再現コマンドが参照する `/tmp/longmemeval-data/longmemeval_s_cleaned.json` は、このリポジトリにも `download_datasets.sh` にも存在しません。W2ではraw版を用い、既存ローダのデータ修正（`eac54add` のラベル補正を含む）を適用します。`BENCHMARKS.md` のcleaned版の既存値とは別の測定として扱い、突合しません。

## 再現コマンド

### W1の8本

```bash
.venv/bin/python benchmarks/comparison/run_comparison.py --bench convomem --system rag --limit 5
.venv/bin/python benchmarks/comparison/run_comparison.py --bench convomem --system bm25 --limit 5
.venv/bin/python benchmarks/comparison/run_comparison.py --bench locomo --system rag --limit 5
.venv/bin/python benchmarks/comparison/run_comparison.py --bench locomo --system bm25 --limit 5
.venv/bin/python benchmarks/comparison/run_comparison.py --bench convomem --system rag
.venv/bin/python benchmarks/comparison/run_comparison.py --bench convomem --system bm25
.venv/bin/python benchmarks/comparison/run_comparison.py --bench locomo --system rag
.venv/bin/python benchmarks/comparison/run_comparison.py --bench locomo --system bm25
```

W1のスモーク4本は2分以内を目安とします。ConvoMem/RAGのフル計測は大きいため、時間とキャッシュを確認してから実行してください。

### W2 LongMemEval-S baseline 4本

```bash
.venv/bin/python benchmarks/comparison/run_comparison.py --bench longmemeval --system rag --top-k 5
.venv/bin/python benchmarks/comparison/run_comparison.py --bench longmemeval --system rag --top-k 10
.venv/bin/python benchmarks/comparison/run_comparison.py --bench longmemeval --system bm25 --top-k 5
.venv/bin/python benchmarks/comparison/run_comparison.py --bench longmemeval --system bm25 --top-k 10
```

4本とも `n=500` のraw版フル計測です。RAGはW1と同じ共有embedding engine、BM25は依存なしのOkapi実装です。

### W2 ERINYS raw再測定（比較表へは混在させない）

```bash
.venv/bin/python benchmarks/longmemeval_bench.py /tmp/longmemeval-data/longmemeval_s.json --mode enhanced_v2_boost --top-k 5 --output benchmarks/comparison/results/longmemeval_raw_erinys/top_k_5
.venv/bin/python benchmarks/longmemeval_bench.py /tmp/longmemeval-data/longmemeval_s.json --mode enhanced_v2_boost --top-k 10 --output benchmarks/comparison/results/longmemeval_raw_erinys/top_k_10
```

出力ディレクトリ名に `raw_erinys` を含め、cleaned版の既存 `benchmarks/results/` と区別します。

### W2 Mem0（コストゲート付きスモークのみ）

```bash
.venv/bin/python benchmarks/comparison/run_mem0_smoke.py --bench locomo --limit 5
.venv/bin/python benchmarks/comparison/run_mem0_smoke.py --bench convomem --limit 2
.venv/bin/python benchmarks/comparison/mem0_cost_estimate.py
```

Mem0の実行可能範囲はこの2つのlimitに固定されています。既定LLMはmem0ai 2.0.18のOSS `openai` / `gpt-5-mini`、vector storeはローカルQdrantです。必要なLLM環境変数が無い場合はMemoryを初期化せず `SKIP` summaryだけを書きます。キーをログやファイルへ出力しません。フル計測は別パックかつ明示承認まで禁止です。

`MEM0_COST_ESTIMATE.md` の件数・文書数・トークン概算はsummaryと既存ローダから生成します。USDは、現在のprovider価格を確認したうえで `MEM0_INPUT_USD_PER_1M` と `MEM0_OUTPUT_USD_PER_1M` を設定した場合だけ算出対象にします。

### CSV/table migration and checks

```bash
.venv/bin/python benchmarks/comparison/rebuild_comparison_csv.py
.venv/bin/python benchmarks/comparison/make_table.py
.venv/bin/python -m unittest benchmarks/comparison/test_w2.py
```

## 生成物の命名規則

- 比較ベースライン: `results_{system}_{benchmark}_YYYYMMDD_HHMMSS.jsonl` と `summary_{system}_{benchmark}_YYYYMMDD_HHMMSS.json`
- Mem0 smoke: `results_mem0_{benchmark}_...` と `summary_mem0_{benchmark}_...`
- ERINYS raw: `results/longmemeval_raw_erinys/top_k_{5,10}/` 下の既存 `longmemeval_bench.py` 命名
- 比較表: `results/comparison.csv`

数値は手入力せず、各実行のsummary JSONおよび `make_table.py` の出力から取得してください。
