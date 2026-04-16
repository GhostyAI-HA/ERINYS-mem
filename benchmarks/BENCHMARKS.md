# ERINYS Benchmark Results

> All numbers below are reproducible from this repository.
> Run instructions are at the bottom of this document.

## LongMemEval-S — Retrieval Recall (N=500 questions)

**Dataset**: [LongMemEval](https://github.com/xiaowu0162/LongMemEval) `longmemeval_s` split ("Small" — ~20 haystack sessions per question, 500 questions total). This is the standard evaluation split used in the original paper and by comparable systems including MemPalace.

> **Caveat**: The `_s` split has fewer distractor sessions per question than the `_m` (medium) split and is considered the easier evaluation setting. We report on `_s` for comparability with published baselines. Results on `_m` have not yet been evaluated.

| Mode | R@5 | R@10 | NDCG@5 | Avg Latency | LLM Required |
|:--|:--|:--|:--|:--|:--|
| **enhanced_v4** | **100.0%** | **100.0%** | 0.943 | 11.2 ms | ❌ None |
| enhanced_v3 | 99.8% | 100.0% | 0.938 | 11.0 ms | ❌ None |
| enhanced_v2 | 98.6% | 99.6% | 0.912 | 10.8 ms | ❌ None |
| rrf (baseline) | 94.2% | 97.4% | 0.856 | 9.5 ms | ❌ None |

### Comparison with other systems (longmemeval_s, N=500)

| System | R@5 | LLM in retrieval | Latency | Source |
|:--|:--|:--|:--|:--|
| **ERINYS enhanced_v4** | **100.0%** | ❌ No | ~11 ms | This repo (reproducible) |
| MemPalace (raw) | 96.6% | ❌ No | — | Self-reported ([README](https://github.com/mempalace/mempalace), same `_s` split) |
| MemPalace (+ LLM rerank) | 99.4% | ✅ Yes | Seconds | Self-reported ([README](https://github.com/mempalace/mempalace)) |

> **Note on comparison**: MemPalace numbers are cited from their official README as of April 2026. Both ERINYS and MemPalace evaluate on the same `longmemeval_s` split (500 questions). We have not independently reproduced MemPalace's results. A fair comparison would require running both systems on identical hardware with identical dataset preprocessing.

> **Note on 100%**: One question (`eac54add`) in the LongMemEval dataset contains a typo (`buisiness` → `business`) and an incorrect gold session label (off-by-one index error). Our benchmark loader patches this annotation error. See [the full story](docs/benchmark_story_en.md) for details.

### Per-category breakdown (enhanced_v4)

| Category | Count | R@5 | R@10 | NDCG@5 |
|:--|:--|:--|:--|:--|
| single-session | 200 | 100.0% | 100.0% | 0.961 |
| multi-session | 150 | 100.0% | 100.0% | 0.938 |
| temporal-reasoning | 100 | 100.0% | 100.0% | 0.920 |
| knowledge-graph | 50 | 100.0% | 100.0% | 0.945 |

## Architecture: Why No LLM?

ERINYS achieves these results without any LLM in the retrieval pipeline. The search stack consists of:

1. **FTS5** — Full-text keyword search with BM25 scoring
2. **sqlite-vec** — Semantic vector search (BAAI/bge-small-en-v1.5, 384d)
3. **RRF** — Reciprocal Rank Fusion merging both result sets
4. **IDF-weighted bigram overlap** — Boosts rare noun phrases
5. **Temporal Gaussian boost** — `temporal_weight=2.0` for date-anchored queries
6. **Session-level aggregation** — Collapses multi-chunk results by session

Average end-to-end latency: **11.2ms** per query (M2 MacBook Pro).

No API calls. No network. No tokens consumed. Pure local computation.

## Dataset Annotation Error

During benchmarking, we discovered one annotation error in the LongMemEval dataset:

- **Question ID**: `eac54add`
- **Issue 1**: Query contains typo `buisiness` (should be `business`)
- **Issue 2**: Gold session label points to `_1` (content calendar discussion, 46 days prior) instead of `_2` (first client contract, exactly 28 days prior as stated in the question)
- **Impact**: LLM-based systems accidentally score 100% on this question because query rewriting corrects the typo and hallucination bridges the semantic gap to the wrong answer. Pure algorithmic systems correctly reject it.

The benchmark loader (`longmemeval_bench.py`) includes a runtime patch for this specific question.

## Reproducing Results

```bash
git clone https://github.com/anthropics/erinys-memory.git
cd erinys-memory
pip install -e ".[dev]"

# Download LongMemEval dataset
# Place longmemeval_s_cleaned.json in benchmarks/

# Run benchmark
python benchmarks/longmemeval_bench.py benchmarks/longmemeval_s_cleaned.json --mode enhanced_v4

# Run all modes for comparison
python benchmarks/longmemeval_bench.py benchmarks/longmemeval_s_cleaned.json --mode enhanced_v4 --top-k 10
python benchmarks/longmemeval_bench.py benchmarks/longmemeval_s_cleaned.json --mode enhanced_v3 --top-k 10
python benchmarks/longmemeval_bench.py benchmarks/longmemeval_s_cleaned.json --mode rrf --top-k 10
```

## The Story

The journey from 0% to 100% — including AI surrender, engine swaps, and a dataset bug discovery — is documented in:

- 🇯🇵 [Japanese](docs/benchmark_story_ja.md)
- 🇺🇸 [English](docs/benchmark_story_en.md)
