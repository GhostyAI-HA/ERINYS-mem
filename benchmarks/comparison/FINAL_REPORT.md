# FINAL REPORT: Baseline Comparison Harness

## Built

- `common.py`: shared benchmark adapters, existing module loading via
  `importlib`, benchmark case construction, existing scorer reuse, run summaries,
  JSONL/JSON/CSV output.
- `baseline_rag.py`: naive cosine top-k over benchmark chunks using the shared
  `erinys_memory.db.embedding_engine`.
- `baseline_bm25.py` and `bm25_impl.py`: pure lexical Okapi BM25 baseline with
  vendored k1=1.5, b=0.75 implementation and no FTS5 or new dependencies.
- `run_comparison.py`: CLI for `--bench {locomo,convomem}`,
  `--system {rag,bm25}`, `--limit N`, and `--top-k K`.
- `make_table.py`: renders Markdown and LaTeX tables from
  `results/comparison.csv` plus existing ERINYS summaries in `../results/`.
- `README.md`: fairness contract, granularity notes, commands, and pip freeze
  instruction.
- `TODO_LONGMEMEVAL.md`: documents the missing cleaned LongMemEval-S path and
  separate acquisition/cleaning step.

## Design Choices

- LoCoMo reuses `../locomo_bench.py` for loading, session text conversion,
  evidence-to-session mapping, and `compute_recall_at_k`.
- ConvoMem reuses `../convomem_bench.py` for cached loading and exact
  `retrieve_for_item` evaluation by temporarily injecting a fixed-result search
  mode. A zero-vector scorer stub is used only inside that existing evaluation
  path so BM25 remains lexical-only and RAG retrieval latencies are not polluted
  by scorer embeddings.
- RAG ranks fixed chunks with cosine similarity only. No FTS, no boosting, no
  RRF, no randomness.
- BM25 ranks fixed chunks with vendored Okapi BM25 only. Ties are deterministic.
- Output files are written only under `benchmarks/comparison/results/`.

## Deviations

- The task requested `BAAI/bge-small-en-v1.5`. In this checkout that exact model
  was not cached; attempting to initialize it caused fastembed to resolve
  HuggingFace and fail under offline DNS. To honor the offline/no-redownload
  premise, RAG uses the same shared `embedding_engine` as the existing benches:
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Each summary
  records both `embedding_model` and `requested_rag_model`.
- The local ConvoMem cache contains filelists for five categories and lacks a
  cached `changing_evidence_filelist.json`. The harness passes only categories
  with cached filelists to the existing loader to avoid network discovery.

## Verification

Syntax check:

```text
$ .venv/bin/python -m py_compile benchmarks/comparison/common.py benchmarks/comparison/baseline_rag.py benchmarks/comparison/baseline_bm25.py benchmarks/comparison/bm25_impl.py benchmarks/comparison/run_comparison.py benchmarks/comparison/make_table.py
```

No output; exit code 0.

## Acceptance Outputs

### 1. ConvoMem RAG

```text
$ .venv/bin/python benchmarks/comparison/run_comparison.py --bench convomem --system rag --limit 5
/Users/fujiyoshi/oss/ERINYS-mem/src/erinys_memory/embedding.py:29: UserWarning: The model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 now uses mean pooling instead of CLS embedding. In order to preserve the previous behaviour, consider either pinning fastembed version to 0.5.1 or using `add_custom_model` functionality.
  self._model: TextEmbeddingModel = TextEmbedding(model_name=model_name)
  User Facts: 5 items loaded
  Assistant Facts: 5 items loaded
  Abstention: 5 items loaded
  Preferences: 5 items loaded
  Implicit Connections: 5 items loaded
{
  "system": "rag",
  "benchmark": "convomem",
  "mode": "session_chunks_cosine_shared_embedding_engine",
  "top_k": 10,
  "n": 5,
  "overall": {
    "r_at_k": 100.0,
    "perfect_pct": 100.0,
    "zero_pct": 0.0
  },
  "latency": {
    "p50_ms": 4.654,
    "p95_ms": 5.744
  },
  "ingest_wall_s": 0.894,
  "scoring_wall_s": 0.049,
  "retrieval_llm_calls": 0,
  "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "requested_rag_model": "BAAI/bge-small-en-v1.5",
  "generated_at": "2026-08-17T03:10:41"
}
results_jsonl=/Users/fujiyoshi/oss/ERINYS-mem/benchmarks/comparison/results/results_rag_convomem_20260817_031041.jsonl
summary_json=/Users/fujiyoshi/oss/ERINYS-mem/benchmarks/comparison/results/summary_rag_convomem_20260817_031041.json
comparison_csv=/Users/fujiyoshi/oss/ERINYS-mem/benchmarks/comparison/results/comparison.csv
```

### 2. ConvoMem BM25

```text
$ .venv/bin/python benchmarks/comparison/run_comparison.py --bench convomem --system bm25 --limit 5
  User Facts: 5 items loaded
  Assistant Facts: 5 items loaded
  Abstention: 5 items loaded
  Preferences: 5 items loaded
  Implicit Connections: 5 items loaded
{
  "system": "bm25",
  "benchmark": "convomem",
  "mode": "okapi_bm25_k1_1.5_b_0.75",
  "top_k": 10,
  "n": 5,
  "overall": {
    "r_at_k": 100.0,
    "perfect_pct": 100.0,
    "zero_pct": 0.0
  },
  "latency": {
    "p50_ms": 0.044,
    "p95_ms": 0.148
  },
  "ingest_wall_s": 0.001,
  "scoring_wall_s": 0.059,
  "retrieval_llm_calls": 0,
  "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "requested_rag_model": "BAAI/bge-small-en-v1.5",
  "generated_at": "2026-08-17T03:10:45"
}
results_jsonl=/Users/fujiyoshi/oss/ERINYS-mem/benchmarks/comparison/results/results_bm25_convomem_20260817_031045.jsonl
summary_json=/Users/fujiyoshi/oss/ERINYS-mem/benchmarks/comparison/results/summary_bm25_convomem_20260817_031045.json
comparison_csv=/Users/fujiyoshi/oss/ERINYS-mem/benchmarks/comparison/results/comparison.csv
```

### 3. LoCoMo RAG

```text
$ .venv/bin/python benchmarks/comparison/run_comparison.py --bench locomo --system rag --limit 5
/Users/fujiyoshi/oss/ERINYS-mem/src/erinys_memory/embedding.py:29: UserWarning: The model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 now uses mean pooling instead of CLS embedding. In order to preserve the previous behaviour, consider either pinning fastembed version to 0.5.1 or using `add_custom_model` functionality.
  self._model: TextEmbeddingModel = TextEmbedding(model_name=model_name)
{
  "system": "rag",
  "benchmark": "locomo",
  "mode": "session_chunks_cosine_shared_embedding_engine",
  "top_k": 10,
  "n": 5,
  "overall": {
    "r_at_k": 80.0,
    "perfect_pct": 80.0,
    "zero_pct": 20.0
  },
  "latency": {
    "p50_ms": 2.19,
    "p95_ms": 2.94
  },
  "ingest_wall_s": 0.58,
  "scoring_wall_s": 0.0,
  "retrieval_llm_calls": 0,
  "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "requested_rag_model": "BAAI/bge-small-en-v1.5",
  "generated_at": "2026-08-17T03:10:49"
}
results_jsonl=/Users/fujiyoshi/oss/ERINYS-mem/benchmarks/comparison/results/results_rag_locomo_20260817_031049.jsonl
summary_json=/Users/fujiyoshi/oss/ERINYS-mem/benchmarks/comparison/results/summary_rag_locomo_20260817_031049.json
comparison_csv=/Users/fujiyoshi/oss/ERINYS-mem/benchmarks/comparison/results/comparison.csv
```

### 4. LoCoMo BM25

```text
$ .venv/bin/python benchmarks/comparison/run_comparison.py --bench locomo --system bm25 --limit 5
{
  "system": "bm25",
  "benchmark": "locomo",
  "mode": "okapi_bm25_k1_1.5_b_0.75",
  "top_k": 10,
  "n": 5,
  "overall": {
    "r_at_k": 60.0,
    "perfect_pct": 60.0,
    "zero_pct": 40.0
  },
  "latency": {
    "p50_ms": 0.029,
    "p95_ms": 0.066
  },
  "ingest_wall_s": 0.003,
  "scoring_wall_s": 0.0,
  "retrieval_llm_calls": 0,
  "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "requested_rag_model": "BAAI/bge-small-en-v1.5",
  "generated_at": "2026-08-17T03:10:52"
}
results_jsonl=/Users/fujiyoshi/oss/ERINYS-mem/benchmarks/comparison/results/results_bm25_locomo_20260817_031052.jsonl
summary_json=/Users/fujiyoshi/oss/ERINYS-mem/benchmarks/comparison/results/summary_bm25_locomo_20260817_031052.json
comparison_csv=/Users/fujiyoshi/oss/ERINYS-mem/benchmarks/comparison/results/comparison.csv
```

### 5. Table Rendering

```text
$ .venv/bin/python benchmarks/comparison/make_table.py
| System | Benchmark | Mode | N | R@k % | Perfect % | p50 ms | p95 ms | Ingest s | LLM calls |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| bm25 | convomem | okapi_bm25_k1_1.5_b_0.75 | 5 | 100.0 | 100.0 | 0.044 | 0.148 | 0.001 | 0 |
| erinys | convomem | enhanced_v2_boost | 250 | 97.600 | 97.200 | - | - | - | 0 |
| rag | convomem | session_chunks_cosine_shared_embedding_engine | 5 | 100.0 | 100.0 | 4.654 | 5.744 | 0.894 | 0 |
| bm25 | locomo | okapi_bm25_k1_1.5_b_0.75 | 5 | 60.0 | 60.0 | 0.029 | 0.066 | 0.003 | 0 |
| erinys | locomo | enhanced_v2_boost | 1982 | 98.083 | 98.083 | 7.300 | - | 11.000 | 0 |
| rag | locomo | session_chunks_cosine_shared_embedding_engine | 5 | 80.0 | 80.0 | 2.19 | 2.94 | 0.58 | 0 |

\begin{tabular}{lllrrrrrrr}
\hline
System & Benchmark & Mode & N & R@k & Perfect & p50 & p95 & Ingest & LLM \\
\hline
bm25 & convomem & okapi\_bm25\_k1\_1.5\_b\_0.75 & 5 & 100.0 & 100.0 & 0.044 & 0.148 & 0.001 & 0 \\
erinys & convomem & enhanced\_v2\_boost & 250 & 97.600 & 97.200 & - & - & - & 0 \\
rag & convomem & session\_chunks\_cosine\_shared\_embedding\_engine & 5 & 100.0 & 100.0 & 4.654 & 5.744 & 0.894 & 0 \\
bm25 & locomo & okapi\_bm25\_k1\_1.5\_b\_0.75 & 5 & 60.0 & 60.0 & 0.029 & 0.066 & 0.003 & 0 \\
erinys & locomo & enhanced\_v2\_boost & 1982 & 98.083 & 98.083 & 7.300 & - & 11.000 & 0 \\
rag & locomo & session\_chunks\_cosine\_shared\_embedding\_engine & 5 & 80.0 & 80.0 & 2.19 & 2.94 & 0.58 & 0 \\
\hline
\end{tabular}
```
