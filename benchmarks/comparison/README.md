# Baseline Comparison Harness

This directory contains deterministic baselines for comparing ERINYS against
trivial retrieval systems under the same benchmark adapters.

## Fairness Contract

- Inputs come from the existing benchmark scripts:
  - LoCoMo uses `../locomo_bench.py` for dataset loading, session text
    conversion, evidence-to-session mapping, and recall scoring.
  - ConvoMem uses `../convomem_bench.py` for cached evidence loading and, via a
    temporary injected search mode, its existing `retrieve_for_item` scorer.
- The RAG baseline uses the shared `erinys_memory.db.embedding_engine` used by
  the existing benchmark scripts. The task requested `BAAI/bge-small-en-v1.5`;
  on this checkout that exact fastembed cache was not present, so the harness
  keeps the repo's active shared engine model and records it in each summary.
- BM25 is pure lexical Okapi BM25. It does not use SQLite FTS5, vector search,
  boosting, RRF, or any new package dependency.
- No LLM calls are made by either baseline. `retrieval_llm_calls` is always `0`.
- Runs are deterministic: documents keep loader order and score ties break by
  document id or original document index.
- Results are written only under `benchmarks/comparison/results/`; existing
  ERINYS summaries in `../results/` are read by `make_table.py` but never
  modified.

## Ingestion Granularity

- LoCoMo uses one chunk per conversation session, matching the gold evidence
  granularity scored by the existing LoCoMo harness.
- ConvoMem uses one chunk per message because the existing ConvoMem scorer checks
  whether gold evidence message text appears in retrieved message text.
- ERINYS benchmark scripts may add synthetic preference or context-window
  observations. These baselines intentionally do not use those augmentations.

## Commands

```bash
.venv/bin/python benchmarks/comparison/run_comparison.py --bench convomem --system rag --limit 5
.venv/bin/python benchmarks/comparison/run_comparison.py --bench convomem --system bm25 --limit 5
.venv/bin/python benchmarks/comparison/run_comparison.py --bench locomo --system rag --limit 5
.venv/bin/python benchmarks/comparison/run_comparison.py --bench locomo --system bm25 --limit 5
.venv/bin/python benchmarks/comparison/make_table.py
```

To capture the Python environment used for paper numbers:

```bash
.venv/bin/python -m pip freeze > benchmarks/comparison/results/pip-freeze.txt
```

`make_table.py` is the only supported way to render paper tables from
`results/comparison.csv` and the checked-in ERINYS summary JSON files.
