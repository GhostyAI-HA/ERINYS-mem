"""Shared adapters and result writing for comparison baselines."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from typing import Callable, Protocol, Sequence

ROOT_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT_DIR / "benchmarks"
SRC_DIR = ROOT_DIR / "src"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
COMPARISON_CSV = RESULTS_DIR / "comparison.csv"
LOCOMO_DATA_PATH = Path("/tmp/locomo-data/locomo10.json")
CONVOMEM_CACHE_DIR = Path("/tmp/convomem_cache")
REQUESTED_RAG_MODEL = "BAAI/bge-small-en-v1.5"
CONVOMEM_EVAL_MODE = "_comparison_fixed_results"
CSV_FIELDS = (
    "system",
    "benchmark",
    "mode",
    "n",
    "r_at_k",
    "perfect_pct",
    "latency_p50_ms",
    "latency_p95_ms",
    "ingest_wall_s",
    "retrieval_llm_calls",
)

for path in (SRC_DIR, BENCHMARK_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from erinys_memory.db import embedding_engine
from erinys_memory.embedding import EMBEDDING_DIM


def _load_module(name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


locomo = _load_module("comparison_locomo_bench", BENCHMARK_DIR / "locomo_bench.py")
convomem = _load_module("comparison_convomem_bench", BENCHMARK_DIR / "convomem_bench.py")


@dataclass(frozen=True)
class Document:
    """One retrievable benchmark chunk."""

    doc_id: str
    text: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class RankedDocument:
    """One ranked result returned by a baseline index."""

    document: Document
    score: float


@dataclass(frozen=True)
class QueryCase:
    """One benchmark query plus its isolated corpus."""

    case_id: str
    benchmark: str
    corpus_key: str
    question: str
    answer: str
    category: str
    documents: tuple[Document, ...]
    gold_ids: frozenset[str]
    raw_item: dict[str, object] | None = None


@dataclass(frozen=True)
class CaseResult:
    """Serializable per-query result."""

    case_id: str
    question: str
    answer: str
    category: str
    recall: float
    retrieved_ids: list[str]
    latency_ms: float
    details: dict[str, object]


class SearchIndex(Protocol):
    """Protocol implemented by baseline indexes."""

    def search(self, query: str, top_k: int) -> list[RankedDocument]:
        """Return ranked documents."""


IndexFactory = Callable[[Sequence[Document]], SearchIndex]


class ZeroEmbeddingEngine:
    """Fast scorer stub used only while reusing ConvoMem evaluation."""

    model_name = "comparison-zero-vector-scorer"

    def embed(self, _text: str) -> list[float]:
        return [0.0] * EMBEDDING_DIM

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * EMBEDDING_DIM for _ in texts]


def configure_benchmark_embedding_engine() -> None:
    """Use the same shared embedding engine as the existing benches."""
    active = getattr(embedding_engine, "_engine", None)
    active_name = getattr(active, "model_name", embedding_engine.model_name)
    embedding_engine.model_name = str(active_name)


def active_embedding_model_name() -> str:
    """Return the model configured on the shared embedding engine."""
    return str(embedding_engine.model_name)


def load_cases(benchmark: str, limit: int | None) -> list[QueryCase]:
    """Load benchmark cases through existing benchmark modules."""
    if benchmark == "locomo":
        return load_locomo_cases(limit)
    if benchmark == "convomem":
        return load_convomem_cases(limit)
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def load_locomo_cases(limit: int | None) -> list[QueryCase]:
    """Load LoCoMo cases with existing loader and evidence mapper."""
    data = locomo.load_dataset(str(LOCOMO_DATA_PATH))
    cases: list[QueryCase] = []
    for conv_idx, sample in enumerate(data):
        cases.extend(_locomo_sample_cases(sample, conv_idx, limit, len(cases)))
        if limit is not None and len(cases) >= limit:
            break
    return cases[:limit] if limit is not None else cases


def _locomo_sample_cases(
    sample: dict[str, object],
    conv_idx: int,
    limit: int | None,
    seen: int,
) -> list[QueryCase]:
    conv_id = str(sample.get("sample_id", f"conv-{conv_idx}"))
    sessions = locomo.load_conversation_sessions(sample["conversation"])
    docs = _locomo_documents(conv_id, sessions)
    qa_pairs = sample.get("qa", [])
    cases: list[QueryCase] = []
    for qa_idx, qa in enumerate(qa_pairs):
        case = _locomo_case(conv_id, qa_idx, qa, docs)
        if case is not None:
            cases.append(case)
        if limit is not None and seen + len(cases) >= limit:
            break
    return cases


def _locomo_documents(conv_id: str, sessions: list[dict[str, object]]) -> tuple[Document, ...]:
    return tuple(
        Document(str(item["session_id"]), locomo.session_to_text(item), {
            "conv_id": conv_id,
            "session_id": str(item["session_id"]),
        })
        for item in sessions
    )


def _locomo_case(
    conv_id: str,
    qa_idx: int,
    qa: dict[str, object],
    docs: tuple[Document, ...],
) -> QueryCase | None:
    gold = locomo.evidence_to_session_ids(qa.get("evidence", []))
    if not gold:
        return None
    category = locomo.CATEGORIES.get(qa["category"], f"unknown-{qa['category']}")
    return QueryCase(
        case_id=f"{conv_id}:{qa_idx}",
        benchmark="locomo",
        corpus_key=f"locomo:{conv_id}",
        question=str(qa["question"]),
        answer=str(qa.get("answer", "")),
        category=str(category),
        documents=docs,
        gold_ids=frozenset(str(item) for item in gold),
    )


def load_convomem_cases(limit: int | None) -> list[QueryCase]:
    """Load ConvoMem cases through the cached existing loader."""
    categories = _cached_convomem_categories()
    # `limit` is items PER CATEGORY, matching ../convomem_bench.py --limit
    # semantics (50 -> 250 total across 5 categories). Do not truncate the
    # combined list again: that would keep only the first category's items.
    per_category = limit if limit is not None else 1_000_000
    raw_items = convomem.load_evidence_items(categories, per_category, str(CONVOMEM_CACHE_DIR))
    return [_convomem_case(idx, item) for idx, item in enumerate(raw_items)]


def _cached_convomem_categories() -> list[str]:
    categories = [
        key for key in convomem.CATEGORIES
        if (CONVOMEM_CACHE_DIR / f"{key}_filelist.json").exists()
    ]
    if not categories:
        raise FileNotFoundError(f"No ConvoMem filelists in {CONVOMEM_CACHE_DIR}")
    return categories


def _convomem_case(index: int, item: dict[str, object]) -> QueryCase:
    docs = _convomem_documents(item)
    category = str(item.get("_category_key", item.get("category", "unknown")))
    return QueryCase(
        case_id=f"convomem:{index}",
        benchmark="convomem",
        corpus_key=f"convomem:{index}",
        question=str(item["question"]),
        answer=str(item.get("answer", "")),
        category=category,
        documents=docs,
        gold_ids=frozenset(),
        raw_item=item,
    )


def _convomem_documents(item: dict[str, object]) -> tuple[Document, ...]:
    docs: list[Document] = []
    for conv_idx, conv in enumerate(item.get("conversations", [])):
        messages = conv.get("messages", [])
        docs.extend(_convomem_message_docs(conv_idx, messages, len(docs)))
    return tuple(docs)


def _convomem_message_docs(
    conv_idx: int,
    messages: list[dict[str, object]],
    offset: int,
) -> list[Document]:
    return [
        Document(f"msg_{offset + idx}", str(msg["text"]), {
            "conversation": conv_idx,
            "msg_idx": offset + idx,
            "speaker": str(msg.get("speaker", "unknown")),
        })
        for idx, msg in enumerate(messages)
    ]


def run_baseline(
    system: str,
    benchmark: str,
    mode: str,
    top_k: int,
    cases: Sequence[QueryCase],
    index_factory: IndexFactory,
) -> dict[str, object]:
    """Run one baseline and return serializable output."""
    cache: dict[str, SearchIndex] = {}
    results: list[CaseResult] = []
    ingest_wall = 0.0
    scoring_wall = 0.0
    for case in cases:
        index, ingest_delta = _index_for_case(case, cache, index_factory)
        ingest_wall += ingest_delta
        result, scoring_delta = _run_case(case, index, top_k)
        results.append(result)
        scoring_wall += scoring_delta
    return _build_run(system, benchmark, mode, top_k, results, ingest_wall, scoring_wall)


def _index_for_case(
    case: QueryCase,
    cache: dict[str, SearchIndex],
    index_factory: IndexFactory,
) -> tuple[SearchIndex, float]:
    if case.corpus_key in cache:
        return cache[case.corpus_key], 0.0
    started = time.perf_counter()
    index = index_factory(case.documents)
    cache[case.corpus_key] = index
    return index, time.perf_counter() - started


def _run_case(
    case: QueryCase,
    index: SearchIndex,
    top_k: int,
) -> tuple[CaseResult, float]:
    started = time.perf_counter()
    ranked = index.search(case.question, top_k)
    latency_ms = (time.perf_counter() - started) * 1000.0
    score_started = time.perf_counter()
    recall, details = evaluate_case(case, ranked, top_k)
    scoring_wall = time.perf_counter() - score_started
    return _case_result(case, ranked, recall, latency_ms, details), scoring_wall


def evaluate_case(
    case: QueryCase,
    ranked: Sequence[RankedDocument],
    top_k: int,
) -> tuple[float, dict[str, object]]:
    """Evaluate a baseline result through benchmark-specific code."""
    if case.benchmark == "locomo":
        return _evaluate_locomo(case, ranked, top_k)
    return _evaluate_convomem(case, ranked, top_k)


def _evaluate_locomo(
    case: QueryCase,
    ranked: Sequence[RankedDocument],
    top_k: int,
) -> tuple[float, dict[str, object]]:
    retrieved = [item.document.doc_id for item in ranked]
    recall = locomo.compute_recall_at_k(retrieved, set(case.gold_ids), top_k)
    return recall, {"gold_ids": sorted(case.gold_ids), "retrieved_ids": retrieved}


def _evaluate_convomem(
    case: QueryCase,
    ranked: Sequence[RankedDocument],
    top_k: int,
) -> tuple[float, dict[str, object]]:
    if case.raw_item is None:
        raise ValueError("ConvoMem case is missing raw_item")
    payload = _convomem_result_payload(ranked)
    recall, details = _score_convomem_with_existing(case.raw_item, payload, top_k)
    details["retrieved_ids"] = [item.document.doc_id for item in ranked]
    return recall, details


def _convomem_result_payload(
    ranked: Sequence[RankedDocument],
) -> list[dict[str, object]]:
    return [
        {
            "content": item.document.text,
            "metadata": json.dumps(item.document.metadata, ensure_ascii=False),
            "score": item.score,
        }
        for item in ranked
    ]


def _score_convomem_with_existing(
    raw_item: dict[str, object],
    payload: list[dict[str, object]],
    top_k: int,
) -> tuple[float, dict[str, object]]:
    previous_mode = convomem.SEARCH_FNS.get(CONVOMEM_EVAL_MODE)
    previous_engine = convomem.embedding_engine
    convomem.SEARCH_FNS[CONVOMEM_EVAL_MODE] = _fixed_convomem_search(payload)
    convomem.embedding_engine = ZeroEmbeddingEngine()
    try:
        return convomem.retrieve_for_item(raw_item, top_k=top_k, mode=CONVOMEM_EVAL_MODE)
    finally:
        _restore_convomem_eval(previous_mode, previous_engine)


def _fixed_convomem_search(
    payload: list[dict[str, object]],
) -> Callable[[object, str, int], list[dict[str, object]]]:
    def search(_db: object, _query: str, top_k: int = 10) -> list[dict[str, object]]:
        return payload[:top_k]

    return search


def _restore_convomem_eval(previous_mode: object, previous_engine: object) -> None:
    if previous_mode is None:
        convomem.SEARCH_FNS.pop(CONVOMEM_EVAL_MODE, None)
    else:
        convomem.SEARCH_FNS[CONVOMEM_EVAL_MODE] = previous_mode
    convomem.embedding_engine = previous_engine


def _case_result(
    case: QueryCase,
    ranked: Sequence[RankedDocument],
    recall: float,
    latency_ms: float,
    details: dict[str, object],
) -> CaseResult:
    return CaseResult(
        case_id=case.case_id,
        question=case.question,
        answer=case.answer,
        category=case.category,
        recall=recall,
        retrieved_ids=[item.document.doc_id for item in ranked],
        latency_ms=latency_ms,
        details=details,
    )


def _build_run(
    system: str,
    benchmark: str,
    mode: str,
    top_k: int,
    results: Sequence[CaseResult],
    ingest_wall: float,
    scoring_wall: float,
) -> dict[str, object]:
    recalls = [item.recall for item in results]
    latencies = [item.latency_ms for item in results]
    return {
        "results": [_result_dict(item) for item in results],
        "summary": _summary(system, benchmark, mode, top_k, recalls, latencies, ingest_wall, scoring_wall),
    }


def _result_dict(result: CaseResult) -> dict[str, object]:
    return {
        "case_id": result.case_id,
        "question": result.question,
        "answer": result.answer,
        "category": result.category,
        "recall": result.recall,
        "retrieved_ids": result.retrieved_ids,
        "latency_ms": round(result.latency_ms, 3),
        "details": result.details,
    }


def _summary(
    system: str,
    benchmark: str,
    mode: str,
    top_k: int,
    recalls: Sequence[float],
    latencies: Sequence[float],
    ingest_wall: float,
    scoring_wall: float,
) -> dict[str, object]:
    return {
        "system": system,
        "benchmark": benchmark,
        "mode": mode,
        "top_k": top_k,
        "n": len(recalls),
        "overall": _overall(recalls),
        "latency": _latency(latencies),
        "ingest_wall_s": round(ingest_wall, 3),
        "scoring_wall_s": round(scoring_wall, 3),
        "retrieval_llm_calls": 0,
        "embedding_model": active_embedding_model_name(),
        "requested_rag_model": REQUESTED_RAG_MODEL,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def _overall(recalls: Sequence[float]) -> dict[str, float]:
    total = max(len(recalls), 1)
    return {
        "r_at_k": round(sum(recalls) / total * 100.0, 3),
        "perfect_pct": round(sum(1 for item in recalls if item >= 1.0) / total * 100.0, 3),
        "zero_pct": round(sum(1 for item in recalls if item == 0.0) / total * 100.0, 3),
    }


def _latency(latencies: Sequence[float]) -> dict[str, float]:
    return {
        "p50_ms": round(_percentile(latencies, 50), 3),
        "p95_ms": round(_percentile(latencies, 95), 3),
    }


def _percentile(values: Sequence[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(math.ceil(percentile / 100.0 * len(ordered)) - 1, 0)
    return ordered[min(index, len(ordered) - 1)]


def write_outputs(run: dict[str, object]) -> tuple[Path, Path, Path]:
    """Write JSONL, summary JSON, and comparison CSV row."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = run["summary"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = RESULTS_DIR / _result_filename(summary, timestamp)
    summary_path = RESULTS_DIR / _summary_filename(summary, timestamp)
    _write_jsonl(result_path, run["results"])
    _write_summary(summary_path, summary)
    _append_csv(summary)
    return result_path, summary_path, COMPARISON_CSV


def _result_filename(summary: object, timestamp: str) -> str:
    item = _dict(summary)
    return f"results_{item['system']}_{item['benchmark']}_{timestamp}.jsonl"


def _summary_filename(summary: object, timestamp: str) -> str:
    item = _dict(summary)
    return f"summary_{item['system']}_{item['benchmark']}_{timestamp}.json"


def _write_jsonl(path: Path, rows: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_summary(path: Path, summary: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def _append_csv(summary: object) -> None:
    row = _csv_row(_dict(summary))
    needs_header = not COMPARISON_CSV.exists() or COMPARISON_CSV.stat().st_size == 0
    with COMPARISON_CSV.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def _csv_row(summary: dict[str, object]) -> dict[str, object]:
    overall = _dict(summary["overall"])
    latency = _dict(summary["latency"])
    return {
        "system": summary["system"],
        "benchmark": summary["benchmark"],
        "mode": summary["mode"],
        "n": summary["n"],
        "r_at_k": overall["r_at_k"],
        "perfect_pct": overall["perfect_pct"],
        "latency_p50_ms": latency["p50_ms"],
        "latency_p95_ms": latency["p95_ms"],
        "ingest_wall_s": summary["ingest_wall_s"],
        "retrieval_llm_calls": summary["retrieval_llm_calls"],
    }


def _dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"Expected dict, got {type(value).__name__}")
    return value
