#!/usr/bin/env python3
"""DBコピー上でbaseline対Current Valid Claim shadowの非介入性を測る。"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable, Sequence
from itertools import cycle, islice
from pathlib import Path
from time import perf_counter
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from erinys_memory._sqlite import sqlite3
from erinys_memory.claim_shadow import load_fixtures
from erinys_memory.claim_shadow_runtime import (
    LiveShadowReceipt,
    ShadowSoakReport,
    build_soak_report,
    observe_shadow_search,
)
from erinys_memory.db import _load_sqlite_vec, embedding_engine
from erinys_memory.search import rrf_hybrid_search

DEFAULT_FIXTURES = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "fixtures"
    / "current_valid_claim_v0.jsonl"
)
DEFAULT_LIVE_DB = Path("~/.erinys/memory.db").expanduser()
SHADOW_MODE_ENV = "ERINYS_CLAIM_SHADOW_MODE"
EMBEDDING_DIM = 384
DUMMY_EMBEDDING = [0.0] * EMBEDDING_DIM
MAX_QUERY_CHARACTERS = 256
SCORE_DRIFT_EPSILON = 1e-12
SHADOW_CANDIDATE_LIMIT = 50
RetrievalMode: TypeAlias = Literal["full_rrf_graph_copy", "fts_graph_copy"]
LINKED_QUERY_SQL = """
SELECT o.title, o.project
FROM observations AS o
JOIN (
  SELECT source_id AS observation_id FROM edges
  UNION
  SELECT target_id AS observation_id FROM edges
) AS linked ON linked.observation_id = o.id
WHERE trim(o.title) <> ''
ORDER BY o.id DESC
LIMIT ?
"""


class CliOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    db: Path
    fixtures: Path
    count: int = Field(gt=0, le=10_000)
    warmup: int = Field(ge=0, le=100)
    query_source: Literal["database", "fixtures"]
    retrieval_mode: RetrievalMode
    pretty: bool


class QueryCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    project: str | None = None


class SoakExecutionReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_count: int = Field(ge=0)
    retrieval_mode: RetrievalMode
    query_source: Literal["database", "fixtures"]
    warmup_count: int = Field(ge=0)
    response_mismatches: int = Field(ge=0)
    score_drift_pairs: int = Field(ge=0)
    max_effective_score_delta: float = Field(ge=0.0)
    baseline_p50_ms: float = Field(ge=0.0)
    baseline_p95_ms: float = Field(ge=0.0)
    shadow_p50_ms: float = Field(ge=0.0)
    shadow_p95_ms: float = Field(ge=0.0)
    delta_p50_ms: float
    delta_p95_ms: float
    receipts: ShadowSoakReport


def parse_args() -> CliOptions:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--query-source", choices=("database", "fixtures"), default="database")
    parser.add_argument(
        "--retrieval-mode",
        choices=("full_rrf_graph_copy", "fts_graph_copy"),
        default="full_rrf_graph_copy",
    )
    parser.add_argument("--pretty", action="store_true")
    return CliOptions.model_validate(vars(parser.parse_args()))


def validate_copy_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == DEFAULT_LIVE_DB.resolve():
        raise ValueError("live ERINYS DB is forbidden; pass a SQLite backup copy")
    if not resolved.is_file():
        raise FileNotFoundError(f"DB copy not found: {resolved}")
    return resolved


def _fixture_cases(path: Path, count: int) -> list[QueryCase]:
    fixtures = load_fixtures(path)
    cases = [
        QueryCase(query=item.request.query, project=item.request.scope.project)
        for item in fixtures
    ]
    if not cases:
        raise ValueError("fixture set must not be empty")
    return list(islice(cycle(cases), count))


def _database_cases(path: Path, count: int) -> list[QueryCase]:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(LINKED_QUERY_SQL, [count]).fetchall()
    finally:
        connection.close()
    cases = [_case_from_row(row) for row in rows]
    if not cases:
        raise ValueError("DB copy has no graph-linked observations")
    return list(islice(cycle(cases), count))


def _case_from_row(row: sqlite3.Row) -> QueryCase:
    query = str(row["title"]).strip()[:MAX_QUERY_CHARACTERS]
    project = row["project"] if isinstance(row["project"], str) else None
    return QueryCase(query=query, project=project)


def _query_cases(options: CliOptions, db_path: Path) -> list[QueryCase]:
    if options.query_source == "database":
        return _database_cases(db_path, options.count)
    return _fixture_cases(options.fixtures, options.count)


def _signature(
    results: Sequence[dict[str, object]],
) -> tuple[list[int], dict[int, float]]:
    ids = [int(str(row["id"])) for row in results]
    scores = {
        int(str(row["id"])): float(str(row.get("effective_score", 0.0)))
        for row in results
    }
    return ids, scores


def _score_delta(
    left: dict[int, float], right: dict[int, float],
) -> float:
    shared = set(left) & set(right)
    return max((abs(left[key] - right[key]) for key in shared), default=0.0)


def _persist_receipt(
    db: sqlite3.Connection, receipt: LiveShadowReceipt | None,
) -> None:
    if receipt is None:
        return
    detail = json.dumps(receipt.model_dump(mode="json"), ensure_ascii=False)
    db.execute(
        "INSERT INTO audit_log(operation, target_type, detail) "
        "VALUES ('claim_shadow', 'retrieval', ?)",
        [detail],
    )
    db.commit()


def _observe_and_persist(
    db: sqlite3.Connection, case: QueryCase, results: list[dict[str, object]],
) -> list[dict[str, object]]:
    receipt = observe_shadow_search(
        db, case.query, case.project, results, "context",
    )
    _persist_receipt(db, receipt)
    return results


class FtsGraphCopySearch:
    def __init__(self, db: sqlite3.Connection) -> None:
        self.db = db

    def __call__(self, case: QueryCase) -> list[dict[str, object]]:
        results = rrf_hybrid_search(
            self.db, case.query, DUMMY_EMBEDDING,
            project=case.project, limit=SHADOW_CANDIDATE_LIMIT, k=60,
        )
        return _observe_and_persist(self.db, case, results)


class FullRrfGraphCopySearch:
    def __init__(self, db: sqlite3.Connection) -> None:
        self.db = db

    def __call__(self, case: QueryCase) -> list[dict[str, object]]:
        embedding = embedding_engine.embed(case.query)
        results = rrf_hybrid_search(
            self.db, case.query, embedding,
            project=case.project, limit=SHADOW_CANDIDATE_LIMIT, k=60,
        )
        return _observe_and_persist(self.db, case, results)


def _search_adapter(
    db: sqlite3.Connection, mode: RetrievalMode,
) -> Callable[[QueryCase], list[dict[str, object]]]:
    if mode == "fts_graph_copy":
        return FtsGraphCopySearch(db)
    try:
        _load_sqlite_vec(db)
    except AttributeError as exc:
        raise RuntimeError(
            "full RRF requires extension-capable SQLite; run with PYTHONPATH=src",
        ) from exc
    return FullRrfGraphCopySearch(db)


def _timed_search(
    search: Callable[[QueryCase], list[dict[str, object]]],
    case: QueryCase, mode: str,
) -> tuple[tuple[list[int], dict[int, float]], float]:
    os.environ[SHADOW_MODE_ENV] = mode
    started = perf_counter()
    results = search(case)
    return _signature(results), (perf_counter() - started) * 1000.0


def _paired_run(
    search: Callable[[QueryCase], list[dict[str, object]]],
    case: QueryCase, audit_first: bool,
) -> tuple[float, float, bool, float]:
    modes = ("audit", "off") if audit_first else ("off", "audit")
    first, first_ms = _timed_search(search, case, modes[0])
    second, second_ms = _timed_search(search, case, modes[1])
    baseline_ms, shadow_ms = (second_ms, first_ms) if audit_first else (first_ms, second_ms)
    baseline, shadow = (second, first) if audit_first else (first, second)
    return baseline_ms, shadow_ms, baseline[0] != shadow[0], _score_delta(baseline[1], shadow[1])


def _percentile(values: Sequence[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * ratio), len(ordered) - 1)
    return ordered[index]


def _receipt_rows(db_path: Path, count: int) -> list[LiveShadowReceipt]:
    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(
            "SELECT detail FROM audit_log WHERE operation='claim_shadow' "
            "ORDER BY id DESC LIMIT ?",
            [count],
        ).fetchall()
    finally:
        connection.close()
    return [LiveShadowReceipt.model_validate_json(str(row[0])) for row in rows]


def _warm_up(
    search: Callable[[QueryCase], list[dict[str, object]]],
    cases: Sequence[QueryCase],
) -> None:
    for case in cases:
        _timed_search(search, case, "off")


def _execute_pairs(
    search: Callable[[QueryCase], list[dict[str, object]]],
    cases: Sequence[QueryCase],
) -> tuple[list[float], list[float], int, int, float]:
    baseline: list[float] = []
    shadow: list[float] = []
    mismatches, drift_pairs, max_drift = 0, 0, 0.0
    for index, case in enumerate(cases):
        baseline_ms, shadow_ms, mismatch, drift = _paired_run(search, case, index % 2 == 1)
        baseline.append(baseline_ms)
        shadow.append(shadow_ms)
        mismatches += int(mismatch)
        drift_pairs += int(drift > SCORE_DRIFT_EPSILON)
        max_drift = max(max_drift, drift)
    return baseline, shadow, mismatches, drift_pairs, max_drift


def _report_identity(options: CliOptions, mismatches: int) -> dict[str, int | str]:
    return {
        "query_count": options.count, "retrieval_mode": options.retrieval_mode,
        "query_source": options.query_source,
        "warmup_count": options.warmup,
        "response_mismatches": mismatches,
    }


def _latency_stats(
    baseline: list[float], shadow: list[float],
) -> dict[str, float]:
    deltas = [right - left for left, right in zip(baseline, shadow)]
    return {
        "baseline_p50_ms": _percentile(baseline, 0.50),
        "baseline_p95_ms": _percentile(baseline, 0.95),
        "shadow_p50_ms": _percentile(shadow, 0.50),
        "shadow_p95_ms": _percentile(shadow, 0.95),
        "delta_p50_ms": _percentile(deltas, 0.50),
        "delta_p95_ms": _percentile(deltas, 0.95),
    }


def _report(
    options: CliOptions, baseline: list[float], shadow: list[float],
    mismatches: int, drift_pairs: int, max_drift: float,
    receipts: ShadowSoakReport,
) -> SoakExecutionReport:
    return SoakExecutionReport(
        **_report_identity(options, mismatches),
        score_drift_pairs=drift_pairs, max_effective_score_delta=max_drift,
        **_latency_stats(baseline, shadow), receipts=receipts,
    )


def _execute_on_copy(
    db_path: Path, cases: list[QueryCase], warmups: list[QueryCase],
    mode: RetrievalMode,
) -> tuple[list[float], list[float], int, int, float]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        logging.getLogger("erinys_memory.search").setLevel(logging.ERROR)
        search = _search_adapter(connection, mode)
        _warm_up(search, warmups)
        return _execute_pairs(search, cases)
    finally:
        connection.close()


def run(options: CliOptions) -> SoakExecutionReport:
    db_path = validate_copy_path(options.db)
    cases = _query_cases(options, db_path)
    warmups = list(islice(cycle(cases), options.warmup))
    result = _execute_on_copy(db_path, cases, warmups, options.retrieval_mode)
    baseline, shadow, mismatches, drift_pairs, max_drift = result
    receipts = build_soak_report(_receipt_rows(db_path, options.count))
    return _report(
        options, baseline, shadow, mismatches, drift_pairs, max_drift, receipts,
    )


def main() -> int:
    options = parse_args()
    try:
        report = run(options)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 2
    indent = 2 if options.pretty else None
    print(report.model_dump_json(indent=indent))
    return int(report.response_mismatches > 0 or report.receipts.error_count > 0)


if __name__ == "__main__":
    sys.exit(main())
