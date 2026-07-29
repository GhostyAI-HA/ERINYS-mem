"""実検索候補からCurrent Valid Claim shadowへ接続する境界を検証する。"""

from __future__ import annotations

import json
import logging
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Literal

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from erinys_memory._sqlite import sqlite3
from erinys_memory.claim_shadow_runtime import (
    LiveShadowReceipt,
    build_soak_report,
    observe_shadow_search,
)

REPORT_SCRIPT = (
    Path(__file__).parent.parent / "scripts" / "current_valid_claim_shadow_report.py"
)
SOAK_SCRIPT = (
    Path(__file__).parent.parent / "scripts" / "current_valid_claim_shadow_soak.py"
)
SCHEMA = """
CREATE TABLE observations (
  id INTEGER PRIMARY KEY, title TEXT NOT NULL, content TEXT NOT NULL,
  type TEXT NOT NULL DEFAULT 'manual', project TEXT, scope TEXT DEFAULT 'project',
  valid_from DATETIME NOT NULL, valid_until DATETIME, superseded_by INTEGER,
  source TEXT DEFAULT 'user', metadata TEXT, created_at DATETIME NOT NULL
);
CREATE TABLE edges (
  id INTEGER PRIMARY KEY, source_id INTEGER NOT NULL, target_id INTEGER NOT NULL,
  relation TEXT NOT NULL, weight REAL NOT NULL DEFAULT 1.0,
  metadata TEXT, created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE audit_log (
  id INTEGER PRIMARY KEY, operation TEXT NOT NULL, target_type TEXT,
  target_id INTEGER, detail TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""
LARGE_OBSERVATION_SQL = """
INSERT INTO observations(id, title, content, project, valid_from, source, created_at)
VALUES (?, ?, ?, 'ERINYS', '2026-07-01T00:00:00Z', 'test', '2026-07-01T00:00:00Z')
"""
LARGE_EDGE_SQL = """
INSERT INTO edges(source_id, target_id, relation)
VALUES (1, ?, 'distilled_from')
"""


def _db() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)
    return connection


def _insert_observation(
    db: sqlite3.Connection, obs_id: int, content: str,
    *, valid_until: str | None = None, metadata: dict[str, object] | None = None,
) -> None:
    db.execute(
        """
        INSERT INTO observations(
          id, title, content, project, valid_from, valid_until,
          source, metadata, created_at
        ) VALUES (?, ?, ?, 'ERINYS', '2026-07-01T00:00:00Z', ?, 'test', ?, ?)
        """,
        _observation_values(obs_id, content, valid_until, metadata),
    )
    db.commit()


def _observation_values(
    obs_id: int, content: str, valid_until: str | None,
    metadata: dict[str, object] | None,
) -> list[object]:
    return [
        obs_id, f"Observation {obs_id}", content, valid_until,
        json.dumps(metadata) if metadata else None,
        f"2026-07-{obs_id:02d}T01:00:00Z",
    ]


def _result(db: sqlite3.Connection, obs_id: int, score: float) -> dict[str, object]:
    row = db.execute("SELECT * FROM observations WHERE id = ?", [obs_id]).fetchone()
    assert row is not None
    result = dict(row)
    result["effective_score"] = score
    return result


def _insert_edge(
    db: sqlite3.Connection, source: int, target: int, relation: str,
    metadata: dict[str, object] | None = None,
) -> None:
    db.execute(
        """
        INSERT INTO edges(source_id, target_id, relation, weight, metadata)
        VALUES (?, ?, ?, 0.9, ?)
        """,
        [source, target, relation, json.dumps(metadata) if metadata else None],
    )
    db.commit()


def _enable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ERINYS_CLAIM_SHADOW_MODE", "audit")


def test_disabled_shadow_does_not_touch_database() -> None:
    db = sqlite3.connect(":memory:")
    assert observe_shadow_search(db, "query", "ERINYS", [{"id": 1}]) is None


def test_legacy_supersede_becomes_verified_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _db()
    _insert_observation(db, 1, "Old policy", valid_until="2026-07-02T00:00:00Z")
    _insert_observation(db, 2, "New policy")
    _insert_edge(db, 1, 2, "supersedes")
    _enable(monkeypatch)
    receipt = observe_shadow_search(
        db, "current policy", "ERINYS", [_result(db, 1, 0.2), _result(db, 2, 0.4)],
    )
    assert receipt is not None
    _assert_supersede_receipt(receipt)


def _assert_supersede_receipt(receipt: LiveShadowReceipt) -> None:
    assert receipt.shadow_active_claim_ids == ["observation:2"]
    assert receipt.historical_claim_ids == ["observation:1"]
    assert receipt.selected_path_ids == ["path:edge:1"]
    assert receipt.removed_baseline_ids == ["observation:1"]


def test_unverified_legacy_edge_is_trace_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _db()
    _insert_observation(db, 1, "Cause A")
    _insert_observation(db, 2, "Cause B")
    _insert_edge(db, 1, 2, "contradicts")
    _enable(monkeypatch)
    receipt = observe_shadow_search(
        db, "root cause", "ERINYS", [_result(db, 1, 0.4), _result(db, 2, 0.5)],
    )
    assert receipt is not None
    assert receipt.selected_path_ids == []
    assert receipt.discarded_path_ids == ["path:edge:1"]
    assert receipt.fallback_reason == "no_verified_evidence_path"


def test_unscoped_search_uses_corpus_without_erasing_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _db()
    _insert_observation(db, 1, "Project A")
    _insert_observation(db, 2, "Project B")
    db.execute("UPDATE observations SET project = 'OTHER' WHERE id = 2")
    _enable(monkeypatch)
    receipt = observe_shadow_search(
        db, "all projects", None, [_result(db, 1, 0.5), _result(db, 2, 0.4)],
    )
    assert receipt is not None
    assert set(receipt.shadow_active_claim_ids) == {"observation:1", "observation:2"}


def test_graph_neighbor_cannot_cross_project_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _db()
    _insert_observation(db, 1, "ERINYS policy")
    _insert_observation(db, 2, "Other policy")
    db.execute("UPDATE observations SET project = 'OTHER' WHERE id = 2")
    _insert_edge(db, 1, 2, "supersedes")
    _enable(monkeypatch)
    receipt = observe_shadow_search(db, "policy", "ERINYS", [_result(db, 1, 0.5)])
    assert receipt is not None
    assert receipt.shadow_active_claim_ids == ["observation:1"]
    assert receipt.selected_path_ids == []


def test_explicit_verified_edge_can_enter_conflict_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _db()
    _insert_observation(db, 1, "Cause A")
    _insert_observation(db, 2, "Cause B")
    _insert_edge(db, 1, 2, "contradicts", _verified_edge_contract())
    _enable(monkeypatch)
    receipt = observe_shadow_search(
        db, "root cause", "ERINYS", [_result(db, 1, 0.4), _result(db, 2, 0.5)],
    )
    assert receipt is not None
    assert receipt.selected_path_ids == ["path:edge:1"]
    assert receipt.conflicting_claim_ids == ["observation:1", "observation:2"]


def _verified_edge_contract() -> dict[str, object]:
    return {
        "claim_edge": {
            "relation": "contradicts", "origin": "VERIFIED_DERIVED",
            "status": "VERIFIED", "verification_id": "human-review-1",
        },
    }


def test_receipt_does_not_persist_query_or_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _db()
    _insert_observation(db, 1, "PRIVATE-CONTENT-XYZ")
    _enable(monkeypatch)
    receipt = observe_shadow_search(
        db, "PRIVATE-QUERY-XYZ", "ERINYS", [_result(db, 1, 0.5)],
    )
    assert receipt is not None
    serialized = receipt.model_dump_json()
    assert "PRIVATE-QUERY-XYZ" not in serialized
    assert "PRIVATE-CONTENT-XYZ" not in serialized


def test_empty_result_produces_observable_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable(monkeypatch)
    receipt = observe_shadow_search(_db(), "missing", "ERINYS", [])
    assert receipt is not None
    assert receipt.status == "no_candidates"
    assert receipt.fallback_reason == "no_baseline_candidates"


def test_adapter_failure_is_fail_open(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    import erinys_memory.claim_shadow_runtime as runtime

    _enable(monkeypatch)
    monkeypatch.setattr(runtime, "_adapt_search", lambda *args: 1 / 0)
    with caplog.at_level(logging.WARNING):
        receipt = runtime.observe_shadow_search(_db(), "query", "ERINYS", [{"id": 1}])
    assert receipt is not None
    assert (receipt.status, receipt.error_type) == ("error", "ZeroDivisionError")
    assert "failed open: ZeroDivisionError" in caplog.text


def _insert_large_graph(db: sqlite3.Connection, size: int) -> None:
    observations = [
        (item, f"Observation {item}", f"Content {item}")
        for item in range(1, size + 1)
    ]
    db.executemany(LARGE_OBSERVATION_SQL, observations)
    edges = [(item,) for item in range(2, size + 1)]
    db.executemany(LARGE_EDGE_SQL, edges)
    db.commit()


def test_graph_expansion_enforces_total_node_bound() -> None:
    import erinys_memory.claim_shadow_runtime as runtime

    db = _db()
    _insert_large_graph(db, runtime.MAX_GRAPH_NODES + 50)
    edges = runtime._bounded_edge_rows(db, {1})
    node_ids = {int(row[key]) for row in edges for key in ("source_id", "target_id")}
    assert len(node_ids) == runtime.MAX_GRAPH_NODES


def test_server_records_shadow_receipt_without_changing_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from erinys_memory import server

    receipt = _receipt("evaluated", 2.0)
    audit_calls: list[tuple[object, ...]] = []
    results = [{"id": 1, "content": "unchanged"}]
    _enable(monkeypatch)
    monkeypatch.setattr(server, "_db", lambda: object())
    monkeypatch.setattr(server, "observe_shadow_search", lambda *args: receipt)
    monkeypatch.setattr(server, "_audit", lambda *args: audit_calls.append(args))
    server._record_claim_shadow("query", "ERINYS", results, "context")
    assert results == [{"id": 1, "content": "unchanged"}]
    assert audit_calls[0][0:3] == ("claim_shadow", "retrieval", None)


def test_server_disabled_shadow_skips_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from erinys_memory import server

    monkeypatch.setattr(server, "_db", lambda: 1 / 0)
    server._record_claim_shadow("query", "ERINYS", [{"id": 1}], "context")


def test_server_persists_valid_audit_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from erinys_memory import server

    db = _db()
    _insert_observation(db, 1, "Current policy")
    results = [_result(db, 1, 0.5)]
    _enable(monkeypatch)
    monkeypatch.setattr(server, "_DB", db)
    server._record_claim_shadow("query", "ERINYS", results, "context")
    receipt = _persisted_receipt(db)
    assert receipt.baseline_claim_ids == ["observation:1"]
    assert results == [_result(db, 1, 0.5)]


def _persisted_receipt(db: sqlite3.Connection) -> LiveShadowReceipt:
    row = db.execute(
        "SELECT detail FROM audit_log WHERE operation = 'claim_shadow'",
    ).fetchone()
    assert row is not None
    return LiveShadowReceipt.model_validate_json(str(row["detail"]))


def _receipt(
    status: Literal["evaluated", "no_candidates"], latency: float,
) -> LiveShadowReceipt:
    return LiveShadowReceipt.model_validate({
        "receipt_id": f"receipt-{status}-{latency}", "status": status,
        "recorded_at": "2026-07-27T00:00:00Z", "query_hash": "a" * 64,
        "query_character_count": 1, "intended_use": "context",
        "candidate_count": 0, "edge_count": 0, "latency_ms": latency,
        "fallback_reason": "no_baseline_candidates" if status == "no_candidates" else None,
    })


def test_soak_report_counts_fallback_and_latency() -> None:
    receipts = [_receipt("evaluated", 1.0), _receipt("no_candidates", 9.0)]
    report = build_soak_report(receipts, invalid_receipts=1)
    assert (report.total, report.valid_receipts) == (3, 2)
    assert report.invalid_receipts == 1
    assert report.fallback_count == 1
    assert report.fallback_rate == 0.5
    assert report.p95_latency_ms == 9.0


def _audit_db(db_path: Path) -> None:
    db = sqlite3.connect(db_path)
    db.execute("CREATE TABLE audit_log(id INTEGER PRIMARY KEY, operation TEXT, detail TEXT)")
    db.execute(
        "INSERT INTO audit_log(operation, detail) VALUES ('claim_shadow', ?)",
        [_receipt("evaluated", 2.0).model_dump_json()],
    )
    db.commit()
    db.close()


def test_report_cli_reads_audit_log(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    _audit_db(db_path)
    completed = subprocess.run(
        [sys.executable, str(REPORT_SCRIPT), "--db", str(db_path)],
        check=False, capture_output=True, text=True,
    )
    payload = json.loads(completed.stdout)
    assert completed.returncode == 0
    assert payload["report"]["total"] == 1


def test_soak_cli_refuses_live_database() -> None:
    live_db = Path("~/.erinys/memory.db").expanduser()
    completed = subprocess.run(
        [sys.executable, str(SOAK_SCRIPT), "--db", str(live_db), "--count", "1"],
        check=False, capture_output=True, text=True,
    )
    payload = json.loads(completed.stdout)
    assert completed.returncode == 2
    assert "live ERINYS DB is forbidden" in payload["error"]


def test_soak_database_cases_use_graph_linked_titles(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    _insert_observation(db, 1, "First")
    _insert_observation(db, 2, "Second")
    _insert_edge(db, 1, 2, "distilled_from")
    db.close()
    namespace = runpy.run_path(str(SOAK_SCRIPT), run_name="soak_test")
    cases = namespace["_database_cases"](db_path, 3)
    assert [case.query for case in cases] == ["Observation 2", "Observation 1", "Observation 2"]


def test_soak_score_drift_is_not_a_response_mismatch() -> None:
    namespace = runpy.run_path(str(SOAK_SCRIPT), run_name="soak_score_test")
    scores = iter((1.0, 0.9))

    def search(_case: object) -> list[dict[str, object]]:
        return [{"id": 1, "effective_score": next(scores)}]

    case = namespace["QueryCase"](query="query")
    _, _, mismatch, drift = namespace["_paired_run"](search, case, False)
    assert mismatch is False
    assert drift == pytest.approx(0.1)
