from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import Thread
from time import monotonic, sleep

import pytest

from conftest import insert_test_observation
from erinys_memory.config import ErinysConfig
from erinys_memory.db import (
    get_db,
    init_db,
    insert_observation_with_embedding,
    reconcile_vec_observations,
    validate_db_metadata,
)
from erinys_memory.embedding import serialize_f32
from erinys_memory.search import rrf_hybrid_search, sanitize_fts
import erinys_memory.server as server


class RecordingConnectionProxy:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self.vec_limits: list[int] = []

    def execute(self, sql: str, params: object = ()) -> sqlite3.Cursor:
        if "WHERE embedding MATCH ?" in sql and "AND k = ?" in sql:
            self.vec_limits.append(int(params[1]))
        return self.connection.execute(sql, params)

    def executemany(self, sql: str, seq_of_params: object) -> sqlite3.Cursor:
        return self.connection.executemany(sql, seq_of_params)

    def __getattr__(self, name: str) -> object:
        return getattr(self.connection, name)


def _file_db(tmp_path: Path) -> tuple[ErinysConfig, sqlite3.Connection]:
    config = ErinysConfig(db_path=str(tmp_path / "erinys.db"), db_backup_on_init=False, auto_distill_on_save=False)
    return config, init_db(config)


def test_vec_rowid_matches_observation_id(db, engine) -> None:
    record = insert_test_observation(db, engine, "Vector Match", "vector rowid test")
    rowid = db.execute("SELECT rowid FROM vec_observations").fetchone()[0]
    assert rowid == record["id"], "vec_observations.rowid should match observations.id exactly"


def test_vec_observation_drift_reconciliation_reports_or_repairs(db, engine) -> None:
    record = insert_test_observation(db, engine, "Repair", "reconcile missing vector")
    db.execute("DELETE FROM vec_observations WHERE rowid = ?", [record["id"]])
    db.execute(
        "INSERT INTO vec_observations(rowid, embedding) VALUES (?, ?)",
        [9999, serialize_f32(engine.embed("orphan vector"))],
    )
    repaired = reconcile_vec_observations(db)
    orphan = db.execute("SELECT COUNT(*) FROM vec_observations WHERE rowid = 9999").fetchone()[0]
    restored = db.execute("SELECT COUNT(*) FROM vec_observations WHERE rowid = ?", [record["id"]]).fetchone()[0]
    assert repaired == {"orphans_removed": 1, "missing_reembedded": 1}, "reconciliation should report one orphan removal and one re-embedding"
    assert orphan == 0, "reconciliation should remove orphan vector rows"
    assert restored == 1, "reconciliation should recreate missing vector rows"


def test_topic_key_upsert_unique_per_project_scope(db, engine) -> None:
    payload = {"title": "One", "content": "same topic", "project": "alpha", "scope": "project", "topic_key": "topic"}
    insert_observation_with_embedding(db, payload, serialize_f32(engine.embed("same topic")))
    with pytest.raises(sqlite3.IntegrityError):
        insert_observation_with_embedding(db, payload, serialize_f32(engine.embed("same topic again")))


def test_topic_key_null_allows_multiple_rows(db, engine) -> None:
    insert_test_observation(db, engine, "First", "first null topic", project="alpha")
    insert_test_observation(db, engine, "Second", "second null topic", project="alpha")
    count = db.execute("SELECT COUNT(*) FROM observations WHERE project = 'alpha'").fetchone()[0]
    assert count == 2, "multiple rows with topic_key=NULL should be allowed by the partial unique index"


def test_invalid_json_observation_metadata_rejected(db, engine) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO observations(title, content, metadata, embedding_model) VALUES (?, ?, ?, ?)",
            ["Bad JSON", "broken metadata", "{bad-json}", engine.model_name],
        )


def test_invalid_json_sessions_edges_audit_rejected(db, engine) -> None:
    left = insert_test_observation(db, engine, "Left", "left edge node")
    right = insert_test_observation(db, engine, "Right", "right edge node")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO sessions(id, project, metadata) VALUES (?, ?, ?)", ["bad-session", "alpha", "{bad-json}"])
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO edges(source_id, target_id, relation, metadata) VALUES (?, ?, ?, ?)",
            [left["id"], right["id"], "relates_to", "{bad-json}"],
        )
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO audit_log(operation, detail) VALUES (?, ?)", ["save", "{bad-json}"])


def test_empty_fts_query_rejected(db, engine) -> None:
    with pytest.raises(ValueError):
        sanitize_fts("")
    with pytest.raises(ValueError):
        sanitize_fts('   ""   ')


def test_sanitize_fts_strips_double_quotes(db, engine) -> None:
    sanitized = sanitize_fts('alpha "be"ta')
    assert sanitized == '"alpha" "beta"', "sanitize_fts should strip embedded double quotes from each token"


def test_project_filter_adaptive_widening(db, engine) -> None:
    query = "Handle transient API failures with exponential backoff and retries."
    for index in range(6):
        insert_test_observation(db, engine, f"Out {index}", query, project="other")
    insert_test_observation(db, engine, "Target", "Tune SQLite busy_timeout to reduce lock contention during writes.", project="target")
    proxy = RecordingConnectionProxy(db)
    results = rrf_hybrid_search(proxy, query, engine.embed(query), project="target", limit=1)
    assert proxy.vec_limits == [5, 10], "project-filtered vector search should widen from 5 to 10 when initial candidates miss the target project"
    assert results[0]["project"] == "target", "adaptive widening should return an in-project result once widening succeeds"


def test_project_filter_vec_exhaustion_stops_widening(db, engine) -> None:
    query = "Handle transient API failures with exponential backoff and retries."
    for index in range(6):
        insert_test_observation(db, engine, f"Out {index}", query, project="other")
    insert_test_observation(db, engine, "Target 1", "Tune SQLite busy_timeout to reduce lock contention during writes.", project="target")
    insert_test_observation(db, engine, "Target 2", "Increase SQLite busy timeout to avoid lock contention for writers.", project="target")
    proxy = RecordingConnectionProxy(db)
    results = rrf_hybrid_search(proxy, query, engine.embed(query), project="target", limit=3)
    assert proxy.vec_limits == [15], "widening should stop when sqlite-vec returns fewer rows than requested"
    assert len(results) == 2, "exhausted vector search should return every available in-project candidate without looping forever"


def test_vec_max_k_caps_initial_fetch(db, engine) -> None:
    insert_test_observation(db, engine, "Cap", "common cap token", project="alpha")
    proxy = RecordingConnectionProxy(db)
    rrf_hybrid_search(proxy, "common", engine.embed("common"), limit=5000)
    assert proxy.vec_limits[0] == 2500, "current implementation should clamp the search limit before the initial vec fetch"
    assert max(proxy.vec_limits) <= 10000, "vector fetch size should never exceed VEC_MAX_K"


def test_large_limit_clamped_to_max_search_limit(db, engine) -> None:
    for index in range(3):
        server.erinys_save(title=f"Clamp {index}", content="clamp token", project="alpha")
    result = server.erinys_search(query="clamp", project="alpha", limit=5000)
    assert server._validate_limit(5000) == 500, "search limit validation should clamp oversized limits to MAX_SEARCH_LIMIT"
    assert len(result["data"]["results"]) == 3, "clamped large-limit searches should still return available results without sqlite bind errors"


def test_metadata_filter_json_extract_single_key(db, engine) -> None:
    server.erinys_save(title="Match", content="metadata token", project="alpha", metadata={"mode": "D2"})
    server.erinys_save(title="Miss", content="metadata token", project="alpha", metadata={"mode": "D1"})
    result = server.erinys_search(query="metadata", project="alpha", metadata_filter={"mode": "D2"})
    titles = [row["title"] for row in result["data"]["results"]]
    assert titles == ["Match"], "single-key metadata_filter should match rows through json_extract"


def test_metadata_filter_json_extract_multi_key(db, engine) -> None:
    server.erinys_save(title="Both", content="risk token", project="alpha", metadata={"mode": "D2", "risk": "L"})
    server.erinys_save(title="One", content="risk token", project="alpha", metadata={"mode": "D2", "risk": "H"})
    result = server.erinys_search(
        query="risk",
        project="alpha",
        metadata_filter={"mode": "D2", "risk": "L"},
    )
    titles = [row["title"] for row in result["data"]["results"]]
    assert titles == ["Both"], "multi-key metadata_filter should apply every json_extract predicate with AND semantics"


def test_embedding_dim_mismatch_fails_startup(db, engine) -> None:
    with pytest.raises(RuntimeError):
        validate_db_metadata(db, ErinysConfig(db_path=":memory:", embedding_dim=999))


def test_embedding_model_mismatch_fails_startup(db, engine) -> None:
    with pytest.raises(RuntimeError):
        validate_db_metadata(db, ErinysConfig(db_path=":memory:", embedding_model="other/model"))


def test_concurrent_read_write_wal_consistency(tmp_path, bind_server, engine) -> None:
    config, primary = _file_db(tmp_path)
    reader = get_db(config)
    writer = get_db(config)
    try:
        bind_server(primary, config)
        insert_observation_with_embedding(
            writer,
            {"title": "Before", "content": "initial row", "project": "alpha", "scope": "project"},
            serialize_f32(engine.embed("initial row")),
        )
        reader.execute("BEGIN")
        before = reader.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        thread = Thread(
            target=insert_observation_with_embedding,
            args=(
                writer,
                {"title": "After", "content": "written later", "project": "alpha", "scope": "project"},
                serialize_f32(engine.embed("written later")),
            ),
        )
        thread.start()
        thread.join()
        during = reader.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        reader.commit()
        after = reader.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        assert before == 1, "reader snapshot should see the initial committed row"
        assert during == 1, "reader transaction should keep a stable snapshot while a writer commits in WAL mode"
        assert after == 2, "reader should observe the writer's commit after ending its snapshot transaction"
    finally:
        reader.close()
        writer.close()
        primary.close()


def test_begin_immediate_serializes_writers(tmp_path, bind_server, engine) -> None:
    config, primary = _file_db(tmp_path)
    first_writer = get_db(config)
    second_writer = get_db(config)
    result: dict[str, object] = {}
    try:
        bind_server(primary, config)
        first_writer.execute("BEGIN IMMEDIATE")
        started = monotonic()

        def _write_later() -> None:
            insert_observation_with_embedding(
                second_writer,
                {"title": "Serialized", "content": "writer waits", "project": "alpha", "scope": "project"},
                serialize_f32(engine.embed("writer waits")),
            )
            result["elapsed"] = monotonic() - started

        thread = Thread(target=_write_later)
        thread.start()
        sleep(0.4)
        first_writer.commit()
        thread.join()
        count = second_writer.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        assert result["elapsed"] >= 0.4, "second writer should wait for the first BEGIN IMMEDIATE transaction to release the write lock"
        assert count == 1, "serialized writers should still commit successfully once the first writer finishes"
    finally:
        first_writer.close()
        second_writer.close()
        primary.close()


def test_backup_restore_roundtrip(tmp_path, bind_server, engine) -> None:
    config, primary = _file_db(tmp_path)
    backup_path = tmp_path / "roundtrip.db"
    try:
        bind_server(primary, config)
        server.erinys_save(title="Roundtrip", content="backup roundtrip payload", project="alpha")
        server.erinys_save_prompt(content="roundtrip prompt", project="alpha")
        backup = server.erinys_backup(path=str(backup_path))
        restored = get_db(ErinysConfig(db_path=str(backup_path), db_backup_on_init=False))
        try:
            obs_count = restored.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
            prompt_count = restored.execute("SELECT COUNT(*) FROM prompts").fetchone()[0]
            assert backup["ok"] is True, "backup roundtrip should start with a successful backup envelope"
            assert obs_count == 1, "restored backup should preserve observation rows"
            assert prompt_count == 1, "restored backup should preserve prompt rows"
        finally:
            restored.close()
    finally:
        primary.close()


def test_migration_rollback_on_failure(db, engine) -> None:
    before = db.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    try:
        db.execute("BEGIN IMMEDIATE")
        db.execute(
            "INSERT INTO observations(title, content, project, scope, embedding_model) VALUES (?, ?, ?, ?, ?)",
            ["Transient", "should rollback", "alpha", "project", engine.model_name],
        )
        db.execute(
            "INSERT INTO edges(source_id, target_id, relation) VALUES (?, ?, ?)",
            [999, 1000, "relates_to"],
        )
        db.commit()
    except sqlite3.IntegrityError:
        db.rollback()
    after = db.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    assert after == before, "failed migration-style multi-step transactions should roll back every partial change"


def test_erinys_backup_envelope(db, engine, tmp_path) -> None:
    server.erinys_save(title="Envelope", content="backup envelope", project="alpha")
    result = server.erinys_backup(path=str(tmp_path / "envelope.db"))
    assert set(result) == {"ok", "data", "error"}, "erinys_backup should always return the unified envelope keys"
    assert result["ok"] is True, "backup envelope should mark successful backups with ok=True"


def test_erinys_delete_cascades_and_envelope(db, engine) -> None:
    left = server.erinys_save(title="Delete A", content="delete shared", project="alpha", topic_key="delete")
    right = server.erinys_save(title="Delete B", content="delete shared", project="beta", topic_key="delete")
    server.erinys_link(left["data"]["id"], right["data"]["id"], "relates_to")
    server.erinys_collide(left["data"]["id"], right["data"]["id"])
    deleted = server.erinys_delete(left["data"]["id"])
    assert set(deleted) == {"ok", "data", "error"}, "erinys_delete should return the unified envelope keys"
    assert deleted["data"]["cascaded"] == {"edges": 1, "collisions": 1}, "erinys_delete should report the rows that will cascade-delete"


def test_restore_recovers_db_metadata_and_vectors(tmp_path, bind_server, engine) -> None:
    config, primary = _file_db(tmp_path)
    backup_path = tmp_path / "restore.db"
    try:
        bind_server(primary, config)
        saved = server.erinys_save(title="Restored", content="vector survives restore", project="alpha")
        server.erinys_backup(path=str(backup_path))
        restored = get_db(ErinysConfig(db_path=str(backup_path), db_backup_on_init=False))
        try:
            validate_db_metadata(restored, ErinysConfig(db_path=str(backup_path), db_backup_on_init=False))
            results = rrf_hybrid_search(restored, "survives", engine.embed("survives"), limit=1)
            assert results[0]["id"] == saved["data"]["id"], "restored database should keep vector search aligned with observations"
            missing = restored.execute(
                "SELECT COUNT(*) FROM observations WHERE id NOT IN (SELECT rowid FROM vec_observations)"
            ).fetchone()[0]
            assert missing == 0, "restored database should not lose vector rows referenced by observations"
        finally:
            restored.close()
    finally:
        primary.close()
