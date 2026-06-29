from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator

import pytest

from erinys_memory.collider import MemoryCollider
from erinys_memory.config import ErinysConfig
import erinys_memory.db as db_module
from erinys_memory.db import init_db, insert_observation_with_embedding
import erinys_memory.distill as distill_module
from erinys_memory.embedding import EmbeddingEngine, serialize_f32
import erinys_memory.server as server
import erinys_memory.session as session_module
import erinys_memory.temporal as temporal_module


@pytest.fixture(scope="session")
def engine() -> EmbeddingEngine:
    return db_module.embedding_engine


@pytest.fixture
def db() -> Iterator[sqlite3.Connection]:
    config = ErinysConfig(db_path=":memory:", db_backup_on_init=False, auto_distill_on_save=False)
    connection = init_db(config)
    try:
        yield connection
    finally:
        connection.close()


# Inline schema for the lightweight `mem_db` fixture: no vec0 dependency, FTS5
# index named `fts_observations` (matches the monorepo harness expectations).
_MEM_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT 'manual',
    project TEXT,
    scope TEXT NOT NULL DEFAULT 'project',
    is_anti_pattern INTEGER NOT NULL DEFAULT 0,
    is_pattern INTEGER NOT NULL DEFAULT 0,
    distillation_level TEXT,
    distilled_from INTEGER REFERENCES observations(id),
    valid_from TIMESTAMP NOT NULL DEFAULT (datetime('now')),
    valid_until TIMESTAMP,
    superseded_by INTEGER REFERENCES observations(id),
    base_strength REAL NOT NULL DEFAULT 1.0,
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TIMESTAMP,
    source TEXT NOT NULL DEFAULT 'user',
    embedding_model TEXT,
    topic_key TEXT,
    metadata TEXT,
    session_id TEXT REFERENCES sessions(id),
    created_at TIMESTAMP NOT NULL DEFAULT (datetime('now')),
    updated_at TIMESTAMP NOT NULL DEFAULT (datetime('now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS fts_observations USING fts5(
    title, content, project,
    content_rowid='id'
);

CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
    target_id INTEGER NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
    relation TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    metadata TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT (datetime('now')),
    UNIQUE(source_id, target_id, relation)
);

CREATE TABLE IF NOT EXISTS collisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_a INTEGER NOT NULL REFERENCES observations(id),
    source_b INTEGER NOT NULL REFERENCES observations(id),
    collision_text TEXT NOT NULL,
    similarity REAL NOT NULL,
    metadata TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    directory TEXT,
    started_at TIMESTAMP NOT NULL DEFAULT (datetime('now')),
    ended_at TIMESTAMP,
    summary TEXT
);

CREATE TABLE IF NOT EXISTS prompts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    project TEXT,
    session_id TEXT REFERENCES sessions(id),
    created_at TIMESTAMP NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation TEXT NOT NULL,
    target_type TEXT,
    target_id INTEGER,
    detail TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT (datetime('now'))
);
"""


@pytest.fixture
def mem_db() -> Iterator[sqlite3.Connection]:
    """In-memory SQLite database with ERINYS schema but no vec0 dependency.

    Used by the monorepo-style tests (test_collider_candidates, test_distill)
    that exercise pure-SQL logic without real embeddings.
    """
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(_MEM_DB_SCHEMA)
    try:
        yield connection
    finally:
        connection.close()


def insert_test_observation(
    db: sqlite3.Connection,
    engine: EmbeddingEngine | None = None,
    title: str = "test",
    content: str = "test content",
    project: str = "test",
    **kwargs: object,
) -> dict[str, object] | int:
    """Insert a test observation, supporting two harness conventions.

    Public harness (real vec0 `db` fixture): pass ``engine`` as the second
    positional argument. Inserts via ``insert_observation_with_embedding`` using
    real embeddings and returns the full row as a dict.

    Monorepo harness (lightweight ``mem_db`` fixture, no vec0): call without
    ``engine``. Performs a plain INSERT plus an ``fts_observations`` index write
    and returns the integer row id.
    """
    if engine is None:
        cursor = db.execute(
            """
            INSERT INTO observations (title, content, type, project, scope)
            VALUES (?, ?, ?, ?, ?)
            """,
            [title, content, kwargs.get("type", "manual"), project, kwargs.get("scope", "project")],
        )
        obs_id = cursor.lastrowid
        db.execute(
            "INSERT INTO fts_observations (rowid, title, content, project) VALUES (?, ?, ?, ?)",
            [obs_id, title, content, project],
        )
        db.commit()
        return obs_id

    payload: dict[str, object] = {
        "title": title,
        "content": content,
        "project": project,
        "type": "manual",
        "scope": "project",
        **kwargs,
    }
    embedding = engine.embed(content)
    obs_id = insert_observation_with_embedding(db, payload, serialize_f32(embedding))
    row = db.execute("SELECT * FROM observations WHERE id = ?", [obs_id]).fetchone()
    assert row is not None, f"observation {obs_id} should exist after insert"
    return dict(row)


@pytest.fixture
def bind_server(
    monkeypatch: pytest.MonkeyPatch,
    engine: EmbeddingEngine,
) -> Callable[[sqlite3.Connection, ErinysConfig | None], sqlite3.Connection]:
    def _bind(
        connection: sqlite3.Connection,
        config: ErinysConfig | None = None,
    ) -> sqlite3.Connection:
        active_config = config or ErinysConfig(db_path=":memory:", db_backup_on_init=False, auto_distill_on_save=False)
        monkeypatch.setattr(db_module, "embedding_engine", engine)
        monkeypatch.setattr(server, "embedding_engine", engine)
        monkeypatch.setattr(session_module, "embedding_engine", engine)
        monkeypatch.setattr(temporal_module, "embedding_engine", engine)
        monkeypatch.setattr(distill_module, "embedding_engine", engine)
        monkeypatch.setattr(server, "_CONFIG", active_config)
        monkeypatch.setattr(server, "_DB", connection)
        monkeypatch.setattr(server, "_COLLIDER", MemoryCollider(active_config))
        return connection

    return _bind


@pytest.fixture(autouse=True)
def _default_server(db: sqlite3.Connection, bind_server: Callable[..., sqlite3.Connection]) -> None:
    bind_server(db)
