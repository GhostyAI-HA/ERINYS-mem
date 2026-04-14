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
    config = ErinysConfig(db_path=":memory:", db_backup_on_init=False)
    connection = init_db(config)
    try:
        yield connection
    finally:
        connection.close()


def insert_test_observation(
    db: sqlite3.Connection,
    engine: EmbeddingEngine,
    title: str,
    content: str,
    project: str = "test",
    **kwargs: object,
) -> dict[str, object]:
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
        active_config = config or ErinysConfig(db_path=":memory:", db_backup_on_init=False)
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
