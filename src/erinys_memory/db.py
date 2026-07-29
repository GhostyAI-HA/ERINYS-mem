"""SQLite 初期化と observation/vector の整合操作を扱う。"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from ._sqlite import sqlite3
from .config import ErinysConfig
from .embedding import EMBEDDING_MODEL, EmbeddingEngine, serialize_f32

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
OBS_INSERT_SQL = """
INSERT INTO observations(
    title,
    content,
    type,
    project,
    scope,
    is_anti_pattern,
    is_pattern,
    distillation_level,
    distilled_from,
    valid_from,
    valid_until,
    superseded_by,
    base_strength,
    access_count,
    last_accessed,
    source,
    embedding_model,
    topic_key,
    metadata,
    created_at,
    updated_at,
    session_id
) VALUES (
    :title,
    :content,
    :type,
    :project,
    :scope,
    :is_anti_pattern,
    :is_pattern,
    :distillation_level,
    :distilled_from,
    :valid_from,
    :valid_until,
    :superseded_by,
    :base_strength,
    :access_count,
    :last_accessed,
    :source,
    :embedding_model,
    :topic_key,
    :metadata,
    :created_at,
    :updated_at,
    :session_id
)
"""
UPDATABLE_FIELDS = {
    "title",
    "content",
    "type",
    "project",
    "scope",
    "is_anti_pattern",
    "is_pattern",
    "distillation_level",
    "distilled_from",
    "valid_from",
    "valid_until",
    "superseded_by",
    "base_strength",
    "access_count",
    "last_accessed",
    "source",
    "embedding_model",
    "topic_key",
    "metadata",
    "session_id",
}


class LazyEmbeddingEngine:
    """fastembed を実際に必要になるまで初期化しない薄いproxy。"""

    def __init__(self) -> None:
        self.model_name = EMBEDDING_MODEL
        self._engine: EmbeddingEngine | None = None

    def _get_engine(self) -> EmbeddingEngine:
        if self._engine is None:
            self._engine = EmbeddingEngine(self.model_name)
        return self._engine

    def embed(self, text: str) -> list[float]:
        return self._get_engine().embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._get_engine().embed_batch(texts)


embedding_engine = LazyEmbeddingEngine()


def _adapt_datetime(value: datetime) -> str:
    utc_value = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return utc_value.astimezone(timezone.utc).isoformat(sep=" ")


def _convert_datetime(value: bytes) -> datetime:
    text = value.decode("utf-8")
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _register_datetime_codec() -> None:
    sqlite3.register_adapter(datetime, _adapt_datetime)
    sqlite3.register_converter("DATETIME", _convert_datetime)


def _load_sqlite_vec(db: sqlite3.Connection) -> None:
    import sqlite_vec

    if not hasattr(db, "enable_load_extension"):
        raise RuntimeError(
            "This Python cannot load SQLite extensions, so sqlite-vec (ERINYS's "
            "vector index) is unavailable. Your interpreter was built without "
            "--enable-loadable-sqlite-extensions (common on macOS system Python "
            "and pyenv builds). Fix by either:\n"
            "  • using a Python that supports it — python.org, Homebrew, and "
            "conda builds all do; or\n"
            "  • installing a compatible pysqlite3 so ERINYS can fall back to it "
            "(e.g. `pip install pysqlite3-binary` where wheels exist for your "
            "platform).\n"
            "See docs/LIMITATIONS.md → 'Python requirements' for details."
        )
    db.enable_load_extension(True)
    try:
        load = getattr(sqlite_vec, "load", None)
        if callable(load):
            load(db)
            return
        loadable_path = getattr(sqlite_vec, "loadable_path", None)
        if callable(loadable_path):
            db.load_extension(loadable_path())
            return
        raise RuntimeError("sqlite_vec loader not available")
    finally:
        db.enable_load_extension(False)


def _resolve_db_path(db_path: str) -> str:
    return db_path if db_path == ":memory:" else os.path.expanduser(db_path)


def _metadata_json(value: object) -> object:
    return json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value


def _normalize_observation_payload(obs_payload: Mapping[str, object]) -> dict[str, object]:
    now = datetime.now(timezone.utc)
    return {
        **_observation_identity(obs_payload),
        **_observation_temporal(obs_payload, now),
        "metadata": _metadata_json(obs_payload.get("metadata")),
        "session_id": obs_payload.get("session_id"),
    }


def _observation_identity(payload: Mapping[str, object]) -> dict[str, object]:
    return {
        "title": payload["title"], "content": payload["content"],
        "type": payload.get("type", "manual"), "project": payload.get("project"),
        "scope": payload.get("scope", "project"),
        "is_anti_pattern": payload.get("is_anti_pattern", 0),
        "is_pattern": payload.get("is_pattern", 0),
        "distillation_level": payload.get("distillation_level"),
        "distilled_from": payload.get("distilled_from"),
        "source": payload.get("source", "user"),
        "embedding_model": payload.get("embedding_model", embedding_engine.model_name),
        "topic_key": payload.get("topic_key"),
    }


def _observation_temporal(
    payload: Mapping[str, object],
    now: datetime,
) -> dict[str, object]:
    return {
        "valid_from": payload.get("valid_from", now),
        "valid_until": payload.get("valid_until"),
        "superseded_by": payload.get("superseded_by"),
        "base_strength": payload.get("base_strength", 1.0),
        "access_count": payload.get("access_count", 0),
        "last_accessed": payload.get("last_accessed"),
        "created_at": payload.get("created_at", now),
        "updated_at": payload.get("updated_at", now),
    }


def _normalize_update_fields(fields: Mapping[str, object]) -> dict[str, object]:
    normalized = {key: value for key, value in fields.items() if key in UPDATABLE_FIELDS}
    if "metadata" in normalized:
        normalized["metadata"] = _metadata_json(normalized["metadata"])
    return normalized


def get_db(config: ErinysConfig | None = None) -> sqlite3.Connection:
    config = config or ErinysConfig()
    _register_datetime_codec()
    db_path = _resolve_db_path(config.db_path)
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(
        db_path,
        timeout=5.0,
        detect_types=sqlite3.PARSE_DECLTYPES,
        check_same_thread=False,
    )
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    db.execute("PRAGMA busy_timeout = 5000")
    # WAL + NORMAL: commit 毎の fsync を削減（WAL では NORMAL でも耐久性は実用十分）
    db.execute("PRAGMA journal_mode = WAL")
    db.execute("PRAGMA synchronous = NORMAL")
    db.execute("PRAGMA cache_size = -16000")  # 16MB page cache
    _load_sqlite_vec(db)
    return db


def init_db(config: ErinysConfig | None = None) -> sqlite3.Connection:
    config = config or ErinysConfig()
    db = get_db(config)
    exists = db.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_version'"
    ).fetchone()
    if exists is None:
        db.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    else:
        _migrate(db)
    validate_db_metadata(db, config)
    reconcile_vec_observations(db)
    db.commit()
    return db


def _migrate(db: sqlite3.Connection) -> None:
    """Run incremental schema migrations based on schema_version."""
    current_version = _schema_version(db)
    if current_version < 2:
        _migrate_v2(db)
    if current_version < 3:
        _migrate_v3(db)


def _schema_version(db: sqlite3.Connection) -> int:
    row = db.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
    return int(row["v"]) if row and row["v"] is not None else 1


def _collision_columns(db: sqlite3.Connection) -> set[str]:
    rows = db.execute("PRAGMA table_info(collisions)").fetchall()
    return {str(row[1]) for row in rows}


def _migrate_v2(db: sqlite3.Connection) -> None:
    if "metadata" not in _collision_columns(db):
        db.execute(
            "ALTER TABLE collisions ADD COLUMN metadata TEXT "
            "CHECK(metadata IS NULL OR json_valid(metadata))"
        )
    db.execute("INSERT OR IGNORE INTO schema_version(version) VALUES (2)")
    db.commit()


def _migrate_v3(db: sqlite3.Connection) -> None:
    db.executescript(_V3_MIGRATION_SQL)


_V3_MIGRATION_SQL = """
BEGIN IMMEDIATE;
DROP TABLE IF EXISTS edges_v3;
CREATE TABLE edges_v3 (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_id INTEGER NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
  target_id INTEGER NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
  relation TEXT NOT NULL CHECK(relation IN (
    'relates_to','depends_on','implements','references','similar_to','contains',
    'contradicts','supersedes','distilled_from','causal','entity','temporal',
    'supports','narrows','derived_from','applicable_to','used_in','verified_by',
    'resulted_in'
  )),
  weight REAL NOT NULL DEFAULT 1.0 CHECK(weight >= 0.0 AND weight <= 1.0),
  metadata TEXT CHECK(metadata IS NULL OR json_valid(metadata)),
  created_at DATETIME NOT NULL DEFAULT (datetime('now')),
  UNIQUE(source_id, target_id, relation)
);
INSERT INTO edges_v3 SELECT * FROM edges;
DROP TABLE edges;
ALTER TABLE edges_v3 RENAME TO edges;
CREATE INDEX idx_edges_source ON edges(source_id);
CREATE INDEX idx_edges_target ON edges(target_id);
CREATE INDEX idx_edges_relation ON edges(relation);
CREATE TABLE IF NOT EXISTS backfill_runs (
  run_id TEXT PRIMARY KEY,
  run_key TEXT NOT NULL,
  request_digest TEXT NOT NULL,
  plan_digest TEXT NOT NULL,
  project TEXT NOT NULL,
  source_kind TEXT NOT NULL,
  source_namespace TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN (
    'PLANNED','APPLYING','APPLIED','PARTIAL','FAILED','ROLLED_BACK'
  )),
  principal TEXT NOT NULL,
  counts TEXT CHECK(counts IS NULL OR json_valid(counts)),
  created_at DATETIME NOT NULL DEFAULT (datetime('now')),
  completed_at DATETIME,
  UNIQUE(project, run_key)
);
CREATE TABLE IF NOT EXISTS observation_event_baselines (
  observation_id INTEGER PRIMARY KEY
    REFERENCES observations(id) ON DELETE CASCADE,
  access_count INTEGER NOT NULL,
  last_accessed DATETIME,
  enrolled_at DATETIME NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS observation_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  observation_id INTEGER NOT NULL
    REFERENCES observations(id) ON DELETE CASCADE,
  run_id TEXT NOT NULL REFERENCES backfill_runs(run_id) ON DELETE CASCADE,
  event_type TEXT NOT NULL CHECK(event_type IN (
    'RECORDED','USED','VERIFIED','OUTCOME','SUPERSEDED'
  )),
  event_key TEXT NOT NULL,
  occurred_at DATETIME NOT NULL,
  source_locator TEXT,
  source_hash TEXT NOT NULL,
  metadata TEXT CHECK(metadata IS NULL OR json_valid(metadata)),
  imported_at DATETIME NOT NULL DEFAULT (datetime('now')),
  UNIQUE(observation_id, event_type, event_key)
);
CREATE TABLE IF NOT EXISTS backfill_run_observations (
  run_id TEXT NOT NULL REFERENCES backfill_runs(run_id) ON DELETE CASCADE,
  observation_id INTEGER NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
  disposition TEXT NOT NULL CHECK(disposition IN ('CREATED','MATCHED')),
  before_projection TEXT CHECK(
    before_projection IS NULL OR json_valid(before_projection)
  ),
  after_projection TEXT CHECK(
    after_projection IS NULL OR json_valid(after_projection)
  ),
  PRIMARY KEY(run_id, observation_id)
);
CREATE INDEX IF NOT EXISTS idx_observation_events_observation
  ON observation_events(observation_id);
CREATE INDEX IF NOT EXISTS idx_observation_events_run
  ON observation_events(run_id);
CREATE INDEX IF NOT EXISTS idx_observation_events_type_time
  ON observation_events(event_type, occurred_at);
CREATE INDEX IF NOT EXISTS idx_backfill_run_observations_observation
  ON backfill_run_observations(observation_id);
INSERT OR IGNORE INTO schema_version(version) VALUES (3);
COMMIT;
"""


def validate_db_metadata(db: sqlite3.Connection, config: ErinysConfig) -> None:
    """
    初回起動時は db_metadata を作成し、
    以後は config の embedding_model / embedding_dim と一致確認する。
    """
    row = db.execute(
        "SELECT embedding_model, embedding_dim FROM db_metadata WHERE id = 1"
    ).fetchone()

    if row is None:
        db.execute(
            """
            INSERT INTO db_metadata(id, embedding_model, embedding_dim, updated_at)
            VALUES (1, ?, ?, datetime('now'))
            """,
            [config.embedding_model, config.embedding_dim],
        )
        return

    if (
        row["embedding_model"] != config.embedding_model
        or row["embedding_dim"] != config.embedding_dim
    ):
        raise RuntimeError("embedding model/dim mismatch: DB metadata vs config")


def resolve_session_id(db: sqlite3.Connection, session_id: object) -> object:
    """
    sessions に実在する場合のみ session_id を返し、それ以外は None。
    セッション一括削除後、孤児 session_id を持つ観測が残ることがある。
    その値を新規 INSERT にコピーすると FOREIGN KEY 違反になる。
    """
    if session_id is None:
        return None
    row = db.execute("SELECT 1 FROM sessions WHERE id = ?", [session_id]).fetchone()
    return session_id if row is not None else None


def insert_observation_with_embedding(
    db: sqlite3.Connection,
    obs_payload: Mapping[str, object],
    embedding_blob: bytes,
) -> int:
    """observationとvectorを単一transactionで保存する。"""
    db.execute("BEGIN IMMEDIATE")
    try:
        obs_id = insert_observation_vector_rows(db, obs_payload, embedding_blob)
        db.commit()
        return obs_id
    except Exception:
        db.rollback()
        raise


def insert_observation_vector_rows(
    db: sqlite3.Connection,
    obs_payload: Mapping[str, object],
    embedding_blob: bytes,
) -> int:
    """Caller-owned transactionへobservationとvectorを追加する。"""
    cursor = db.execute(OBS_INSERT_SQL, _normalize_observation_payload(obs_payload))
    if cursor.lastrowid is None:
        raise RuntimeError("observation insert did not return an id")
    obs_id = int(cursor.lastrowid)
    db.execute(
        "INSERT INTO vec_observations(rowid, embedding) VALUES (?, ?)",
        [obs_id, embedding_blob],
    )
    return obs_id


def delete_observation_with_embedding(db: sqlite3.Connection, obs_id: int) -> None:
    """
    observations と vec_observations を単一トランザクションで同時削除。
    edges/collisions は ON DELETE CASCADE で自動削除。
    """
    db.execute("BEGIN IMMEDIATE")
    try:
        db.execute("DELETE FROM vec_observations WHERE rowid = ?", [obs_id])
        db.execute("DELETE FROM observations WHERE id = ?", [obs_id])
        db.commit()
    except Exception:
        db.rollback()
        raise


def reconcile_vec_observations(db: sqlite3.Connection) -> dict[str, int]:
    """
    Startup reconciliation: observations ↔ vec_observations の整合性検査。
    1. orphan vec rows (vec にあるが obs にない) → DELETE
    2. missing vec rows (obs にあるが vec にない) → 再embedding + INSERT
    Returns: {"orphans_removed": int, "missing_reembedded": int}
    """
    orphans = db.execute(
        "SELECT rowid FROM vec_observations WHERE rowid NOT IN (SELECT id FROM observations)"
    ).fetchall()
    for row in orphans:
        db.execute("DELETE FROM vec_observations WHERE rowid = ?", [row[0]])

    missing = db.execute(
        "SELECT id, content FROM observations WHERE id NOT IN (SELECT rowid FROM vec_observations)"
    ).fetchall()
    for row in missing:
        embedding = embedding_engine.embed(row[1])
        db.execute(
            "INSERT INTO vec_observations(rowid, embedding) VALUES (?, ?)",
            [row[0], serialize_f32(embedding)],
        )

    return {"orphans_removed": len(orphans), "missing_reembedded": len(missing)}


def update_observation(
    db: sqlite3.Connection, obs_id: int, fields: dict[str, object]
) -> None:
    """updated_at は UPDATE SET で明示更新する。"""
    normalized = _normalize_update_fields(fields)
    if not normalized:
        return
    if "content" in normalized and "embedding_model" not in normalized:
        normalized["embedding_model"] = embedding_engine.model_name
    assignments = ", ".join(f"{field} = ?" for field in normalized)
    params = list(normalized.values())
    db.execute("BEGIN IMMEDIATE")
    try:
        cursor = db.execute(
            f"UPDATE observations SET {assignments}, updated_at = datetime('now') WHERE id = ?",
            [*params, obs_id],
        )
        if cursor.rowcount == 0:
            raise LookupError(f"observation not found: {obs_id}")
        if "content" in normalized:
            db.execute("DELETE FROM vec_observations WHERE rowid = ?", [obs_id])
            db.execute(
                "INSERT INTO vec_observations(rowid, embedding) VALUES (?, ?)",
                [obs_id, serialize_f32(embedding_engine.embed(str(normalized['content'])))],
            )
        db.commit()
    except Exception:
        db.rollback()
        raise
