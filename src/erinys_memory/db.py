"""SQLite 初期化と observation/vector の整合操作を扱う。"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
import sqlite3

from .config import ErinysConfig
from .embedding import EmbeddingEngine, serialize_f32

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
embedding_engine = EmbeddingEngine()
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
    return {
        "title": obs_payload["title"],
        "content": obs_payload["content"],
        "type": obs_payload.get("type", "manual"),
        "project": obs_payload.get("project"),
        "scope": obs_payload.get("scope", "project"),
        "is_anti_pattern": obs_payload.get("is_anti_pattern", 0),
        "is_pattern": obs_payload.get("is_pattern", 0),
        "distillation_level": obs_payload.get("distillation_level"),
        "distilled_from": obs_payload.get("distilled_from"),
        "valid_from": obs_payload.get("valid_from", datetime.now(timezone.utc)),
        "valid_until": obs_payload.get("valid_until"),
        "superseded_by": obs_payload.get("superseded_by"),
        "base_strength": obs_payload.get("base_strength", 1.0),
        "access_count": obs_payload.get("access_count", 0),
        "last_accessed": obs_payload.get("last_accessed"),
        "source": obs_payload.get("source", "user"),
        "embedding_model": obs_payload.get("embedding_model", embedding_engine.model_name),
        "topic_key": obs_payload.get("topic_key"),
        "metadata": _metadata_json(obs_payload.get("metadata")),
        "session_id": obs_payload.get("session_id"),
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
    validate_db_metadata(db, config)
    reconcile_vec_observations(db)
    db.commit()
    return db


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


def insert_observation_with_embedding(
    db: sqlite3.Connection,
    obs_payload: Mapping[str, object],
    embedding_blob: bytes,
) -> int:
    """
    observations と vec_observations は rowid を明示一致させる。
    drift 防止のため、単一トランザクションで両方を書き込む。
    """
    db.execute("BEGIN IMMEDIATE")
    try:
        cursor = db.execute(OBS_INSERT_SQL, _normalize_observation_payload(obs_payload))
        obs_id = cursor.lastrowid
        db.execute(
            "INSERT INTO vec_observations(rowid, embedding) VALUES (?, ?)",
            [obs_id, embedding_blob],
        )
        db.commit()
        return obs_id
    except Exception:
        db.rollback()
        raise


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
