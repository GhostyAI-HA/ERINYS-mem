"""セッションの開始・終了・要約保存を扱う。"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from .db import embedding_engine, insert_observation_with_embedding
from .embedding import serialize_f32


def _decode_json(value: object) -> object:
    if isinstance(value, str) and value:
        return json.loads(value)
    return value


def _normalize_metadata(metadata: dict[str, object] | None) -> str | None:
    return json.dumps(metadata, ensure_ascii=False) if metadata is not None else None


def _session_record(row: sqlite3.Row) -> dict[str, Any]:
    session = dict(row)
    session["metadata"] = _decode_json(session.get("metadata"))
    return session


def _fetch_session(db: sqlite3.Connection, session_id: str) -> dict[str, Any]:
    row = db.execute(
        """
        SELECT id, project, directory, started_at, ended_at, summary, metadata
        FROM sessions
        WHERE id = ?
        """,
        [session_id],
    ).fetchone()
    if row is None:
        raise LookupError(f"session not found: {session_id}")
    return _session_record(row)


def _session_observations(db: sqlite3.Connection, session_id: str) -> list[dict[str, Any]]:
    rows = db.execute(
        """
        SELECT id, title, content, type, project, scope, session_id,
               metadata, created_at, updated_at, valid_from, valid_until
        FROM observations
        WHERE session_id = ?
        ORDER BY created_at ASC, id ASC
        """,
        [session_id],
    ).fetchall()
    return [_decode_observation(row) for row in rows]


def _decode_observation(row: sqlite3.Row) -> dict[str, Any]:
    observation = dict(row)
    observation["metadata"] = _decode_json(observation.get("metadata"))
    return observation


def start_session(
    db: sqlite3.Connection,
    session_id: str,
    project: str,
    directory: str | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, Any]:
    """sessions 行を作成する。"""
    db.execute(
        """
        INSERT INTO sessions(id, project, directory, metadata)
        VALUES (?, ?, ?, ?)
        """,
        [session_id, project, directory, _normalize_metadata(metadata)],
    )
    db.commit()
    return _fetch_session(db, session_id)


def end_session(
    db: sqlite3.Connection,
    session_id: str,
    summary: str | None = None,
) -> dict[str, Any]:
    """終了時刻と任意 summary を記録し、WAL をフラッシュする。"""
    if summary is None:
        cursor = db.execute(
            "UPDATE sessions SET ended_at = datetime('now') WHERE id = ?",
            [session_id],
        )
    else:
        cursor = db.execute(
            "UPDATE sessions SET ended_at = datetime('now'), summary = ? WHERE id = ?",
            [summary, session_id],
        )
    if cursor.rowcount == 0:
        raise LookupError(f"session not found: {session_id}")
    db.commit()
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return _fetch_session(db, session_id)


def save_session_summary(
    db: sqlite3.Connection,
    content: str,
    project: str,
    session_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, Any]:
    """セッション要約を observation として保存する。"""
    if session_id is not None:
        _fetch_session(db, session_id)
    summary_metadata = dict(metadata or {})
    summary_metadata["session_summary"] = True
    payload = {
        "title": f"Session Summary: {project}",
        "content": content,
        "type": "learning",
        "project": project,
        "scope": "project",
        "source": "agent",
        "session_id": session_id,
        "metadata": summary_metadata,
    }
    embedding = embedding_engine.embed(content)
    obs_id = insert_observation_with_embedding(db, payload, serialize_f32(embedding))
    if session_id is not None:
        cursor = db.execute(
            "UPDATE sessions SET summary = ? WHERE id = ?",
            [content, session_id],
        )
        if cursor.rowcount == 0:
            raise LookupError(f"session not found: {session_id}")
        db.commit()
    row = db.execute("SELECT * FROM observations WHERE id = ?", [obs_id]).fetchone()
    if row is None:
        raise LookupError("session summary observation not found after insert")
    return _decode_observation(row)


def get_recent_sessions(
    db: sqlite3.Connection,
    project: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """最近の sessions と紐付いた observations を返す。"""
    if project is None:
        rows = db.execute(
            """
            SELECT id, project, directory, started_at, ended_at, summary, metadata
            FROM sessions
            ORDER BY started_at DESC, id DESC
            LIMIT ?
            """,
            [limit],
        ).fetchall()
    else:
        rows = db.execute(
            """
            SELECT id, project, directory, started_at, ended_at, summary, metadata
            FROM sessions
            WHERE project = ?
            ORDER BY started_at DESC, id DESC
            LIMIT ?
            """,
            [project, limit],
        ).fetchall()
    sessions = [_session_record(row) for row in rows]
    for session in sessions:
        session["observations"] = _session_observations(db, str(session["id"]))
    return sessions


class SessionManager:
    """server.py 向けの薄い stateful wrapper。"""

    def __init__(self, db: sqlite3.Connection) -> None:
        self.db = db

    def start_session(
        self,
        session_id: str,
        project: str,
        directory: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        return start_session(self.db, session_id, project, directory, metadata)

    def end_session(
        self,
        session_id: str,
        summary: str | None = None,
    ) -> dict[str, Any]:
        return end_session(self.db, session_id, summary)

    def save_session_summary(
        self,
        content: str,
        project: str,
        session_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        return save_session_summary(self.db, content, project, session_id, metadata)

    def get_recent_sessions(
        self,
        project: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        return get_recent_sessions(self.db, project, limit)
