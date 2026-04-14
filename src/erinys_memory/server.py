"""ERINYS FastMCP server と全 tool 実装。"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
import sqlite3
import struct
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Mapping

from fastmcp import FastMCP

from .collider import MemoryCollider, get_collision, pair_similarity, save_collision
from .config import ErinysConfig
from .db import (
    delete_observation_with_embedding,
    embedding_engine,
    get_db,
    init_db,
    insert_observation_with_embedding,
    update_observation,
)
from .decay import current_strength
from .distill import distill_observation
from .embedding import EmbeddingEngine, serialize_f32
from .graph import GraphEngine, VALID_RELATIONS, create_edge, traverse
from .search import rrf_hybrid_search
from .session import SessionManager, end_session, get_recent_sessions, save_session_summary, start_session
from .temporal import conflict_check, query_as_of, supersede_observation

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

VALID_TYPES = {
    "manual",
    "decision",
    "architecture",
    "bugfix",
    "pattern",
    "config",
    "discovery",
    "learning",
    "anti_pattern",
    "meta_knowledge",
}
VALID_SCOPES = {"project", "personal", "global"}
VALID_DISTILLATION_LEVELS = {"concrete", "abstract", "meta"}
SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|token|secret)\s*[:=]\s*([^\s,;]+)"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
]

mcp = FastMCP("ERINYS")
_CONFIG = ErinysConfig()
_DB: sqlite3.Connection | None = None
_DB_LOCK = Lock()
_COLLIDER = MemoryCollider(_CONFIG)


class ContentTooLongError(ValueError):
    """max_content_length / max_title_length 違反。"""


def _ok(data: Any) -> dict[str, Any]:
    return {"ok": True, "data": data, "error": None}


def _error(code: str, message: str) -> dict[str, Any]:
    return {"ok": False, "data": None, "error": {"message": message, "code": code}}


def _map_integrity_error(exc: sqlite3.IntegrityError) -> dict[str, Any]:
    message = str(exc)
    if "UNIQUE constraint failed" in message:
        return _error("DUPLICATE", message)
    if "CHECK constraint failed" in message or "FOREIGN KEY constraint failed" in message:
        return _error("VALIDATION", message)
    return _error("DB_ERROR", message)


def _envelope(action: Callable[[], Any]) -> dict[str, Any]:
    try:
        return _ok(action())
    except LookupError as exc:
        return _error("NOT_FOUND", str(exc))
    except ContentTooLongError as exc:
        return _error("CONTENT_TOO_LONG", str(exc))
    except sqlite3.IntegrityError as exc:
        return _map_integrity_error(exc)
    except ValueError as exc:
        return _error("VALIDATION", str(exc))
    except RuntimeError as exc:
        code = "EMBEDDING_ERROR" if "embed" in str(exc).lower() else "DB_ERROR"
        return _error(code, str(exc))
    except sqlite3.DatabaseError as exc:
        return _error("DB_ERROR", str(exc))
    except Exception as exc:
        logger.exception("Unhandled ERINYS error")
        return _error("DB_ERROR", str(exc))


def _db_path() -> str:
    return _CONFIG.db_path


def _maybe_backup_on_init() -> None:
    if not _CONFIG.db_backup_on_init or _db_path() == ":memory:":
        return
    source = Path(_db_path()).expanduser()
    if not source.exists():
        return
    backup = source.with_suffix(f"{source.suffix}.bak")
    shutil.copy2(source, backup)


def _db() -> sqlite3.Connection:
    global _DB
    with _DB_LOCK:
        if _DB is None:
            _maybe_backup_on_init()
            _DB = init_db(_CONFIG)
        return _DB


def _embedding() -> EmbeddingEngine:
    return embedding_engine


def _graph() -> GraphEngine:
    return GraphEngine(_db())


def _sessions() -> SessionManager:
    return SessionManager(_db())


def _decode_json(value: object) -> object:
    if isinstance(value, str) and value:
        return json.loads(value)
    return value


def _normalize_observation(row: sqlite3.Row | Mapping[str, object]) -> dict[str, Any]:
    record = dict(row)
    record["metadata"] = _decode_json(record.get("metadata"))
    return record


def _normalize_prompt(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _redact_text(text: str) -> str:
    if not _CONFIG.redact_secret_patterns:
        return text
    masked = text
    for pattern in SECRET_PATTERNS:
        masked = pattern.sub(lambda match: f"{match.group(1)}=[REDACTED]" if match.groups() else "[REDACTED]", masked)
    return masked


def _validate_title(title: str) -> None:
    if not title.strip():
        raise ValueError("title must not be empty")
    if len(title) > _CONFIG.max_title_length:
        raise ContentTooLongError("title exceeds max_title_length")


def _validate_content(content: str) -> None:
    if not content.strip():
        raise ValueError("content must not be empty")
    if len(content) > _CONFIG.max_content_length:
        raise ContentTooLongError("content exceeds max_content_length")


def _validate_type(obs_type: str) -> None:
    if obs_type not in VALID_TYPES:
        raise ValueError(f"invalid type: {obs_type}")


def _validate_scope(scope: str) -> None:
    if scope not in VALID_SCOPES:
        raise ValueError(f"invalid scope: {scope}")


def _validate_relation(relation: str) -> None:
    if relation not in VALID_RELATIONS:
        raise ValueError(f"invalid relation: {relation}")


def _validate_limit(limit: int) -> int:
    if limit <= 0:
        raise ValueError("limit must be positive")
    return min(limit, _CONFIG.max_search_limit)


def _validate_metadata_filter(metadata_filter: dict[str, object] | None) -> None:
    if metadata_filter is not None and not isinstance(metadata_filter, dict):
        raise ValueError("metadata_filter must be a dict")


def _session_exists(session_id: str) -> bool:
    row = _db().execute("SELECT 1 FROM sessions WHERE id = ?", [session_id]).fetchone()
    return row is not None


def _fetch_observation(obs_id: int) -> dict[str, Any]:
    row = _db().execute("SELECT * FROM observations WHERE id = ?", [obs_id]).fetchone()
    if row is None:
        raise LookupError(f"observation not found: {obs_id}")
    return _normalize_observation(row)


def _fetch_prompt(prompt_id: int) -> dict[str, Any]:
    row = _db().execute("SELECT * FROM prompts WHERE id = ?", [prompt_id]).fetchone()
    if row is None:
        raise LookupError(f"prompt not found: {prompt_id}")
    return _normalize_prompt(row)


def _fetch_embedding_blob(obs_id: int) -> bytes:
    row = _db().execute(
        "SELECT embedding FROM vec_observations WHERE rowid = ?",
        [obs_id],
    ).fetchone()
    if row is None:
        raise LookupError(f"embedding not found: {obs_id}")
    return bytes(row[0])


def _vector_from_blob(blob: bytes) -> list[float]:
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob))


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _infer_flags(obs_type: str) -> tuple[int, int]:
    return int(obs_type == "anti_pattern"), int(obs_type == "pattern")


def _find_topic_key_owner(project: str | None, scope: str, topic_key: str) -> int | None:
    row = _db().execute(
        """
        SELECT id
        FROM observations
        WHERE ((project = ?) OR (project IS NULL AND ? IS NULL))
          AND scope = ?
          AND topic_key = ?
        """,
        [project, project, scope, topic_key],
    ).fetchone()
    return None if row is None else int(row["id"])


def _upsert_fields(payload: Mapping[str, object]) -> dict[str, object]:
    return {
        "title": payload["title"],
        "content": payload["content"],
        "type": payload["type"],
        "project": payload["project"],
        "scope": payload["scope"],
        "is_anti_pattern": payload["is_anti_pattern"],
        "is_pattern": payload["is_pattern"],
        "source": payload["source"],
        "embedding_model": payload["embedding_model"],
        "topic_key": payload["topic_key"],
        "metadata": payload["metadata"],
        "session_id": payload["session_id"],
    }


def _audit(
    operation: str,
    target_type: str | None = None,
    target_id: int | None = None,
    detail: dict[str, object] | None = None,
) -> None:
    if not _CONFIG.enable_audit_log:
        return
    try:
        _db().execute(
            """
            INSERT INTO audit_log(operation, target_type, target_id, detail)
            VALUES (?, ?, ?, ?)
            """,
            [
                operation,
                target_type,
                target_id,
                json.dumps(detail, ensure_ascii=False) if detail is not None else None,
            ],
        )
        _db().commit()
    except sqlite3.DatabaseError:
        logger.warning("Failed to write audit_log for %s", operation)


def _observation_payload(
    title: str,
    content: str,
    obs_type: str,
    project: str | None,
    scope: str,
    topic_key: str | None,
    session_id: str | None,
    metadata: dict[str, object] | None,
    source: str = "user",
) -> dict[str, object]:
    _validate_title(title)
    _validate_content(content)
    _validate_type(obs_type)
    _validate_scope(scope)
    if session_id is not None and not _session_exists(session_id):
        raise LookupError(f"session not found: {session_id}")
    is_anti_pattern, is_pattern = _infer_flags(obs_type)
    return {
        "title": _redact_text(title),
        "content": _redact_text(content),
        "type": obs_type,
        "project": project,
        "scope": scope,
        "is_anti_pattern": is_anti_pattern,
        "is_pattern": is_pattern,
        "source": source,
        "embedding_model": _embedding().model_name,
        "topic_key": topic_key,
        "metadata": metadata,
        "session_id": session_id,
    }


def _persist_observation(
    payload: dict[str, object],
    *,
    embedding: list[float] | None = None,
) -> tuple[dict[str, Any], str]:
    existing_id = None
    topic_key = payload.get("topic_key")
    if isinstance(topic_key, str):
        existing_id = _find_topic_key_owner(
            payload.get("project"),
            str(payload["scope"]),
            topic_key,
        )
    if existing_id is not None:
        update_observation(_db(), existing_id, _upsert_fields(payload))
        return _fetch_observation(existing_id), "updated"
    vector = embedding or _embedding().embed(str(payload["content"]))
    obs_id = insert_observation_with_embedding(_db(), payload, serialize_f32(vector))
    return _fetch_observation(obs_id), "created"


def _search_results(
    query: str,
    project: str | None,
    limit: int,
    include_anti_patterns: bool,
    include_distilled: bool,
    metadata_filter: dict[str, object] | None,
) -> list[dict[str, Any]]:
    _validate_metadata_filter(metadata_filter)
    normalized_limit = _validate_limit(limit)
    query_embedding = _embedding().embed(query)
    results = rrf_hybrid_search(
        _db(),
        query,
        query_embedding,
        project=project,
        metadata_filter=metadata_filter,
        limit=min(normalized_limit * 5, _CONFIG.max_search_limit),
        k=_CONFIG.rrf_k,
        fts_weight=_CONFIG.fts_weight,
        vec_weight=_CONFIG.vec_weight,
    )
    filtered: list[dict[str, Any]] = []
    for row in results:
        record = _normalize_observation(row)
        if not include_anti_patterns and bool(record.get("is_anti_pattern")):
            continue
        if not include_distilled and record.get("distillation_level") is not None:
            continue
        filtered.append(record)
    return filtered[:normalized_limit]


def _recent_observations(project: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
    normalized_limit = _validate_limit(limit)
    if project is None:
        rows = _db().execute(
            """
            SELECT *
            FROM observations
            WHERE valid_until IS NULL
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            [normalized_limit],
        ).fetchall()
    else:
        rows = _db().execute(
            """
            SELECT *
            FROM observations
            WHERE project = ? AND valid_until IS NULL
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            [project, normalized_limit],
        ).fetchall()
    records = [_normalize_observation(row) for row in rows]
    for record in records:
        record["effective_strength"] = current_strength(
            float(record["base_strength"]),
            record["created_at"],
            record.get("last_accessed"),
            int(record["access_count"]),
        )
    return records


def _count_cascade_targets(obs_id: int) -> dict[str, int]:
    edges = _db().execute(
        "SELECT COUNT(*) AS count FROM edges WHERE source_id = ? OR target_id = ?",
        [obs_id, obs_id],
    ).fetchone()
    collisions = _db().execute(
        "SELECT COUNT(*) AS count FROM collisions WHERE source_a = ? OR source_b = ?",
        [obs_id, obs_id],
    ).fetchone()
    return {
        "edges": int(edges["count"]) if edges is not None else 0,
        "collisions": int(collisions["count"]) if collisions is not None else 0,
    }


def _save_prompt_record(content: str, project: str | None, session_id: str | None) -> dict[str, Any]:
    _validate_content(content)
    if session_id is not None and not _session_exists(session_id):
        raise LookupError(f"session not found: {session_id}")
    cursor = _db().execute(
        """
        INSERT INTO prompts(content, project, session_id)
        VALUES (?, ?, ?)
        """,
        [_redact_text(content), project, session_id],
    )
    _db().commit()
    return _fetch_prompt(int(cursor.lastrowid))


def _related_wikilinks(obs_id: int) -> str:
    rows = _db().execute(
        """
        SELECT o.title
        FROM edges e
        JOIN observations o ON o.id = e.target_id
        WHERE e.source_id = ?
        ORDER BY e.created_at ASC, e.id ASC
        """,
        [obs_id],
    ).fetchall()
    return " ".join(f"[[{row['title']}]]" for row in rows)


def _render_markdown_export(project: str | None = None) -> str:
    if project is None:
        rows = _db().execute("SELECT * FROM observations ORDER BY created_at ASC, id ASC").fetchall()
    else:
        rows = _db().execute(
            "SELECT * FROM observations WHERE project = ? ORDER BY created_at ASC, id ASC",
            [project],
        ).fetchall()
    sections: list[str] = []
    for row in rows:
        record = _normalize_observation(row)
        lines = [
            f"## {record['title']}",
            f"- ID: {record['id']}",
            f"- Type: {record['type']}",
            f"- Project: {record['project'] or 'global'}",
            f"- Scope: {record['scope']}",
            f"- Valid: {record['valid_from']} -> {record['valid_until'] or 'current'}",
            "",
            str(record["content"]),
        ]
        wikilinks = _related_wikilinks(int(record["id"]))
        if wikilinks:
            lines.extend(["", f"Related: {wikilinks}"])
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def _default_backup_path() -> Path:
    timestamp = _now().strftime("%Y%m%d%H%M%S")
    if _db_path() == ":memory:":
        return Path(tempfile.gettempdir()) / f"erinys-memory-{timestamp}.db"
    source = Path(_db_path()).expanduser()
    source.parent.mkdir(parents=True, exist_ok=True)
    return source.with_name(f"{source.stem}-{timestamp}{source.suffix or '.db'}")


def _backup_database(path: str | None = None) -> dict[str, Any]:
    backup_path = Path(path).expanduser() if path is not None else _default_backup_path()
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    _db().execute("PRAGMA wal_checkpoint(PASSIVE)")
    target = get_db(ErinysConfig(db_path=str(backup_path)))
    try:
        _db().backup(target)
    finally:
        target.close()
    return {
        "path": str(backup_path),
        "size_bytes": backup_path.stat().st_size,
        "created_at": _now().isoformat(),
    }


def _vector_health() -> dict[str, int]:
    orphan = _db().execute(
        "SELECT COUNT(*) AS count FROM vec_observations WHERE rowid NOT IN (SELECT id FROM observations)"
    ).fetchone()
    missing = _db().execute(
        "SELECT COUNT(*) AS count FROM observations WHERE id NOT IN (SELECT rowid FROM vec_observations)"
    ).fetchone()
    return {
        "orphan_vectors": int(orphan["count"]) if orphan is not None else 0,
        "missing_vectors": int(missing["count"]) if missing is not None else 0,
    }


def _auto_link(created: list[dict[str, Any]], embeddings: list[list[float]]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for index, left in enumerate(created):
        for right_index in range(index + 1, len(created)):
            similarity = _cosine_similarity(embeddings[index], embeddings[right_index])
            if 0.80 <= similarity < 0.99:
                edges.append(
                    create_edge(
                        _db(),
                        int(left["id"]),
                        int(created[right_index]["id"]),
                        "similar_to",
                        weight=round(similarity, 4),
                        metadata={"auto_link": True},
                    )
                )
    return edges


def _evaluation_metrics(project: str | None = None) -> dict[str, Any]:
    if project is None:
        rows = _db().execute("SELECT * FROM observations").fetchall()
    else:
        rows = _db().execute("SELECT * FROM observations WHERE project = ?", [project]).fetchall()
    observations = [_normalize_observation(row) for row in rows]
    total = len(observations)
    active = sum(1 for record in observations if record.get("valid_until") is None)
    distilled = sum(1 for record in observations if record.get("distillation_level") is not None)
    linked = _db().execute("SELECT COUNT(*) AS count FROM edges").fetchone()
    session_bound = sum(1 for record in observations if record.get("session_id"))
    avg_strength = (
        sum(
            current_strength(
                float(record["base_strength"]),
                record["created_at"],
                record.get("last_accessed"),
                int(record["access_count"]),
            )
            for record in observations
        )
        / total
        if total
        else 0.0
    )
    metrics = {
        "total_observations": total,
        "active_ratio": active / total if total else 0.0,
        "distillation_ratio": distilled / total if total else 0.0,
        "session_coverage": session_bound / total if total else 0.0,
        "avg_effective_strength": avg_strength,
        "edge_density": (int(linked["count"]) / total) if total and linked is not None else 0.0,
        "vector_health": _vector_health(),
    }
    metrics["score"] = round(
        (
            metrics["active_ratio"]
            + metrics["distillation_ratio"]
            + metrics["session_coverage"]
            + min(metrics["avg_effective_strength"] / 2.0, 1.0)
            + min(metrics["edge_density"], 1.0)
        )
        / 5.0,
        4,
    )
    return metrics


@mcp.tool
def erinys_save(
    title: str,
    content: str,
    type: str = "manual",
    project: str | None = None,
    scope: str = "project",
    topic_key: str | None = None,
    session_id: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """Save a structured observation to ERINYS memory."""

    def action() -> dict[str, Any]:
        payload = _observation_payload(title, content, type, project, scope, topic_key, session_id, metadata)
        record, status = _persist_observation(payload)
        _audit("save", "observation", int(record["id"]), {"status": status, "topic_key": topic_key})
        auto_distilled: list[dict[str, Any]] = []
        if status == "created" and _CONFIG.auto_distill_on_save:
            try:
                result = distill_observation(_db(), int(record["id"]), "meta")
                auto_distilled = result.get("created", [])
                _audit("auto_distill", "observation", int(record["id"]), {"count": len(auto_distilled)})
            except Exception as exc:
                logger.warning("auto-distill failed for observation %s: %s", record["id"], exc)
        return {
            "id": record["id"],
            "status": status,
            "observation": record,
            "auto_distilled": [{"id": d["id"], "level": d.get("distillation_level")} for d in auto_distilled],
        }

    return _envelope(action)


@mcp.tool
def erinys_get(
    id: int,
) -> dict:
    """Get a single observation by ID (full content, untruncated)."""
    return _envelope(lambda: {"observation": _fetch_observation(id)})


@mcp.tool
def erinys_update(
    id: int,
    title: str | None = None,
    content: str | None = None,
    type: str | None = None,
    project: str | None = None,
    scope: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """Update an existing observation. Only provided fields are changed."""

    def action() -> dict[str, Any]:
        _fetch_observation(id)
        fields: dict[str, object] = {}
        if title is not None:
            _validate_title(title)
            fields["title"] = _redact_text(title)
        if content is not None:
            _validate_content(content)
            fields["content"] = _redact_text(content)
        if type is not None:
            _validate_type(type)
            anti, pattern = _infer_flags(type)
            fields["type"] = type
            fields["is_anti_pattern"] = anti
            fields["is_pattern"] = pattern
        if project is not None:
            fields["project"] = project
        if scope is not None:
            _validate_scope(scope)
            fields["scope"] = scope
        if metadata is not None:
            fields["metadata"] = metadata
        update_observation(_db(), id, fields)
        record = _fetch_observation(id)
        _audit("update", "observation", id, {"fields": sorted(fields)})
        return {"id": id, "observation": record}

    return _envelope(action)


@mcp.tool
def erinys_delete(
    id: int,
) -> dict:
    """Delete an observation and cascade dependent rows via FK rules."""

    def action() -> dict[str, Any]:
        _fetch_observation(id)
        cascaded = _count_cascade_targets(id)
        delete_observation_with_embedding(_db(), id)
        _audit("delete", "observation", id, cascaded)
        return {"id": id, "deleted": True, "cascaded": cascaded}

    return _envelope(action)


@mcp.tool
def erinys_search(
    query: str,
    project: str | None = None,
    limit: int = 10,
    include_anti_patterns: bool = True,
    include_distilled: bool = True,
    metadata_filter: dict | None = None,
) -> dict:
    """RRF hybrid search (FTS5 keyword + vector similarity)."""

    def action() -> dict[str, Any]:
        results = _search_results(query, project, limit, include_anti_patterns, include_distilled, metadata_filter)
        _audit("search", "observation", None, {"query": query, "count": len(results), "project": project})
        return {"query": query, "results": results}

    return _envelope(action)


@mcp.tool
def erinys_save_prompt(
    content: str,
    project: str | None = None,
    session_id: str | None = None,
) -> dict:
    """Save a user prompt to track intent and goals."""

    def action() -> dict[str, Any]:
        prompt = _save_prompt_record(content, project, session_id)
        _audit("save_prompt", "prompt", int(prompt["id"]), {"project": project, "session_id": session_id})
        return {"prompt": prompt}

    return _envelope(action)


@mcp.tool
def erinys_session_start(
    id: str,
    project: str,
    directory: str | None = None,
) -> dict:
    """Start a new session."""

    def action() -> dict[str, Any]:
        session = start_session(_db(), id, project, directory)
        _audit("session_start", "session", None, {"id": id, "project": project})
        return {"session": session}

    return _envelope(action)


@mcp.tool
def erinys_session_end(
    id: str,
    summary: str | None = None,
) -> dict:
    """End a session with optional summary."""

    def action() -> dict[str, Any]:
        session = end_session(_db(), id, _redact_text(summary) if summary else None)
        _audit("session_end", "session", None, {"id": id})
        return {"session": session}

    return _envelope(action)


@mcp.tool
def erinys_session_summary(
    content: str,
    project: str,
    session_id: str | None = None,
) -> dict:
    """Save comprehensive end-of-session summary (Goal/Discoveries/Accomplished)."""

    def action() -> dict[str, Any]:
        _validate_content(content)
        summary = save_session_summary(_db(), _redact_text(content), project, session_id)
        _audit("session_summary", "observation", int(summary["id"]), {"project": project, "session_id": session_id})
        return {"summary": _normalize_observation(summary)}

    return _envelope(action)


@mcp.tool
def erinys_recall(
    project: str | None = None,
    limit: int = 10,
) -> dict:
    """Recall recent observations for context."""

    def action() -> dict[str, Any]:
        observations = _recent_observations(project, limit)
        _audit("recall", "observation", None, {"count": len(observations), "project": project})
        return {"observations": observations}

    return _envelope(action)


@mcp.tool
def erinys_context(
    project: str,
    limit: int = 20,
) -> dict:
    """Get recent session context for a project."""

    def action() -> dict[str, Any]:
        sessions = get_recent_sessions(_db(), project, _validate_limit(limit))
        observations = _recent_observations(project, min(limit, 10))
        _audit("context", "session", None, {"project": project, "session_count": len(sessions)})
        return {"project": project, "sessions": sessions, "observations": observations}

    return _envelope(action)


@mcp.tool
def erinys_export(
    project: str | None = None,
    format: str = "markdown",
) -> dict:
    """Export observations as markdown payload (Obsidian-compatible [[wikilinks]])."""

    def action() -> dict[str, Any]:
        if format != "markdown":
            raise ValueError("only markdown export is supported")
        payload = _render_markdown_export(project)
        _audit("export", "observation", None, {"project": project, "format": format})
        return {"format": format, "content": payload}

    return _envelope(action)


@mcp.tool
def erinys_backup(
    path: str | None = None,
) -> dict:
    """Create a consistent SQLite backup and return backup metadata."""

    def action() -> dict[str, Any]:
        backup = _backup_database(path)
        _audit("backup", "observation", None, {"path": backup["path"]})
        return backup

    return _envelope(action)


@mcp.tool
def erinys_stats(
    project: str | None = None,
) -> dict:
    """Database statistics: observation count, project breakdown, health."""

    def action() -> dict[str, Any]:
        if project is None:
            total_row = _db().execute("SELECT COUNT(*) AS count FROM observations").fetchone()
            project_rows = _db().execute(
                """
                SELECT COALESCE(project, '__NULL__') AS project_name, COUNT(*) AS count
                FROM observations
                GROUP BY COALESCE(project, '__NULL__')
                ORDER BY count DESC
                """
            ).fetchall()
        else:
            total_row = _db().execute(
                "SELECT COUNT(*) AS count FROM observations WHERE project = ?",
                [project],
            ).fetchone()
            project_rows = _db().execute(
                "SELECT ? AS project_name, COUNT(*) AS count FROM observations WHERE project = ?",
                [project, project],
            ).fetchall()
        stats = {
            "project": project,
            "observation_count": int(total_row["count"]) if total_row is not None else 0,
            "session_count": int(_db().execute("SELECT COUNT(*) AS count FROM sessions").fetchone()["count"]),
            "edge_count": int(_db().execute("SELECT COUNT(*) AS count FROM edges").fetchone()["count"]),
            "collision_count": int(_db().execute("SELECT COUNT(*) AS count FROM collisions").fetchone()["count"]),
            "prompt_count": int(_db().execute("SELECT COUNT(*) AS count FROM prompts").fetchone()["count"]),
            "projects": [{row["project_name"]: row["count"]} for row in project_rows],
            "vector_health": _vector_health(),
        }
        if _db_path() != ":memory:":
            db_file = Path(_db_path()).expanduser()
            stats["db_size_bytes"] = db_file.stat().st_size if db_file.exists() else 0
        _audit("stats", "observation", None, {"project": project})
        return stats

    return _envelope(action)


@mcp.tool
def erinys_link(
    source_id: int,
    target_id: int,
    relation: str,
    weight: float = 1.0,
) -> dict:
    """Create typed edge between observations."""

    def action() -> dict[str, Any]:
        _validate_relation(relation)
        edge = create_edge(_db(), source_id, target_id, relation, weight)
        _audit("link", "edge", int(edge["id"]), {"relation": relation})
        return {"edge": edge}

    return _envelope(action)


@mcp.tool
def erinys_traverse(
    start_id: int,
    max_depth: int = 2,
    relation_filter: list[str] | None = None,
) -> dict:
    """Traverse graph from a starting observation."""

    def action() -> dict[str, Any]:
        if relation_filter:
            for relation in relation_filter:
                _validate_relation(relation)
        graph = traverse(_db(), start_id, max_depth, relation_filter)
        _audit("traverse", "observation", start_id, {"max_depth": max_depth})
        return graph

    return _envelope(action)


@mcp.tool
def erinys_prune(
    threshold: float = 0.1,
    dry_run: bool = True,
) -> dict:
    """Prune decayed observations below threshold."""

    def action() -> dict[str, Any]:
        rows = _db().execute("SELECT * FROM observations").fetchall()
        candidates: list[dict[str, Any]] = []
        for row in rows:
            record = _normalize_observation(row)
            strength = current_strength(
                float(record["base_strength"]),
                record["created_at"],
                record.get("last_accessed"),
                int(record["access_count"]),
            )
            if strength < threshold:
                record["effective_strength"] = strength
                candidates.append(record)
        deleted_ids: list[int] = []
        if not dry_run:
            for record in candidates:
                delete_observation_with_embedding(_db(), int(record["id"]))
                deleted_ids.append(int(record["id"]))
        _audit("prune", "observation", None, {"threshold": threshold, "count": len(candidates), "dry_run": dry_run})
        return {"threshold": threshold, "dry_run": dry_run, "candidates": candidates, "deleted_ids": deleted_ids}

    return _envelope(action)


@mcp.tool
def erinys_batch_save(
    items: list[dict],
    auto_link: bool = True,
) -> dict:
    """Bulk-add observations with automatic similarity linking."""

    def action() -> dict[str, Any]:
        if not items:
            raise ValueError("items must not be empty")
        payloads: list[dict[str, object]] = []
        contents: list[str] = []
        for item in items:
            if "title" not in item or "content" not in item:
                raise ValueError("each batch item must include title and content")
            payload = _observation_payload(
                str(item["title"]),
                str(item["content"]),
                str(item.get("type", "manual")),
                item.get("project"),
                str(item.get("scope", "project")),
                item.get("topic_key"),
                item.get("session_id"),
                item.get("metadata"),
            )
            payloads.append(payload)
            contents.append(str(payload["content"]))
        embeddings = _embedding().embed_batch(contents)
        created: list[dict[str, Any]] = []
        for payload, vector in zip(payloads, embeddings):
            record, _ = _persist_observation(payload, embedding=vector)
            created.append(record)
        edges = _auto_link(created, embeddings) if auto_link else []
        _audit("batch_save", "observation", None, {"count": len(created), "auto_link": auto_link, "edges": len(edges)})
        return {"observations": created, "edges": edges}

    return _envelope(action)


@mcp.tool
def erinys_reinforce(
    observation_id: int,
) -> dict:
    """Reinforce an observation by updating access_count and last_accessed."""

    def action() -> dict[str, Any]:
        record = _fetch_observation(observation_id)
        update_observation(
            _db(),
            observation_id,
            {
                "access_count": int(record["access_count"]) + 1,
                "last_accessed": _now(),
            },
        )
        updated = _fetch_observation(observation_id)
        updated["effective_strength"] = current_strength(
            float(updated["base_strength"]),
            updated["created_at"],
            updated.get("last_accessed"),
            int(updated["access_count"]),
        )
        _audit("reinforce", "observation", observation_id, {"access_count": updated["access_count"]})
        return {"observation": updated}

    return _envelope(action)


@mcp.tool
def erinys_conflict_check(
    observation_id: int,
) -> dict:
    """Detect contradicting observations."""

    def action() -> dict[str, Any]:
        conflicts = conflict_check(_db(), observation_id)
        _audit("conflict_check", "observation", observation_id, {"count": len(conflicts)})
        return {"observation_id": observation_id, "conflicts": conflicts}

    return _envelope(action)


@mcp.tool
def erinys_supersede(
    old_id: int,
    new_content: str,
    reason: str,
) -> dict:
    """Supersede an old observation with updated fact."""

    def action() -> dict[str, Any]:
        _validate_content(new_content)
        result = supersede_observation(_db(), old_id, _redact_text(new_content), reason)
        _audit("supersede", "observation", old_id, {"new_id": result["new"]["id"], "reason": reason})
        return result

    return _envelope(action)


@mcp.tool
def erinys_timeline(
    query: str,
    as_of: str | None = None,
) -> dict:
    """Query facts valid at a specific point in time."""

    def action() -> dict[str, Any]:
        results = query_as_of(_db(), query, as_of, limit=_CONFIG.default_search_limit)
        _audit("timeline", "observation", None, {"query": query, "as_of": as_of, "count": len(results)})
        return {"query": query, "as_of": as_of or _now().isoformat(), "results": results}

    return _envelope(action)


@mcp.tool
def erinys_collide(
    obs_a_id: int,
    obs_b_id: int,
) -> dict:
    """Manually collide two observations to generate insight."""

    def action() -> dict[str, Any]:
        existing = get_collision(_db(), obs_a_id, obs_b_id)
        if existing is not None:
            return {"created": False, "collision": existing}
        obs_a = _fetch_observation(obs_a_id)
        obs_b = _fetch_observation(obs_b_id)
        insight = _COLLIDER.collide(_db(), obs_a, obs_b)
        if insight is None:
            return {"created": False, "collision": None}
        collision = save_collision(_db(), obs_a_id, obs_b_id, insight, pair_similarity(_db(), obs_a_id, obs_b_id))
        _audit("collide", "observation", None, {"source_a": obs_a_id, "source_b": obs_b_id})
        return {"created": True, "collision": collision}

    return _envelope(action)


@mcp.tool
def erinys_dream(
    max_collisions: int = 10,
) -> dict:
    """Run Dream Cycle: batch collision of candidate memory pairs."""

    def action() -> dict[str, Any]:
        collisions = _COLLIDER.dream_cycle(_db(), max_collisions=max_collisions)
        _audit("dream", "observation", None, {"count": len(collisions)})
        return {"collisions": collisions}

    return _envelope(action)


@mcp.tool
def erinys_distill(
    observation_id: int,
    level: str = "abstract",
) -> dict:
    """Distill observation to higher abstraction level."""

    def action() -> dict[str, Any]:
        if level not in VALID_DISTILLATION_LEVELS:
            raise ValueError(f"invalid distillation level: {level}")
        result = distill_observation(_db(), observation_id, level)
        _audit("distill", "observation", observation_id, {"level": level, "count": len(result['created'])})
        return result

    return _envelope(action)


@mcp.tool
def erinys_eval(
    project: str | None = None,
) -> dict:
    """Self-evaluate memory quality (LOCOMO-inspired metrics)."""

    def action() -> dict[str, Any]:
        metrics = _evaluation_metrics(project)
        _audit("eval", "observation", None, {"project": project, "score": metrics["score"]})
        return metrics

    return _envelope(action)


def main() -> None:
    """Console script entry point."""
    mcp.run()


if __name__ == "__main__":
    mcp.run()
