"""時系列の有効期間と supersede 連鎖を扱う。"""

from __future__ import annotations

import json
import math
import sqlite3
import struct
from datetime import datetime, timezone
from typing import Any

from .config import ErinysConfig
from .db import embedding_engine, insert_observation_with_embedding, update_observation
from .embedding import serialize_f32
from .graph import create_edge
from .search import rrf_hybrid_search

NEGATION_TOKENS = {
    "not",
    "never",
    "no",
    "without",
    "cannot",
    "can't",
    "failed",
    "broken",
    "disabled",
    "avoid",
    "invalid",
    "deprecated",
    "ない",
    "無効",
    "禁止",
    "失敗",
    "壊れ",
    "不可",
}

ANTONYM_PAIRS = [
    ("enable", "disable"),
    ("enabled", "disabled"),
    ("success", "failure"),
    ("works", "broken"),
    ("required", "optional"),
    ("allow", "deny"),
    ("recommended", "avoid"),
    ("supported", "unsupported"),
    ("有効", "無効"),
    ("必要", "不要"),
    ("成功", "失敗"),
]


def _decode_json(value: object) -> object:
    if isinstance(value, str) and value:
        return json.loads(value)
    return value


def _observation_record(row: sqlite3.Row) -> dict[str, Any]:
    record = dict(row)
    record["metadata"] = _decode_json(record.get("metadata"))
    return record


def _fetch_observation(db: sqlite3.Connection, obs_id: int) -> dict[str, Any]:
    row = db.execute("SELECT * FROM observations WHERE id = ?", [obs_id]).fetchone()
    if row is None:
        raise LookupError(f"observation not found: {obs_id}")
    return _observation_record(row)


def _fetch_embedding_blob(db: sqlite3.Connection, obs_id: int) -> bytes:
    row = db.execute(
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


def _parse_as_of(as_of: str | datetime | None) -> datetime:
    if as_of is None:
        return datetime.now(timezone.utc)
    if isinstance(as_of, datetime):
        return as_of if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc)
    parsed = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _extract_keywords(text: str) -> set[str]:
    normalized = text.lower().replace("\n", " ")
    tokens = [token.strip(".,:;!?()[]{}\"'") for token in normalized.split()]
    return {token for token in tokens if len(token) >= 4}


def _has_negation(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in NEGATION_TOKENS)


def _has_antonym_conflict(text_a: str, text_b: str) -> bool:
    lowered_a = text_a.lower()
    lowered_b = text_b.lower()
    for left, right in ANTONYM_PAIRS:
        if left in lowered_a and right in lowered_b:
            return True
        if right in lowered_a and left in lowered_b:
            return True
    return False


def _looks_contradictory(obs_a: dict[str, Any], obs_b: dict[str, Any]) -> bool:
    text_a = f"{obs_a['title']} {obs_a['content']}"
    text_b = f"{obs_b['title']} {obs_b['content']}"
    keywords_a = _extract_keywords(text_a)
    keywords_b = _extract_keywords(text_b)
    overlap = keywords_a & keywords_b
    if obs_a.get("topic_key") and obs_a.get("topic_key") == obs_b.get("topic_key"):
        return True
    if not overlap:
        return False
    if bool(obs_a.get("is_anti_pattern")) != bool(obs_b.get("is_anti_pattern")):
        return True
    if _has_negation(text_a) != _has_negation(text_b):
        return True
    return _has_antonym_conflict(text_a, text_b)


def _is_valid_at(observation: dict[str, Any], moment: datetime) -> bool:
    valid_from = observation["valid_from"]
    valid_until = observation.get("valid_until")
    if valid_from > moment:
        return False
    if valid_until is None:
        return True
    return valid_until > moment


def supersede_observation(
    db: sqlite3.Connection,
    old_id: int,
    new_content: str,
    reason: str,
) -> dict[str, Any]:
    """旧 observation を閉じ、新 observation を作成する。"""
    old = _fetch_observation(db, old_id)
    now = datetime.now(timezone.utc)
    old_topic_key = old.get("topic_key")
    old_valid_until = old.get("valid_until")
    update_observation(
        db,
        old_id,
        {"valid_until": now, "topic_key": None if old_topic_key is not None else old_topic_key},
    )
    payload = {
        "title": old["title"],
        "content": new_content,
        "type": old["type"],
        "project": old["project"],
        "scope": old["scope"],
        "is_anti_pattern": old["is_anti_pattern"],
        "is_pattern": old["is_pattern"],
        "distillation_level": old["distillation_level"],
        "distilled_from": old["distilled_from"],
        "valid_from": now,
        "base_strength": old["base_strength"],
        "source": old["source"],
        "embedding_model": old["embedding_model"],
        "topic_key": old_topic_key,
        "metadata": {**dict(old.get("metadata") or {}), "supersede_reason": reason, "superseded_from": old_id},
        "session_id": old["session_id"],
    }
    try:
        embedding = embedding_engine.embed(new_content)
        new_id = insert_observation_with_embedding(db, payload, serialize_f32(embedding))
    except Exception:
        update_observation(db, old_id, {"valid_until": old_valid_until, "topic_key": old_topic_key})
        raise
    update_observation(db, old_id, {"superseded_by": new_id})
    edge = create_edge(
        db,
        old_id,
        new_id,
        "supersedes",
        1.0,
        {"reason": reason},
    )
    return {"old": _fetch_observation(db, old_id), "new": _fetch_observation(db, new_id), "edge": edge}


def query_as_of(
    db: sqlite3.Connection,
    query: str,
    as_of: str | datetime | None = None,
    *,
    project: str | None = None,
    limit: int = 10,
    metadata_filter: dict[str, object] | None = None,
    query_embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    """指定時点で有効な observation だけを返す。"""
    config = ErinysConfig()
    embedding = query_embedding or embedding_engine.embed(query)
    search_limit = min(max(limit * 5, limit), config.max_search_limit)
    moment = _parse_as_of(as_of)
    results = rrf_hybrid_search(
        db,
        query,
        embedding,
        project=project,
        metadata_filter=metadata_filter,
        limit=search_limit,
        k=config.rrf_k,
        fts_weight=config.fts_weight,
        vec_weight=config.vec_weight,
    )
    normalized = []
    for row in results:
        record = dict(row)
        record["metadata"] = _decode_json(record.get("metadata"))
        if _is_valid_at(record, moment):
            normalized.append(record)
    return normalized[:limit]


def conflict_check(
    db: sqlite3.Connection,
    observation_id: int,
    limit: int = 10,
    similarity_threshold: float = 0.75,
) -> list[dict[str, Any]]:
    """embedding 類似度と否定表現で矛盾候補を検出する。"""
    source = _fetch_observation(db, observation_id)
    source_blob = _fetch_embedding_blob(db, observation_id)
    source_vector = _vector_from_blob(source_blob)
    neighbor_count = max(limit * 10, 20)
    rows = db.execute(
        """
        SELECT rowid
        FROM vec_observations
        WHERE embedding MATCH ? AND k = ?
        """,
        [source_blob, neighbor_count],
    ).fetchall()
    conflicts: list[dict[str, Any]] = []
    for row in rows:
        candidate_id = int(row[0])
        if candidate_id == observation_id:
            continue
        candidate = _fetch_observation(db, candidate_id)
        candidate_vector = _vector_from_blob(_fetch_embedding_blob(db, candidate_id))
        similarity = _cosine_similarity(source_vector, candidate_vector)
        if similarity < similarity_threshold:
            continue
        if not _looks_contradictory(source, candidate):
            continue
        conflicts.append({"observation": candidate, "similarity": similarity})
    conflicts.sort(key=lambda item: float(item["similarity"]), reverse=True)
    return conflicts[:limit]
