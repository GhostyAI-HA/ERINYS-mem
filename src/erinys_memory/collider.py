"""異なる文脈の記憶を衝突させて新しい洞察を生成する。"""

from __future__ import annotations

import json
import math
import sqlite3
import struct
from typing import Any

from .config import ErinysConfig

STOPWORDS = {
    "the",
    "and",
    "with",
    "that",
    "this",
    "from",
    "have",
    "into",
    "when",
    "where",
    "then",
    "than",
    "about",
    "because",
    "project",
    "session",
    "using",
    "used",
    "there",
    "their",
    "agent",
}


def _decode_json(value: object) -> object:
    if isinstance(value, str) and value:
        return json.loads(value)
    return value


def _normalize_pair(source_a: int, source_b: int) -> tuple[int, int]:
    return (source_a, source_b) if source_a < source_b else (source_b, source_a)


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


def _extract_keywords(text: str) -> list[str]:
    lowered = text.lower().replace("\n", " ")
    raw_tokens = [token.strip(".,:;!?()[]{}\"'") for token in lowered.split()]
    keywords = [token for token in raw_tokens if len(token) >= 4 and token not in STOPWORDS]
    seen: set[str] = set()
    ordered: list[str] = []
    for token in keywords:
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def _context_differs(obs_a: dict[str, Any], obs_b: dict[str, Any]) -> bool:
    return obs_a.get("project") != obs_b.get("project") or obs_a.get("session_id") != obs_b.get("session_id")


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


def _fetch_observations_with_embeddings(
    db: sqlite3.Connection,
) -> list[tuple[dict[str, Any], list[float]]]:
    rows = db.execute(
        """
        SELECT o.*, v.embedding
        FROM observations o
        JOIN vec_observations v ON v.rowid = o.id
        ORDER BY o.id ASC
        """
    ).fetchall()
    return [(_observation_record(row), _vector_from_blob(bytes(row["embedding"]))) for row in rows]


def get_collision(
    db: sqlite3.Connection,
    source_a: int,
    source_b: int,
) -> dict[str, Any] | None:
    """正規化済みペアで collision を取得する。"""
    normalized_a, normalized_b = _normalize_pair(source_a, source_b)
    row = db.execute(
        """
        SELECT id, source_a, source_b, insight, confidence, accepted, created_at
        FROM collisions
        WHERE source_a = ? AND source_b = ?
        """,
        [normalized_a, normalized_b],
    ).fetchone()
    return dict(row) if row is not None else None


def save_collision(
    db: sqlite3.Connection,
    source_a: int,
    source_b: int,
    insight: str,
    confidence: float | None,
    accepted: bool | None = None,
) -> dict[str, Any]:
    """collision を正規化して保存する。"""
    normalized_a, normalized_b = _normalize_pair(source_a, source_b)
    db.execute(
        """
        INSERT INTO collisions(source_a, source_b, insight, confidence, accepted)
        VALUES (?, ?, ?, ?, ?)
        """,
        [normalized_a, normalized_b, insight, confidence, accepted],
    )
    db.commit()
    collision = get_collision(db, normalized_a, normalized_b)
    if collision is None:
        raise LookupError("collision not found after insert")
    return collision


def pair_similarity(
    db: sqlite3.Connection,
    source_a: int,
    source_b: int,
) -> float:
    """2 observation の cosine similarity を返す。"""
    left = _vector_from_blob(_fetch_embedding_blob(db, source_a))
    right = _vector_from_blob(_fetch_embedding_blob(db, source_b))
    return _cosine_similarity(left, right)


class MemoryCollider:
    """
    異なる文脈の記憶をぶつけて新しい洞察を生成する。

    衝突条件:
    1. cosine similarity が 0.65-0.90
    2. project または session が異なる
    3. collisions テーブルに未登録
    """

    def __init__(self, config: ErinysConfig | None = None) -> None:
        self.config = config or ErinysConfig()

    def find_collision_candidates(
        self,
        db: sqlite3.Connection,
        limit: int = 20,
    ) -> list[tuple[int, int, float]]:
        """全 observation のペアから衝突候補を見つける。"""
        observations = _fetch_observations_with_embeddings(db)
        candidates: list[tuple[int, int, float]] = []
        for index, (left_obs, left_vector) in enumerate(observations):
            for right_obs, right_vector in observations[index + 1 :]:
                if not _context_differs(left_obs, right_obs):
                    continue
                if get_collision(db, int(left_obs["id"]), int(right_obs["id"])) is not None:
                    continue
                similarity = _cosine_similarity(left_vector, right_vector)
                if self.config.collider_sim_min < similarity < self.config.collider_sim_max:
                    candidates.append((int(left_obs["id"]), int(right_obs["id"]), similarity))
        candidates.sort(key=lambda item: item[2], reverse=True)
        return candidates[:limit]

    def collide(
        self,
        db: sqlite3.Connection,
        obs_a: dict[str, Any],
        obs_b: dict[str, Any],
    ) -> str | None:
        """
        共通キーワードと pattern/anti-pattern の組み合わせから洞察を作る。
        """
        del db
        keywords_a = _extract_keywords(f"{obs_a['title']} {obs_a['content']}")
        keywords_b = _extract_keywords(f"{obs_b['title']} {obs_b['content']}")
        shared = [keyword for keyword in keywords_a if keyword in keywords_b][:5]
        if not shared and obs_a.get("topic_key") != obs_b.get("topic_key"):
            return None
        topic = ", ".join(shared or [str(obs_a["title"]), str(obs_b["title"])])
        if bool(obs_a.get("is_anti_pattern")) != bool(obs_b.get("is_anti_pattern")):
            pattern = obs_a if obs_a.get("is_pattern") else obs_b
            anti = obs_a if obs_a.get("is_anti_pattern") else obs_b
            return (
                f"Collision insight: when working with {topic}, apply '{pattern['title']}' "
                f"to avoid the failure mode in '{anti['title']}'."
            )
        if obs_a.get("topic_key") and obs_a.get("topic_key") == obs_b.get("topic_key"):
            return (
                f"Collision insight: '{obs_a['title']}' and '{obs_b['title']}' describe the same theme "
                f"across different contexts; promote {topic} into a reusable cross-project rule."
            )
        if shared:
            return (
                f"Collision insight: '{obs_a['title']}' and '{obs_b['title']}' converge on {topic}; "
                f"combine the two perspectives the next time this theme appears."
            )
        return None

    def dream_cycle(
        self,
        db: sqlite3.Connection,
        max_collisions: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Dream Cycle: 候補 discovery → collide → collisions 保存を一括実行する。
        """
        results: list[dict[str, Any]] = []
        candidates = self.find_collision_candidates(db, limit=max_collisions)
        for obs_a_id, obs_b_id, similarity in candidates:
            obs_a = _fetch_observation(db, obs_a_id)
            obs_b = _fetch_observation(db, obs_b_id)
            insight = self.collide(db, obs_a, obs_b)
            if insight:
                results.append(save_collision(db, obs_a_id, obs_b_id, insight, similarity))
        return results
