"""異なる文脈の記憶を衝突させて新しい洞察を生成する。"""

from __future__ import annotations

import json
import logging
import math
from ._sqlite import sqlite3
import struct
from typing import Any

import numpy as np

from .config import ErinysConfig
from .db import embedding_engine

logger = logging.getLogger(__name__)

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


def _compute_collision_score(
    source_a_content: str,
    source_b_content: str,
    collision_content: str,
    similarity: float,
) -> dict[str, float]:
    """Score a dream collision's usefulness.

    Returns dict with:
    - novelty: fraction of collision keywords NOT in sources (0-1)
    - relevance: embedding similarity of collision to combined sources (0-1)
    - serendipity_score: novelty * (1 - similarity) + relevance * similarity
    """
    collision_keywords = _extract_keywords(collision_content)
    if collision_keywords:
        source_keywords = set(_extract_keywords(source_a_content + " " + source_b_content))
        novel_count = sum(1 for kw in collision_keywords if kw not in source_keywords)
        novelty = novel_count / len(collision_keywords)
    else:
        novelty = 0.0

    combined_source = source_a_content + " " + source_b_content
    combined_embedding = embedding_engine.embed(combined_source)
    collision_embedding = embedding_engine.embed(collision_content)
    relevance = max(0.0, _cosine_similarity(combined_embedding, collision_embedding))

    # W8: Clamp inputs to [0, 1] for safety when reused outside bounded search
    similarity = max(0.0, min(similarity, 1.0))
    novelty = max(0.0, min(novelty, 1.0))
    relevance = max(0.0, min(relevance, 1.0))

    serendipity_score = novelty * (1.0 - similarity) + relevance * similarity

    return {
        "novelty": round(novelty, 4),
        "relevance": round(relevance, 4),
        "serendipity_score": round(serendipity_score, 4),
    }


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


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0.0] = 1.0  # zero ベクトルは similarity 0 扱い（band 外）
    return matrix / norms[:, None]


def _fetch_observations_with_embeddings(
    db: sqlite3.Connection,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """全 observation と正規化済み embedding 行列を返す。"""
    rows = db.execute(
        """
        SELECT o.*, v.embedding
        FROM observations o
        JOIN vec_observations v ON v.rowid = o.id
        ORDER BY o.id ASC
        """
    ).fetchall()
    records = [_observation_record(row) for row in rows]
    if not rows:
        return records, np.empty((0, 0), dtype=np.float32)
    matrix = np.vstack([np.frombuffer(bytes(row["embedding"]), dtype=np.float32) for row in rows])
    return records, _normalize_rows(matrix)


def _existing_collision_pairs(db: sqlite3.Connection) -> set[tuple[int, int]]:
    """既存 collision ペアを1クエリで取得する（ペア毎の照会を回避）。"""
    rows = db.execute("SELECT source_a, source_b FROM collisions").fetchall()
    return {(int(row[0]), int(row[1])) for row in rows}


def get_collision(
    db: sqlite3.Connection,
    source_a: int,
    source_b: int,
) -> dict[str, Any] | None:
    """正規化済みペアで collision を取得する。"""
    normalized_a, normalized_b = _normalize_pair(source_a, source_b)
    row = db.execute(
        """
        SELECT id, source_a, source_b, insight, confidence, accepted, metadata, created_at
        FROM collisions
        WHERE source_a = ? AND source_b = ?
        """,
        [normalized_a, normalized_b],
    ).fetchone()
    if row is None:
        return None
    record = dict(row)
    # Decode collision_score from metadata JSON
    if record.get("metadata") and isinstance(record["metadata"], str):
        try:
            record["collision_score"] = json.loads(record["metadata"])
        except (json.JSONDecodeError, TypeError):
            pass
    return record


def save_collision(
    db: sqlite3.Connection,
    source_a: int,
    source_b: int,
    insight: str,
    confidence: float | None,
    accepted: bool | None = None,
    collision_score: dict[str, float] | None = None,
) -> dict[str, Any]:
    """collision を正規化して保存する。score が未指定なら自動計算する。"""
    normalized_a, normalized_b = _normalize_pair(source_a, source_b)

    # W4: Auto-compute collision_score if not provided (consistent across entry points)
    if collision_score is None and confidence is not None:
        try:
            obs_a = _fetch_observation(db, normalized_a)
            obs_b = _fetch_observation(db, normalized_b)
            collision_score = _compute_collision_score(
                str(obs_a["content"]),
                str(obs_b["content"]),
                insight,
                confidence,
            )
        except (LookupError, Exception):
            pass  # Degrade gracefully if observations unavailable

    # W3: Persist collision_score in metadata JSON column
    score_json = json.dumps(collision_score, ensure_ascii=False) if collision_score else None
    db.execute(
        """
        INSERT INTO collisions(source_a, source_b, insight, confidence, accepted, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [normalized_a, normalized_b, insight, confidence, accepted, score_json],
    )
    db.commit()
    collision = get_collision(db, normalized_a, normalized_b)
    if collision is None:
        raise LookupError("collision not found after insert")
    if collision_score is not None:
        collision["collision_score"] = collision_score
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

    # index 行と後続行の類似度を一括計算し、衝突条件を満たすペアを返す。
    def _band_candidates_for_row(
        self,
        records: list[dict[str, Any]],
        normalized: np.ndarray,
        index: int,
        existing: set[tuple[int, int]],
    ) -> list[tuple[int, int, float]]:
        left_obs = records[index]
        sims = normalized[index + 1 :] @ normalized[index]
        band = np.nonzero(
            (sims > self.config.collider_sim_min) & (sims < self.config.collider_sim_max)
        )[0]
        found: list[tuple[int, int, float]] = []
        for offset in band:
            right_obs = records[index + 1 + int(offset)]
            if not _context_differs(left_obs, right_obs):
                continue
            pair = _normalize_pair(int(left_obs["id"]), int(right_obs["id"]))
            if pair in existing:
                continue
            found.append((int(left_obs["id"]), int(right_obs["id"]), float(sims[int(offset)])))
        return found

    def find_collision_candidates(
        self,
        db: sqlite3.Connection,
        limit: int = 20,
    ) -> list[tuple[int, int, float]]:
        """全 observation のペアから衝突候補を見つける（numpy ベクトル化）。"""
        records, normalized = _fetch_observations_with_embeddings(db)
        if len(records) < 2:
            return []
        existing = _existing_collision_pairs(db)
        candidates: list[tuple[int, int, float]] = []
        for index in range(len(records) - 1):
            candidates.extend(
                self._band_candidates_for_row(records, normalized, index, existing)
            )
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
                score = _compute_collision_score(
                    str(obs_a["content"]),
                    str(obs_b["content"]),
                    insight,
                    similarity,
                )
                results.append(
                    save_collision(
                        db, obs_a_id, obs_b_id, insight, similarity,
                        collision_score=score,
                    )
                )
        return results
