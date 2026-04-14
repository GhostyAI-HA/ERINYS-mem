"""FTS5 と sqlite-vec を RRF で統合する検索ロジック。"""

from __future__ import annotations

from datetime import datetime
import sqlite3
from typing import Any, Sequence, cast

from .decay import current_strength
from .embedding import serialize_f32


def rrf_hybrid_search(
    db: sqlite3.Connection,
    query: str,
    query_embedding: list[float],
    project: str | None = None,
    metadata_filter: dict[str, object] | None = None,
    limit: int = 10,
    k: int = 60,
    fts_weight: float = 0.4,
    vec_weight: float = 0.6,
) -> list[dict[str, Any]]:
    """
    FTS5 keyword search + sqlite-vec vector search → RRF fusion.

    RRF score = Σ (weight / (k + rank))

    Project filter — adaptive widening:
    - FTS5: project条件をMATCH句に含めるため、候補段階でproject絞り込み済み
    - sqlite-vec: project列を持たないため、iterative widening で対応
      初期k=limit*5 → in-project候補不足ならk*2に拡大 → 繰り返し
      終了条件: in_project >= limit OR vec候補が枯渇 (returned < requested)
      上限: VEC_MAX_K (10000) で安全停止
    """
    MAX_SEARCH_LIMIT = 500
    VEC_INITIAL_MULTIPLIER = 5
    VEC_MAX_K = 10000
    limit = min(limit, MAX_SEARCH_LIMIT)
    fts_match = sanitize_fts(query)
    if project:
        safe_project = project.replace('"', "").strip()
        if safe_project:
            fts_match += f' AND project:"{safe_project}"'
    fts_sql = """
        SELECT rowid,
               row_number() OVER (ORDER BY rank) as rank_num
        FROM observations_fts
        WHERE observations_fts MATCH ?
        LIMIT ?
    """
    fts_rows = db.execute(
        fts_sql, [fts_match, min(limit * VEC_INITIAL_MULTIPLIER, VEC_MAX_K)]
    ).fetchall()

    def fetch_vec_rows(vec_limit: int) -> list[sqlite3.Row]:
        vec_sql = """
            SELECT rowid,
                   row_number() OVER (ORDER BY distance) as rank_num
            FROM vec_observations
            WHERE embedding MATCH ?
              AND k = ?
        """
        return db.execute(
            vec_sql, [serialize_f32(query_embedding), vec_limit]
        ).fetchall()

    def compute_rrf(
        fts: Sequence[sqlite3.Row], vec: Sequence[sqlite3.Row]
    ) -> dict[int, float]:
        scores: dict[int, float] = {}
        for rowid, rank_num in fts:
            scores[rowid] = scores.get(rowid, 0.0) + fts_weight / (k + rank_num)
        for rowid, rank_num in vec:
            scores[rowid] = scores.get(rowid, 0.0) + vec_weight / (k + rank_num)
        return scores

    vec_k = min(limit * VEC_INITIAL_MULTIPLIER, VEC_MAX_K)
    vec_rows = fetch_vec_rows(vec_k)
    scores = compute_rrf(fts_rows, vec_rows)
    if project:
        while vec_k < VEC_MAX_K:
            ranked_ids = sorted(scores, key=scores.__getitem__, reverse=True)
            in_project_count = sum(
                1
                for rid in ranked_ids
                if db.execute(
                    "SELECT 1 FROM observations WHERE id = ? AND project = ?",
                    [rid, project],
                ).fetchone()
            )
            if in_project_count >= limit:
                break
            if len(vec_rows) < vec_k:
                break
            vec_k = min(vec_k * 2, VEC_MAX_K)
            vec_rows = fetch_vec_rows(vec_k)
            scores = compute_rrf(fts_rows, vec_rows)

    ranked_ids = sorted(scores, key=scores.__getitem__, reverse=True)
    if not ranked_ids:
        return []
    db.execute("CREATE TEMP TABLE IF NOT EXISTS _rrf_ranked(id INTEGER, ord INTEGER)")
    db.execute("DELETE FROM _rrf_ranked")
    db.executemany(
        "INSERT INTO _rrf_ranked(id, ord) VALUES (?, ?)",
        [(obs_id, ord_num) for ord_num, obs_id in enumerate(ranked_ids)],
    )
    where_clauses = ["1 = 1"]
    params: list[object] = []
    if project:
        where_clauses.append("o.project = ?")
        params.append(project)
    if metadata_filter:
        for key, value in metadata_filter.items():
            where_clauses.append("json_extract(o.metadata, ?) = ?")
            params.extend([f"$.{key}", value])
    join_sql = f"""
        SELECT o.*, r.ord
        FROM _rrf_ranked r
        JOIN observations o ON o.id = r.id
        WHERE {" AND ".join(where_clauses)}
        ORDER BY r.ord
    """
    fused_rows = db.execute(join_sql, params).fetchall()
    results: list[dict[str, Any]] = []
    for row in fused_rows:
        obs = dict(row)
        effective_strength = current_strength(
            float(obs["base_strength"]),
            cast(datetime, obs["created_at"]),
            cast(datetime | None, obs["last_accessed"]),
            int(obs["access_count"]),
        )
        obs_id = int(obs["id"])
        obs["rrf_score"] = scores[obs_id]
        obs["effective_strength"] = effective_strength
        obs["effective_score"] = scores[obs_id] * effective_strength
        results.append(obs)
    results.sort(key=lambda item: float(item["effective_score"]), reverse=True)
    return results[:limit]


def sanitize_fts(query: str) -> str:
    """
    FTS5 MATCH構文のために入力を正規化する。
    空文字は reject し、各トークンからダブルクォートを除去する。
    """
    raw_tokens = query.strip().split()
    tokens = [token.replace('"', "").strip() for token in raw_tokens]
    tokens = [token for token in tokens if token]

    if not tokens:
        raise ValueError("FTS query must not be empty")

    return " ".join(f'"{token}"' for token in tokens)
