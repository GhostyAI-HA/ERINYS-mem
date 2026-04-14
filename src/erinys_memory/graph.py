"""観測グラフのエッジ管理と BFS 走査を提供する。"""

from __future__ import annotations

import json
import sqlite3
from collections import deque
from typing import Any

from .decay import current_strength

VALID_RELATIONS = {
    "relates_to",
    "depends_on",
    "implements",
    "references",
    "similar_to",
    "contains",
    "contradicts",
    "supersedes",
    "distilled_from",
}


def _decode_json(value: object) -> object:
    if isinstance(value, str) and value:
        return json.loads(value)
    return value


def _normalize_metadata(metadata: dict[str, object] | None) -> str | None:
    return json.dumps(metadata, ensure_ascii=False) if metadata is not None else None


def _ensure_observation_exists(db: sqlite3.Connection, obs_id: int) -> None:
    row = db.execute("SELECT 1 FROM observations WHERE id = ?", [obs_id]).fetchone()
    if row is None:
        raise LookupError(f"observation not found: {obs_id}")


def _edge_record(row: sqlite3.Row) -> dict[str, Any]:
    edge = dict(row)
    edge["metadata"] = _decode_json(edge.get("metadata"))
    edge["effective_weight"] = current_strength(
        float(edge["weight"]),
        edge["created_at"],
        None,
        0,
    )
    return edge


def _observation_summary(db: sqlite3.Connection, obs_id: int) -> dict[str, Any]:
    row = db.execute(
        """
        SELECT id, title, type, project, scope, session_id, created_at, updated_at
        FROM observations
        WHERE id = ?
        """,
        [obs_id],
    ).fetchone()
    if row is None:
        raise LookupError(f"observation not found: {obs_id}")
    return dict(row)


def _relation_clause(relation_filter: list[str] | None) -> tuple[str, list[object]]:
    if not relation_filter:
        return "", []
    placeholders = ", ".join("?" for _ in relation_filter)
    return f" AND relation IN ({placeholders})", list(relation_filter)


def create_edge(
    db: sqlite3.Connection,
    source_id: int,
    target_id: int,
    relation: str,
    weight: float = 1.0,
    metadata: dict[str, object] | None = None,
) -> dict[str, Any]:
    """UNIQUE 制約を利用して重複 edge を upsert する。"""
    if relation not in VALID_RELATIONS:
        raise ValueError(f"invalid relation: {relation}")
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be between 0.0 and 1.0")
    _ensure_observation_exists(db, source_id)
    _ensure_observation_exists(db, target_id)
    db.execute(
        """
        INSERT INTO edges(source_id, target_id, relation, weight, metadata)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(source_id, target_id, relation)
        DO UPDATE SET weight = excluded.weight, metadata = excluded.metadata
        """,
        [source_id, target_id, relation, weight, _normalize_metadata(metadata)],
    )
    db.commit()
    row = db.execute(
        """
        SELECT id, source_id, target_id, relation, weight, metadata, created_at
        FROM edges
        WHERE source_id = ? AND target_id = ? AND relation = ?
        """,
        [source_id, target_id, relation],
    ).fetchone()
    if row is None:
        raise LookupError("edge not found after upsert")
    return _edge_record(row)


def get_edges(
    db: sqlite3.Connection,
    observation_id: int,
    relation_filter: list[str] | None = None,
    direction: str = "outgoing",
) -> list[dict[str, Any]]:
    """指定 observation に接続した edge を取得する。"""
    _ensure_observation_exists(db, observation_id)
    if direction not in {"outgoing", "incoming", "both"}:
        raise ValueError(f"invalid direction: {direction}")
    relation_sql, relation_params = _relation_clause(relation_filter)
    if direction == "outgoing":
        where_sql, params = "source_id = ?", [observation_id]
    elif direction == "incoming":
        where_sql, params = "target_id = ?", [observation_id]
    else:
        where_sql, params = "(source_id = ? OR target_id = ?)", [observation_id, observation_id]
    rows = db.execute(
        f"""
        SELECT id, source_id, target_id, relation, weight, metadata, created_at
        FROM edges
        WHERE {where_sql}{relation_sql}
        ORDER BY created_at DESC, id DESC
        """,
        [*params, *relation_params],
    ).fetchall()
    return [_edge_record(row) for row in rows]


def traverse(
    db: sqlite3.Connection,
    start_id: int,
    max_depth: int = 2,
    relation_filter: list[str] | None = None,
) -> dict[str, Any]:
    """深さ制限つき BFS で outgoing edge を辿る。"""
    _ensure_observation_exists(db, start_id)
    queue: deque[tuple[int, int]] = deque([(start_id, 0)])
    visited = {start_id}
    edge_ids: set[int] = set()
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    while queue:
        current_id, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for edge in get_edges(db, current_id, relation_filter, "outgoing"):
            next_id = int(edge["target_id"])
            if int(edge["id"]) not in edge_ids:
                edge_ids.add(int(edge["id"]))
                edges.append(edge)
            if next_id in visited:
                continue
            visited.add(next_id)
            node = _observation_summary(db, next_id)
            node["depth"] = depth + 1
            nodes.append(node)
            queue.append((next_id, depth + 1))

    start = _observation_summary(db, start_id)
    start["depth"] = 0
    return {"start": start, "nodes": nodes, "edges": edges}


class GraphEngine:
    """server.py から使う薄いラッパー。"""

    def __init__(self, db: sqlite3.Connection) -> None:
        self.db = db

    def create_edge(
        self,
        source_id: int,
        target_id: int,
        relation: str,
        weight: float = 1.0,
        metadata: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        return create_edge(self.db, source_id, target_id, relation, weight, metadata)

    def get_edges(
        self,
        observation_id: int,
        relation_filter: list[str] | None = None,
        direction: str = "outgoing",
    ) -> list[dict[str, Any]]:
        return get_edges(self.db, observation_id, relation_filter, direction)

    def traverse(
        self,
        start_id: int,
        max_depth: int = 2,
        relation_filter: list[str] | None = None,
    ) -> dict[str, Any]:
        return traverse(self.db, start_id, max_depth, relation_filter)
