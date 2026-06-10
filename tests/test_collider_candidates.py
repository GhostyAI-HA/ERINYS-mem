"""find_collision_candidates のベクトル化実装の回帰テスト。"""

from __future__ import annotations

import sqlite3
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from erinys_memory.collider import MemoryCollider, _existing_collision_pairs

from conftest import insert_test_observation


def _blob(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _setup_vec_table(db: sqlite3.Connection) -> None:
    # vec0 拡張なしでも JOIN できる plain table で代替
    db.execute("CREATE TABLE vec_observations (rowid INTEGER PRIMARY KEY, embedding BLOB)")


def _insert_with_embedding(
    db: sqlite3.Connection,
    title: str,
    project: str,
    vector: list[float],
) -> int:
    obs_id = insert_test_observation(db, title=title, content=f"content {title}", project=project)
    db.execute(
        "INSERT INTO vec_observations (rowid, embedding) VALUES (?, ?)",
        [obs_id, _blob(vector)],
    )
    db.commit()
    return obs_id


class TestFindCollisionCandidates:
    def test_band_and_context_filtering(self, mem_db):
        _setup_vec_table(mem_db)
        # A-B: sim=0.8（band内・project異なる）→ 候補
        a = _insert_with_embedding(mem_db, "A", "p1", [1.0, 0.0, 0.0, 0.0])
        b = _insert_with_embedding(mem_db, "B", "p2", [0.8, 0.6, 0.0, 0.0])
        # C: Aと sim=0.95（band外）、Dとは sim=0.76（band内・project異なる）→ 候補
        c = _insert_with_embedding(mem_db, "C", "p3", [0.95, 0.31225, 0.0, 0.0])
        # D: Aと sim=0.8 だが同一project → A-D は除外
        d = _insert_with_embedding(mem_db, "D", "p1", [0.8, 0.0, 0.6, 0.0])

        collider = MemoryCollider()
        candidates = collider.find_collision_candidates(mem_db)
        pairs = {(x, y) for x, y, _ in candidates}
        assert pairs == {(a, b), (c, d)}, f"unexpected candidates: {candidates}"
        # 降順ソート: A-B (0.8) が C-D (0.76) より先
        assert candidates[0][:2] == (a, b)
        assert abs(candidates[0][2] - 0.8) < 1e-4

    def test_existing_collision_excluded(self, mem_db):
        _setup_vec_table(mem_db)
        a = _insert_with_embedding(mem_db, "A", "p1", [1.0, 0.0, 0.0, 0.0])
        b = _insert_with_embedding(mem_db, "B", "p2", [0.8, 0.6, 0.0, 0.0])
        mem_db.execute(
            "INSERT INTO collisions (source_a, source_b, collision_text, similarity) VALUES (?, ?, ?, ?)",
            [min(a, b), max(a, b), "existing", 0.8],
        )
        mem_db.commit()
        assert _existing_collision_pairs(mem_db) == {(min(a, b), max(a, b))}
        collider = MemoryCollider()
        assert collider.find_collision_candidates(mem_db) == []

    def test_empty_db(self, mem_db):
        _setup_vec_table(mem_db)
        collider = MemoryCollider()
        assert collider.find_collision_candidates(mem_db) == []

    def test_zero_vector_not_in_band(self, mem_db):
        _setup_vec_table(mem_db)
        _insert_with_embedding(mem_db, "A", "p1", [1.0, 0.0, 0.0, 0.0])
        _insert_with_embedding(mem_db, "Z", "p2", [0.0, 0.0, 0.0, 0.0])
        collider = MemoryCollider()
        assert collider.find_collision_candidates(mem_db) == []
