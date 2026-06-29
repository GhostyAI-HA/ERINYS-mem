"""Tests for distill.py: distillation quality scoring."""

from __future__ import annotations

import math
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from erinys_memory.db import resolve_session_id
from erinys_memory.distill import _compute_distillation_quality


class TestComputeDistillationQuality:
    """Tests for _compute_distillation_quality (#128/#108)."""

    def _make_embedding(self, values: list[float]) -> list[float]:
        """Create a simple embedding vector."""
        return values

    def test_identical_content_high_quality(self):
        emb = [1.0, 0.0, 0.5]
        result = _compute_distillation_quality(
            "ERINYS memory search configuration details",
            "ERINYS memory search configuration details",
            emb, emb,
        )
        assert result["semantic_preservation"] > 0.99
        assert result["keyword_retention"] > 0.99
        assert result["quality_score"] > 0.7

    def test_completely_different_low_quality(self):
        emb_source = [1.0, 0.0, 0.0]
        emb_distilled = [0.0, 1.0, 0.0]
        result = _compute_distillation_quality(
            "ERINYS memory search configuration",
            "completely unrelated random words",
            emb_source, emb_distilled,
        )
        assert result["semantic_preservation"] < 0.1
        assert result["keyword_retention"] < 0.5
        assert result["quality_score"] < 0.7

    def test_compression_ratio(self):
        emb = [1.0, 0.5, 0.3]
        result = _compute_distillation_quality(
            "a" * 100,  # long source
            "a" * 50,   # shorter distilled
            emb, emb,
        )
        assert abs(result["compression_ratio"] - 0.5) < 0.01

    def test_zero_length_source(self):
        emb = [1.0, 0.0]
        result = _compute_distillation_quality("", "distilled", emb, emb)
        # Should handle gracefully
        assert "quality_score" in result

    def test_zero_vectors(self):
        result = _compute_distillation_quality(
            "source content here",
            "distilled version",
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        )
        assert result["semantic_preservation"] == 0.0
        assert "quality_score" in result

    def test_quality_score_bounded(self):
        emb = [0.5, 0.5, 0.5]
        result = _compute_distillation_quality(
            "some content", "some content", emb, emb,
        )
        assert 0.0 <= result["quality_score"] <= 1.0

    def test_return_keys(self):
        emb = [1.0]
        result = _compute_distillation_quality("a", "b", emb, emb)
        expected_keys = {
            "semantic_preservation",
            "keyword_retention",
            "compression_ratio",
            "compression_score",
            "quality_score",
        }
        assert expected_keys.issubset(set(result.keys()))


class TestResolveSessionId:
    """resolve_session_id: 孤児 session_id の蒸留FK違反回帰 (2026-06-11).

    セッション一括削除後、観測の session_id が sessions に存在しない状態が残る。
    その値を蒸留・supersede の新規 INSERT にコピーすると FOREIGN KEY 違反になる。
    """

    def _insert_orphan_observation(self, db: sqlite3.Connection) -> str:
        """session を削除して孤児 session_id を持つ観測を作る。"""
        db.execute("INSERT INTO sessions (id, project) VALUES ('s-gone', 'p')")
        db.execute(
            "INSERT INTO observations (title, content, session_id)"
            " VALUES ('src', 'content', 's-gone')"
        )
        db.commit()
        db.execute("PRAGMA foreign_keys = OFF")
        db.execute("DELETE FROM sessions WHERE id = 's-gone'")
        db.commit()
        db.execute("PRAGMA foreign_keys = ON")
        return "s-gone"

    def test_existing_session_passthrough(self, mem_db):
        mem_db.execute("INSERT INTO sessions (id, project) VALUES ('s1', 'p')")
        assert resolve_session_id(mem_db, "s1") == "s1"

    def test_none_returns_none(self, mem_db):
        assert resolve_session_id(mem_db, None) is None

    def test_orphan_session_returns_none(self, mem_db):
        orphan_sid = self._insert_orphan_observation(mem_db)
        assert resolve_session_id(mem_db, orphan_sid) is None

    def test_orphan_copy_insert_fails_without_resolve(self, mem_db):
        """元バグの再現: 孤児 session_id をそのままコピーすると FK 違反。"""
        orphan_sid = self._insert_orphan_observation(mem_db)
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            mem_db.execute(
                "INSERT INTO observations (title, content, session_id)"
                " VALUES ('distilled', 'c', ?)",
                [orphan_sid],
            )

    def test_resolve_unblocks_insert(self, mem_db):
        orphan_sid = self._insert_orphan_observation(mem_db)
        mem_db.execute(
            "INSERT INTO observations (title, content, session_id)"
            " VALUES ('distilled', 'c', ?)",
            [resolve_session_id(mem_db, orphan_sid)],
        )
        row = mem_db.execute(
            "SELECT session_id FROM observations WHERE title = 'distilled'"
        ).fetchone()
        assert row["session_id"] is None


class TestQualityGate:
    """SSGM quality gate 配線 (#147, 2026-06-22): 低品質蒸留を可逆フラグする。"""

    def test_low_quality_sets_flag(self):
        from erinys_memory.distill import _apply_quality_gate
        md: dict = {}
        _apply_quality_gate({"title": "X"}, "meta", "llm", {"quality_score": 0.30}, md)
        assert md["quality_gate"] == {"status": "low", "score": 0.30}

    def test_high_quality_no_flag(self):
        from erinys_memory.distill import _apply_quality_gate
        md: dict = {}
        _apply_quality_gate({"title": "Y"}, "meta", "llm", {"quality_score": 0.80}, md)
        assert "quality_gate" not in md

    def test_threshold_boundary_no_flag(self):
        from erinys_memory.distill import _apply_quality_gate, QUALITY_GATE_THRESHOLD
        md: dict = {}
        _apply_quality_gate({"title": "Z"}, "meta", "llm", {"quality_score": QUALITY_GATE_THRESHOLD}, md)
        assert "quality_gate" not in md

    def test_gate_is_wired_into_consolidation(self):
        """配線漏れ防止: consolidate パスが実際に gate を呼ぶことを保証する。"""
        import inspect
        from erinys_memory.distill import _create_distillation_record
        assert "_apply_quality_gate(" in inspect.getsource(_create_distillation_record)

    def test_stale_flag_cleared_on_good_quality(self):
        """Codex P2: 源泉から継承した古い low フラグを高品質時に消す。"""
        from erinys_memory.distill import _apply_quality_gate
        md: dict = {"quality_gate": {"status": "low", "score": 0.20}}
        _apply_quality_gate({"title": "X"}, "meta", "llm", {"quality_score": 0.90}, md)
        assert "quality_gate" not in md
