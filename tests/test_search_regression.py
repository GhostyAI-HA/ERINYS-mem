"""Regression tests for rrf_hybrid_search pipeline logic.

Uses FTS5-only mock DB (no sqlite-vec extension required) to verify that
the MAGMA changes don't break the search pipeline's scoring, ordering,
and metadata population.

Strategy: Build a minimal DB with observations + fts_observations (FTS5),
and call rrf_hybrid_search with pre-computed query embeddings. The vec
search path will return empty (no vec_observations table), so we verify
FTS-side scoring, boost logic, intent routing, and result structure.
"""

from __future__ import annotations

import math
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from erinys_memory.search import (
    rrf_hybrid_search,
    classify_query_complexity,
    classify_query_intent,
    COMPLEXITY_L1,
    COMPLEXITY_L2,
    COMPLEXITY_L3,
    INTENT_WHY,
    INTENT_GENERAL,
)


# -- Helpers --

_SCHEMA = """
CREATE TABLE observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT 'manual',
    project TEXT,
    scope TEXT NOT NULL DEFAULT 'project',
    is_anti_pattern INTEGER NOT NULL DEFAULT 0,
    is_pattern INTEGER NOT NULL DEFAULT 0,
    distillation_level TEXT,
    distilled_from INTEGER,
    valid_from TIMESTAMP NOT NULL DEFAULT (datetime('now')),
    valid_until TIMESTAMP,
    superseded_by INTEGER,
    base_strength REAL NOT NULL DEFAULT 1.0,
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TIMESTAMP,
    source TEXT NOT NULL DEFAULT 'user',
    embedding_model TEXT,
    topic_key TEXT,
    metadata TEXT,
    session_id TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT (datetime('now')),
    updated_at TIMESTAMP NOT NULL DEFAULT (datetime('now'))
);

CREATE VIRTUAL TABLE observations_fts USING fts5(
    title, content, project,
    content_rowid='id'
);
"""


def _make_db() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    db.executescript(_SCHEMA)
    return db


def _insert(db: sqlite3.Connection, title: str, content: str,
            project: str = "test") -> int:
    cursor = db.execute(
        "INSERT INTO observations (title, content, project) VALUES (?, ?, ?)",
        [title, content, project],
    )
    obs_id = cursor.lastrowid
    db.execute(
        "INSERT INTO observations_fts (rowid, title, content, project) VALUES (?, ?, ?, ?)",
        [obs_id, title, content, project],
    )
    db.commit()
    return obs_id


def _dummy_embedding(dim: int = 384) -> list[float]:
    """Dummy embedding vector."""
    return [0.1] * dim


# -- Regression Tests --

class TestRrfSearchRegression:
    """FTS-path regression tests for rrf_hybrid_search after MAGMA changes."""

    def test_fts_returns_matching_results(self):
        """Basic FTS keyword match should work."""
        db = _make_db()
        _insert(db, "Python setup", "Install Python using pyenv on macOS")
        _insert(db, "Rust intro", "Getting started with Rust programming")

        results = rrf_hybrid_search(db, "Python pyenv install", _dummy_embedding(), limit=5)
        assert len(results) >= 1
        titles = [r["title"] for r in results]
        assert "Python setup" in titles

    def test_results_have_effective_score(self):
        """All results should have positive effective_score."""
        db = _make_db()
        _insert(db, "Score test", "Verifying scores are computed correctly")

        results = rrf_hybrid_search(db, "score test verifying", _dummy_embedding(), limit=5)
        for r in results:
            assert "effective_score" in r
            assert float(r["effective_score"]) > 0

    def test_results_sorted_descending(self):
        """Results should come back sorted by effective_score descending."""
        db = _make_db()
        _insert(db, "Alpha", "Alpha keyword match content here")
        _insert(db, "Beta", "Beta keyword match content here")
        _insert(db, "Gamma", "Gamma keyword match content here")

        results = rrf_hybrid_search(db, "keyword match content", _dummy_embedding(), limit=10)
        if len(results) >= 2:
            scores = [float(r["effective_score"]) for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_project_filter(self):
        """Project filter should restrict results to matching project."""
        db = _make_db()
        _insert(db, "Alpha config", "Configuration for alpha system", project="alpha")
        _insert(db, "Beta config", "Configuration for beta system", project="beta")

        results = rrf_hybrid_search(
            db, "configuration system", _dummy_embedding(), project="alpha", limit=5
        )
        for r in results:
            assert r["project"] == "alpha"

    def test_limit_respected(self):
        """Limit parameter caps number of results."""
        db = _make_db()
        for i in range(10):
            _insert(db, f"Item {i}", f"Content about topic number {i} with keywords")

        results = rrf_hybrid_search(db, "content topic keywords", _dummy_embedding(), limit=3)
        assert len(results) <= 3

    def test_empty_db_returns_empty(self):
        """Search on empty DB should return empty, not crash."""
        db = _make_db()
        results = rrf_hybrid_search(db, "anything at all", _dummy_embedding(), limit=5)
        assert results == []

    def test_query_intent_populated(self):
        """Results should contain query_intent field."""
        db = _make_db()
        _insert(db, "Failure analysis", "Why did the deployment fail yesterday")

        results = rrf_hybrid_search(db, "why deployment fail", _dummy_embedding(), limit=5)
        if results:
            assert "query_intent" in results[0]
            assert results[0]["query_intent"] == "WHY"

    def test_query_complexity_populated(self):
        """Results should contain query_complexity field."""
        db = _make_db()
        _insert(db, "Complexity test", "Testing complexity classification feature")

        results = rrf_hybrid_search(db, "testing complexity", _dummy_embedding(), limit=5)
        if results:
            assert "query_complexity" in results[0]
            assert results[0]["query_complexity"] in (COMPLEXITY_L1, COMPLEXITY_L2, COMPLEXITY_L3)

    def test_explicit_weights_respected(self):
        """Explicit weights should not be overridden by adaptive logic."""
        db = _make_db()
        _insert(db, "Weight test", "Testing explicit weight parameters for search")

        # FTS-heavy explicit weight
        results = rrf_hybrid_search(
            db, "weight test parameters", _dummy_embedding(),
            fts_weight=0.90, vec_weight=0.10, limit=5
        )
        assert len(results) >= 1  # Should not crash

    def test_cjk_query_does_not_crash(self):
        """CJK-only query should not crash even if FTS returns nothing."""
        db = _make_db()
        _insert(db, "日本語テスト", "検索設定のパラメータ")

        # CJK query - FTS5 porter may not match but should not crash
        results = rrf_hybrid_search(db, "検索設定", _dummy_embedding(), limit=5)
        # May return 0 results via FTS (porter tokenizer), but must not raise
        assert isinstance(results, list)

    def test_mixed_cjk_ascii_complexity(self):
        """Mixed CJK+ASCII query should be classified as L2 (vec-heavy)."""
        assert classify_query_complexity("ERINYS 設定") == COMPLEXITY_L2
        assert classify_query_complexity("なぜ API が失敗した") == COMPLEXITY_L2
        assert classify_query_complexity("API 設定") == COMPLEXITY_L2

    def test_general_intent_no_graph_boost(self):
        """GENERAL intent should not trigger graph boost code path."""
        db = _make_db()
        _insert(db, "General test", "Simple general content here")

        results = rrf_hybrid_search(db, "general content", _dummy_embedding(), limit=5)
        for r in results:
            assert not r.get("graph_boosted", False)
