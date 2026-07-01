"""Tests for schema migration v1→v2 and collision metadata persistence."""

from __future__ import annotations

import json
from erinys_memory._sqlite import sqlite3
import sys
import types
from pathlib import Path
from unittest.mock import patch

# Stub fastembed before importing erinys_memory
_fake_fastembed = types.ModuleType("fastembed")


class _FakeTextEmbedding:
    def __init__(self, model_name: str = "test", **kwargs):
        self.model_name = model_name

    def embed(self, texts):
        class V(list):
            def tolist(self):
                return list(self)
        return [V([0.1] * 384) for _ in texts]


_fake_fastembed.TextEmbedding = _FakeTextEmbedding
sys.modules["fastembed"] = _fake_fastembed

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from erinys_memory.db import _migrate  # noqa: E402


def _create_v1_db() -> sqlite3.Connection:
    """Create an in-memory DB with v1 schema (edges without causal/entity/temporal,
    collisions without metadata column)."""
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript("""
        CREATE TABLE observations(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          content TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE edges(
          id          INTEGER PRIMARY KEY AUTOINCREMENT,
          source_id   INTEGER NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
          target_id   INTEGER NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
          relation    TEXT    NOT NULL
                      CHECK(relation IN (
                        'relates_to','depends_on','implements',
                        'references','similar_to','contains',
                        'contradicts','supersedes','distilled_from'
                      )),
          weight      REAL    NOT NULL DEFAULT 1.0
                      CHECK(weight >= 0.0 AND weight <= 1.0),
          metadata    TEXT CHECK(metadata IS NULL OR json_valid(metadata)),
          created_at  DATETIME NOT NULL DEFAULT (datetime('now')),
          UNIQUE(source_id, target_id, relation)
        );
        CREATE INDEX idx_edges_source   ON edges(source_id);
        CREATE INDEX idx_edges_target   ON edges(target_id);
        CREATE INDEX idx_edges_relation ON edges(relation);
        CREATE TABLE collisions(
          id          INTEGER PRIMARY KEY AUTOINCREMENT,
          source_a    INTEGER NOT NULL,
          source_b    INTEGER NOT NULL,
          insight     TEXT    NOT NULL,
          confidence  REAL,
          accepted    BOOLEAN,
          created_at  DATETIME NOT NULL DEFAULT (datetime('now')),
          UNIQUE(source_a, source_b),
          CHECK(source_a < source_b)
        );
        CREATE TABLE schema_version(
          version     INTEGER PRIMARY KEY,
          applied_at  DATETIME NOT NULL DEFAULT (datetime('now'))
        );
        INSERT INTO schema_version(version) VALUES (1);
        INSERT INTO observations(id, content) VALUES (1, 'obs1'), (2, 'obs2');
        INSERT INTO edges(source_id, target_id, relation) VALUES (1, 2, 'relates_to');
    """)
    return con


class TestMigrationV1ToV2:
    """Tests for _migrate() v1→v2."""

    def test_migration_bumps_version(self):
        db = _create_v1_db()
        _migrate(db)
        version = db.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        assert version == 2

    def test_migration_preserves_existing_edges(self):
        db = _create_v1_db()
        _migrate(db)
        count = db.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        assert count == 1
        row = db.execute("SELECT relation FROM edges").fetchone()
        assert row[0] == "relates_to"

    def test_migration_allows_new_relation_types(self):
        db = _create_v1_db()
        _migrate(db)
        for relation in ("causal", "entity", "temporal"):
            db.execute(
                "INSERT INTO edges(source_id, target_id, relation) VALUES (?, ?, ?)",
                (1, 2, relation),
            )
        count = db.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        assert count == 4  # original + 3 new

    def test_migration_adds_collisions_metadata_column(self):
        db = _create_v1_db()
        _migrate(db)
        cols = [row[1] for row in db.execute("PRAGMA table_info(collisions)").fetchall()]
        assert "metadata" in cols

    def test_migration_idempotent(self):
        """Running _migrate twice should not error (already at v2)."""
        db = _create_v1_db()
        _migrate(db)
        _migrate(db)  # Should be a no-op
        version = db.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        assert version == 2


class TestCollisionMetadataRoundTrip:
    """Tests for collision_score metadata persistence."""

    def test_collision_score_persisted_and_retrieved(self):
        db = _create_v1_db()
        _migrate(db)

        score = {"novelty": 0.75, "relevance": 0.82, "serendipity_score": 0.68}
        score_json = json.dumps(score, ensure_ascii=False)

        db.execute(
            "INSERT INTO collisions(source_a, source_b, insight, confidence, metadata) "
            "VALUES (?, ?, ?, ?, ?)",
            (1, 2, "test insight", 0.8, score_json),
        )
        db.commit()

        row = db.execute(
            "SELECT metadata FROM collisions WHERE source_a = 1 AND source_b = 2"
        ).fetchone()
        assert row is not None
        decoded = json.loads(row[0])
        assert decoded["novelty"] == 0.75
        assert decoded["relevance"] == 0.82
        assert decoded["serendipity_score"] == 0.68

    def test_null_metadata_accepted(self):
        db = _create_v1_db()
        _migrate(db)

        db.execute(
            "INSERT INTO collisions(source_a, source_b, insight) VALUES (?, ?, ?)",
            (1, 2, "no score"),
        )
        db.commit()

        row = db.execute(
            "SELECT metadata FROM collisions WHERE source_a = 1 AND source_b = 2"
        ).fetchone()
        assert row[0] is None

    def test_invalid_json_rejected(self):
        db = _create_v1_db()
        _migrate(db)

        import pytest
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO collisions(source_a, source_b, insight, metadata) "
                "VALUES (?, ?, ?, ?)",
                (1, 2, "bad json", "not valid json{{{"),
            )
