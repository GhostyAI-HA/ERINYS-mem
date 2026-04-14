#!/usr/bin/env python3
"""engram + episodic JSONL → ERINYS 一括移行スクリプト"""

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from erinys_memory.config import ErinysConfig
from erinys_memory.db import init_db, insert_observation_with_embedding
from erinys_memory.embedding import EmbeddingEngine, serialize_f32

ENGRAM_DB = Path.home() / ".engram" / "engram.db"
EPISODIC_JSONL = Path(__file__).parent.parent.parent.parent / "framework" / "skills" / "episodic-memory" / "data" / "session_summaries.jsonl"

config = ErinysConfig()
erinys_db = init_db(config)
engine = EmbeddingEngine()

stats = {"engram_obs": 0, "engram_sessions": 0, "episodic": 0, "errors": 0}


def migrate_engram():
    if not ENGRAM_DB.exists():
        print(f"SKIP: engram DB not found at {ENGRAM_DB}")
        return

    src = sqlite3.connect(str(ENGRAM_DB))
    src.text_factory = lambda b: b.decode("utf-8", errors="replace")
    src.row_factory = sqlite3.Row

    # Sessions first
    for row in src.execute("SELECT * FROM sessions ORDER BY started_at ASC"):
        try:
            erinys_db.execute(
                """INSERT OR IGNORE INTO sessions(id, project, directory, started_at, ended_at, summary)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                [row["id"], row["project"], row["directory"] or ".", row["started_at"], row["ended_at"], row["summary"]],
            )
            stats["engram_sessions"] += 1
        except Exception as e:
            print(f"  session error: {e}")
            stats["errors"] += 1
    erinys_db.commit()

    # Observations
    for row in src.execute("SELECT * FROM observations WHERE deleted_at IS NULL ORDER BY id ASC"):
        try:
            obs_type = row["type"] if row["type"] in {
                "manual", "decision", "architecture", "bugfix", "pattern",
                "config", "discovery", "learning", "anti_pattern", "meta_knowledge"
            } else "manual"

            content = row["content"]
            embedding = engine.embed(content)
            blob = serialize_f32(embedding)

            payload = {
                "title": row["title"],
                "content": content,
                "type": obs_type,
                "project": row["project"],
                "scope": row["scope"] or "project",
                "is_anti_pattern": 1 if obs_type == "anti_pattern" else 0,
                "is_pattern": 1 if obs_type == "pattern" else 0,
                "source": "migration",
                "embedding_model": engine.model_name,
                "topic_key": row["topic_key"],
                "metadata": json.dumps({"migrated_from": "engram", "original_id": row["id"]}),
                "session_id": row["session_id"] if row["session_id"] in _session_ids else None,
            }
            insert_observation_with_embedding(erinys_db, payload, blob)
            stats["engram_obs"] += 1
            if stats["engram_obs"] % 50 == 0:
                print(f"  migrated {stats['engram_obs']} observations...")
        except Exception as e:
            print(f"  obs error (id={row['id']}): {e}")
            stats["errors"] += 1

    src.close()


def migrate_episodic():
    if not EPISODIC_JSONL.exists():
        print(f"SKIP: episodic JSONL not found at {EPISODIC_JSONL}")
        return

    for line in EPISODIC_JSONL.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            summary = entry.get("summary", "")
            if not summary:
                continue

            topics = entry.get("topics", [])
            decisions = entry.get("decisions", [])
            session_id = entry.get("session_id", "")
            timestamp = entry.get("timestamp", "")

            content_parts = [summary]
            if topics:
                content_parts.append(f"Topics: {', '.join(topics)}")
            if decisions:
                content_parts.append(f"Decisions: {'; '.join(decisions)}")
            content = "\n".join(content_parts)

            title = summary[:100] + ("..." if len(summary) > 100 else "")

            embedding = engine.embed(content)
            blob = serialize_f32(embedding)

            payload = {
                "title": title,
                "content": content,
                "type": "learning",
                "project": entry.get("project"),
                "scope": "project",
                "is_anti_pattern": 0,
                "is_pattern": 0,
                "source": "migration",
                "embedding_model": engine.model_name,
                "topic_key": None,
                "metadata": json.dumps({
                    "migrated_from": "episodic",
                    "original_session_id": session_id,
                    "original_timestamp": timestamp,
                    "topics": topics,
                    "decisions": decisions,
                }),
                "session_id": None,
            }
            insert_observation_with_embedding(erinys_db, payload, blob)
            stats["episodic"] += 1
            if stats["episodic"] % 50 == 0:
                print(f"  migrated {stats['episodic']} episodic entries...")
        except Exception as e:
            print(f"  episodic error: {e}")
            stats["errors"] += 1


# Collect existing session IDs for FK validation
_session_ids = set()
if ENGRAM_DB.exists():
    _src = sqlite3.connect(str(ENGRAM_DB))
    _src.text_factory = lambda b: b.decode("utf-8", errors="replace")
    _session_ids = {row[0] for row in _src.execute("SELECT id FROM sessions")}
    _src.close()

print("=== ERINYS Migration ===")
print(f"engram DB: {ENGRAM_DB} (exists={ENGRAM_DB.exists()})")
print(f"episodic JSONL: {EPISODIC_JSONL} (exists={EPISODIC_JSONL.exists()})")
print()

print("--- Migrating engram ---")
migrate_engram()
print(f"  Done: {stats['engram_obs']} observations, {stats['engram_sessions']} sessions")

print("--- Migrating episodic ---")
migrate_episodic()
print(f"  Done: {stats['episodic']} entries")

print()
print(f"=== Migration Complete ===")
print(f"  engram: {stats['engram_obs']} obs + {stats['engram_sessions']} sessions")
print(f"  episodic: {stats['episodic']} entries")
print(f"  errors: {stats['errors']}")

# Verify
total = erinys_db.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
vec_total = erinys_db.execute("SELECT COUNT(*) FROM vec_observations").fetchone()[0]
print(f"  ERINYS total: {total} observations, {vec_total} vectors")
print(f"  Sync OK: {total == vec_total}")
