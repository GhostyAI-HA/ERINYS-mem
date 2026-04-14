from __future__ import annotations

from pathlib import Path

import erinys_memory.server as server
from erinys_memory.config import ErinysConfig
from erinys_memory.db import get_db


def test_save_observation(db, engine) -> None:
    result = server.erinys_save(title="Save Test", content="alpha beta gamma", project="alpha")
    assert result["ok"] is True, "erinys_save should succeed for a valid observation"
    record = result["data"]["observation"]
    assert record["project"] == "alpha", "saved observation should keep the requested project"
    count = db.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    assert count == 1, "saving one observation should create exactly one row"
    vec_count = db.execute("SELECT COUNT(*) FROM vec_observations").fetchone()[0]
    assert vec_count == 1, "saving one observation should create exactly one vector row"


def test_save_with_topic_key_upsert(db, engine) -> None:
    first = server.erinys_save(
        title="First",
        content="original content",
        project="alpha",
        topic_key="release-plan",
    )
    second = server.erinys_save(
        title="Second",
        content="updated content",
        project="alpha",
        topic_key="release-plan",
    )
    assert first["data"]["id"] == second["data"]["id"], "topic_key upsert should reuse the same observation id"
    assert second["data"]["status"] == "updated", "second save with same topic_key should report updated"
    count = db.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    assert count == 1, "topic_key upsert should keep a single observation row"


def test_update_observation(db, engine) -> None:
    saved = server.erinys_save(title="Mutable", content="old phrase", project="alpha")
    obs_id = saved["data"]["id"]
    updated = server.erinys_update(
        id=obs_id,
        content="new phrase",
        type="pattern",
        metadata={"mode": "D2"},
    )
    assert updated["ok"] is True, "erinys_update should succeed for an existing observation"
    record = updated["data"]["observation"]
    assert record["content"] == "new phrase", "updated observation should expose the new content"
    assert record["is_pattern"] == 1, "changing type to pattern should update inferred pattern flags"
    search = server.erinys_search(query="new", project="alpha")
    assert search["data"]["results"][0]["id"] == obs_id, "updated embedding should make the new content searchable"


def test_delete_cascades(db, engine) -> None:
    left = server.erinys_save(title="Left", content="shared topic", project="alpha", topic_key="topic")
    right = server.erinys_save(title="Right", content="shared topic", project="beta", topic_key="topic")
    server.erinys_link(left["data"]["id"], right["data"]["id"], "relates_to")
    collision = server.erinys_collide(left["data"]["id"], right["data"]["id"])
    assert collision["data"]["created"] is True, "collision setup should create a collision row"
    deleted = server.erinys_delete(left["data"]["id"])
    assert deleted["ok"] is True, "erinys_delete should succeed for an existing observation"
    edge_count = db.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    assert edge_count == 0, "deleting an observation should cascade-delete connected edges"
    collision_count = db.execute("SELECT COUNT(*) FROM collisions").fetchone()[0]
    assert collision_count == 0, "deleting an observation should cascade-delete connected collisions"


def test_search_basic(db, engine) -> None:
    server.erinys_save(title="Relevant", content="vector search alpha token", project="alpha")
    server.erinys_save(title="Noise", content="completely unrelated gardening note", project="alpha")
    result = server.erinys_search(query="alpha", project="alpha")
    titles = [row["title"] for row in result["data"]["results"]]
    assert "Relevant" in titles, "basic search should return the matching observation"
    assert result["data"]["results"][0]["title"] == "Relevant", "the strongest lexical match should rank first here"


def test_search_with_project_filter(db, engine) -> None:
    server.erinys_save(title="Alpha", content="retry transient api failures", project="alpha")
    server.erinys_save(title="Beta", content="retry transient api failures", project="beta")
    result = server.erinys_search(query="retry", project="beta")
    projects = {row["project"] for row in result["data"]["results"]}
    assert projects == {"beta"}, "project filter should exclude observations from other projects"


def test_save_prompt(db, engine) -> None:
    result = server.erinys_save_prompt(content="Investigate lock contention", project="alpha")
    assert result["ok"] is True, "erinys_save_prompt should persist prompt records"
    count = db.execute("SELECT COUNT(*) FROM prompts").fetchone()[0]
    assert count == 1, "saving one prompt should create exactly one prompt row"
    assert result["data"]["prompt"]["project"] == "alpha", "saved prompt should keep the requested project"


def test_recall(db, engine) -> None:
    server.erinys_save(title="Older", content="alpha note", project="alpha")
    server.erinys_save(title="Newer", content="beta note", project="alpha")
    result = server.erinys_recall(project="alpha", limit=2)
    titles = [row["title"] for row in result["data"]["observations"]]
    assert titles == ["Newer", "Older"], "recall should return most recent observations first"
    strength = result["data"]["observations"][0]["effective_strength"]
    assert strength > 0.0, "recall should annotate each observation with an effective strength"


def test_context(db, engine) -> None:
    session = server.erinys_session_start(id="session-core", project="alpha", directory="/tmp/alpha")
    assert session["ok"] is True, "context setup should start a session successfully"
    server.erinys_save(title="Session Note", content="alpha context detail", project="alpha", session_id="session-core")
    result = server.erinys_context(project="alpha", limit=5)
    assert result["ok"] is True, "erinys_context should succeed for an existing project"
    assert result["data"]["sessions"][0]["id"] == "session-core", "context should include recent project sessions"
    assert result["data"]["sessions"][0]["observations"][0]["title"] == "Session Note", "session context should include linked observations"


def test_export_markdown(db, engine) -> None:
    first = server.erinys_save(title="Root", content="export me", project="alpha")
    second = server.erinys_save(title="Leaf", content="linked note", project="alpha")
    server.erinys_link(first["data"]["id"], second["data"]["id"], "references")
    exported = server.erinys_export(project="alpha")
    content = exported["data"]["content"]
    assert "## Root" in content, "markdown export should include section headers for observations"
    assert "Related: [[Leaf]]" in content, "markdown export should render outgoing links as wikilinks"


def test_backup(db, engine, tmp_path) -> None:
    server.erinys_save(title="Backup Source", content="persist me", project="alpha")
    backup_path = tmp_path / "backup.db"
    result = server.erinys_backup(path=str(backup_path))
    assert result["ok"] is True, "erinys_backup should return a successful envelope"
    assert backup_path.exists(), "backup tool should create the requested sqlite file"
    backup_db = get_db(ErinysConfig(db_path=str(backup_path), db_backup_on_init=False))
    try:
        count = backup_db.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        assert count == 1, "backup database should contain the saved observation"
    finally:
        backup_db.close()


def test_stats(db, engine) -> None:
    server.erinys_session_start(id="session-stats", project="alpha")
    left = server.erinys_save(title="Stats A", content="shared stats topic", project="alpha", topic_key="stats")
    right = server.erinys_save(title="Stats B", content="shared stats topic", project="beta", topic_key="stats")
    server.erinys_save_prompt(content="stats prompt", project="alpha")
    server.erinys_link(left["data"]["id"], right["data"]["id"], "relates_to")
    server.erinys_collide(left["data"]["id"], right["data"]["id"])
    stats = server.erinys_stats()
    data = stats["data"]
    assert data["observation_count"] == 2, "stats should count all saved observations"
    assert data["session_count"] == 1, "stats should count persisted sessions"
    assert data["edge_count"] == 1, "stats should count graph edges"
    assert data["collision_count"] == 1, "stats should count saved collisions"
    assert data["prompt_count"] == 1, "stats should count saved prompts"
