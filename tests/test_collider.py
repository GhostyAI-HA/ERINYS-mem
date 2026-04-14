from __future__ import annotations

from conftest import insert_test_observation
from erinys_memory.collider import MemoryCollider, pair_similarity, save_collision
import erinys_memory.server as server


def test_find_candidates(db, engine) -> None:
    first = insert_test_observation(
        db,
        engine,
        "Nightly backups",
        "Store database backups in object storage every night.",
        project="alpha",
        topic_key="backup",
    )
    second = insert_test_observation(
        db,
        engine,
        "Backup protection",
        "Nightly object storage backups protect the database from data loss.",
        project="beta",
        topic_key="backup",
    )
    candidates = MemoryCollider().find_collision_candidates(db)
    pairs = {(left, right) for left, right, _ in candidates}
    assert (int(first["id"]), int(second["id"])) in pairs, "find_collision_candidates should detect similar cross-project memory pairs"


def test_collide_generates_insight(db, engine) -> None:
    first = server.erinys_save(
        title="Nightly backups",
        content="Store database backups in object storage every night.",
        project="alpha",
        topic_key="backup",
    )
    second = server.erinys_save(
        title="Backup protection",
        content="Nightly object storage backups protect the database from data loss.",
        project="beta",
        topic_key="backup",
    )
    result = server.erinys_collide(first["data"]["id"], second["data"]["id"])
    collision = result["data"]["collision"]
    assert result["data"]["created"] is True, "erinys_collide should create a collision when insight generation succeeds"
    assert "Collision insight:" in collision["insight"], "generated collision should include the synthesized insight text"


def test_dream_cycle(db, engine, monkeypatch) -> None:
    first = insert_test_observation(db, engine, "Backup A", "Store database backups nightly.", project="alpha", topic_key="backup")
    second = insert_test_observation(db, engine, "Backup B", "Nightly backups keep database snapshots safe.", project="beta", topic_key="backup")
    third = insert_test_observation(db, engine, "Debug A", "Disable caching while debugging stale responses.", project="gamma", topic_key="debug")
    fourth = insert_test_observation(db, engine, "Debug B", "Turn off caches during stale-response debugging.", project="delta", topic_key="debug")
    monkeypatch.setattr(
        server._COLLIDER,
        "find_collision_candidates",
        lambda db, limit=20: [
            (int(first["id"]), int(second["id"]), 0.88),
            (int(third["id"]), int(fourth["id"]), 0.7),
        ][:limit],
    )
    result = server.erinys_dream(max_collisions=2)
    collisions = result["data"]["collisions"]
    assert len(collisions) == 2, "dream cycle should persist each candidate that produces an insight"


def test_no_duplicate_collision(db, engine) -> None:
    first = server.erinys_save(title="Fact A", content="shared collision topic", project="alpha", topic_key="shared")
    second = server.erinys_save(title="Fact B", content="shared collision topic", project="beta", topic_key="shared")
    created = server.erinys_collide(first["data"]["id"], second["data"]["id"])
    repeated = server.erinys_collide(first["data"]["id"], second["data"]["id"])
    count = db.execute("SELECT COUNT(*) FROM collisions").fetchone()[0]
    assert created["data"]["created"] is True, "first collision call should create a collision row"
    assert repeated["data"]["created"] is False, "second collision call should reuse the existing collision"
    assert count == 1, "collisions table should keep a single row per normalized observation pair"


def test_candidate_sim_range(db, engine) -> None:
    first = insert_test_observation(
        db,
        engine,
        "Nightly backups",
        "Store database backups in object storage every night.",
        project="alpha",
        topic_key="backup",
    )
    second = insert_test_observation(
        db,
        engine,
        "Backup protection",
        "Nightly object storage backups protect the database from data loss.",
        project="beta",
        topic_key="backup",
    )
    similarity = pair_similarity(db, int(first["id"]), int(second["id"]))
    assert 0.65 < similarity < 0.9, "candidate similarity should stay inside the configured collider range"


def test_collision_accept_reject(db, engine) -> None:
    first = insert_test_observation(db, engine, "Accepted", "accept collision", project="alpha")
    second = insert_test_observation(db, engine, "Accepted Pair", "accept collision", project="beta")
    third = insert_test_observation(db, engine, "Rejected", "reject collision", project="gamma")
    fourth = insert_test_observation(db, engine, "Rejected Pair", "reject collision", project="delta")
    accepted = save_collision(db, int(first["id"]), int(second["id"]), "accepted insight", 0.8, accepted=True)
    rejected = save_collision(db, int(third["id"]), int(fourth["id"]), "rejected insight", 0.8, accepted=False)
    assert accepted["accepted"] == 1, "save_collision should persist accepted=True as a truthy database flag"
    assert rejected["accepted"] == 0, "save_collision should persist accepted=False as a falsy database flag"
