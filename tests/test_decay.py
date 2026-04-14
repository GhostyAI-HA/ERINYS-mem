from __future__ import annotations

from datetime import datetime, timedelta, timezone

from conftest import insert_test_observation
from erinys_memory.decay import current_strength, should_prune
import erinys_memory.server as server


def test_initial_strength_is_one(db, engine) -> None:
    record = insert_test_observation(db, engine, "Fresh", "fresh memory")
    strength = current_strength(1.0, record["created_at"], None, 0, now=record["created_at"])
    assert strength == 1.0, "a newly created observation should start with effective strength 1.0"


def test_decay_over_time(db, engine) -> None:
    created_at = datetime.now(timezone.utc) - timedelta(days=30)
    strength = current_strength(1.0, created_at, None, 0, now=datetime.now(timezone.utc))
    assert strength < 1.0, "effective strength should decay as time passes"


def test_reinforce_boosts_strength(db, engine) -> None:
    created_at = datetime.now(timezone.utc)
    baseline = current_strength(1.0, created_at, None, 0, now=created_at)
    reinforced = current_strength(1.0, created_at, None, 3, now=created_at)
    assert reinforced > baseline, "higher access_count should boost effective strength"


def test_prune_threshold(db, engine) -> None:
    assert should_prune(0.09) is True, "strength below the prune threshold should be prunable"
    assert should_prune(0.1) is False, "strength at the prune threshold should not be prunable"


def test_prune_candidates(db, engine) -> None:
    record = insert_test_observation(db, engine, "Stale", "stale memory")
    stale_at = datetime.now(timezone.utc) - timedelta(days=400)
    db.execute(
        "UPDATE observations SET created_at = ?, base_strength = ? WHERE id = ?",
        [stale_at, 0.2, record["id"]],
    )
    db.commit()
    result = server.erinys_prune(threshold=0.1, dry_run=True)
    candidate_ids = {row["id"] for row in result["data"]["candidates"]}
    assert int(record["id"]) in candidate_ids, "dry-run prune should identify stale observations as candidates"


def test_reinforce_on_access(db, engine) -> None:
    record = insert_test_observation(db, engine, "Accessed", "accessed memory")
    result = server.erinys_reinforce(int(record["id"]))
    observation = result["data"]["observation"]
    assert observation["access_count"] == 1, "reinforce should increment access_count"
    assert observation["last_accessed"] is not None, "reinforce should stamp last_accessed"


def test_frequently_accessed_survives(db, engine) -> None:
    created_at = datetime.now(timezone.utc) - timedelta(days=180)
    weak = current_strength(1.0, created_at, None, 0, now=datetime.now(timezone.utc))
    strong = current_strength(1.0, created_at, None, 6, now=datetime.now(timezone.utc))
    assert strong > weak, "frequently accessed memories should decay more slowly in practice"


def test_prune_execute(db, engine) -> None:
    record = insert_test_observation(db, engine, "Disposable", "obsolete memory")
    stale_at = datetime.now(timezone.utc) - timedelta(days=500)
    db.execute(
        "UPDATE observations SET created_at = ?, base_strength = ? WHERE id = ?",
        [stale_at, 0.1, record["id"]],
    )
    db.commit()
    result = server.erinys_prune(threshold=0.1, dry_run=False)
    deleted_ids = set(result["data"]["deleted_ids"])
    assert int(record["id"]) in deleted_ids, "execute prune should delete stale candidates"


def test_strength_cap(db, engine) -> None:
    now = datetime.now(timezone.utc)
    strength = current_strength(1.9, now, None, 10, now=now)
    assert strength <= 2.0, "effective strength should never exceed the configured cap of 2.0"
