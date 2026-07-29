"""Run-scoped backfill rollback tests."""

from __future__ import annotations

import json

import pytest

from erinys_memory.backfill import BackfillError, BackfillService
from erinys_memory.backfill_contracts import BackfillRequest


def item(
    topic: str,
    event: str,
    recorded_at: str = "2024-01-01T00:00:00Z",
    **overrides: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "topic_key": topic,
        "title": topic,
        "content": f"content for {topic}",
        "project": "Project",
        "memory_role": "CLAIM",
        "claim_status": "ACTIVE",
        "source": {
            "kind": "slack",
            "namespace": "slack:workspace-1",
            "event_key": event,
            "recorded_at": recorded_at,
        },
        "use_events": [],
    }
    payload.update(overrides)
    return payload


def request(
    items: list[dict[str, object]],
    run_key: str,
    plan_digest: str | None = None,
) -> BackfillRequest:
    payload: dict[str, object] = {"items": items, "run_key": run_key}
    if plan_digest is not None:
        payload.update(mode="apply", expected_plan_digest=plan_digest)
    return BackfillRequest.model_validate(payload)


def apply(
    service: BackfillService,
    items: list[dict[str, object]],
    run_key: str,
) -> dict[str, object]:
    plan = service.plan(request(items, run_key))
    return service.apply(request(items, run_key, str(plan["plan_digest"])))


def test_created_observation_rolls_back_cleanly(
    backfill_db,
    fake_embed_batch,
) -> None:
    db, config = backfill_db
    service = BackfillService(db, config, fake_embed_batch)
    applied = apply(service, [item("topic-1", "record-1")], "run-created")

    dry_run = service.rollback(str(applied["run_id"]))
    executed = service.rollback(str(applied["run_id"]), dry_run=False)

    assert dry_run["status"] == "READY"
    assert executed["status"] == "ROLLED_BACK"
    assert executed["vector_health"] == {"missing": 0, "orphan": 0}
    assert executed["membership"]["complete"] is True
    assert set(executed["membership"]["residual"].values()) == {0}
    assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM observation_events").fetchone()[0] == 0


def test_matched_run_rollback_restores_projection(
    backfill_db,
    fake_embed_batch,
) -> None:
    db, config = backfill_db
    service = BackfillService(db, config, fake_embed_batch)
    base = item(
        "topic-1",
        "record-1",
        use_events=[
            {"event_key": "use-1", "occurred_at": "2024-02-01T00:00:00Z"},
        ],
    )
    apply(service, [base], "run-base")
    extended = item(
        "topic-1",
        "record-1",
        use_events=[
            {"event_key": "use-1", "occurred_at": "2024-02-01T00:00:00Z"},
            {"event_key": "use-2", "occurred_at": "2024-03-01T00:00:00Z"},
        ],
    )
    second = apply(service, [extended], "run-second")

    service.rollback(str(second["run_id"]), dry_run=False)
    row = db.execute(
        "SELECT access_count, last_accessed FROM observations"
    ).fetchone()

    assert row["access_count"] == 1
    assert row["last_accessed"].isoformat() == "2024-02-01T00:00:00+00:00"


def test_later_run_blocks_original_created_rollback(
    backfill_db,
    fake_embed_batch,
) -> None:
    db, config = backfill_db
    service = BackfillService(db, config, fake_embed_batch)
    base = item("topic-1", "record-1")
    first = apply(service, [base], "run-first")
    used = item(
        "topic-1",
        "record-1",
        use_events=[
            {"event_key": "use-later", "occurred_at": "2024-04-01T00:00:00Z"},
        ],
    )
    apply(service, [used], "run-later")

    dry_run = service.rollback(str(first["run_id"]))
    with pytest.raises(BackfillError) as exc:
        service.rollback(str(first["run_id"]), dry_run=False)

    assert dry_run["status"] == "BLOCKED"
    assert exc.value.code == "ROLLBACK_DEPENDENCY"


def test_later_projection_update_blocks_created_rollback(
    backfill_db,
    fake_embed_batch,
) -> None:
    db, config = backfill_db
    service = BackfillService(db, config, fake_embed_batch)
    applied = apply(service, [item("topic-1", "record-1")], "run-projection")
    db.execute(
        """
        UPDATE observations
        SET access_count = 1, last_accessed = ?, updated_at = ?
        WHERE topic_key = ?
        """,
        [
            "2025-01-01T00:00:00+00:00",
            "2025-01-01T00:00:00+00:00",
            "topic-1",
        ],
    )
    db.commit()

    result = service.rollback(str(applied["run_id"]))

    assert result["status"] == "BLOCKED"
    assert result["blocked_observations"] == [
        applied["items"][0]["observation_id"],
    ]


def test_external_edge_blocks_created_rollback(
    backfill_db,
    fake_embed_batch,
) -> None:
    db, config = backfill_db
    service = BackfillService(db, config, fake_embed_batch)
    first = apply(service, [item("topic-1", "record-1")], "run-edge")
    second = apply(service, [item("topic-2", "record-2")], "run-peer")
    db.execute(
        """
        INSERT INTO edges(source_id, target_id, relation, metadata)
        VALUES (?, ?, 'relates_to', ?)
        """,
        [
            first["items"][0]["observation_id"],
            second["items"][0]["observation_id"],
            json.dumps({"manual": True}),
        ],
    )
    db.commit()

    result = service.rollback(str(first["run_id"]))

    assert result["status"] == "BLOCKED"
    assert result["blocked_observations"] == [
        first["items"][0]["observation_id"],
    ]


def test_provenance_child_blocks_created_rollback(
    backfill_db,
    fake_embed_batch,
) -> None:
    db, config = backfill_db
    service = BackfillService(db, config, fake_embed_batch)
    applied = apply(service, [item("topic-1", "record-1")], "run-parent")
    parent_id = int(applied["items"][0]["observation_id"])
    db.execute(
        """
        INSERT INTO observations(title, content, project, metadata)
        VALUES (?, ?, ?, ?)
        """,
        [
            "derived", "derived content", "Project",
            json.dumps({"provenance": {"parents": [parent_id]}}),
        ],
    )
    db.commit()

    result = service.rollback(str(applied["run_id"]))

    assert result["status"] == "BLOCKED"
    assert result["blocked_observations"] == [parent_id]


def test_supersede_edge_and_temporal_closure_are_reversible(
    backfill_db,
    fake_embed_batch,
) -> None:
    db, config = backfill_db
    service = BackfillService(db, config, fake_embed_batch)
    old = item("old-claim", "old-record")
    apply(service, [old], "run-old")
    new = item(
        "new-claim",
        "new-record",
        recorded_at="2025-01-01T00:00:00Z",
        supersedes_topic_key="old-claim",
    )
    applied = apply(service, [new], "run-supersede")
    old_row = db.execute(
        "SELECT * FROM observations WHERE topic_key = 'old-claim'"
    ).fetchone()
    old_metadata = json.loads(old_row["metadata"])

    assert old_row["valid_until"].isoformat() == "2025-01-01T00:00:00+00:00"
    assert old_metadata["claim_status"] == "SUPERSEDED"
    assert db.execute(
        "SELECT COUNT(*) FROM edges WHERE relation = 'supersedes'"
    ).fetchone()[0] == 1

    service.rollback(str(applied["run_id"]), dry_run=False)
    restored = db.execute(
        "SELECT * FROM observations WHERE topic_key = 'old-claim'"
    ).fetchone()

    assert restored["valid_until"] is None
    assert restored["superseded_by"] is None
    assert json.loads(restored["metadata"])["claim_status"] == "ACTIVE"
    assert db.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 1


def test_temporal_repair_records_and_restores_projection(
    backfill_db,
    fake_embed_batch,
) -> None:
    db, config = backfill_db
    service = BackfillService(db, config, fake_embed_batch)
    source = item("topic-1", "record-1")
    apply(service, [source], "run-original")
    db.execute(
        """
        UPDATE observations
        SET created_at = ?, valid_from = ?, valid_until = ?
        WHERE topic_key = ?
        """,
        [
            "2024-01-02T00:00:00+00:00",
            "2024-01-03T00:00:00+00:00",
            "2024-12-31T00:00:00+00:00",
            "topic-1",
        ],
    )
    db.commit()
    payload = {
        "items": [source],
        "run_key": "run-repair",
        "repair_temporal_projection": True,
    }
    planned_request = BackfillRequest.model_validate(payload)
    plan = service.plan(planned_request)
    payload.update(mode="apply", expected_plan_digest=plan["plan_digest"])
    applied = service.apply(BackfillRequest.model_validate(payload))

    repaired = db.execute(
        "SELECT created_at, valid_from, valid_until FROM observations"
    ).fetchone()
    assert plan["items"][0]["status"] == "MATCH_REPAIR"
    assert repaired["created_at"].isoformat() == "2024-01-01T00:00:00+00:00"
    assert repaired["valid_from"].isoformat() == "2024-01-01T00:00:00+00:00"
    assert repaired["valid_until"] is None

    service.rollback(str(applied["run_id"]), dry_run=False)
    restored = db.execute(
        "SELECT created_at, valid_from, valid_until FROM observations"
    ).fetchone()
    assert restored["created_at"].isoformat() == "2024-01-02T00:00:00+00:00"
    assert restored["valid_from"].isoformat() == "2024-01-03T00:00:00+00:00"
    assert restored["valid_until"].isoformat() == "2024-12-31T00:00:00+00:00"
