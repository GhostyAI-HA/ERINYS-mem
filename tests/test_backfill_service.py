"""Backfill plan/apply integration tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from erinys_memory.backfill import BackfillError, BackfillService, canonical_json
from erinys_memory.backfill_contracts import BackfillRequest


def backfill_item(
    topic_key: str = "topic-1",
    event_key: str = "record-1",
    **overrides: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "topic_key": topic_key,
        "title": "Historical token=secret answer",
        "content": "Send token=secret to <@U123> user@example.com",
        "project": "Project",
        "memory_role": "CLAIM",
        "claim_status": "ACTIVE",
        "source": {
            "kind": "slack",
            "namespace": "slack:workspace-1",
            "event_key": event_key,
            "recorded_at": "2024-01-01T00:00:00Z",
        },
        "use_events": [
            {"event_key": "use-1", "occurred_at": "2024-02-01T00:00:00Z"},
        ],
    }
    payload.update(overrides)
    return payload


def make_request(
    items: list[dict[str, object]],
    run_key: str,
    digest: str | None = None,
) -> BackfillRequest:
    payload: dict[str, object] = {"run_key": run_key, "items": items}
    if digest is not None:
        payload.update(mode="apply", expected_plan_digest=digest)
    return BackfillRequest.model_validate(payload)


def apply_items(
    service: BackfillService,
    items: list[dict[str, object]],
    run_key: str,
) -> tuple[dict[str, object], dict[str, object]]:
    plan = service.plan(make_request(items, run_key))
    applied = service.apply(make_request(items, run_key, str(plan["plan_digest"])))
    return plan, applied


def test_plan_is_content_free_and_does_not_mutate(
    backfill_db,
    fake_embed_batch,
) -> None:
    db, config = backfill_db
    service = BackfillService(db, config, fake_embed_batch)

    plan = service.plan(make_request([backfill_item()], "run-plan"))

    assert plan["counts"]["create"] == 1
    assert plan["vector_health"] == "not_mutated"
    assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0
    serialized = canonical_json(plan)
    assert "token=secret" not in serialized
    assert "user@example.com" not in serialized


def test_apply_writes_projection_vector_fts_and_safe_metadata(
    backfill_db,
    fake_embed_batch,
) -> None:
    db, config = backfill_db
    service = BackfillService(db, config, fake_embed_batch)

    _, applied = apply_items(service, [backfill_item()], "run-apply")
    row = db.execute("SELECT * FROM observations").fetchone()
    metadata = json.loads(row["metadata"])

    assert row["created_at"].isoformat() == "2024-01-01T00:00:00+00:00"
    assert row["access_count"] == 1
    assert row["last_accessed"].isoformat() == "2024-02-01T00:00:00+00:00"
    assert row["content"] == "Send [REDACTED] to [REDACTED] [REDACTED]"
    assert metadata["memory_role"] == "CLAIM"
    assert metadata["verification_state"] == "UNVERIFIED"
    assert metadata["memory_safety"]["source_class"] == "first_party_record"
    assert metadata["memory_safety"].get("action_authority") is None
    assert applied["vector_health"] == {"missing": 0, "orphan": 0}
    assert applied["storage_health"] == {
        "missing_observations": 0,
        "missing_vectors": 0,
        "missing_fts": 0,
        "missing_events": 0,
        "missing_edges": 0,
        "complete": True,
    }
    assert Path(str(applied["backup_path"])).exists()
    assert db.execute("SELECT COUNT(*) FROM observation_events").fetchone()[0] == 2
    match = db.execute(
        "SELECT rowid FROM observations_fts WHERE observations_fts MATCH 'Historical'"
    ).fetchone()
    assert match["rowid"] == row["id"]


def test_apply_redacts_nested_metadata(
    backfill_db,
    fake_embed_batch,
) -> None:
    db, config = backfill_db
    service = BackfillService(db, config, fake_embed_batch)
    record = backfill_item(metadata={
        "review": {"contact": "user@example.com", "note": "token=secret"},
    })

    apply_items(service, [record], "run-metadata-redaction")
    row = db.execute("SELECT metadata FROM observations").fetchone()
    metadata = json.loads(row["metadata"])

    assert metadata["review"] == {
        "contact": "[REDACTED]",
        "note": "[REDACTED]",
    }


def test_second_apply_is_fully_idempotent(
    backfill_db,
    fake_embed_batch,
) -> None:
    db, config = backfill_db
    service = BackfillService(db, config, fake_embed_batch)
    items = [backfill_item()]
    plan, first = apply_items(service, items, "run-idempotent")
    before = dict(db.execute("SELECT access_count, last_accessed FROM observations").fetchone())

    second = service.apply(
        make_request(items, "run-idempotent", str(plan["plan_digest"])),
    )
    after = dict(db.execute("SELECT access_count, last_accessed FROM observations").fetchone())

    assert first["status"] == "APPLIED"
    assert second["outcome"] == "already_applied"
    assert second["plan_digest"] == plan["plan_digest"]
    assert second["items"][0]["strength_before"] == second["items"][0]["strength_after"]
    assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 1
    assert db.execute("SELECT COUNT(*) FROM observation_events").fetchone()[0] == 2
    assert after == before


def test_new_used_event_in_new_run_increments_once(
    backfill_db,
    fake_embed_batch,
) -> None:
    db, config = backfill_db
    service = BackfillService(db, config, fake_embed_batch)
    base = backfill_item()
    apply_items(service, [base], "run-base")
    extended = backfill_item(use_events=[
        {"event_key": "use-1", "occurred_at": "2024-02-01T00:00:00Z"},
        {"event_key": "use-2", "occurred_at": "2024-03-01T00:00:00Z"},
    ])

    plan, _ = apply_items(service, [extended], "run-extended")
    row = db.execute("SELECT access_count, last_accessed FROM observations").fetchone()

    assert plan["counts"]["match"] == 1
    assert plan["counts"]["new_used_events"] == 1
    assert row["access_count"] == 2
    assert row["last_accessed"].isoformat() == "2024-03-01T00:00:00+00:00"


def test_source_mutation_fails_closed(
    backfill_db,
    fake_embed_batch,
) -> None:
    db, config = backfill_db
    service = BackfillService(db, config, fake_embed_batch)
    apply_items(service, [backfill_item()], "run-source")
    source = dict(backfill_item()["source"])
    source["recorded_at"] = "2024-01-02T00:00:00Z"
    mutated = backfill_item(source=source)
    plan = service.plan(make_request([mutated], "run-mutated"))

    with pytest.raises(BackfillError) as exc:
        service.apply(
            make_request([mutated], "run-mutated", str(plan["plan_digest"])),
        )

    assert plan["items"][0]["status"] == "SOURCE_MUTATION"
    assert exc.value.code == "SOURCE_MUTATION"


def test_role_mutation_fails_closed(
    backfill_db,
    fake_embed_batch,
) -> None:
    db, config = backfill_db
    service = BackfillService(db, config, fake_embed_batch)
    evidence = backfill_item(memory_role="EVIDENCE", claim_status=None)
    apply_items(service, [evidence], "run-evidence")
    promoted = backfill_item(memory_role="CLAIM", claim_status="ACTIVE")
    plan = service.plan(make_request([promoted], "run-role-mutation"))

    with pytest.raises(BackfillError) as exc:
        service.apply(make_request(
            [promoted], "run-role-mutation", str(plan["plan_digest"]),
        ))

    assert plan["items"][0]["status"] == "SOURCE_MUTATION"
    assert exc.value.code == "SOURCE_MUTATION"


def test_title_mutation_is_topic_conflict(
    backfill_db,
    fake_embed_batch,
) -> None:
    db, config = backfill_db
    service = BackfillService(db, config, fake_embed_batch)
    apply_items(service, [backfill_item()], "run-title")
    changed = backfill_item(title="Changed historical title")
    plan = service.plan(make_request([changed], "run-title-mutation"))

    with pytest.raises(BackfillError) as exc:
        service.apply(make_request(
            [changed], "run-title-mutation", str(plan["plan_digest"]),
        ))

    assert plan["items"][0]["status"] == "TOPIC_CONFLICT"
    assert exc.value.code == "TOPIC_CONFLICT"


def test_plan_digest_change_fails_closed(
    backfill_db,
    fake_embed_batch,
) -> None:
    db, config = backfill_db
    service = BackfillService(db, config, fake_embed_batch)
    base = backfill_item()
    apply_items(service, [base], "run-before-digest")
    extended = backfill_item(use_events=[
        *base["use_events"],
        {"event_key": "use-2", "occurred_at": "2024-03-01T00:00:00Z"},
    ])
    plan = service.plan(make_request([extended], "run-stale"))
    db.execute(
        "UPDATE observations SET updated_at = '2026-01-01T00:00:00+00:00'"
    )
    db.commit()

    with pytest.raises(BackfillError) as exc:
        service.apply(
            make_request([extended], "run-stale", str(plan["plan_digest"])),
        )

    assert exc.value.code == "PLAN_DIGEST_MISMATCH"


def test_referenced_target_change_invalidates_plan(
    backfill_db,
    fake_embed_batch,
) -> None:
    db, config = backfill_db
    service = BackfillService(db, config, fake_embed_batch)
    apply_items(service, [backfill_item()], "run-reference-base")
    successor = backfill_item(
        topic_key="topic-2",
        event_key="record-2",
        source={
            "kind": "slack",
            "namespace": "slack:workspace-1",
            "event_key": "record-2",
            "recorded_at": "2025-01-01T00:00:00Z",
        },
        use_events=[],
        supersedes_topic_key="topic-1",
    )
    plan = service.plan(make_request([successor], "run-reference-change"))
    db.execute(
        "UPDATE observations SET updated_at = '2026-01-01T00:00:00+00:00' "
        "WHERE topic_key = 'topic-1'"
    )
    db.commit()

    with pytest.raises(BackfillError) as exc:
        service.apply(make_request(
            [successor],
            "run-reference-change",
            str(plan["plan_digest"]),
        ))

    assert exc.value.code == "PLAN_DIGEST_MISMATCH"


def test_embedding_failure_starts_no_write(
    backfill_db,
) -> None:
    db, config = backfill_db

    def fail(_: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding unavailable")

    service = BackfillService(db, config, fail)
    items = [backfill_item()]
    plan = service.plan(make_request(items, "run-embed-fail"))

    with pytest.raises(BackfillError) as exc:
        service.apply(
            make_request(items, "run-embed-fail", str(plan["plan_digest"])),
        )

    assert exc.value.code == "EMBEDDING_FAILED"
    assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM backfill_runs").fetchone()[0] == 0


def test_partial_run_resumes_with_original_digest(
    backfill_db,
    fake_embed_batch,
    monkeypatch,
) -> None:
    db, config = backfill_db
    config.backfill_chunk_size = 1
    service = BackfillService(db, config, fake_embed_batch)
    items = [
        backfill_item("topic-1", "record-1"),
        backfill_item("topic-2", "record-2", use_events=[]),
    ]
    plan = service.plan(make_request(items, "run-partial"))
    request = make_request(items, "run-partial", str(plan["plan_digest"]))
    original = service._apply_item

    def fail_second(run_id, entry, embeddings):
        if entry.item.topic_key == "topic-2":
            raise RuntimeError("forced chunk failure")
        original(run_id, entry, embeddings)

    monkeypatch.setattr(service, "_apply_item", fail_second)
    with pytest.raises(BackfillError):
        service.apply(request)
    assert db.execute(
        "SELECT status FROM backfill_runs"
    ).fetchone()["status"] == "PARTIAL"
    monkeypatch.setattr(service, "_apply_item", original)

    resumed = service.apply(request)

    assert resumed["status"] == "APPLIED"
    assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 2
    assert db.execute("SELECT COUNT(*) FROM observation_events").fetchone()[0] == 3


def test_storage_health_failure_resumes_and_repairs_vector(
    backfill_db,
    fake_embed_batch,
    monkeypatch,
) -> None:
    db, config = backfill_db
    service = BackfillService(db, config, fake_embed_batch)
    items = [backfill_item()]
    plan = service.plan(make_request(items, "run-storage-repair"))
    request = make_request(
        items, "run-storage-repair", str(plan["plan_digest"]),
    )
    original = service._apply_edges

    def corrupt_vector(bundle) -> None:
        original(bundle)
        db.execute("DELETE FROM vec_observations")

    monkeypatch.setattr(service, "_apply_edges", corrupt_vector)
    with pytest.raises(BackfillError) as exc:
        service.apply(request)
    assert exc.value.code == "TRANSACTION_FAILED"
    assert db.execute(
        "SELECT status FROM backfill_runs"
    ).fetchone()["status"] == "PARTIAL"
    monkeypatch.setattr(service, "_apply_edges", original)

    resumed = service.apply(request)

    assert resumed["status"] == "APPLIED"
    assert resumed["storage_health"]["complete"] is True
    assert resumed["vector_health"] == {"missing": 0, "orphan": 0}


def test_retention_and_local_authority_are_visible(
    backfill_db,
    fake_embed_batch,
) -> None:
    db, config = backfill_db
    service = BackfillService(db, config, fake_embed_batch)
    historical = backfill_item(historical_value=True)
    request = make_request([historical], "run-retention")

    with pytest.raises(BackfillError):
        service.plan(request)
    plan = service.plan(request, allow_historical=True)

    assert plan["items"][0]["retention_protected"] is True
    assert plan["items"][0]["would_prune"] is False


def test_apply_is_disabled_by_default_flag(
    backfill_db,
    fake_embed_batch,
) -> None:
    db, config = backfill_db
    config.backfill_apply_enabled = False
    service = BackfillService(db, config, fake_embed_batch)
    items = [backfill_item()]
    plan = service.plan(make_request(items, "run-disabled"))

    with pytest.raises(BackfillError) as exc:
        service.apply(
            make_request(items, "run-disabled", str(plan["plan_digest"])),
        )

    assert exc.value.code == "BACKFILL_DISABLED"
    assert db.execute("SELECT COUNT(*) FROM backfill_runs").fetchone()[0] == 0


def test_global_scope_requires_local_authority(
    backfill_db,
    fake_embed_batch,
) -> None:
    db, config = backfill_db
    service = BackfillService(db, config, fake_embed_batch)
    request = make_request([backfill_item(scope="global")], "run-global")

    with pytest.raises(BackfillError) as exc:
        service.plan(request)
    plan = service.plan(request, allow_global=True)

    assert exc.value.code == "INVALID_BACKFILL_ITEM"
    assert plan["counts"]["create"] == 1
