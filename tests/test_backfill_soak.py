"""Synthetic 100-record copy-DB release drill."""

from __future__ import annotations

from time import perf_counter

from erinys_memory.backfill import BackfillService
from erinys_memory.backfill_contracts import BackfillRequest

SOAK_RECORDS = 100
DB_PIPELINE_SLO_SECONDS = 5.0


def record(index: int) -> dict[str, object]:
    return {
        "topic_key": f"soak-{index:03d}",
        "title": f"Synthetic historical record {index}",
        "content": f"Reviewed synthetic content {index}",
        "project": "SoakProject",
        "memory_role": "EVIDENCE",
        "source": {
            "kind": "slack",
            "namespace": "slack:workspace-1",
            "event_key": f"record-{index:03d}",
            "recorded_at": "2024-01-01T00:00:00Z",
        },
        "use_events": _use_events(index),
    }


def _use_events(index: int) -> list[dict[str, str]]:
    if index % 2:
        return []
    return [{
        "event_key": f"use-{index:03d}",
        "occurred_at": "2024-02-01T00:00:00Z",
    }]


def request(
    items: list[dict[str, object]],
    digest: str | None = None,
) -> BackfillRequest:
    payload: dict[str, object] = {"run_key": "synthetic-soak-100", "items": items}
    if digest is not None:
        payload.update(mode="apply", expected_plan_digest=digest)
    return BackfillRequest.model_validate(payload)


def test_synthetic_copy_db_plan_apply_retry_and_rollback(
    backfill_db,
    fake_embed_batch,
) -> None:
    db, config = backfill_db
    service = BackfillService(db, config, fake_embed_batch)
    items = [record(index) for index in range(SOAK_RECORDS)]
    started = perf_counter()
    plan = service.plan(request(items))
    apply_request = request(items, str(plan["plan_digest"]))

    applied = service.apply(apply_request)
    retried = service.apply(apply_request)

    assert plan["counts"] == {
        "requested": 100,
        "create": 100,
        "match": 0,
        "skip_idempotent": 0,
        "conflict": 0,
        "new_used_events": 50,
        "would_prune": 100,
        "sentry_restricted": 0,
    }
    assert applied["vector_health"] == {"missing": 0, "orphan": 0}
    assert retried["outcome"] == "already_applied"
    assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 100
    assert db.execute("SELECT COUNT(*) FROM observation_events").fetchone()[0] == 150

    rolled_back = service.rollback(str(applied["run_id"]), dry_run=False)

    assert rolled_back["membership"]["complete"] is True
    assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0
    assert perf_counter() - started < DB_PIPELINE_SLO_SECONDS
