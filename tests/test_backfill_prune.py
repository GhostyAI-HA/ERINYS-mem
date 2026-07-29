"""Prune retention protection for historical backfills."""

from __future__ import annotations

from erinys_memory import server
from erinys_memory.backfill import BackfillService
from erinys_memory.backfill_contracts import BackfillRequest


def records() -> list[dict[str, object]]:
    common = {
        "title": "Historical memory",
        "content": "Historical content",
        "project": "Project",
        "memory_role": "EVIDENCE",
    }
    return [
        {
            **common,
            "topic_key": "protected",
            "historical_value": True,
            "source": {
                "kind": "slack",
                "namespace": "slack:workspace-1",
                "event_key": "record-protected",
                "recorded_at": "2020-01-01T00:00:00Z",
            },
        },
        {
            **common,
            "topic_key": "ordinary",
            "source": {
                "kind": "slack",
                "namespace": "slack:workspace-1",
                "event_key": "record-ordinary",
                "recorded_at": "2020-01-01T00:00:00Z",
            },
        },
    ]


def apply_records(service: BackfillService) -> None:
    plan_request = BackfillRequest.model_validate({
        "run_key": "prune-records",
        "items": records(),
    })
    plan = service.plan(plan_request, allow_historical=True)
    apply_request = BackfillRequest.model_validate({
        "run_key": "prune-records",
        "items": records(),
        "mode": "apply",
        "expected_plan_digest": plan["plan_digest"],
    })
    service.apply(apply_request, allow_historical=True)


def test_prune_reports_and_preserves_historical_value(
    backfill_db,
    fake_embed_batch,
    monkeypatch,
) -> None:
    db, config = backfill_db
    service = BackfillService(db, config, fake_embed_batch)
    apply_records(service)
    monkeypatch.setattr(server, "_DB", db)
    monkeypatch.setattr(server, "_CONFIG", config)

    plan = server.erinys_prune(threshold=10.0)
    executed = server.erinys_prune(threshold=10.0, dry_run=False)

    assert plan["ok"] is True
    assert [row["topic_key"] for row in plan["data"]["protected"]] == ["protected"]
    assert [row["topic_key"] for row in plan["data"]["candidates"]] == ["ordinary"]
    assert executed["data"]["deleted_ids"]
    remaining = db.execute("SELECT topic_key FROM observations").fetchall()
    assert [row["topic_key"] for row in remaining] == ["protected"]
