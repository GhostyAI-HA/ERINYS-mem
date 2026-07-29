"""Backfilled memory roles in Current Valid Claim shadow."""

from __future__ import annotations

from datetime import datetime, timezone

from erinys_memory.backfill import BackfillService
from erinys_memory.backfill_contracts import BackfillRequest
from erinys_memory.claim_shadow import build_shadow_result
from erinys_memory.claim_shadow_runtime import _adapt_claims, _adapt_search


def source(event: str, recorded_at: str) -> dict[str, object]:
    return {
        "kind": "slack",
        "namespace": "slack:workspace-1",
        "event_key": event,
        "recorded_at": recorded_at,
    }


def evidence() -> dict[str, object]:
    return {
        "topic_key": "evidence-1",
        "title": "Inquiry",
        "content": "Was the historical limit 50 percent?",
        "project": "Project",
        "memory_role": "EVIDENCE",
        "source": source("record-evidence", "2024-02-01T00:00:00Z"),
    }


def claim() -> dict[str, object]:
    return {
        "topic_key": "claim-1",
        "title": "Historical limit",
        "content": "The historical limit was 50 percent.",
        "project": "Project",
        "memory_role": "CLAIM",
        "claim_status": "ACTIVE",
        "source": source("record-claim", "2024-02-02T00:00:00Z"),
        "valid_from": "2024-01-01T00:00:00Z",
        "edges": [
            {"relation": "verified_by", "target_topic_key": "evidence-1"},
        ],
    }


def apply_records(service: BackfillService) -> None:
    items = [evidence(), claim()]
    plan_request = BackfillRequest.model_validate({
        "run_key": "claim-shadow",
        "items": items,
    })
    plan = service.plan(plan_request)
    apply_request = BackfillRequest.model_validate({
        "run_key": "claim-shadow",
        "items": items,
        "mode": "apply",
        "expected_plan_digest": plan["plan_digest"],
    })
    service.apply(apply_request)


def test_evidence_is_not_adapted_as_active_claim(
    backfill_db,
    fake_embed_batch,
) -> None:
    db, config = backfill_db
    service = BackfillService(db, config, fake_embed_batch)
    apply_records(service)
    rows = [
        dict(row) for row in db.execute(
            "SELECT * FROM observations ORDER BY id"
        ).fetchall()
    ]

    claims, warnings = _adapt_claims(
        rows, datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    assert [node.proposition for node in claims] == [
        "The historical limit was 50 percent.",
    ]
    assert any(value.startswith("non_claim_node_excluded:") for value in warnings)


def test_recorded_time_and_valid_time_remain_separate(
    backfill_db,
    fake_embed_batch,
) -> None:
    db, config = backfill_db
    service = BackfillService(db, config, fake_embed_batch)
    apply_records(service)
    row = dict(db.execute(
        "SELECT * FROM observations WHERE topic_key = 'claim-1'"
    ).fetchone())

    claims, _ = _adapt_claims(
        [row], datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    assert claims[0].recorded_at.isoformat() == "2024-02-02T00:00:00+00:00"
    assert claims[0].valid_time.from_.isoformat() == "2024-01-01T00:00:00+00:00"


def test_verified_evidence_edge_is_available_to_shadow_without_baseline_drift(
    backfill_db,
    fake_embed_batch,
) -> None:
    db, config = backfill_db
    service = BackfillService(db, config, fake_embed_batch)
    apply_records(service)
    rows = [
        {**dict(row), "effective_score": 1.0}
        for row in db.execute(
            "SELECT * FROM observations ORDER BY id"
        ).fetchall()
    ]

    adapted = _adapt_search(
        db,
        "historical limit",
        "Project",
        rows,
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    result = build_shadow_result(adapted.shadow_input)

    assert adapted.shadow_input.baseline_claim_ids == ["observation:2"]
    assert len(adapted.shadow_input.claims) == 1
    assert len(adapted.shadow_input.edges) == 1
    assert [node.claim_id for node in result.active_claims] == ["observation:2"]
    assert result.supporting_paths[0].node_ids == [
        "observation:1",
        "observation:2",
    ]
