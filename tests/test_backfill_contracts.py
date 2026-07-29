"""Public timestamp-backfill contract tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from erinys_memory.backfill import (
    BackfillError,
    BackfillService,
    content_digest,
    digest,
    request_digest,
)
from erinys_memory.backfill_contracts import BackfillRelation, BackfillRequest


def item(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "topic_key": "topic-1",
        "title": "Historical answer",
        "content": "The historical answer.",
        "project": "Project",
        "memory_role": "CLAIM",
        "claim_status": "ACTIVE",
        "source": {
            "kind": "slack",
            "namespace": "slack:workspace-1",
            "event_key": "record-1",
            "recorded_at": "2024-01-01T09:00:00+09:00",
        },
    }
    payload.update(overrides)
    return payload


def request(**overrides: object) -> BackfillRequest:
    payload: dict[str, object] = {"run_key": "run-1", "items": [item()]}
    payload.update(overrides)
    return BackfillRequest.model_validate(payload)


def test_relation_string_uses_wire_value() -> None:
    assert str(BackfillRelation.SUPPORTS) == "supports"


def test_times_are_normalized_to_utc() -> None:
    parsed = request()

    assert parsed.items[0].source.recorded_at == datetime(
        2024, 1, 1, 0, 0, tzinfo=timezone.utc,
    )


def test_naive_timestamp_is_rejected() -> None:
    source = dict(item()["source"])
    source["recorded_at"] = "2024-01-01T09:00:00"

    with pytest.raises(ValidationError):
        request(items=[item(source=source)])


def test_invalid_valid_interval_is_rejected() -> None:
    with pytest.raises(ValidationError, match="valid_until"):
        request(items=[
            item(
                valid_from="2024-02-01T00:00:00Z",
                valid_until="2024-01-01T00:00:00Z",
            ),
        ])


def test_unknown_and_server_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        request(items=[item(source_class="policy")])
    with pytest.raises(ValidationError, match="server-controlled"):
        request(items=[item(metadata={"legal_retention": True})])
    with pytest.raises(ValidationError, match="server-controlled"):
        request(items=[item(metadata={"risk_class": "CRITICAL"})])


def test_claim_status_boundary_is_strict() -> None:
    with pytest.raises(ValidationError, match="CLAIM requires"):
        request(items=[item(claim_status=None)])
    with pytest.raises(ValidationError, match="only valid"):
        request(items=[item(memory_role="EVIDENCE", claim_status="ACTIVE")])


def test_duplicate_topic_and_event_keys_are_rejected() -> None:
    duplicate_topic = item(topic_key="topic-1")
    with pytest.raises(ValidationError, match="duplicate topic_key"):
        request(items=[item(), duplicate_topic])
    duplicate_use = [
        {"event_key": "use-1", "occurred_at": "2024-02-01T00:00:00Z"},
        {"event_key": "use-1", "occurred_at": "2024-03-01T00:00:00Z"},
    ]
    with pytest.raises(ValidationError, match="duplicate use event_key"):
        request(items=[item(use_events=duplicate_use)])


def test_run_requires_one_source_boundary() -> None:
    second_source = dict(item()["source"])
    second_source["namespace"] = "slack:other"
    with pytest.raises(ValidationError, match="source boundary"):
        request(items=[
            item(topic_key="topic-1"),
            item(topic_key="topic-2", source=second_source),
        ])


def test_digest_helpers_are_deterministic() -> None:
    first = request()
    second = request()

    assert digest({"b": 2, "a": 1}) == digest({"a": 1, "b": 2})
    assert content_digest("same") == content_digest("same")
    assert request_digest(first, lambda value: value) == request_digest(
        second, lambda value: value,
    )


def test_future_timestamp_is_rejected_by_service(
    backfill_db,
    fake_embed_batch,
) -> None:
    db, config = backfill_db
    future = "2999-01-01T00:00:00Z"
    service = BackfillService(db, config, fake_embed_batch)

    with pytest.raises(BackfillError) as exc:
        service.plan(request(items=[item(source={
            "kind": "slack",
            "namespace": "slack:workspace-1",
            "event_key": "record-1",
            "recorded_at": future,
        })]))

    assert exc.value.code == "FUTURE_TIMESTAMP"
