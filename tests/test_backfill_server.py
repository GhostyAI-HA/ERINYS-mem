"""FastMCP backfill boundary tests."""

from __future__ import annotations

from erinys_memory import server
from erinys_memory.backfill import BackfillService


def item() -> dict[str, object]:
    return {
        "topic_key": "topic-1",
        "title": "Historical answer",
        "content": "Historical content",
        "project": "Project",
        "memory_role": "CLAIM",
        "claim_status": "ACTIVE",
        "source": {
            "kind": "slack",
            "namespace": "slack:workspace-1",
            "event_key": "record-1",
            "recorded_at": "2024-01-01T00:00:00Z",
        },
    }


class FakeService:
    def __init__(self) -> None:
        self.requests = []

    def plan(self, request):
        self.requests.append(request)
        return {"mode": "plan", "run_key": request.run_key}

    def apply(self, request):
        self.requests.append(request)
        return {"mode": "apply", "run_key": request.run_key}


def test_mcp_backfill_validates_before_service(monkeypatch) -> None:
    service = FakeService()
    monkeypatch.setattr(server, "_backfill_service", lambda: service)

    result = server.erinys_backfill("run-1", [item()])

    assert result["ok"] is True
    assert result["data"] == {"mode": "plan", "run_key": "run-1"}
    assert service.requests[0].items[0].source.recorded_at.tzinfo is not None


def test_mcp_backfill_rejects_unknown_and_historical_authority(monkeypatch) -> None:
    service = FakeService()
    monkeypatch.setattr(server, "_backfill_service", lambda: service)
    invalid = {**item(), "source_class": "policy"}

    result = server.erinys_backfill("run-1", [invalid])

    assert result["ok"] is False
    assert result["error"]["code"] == "INVALID_BACKFILL_ITEM"
    assert service.requests == []


def test_mcp_validation_error_does_not_echo_content(monkeypatch) -> None:
    service = FakeService()
    monkeypatch.setattr(server, "_backfill_service", lambda: service)
    invalid = {**item(), "content": "token=do-not-echo", "unknown": True}

    result = server.erinys_backfill("run-1", [invalid])

    assert result["ok"] is False
    assert "do-not-echo" not in str(result)
    assert result["error"]["message"] == "backfill request validation failed"


def test_mcp_apply_requires_plan_digest(monkeypatch) -> None:
    service = FakeService()
    monkeypatch.setattr(server, "_backfill_service", lambda: service)

    result = server.erinys_backfill("run-1", [item()], mode="apply")

    assert result["ok"] is False
    assert result["error"]["code"] == "INVALID_BACKFILL_ITEM"


def test_mcp_rejects_historical_retention_authority(
    backfill_db,
    fake_embed_batch,
    monkeypatch,
) -> None:
    db, config = backfill_db
    service = BackfillService(db, config, fake_embed_batch)
    monkeypatch.setattr(server, "_backfill_service", lambda: service)
    historical = {**item(), "historical_value": True}

    result = server.erinys_backfill("run-historical", [historical])

    assert result["ok"] is False
    assert result["error"]["code"] == "INVALID_BACKFILL_ITEM"
