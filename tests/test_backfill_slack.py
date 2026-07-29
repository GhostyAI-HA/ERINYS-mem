"""Slack normalization adapter tests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "scripts" / "backfill_slack.py"


def load_adapter():
    spec = importlib.util.spec_from_file_location("backfill_slack", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_slack_adapter_redacts_and_hashes_identifiers(tmp_path: Path) -> None:
    adapter = load_adapter()
    source = tmp_path / "slack.jsonl"
    raw = {
        "workspace": "W123",
        "channel": "C456",
        "message_ts": "1710000000.000100",
        "recorded_at": "2024-03-09T00:00:00Z",
        "topic_key": "ticket-1",
        "title": "Question from user@example.com",
        "content": "Send token=secret to <@U123>",
        "project": "Project",
        "memory_role": "EVIDENCE",
    }
    source.write_text(json.dumps(raw) + "\n", encoding="utf-8")

    first = json.loads(adapter.convert(str(source), "test-secret")[0])
    second = json.loads(adapter.convert(str(source), "test-secret")[0])

    assert first == second
    assert first["content"] == "Send [REDACTED] to [REDACTED]"
    assert first["title"] == "Question from [REDACTED]"
    assert first["source"]["namespace"] == "slack:W123"
    assert "W123" not in first["source"]["event_key"]
    assert "C456" not in first["source"]["event_key"]
    assert "1710000000" not in first["source"]["event_key"]


def test_slack_adapter_redacts_nested_metadata(tmp_path: Path) -> None:
    adapter = load_adapter()
    source = tmp_path / "slack.jsonl"
    raw = {
        "workspace": "W123",
        "channel": "C456",
        "message_ts": "1710000000.000100",
        "recorded_at": "2024-03-09T00:00:00Z",
        "topic_key": "ticket-1",
        "title": "Question",
        "content": "Historical question",
        "project": "Project",
        "memory_role": "EVIDENCE",
        "metadata": {"contact": "user@example.com", "secret": "token=secret"},
    }
    source.write_text(json.dumps(raw) + "\n", encoding="utf-8")

    normalized = json.loads(adapter.convert(str(source), None)[0])

    assert normalized["metadata"] == {
        "contact": "[REDACTED]",
        "secret": "[REDACTED]",
    }


def test_slack_adapter_emits_distinct_used_events(tmp_path: Path) -> None:
    adapter = load_adapter()
    source = tmp_path / "slack.jsonl"
    raw = {
        "workspace": "W123",
        "channel": "C456",
        "message_ts": "1710000000.000100",
        "recorded_at": "2024-03-09T00:00:00Z",
        "topic_key": "answer-1",
        "title": "Answer",
        "content": "Historical answer",
        "project": "Project",
        "memory_role": "CLAIM",
        "claim_status": "ACTIVE",
        "edges": [
            {"relation": "verified_by", "target_topic_key": "evidence-1"},
        ],
        "use_events": [
            {
                "message_ts": "1711000000.000100",
                "occurred_at": "2024-03-20T00:00:00Z",
            },
            {
                "message_ts": "1712000000.000100",
                "occurred_at": "2024-04-01T00:00:00Z",
            },
        ],
    }
    source.write_text(json.dumps(raw) + "\n", encoding="utf-8")

    normalized = json.loads(adapter.convert(str(source), None)[0])
    keys = [event["event_key"] for event in normalized["use_events"]]

    assert len(keys) == len(set(keys)) == 2
    assert normalized["verification_state"] == "UNVERIFIED"
    assert normalized["edges"] == [
        {"relation": "verified_by", "target_topic_key": "evidence-1"},
    ]
