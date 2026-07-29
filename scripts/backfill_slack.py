#!/usr/bin/env python3
"""Normalize reviewed Slack records into ERINYS backfill JSONL."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from erinys_memory.backfill import (
    redact_backfill_metadata,
    redact_backfill_text,
)
from erinys_memory.backfill_contracts import BackfillEdge, BackfillItem

HMAC_ENV = "ERINYS_BACKFILL_EVENT_HMAC_KEY"
MAX_KEY_LENGTH = 500


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SlackUse(StrictModel):
    occurred_at: AwareDatetime
    message_ts: str = Field(min_length=1, max_length=MAX_KEY_LENGTH)


class SlackRecord(StrictModel):
    workspace: str = Field(min_length=1, max_length=MAX_KEY_LENGTH)
    channel: str = Field(min_length=1, max_length=MAX_KEY_LENGTH)
    message_ts: str = Field(min_length=1, max_length=MAX_KEY_LENGTH)
    recorded_at: AwareDatetime
    topic_key: str = Field(min_length=1, max_length=MAX_KEY_LENGTH)
    title: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1, max_length=50_000)
    project: str = Field(min_length=1, max_length=MAX_KEY_LENGTH)
    memory_role: Literal["EVIDENCE", "CLAIM", "DECISION", "OUTCOME"]
    type: str = "manual"
    valid_from: AwareDatetime | None = None
    valid_until: AwareDatetime | None = None
    use_events: list[SlackUse] = Field(default_factory=list)
    edges: list[BackfillEdge] = Field(default_factory=list)
    supersedes_topic_key: str | None = None
    claim_status: str | None = None
    historical_value: bool = False
    metadata: dict[str, object] = Field(default_factory=dict)


SlackUse.model_rebuild(_types_namespace={"AwareDatetime": AwareDatetime})
SlackRecord.model_rebuild(
    _types_namespace={
        "AwareDatetime": AwareDatetime,
        "Literal": Literal,
        "BackfillEdge": BackfillEdge,
        "SlackUse": SlackUse,
    },
)


def _event_key(parts: list[str], secret: str | None) -> str:
    payload = "\x1f".join(parts).encode("utf-8")
    raw = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest() if secret else hashlib.sha256(payload).hexdigest()
    return "slack:" + raw


def _locator(record: SlackRecord, message_ts: str) -> str:
    return f"slack://{record.workspace}/{record.channel}/{message_ts}"


def _use_event(
    record: SlackRecord,
    use: SlackUse,
    secret: str | None,
) -> dict[str, object]:
    parts = [record.workspace, record.channel, use.message_ts, "USED"]
    return {
        "event_key": _event_key(parts, secret),
        "occurred_at": use.occurred_at.isoformat(),
        "locator": _locator(record, use.message_ts),
    }


def normalize(record: SlackRecord, secret: str | None) -> dict[str, object]:
    parts = [
        record.workspace, record.channel, record.message_ts,
        record.memory_role, "RECORDED",
    ]
    result = {
        **_content_fields(record),
        **_temporal_fields(record),
        "source": _source_fields(record, parts, secret),
        "use_events": [
            _use_event(record, use, secret) for use in record.use_events
        ],
    }
    cleaned = {key: value for key, value in result.items() if value is not None}
    return BackfillItem.model_validate(cleaned).model_dump(mode="json")


def _content_fields(record: SlackRecord) -> dict[str, object]:
    return {
        "topic_key": record.topic_key,
        "title": redact_backfill_text(record.title),
        "content": redact_backfill_text(record.content),
        "project": record.project, "scope": "project", "type": record.type,
        "memory_role": record.memory_role,
        "metadata": redact_backfill_metadata(record.metadata, redact_backfill_text),
    }


def _temporal_fields(record: SlackRecord) -> dict[str, object]:
    return {
        "valid_from": _iso(record.valid_from),
        "valid_until": _iso(record.valid_until),
        "edges": [edge.model_dump(mode="json") for edge in record.edges],
        "supersedes_topic_key": record.supersedes_topic_key,
        "claim_status": record.claim_status,
        "verification_state": "UNVERIFIED",
        "historical_value": record.historical_value,
    }


def _source_fields(
    record: SlackRecord,
    parts: list[str],
    secret: str | None,
) -> dict[str, object]:
    return {
        "kind": "slack", "namespace": f"slack:{record.workspace}",
        "event_key": _event_key(parts, secret),
        "recorded_at": record.recorded_at.isoformat(),
        "locator": _locator(record, record.message_ts),
    }


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _read_lines(path: str) -> list[str]:
    text = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    return [line for line in text.splitlines() if line.strip()]


def convert(path: str, secret: str | None) -> list[str]:
    records = [
        SlackRecord.model_validate_json(line)
        for line in _read_lines(path)
    ]
    return [
        json.dumps(normalize(record, secret), ensure_ascii=False, sort_keys=True)
        for record in records
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="-")
    return parser


def _write(lines: list[str], output: str) -> None:
    text = "\n".join(lines) + ("\n" if lines else "")
    if output == "-":
        sys.stdout.write(text)
        return
    Path(output).write_text(text, encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    _write(convert(args.input, os.environ.get(HMAC_ENV)), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
