"""Timestamp-aware historical backfill service.

The event ledger is authoritative. Observation temporal/access columns remain a
search-compatible projection rebuilt from lazy legacy baselines plus unique
historical events.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ._sqlite import sqlite3
from .backfill_contracts import BackfillItem, BackfillMode, BackfillRequest
from .config import ErinysConfig
from .db import insert_observation_vector_rows
from .decay import current_strength
from .embedding import serialize_f32
from .provenance import build_provenance
from .sentry import SourceClass, build_record_memory_safety, classify_record_source

HASH_PREFIX = "sha256:"
RUN_PREFIX = "bf_"
DEFAULT_PRINCIPAL = "erinys_backfill"
BACKFILL_SOURCE = "migration"
RECORD_EVENT = "RECORDED"
USE_EVENT = "USED"
SUPERSEDED_EVENT = "SUPERSEDED"
PLAN_CONFLICTS = {
    "SOURCE_MUTATION",
    "TOPIC_CONFLICT",
    "TEMPORAL_CONFLICT",
}
SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|token|password|secret)\s*[:=]\s*\S+"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]+\b"),
)
PII_PATTERNS = (
    re.compile(r"<@[A-Z0-9]+>"),
    re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
)
EVENT_INSERT_SQL = """
INSERT OR IGNORE INTO observation_events(
  observation_id, run_id, event_type, event_key, occurred_at,
  source_locator, source_hash, metadata
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""
RUN_OBSERVATION_SQL = """
INSERT INTO backfill_run_observations(
  run_id, observation_id, disposition, before_projection, after_projection
) VALUES (?, ?, ?, ?, ?)
ON CONFLICT(run_id, observation_id) DO UPDATE SET
  disposition = CASE
    WHEN disposition = 'CREATED' THEN disposition
    ELSE excluded.disposition
  END,
  before_projection = COALESCE(before_projection, excluded.before_projection),
  after_projection = excluded.after_projection
"""
EDGE_INSERT_SQL = """
INSERT INTO edges(source_id, target_id, relation, weight, metadata)
VALUES (?, ?, ?, 1.0, ?)
ON CONFLICT(source_id, target_id, relation) DO NOTHING
"""
ACCESS_UPDATE_SQL = """
UPDATE observations
SET access_count = ?, last_accessed = ?, updated_at = datetime('now')
WHERE id = ?
"""
SUPERSEDE_UPDATE_SQL = """
UPDATE observations
SET valid_until = ?, superseded_by = ?, metadata = ?,
    updated_at = datetime('now')
WHERE id = ?
"""
AUDIT_INSERT_SQL = """
INSERT INTO audit_log(operation, target_type, detail)
VALUES (?, 'backfill_run', ?)
"""
DERIVED_OBSERVATION_SQL = """
SELECT 1 FROM observations AS observation
WHERE (
  distilled_from = ?1 OR superseded_by = ?1
  OR (
    metadata IS NOT NULL AND json_valid(metadata)
    AND EXISTS (
      SELECT 1 FROM json_each(
        metadata, '$.provenance.parents'
      ) AS parent WHERE parent.value = ?1
    )
  )
)
AND NOT EXISTS (
  SELECT 1 FROM backfill_run_observations AS member
  WHERE member.run_id = ?2 AND member.observation_id = observation.id
)
LIMIT 1
"""

EmbedBatch = Callable[[list[str]], list[list[float]]]
Redactor = Callable[[str], str]


class BackfillError(RuntimeError):
    """Stable public error code plus a content-free message."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class PlannedItem:
    item: BackfillItem
    content: str
    content_hash: str
    source_hash: str
    owner: dict[str, object] | None
    status: str
    new_used_events: int
    before: dict[str, object] | None
    after: dict[str, object]


@dataclass(frozen=True)
class PlanBundle:
    run_id: str
    request_digest: str
    plan_digest: str
    entries: list[PlannedItem]
    receipt: dict[str, object]


def redact_backfill_text(text: str) -> str:
    redacted = text
    for pattern in (*SECRET_PATTERNS, *PII_PATTERNS):
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def redact_backfill_value(value: object, redactor: Redactor) -> object:
    if isinstance(value, str):
        return redactor(value)
    if isinstance(value, Mapping):
        return {
            str(key): redact_backfill_value(item, redactor)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_backfill_value(item, redactor) for item in value]
    return value


def redact_backfill_metadata(
    metadata: Mapping[str, object],
    redactor: Redactor,
) -> dict[str, object]:
    return {
        str(key): redact_backfill_value(value, redactor)
        for key, value in metadata.items()
    }


def _utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc)


def _json_value(value: object) -> object:
    if isinstance(value, datetime):
        return _utc(value).isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def canonical_json(value: object) -> str:
    normalized = _json_value(value)
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: object) -> str:
    encoded = canonical_json(value).encode("utf-8")
    return HASH_PREFIX + hashlib.sha256(encoded).hexdigest()


def content_digest(content: str) -> str:
    return digest({"content": content})


def observation_content_digest(title: str, content: str) -> str:
    return digest({"title": title, "content": content})


def _decode_metadata(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str) or not value:
        return {}
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return dict(decoded) if isinstance(decoded, dict) else {}


def _as_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        return None
    aware = parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return _utc(aware)


def _same_time(left: object, right: object) -> bool:
    return _as_datetime(left) == _as_datetime(right)


def _row_dict(row: sqlite3.Row | None) -> dict[str, object] | None:
    return None if row is None else dict(row)


def _source_payload(item: BackfillItem, content_hash: str) -> dict[str, object]:
    source = item.source
    return {
        "kind": source.kind, "namespace": source.namespace,
        "event_key": source.event_key, "recorded_at": source.recorded_at,
        "locator": source.locator, "content_hash": content_hash,
        "project": item.project, "scope": item.scope, "type": item.type,
        "memory_role": item.memory_role, "claim_status": item.claim_status,
        "valid_from": item.resolved_valid_from, "valid_until": item.valid_until,
        "historical_value": item.historical_value,
        "supersedes_topic_key": item.supersedes_topic_key,
        "metadata": item.metadata,
    }


def source_digest(item: BackfillItem, content_hash: str) -> str:
    return digest(_source_payload(item, content_hash))


def use_event_digest(item: BackfillItem, event_index: int, content_hash: str) -> str:
    event = item.use_events[event_index]
    payload = _source_payload(item, content_hash)
    payload.update(
        event_key=event.event_key,
        occurred_at=event.occurred_at,
        locator=event.locator,
        event_type=USE_EVENT,
    )
    return digest(payload)


def _request_payload(request: BackfillRequest, redactor: Redactor) -> dict[str, object]:
    payload = request.model_dump(mode="json", exclude={"mode", "expected_plan_digest"})
    items = list(payload["items"])
    for raw, item in zip(items, request.items):
        raw["title"] = redactor(item.title)
        raw["content"] = redactor(item.content)
    payload["items"] = items
    return payload


def request_digest(request: BackfillRequest, redactor: Redactor) -> str:
    return digest(_request_payload(request, redactor))


def _run_id(request: BackfillRequest) -> str:
    value = digest({"project": request.project, "run_key": request.run_key})
    return RUN_PREFIX + value.removeprefix(HASH_PREFIX)[:24]


def _item_times(item: BackfillItem) -> list[datetime]:
    times = [item.source.recorded_at, item.resolved_valid_from]
    if item.valid_until is not None:
        times.append(item.valid_until)
    times.extend(event.occurred_at for event in item.use_events)
    return times


def _validate_future_times(
    request: BackfillRequest,
    now: datetime,
    skew_seconds: int,
) -> None:
    ceiling = now + timedelta(seconds=skew_seconds)
    if any(value > ceiling for item in request.items for value in _item_times(item)):
        raise BackfillError("FUTURE_TIMESTAMP", "historical timestamp exceeds clock skew")


def _validate_scope(request: BackfillRequest, allow_global: bool) -> None:
    if not allow_global and any(item.scope == "global" for item in request.items):
        raise BackfillError("INVALID_BACKFILL_ITEM", "global scope requires local authority")


def _validate_retention(request: BackfillRequest, allow_historical: bool) -> None:
    if not allow_historical and any(item.historical_value for item in request.items):
        raise BackfillError(
            "INVALID_BACKFILL_ITEM",
            "historical_value requires local management authority",
        )


def _owner(
    db: sqlite3.Connection,
    item: BackfillItem,
) -> dict[str, object] | None:
    row = db.execute(
        """
        SELECT * FROM observations
        WHERE project = ? AND scope = ? AND topic_key = ?
        """,
        [item.project, item.scope, item.topic_key],
    ).fetchone()
    return _row_dict(row)


def _event(
    db: sqlite3.Connection,
    observation_id: int,
    event_type: str,
    event_key: str,
) -> dict[str, object] | None:
    row = db.execute(
        """
        SELECT * FROM observation_events
        WHERE observation_id = ? AND event_type = ? AND event_key = ?
        """,
        [observation_id, event_type, event_key],
    ).fetchone()
    return _row_dict(row)


def _baseline(
    db: sqlite3.Connection,
    owner: dict[str, object] | None,
) -> tuple[int, datetime | None]:
    if owner is None:
        return 0, None
    row = db.execute(
        "SELECT * FROM observation_event_baselines WHERE observation_id = ?",
        [owner["id"]],
    ).fetchone()
    if row is None:
        return int(owner["access_count"]), _as_datetime(owner.get("last_accessed"))
    return int(row["access_count"]), _as_datetime(row["last_accessed"])


def _existing_used(
    db: sqlite3.Connection,
    observation_id: int | None,
) -> list[dict[str, object]]:
    if observation_id is None:
        return []
    rows = db.execute(
        """
        SELECT event_key, occurred_at, source_hash
        FROM observation_events
        WHERE observation_id = ? AND event_type = ?
        """,
        [observation_id, USE_EVENT],
    ).fetchall()
    return [dict(row) for row in rows]


def _new_use_events(
    item: BackfillItem,
    existing: Sequence[Mapping[str, object]],
) -> list[int]:
    keys = {str(row["event_key"]) for row in existing}
    return [
        index for index, event in enumerate(item.use_events)
        if event.event_key not in keys
    ]


def _max_time(values: Sequence[datetime | None]) -> datetime | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def _retention_protected(
    item: BackfillItem,
    owner: dict[str, object] | None,
) -> bool:
    metadata = _decode_metadata(owner.get("metadata")) if owner else {}
    risk = str(metadata.get("risk_class", "LOW"))
    return bool(
        item.historical_value
        or metadata.get("historical_value")
        or metadata.get("legal_retention")
        or risk in {"HIGH", "CRITICAL"}
    )


def _projection(
    db: sqlite3.Connection, item: BackfillItem, owner: dict[str, object] | None,
    now: datetime, threshold: float,
) -> tuple[dict[str, object], int]:
    owner_id = int(owner["id"]) if owner else None
    existing = _existing_used(db, owner_id)
    new_indexes = _new_use_events(item, existing)
    base_count, base_last = _baseline(db, owner)
    last_accessed = _projected_last_accessed(item, existing, new_indexes, base_last)
    access_count = base_count + len(existing) + len(new_indexes)
    projection = _projection_receipt(
        item, owner, last_accessed, access_count, now, threshold,
    )
    return projection, len(new_indexes)


def _projected_last_accessed(
    item: BackfillItem,
    existing: Sequence[Mapping[str, object]],
    new_indexes: Sequence[int],
    baseline: datetime | None,
) -> datetime | None:
    existing_times = [_as_datetime(row["occurred_at"]) for row in existing]
    new_times = [item.use_events[index].occurred_at for index in new_indexes]
    return _max_time([baseline, *existing_times, *new_times])


def _projection_receipt(
    item: BackfillItem, owner: dict[str, object] | None,
    last_accessed: datetime | None, access_count: int,
    now: datetime, threshold: float,
) -> dict[str, object]:
    before, strength = _projection_strengths(
        item, owner, last_accessed, access_count, now,
    )
    protected = _retention_protected(item, owner)
    return {
        **_projection_times(item, last_accessed, access_count),
        "effective_strength_before": before, "effective_strength": strength,
        "would_prune": strength < threshold and not protected,
        "retention_protected": protected,
    }


def _projection_strengths(
    item: BackfillItem, owner: dict[str, object] | None,
    last_accessed: datetime | None, access_count: int, now: datetime,
) -> tuple[float, float]:
    base = float(owner["base_strength"]) if owner else 1.0
    after = current_strength(
        base, item.source.recorded_at, last_accessed, access_count, now,
    )
    return _owner_strength(owner, item, now), after


def _owner_strength(
    owner: dict[str, object] | None,
    item: BackfillItem,
    now: datetime,
) -> float:
    if owner is None:
        return 0.0
    return current_strength(
        float(owner["base_strength"]),
        _as_datetime(owner["created_at"]) or item.source.recorded_at,
        _as_datetime(owner.get("last_accessed")), int(owner["access_count"]), now,
    )


def _projection_times(
    item: BackfillItem,
    last_accessed: datetime | None,
    access_count: int,
) -> dict[str, object]:
    created_at = item.source.recorded_at
    return {
        "created_at": created_at, "valid_from": item.resolved_valid_from,
        "valid_until": item.valid_until, "access_count": access_count,
        "last_accessed": last_accessed,
        "decay_reference_time": last_accessed or created_at,
    }


def _snapshot(owner: dict[str, object] | None) -> dict[str, object] | None:
    if owner is None:
        return None
    return {
        "observation_id": owner["id"],
        "created_at": owner["created_at"],
        "updated_at": owner["updated_at"],
        "valid_from": owner["valid_from"],
        "valid_until": owner["valid_until"],
        "access_count": owner["access_count"],
        "last_accessed": owner["last_accessed"],
        "superseded_by": owner.get("superseded_by"),
        "metadata": owner.get("metadata"),
    }


def _temporal_match(owner: dict[str, object], item: BackfillItem) -> bool:
    return (
        _same_time(owner["created_at"], item.source.recorded_at)
        and _same_time(owner["valid_from"], item.resolved_valid_from)
        and _same_time(owner.get("valid_until"), item.valid_until)
    )


def _event_mutated(
    db: sqlite3.Connection,
    owner_id: int,
    item: BackfillItem,
    content_hash: str,
) -> bool:
    recorded = _event(db, owner_id, RECORD_EVENT, item.source.event_key)
    if recorded and recorded["source_hash"] != source_digest(item, content_hash):
        return True
    for index, use in enumerate(item.use_events):
        current = _event(db, owner_id, USE_EVENT, use.event_key)
        if current and current["source_hash"] != use_event_digest(item, index, content_hash):
            return True
    return False


def _repairable(
    db: sqlite3.Connection,
    owner_id: int,
    item: BackfillItem,
    source_hash: str,
) -> bool:
    recorded = _event(db, owner_id, RECORD_EVENT, item.source.event_key)
    return bool(recorded and recorded["source_hash"] == source_hash)


def _status(
    db: sqlite3.Connection, item: BackfillItem,
    owner: dict[str, object] | None, content_hash: str,
    source_hash: str, repair: bool,
) -> str:
    if owner is None:
        return "CREATE"
    owner_id = int(owner["id"])
    if not _owner_content_matches(owner, content_hash):
        return "TOPIC_CONFLICT"
    if _event_mutated(db, owner_id, item, content_hash):
        return "SOURCE_MUTATION"
    if not _temporal_match(owner, item):
        return _temporal_status(db, owner_id, item, source_hash, repair)
    return _event_status(db, owner_id, item)


def _owner_content_matches(owner: Mapping[str, object], expected: str) -> bool:
    actual = observation_content_digest(
        str(owner["title"]), str(owner["content"]),
    )
    return actual == expected


def _temporal_status(
    db: sqlite3.Connection,
    owner_id: int,
    item: BackfillItem,
    source_hash: str,
    repair: bool,
) -> str:
    allowed = repair and _repairable(db, owner_id, item, source_hash)
    return "MATCH_REPAIR" if allowed else "TEMPORAL_CONFLICT"


def _event_status(
    db: sqlite3.Connection,
    owner_id: int,
    item: BackfillItem,
) -> str:
    existing = _event(db, owner_id, RECORD_EVENT, item.source.event_key)
    new_used = _new_use_events(item, _existing_used(db, owner_id))
    return "SKIP_IDEMPOTENT" if existing and not new_used else "MATCH"


def _target_exists(
    db: sqlite3.Connection,
    project: str,
    topic_key: str,
    request_topics: set[str],
) -> bool:
    if topic_key in request_topics:
        return True
    row = db.execute(
        "SELECT 1 FROM observations WHERE project = ? AND topic_key = ?",
        [project, topic_key],
    ).fetchone()
    return row is not None


def _references_valid(
    db: sqlite3.Connection,
    item: BackfillItem,
    request_items: Mapping[str, BackfillItem],
) -> bool:
    targets = [edge.target_topic_key for edge in item.edges]
    if item.supersedes_topic_key:
        targets.append(item.supersedes_topic_key)
    if item.topic_key in targets:
        return False
    return all(
        _target_exists(db, item.project, target, set(request_items))
        for target in targets
    ) and _supersedes_time_valid(db, item, request_items)


def _supersedes_time_valid(
    db: sqlite3.Connection,
    item: BackfillItem,
    request_items: Mapping[str, BackfillItem],
) -> bool:
    target_key = item.supersedes_topic_key
    if not target_key:
        return True
    target = request_items.get(target_key)
    if target is not None:
        return target.resolved_valid_from < item.resolved_valid_from
    target_time = _target_valid_from(db, item.project, target_key)
    return target_time is not None and target_time < item.resolved_valid_from


def _target_valid_from(
    db: sqlite3.Connection,
    project: str,
    topic_key: str,
) -> datetime | None:
    row = db.execute(
        "SELECT valid_from FROM observations WHERE project = ? AND topic_key = ?",
        [project, topic_key],
    ).fetchone()
    return _as_datetime(row["valid_from"]) if row else None


def _planned_item(
    db: sqlite3.Connection, request: BackfillRequest, item: BackfillItem,
    redactor: Redactor, now: datetime, threshold: float,
) -> PlannedItem:
    title = redactor(item.title)
    content = redactor(item.content)
    content_hash = observation_content_digest(title, content)
    source_hash = source_digest(item, content_hash)
    owner = _owner(db, item)
    status = _planned_status(db, request, item, owner, content_hash, source_hash)
    projection, new_used = _projection(db, item, owner, now, threshold)
    return PlannedItem(
        item, content, content_hash, source_hash, owner, status,
        new_used, _snapshot(owner), projection,
    )


def _planned_status(
    db: sqlite3.Connection,
    request: BackfillRequest,
    item: BackfillItem,
    owner: dict[str, object] | None,
    content_hash: str,
    source_hash: str,
) -> str:
    status = _status(
        db, item, owner, content_hash, source_hash,
        request.repair_temporal_projection,
    )
    topics = {entry.topic_key: entry for entry in request.items}
    return status if _references_valid(db, item, topics) else "TOPIC_CONFLICT"


def _entry_state(
    db: sqlite3.Connection,
    entry: PlannedItem,
) -> dict[str, object]:
    owner_id = int(entry.owner["id"]) if entry.owner else None
    events = _existing_used(db, owner_id)
    recorded = _recorded_state(db, owner_id, entry)
    return {
        "topic_key": entry.item.topic_key, "owner": _snapshot(entry.owner),
        "recorded_event": recorded, "used_events": events, "status": entry.status,
        "references": _reference_state(db, entry),
    }


def _reference_state(
    db: sqlite3.Connection,
    entry: PlannedItem,
) -> list[dict[str, object] | None]:
    targets = [edge.target_topic_key for edge in entry.item.edges]
    if entry.item.supersedes_topic_key:
        targets.append(entry.item.supersedes_topic_key)
    return [
        _snapshot(_owner_by_topic(db, entry.item, topic))
        for topic in sorted(set(targets))
    ]


def _owner_by_topic(
    db: sqlite3.Connection,
    item: BackfillItem,
    topic_key: str,
) -> dict[str, object] | None:
    row = db.execute(
        "SELECT * FROM observations WHERE project = ? AND topic_key = ?",
        [item.project, topic_key],
    ).fetchone()
    return _row_dict(row)


def _recorded_state(
    db: sqlite3.Connection,
    owner_id: int | None,
    entry: PlannedItem,
) -> dict[str, object] | None:
    if owner_id is None:
        return None
    return _event(
        db, owner_id, RECORD_EVENT, entry.item.source.event_key,
    )


def _counts(entries: Sequence[PlannedItem]) -> dict[str, int]:
    counts = {
        "requested": len(entries),
        "create": 0,
        "match": 0,
        "skip_idempotent": 0,
        "conflict": 0,
        "new_used_events": 0,
        "would_prune": 0,
        "sentry_restricted": 0,
    }
    for entry in entries:
        _increment_counts(counts, entry)
    return counts


def _increment_counts(counts: dict[str, int], entry: PlannedItem) -> None:
    counts[_count_key(entry.status)] += 1
    counts["new_used_events"] += entry.new_used_events
    counts["would_prune"] += int(bool(entry.after["would_prune"]))


def _count_key(status: str) -> str:
    if status == "CREATE":
        return "create"
    if status in {"MATCH", "MATCH_REPAIR"}:
        return "match"
    if status == "SKIP_IDEMPOTENT":
        return "skip_idempotent"
    return "conflict"


def _receipt_item(
    entry: PlannedItem,
    allowed_namespaces: tuple[str, ...],
) -> dict[str, object]:
    source_class = classify_record_source(
        entry.item.source.kind,
        entry.item.source.namespace,
        allowed_namespaces,
    )
    return {
        **_receipt_identity(entry, source_class),
        **_receipt_temporal(entry),
        **_receipt_decay(entry),
    }


def _receipt_identity(entry: PlannedItem, source_class: object) -> dict[str, object]:
    return {
        "topic_key": entry.item.topic_key, "source_hash": entry.source_hash,
        "status": entry.status,
        "observation_id": entry.owner["id"] if entry.owner else None,
        "memory_role": entry.item.memory_role,
        "sentry_source_class": source_class,
    }


def _receipt_temporal(entry: PlannedItem) -> dict[str, object]:
    return {
        "recorded_at": entry.item.source.recorded_at,
        "valid_from": entry.item.resolved_valid_from,
        "valid_until": entry.item.valid_until,
    }


def _receipt_decay(entry: PlannedItem) -> dict[str, object]:
    return {
        "strength_before": _before_strength(entry),
        "strength_after": entry.after["effective_strength"],
        "decay_reference_time": entry.after["decay_reference_time"],
        "would_prune": entry.after["would_prune"],
        "retention_protected": entry.after["retention_protected"],
    }


def _before_strength(entry: PlannedItem) -> float:
    return float(entry.after["effective_strength_before"])


def _observation_temporal(item: BackfillItem) -> dict[str, object]:
    return {
        "valid_from": item.resolved_valid_from,
        "valid_until": item.valid_until,
        "base_strength": 1.0,
        "access_count": 0,
        "created_at": item.source.recorded_at,
        "updated_at": datetime.now(timezone.utc),
    }


def _baseline_values(
    disposition: str,
    before: dict[str, object] | None,
) -> tuple[int, object | None]:
    count = 0 if disposition == "CREATED" else int(before["access_count"])
    last = None if before is None else before["last_accessed"]
    return count, last


def _optional_json(value: dict[str, object] | None) -> str | None:
    return canonical_json(value) if value is not None else None


def _edge_metadata(
    run_id: str,
    relation: str,
    verified: bool,
) -> dict[str, object]:
    return {
        "backfill_run_id": run_id,
        "claim_edge": {
            "relation": relation,
            "origin": "DETERMINISTIC" if verified else "CANDIDATE",
            "status": "VERIFIED" if verified else "CANDIDATE",
        },
    }


def _rollback_receipt(
    run_id: str,
    dry_run: bool,
    created: Sequence[int],
    dependencies: set[int],
) -> dict[str, object]:
    return {
        "run_id": run_id, "dry_run": dry_run,
        "created_observations": list(created),
        "blocked_observations": sorted(dependencies),
        "status": "BLOCKED" if dependencies else "READY",
    }


class BackfillService:
    """Plan/apply/rollback orchestration over one ERINYS SQLite connection."""

    def __init__(
        self,
        db: sqlite3.Connection,
        config: ErinysConfig,
        embed_batch: EmbedBatch,
        redactor: Redactor = redact_backfill_text,
        principal: str = DEFAULT_PRINCIPAL,
        max_items: int | None = None,
    ) -> None:
        self.db = db
        self.config = config
        self.embed_batch = embed_batch
        self.redactor = redactor
        self.principal = principal
        self.max_items = max_items or config.backfill_max_items

    def _validate(
        self,
        request: BackfillRequest,
        allow_historical: bool,
        allow_global: bool,
    ) -> None:
        if len(request.items) > self.max_items:
            raise BackfillError("INVALID_BACKFILL_ITEM", "backfill item limit exceeded")
        _validate_future_times(
            request, datetime.now(timezone.utc),
            self.config.backfill_clock_skew_seconds,
        )
        _validate_scope(request, allow_global)
        _validate_retention(request, allow_historical)
        self._validate_source_authority(request)

    def _validate_source_authority(self, request: BackfillRequest) -> None:
        for item in request.items:
            source_class = classify_record_source(
                item.source.kind, item.source.namespace,
                self.config.first_party_record_namespaces,
            )
            if item.supersedes_topic_key and source_class is not SourceClass.FIRST_PARTY_RECORD:
                raise BackfillError(
                    "INVALID_BACKFILL_ITEM",
                    "supersedes requires an allowlisted first-party record",
                )

    def plan(
        self,
        request: BackfillRequest,
        *,
        allow_historical: bool = False,
        allow_global: bool = False,
    ) -> dict[str, object]:
        return self._build_plan(request, allow_historical, allow_global).receipt

    def _build_plan(
        self,
        request: BackfillRequest,
        allow_historical: bool,
        allow_global: bool,
    ) -> PlanBundle:
        self._validate(request, allow_historical, allow_global)
        entries = self._plan_entries(request)
        req_digest = request_digest(request, self.redactor)
        state = [_entry_state(self.db, entry) for entry in entries]
        plan_digest = digest({"request_digest": req_digest, "db_state": state})
        receipt = self._plan_receipt(request, entries, req_digest, plan_digest)
        return PlanBundle(_run_id(request), req_digest, plan_digest, entries, receipt)

    def _plan_entries(self, request: BackfillRequest) -> list[PlannedItem]:
        now = datetime.now(timezone.utc)
        return [
            _planned_item(
                self.db, request, item, self.redactor, now,
                self.config.prune_threshold,
            )
            for item in request.items
        ]

    def _plan_receipt(
        self, request: BackfillRequest, entries: list[PlannedItem],
        req_digest: str, plan_digest: str,
    ) -> dict[str, object]:
        counts = _counts(entries)
        items = self._receipt_items(entries)
        counts["sentry_restricted"] = sum(
            int(item["sentry_source_class"] == "external") for item in items
        )
        return {
            "run_id": _run_id(request), "run_key": request.run_key,
            "mode": BackfillMode.PLAN, "request_digest": req_digest,
            "plan_digest": plan_digest, "counts": counts, "items": items,
            "vector_health": "not_mutated", "warnings": [],
        }

    def _receipt_items(
        self,
        entries: Sequence[PlannedItem],
    ) -> list[dict[str, object]]:
        return [
            _receipt_item(entry, self.config.first_party_record_namespaces)
            for entry in entries
        ]

    def apply(
        self,
        request: BackfillRequest,
        *,
        allow_historical: bool = False,
        allow_global: bool = False,
    ) -> dict[str, object]:
        if not self.config.backfill_apply_enabled:
            raise BackfillError("BACKFILL_DISABLED", "backfill apply is disabled")
        bundle = self._build_plan(request, allow_historical, allow_global)
        existing = self._existing_run(bundle.run_id)
        self._validate_digest(request, bundle, existing)
        if existing and existing["status"] == "APPLIED":
            return self._applied_receipt(bundle, existing, "already_applied", None)
        return self._apply_new(request, bundle, existing)

    def _apply_new(
        self, request: BackfillRequest, bundle: PlanBundle,
        existing: dict[str, object] | None,
    ) -> dict[str, object]:
        self._raise_conflicts(bundle.entries)
        embeddings = self._embed_creates(bundle.entries)
        backup = backup_database(self.db, bundle.run_id)
        self._start_run(request, bundle, existing)
        completed = self._apply_chunks(bundle, embeddings)
        self._apply_edges(bundle)
        self._ensure_storage_health(bundle)
        self._complete_run(bundle, completed)
        row = self._existing_run(bundle.run_id)
        return self._applied_receipt(bundle, row, "applied", backup)

    def _existing_run(self, run_id: str) -> dict[str, object] | None:
        row = self.db.execute(
            "SELECT * FROM backfill_runs WHERE run_id = ?",
            [run_id],
        ).fetchone()
        return _row_dict(row)

    def _validate_digest(
        self,
        request: BackfillRequest,
        bundle: PlanBundle,
        existing: dict[str, object] | None,
    ) -> None:
        expected = request.expected_plan_digest
        actual = existing["plan_digest"] if existing else bundle.plan_digest
        if expected != actual:
            raise BackfillError("PLAN_DIGEST_MISMATCH", "plan digest changed")
        if existing and existing["request_digest"] != bundle.request_digest:
            raise BackfillError("PLAN_DIGEST_MISMATCH", "run key request changed")

    def _raise_conflicts(self, entries: Sequence[PlannedItem]) -> None:
        conflicts = [entry.status for entry in entries if entry.status in PLAN_CONFLICTS]
        if not conflicts:
            return
        status = conflicts[0]
        code = "TOPIC_CONFLICT" if status == "TEMPORAL_CONFLICT" else status
        raise BackfillError(code, f"backfill plan contains {status}")

    def _embed_creates(self, entries: Sequence[PlannedItem]) -> dict[str, bytes]:
        targets = [entry for entry in entries if self._needs_embedding(entry)]
        if not targets:
            return {}
        try:
            vectors = self.embed_batch([entry.content for entry in targets])
        except Exception as exc:
            raise BackfillError("EMBEDDING_FAILED", "embedding failed before write") from exc
        if len(vectors) != len(targets):
            raise BackfillError("EMBEDDING_FAILED", "embedding count mismatch")
        return {
            entry.item.topic_key: serialize_f32(vector)
            for entry, vector in zip(targets, vectors)
        }

    def _needs_embedding(self, entry: PlannedItem) -> bool:
        if entry.status == "CREATE":
            return True
        owner_id = int(entry.owner["id"]) if entry.owner else None
        return owner_id is not None and not self._row_exists(
            "vec_observations", owner_id,
        )

    def _start_run(
        self,
        request: BackfillRequest,
        bundle: PlanBundle,
        existing: dict[str, object] | None,
    ) -> None:
        if existing is None:
            self._insert_run(request, bundle)
        self.db.execute(
            "UPDATE backfill_runs SET status = 'APPLYING', completed_at = NULL WHERE run_id = ?",
            [bundle.run_id],
        )
        self.db.commit()

    def _insert_run(self, request: BackfillRequest, bundle: PlanBundle) -> None:
        self.db.execute(
            """
            INSERT INTO backfill_runs(
              run_id, run_key, request_digest, plan_digest, project,
              source_kind, source_namespace, status, principal, counts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'PLANNED', ?, ?)
            """,
            [
                bundle.run_id, request.run_key, bundle.request_digest,
                bundle.plan_digest, request.project, request.source_kind,
                request.source_namespace, self.principal,
                canonical_json(bundle.receipt["counts"]),
            ],
        )

    def _chunks(self, entries: Sequence[PlannedItem]) -> list[Sequence[PlannedItem]]:
        size = min(self.config.backfill_chunk_size, self.config.backfill_max_items)
        return [entries[index:index + size] for index in range(0, len(entries), size)]

    def _apply_chunks(
        self,
        bundle: PlanBundle,
        embeddings: dict[str, bytes],
    ) -> int:
        completed = 0
        for chunk in self._chunks(bundle.entries):
            try:
                self._apply_chunk(bundle.run_id, chunk, embeddings)
                completed += len(chunk)
            except Exception as exc:
                self._mark_failed(bundle.run_id, completed)
                raise BackfillError("TRANSACTION_FAILED", "backfill chunk failed") from exc
        return completed

    def _apply_chunk(
        self,
        run_id: str,
        entries: Sequence[PlannedItem],
        embeddings: dict[str, bytes],
    ) -> None:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            for entry in entries:
                self._apply_item(run_id, entry, embeddings)
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def _apply_item(
        self,
        run_id: str,
        entry: PlannedItem,
        embeddings: dict[str, bytes],
    ) -> None:
        owner = _owner(self.db, entry.item)
        obs_id, disposition, before = self._resolve_item(
            entry, embeddings, owner,
        )
        self._ensure_baseline(obs_id, disposition, before)
        self._insert_events(run_id, obs_id, entry)
        self._rebuild_projection(obs_id)
        after = _snapshot(self._observation(obs_id))
        self._record_run_observation(run_id, obs_id, disposition, before, after)

    def _resolve_item(
        self,
        entry: PlannedItem,
        embeddings: dict[str, bytes],
        owner: dict[str, object] | None,
    ) -> tuple[int, str, dict[str, object] | None]:
        if owner is None:
            return self._create_observation(entry, embeddings), "CREATED", None
        obs_id = int(owner["id"])
        self._repair_existing_membership(obs_id, entry, embeddings)
        if entry.status == "MATCH_REPAIR":
            self._repair_temporal(obs_id, entry)
        return obs_id, "MATCHED", _snapshot(owner)

    def _repair_existing_membership(
        self, obs_id: int, entry: PlannedItem, embeddings: dict[str, bytes],
    ) -> None:
        if not self._row_exists("vec_observations", obs_id):
            blob = embeddings.get(entry.item.topic_key)
            if blob is None:
                raise RuntimeError("missing recovery embedding")
            self.db.execute(
                "INSERT INTO vec_observations(rowid, embedding) VALUES (?, ?)",
                [obs_id, blob],
            )
        if not self._row_exists("observations_fts", obs_id):
            self._repair_fts(obs_id)

    def _repair_fts(self, obs_id: int) -> None:
        row = self._observation(obs_id)
        self.db.execute(
            """
            INSERT INTO observations_fts(rowid, title, content, project, type)
            VALUES (?, ?, ?, ?, ?)
            """,
            [obs_id, row["title"], row["content"], row["project"], row["type"]],
        )

    def _create_observation(
        self,
        entry: PlannedItem,
        embeddings: dict[str, bytes],
    ) -> int:
        blob = embeddings.get(entry.item.topic_key)
        if blob is None:
            raise RuntimeError("missing precomputed embedding")
        payload = self._observation_payload(entry)
        return insert_observation_vector_rows(self.db, payload, blob)

    def _observation_payload(self, entry: PlannedItem) -> dict[str, object]:
        item = entry.item
        metadata = redact_backfill_metadata(item.metadata, self.redactor)
        metadata.update(self._server_metadata(entry))
        return {
            **self._observation_identity(entry),
            **_observation_temporal(item),
            "embedding_model": self.config.embedding_model,
            "metadata": _json_value(metadata),
        }

    def _observation_identity(self, entry: PlannedItem) -> dict[str, object]:
        item = entry.item
        return {
            "title": self.redactor(item.title), "content": entry.content,
            "type": item.type, "project": item.project, "scope": item.scope,
            "is_anti_pattern": int(item.type == "anti_pattern"),
            "is_pattern": int(item.type == "pattern"),
            "source": BACKFILL_SOURCE, "topic_key": item.topic_key,
        }

    def _server_metadata(self, entry: PlannedItem) -> dict[str, object]:
        item = entry.item
        return {
            "memory_role": item.memory_role, "claim_status": item.claim_status,
            "verification_state": item.verification_state,
            "historical_value": item.historical_value,
            "source_recorded_at": item.source.recorded_at.isoformat(),
            "source_namespace": item.source.namespace,
            "source_event_hash": entry.source_hash,
            "imported_at": datetime.now(timezone.utc).isoformat(),
            "provenance": self._backfill_provenance(item),
            "memory_safety": self._record_safety(item),
        }

    def _backfill_provenance(self, item: BackfillItem) -> dict[str, object]:
        return build_provenance(
            BACKFILL_SOURCE, self.principal, "backfill", [],
            item.source.recorded_at.isoformat(),
        )

    def _record_safety(self, item: BackfillItem) -> dict[str, object]:
        return build_record_memory_safety(
            item.source.kind, item.source.namespace,
            self.config.first_party_record_namespaces,
        )

    def _repair_temporal(self, obs_id: int, entry: PlannedItem) -> None:
        self.db.execute(
            """
            UPDATE observations
            SET created_at = ?, valid_from = ?, valid_until = ?,
                updated_at = datetime('now')
            WHERE id = ?
            """,
            [
                entry.item.source.recorded_at,
                entry.item.resolved_valid_from,
                entry.item.valid_until,
                obs_id,
            ],
        )

    def _ensure_baseline(
        self,
        obs_id: int,
        disposition: str,
        before: dict[str, object] | None,
    ) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO observation_event_baselines"
            "(observation_id, access_count, last_accessed) VALUES (?, ?, ?)",
            [obs_id, *_baseline_values(disposition, before)],
        )

    def _insert_events(self, run_id: str, obs_id: int, entry: PlannedItem) -> None:
        item = entry.item
        self._insert_event(
            run_id, obs_id, RECORD_EVENT, item.source.event_key,
            item.source.recorded_at, item.source.locator, entry.source_hash,
            {"memory_role": item.memory_role},
        )
        for index, event in enumerate(item.use_events):
            self._insert_event(
                run_id, obs_id, USE_EVENT, event.event_key,
                event.occurred_at, event.locator,
                use_event_digest(item, index, entry.content_hash),
                {"verified_historical_use": True},
            )

    def _insert_event(
        self, run_id: str, obs_id: int, event_type: str, event_key: str,
        occurred_at: datetime, locator: str | None, source_hash: str,
        metadata: dict[str, object],
    ) -> None:
        current = _event(self.db, obs_id, event_type, event_key)
        if current and current["source_hash"] != source_hash:
            raise BackfillError("SOURCE_MUTATION", "event source hash changed")
        self.db.execute(
            EVENT_INSERT_SQL,
            [obs_id, run_id, event_type, event_key, occurred_at,
             locator, source_hash, canonical_json(metadata)],
        )

    def _rebuild_projection(self, obs_id: int) -> None:
        baseline = self._event_baseline(obs_id)
        aggregate = self._used_aggregate(obs_id)
        count = int(baseline["access_count"]) + int(aggregate["count"])
        last = _max_time([
            _as_datetime(baseline["last_accessed"]),
            _as_datetime(aggregate["last"]),
        ])
        self.db.execute(ACCESS_UPDATE_SQL, [count, last, obs_id])

    def _event_baseline(self, obs_id: int) -> sqlite3.Row:
        row = self.db.execute(
            "SELECT * FROM observation_event_baselines WHERE observation_id = ?",
            [obs_id],
        ).fetchone()
        if row is None:
            raise LookupError(f"event baseline not found: {obs_id}")
        return row

    def _used_aggregate(self, obs_id: int) -> sqlite3.Row:
        return self.db.execute(
            "SELECT COUNT(*) AS count, MAX(occurred_at) AS last "
            "FROM observation_events WHERE observation_id = ? AND event_type = ?",
            [obs_id, USE_EVENT],
        ).fetchone()

    def _observation(self, obs_id: int) -> dict[str, object]:
        row = self.db.execute(
            "SELECT * FROM observations WHERE id = ?",
            [obs_id],
        ).fetchone()
        if row is None:
            raise LookupError(f"observation not found: {obs_id}")
        return dict(row)

    def _record_run_observation(
        self,
        run_id: str,
        obs_id: int,
        disposition: str,
        before: dict[str, object] | None,
        after: dict[str, object] | None,
    ) -> None:
        self.db.execute(
            RUN_OBSERVATION_SQL,
            [run_id, obs_id, disposition, _optional_json(before), _optional_json(after)],
        )

    def _apply_edges(self, bundle: PlanBundle) -> None:
        try:
            self.db.execute("BEGIN IMMEDIATE")
            for entry in bundle.entries:
                self._item_edges(bundle.run_id, entry)
            self.db.commit()
        except Exception as exc:
            self.db.rollback()
            self._mark_failed(bundle.run_id, len(bundle.entries))
            raise BackfillError("TRANSACTION_FAILED", "backfill edge write failed") from exc

    def _item_edges(self, run_id: str, entry: PlannedItem) -> None:
        source_id = self._topic_id(entry.item)
        verified = self._edge_verified(entry.item)
        for edge in entry.item.edges:
            target_id = self._topic_id(entry.item, edge.target_topic_key)
            self._insert_edge(
                run_id, source_id, target_id, edge.relation, verified,
            )
        if entry.item.supersedes_topic_key:
            target_id = self._topic_id(entry.item, entry.item.supersedes_topic_key)
            self._insert_edge(run_id, source_id, target_id, "supersedes", verified)
            self._supersede_target(run_id, source_id, target_id, entry)

    def _edge_verified(self, item: BackfillItem) -> bool:
        source_class = classify_record_source(
            item.source.kind, item.source.namespace,
            self.config.first_party_record_namespaces,
        )
        return source_class is SourceClass.FIRST_PARTY_RECORD

    def _topic_id(self, item: BackfillItem, topic_key: str | None = None) -> int:
        row = self.db.execute(
            "SELECT id FROM observations WHERE project = ? AND topic_key = ?",
            [item.project, topic_key or item.topic_key],
        ).fetchone()
        if row is None:
            raise LookupError("edge topic not found")
        return int(row["id"])

    def _insert_edge(
        self, run_id: str, source_id: int, target_id: int,
        relation: object, verified: bool,
    ) -> None:
        relation_value = str(relation)
        self.db.execute(
            EDGE_INSERT_SQL,
            [
                source_id, target_id, relation_value,
                canonical_json(_edge_metadata(run_id, relation_value, verified)),
            ],
        )

    def _supersede_target(
        self,
        run_id: str,
        source_id: int,
        target_id: int,
        entry: PlannedItem,
    ) -> None:
        before = _snapshot(self._observation(target_id))
        valid_until = entry.item.resolved_valid_from
        self._mark_superseded(source_id, target_id, valid_until, before)
        self._record_supersession(run_id, target_id, source_id, entry)
        after = _snapshot(self._observation(target_id))
        self._record_run_observation(
            run_id, target_id, "MATCHED", before, after,
        )

    def _mark_superseded(
        self,
        source_id: int,
        target_id: int,
        valid_until: datetime,
        before: dict[str, object],
    ) -> None:
        metadata = _decode_metadata(before["metadata"])
        metadata["claim_status"] = "SUPERSEDED"
        self.db.execute(
            SUPERSEDE_UPDATE_SQL,
            [valid_until, source_id, canonical_json(metadata), target_id],
        )

    def _record_supersession(
        self,
        run_id: str,
        target_id: int,
        source_id: int,
        entry: PlannedItem,
    ) -> None:
        valid_until = entry.item.resolved_valid_from
        self._insert_event(
            run_id, target_id, SUPERSEDED_EVENT, entry.item.source.event_key,
            valid_until, entry.item.source.locator, entry.source_hash,
            {"superseded_by": source_id},
        )

    def _mark_failed(self, run_id: str, completed: int) -> None:
        status = "PARTIAL" if completed else "FAILED"
        self.db.execute(
            "UPDATE backfill_runs SET status = ?, completed_at = datetime('now') WHERE run_id = ?",
            [status, run_id],
        )
        self.db.commit()

    def _ensure_storage_health(self, bundle: PlanBundle) -> None:
        health = self._storage_health(bundle.entries)
        if bool(health["complete"]):
            return
        self._mark_failed(bundle.run_id, len(bundle.entries))
        raise BackfillError(
            "TRANSACTION_FAILED", "backfill storage membership check failed",
        )

    def _complete_run(self, bundle: PlanBundle, completed: int) -> None:
        counts = dict(bundle.receipt["counts"])
        counts["completed"] = completed
        self.db.execute(
            """
            UPDATE backfill_runs
            SET status = 'APPLIED', counts = ?, completed_at = datetime('now')
            WHERE run_id = ?
            """,
            [canonical_json(counts), bundle.run_id],
        )
        self._audit("backfill_apply", bundle.run_id, counts)
        self.db.commit()

    def _audit(
        self,
        operation: str,
        run_id: str,
        detail: dict[str, object],
    ) -> None:
        if not self.config.enable_audit_log:
            return
        safe = {"run_id": run_id, "counts": detail}
        self.db.execute(AUDIT_INSERT_SQL, [operation, canonical_json(safe)])

    def _applied_receipt(
        self,
        bundle: PlanBundle,
        run: dict[str, object] | None,
        outcome: str,
        backup: str | None,
    ) -> dict[str, object]:
        receipt = dict(bundle.receipt)
        receipt.update(self._apply_receipt_status(run, outcome, backup))
        if run is not None:
            receipt["request_digest"] = run["request_digest"]
            receipt["plan_digest"] = run["plan_digest"]
        receipt["items"] = self._refresh_item_receipts(bundle.entries)
        receipt["storage_health"] = self._storage_health(bundle.entries)
        return receipt

    def _apply_receipt_status(
        self,
        run: dict[str, object] | None,
        outcome: str,
        backup: str | None,
    ) -> dict[str, object]:
        return {
            "mode": BackfillMode.APPLY, "outcome": outcome,
            "status": run["status"] if run else "APPLIED",
            "backup_path": backup, "vector_health": self._vector_health(),
        }

    def _refresh_item_receipts(
        self,
        entries: Sequence[PlannedItem],
    ) -> list[dict[str, object]]:
        receipts: list[dict[str, object]] = []
        for entry in entries:
            receipt = _receipt_item(
                entry, self.config.first_party_record_namespaces,
            )
            owner = _owner(self.db, entry.item)
            receipt["observation_id"] = owner["id"] if owner else None
            receipts.append(receipt)
        return receipts

    def _vector_health(self) -> dict[str, int]:
        missing = self.db.execute(
            """
            SELECT COUNT(*) FROM observations
            WHERE id NOT IN (SELECT rowid FROM vec_observations)
            """
        ).fetchone()[0]
        orphan = self.db.execute(
            """
            SELECT COUNT(*) FROM vec_observations
            WHERE rowid NOT IN (SELECT id FROM observations)
            """
        ).fetchone()[0]
        return {"missing": int(missing), "orphan": int(orphan)}

    def _storage_health(
        self,
        entries: Sequence[PlannedItem],
    ) -> dict[str, object]:
        counts = {
            "missing_observations": 0, "missing_vectors": 0,
            "missing_fts": 0, "missing_events": 0, "missing_edges": 0,
        }
        for entry in entries:
            self._add_item_health(counts, entry)
        return {**counts, "complete": not any(counts.values())}

    def _add_item_health(
        self,
        counts: dict[str, int],
        entry: PlannedItem,
    ) -> None:
        owner = _owner(self.db, entry.item)
        if owner is None:
            counts["missing_observations"] += 1
            return
        obs_id = int(owner["id"])
        counts["missing_vectors"] += int(not self._row_exists("vec_observations", obs_id))
        counts["missing_fts"] += int(not self._row_exists("observations_fts", obs_id))
        counts["missing_events"] += self._missing_item_events(obs_id, entry)
        counts["missing_edges"] += self._missing_item_edges(obs_id, entry)

    def _row_exists(self, table: str, obs_id: int) -> bool:
        column = "id" if table == "observations" else "rowid"
        row = self.db.execute(
            f"SELECT 1 FROM {table} WHERE {column} = ? LIMIT 1",
            [obs_id],
        ).fetchone()
        return row is not None

    def _missing_item_events(self, obs_id: int, entry: PlannedItem) -> int:
        expected = [(RECORD_EVENT, entry.item.source.event_key)]
        expected.extend((USE_EVENT, event.event_key) for event in entry.item.use_events)
        return sum(
            int(_event(self.db, obs_id, event_type, event_key) is None)
            for event_type, event_key in expected
        )

    def _missing_item_edges(self, obs_id: int, entry: PlannedItem) -> int:
        expected = [
            (str(edge.relation), edge.target_topic_key) for edge in entry.item.edges
        ]
        if entry.item.supersedes_topic_key:
            expected.append(("supersedes", entry.item.supersedes_topic_key))
        return sum(
            int(not self._edge_exists(obs_id, relation, entry.item, target))
            for relation, target in expected
        )

    def _edge_exists(
        self, source_id: int, relation: str,
        item: BackfillItem, target_topic: str,
    ) -> bool:
        target_id = self._topic_id(item, target_topic)
        row = self.db.execute(
            """
            SELECT 1 FROM edges
            WHERE source_id = ? AND target_id = ? AND relation = ? LIMIT 1
            """,
            [source_id, target_id, relation],
        ).fetchone()
        return row is not None

    def rollback(self, run_id: str, dry_run: bool = True) -> dict[str, object]:
        run = self._existing_run(run_id)
        if run is None:
            raise BackfillError("INVALID_BACKFILL_ITEM", "backfill run not found")
        created = self._run_observation_ids(run_id, "CREATED")
        members = self._run_member_ids(run_id)
        dependencies = self._rollback_dependencies(run_id, created, members)
        receipt = _rollback_receipt(run_id, dry_run, created, dependencies)
        if dry_run or dependencies:
            self._raise_rollback_dependency(dry_run, dependencies)
            return receipt
        self._execute_rollback(run_id, created)
        return self._finish_rollback_receipt(receipt, run_id, created)

    def _finish_rollback_receipt(
        self, receipt: dict[str, object], run_id: str, created: Sequence[int],
    ) -> dict[str, object]:
        receipt["status"] = "ROLLED_BACK"
        receipt["vector_health"] = self._vector_health()
        receipt["membership"] = self._rollback_membership(run_id, created)
        return receipt

    def _raise_rollback_dependency(
        self,
        dry_run: bool,
        dependencies: set[int],
    ) -> None:
        if dependencies and not dry_run:
            raise BackfillError("ROLLBACK_DEPENDENCY", "rollback has dependencies")

    def _run_observation_ids(self, run_id: str, disposition: str) -> list[int]:
        rows = self.db.execute(
            """
            SELECT observation_id FROM backfill_run_observations
            WHERE run_id = ? AND disposition = ?
            """,
            [run_id, disposition],
        ).fetchall()
        return [int(row["observation_id"]) for row in rows]

    def _run_member_ids(self, run_id: str) -> list[int]:
        rows = self.db.execute(
            "SELECT observation_id FROM backfill_run_observations WHERE run_id = ?",
            [run_id],
        ).fetchall()
        return [int(row["observation_id"]) for row in rows]

    def _rollback_dependencies(
        self, run_id: str, created: Sequence[int], members: Sequence[int],
    ) -> set[int]:
        blocked = {
            obs_id for obs_id in members
            if self._has_projection_drift(run_id, obs_id)
        }
        blocked.update(
            obs_id for obs_id in created
            if self._has_created_dependency(run_id, obs_id)
        )
        return blocked

    def _has_created_dependency(self, run_id: str, obs_id: int) -> bool:
        return any((
            self._has_later_event(run_id, obs_id),
            self._has_external_edge(run_id, obs_id),
            self._has_derived_observation(run_id, obs_id),
            self._has_collision(obs_id),
        ))

    def _has_projection_drift(self, run_id: str, obs_id: int) -> bool:
        row = self.db.execute(
            """
            SELECT after_projection FROM backfill_run_observations
            WHERE run_id = ? AND observation_id = ?
            """,
            [run_id, obs_id],
        ).fetchone()
        expected = _decode_metadata(row["after_projection"]) if row else {}
        current = _snapshot(self._observation(obs_id))
        return canonical_json(current) != canonical_json(expected)

    def _has_later_event(self, run_id: str, obs_id: int) -> bool:
        row = self.db.execute(
            """
            SELECT 1 FROM observation_events
            WHERE observation_id = ? AND run_id <> ? LIMIT 1
            """,
            [obs_id, run_id],
        ).fetchone()
        return row is not None

    def _has_external_edge(self, run_id: str, obs_id: int) -> bool:
        rows = self.db.execute(
            "SELECT metadata FROM edges WHERE source_id = ? OR target_id = ?",
            [obs_id, obs_id],
        ).fetchall()
        return any(
            _decode_metadata(row["metadata"]).get("backfill_run_id") != run_id
            for row in rows
        )

    def _has_derived_observation(self, run_id: str, obs_id: int) -> bool:
        row = self.db.execute(
            DERIVED_OBSERVATION_SQL,
            [obs_id, run_id],
        ).fetchone()
        return row is not None

    def _has_collision(self, obs_id: int) -> bool:
        row = self.db.execute(
            "SELECT 1 FROM collisions WHERE source_a = ? OR source_b = ? LIMIT 1",
            [obs_id, obs_id],
        ).fetchone()
        return row is not None

    def _execute_rollback(self, run_id: str, created: Sequence[int]) -> None:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self._rollback_rows(run_id, created)
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def _rollback_rows(self, run_id: str, created: Sequence[int]) -> None:
        self._delete_run_edges(run_id)
        self.db.execute("DELETE FROM observation_events WHERE run_id = ?", [run_id])
        self._restore_matched(run_id)
        self._delete_created(created)
        self.db.execute(
            "UPDATE backfill_runs SET status = 'ROLLED_BACK', "
            "completed_at = datetime('now') WHERE run_id = ?",
            [run_id],
        )
        self._audit("backfill_rollback", run_id, {"created": len(created)})

    def _delete_run_edges(self, run_id: str) -> None:
        rows = self.db.execute("SELECT id, metadata FROM edges").fetchall()
        ids = [
            int(row["id"]) for row in rows
            if _decode_metadata(row["metadata"]).get("backfill_run_id") == run_id
        ]
        if ids:
            marks = ",".join("?" for _ in ids)
            self.db.execute(f"DELETE FROM edges WHERE id IN ({marks})", ids)

    def _restore_matched(self, run_id: str) -> None:
        rows = self.db.execute(
            """
            SELECT observation_id, before_projection
            FROM backfill_run_observations
            WHERE run_id = ? AND disposition = 'MATCHED'
            """,
            [run_id],
        ).fetchall()
        for row in rows:
            before = _decode_metadata(row["before_projection"])
            self._restore_snapshot(int(row["observation_id"]), before)
            self._rebuild_projection(int(row["observation_id"]))

    def _restore_snapshot(self, obs_id: int, before: dict[str, object]) -> None:
        self.db.execute(
            """
            UPDATE observations
            SET created_at = ?, valid_from = ?, valid_until = ?,
                superseded_by = ?, metadata = ?, updated_at = datetime('now')
            WHERE id = ?
            """,
            [
                before.get("created_at"), before.get("valid_from"),
                before.get("valid_until"), before.get("superseded_by"),
                before.get("metadata"), obs_id,
            ],
        )

    def _delete_created(self, ids: Sequence[int]) -> None:
        for obs_id in ids:
            self.db.execute(
                "DELETE FROM vec_observations WHERE rowid = ?",
                [obs_id],
            )
            self.db.execute("DELETE FROM observations WHERE id = ?", [obs_id])

    def _rollback_membership(
        self,
        run_id: str,
        created: Sequence[int],
    ) -> dict[str, object]:
        residual = _created_residuals(self.db, created)
        row = self.db.execute(
            "SELECT COUNT(*) FROM observation_events WHERE run_id = ?",
            [run_id],
        ).fetchone()
        residual["run_events"] = int(row[0])
        return {"residual": residual, "complete": not any(residual.values())}


def _database_path(db: sqlite3.Connection) -> Path | None:
    rows = db.execute("PRAGMA database_list").fetchall()
    path = str(rows[0][2]) if rows else ""
    return Path(path) if path else None


def backup_database(db: sqlite3.Connection, run_id: str) -> str | None:
    source = _database_path(db)
    if source is None:
        return None
    backup_dir = source.parent / ".erinys-backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = backup_dir / f"{source.name}.{run_id}.{stamp}.{uuid.uuid4().hex[:8]}.db"
    target_db = sqlite3.connect(target)
    try:
        db.backup(target_db)
    finally:
        target_db.close()
    return str(target)


def _created_residuals(
    db: sqlite3.Connection,
    ids: Sequence[int],
) -> dict[str, int]:
    if not ids:
        return _empty_residuals()
    marks = ",".join("?" for _ in ids)
    return {
        "observations": _count(db, f"id IN ({marks})", ids, "observations"),
        "vectors": _count(db, f"rowid IN ({marks})", ids, "vec_observations"),
        "fts": _count(db, f"rowid IN ({marks})", ids, "observations_fts"),
        "edges": _edge_count(db, marks, ids),
        "collisions": _collision_count(db, marks, ids),
    }


def _empty_residuals() -> dict[str, int]:
    return {
        "observations": 0, "vectors": 0, "fts": 0,
        "edges": 0, "collisions": 0,
    }


def _count(
    db: sqlite3.Connection,
    where: str,
    params: Sequence[int],
    table: str,
) -> int:
    row = db.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}", params).fetchone()
    return int(row[0])


def _edge_count(
    db: sqlite3.Connection,
    marks: str,
    ids: Sequence[int],
) -> int:
    where = f"source_id IN ({marks}) OR target_id IN ({marks})"
    return _count(db, where, [*ids, *ids], "edges")


def _collision_count(
    db: sqlite3.Connection,
    marks: str,
    ids: Sequence[int],
) -> int:
    where = f"source_a IN ({marks}) OR source_b IN ({marks})"
    return _count(db, where, [*ids, *ids], "collisions")
