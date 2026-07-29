"""Typed public contracts for timestamp-aware historical backfill."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, TypeAlias

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from ._compat import StrEnum

MAX_BACKFILL_ITEMS = 1_000
MAX_EVENT_KEY_LENGTH = 500
MAX_LOCATOR_LENGTH = 2_000
MAX_METADATA_KEYS = 100
MAX_TITLE_LENGTH = 500
MAX_CONTENT_LENGTH = 50_000
MAX_EDGES_PER_ITEM = 1_000
MAX_EVENTS_PER_REQUEST = 50_000
MAX_USE_EVENTS_PER_ITEM = 10_000
RESERVED_METADATA_KEYS = {
    "action_authority",
    "backfill_run_id",
    "base_strength",
    "claim_status",
    "created_at",
    "historical_value",
    "imported_at",
    "legal_retention",
    "memory_role",
    "memory_safety",
    "policy_authority",
    "principal",
    "provenance",
    "retention_protected",
    "risk_class",
    "source_event_hash",
    "source_class",
    "source_namespace",
    "source_recorded_at",
    "updated_at",
    "verification_state",
}

ObservationType: TypeAlias = Literal[
    "manual",
    "decision",
    "architecture",
    "bugfix",
    "pattern",
    "config",
    "discovery",
    "learning",
    "anti_pattern",
    "meta_knowledge",
]


class BackfillMode(StrEnum):
    PLAN = "plan"
    APPLY = "apply"


class MemoryRole(StrEnum):
    EVIDENCE = "EVIDENCE"
    CLAIM = "CLAIM"
    DECISION = "DECISION"
    OUTCOME = "OUTCOME"


class ClaimStatus(StrEnum):
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    CONTESTED = "CONTESTED"
    REFUTED = "REFUTED"
    RETRACTED = "RETRACTED"
    ARCHIVED = "ARCHIVED"


class SourceKind(StrEnum):
    SLACK = "slack"
    EMAIL = "email"
    CHAT = "chat"
    WEB = "web"
    ARCHIVE = "archive"


class BackfillRelation(StrEnum):
    DERIVED_FROM = "derived_from"
    USED_IN = "used_in"
    RESULTED_IN = "resulted_in"
    VERIFIED_BY = "verified_by"
    SUPPORTS = "supports"
    NARROWS = "narrows"
    APPLICABLE_TO = "applicable_to"


class StrictModel(BaseModel):
    """Reject fields that would otherwise be silently discarded."""

    model_config = ConfigDict(extra="forbid")


def _as_utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc)


class BackfillSource(StrictModel):
    kind: SourceKind
    namespace: str = Field(min_length=1, max_length=MAX_EVENT_KEY_LENGTH)
    event_key: str = Field(min_length=1, max_length=MAX_EVENT_KEY_LENGTH)
    recorded_at: AwareDatetime
    locator: str | None = Field(default=None, max_length=MAX_LOCATOR_LENGTH)

    @field_validator("recorded_at")
    @classmethod
    def normalize_recorded_at(cls, value: datetime) -> datetime:
        return _as_utc(value)


class HistoricalUseEvent(StrictModel):
    event_key: str = Field(min_length=1, max_length=MAX_EVENT_KEY_LENGTH)
    occurred_at: AwareDatetime
    locator: str | None = Field(default=None, max_length=MAX_LOCATOR_LENGTH)

    @field_validator("occurred_at")
    @classmethod
    def normalize_occurred_at(cls, value: datetime) -> datetime:
        return _as_utc(value)


class BackfillEdge(StrictModel):
    relation: BackfillRelation
    target_topic_key: str = Field(min_length=1, max_length=MAX_EVENT_KEY_LENGTH)


class BackfillItem(StrictModel):
    topic_key: str = Field(min_length=1, max_length=MAX_EVENT_KEY_LENGTH)
    title: str = Field(min_length=1, max_length=MAX_TITLE_LENGTH)
    content: str = Field(min_length=1, max_length=MAX_CONTENT_LENGTH)
    project: str = Field(min_length=1, max_length=MAX_EVENT_KEY_LENGTH)
    scope: Literal["project", "personal", "global"] = "project"
    type: ObservationType = "manual"
    memory_role: MemoryRole
    source: BackfillSource
    valid_from: AwareDatetime | None = None
    valid_until: AwareDatetime | None = None
    use_events: list[HistoricalUseEvent] = Field(
        default_factory=list, max_length=MAX_USE_EVENTS_PER_ITEM,
    )
    edges: list[BackfillEdge] = Field(
        default_factory=list, max_length=MAX_EDGES_PER_ITEM,
    )
    supersedes_topic_key: str | None = Field(default=None, max_length=MAX_EVENT_KEY_LENGTH)
    claim_status: ClaimStatus | None = None
    verification_state: Literal["UNVERIFIED"] = "UNVERIFIED"
    historical_value: bool = False
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("valid_from", "valid_until")
    @classmethod
    def normalize_optional_time(cls, value: datetime | None) -> datetime | None:
        return _as_utc(value) if value is not None else None

    @model_validator(mode="after")
    def validate_claim_shape(self) -> BackfillItem:
        if self.memory_role is MemoryRole.CLAIM and self.claim_status is None:
            raise ValueError("CLAIM requires claim_status")
        if self.memory_role is not MemoryRole.CLAIM and self.claim_status is not None:
            raise ValueError("claim_status is only valid for CLAIM")
        return self

    @model_validator(mode="after")
    def validate_valid_time(self) -> BackfillItem:
        start = self.valid_from or self.source.recorded_at
        if self.valid_until is not None and self.valid_until <= start:
            raise ValueError("valid_until must be after valid_from")
        return self

    @model_validator(mode="after")
    def validate_historical_closure(self) -> BackfillItem:
        historical = {ClaimStatus.SUPERSEDED, ClaimStatus.REFUTED, ClaimStatus.RETRACTED}
        if self.claim_status in historical and self.valid_until is None:
            raise ValueError("historical CLAIM requires valid_until")
        return self

    @model_validator(mode="after")
    def validate_events(self) -> BackfillItem:
        keys = [event.event_key for event in self.use_events]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate use event_key")
        if any(event.occurred_at < self.source.recorded_at for event in self.use_events):
            raise ValueError("use event cannot predate source.recorded_at")
        return self

    @model_validator(mode="after")
    def validate_metadata_size(self) -> BackfillItem:
        if len(self.metadata) > MAX_METADATA_KEYS:
            raise ValueError("metadata has too many keys")
        forbidden = RESERVED_METADATA_KEYS.intersection(self.metadata)
        if forbidden:
            raise ValueError(f"server-controlled metadata: {sorted(forbidden)}")
        return self

    @property
    def resolved_valid_from(self) -> datetime:
        return self.valid_from or self.source.recorded_at


class BackfillRequest(StrictModel):
    run_key: str = Field(min_length=1, max_length=MAX_EVENT_KEY_LENGTH)
    items: list[BackfillItem] = Field(
        min_length=1,
        max_length=MAX_BACKFILL_ITEMS,
    )
    mode: BackfillMode = BackfillMode.PLAN
    expected_plan_digest: str | None = None
    repair_temporal_projection: bool = False

    @model_validator(mode="after")
    def validate_boundary(self) -> BackfillRequest:
        projects = {item.project for item in self.items}
        namespaces = {item.source.namespace for item in self.items}
        kinds = {item.source.kind for item in self.items}
        if len(projects) != 1 or len(namespaces) != 1 or len(kinds) != 1:
            raise ValueError("one run requires one project and one source boundary")
        return self

    @model_validator(mode="after")
    def validate_event_keys(self) -> BackfillRequest:
        topics = [item.topic_key for item in self.items]
        if len(topics) != len(set(topics)):
            raise ValueError("duplicate topic_key within request")
        keys: list[tuple[str, str, str]] = []
        for item in self.items:
            keys.append((item.topic_key, "RECORDED", item.source.event_key))
            keys.extend(
                (item.topic_key, "USED", event.event_key)
                for event in item.use_events
            )
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate event_key within request")
        return self

    @model_validator(mode="after")
    def validate_apply_digest(self) -> BackfillRequest:
        if self.mode is BackfillMode.APPLY and not self.expected_plan_digest:
            raise ValueError("apply requires expected_plan_digest")
        return self

    @model_validator(mode="after")
    def validate_event_volume(self) -> BackfillRequest:
        total = sum(1 + len(item.use_events) for item in self.items)
        if total > MAX_EVENTS_PER_REQUEST:
            raise ValueError("backfill event limit exceeded")
        return self

    @property
    def project(self) -> str:
        return self.items[0].project

    @property
    def source_kind(self) -> SourceKind:
        return self.items[0].source.kind

    @property
    def source_namespace(self) -> str:
        return self.items[0].source.namespace


def backfill_validation_code(exc: ValidationError) -> str:
    message = str(exc).lower()
    if "timezone" in message:
        return "TIMESTAMP_TIMEZONE_REQUIRED"
    if "valid_until" in message:
        return "INVALID_VALID_TIME"
    if "duplicate event_key" in message or "duplicate use event_key" in message:
        return "DUPLICATE_EVENT_KEY"
    return "INVALID_BACKFILL_ITEM"


def backfill_validation_message(code: str) -> str:
    messages = {
        "TIMESTAMP_TIMEZONE_REQUIRED": "backfill timestamps require a timezone",
        "INVALID_VALID_TIME": "valid_until must be after valid_from",
        "DUPLICATE_EVENT_KEY": "backfill event keys must be unique",
    }
    return messages.get(code, "backfill request validation failed")
