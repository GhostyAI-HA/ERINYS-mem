"""ERINYS memory admission / activation sentry (#171)."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ._compat import StrEnum

SENTRY_SCHEMA = 1
BLOCKED_CONTENT = "[BLOCKED_BY_ERINYS_SENTRY]"
DERIVED_VIA = frozenset({"distill", "collide", "supersede", "session_summary"})
EXTERNAL_SOURCES = frozenset({"browser", "external", "import", "reddit", "web", "x"})
PROMPT_INJECTION_PATTERNS = (
    re.compile(r"(?i)\b(ignore (all |the )?(previous|prior)|system prompt)\b"),
    re.compile(r"(?i)\b(reveal|extract|print)\b.{0,32}\b(secret|token|password|prompt)\b"),
    re.compile(r"(前の指示を無視|システムプロンプト|秘密.{0,16}(表示|抽出))"),
)
OPERATION_DIRECTIVE_PATTERNS = (
    re.compile(r"(?i)\b(delete|execute|publish|reveal|run|send|shell|upload)\b"),
    re.compile(r"(実行して|削除して|送信して|公開して)"),
)


class SourceClass(StrEnum):
    DIRECT_USER = "direct_user"
    FIRST_PARTY_AGENT = "first_party_agent"
    FIRST_PARTY_RECORD = "first_party_record"
    POLICY = "policy"
    EXTERNAL = "external"
    DERIVED = "derived"
    UNKNOWN = "unknown"


class IntendedUse(StrEnum):
    CONTEXT = "context"
    EVIDENCE = "evidence"
    PREFERENCE = "preference"
    DIRECTIVE = "directive"


class AdmissionVerdict(StrEnum):
    ALLOW = "allow"
    RESTRICT = "restrict"
    QUARANTINE = "quarantine"


class ActivationVerdict(StrEnum):
    ALLOW = "allow"
    CONTEXT_ONLY = "context_only"
    BLOCK = "block"


class MemorySafety(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = SENTRY_SCHEMA
    source_class: SourceClass
    trusted_for: list[IntendedUse]
    admission_verdict: AdmissionVerdict
    reasons: list[str] = Field(default_factory=list)


class ActivationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = SENTRY_SCHEMA
    intended_use: IntendedUse
    verdict: ActivationVerdict
    action_authority: bool
    reasons: list[str] = Field(default_factory=list)
    blocked_content_hash: str | None = None


def _requested_source_class(value: str | None) -> SourceClass | None:
    if value == SourceClass.EXTERNAL:
        return SourceClass.EXTERNAL
    if value == SourceClass.DIRECT_USER:
        return SourceClass.DIRECT_USER
    return None


def infer_source_class(source: str, derived_via: str, requested: str | None = None) -> SourceClass:
    explicit = _requested_source_class(requested)
    if explicit is not None:
        return explicit
    if derived_via in DERIVED_VIA:
        return SourceClass.DERIVED
    if source in EXTERNAL_SOURCES:
        return SourceClass.EXTERNAL
    source_map = {
        "policy": SourceClass.POLICY,
        "user": SourceClass.DIRECT_USER,
        "agent": SourceClass.FIRST_PARTY_AGENT,
    }
    return source_map.get(source, SourceClass.UNKNOWN)


def _trusted_uses(source_class: SourceClass) -> list[IntendedUse]:
    if source_class is SourceClass.POLICY:
        return list(IntendedUse)
    if source_class is SourceClass.DIRECT_USER:
        return [IntendedUse.CONTEXT, IntendedUse.EVIDENCE, IntendedUse.PREFERENCE]
    if source_class in {
        SourceClass.FIRST_PARTY_AGENT,
        SourceClass.FIRST_PARTY_RECORD,
        SourceClass.DERIVED,
    }:
        return [IntendedUse.CONTEXT, IntendedUse.EVIDENCE]
    return [IntendedUse.CONTEXT]


def build_memory_safety(
    source: str,
    derived_via: str,
    requested_source_class: str | None = None,
) -> dict[str, object]:
    source_class = infer_source_class(source, derived_via, requested_source_class)
    restricted = source_class in {SourceClass.EXTERNAL, SourceClass.UNKNOWN}
    safety = MemorySafety(
        source_class=source_class,
        trusted_for=_trusted_uses(source_class),
        admission_verdict=AdmissionVerdict.RESTRICT if restricted else AdmissionVerdict.ALLOW,
        reasons=["untrusted_source_class"] if restricted else [],
    )
    return safety.model_dump(mode="json")


def classify_record_source(
    source_kind: str,
    namespace: str,
    allowed_namespaces: tuple[str, ...],
) -> SourceClass:
    if source_kind == "web" or namespace not in allowed_namespaces:
        return SourceClass.EXTERNAL
    return SourceClass.FIRST_PARTY_RECORD


def build_record_memory_safety(
    source_kind: str,
    namespace: str,
    allowed_namespaces: tuple[str, ...],
) -> dict[str, object]:
    source_class = classify_record_source(source_kind, namespace, allowed_namespaces)
    restricted = source_class is SourceClass.EXTERNAL
    safety = MemorySafety(
        source_class=source_class,
        trusted_for=_trusted_uses(source_class),
        admission_verdict=AdmissionVerdict.RESTRICT if restricted else AdmissionVerdict.ALLOW,
        reasons=["source_namespace_not_allowlisted"] if restricted else [],
    )
    return safety.model_dump(mode="json")


def _metadata(record: Mapping[str, object]) -> Mapping[str, object]:
    metadata = record.get("metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _fallback_safety(record: Mapping[str, object]) -> MemorySafety:
    provenance = _metadata(record).get("provenance")
    via = provenance.get("derived_via", "save") if isinstance(provenance, Mapping) else "save"
    payload = build_memory_safety(str(record.get("source", "unknown")), str(via))
    return MemorySafety.model_validate(payload)


def memory_safety(record: Mapping[str, object]) -> MemorySafety:
    payload = _metadata(record).get("memory_safety")
    if isinstance(payload, Mapping):
        try:
            return MemorySafety.model_validate(payload)
        except ValidationError:
            pass
    return _fallback_safety(record)


def derived_memory_safety(record: Mapping[str, object]) -> dict[str, object]:
    parent = memory_safety(record)
    trusted = [use for use in parent.trusted_for if use is not IntendedUse.DIRECTIVE]
    safety = MemorySafety(
        source_class=SourceClass.DERIVED,
        trusted_for=trusted,
        admission_verdict=parent.admission_verdict,
        reasons=[*parent.reasons, "inherited_from_parent"],
    )
    return safety.model_dump(mode="json")


def has_risky_directive(content: str) -> bool:
    patterns = (*PROMPT_INJECTION_PATTERNS, *OPERATION_DIRECTIVE_PATTERNS)
    return any(pattern.search(content) for pattern in patterns)


def has_prompt_injection(content: str) -> bool:
    return any(pattern.search(content) for pattern in PROMPT_INJECTION_PATTERNS)


def has_operation_directive(content: str) -> bool:
    return any(pattern.search(content) for pattern in OPERATION_DIRECTIVE_PATTERNS)


def _blocked_activation(use: IntendedUse, reason: str, content: str) -> ActivationResult:
    return ActivationResult(
        intended_use=use,
        verdict=ActivationVerdict.BLOCK,
        action_authority=False,
        reasons=[reason],
        blocked_content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )


def _block_reason(
    safety: MemorySafety,
    content: str,
    use: IntendedUse,
) -> str | None:
    if safety.admission_verdict is AdmissionVerdict.QUARANTINE:
        return "quarantined_at_admission"
    if has_prompt_injection(content):
        return "prompt_injection_pattern"
    if use is IntendedUse.DIRECTIVE and use not in safety.trusted_for:
        return "directive_not_trusted"
    return None


def assess_activation(safety: MemorySafety, content: str, intended_use: str) -> ActivationResult:
    use = IntendedUse(intended_use)
    reason = _block_reason(safety, content, use)
    if reason is not None:
        return _blocked_activation(use, reason, content)
    quoted = has_operation_directive(content)
    trusted = use in safety.trusted_for and not (
        quoted and use in {IntendedUse.CONTEXT, IntendedUse.PREFERENCE}
    )
    verdict = ActivationVerdict.ALLOW if trusted else ActivationVerdict.CONTEXT_ONLY
    return _activation_result(use, verdict, quoted)


def _activation_result(
    use: IntendedUse,
    verdict: ActivationVerdict,
    quoted: bool,
) -> ActivationResult:
    return ActivationResult(
        intended_use=use,
        verdict=verdict,
        action_authority=use is IntendedUse.DIRECTIVE and verdict is ActivationVerdict.ALLOW,
        reasons=_activation_reasons(verdict, quoted),
    )


def _activation_reasons(
    verdict: ActivationVerdict,
    quoted_directive: bool,
) -> list[str]:
    if verdict is ActivationVerdict.ALLOW:
        return []
    if quoted_directive:
        return ["quoted_directive_no_action_authority"]
    return ["use_not_trusted"]


def screen_observation(
    record: Mapping[str, object],
    intended_use: str = IntendedUse.CONTEXT,
) -> dict[str, object]:
    screened = dict(record)
    activation = assess_activation(memory_safety(record), str(record.get("content", "")), intended_use)
    screened["activation"] = activation.model_dump(mode="json")
    if activation.verdict is ActivationVerdict.BLOCK:
        screened["content"] = BLOCKED_CONTENT
    return screened


def screen_observations(
    records: list[dict[str, object]],
    intended_use: str = IntendedUse.CONTEXT,
) -> list[dict[str, object]]:
    return [screen_observation(record, intended_use) for record in records]


def blocked_count(records: list[Mapping[str, object]]) -> int:
    return sum(
        1
        for record in records
        if isinstance(record.get("activation"), Mapping)
        and record["activation"].get("verdict") == ActivationVerdict.BLOCK
    )
