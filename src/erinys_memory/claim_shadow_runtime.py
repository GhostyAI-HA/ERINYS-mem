"""Live RRF candidatesをCurrent Valid Claim shadowへ接続するadapter。"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from time import perf_counter
from typing import Literal
from uuid import uuid4

from pydantic import Field, TypeAdapter, ValidationError

from ._sqlite import sqlite3
from .claim_contracts import (
    ClaimNode,
    ClaimScope,
    ClaimSearchRequest,
    ClaimStatus,
    ContractModel,
    CurrentValidClaimResult,
    EdgeOrigin,
    EdgeStatus,
    EdgeType,
    MemoryEdge,
    ProvenanceBundle,
    QueryMode,
    RiskClass,
    SourceProvenance,
    TransformationLineage,
    VerificationState,
)
from .claim_shadow import ShadowInput, build_shadow_result

logger = logging.getLogger(__name__)

SHADOW_MODE_ENV = "ERINYS_CLAIM_SHADOW_MODE"
SHADOW_MODE_AUDIT = "audit"
SHADOW_SCHEMA_VERSION = 1
ERINYS_CORPUS = "ERINYS"
MAX_GRAPH_HOPS = 2
MAX_GRAPH_NODES = 128
MAX_GRAPH_EDGES = 256
DEFAULT_CLAIM_CONFIDENCE = 0.5
SCOPE_FIELDS = (
    "project", "tenant", "environment", "jurisdiction",
    "product_version", "source_version",
)
HISTORICAL_STATUSES = frozenset({
    ClaimStatus.SUPERSEDED, ClaimStatus.REFUTED, ClaimStatus.RETRACTED,
})
LEGACY_RELATIONS = {
    "contradicts": EdgeType.CONTRADICTS,
    "supersedes": EdgeType.SUPERSEDES,
    "distilled_from": EdgeType.DERIVED_FROM,
    "references": EdgeType.DERIVED_FROM,
}
JSON_OBJECT = TypeAdapter(dict[str, object])


class AdaptedShadowSearch(ContractModel):
    shadow_input: ShadowInput
    baseline_scores: dict[str, float] = Field(default_factory=dict)
    graph_candidate_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class LiveShadowReceipt(ContractModel):
    schema_version: Literal[1] = SHADOW_SCHEMA_VERSION
    receipt_id: str = Field(min_length=1)
    status: Literal["evaluated", "no_candidates", "error"]
    recorded_at: datetime
    query_hash: str = Field(min_length=64, max_length=64)
    query_character_count: int = Field(ge=0)
    project: str | None = None
    intended_use: str = Field(min_length=1)
    candidate_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    baseline_claim_ids: list[str] = Field(default_factory=list)
    baseline_scores: dict[str, float] = Field(default_factory=dict)
    graph_candidate_ids: list[str] = Field(default_factory=list)
    shadow_active_claim_ids: list[str] = Field(default_factory=list)
    conflicting_claim_ids: list[str] = Field(default_factory=list)
    historical_claim_ids: list[str] = Field(default_factory=list)
    removed_baseline_ids: list[str] = Field(default_factory=list)
    added_by_policy_ids: list[str] = Field(default_factory=list)
    selected_path_ids: list[str] = Field(default_factory=list)
    discarded_path_ids: list[str] = Field(default_factory=list)
    fallback_reason: str | None = None
    error_type: str | None = None
    uncertainty: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    latency_ms: float = Field(ge=0.0)


class ShadowSoakReport(ContractModel):
    total: int = Field(ge=0)
    valid_receipts: int = Field(ge=0)
    invalid_receipts: int = Field(ge=0)
    evaluated: int = Field(ge=0)
    no_candidates: int = Field(ge=0)
    error_count: int = Field(ge=0)
    fallback_count: int = Field(ge=0)
    disagreement_count: int = Field(ge=0)
    graph_path_count: int = Field(ge=0)
    invalid_rate: float = Field(ge=0.0, le=1.0)
    fallback_rate: float = Field(ge=0.0, le=1.0)
    disagreement_rate: float = Field(ge=0.0, le=1.0)
    graph_path_rate: float = Field(ge=0.0, le=1.0)
    error_rate: float = Field(ge=0.0, le=1.0)
    p50_latency_ms: float = Field(ge=0.0)
    p95_latency_ms: float = Field(ge=0.0)
    warning_counts: dict[str, int] = Field(default_factory=dict)


def claim_shadow_enabled() -> bool:
    mode = os.environ.get(SHADOW_MODE_ENV, "off")
    return mode.strip().lower() == SHADOW_MODE_AUDIT


def _metadata(value: object) -> dict[str, object]:
    try:
        if isinstance(value, str):
            return JSON_OBJECT.validate_json(value)
        return JSON_OBJECT.validate_python(value or {})
    except ValidationError:
        return {}


def _aware_datetime(value: object, fallback: datetime) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        return fallback
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _observation_id(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError("boolean observation id is invalid")
    return int(str(value))


def _claim_id(value: object) -> str:
    return f"observation:{_observation_id(value)}"


def _safe_float(value: object, fallback: float) -> float:
    if isinstance(value, bool):
        return fallback
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return fallback
    return min(max(parsed, 0.0), 1.0)


def _scope(row: Mapping[str, object], metadata: dict[str, object]) -> ClaimScope:
    values: dict[str, object] = {"corpus": ERINYS_CORPUS}
    project = row.get("project")
    if isinstance(project, str) and project:
        values["project"] = project
    explicit = _metadata(metadata.get("claim_scope"))
    values.update(_scope_overrides(explicit))
    return ClaimScope.model_validate(values)


def _scope_overrides(explicit: dict[str, object]) -> dict[str, str]:
    return {
        key: value for key in SCOPE_FIELDS
        if isinstance((value := explicit.get(key)), str) and value
    }


def _claim_status(row: Mapping[str, object], metadata: dict[str, object]) -> ClaimStatus:
    raw = metadata.get("claim_status")
    try:
        status = ClaimStatus(str(raw))
    except ValueError:
        status = _legacy_status(row)
    if status in HISTORICAL_STATUSES and row.get("valid_until") is None:
        return ClaimStatus.CONTESTED
    return status


def _legacy_status(row: Mapping[str, object]) -> ClaimStatus:
    historical = row.get("valid_until") is not None or row.get("superseded_by") is not None
    return ClaimStatus.SUPERSEDED if historical else ClaimStatus.ACTIVE


def _verification(metadata: dict[str, object]) -> VerificationState:
    try:
        return VerificationState(str(metadata.get("verification_state")))
    except ValueError:
        return VerificationState.UNVERIFIED


def _risk(metadata: dict[str, object]) -> RiskClass:
    try:
        return RiskClass(str(metadata.get("risk_class")))
    except ValueError:
        return RiskClass.LOW


def _transformation(row: Mapping[str, object], metadata: dict[str, object]) -> list[TransformationLineage]:
    provenance = _metadata(metadata.get("provenance"))
    operation = provenance.get("derived_via")
    if not isinstance(operation, str) or not operation:
        return []
    parents = provenance.get("parents")
    parent_ids = [_claim_id(item) for item in parents] if isinstance(parents, list) else []
    return [_transformation_lineage(row, operation, parent_ids)]


def _transformation_lineage(
    row: Mapping[str, object], operation: str, parent_ids: list[str],
) -> TransformationLineage:
    return TransformationLineage(
        transformation_id=f"observation:{_observation_id(row['id'])}:{operation}",
        operation=operation,
        parent_node_ids=parent_ids,
        implementation_version="legacy-observation-adapter-v1",
    )


def _provenance(
    row: Mapping[str, object], metadata: dict[str, object], recorded_at: datetime,
) -> ProvenanceBundle:
    source_type = str(row.get("source") or row.get("type") or "observation")
    source = SourceProvenance(
        source_id=f"observation:{_observation_id(row['id'])}:source",
        source_type=source_type,
        locator=f"erinys://observation/{_observation_id(row['id'])}",
        acquired_at=recorded_at,
    )
    return ProvenanceBundle(sources=[source], transformations=_transformation(row, metadata))


def _claim_policy(
    row: Mapping[str, object], metadata: dict[str, object],
) -> dict[str, object]:
    return {
        "status": _claim_status(row, metadata),
        "confidence": _safe_float(
            metadata.get("claim_confidence"), DEFAULT_CLAIM_CONFIDENCE,
        ),
        "verification_state": _verification(metadata),
        "risk_class": _risk(metadata),
        "legal_retention": bool(metadata.get("legal_retention")),
        "historical_value": bool(metadata.get("historical_value")),
    }


def _claim_payload(row: Mapping[str, object], now: datetime) -> dict[str, object]:
    metadata = _metadata(row.get("metadata"))
    recorded_at = _aware_datetime(row.get("created_at"), now)
    valid_from = _aware_datetime(row.get("valid_from"), recorded_at)
    proposition = str(row.get("content") or row.get("title") or "").strip()
    return {
        "claim_id": _claim_id(row["id"]), "proposition": proposition,
        "scope": _scope(row, metadata),
        "valid_time": {"from": valid_from, "to": row.get("valid_until")},
        "recorded_at": recorded_at, **_claim_policy(row, metadata),
        "provenance": _provenance(row, metadata, recorded_at),
    }


def _is_claim_row(row: Mapping[str, object]) -> bool:
    role = _metadata(row.get("metadata")).get("memory_role")
    return role in {None, "CLAIM"}


def _adapt_claims(
    rows: Sequence[Mapping[str, object]], now: datetime,
) -> tuple[list[ClaimNode], list[str]]:
    claims: list[ClaimNode] = []
    warnings: list[str] = []
    for row in rows:
        if not _is_claim_row(row):
            warnings.append(f"non_claim_node_excluded:{row.get('id', 'unknown')}")
            continue
        try:
            claims.append(ClaimNode.model_validate(_claim_payload(row, now)))
        except (KeyError, TypeError, ValueError, ValidationError):
            warnings.append(f"invalid_observation:{row.get('id', 'unknown')}")
    return claims, warnings


def _edge_rows(
    db: sqlite3.Connection, frontier: set[int],
) -> list[dict[str, object]]:
    if not frontier:
        return []
    marks = ",".join("?" for _ in frontier)
    sql = (
        "SELECT * FROM edges WHERE source_id IN (" + marks + ") "
        "OR target_id IN (" + marks + ") ORDER BY id LIMIT ?"
    )
    params = [*sorted(frontier), *sorted(frontier), MAX_GRAPH_EDGES]
    return [dict(row) for row in db.execute(sql, params).fetchall()]


def _bounded_edge_rows(
    db: sqlite3.Connection, start_ids: set[int],
) -> list[dict[str, object]]:
    rows: dict[int, dict[str, object]] = {}
    seen, frontier = set(start_ids), set(start_ids)
    for _ in range(MAX_GRAPH_HOPS):
        batch = _edge_rows(db, frontier)
        accepted, frontier = _admit_edge_rows(batch, seen)
        rows.update({_observation_id(row["id"]): row for row in accepted})
        if not frontier or len(rows) >= MAX_GRAPH_EDGES:
            break
    return [rows[key] for key in sorted(rows)[:MAX_GRAPH_EDGES]]


def _admit_edge_rows(
    rows: Sequence[Mapping[str, object]], seen: set[int],
) -> tuple[list[dict[str, object]], set[int]]:
    accepted: list[dict[str, object]] = []
    admitted: set[int] = set()
    for row in rows:
        endpoints = {_observation_id(row[key]) for key in ("source_id", "target_id")}
        new_nodes = endpoints - seen
        if len(seen) + len(new_nodes) > MAX_GRAPH_NODES:
            continue
        accepted.append(dict(row))
        seen.update(new_nodes)
        admitted.update(new_nodes)
    return accepted, admitted


def _observation_rows(
    db: sqlite3.Connection, ids: set[int],
) -> list[dict[str, object]]:
    if not ids:
        return []
    marks = ",".join("?" for _ in ids)
    sql = f"SELECT * FROM observations WHERE id IN ({marks}) ORDER BY id"
    return [dict(row) for row in db.execute(sql, sorted(ids)).fetchall()]


def _contract_edge(metadata: dict[str, object]) -> dict[str, object]:
    return _metadata(metadata.get("claim_edge"))


def _edge_relation(contract: dict[str, object], legacy: str) -> EdgeType:
    raw = contract.get("relation")
    try:
        return EdgeType(str(raw))
    except ValueError:
        return LEGACY_RELATIONS.get(legacy, EdgeType.APPLICABLE_TO)


def _edge_trust(
    contract: dict[str, object], legacy: str,
) -> tuple[EdgeOrigin, EdgeStatus]:
    if contract:
        try:
            return EdgeOrigin(str(contract.get("origin"))), EdgeStatus(str(contract.get("status")))
        except ValueError:
            return EdgeOrigin.CANDIDATE, EdgeStatus.CANDIDATE
    if legacy == "supersedes":
        return EdgeOrigin.DETERMINISTIC, EdgeStatus.VERIFIED
    if legacy == "distilled_from":
        return EdgeOrigin.VERIFIED_DERIVED, EdgeStatus.VERIFIED
    return EdgeOrigin.CANDIDATE, EdgeStatus.CANDIDATE


def _edge_endpoints(
    row: Mapping[str, object], legacy: str, explicit: bool,
) -> tuple[str, str]:
    source = _claim_id(row["source_id"])
    target = _claim_id(row["target_id"])
    reversed_relations = {"derived_from", "verified_by"}
    reverse = legacy == "distilled_from" and not explicit
    reverse = reverse or (explicit and legacy in reversed_relations)
    return (target, source) if reverse else (source, target)


def _edge_payload(row: Mapping[str, object]) -> dict[str, object]:
    metadata = _metadata(row.get("metadata"))
    contract = _contract_edge(metadata)
    legacy = str(row.get("relation") or "")
    origin, status = _edge_trust(contract, legacy)
    source, target = _edge_endpoints(row, legacy, bool(contract))
    return {
        "edge_id": f"edge:{_observation_id(row['id'])}",
        "source_id": source, "target_id": target,
        "relation": _edge_relation(contract, legacy),
        "origin": origin, "status": status,
        "confidence": _safe_float(row.get("weight"), DEFAULT_CLAIM_CONFIDENCE),
        "verification_id": contract.get("verification_id"),
    }


def _adapt_edges(
    rows: Sequence[Mapping[str, object]],
) -> tuple[list[MemoryEdge], list[str]]:
    edges: list[MemoryEdge] = []
    warnings: list[str] = []
    for row in rows:
        try:
            edges.append(MemoryEdge.model_validate(_edge_payload(row)))
        except (KeyError, TypeError, ValueError, ValidationError):
            warnings.append(f"invalid_edge:{row.get('id', 'unknown')}")
    candidate_count = sum(edge.status is EdgeStatus.CANDIDATE for edge in edges)
    if candidate_count:
        warnings.append(f"unverified_edges_excluded:{candidate_count}")
    return edges, warnings


def _baseline_ids(results: Sequence[Mapping[str, object]]) -> list[str]:
    return [
        _claim_id(row["id"]) for row in results
        if row.get("id") is not None and _is_claim_row(row)
    ]


def _baseline_scores(results: Sequence[Mapping[str, object]]) -> dict[str, float]:
    return {
        _claim_id(row["id"]): max(float(str(row.get("effective_score", 0.0))), 0.0)
        for row in results if row.get("id") is not None and _is_claim_row(row)
    }


def _request(query: str, project: str | None, now: datetime) -> ClaimSearchRequest:
    scope = ClaimScope(corpus=ERINYS_CORPUS, project=project)
    return ClaimSearchRequest(
        query=query, as_of=now, knowledge_as_of=now, scope=scope,
        mode=QueryMode.CURRENT_VALID_CLAIM, max_hops=MAX_GRAPH_HOPS,
    )


def _graph_context(
    db: sqlite3.Connection, baseline_ids: set[int],
) -> tuple[list[dict[str, object]], set[int], list[dict[str, object]]]:
    edge_rows = _bounded_edge_rows(db, baseline_ids)
    graph_ids = {
        _observation_id(row[key])
        for row in edge_rows for key in ("source_id", "target_id")
    }
    rows = _observation_rows(db, baseline_ids | graph_ids)
    return edge_rows, graph_ids, rows


def _adapt_graph_rows(
    rows: Sequence[Mapping[str, object]],
    edge_rows: Sequence[Mapping[str, object]],
    now: datetime,
) -> tuple[list[ClaimNode], list[MemoryEdge], list[str]]:
    claims, claim_warnings = _adapt_claims(rows, now)
    edges, edge_warnings = _adapt_edges(edge_rows)
    return claims, edges, [*claim_warnings, *edge_warnings]


def _adapted_search(
    shadow_input: ShadowInput, results: Sequence[Mapping[str, object]],
    graph_ids: set[int], baseline_ids: set[int], warnings: list[str],
) -> AdaptedShadowSearch:
    graph_claim_ids = {_claim_id(item) for item in graph_ids - baseline_ids}
    return AdaptedShadowSearch(
        shadow_input=shadow_input, baseline_scores=_baseline_scores(results),
        graph_candidate_ids=sorted(graph_claim_ids), warnings=warnings,
    )


def _adapt_search(
    db: sqlite3.Connection, query: str, project: str | None,
    results: Sequence[Mapping[str, object]], now: datetime,
) -> AdaptedShadowSearch:
    start_ids = {_observation_id(row["id"]) for row in results}
    edge_rows, graph_ids, rows = _graph_context(db, start_ids)
    claims, edges, warnings = _adapt_graph_rows(rows, edge_rows, now)
    claim_ids = _adapted_claim_ids(claims)
    baseline_ids = _baseline_observation_ids(results)
    shadow_input = ShadowInput(
        request=_request(query, project, now), claims=claims, edges=edges,
        baseline_claim_ids=_baseline_ids(results),
    )
    return _adapted_search(
        shadow_input, results, graph_ids & claim_ids, baseline_ids, warnings)


def _adapted_claim_ids(claims: Sequence[ClaimNode]) -> set[int]:
    return {
        _observation_id(claim.claim_id.removeprefix("observation:"))
        for claim in claims
    }


def _baseline_observation_ids(
    results: Sequence[Mapping[str, object]],
) -> set[int]:
    return {
        _observation_id(row["id"]) for row in results if _is_claim_row(row)
    }


def _conflicting_ids(result: CurrentValidClaimResult) -> list[str]:
    claims = [*result.active_claims, *result.historical_claims]
    node_ids = {
        node_id for path in result.conflicting_paths for node_id in path.node_ids
    }
    return sorted({claim.claim_id for claim in claims} & node_ids)


def _selection_payload(
    adapted: AdaptedShadowSearch, result: CurrentValidClaimResult,
) -> dict[str, object]:
    baseline = adapted.shadow_input.baseline_claim_ids
    active = [claim.claim_id for claim in result.active_claims]
    history = [claim.claim_id for claim in result.historical_claims]
    return {
        "baseline_claim_ids": baseline, "baseline_scores": adapted.baseline_scores,
        "graph_candidate_ids": adapted.graph_candidate_ids,
        "shadow_active_claim_ids": active, "conflicting_claim_ids": _conflicting_ids(result),
        "historical_claim_ids": history,
        "removed_baseline_ids": sorted(set(baseline) - set(active)),
        "added_by_policy_ids": sorted(set(active) - set(baseline)),
    }


def _trace_payload(result: CurrentValidClaimResult) -> dict[str, object]:
    trace = result.retrieval_trace
    return {
        "selected_path_ids": trace.selected_path_ids,
        "discarded_path_ids": trace.discarded_path_ids,
        "fallback_reason": trace.fallback_reason,
        "uncertainty": result.uncertainty,
    }


def _receipt_payload(
    adapted: AdaptedShadowSearch, result: CurrentValidClaimResult, query: str,
    project: str | None, intended_use: str, now: datetime, latency_ms: float,
) -> dict[str, object]:
    return {
        **_receipt_identity(query, project, intended_use, now),
        "status": "evaluated", "candidate_count": len(adapted.shadow_input.claims),
        "edge_count": len(adapted.shadow_input.edges),
        **_selection_payload(adapted, result), **_trace_payload(result),
        "warnings": adapted.warnings, "latency_ms": latency_ms,
    }


def _receipt_identity(
    query: str, project: str | None, intended_use: str, now: datetime,
) -> dict[str, object]:
    return {
        "receipt_id": uuid4().hex, "recorded_at": now,
        "query_hash": hashlib.sha256(query.encode("utf-8")).hexdigest(),
        "query_character_count": len(query), "project": project,
        "intended_use": intended_use,
    }


def _empty_receipt(
    query: str, project: str | None, intended_use: str,
    now: datetime, latency_ms: float,
) -> LiveShadowReceipt:
    payload = {
        **_receipt_identity(query, project, intended_use, now),
        "status": "no_candidates", "candidate_count": 0, "edge_count": 0,
        "fallback_reason": "no_baseline_candidates", "latency_ms": latency_ms,
    }
    return LiveShadowReceipt.model_validate(payload)


def _error_receipt(
    query: str, project: str | None, intended_use: str, now: datetime,
    candidate_count: int, error_type: str, latency_ms: float,
) -> LiveShadowReceipt:
    payload = {
        **_receipt_identity(query, project, intended_use, now),
        "status": "error", "candidate_count": candidate_count, "edge_count": 0,
        "fallback_reason": "shadow_evaluation_error", "error_type": error_type,
        "warnings": [f"shadow_error:{error_type}"], "latency_ms": latency_ms,
    }
    return LiveShadowReceipt.model_validate(payload)


def _evaluate_shadow(
    db: sqlite3.Connection, query: str, project: str | None,
    results: Sequence[Mapping[str, object]], intended_use: str,
    now: datetime, started: float,
) -> LiveShadowReceipt:
    if not results:
        return _empty_receipt(query, project, intended_use, now, _elapsed_ms(started))
    adapted = _adapt_search(db, query, project, results, now)
    result = build_shadow_result(adapted.shadow_input)
    payload = _receipt_payload(
        adapted, result, query, project, intended_use, now, _elapsed_ms(started),
    )
    return LiveShadowReceipt.model_validate(payload)


def _failed_open_receipt(
    exc: Exception, query: str, project: str | None, intended_use: str,
    now: datetime, candidate_count: int, started: float,
) -> LiveShadowReceipt:
    error_type = type(exc).__name__
    logger.warning("Current Valid Claim shadow failed open: %s", error_type)
    return _error_receipt(
        query, project, intended_use, now, candidate_count,
        error_type, _elapsed_ms(started),
    )


def observe_shadow_search(
    db: sqlite3.Connection, query: str, project: str | None,
    results: Sequence[Mapping[str, object]], intended_use: str = "context",
) -> LiveShadowReceipt | None:
    if not claim_shadow_enabled():
        return None
    started, now = perf_counter(), datetime.now(timezone.utc)
    try:
        return _evaluate_shadow(
            db, query, project, results, intended_use, now, started,
        )
    except Exception as exc:  # noqa: BLE001 - shadow must fail open
        return _failed_open_receipt(
            exc, query, project, intended_use, now, len(results), started,
        )


def _elapsed_ms(started: float) -> float:
    return max((perf_counter() - started) * 1000.0, 0.0)


def _percentile(values: Sequence[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * ratio), len(ordered) - 1)
    return ordered[index]


def _warning_counts(receipts: Sequence[LiveShadowReceipt]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for receipt in receipts:
        for warning in receipt.warnings:
            counts[warning] = counts.get(warning, 0) + 1
    return counts


def _soak_counts(receipts: Sequence[LiveShadowReceipt]) -> tuple[int, int, int, int, int]:
    evaluated = sum(receipt.status == "evaluated" for receipt in receipts)
    errors = sum(receipt.status == "error" for receipt in receipts)
    fallback = sum(receipt.fallback_reason is not None for receipt in receipts)
    disagreement = sum(
        bool(receipt.removed_baseline_ids or receipt.added_by_policy_ids)
        for receipt in receipts
    )
    graph_paths = sum(bool(receipt.selected_path_ids) for receipt in receipts)
    return evaluated, errors, fallback, disagreement, graph_paths


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _soak_rates(
    invalid: int, total: int, fallback: int, valid: int,
    disagreement: int, graph_paths: int, evaluated: int, errors: int,
) -> dict[str, float]:
    return {
        "invalid_rate": _ratio(invalid, total),
        "fallback_rate": _ratio(fallback, valid),
        "disagreement_rate": _ratio(disagreement, evaluated),
        "graph_path_rate": _ratio(graph_paths, evaluated),
        "error_rate": _ratio(errors, valid),
    }


def _soak_cardinalities(
    total: int, valid: int, invalid: int, evaluated: int,
    no_candidates: int, errors: int, fallback: int,
    disagreement: int, graph_paths: int,
) -> dict[str, int]:
    return {
        "total": total, "valid_receipts": valid, "invalid_receipts": invalid,
        "evaluated": evaluated, "no_candidates": no_candidates,
        "error_count": errors, "fallback_count": fallback,
        "disagreement_count": disagreement, "graph_path_count": graph_paths,
    }


def _soak_report_counts(
    receipts: Sequence[LiveShadowReceipt], invalid_receipts: int = 0,
) -> dict[str, object]:
    evaluated, errors, fallback, disagreement, graph_paths = _soak_counts(receipts)
    valid, total = len(receipts), len(receipts) + invalid_receipts
    no_candidates = sum(receipt.status == "no_candidates" for receipt in receipts)
    rates = _soak_rates(
        invalid_receipts, total, fallback, valid,
        disagreement, graph_paths, evaluated, errors,
    )
    counts = _soak_cardinalities(
        total, valid, invalid_receipts, evaluated, no_candidates,
        errors, fallback, disagreement, graph_paths,
    )
    return {**counts, **rates}


def build_soak_report(
    receipts: Sequence[LiveShadowReceipt], invalid_receipts: int = 0,
) -> ShadowSoakReport:
    latencies = [receipt.latency_ms for receipt in receipts]
    return ShadowSoakReport(
        **_soak_report_counts(receipts, invalid_receipts),
        p50_latency_ms=_percentile(latencies, 0.50),
        p95_latency_ms=_percentile(latencies, 0.95),
        warning_counts=_warning_counts(receipts),
    )
