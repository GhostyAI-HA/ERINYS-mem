"""Pure shadow evaluator for ERINYS Current Valid Claim contract v0."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from pydantic import Field

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
    EvidencePath,
    MemoryEdge,
    OutcomeAttributionReceipt,
    RetrievalTrace,
)

NO_VERIFIED_PATH = "no_verified_evidence_path"
ACTIVE_STATUSES = frozenset({ClaimStatus.ACTIVE, ClaimStatus.CONTESTED})
HISTORY_STATUSES = frozenset(
    {ClaimStatus.SUPERSEDED, ClaimStatus.REFUTED, ClaimStatus.RETRACTED},
)
TRUSTED_EDGE_ORIGINS = frozenset(
    {EdgeOrigin.DETERMINISTIC, EdgeOrigin.VERIFIED_DERIVED},
)
MAX_SELECTED_PATHS = 64


class ShadowInput(ContractModel):
    """Expected-outputを含まない、本番shadow評価用の入力。"""

    request: ClaimSearchRequest
    claims: list[ClaimNode] = Field(min_length=1)
    edges: list[MemoryEdge] = Field(default_factory=list)
    baseline_claim_ids: list[str] = Field(default_factory=list)
    attribution_receipt: OutcomeAttributionReceipt | None = None


class ShadowExpected(ContractModel):
    active_claim_ids: list[str] = Field(default_factory=list)
    conflicting_claim_ids: list[str] = Field(default_factory=list)
    historical_claim_ids: list[str] = Field(default_factory=list)
    fallback_reason: str | None = None
    reinforcement_eligible: bool | None = None


class ShadowFixture(ShadowInput):
    fixture_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    expected: ShadowExpected


class ShadowComparison(ContractModel):
    fixture_id: str
    category: str
    baseline_claim_ids: list[str]
    shadow_active_claim_ids: list[str]
    removed_baseline_ids: list[str]
    added_by_policy_ids: list[str]
    conflicting_claim_ids: list[str]
    historical_claim_ids: list[str]
    reinforcement_eligible: bool | None
    mismatches: list[str] = Field(default_factory=list)
    result: CurrentValidClaimResult


class ShadowSuiteReport(ContractModel):
    total: int
    passed: int
    failed: int
    category_counts: dict[str, int]
    failures: dict[str, list[str]]


def load_fixtures(path: Path) -> list[ShadowFixture]:
    lines = path.read_text(encoding="utf-8").splitlines()
    payloads = [json.loads(line) for line in lines if line.strip()]
    return [ShadowFixture.model_validate(payload) for payload in payloads]


def _scope_items(scope: ClaimScope) -> Iterable[tuple[str, object]]:
    return ((key, value) for key, value in scope.model_dump().items() if value is not None)


def scope_matches(requested: ClaimScope, candidate: ClaimScope) -> bool:
    candidate_data = candidate.model_dump()
    return all(candidate_data[key] == value for key, value in _scope_items(requested))


def known_at(claim: ClaimNode, moment: datetime) -> bool:
    return claim.recorded_at <= moment


def valid_at(claim: ClaimNode, moment: datetime) -> bool:
    interval = claim.valid_time
    return interval.from_ <= moment and (interval.to is None or moment < interval.to)


def _eligible_claims(fixture: ShadowInput) -> list[ClaimNode]:
    request = fixture.request
    return [
        claim for claim in fixture.claims
        if scope_matches(request.scope, claim.scope)
        and known_at(claim, request.knowledge_as_of)
    ]


def _active_claims(claims: list[ClaimNode], as_of: datetime) -> list[ClaimNode]:
    eligible = ACTIVE_STATUSES | HISTORY_STATUSES
    active = [claim for claim in claims if claim.status in eligible and valid_at(claim, as_of)]
    return sorted(active, key=lambda claim: (claim.confidence, claim.recorded_at), reverse=True)


def _history_claims(
    claims: list[ClaimNode],
    active_ids: set[str],
    as_of: datetime,
) -> list[ClaimNode]:
    history = [
        claim for claim in claims
        if claim.status in HISTORY_STATUSES
        and claim.claim_id not in active_ids
        and claim.valid_time.from_ <= as_of
    ]
    return sorted(history, key=lambda claim: claim.valid_time.from_, reverse=True)


def _edge_valid_at(edge: MemoryEdge, moment: datetime) -> bool:
    interval = edge.valid_time
    if interval is None:
        return True
    return interval.from_ <= moment and (interval.to is None or moment < interval.to)


def _usable_edge(edge: MemoryEdge, moment: datetime) -> bool:
    trusted = edge.origin in TRUSTED_EDGE_ORIGINS
    return trusted and edge.status is EdgeStatus.VERIFIED and _edge_valid_at(edge, moment)


def _claim_endpoints_allowed(
    edge: MemoryEdge,
    all_claim_ids: set[str],
    path_claim_ids: set[str],
) -> bool:
    claim_endpoints = {edge.source_id, edge.target_id} & all_claim_ids
    return claim_endpoints <= path_claim_ids


def _usable_edges(
    fixture: ShadowInput,
    path_claim_ids: set[str],
) -> list[MemoryEdge]:
    all_claim_ids = {claim.claim_id for claim in fixture.claims}
    return [
        edge for edge in fixture.edges
        if _usable_edge(edge, fixture.request.as_of)
        and _claim_endpoints_allowed(edge, all_claim_ids, path_claim_ids)
    ]


def _incoming_edges(edges: list[MemoryEdge]) -> dict[str, list[MemoryEdge]]:
    incoming: dict[str, list[MemoryEdge]] = {}
    for edge in edges:
        incoming.setdefault(edge.target_id, []).append(edge)
    return incoming


def _walk_paths(
    target_id: str, remaining: int, incoming: dict[str, list[MemoryEdge]],
    node_ids: list[str], edges: list[MemoryEdge],
) -> list[tuple[list[str], list[MemoryEdge]]]:
    options = [edge for edge in incoming.get(target_id, []) if edge.source_id not in node_ids]
    if remaining == 0 or not options:
        return [(node_ids, edges)] if edges else []
    paths: list[tuple[list[str], list[MemoryEdge]]] = []
    for edge in options:
        paths.extend(_walk_paths(
            edge.source_id, remaining - 1, incoming,
            [edge.source_id, *node_ids], [edge, *edges],
        ))
    return paths


def _path(
    node_ids: list[str],
    edges: list[MemoryEdge],
) -> EvidencePath:
    relations = [edge.relation for edge in edges]
    conflict = EdgeType.CONTRADICTS in relations
    reason = "conflicts_with_active_claim" if conflict else "supports_active_claim"
    return EvidencePath(
        path_id="path:" + ">".join(edge.edge_id for edge in edges),
        node_ids=node_ids,
        edge_ids=[edge.edge_id for edge in edges],
        relations=relations,
        score=min(edge.confidence for edge in edges),
        selected_reason=reason,
    )


def _evidence_paths(
    edges: list[MemoryEdge],
    active_ids: set[str],
    max_hops: int,
) -> list[EvidencePath]:
    incoming = _incoming_edges(edges)
    raw_paths: list[tuple[list[str], list[MemoryEdge]]] = []
    for claim_id in sorted(active_ids):
        raw_paths.extend(_walk_paths(claim_id, max_hops, incoming, [claim_id], []))
    built = [_path(nodes, path_edges) for nodes, path_edges in raw_paths]
    paths = {path.path_id: path for path in built}
    ranked = sorted(paths.values(), key=lambda path: (-path.score, path.path_id))
    return ranked[:MAX_SELECTED_PATHS]


def _partition_paths(
    paths: list[EvidencePath],
) -> tuple[list[EvidencePath], list[EvidencePath]]:
    supporting = [path for path in paths if EdgeType.CONTRADICTS not in path.relations]
    conflicting = [path for path in paths if EdgeType.CONTRADICTS in path.relations]
    return supporting, conflicting


def _conflicting_ids(paths: list[EvidencePath], claim_ids: set[str]) -> list[str]:
    node_ids = {node_id for path in paths for node_id in path.node_ids}
    return sorted(node_ids & claim_ids)


def _returned_ids(
    active: list[ClaimNode],
    history: list[ClaimNode],
    paths: list[EvidencePath],
) -> list[str]:
    ids = {claim.claim_id for claim in [*active, *history]}
    path_ids = {node_id for path in paths for node_id in path.node_ids}
    return sorted(ids | path_ids)


def _discarded_paths(fixture: ShadowInput, paths: list[EvidencePath]) -> list[str]:
    selected = {edge_id for path in paths for edge_id in path.edge_ids}
    return [f"path:{edge.edge_id}" for edge in fixture.edges if edge.edge_id not in selected]


def _trace_identity(fixture: ShadowInput, paths: list[EvidencePath]) -> dict[str, object]:
    return {
        "vector_used": True,
        "graph_used": bool(paths),
        "max_hops": fixture.request.max_hops,
        "strategy": ["baseline_candidates", "claim_policy", "typed_edges"],
        "baseline_node_ids": fixture.baseline_claim_ids,
    }


def _visited_ids(fixture: ShadowInput) -> list[str]:
    claim_ids = {claim.claim_id for claim in fixture.claims}
    edge_ids = {node_id for edge in fixture.edges for node_id in (edge.source_id, edge.target_id)}
    return sorted(claim_ids | edge_ids)


def _trace(
    fixture: ShadowInput,
    paths: list[EvidencePath],
    returned: list[str],
) -> RetrievalTrace:
    visited = _visited_ids(fixture)
    detail = {
        "visited_node_ids": visited,
        "returned_node_ids": returned,
        "discarded_node_ids": sorted(set(visited) - set(returned)),
        "selected_path_ids": [path.path_id for path in paths],
        "discarded_path_ids": _discarded_paths(fixture, paths),
        "fallback_reason": None if paths else NO_VERIFIED_PATH,
    }
    return RetrievalTrace.model_validate({**_trace_identity(fixture, paths), **detail})


def _selection(
    fixture: ShadowInput,
) -> tuple[list[ClaimNode], list[ClaimNode], list[EvidencePath], list[EvidencePath]]:
    eligible = _eligible_claims(fixture)
    active = _active_claims(eligible, fixture.request.as_of)
    active_ids = {claim.claim_id for claim in active}
    history = _history_claims(eligible, active_ids, fixture.request.as_of)
    path_claim_ids = active_ids | {claim.claim_id for claim in history}
    edges = _usable_edges(fixture, path_claim_ids)
    paths = _evidence_paths(edges, active_ids, fixture.request.max_hops)
    supporting, conflicting = _partition_paths(paths)
    return active, history, supporting, conflicting


def _uncertainty(active: list[ClaimNode]) -> list[str]:
    contested = any(claim.status is ClaimStatus.CONTESTED for claim in active)
    return ["contested_claims_present"] if contested else []


def build_shadow_result(fixture: ShadowInput) -> CurrentValidClaimResult:
    active, history, supporting, conflicting = _selection(fixture)
    paths = [*supporting, *conflicting]
    returned = _returned_ids(active, history, paths)
    return CurrentValidClaimResult(
        request=fixture.request,
        active_claims=active,
        supporting_paths=supporting,
        conflicting_paths=conflicting,
        historical_claims=history,
        uncertainty=_uncertainty(active),
        retrieval_trace=_trace(fixture, paths, returned),
    )


def _result_ids(result: CurrentValidClaimResult) -> tuple[list[str], list[str], list[str]]:
    active = [claim.claim_id for claim in result.active_claims]
    history = [claim.claim_id for claim in result.historical_claims]
    conflicts = _conflicting_ids(result.conflicting_paths, {*active, *history})
    return active, conflicts, history


def _reinforcement(fixture: ShadowFixture) -> bool | None:
    receipt = fixture.attribution_receipt
    return receipt.reinforcement_eligible() if receipt is not None else None


def _expected_checks(
    fixture: ShadowFixture,
    active: list[str],
    conflicts: list[str],
    history: list[str],
    result: CurrentValidClaimResult,
) -> dict[str, tuple[object, object]]:
    expected = fixture.expected
    return {
        "active_claim_ids": (active, expected.active_claim_ids),
        "conflicting_claim_ids": (conflicts, expected.conflicting_claim_ids),
        "historical_claim_ids": (history, expected.historical_claim_ids),
        "fallback_reason": (result.retrieval_trace.fallback_reason, expected.fallback_reason),
        "reinforcement_eligible": (_reinforcement(fixture), expected.reinforcement_eligible),
    }


def _mismatches(
    fixture: ShadowFixture,
    active: list[str],
    conflicts: list[str],
    history: list[str],
    result: CurrentValidClaimResult,
) -> list[str]:
    checks = _expected_checks(fixture, active, conflicts, history, result)
    return [name for name, values in checks.items() if values[0] != values[1]]


def _baseline_delta(baseline: list[str], active: list[str]) -> tuple[list[str], list[str]]:
    removed = sorted(set(baseline) - set(active))
    added = sorted(set(active) - set(baseline))
    return removed, added


def _comparison_payload(fixture: ShadowFixture, result: CurrentValidClaimResult,
                        active: list[str], conflicts: list[str],
                        history: list[str]) -> dict[str, object]:
    removed, added = _baseline_delta(fixture.baseline_claim_ids, active)
    return {
        "fixture_id": fixture.fixture_id, "category": fixture.category,
        "baseline_claim_ids": fixture.baseline_claim_ids, "shadow_active_claim_ids": active,
        "removed_baseline_ids": removed, "added_by_policy_ids": added,
        "conflicting_claim_ids": conflicts, "historical_claim_ids": history,
        "reinforcement_eligible": _reinforcement(fixture),
        "mismatches": _mismatches(fixture, active, conflicts, history, result), "result": result,
    }


def compare_fixture(fixture: ShadowFixture) -> ShadowComparison:
    result = build_shadow_result(fixture)
    active, conflicts, history = _result_ids(result)
    payload = _comparison_payload(fixture, result, active, conflicts, history)
    return ShadowComparison.model_validate(payload)


def suite_report(comparisons: list[ShadowComparison]) -> ShadowSuiteReport:
    failures = {item.fixture_id: item.mismatches for item in comparisons if item.mismatches}
    categories: dict[str, int] = {}
    for item in comparisons:
        categories[item.category] = categories.get(item.category, 0) + 1
    return ShadowSuiteReport(
        total=len(comparisons),
        passed=len(comparisons) - len(failures),
        failed=len(failures),
        category_counts=categories,
        failures=failures,
    )
