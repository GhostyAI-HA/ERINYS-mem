"""Typed Contract v0 の境界と安全不変条件を検証する。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from erinys_memory.claim_contracts import (
    AttributionStatus,
    ClaimNode,
    ClaimScope,
    ClaimStatus,
    EdgeOrigin,
    EdgeStatus,
    EdgeType,
    EvidencePath,
    MemoryEdge,
    OutcomeAttributionReceipt,
    OutcomeStatus,
    ReinforcementTarget,
    RiskClass,
)


def _claim_payload(status: str = "ACTIVE") -> dict[str, object]:
    valid_to = "2026-07-20T00:00:00Z" if status != "ACTIVE" else None
    return {
        "claim_id": "claim-1", "proposition": "Policy A is valid",
        "scope": {"project": "ERINYS"},
        "valid_time": {"from": "2026-07-01T00:00:00Z", "to": valid_to},
        "recorded_at": "2026-07-02T00:00:00Z", "status": status,
        "confidence": 0.9,
        "provenance": {"sources": [{"source_id": "source-1", "source_type": "policy"}]},
    }


def _receipt_payload() -> dict[str, object]:
    return {
        "memory_id": "claim-1", "usage": "decision_basis",
        "decision_id": "decision-1", "outcome_id": "outcome-1",
        "outcome_status": "POSITIVE", "attribution_status": "ACCEPTED",
        "attribution_confidence": 0.8, "verified_by": ["human"],
        "scope": {"project": "ERINYS"},
        "reinforcement_targets": ["RETRIEVAL_PRIORITY"],
    }


def test_claim_requires_source_provenance() -> None:
    payload = _claim_payload()
    payload["provenance"] = {}
    with pytest.raises(ValidationError, match="source provenance"):
        ClaimNode.model_validate(payload)


def test_historical_states_remain_distinct() -> None:
    statuses = ("SUPERSEDED", "REFUTED", "RETRACTED")
    parsed = [ClaimNode.model_validate(_claim_payload(status)).status for status in statuses]
    assert parsed == [ClaimStatus.SUPERSEDED, ClaimStatus.REFUTED, ClaimStatus.RETRACTED]


def test_historical_claim_requires_closed_valid_time() -> None:
    payload = _claim_payload("REFUTED")
    payload["valid_time"] = {"from": "2026-07-01T00:00:00Z", "to": None}
    with pytest.raises(ValidationError, match="valid_time.to"):
        ClaimNode.model_validate(payload)


def test_claim_contract_rejects_unknown_fields() -> None:
    payload = _claim_payload()
    payload["truth"] = True
    with pytest.raises(ValidationError, match="Extra inputs"):
        ClaimNode.model_validate(payload)


def test_evidence_path_is_shape_checked() -> None:
    with pytest.raises(ValidationError, match="edge count"):
        EvidencePath(
            path_id="path-1", node_ids=["a", "b", "c"], edge_ids=["edge-1"],
            relations=[EdgeType.SUPPORTS], score=1.0, selected_reason="test",
        )


def test_candidate_edge_cannot_be_marked_verified() -> None:
    with pytest.raises(ValidationError, match="cannot be verified"):
        MemoryEdge(
            edge_id="edge-1", source_id="a", target_id="b",
            relation=EdgeType.SUPPORTS, origin=EdgeOrigin.CANDIDATE,
            status=EdgeStatus.VERIFIED, confidence=0.8,
        )


def test_risk_aware_retention_protects_rare_critical_claim() -> None:
    payload = _claim_payload()
    payload["risk_class"] = RiskClass.CRITICAL
    assert ClaimNode.model_validate(payload).retention_protected()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("outcome_status", OutcomeStatus.NEGATIVE),
        ("attribution_status", AttributionStatus.UNRESOLVED),
        ("attribution_confidence", 0.69),
        ("verified_by", []),
        ("reinforcement_targets", []),
    ],
)
def test_reinforcement_fails_closed(field: str, value: object) -> None:
    payload = _receipt_payload()
    payload[field] = value
    receipt = OutcomeAttributionReceipt.model_validate(payload)
    assert not receipt.reinforcement_eligible()


def test_reinforcement_requires_positive_attributed_outcome() -> None:
    receipt = OutcomeAttributionReceipt.model_validate(_receipt_payload())
    assert receipt.reinforcement_eligible()
    assert receipt.reinforcement_targets == [ReinforcementTarget.RETRIEVAL_PRIORITY]


def test_scope_requires_at_least_one_dimension() -> None:
    with pytest.raises(ValidationError, match="at least one dimension"):
        ClaimScope()
