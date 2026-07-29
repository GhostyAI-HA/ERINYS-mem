"""Current Valid Claim shadow evaluator の回帰テスト。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from erinys_memory.claim_contracts import ClaimStatus
from erinys_memory.claim_shadow import (
    NO_VERIFIED_PATH,
    ShadowFixture,
    build_shadow_result,
    compare_fixture,
    load_fixtures,
    suite_report,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "current_valid_claim_v0.jsonl"
SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "current_valid_claim_shadow.py"
EXPECTED_CATEGORIES = {
    "simple_update": 3, "as_of": 3, "delayed_recording": 3,
    "conflicting_evidence": 3, "scope_difference": 4, "retraction": 3,
    "rare_critical": 3, "attribution": 3, "traversal_leakage": 3,
    "no_path_fallback": 2,
}


def _fixtures() -> list[ShadowFixture]:
    return load_fixtures(FIXTURE_PATH)


def _fixture(fixture_id: str) -> ShadowFixture:
    return next(item for item in _fixtures() if item.fixture_id == fixture_id)


def _active_claim_payload(project: str = "ERINYS") -> dict[str, object]:
    return {
        "claim_id": "claim-1", "proposition": "Bounded paths are active",
        "scope": {"project": project},
        "valid_time": {"from": "2026-07-01T00:00:00Z", "to": None},
        "recorded_at": "2026-07-01T01:00:00Z", "status": "ACTIVE",
        "confidence": 0.9,
        "provenance": {"sources": [{"source_id": "source-1", "source_type": "policy"}]},
    }


def _fixture_payload(edges: list[dict[str, object]], max_hops: int = 2) -> dict[str, object]:
    return {
        "fixture_id": "synthetic", "category": "synthetic",
        "request": {
            "query": "path?", "as_of": "2026-07-27T00:00:00Z",
            "knowledge_as_of": "2026-07-27T00:00:00Z",
            "scope": {"project": "ERINYS"}, "max_hops": max_hops,
        },
        "claims": [_active_claim_payload()], "edges": edges,
        "baseline_claim_ids": ["claim-1"],
        "expected": {"active_claim_ids": ["claim-1"]},
    }


def test_fixed_30_fixture_suite_passes() -> None:
    comparisons = [compare_fixture(item) for item in _fixtures()]
    report = suite_report(comparisons)
    assert (report.total, report.passed, report.failed) == (30, 30, 0)
    assert report.category_counts == EXPECTED_CATEGORIES


def test_shadow_cli_returns_machine_readable_report() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--fixtures", str(FIXTURE_PATH)],
        check=False, capture_output=True, text=True,
    )
    payload = json.loads(completed.stdout)
    assert completed.returncode == 0
    assert payload["report"]["passed"] == 30


def test_trace_records_unreturned_and_discarded_candidates() -> None:
    trace = build_shadow_result(_fixture("traversal-01")).retrieval_trace
    assert "tv01-archive" in trace.visited_node_ids
    assert "tv01-archive" in trace.discarded_node_ids
    assert "path:edge-tv01-candidate" in trace.discarded_path_ids


def test_refuted_claim_is_history_not_superseded_alias() -> None:
    result = build_shadow_result(_fixture("retraction-02"))
    assert result.historical_claims[0].status is ClaimStatus.REFUTED
    assert result.active_claims[0].claim_id == "rt02-new"


def test_two_hop_evidence_path_honors_bound() -> None:
    edges = [
        _edge("edge-source", "source-1", "evidence-1", "derived_from"),
        _edge("edge-claim", "evidence-1", "claim-1", "supports"),
    ]
    fixture = ShadowFixture.model_validate(_fixture_payload(edges))
    path = build_shadow_result(fixture).supporting_paths[0]
    assert path.edge_ids == ["edge-source", "edge-claim"]


def test_one_hop_limit_does_not_leak_predecessor() -> None:
    edges = [
        _edge("edge-source", "source-1", "evidence-1", "derived_from"),
        _edge("edge-claim", "evidence-1", "claim-1", "supports"),
    ]
    fixture = ShadowFixture.model_validate(_fixture_payload(edges, max_hops=1))
    path = build_shadow_result(fixture).supporting_paths[0]
    assert path.edge_ids == ["edge-claim"]


def _edge(edge_id: str, source: str, target: str, relation: str) -> dict[str, object]:
    return {
        "edge_id": edge_id, "source_id": source, "target_id": target,
        "relation": relation, "origin": "DETERMINISTIC",
        "status": "VERIFIED", "confidence": 1.0,
    }


def test_rejected_deterministic_edge_is_not_retrieval_evidence() -> None:
    edge = _edge("edge-rejected", "source-1", "claim-1", "supports")
    edge["status"] = "REJECTED"
    fixture = ShadowFixture.model_validate(_fixture_payload([edge]))
    trace = build_shadow_result(fixture).retrieval_trace
    assert trace.fallback_reason == NO_VERIFIED_PATH


def test_expired_edge_is_not_retrieval_evidence() -> None:
    edge = _edge("edge-expired", "source-1", "claim-1", "supports")
    edge["valid_time"] = {
        "from": "2026-07-01T00:00:00Z", "to": "2026-07-10T00:00:00Z",
    }
    fixture = ShadowFixture.model_validate(_fixture_payload([edge]))
    trace = build_shadow_result(fixture).retrieval_trace
    assert trace.fallback_reason == NO_VERIFIED_PATH


def test_out_of_scope_claim_cannot_enter_conflict_path() -> None:
    payload = _fixture_payload([_edge("edge-cross", "claim-other", "claim-1", "contradicts")])
    payload["claims"] = [_active_claim_payload(), _other_claim_payload()]
    result = build_shadow_result(ShadowFixture.model_validate(payload))
    assert result.conflicting_paths == []
    assert "claim-other" in result.retrieval_trace.discarded_node_ids


def _other_claim_payload() -> dict[str, object]:
    payload = _active_claim_payload("OTHER")
    payload["claim_id"] = "claim-other"
    payload["provenance"] = {
        "sources": [{"source_id": "source-other", "source_type": "policy"}],
    }
    return payload
