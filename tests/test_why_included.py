"""Tests for search result explainability (`why_included`)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from erinys_memory.db import insert_observation_with_embedding  # noqa: E402
from erinys_memory.embedding import serialize_f32  # noqa: E402
from erinys_memory.search import rrf_hybrid_search  # noqa: E402


def _insert(db, engine, title: str, content: str, project: str = "alpha", metadata=None) -> int:
    payload = {"title": title, "content": content, "project": project, "scope": "project"}
    if metadata is not None:
        payload["metadata"] = json.dumps(metadata)
    return insert_observation_with_embedding(db, payload, serialize_f32(engine.embed(content)))


def test_why_included_is_present_and_structured(db, engine) -> None:
    _insert(db, engine, "JWT bug", "JWT httpOnly flag was missing on the /api/auth endpoint")
    results = rrf_hybrid_search(
        db, "httpOnly auth flag", engine.embed("httpOnly auth flag"), project="alpha", limit=5
    )
    assert results, "search should return the inserted observation"
    why = results[0]["why_included"]
    assert {"summary", "channels", "signals", "score", "query", "provenance"} <= set(why)
    assert isinstance(why["summary"], str) and why["summary"]
    assert why["channels"], "a returned result must have surfaced via >=1 channel"
    for ch in why["channels"]:
        assert ch["channel"] in {"keyword", "semantic"}
        assert isinstance(ch["rank"], int)
    for field in ("rrf_score", "boost", "effective_strength", "effective_score", "graph_boosted"):
        assert field in why["score"]


def test_why_included_reports_keyword_channel_for_exact_terms(db, engine) -> None:
    _insert(db, engine, "Postgres choice", "We decided to use PostgreSQL for the primary datastore")
    results = rrf_hybrid_search(
        db, "PostgreSQL datastore", engine.embed("PostgreSQL datastore"), project="alpha", limit=5
    )
    top = results[0]["why_included"]
    channels = {c["channel"] for c in top["channels"]}
    assert "keyword" in channels, f"exact-term query should match the keyword channel: {top}"


def test_why_included_surfaces_provenance(db, engine) -> None:
    _insert(
        db,
        engine,
        "Provenanced fact",
        "The deployment moved from AWS to GCP in Q2",
        metadata={"provenance": {"principal": "tester", "derived_via": "save"}},
    )
    results = rrf_hybrid_search(
        db, "AWS to GCP deployment", engine.embed("AWS to GCP deployment"), project="alpha", limit=5
    )
    prov = results[0]["why_included"]["provenance"]
    assert prov is not None and prov.get("principal") == "tester"


def test_why_included_does_not_leak_internal_signal_key(db, engine) -> None:
    _insert(db, engine, "Leak check", "some searchable content about caching layers")
    results = rrf_hybrid_search(
        db, "caching layers", engine.embed("caching layers"), project="alpha", limit=5
    )
    assert "_why_signals" not in results[0], "internal scratch key must be stripped from results"
