"""Tests for the zero-LLM answerability / abstention signal.

The signal distinguishes "the top result answers the query" from "retrieval
returned something topically related but not an answer" — so an agent can abstain
instead of asserting a plausible-but-wrong memory.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from erinys_memory.db import insert_observation_with_embedding  # noqa: E402
from erinys_memory.embedding import serialize_f32  # noqa: E402
from erinys_memory.search import rrf_hybrid_search, assess_answerability  # noqa: E402

_QUERY = "what is my hamster called"


def _insert(db, engine, content, project="alpha"):
    payload = {"title": content[:40], "content": content, "project": project, "scope": "project"}
    return insert_observation_with_embedding(db, payload, serialize_f32(engine.embed(content)))


def test_answerable_when_top_result_grounds_the_query(db, engine) -> None:
    _insert(db, engine, "My hamster is called Nibbles and loves his exercise wheel")
    _insert(db, engine, "I went hiking at Muir Woods last weekend")
    results = rrf_hybrid_search(db, _QUERY, engine.embed(_QUERY), project="alpha", limit=5)
    a = assess_answerability(_QUERY, results)
    assert a["answerable"] is True, a
    assert a["grounding"] >= 0.5


def test_abstains_when_answer_absent(db, engine) -> None:
    # No hamster anywhere — retrieval will still return the nearest neighbours,
    # but the query's entity is not grounded, so we must flag "likely no answer".
    _insert(db, engine, "My cat Luna sleeps all day on the couch")
    _insert(db, engine, "I enjoy hiking on the weekends")
    results = rrf_hybrid_search(db, _QUERY, engine.embed(_QUERY), project="alpha", limit=5)
    assert results, "retrieval should still return topically-near results"
    a = assess_answerability(_QUERY, results)
    assert a["answerable"] is False, a
    assert a["grounding"] < 0.5


def test_empty_results_are_unanswerable(db, engine) -> None:
    a = assess_answerability(_QUERY, [])
    assert a["answerable"] is False
    assert a["score"] == 0.0
