"""Tests for the product-side temporal date-proximity boost in rrf_hybrid_search.

The boost is opt-in via `as_of`: when the caller anchors the query in time and the
query carries a resolvable relative date ("last Tuesday"), observations whose own
date (metadata.date) is near the resolved target are promoted. Without `as_of`,
behaviour is unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from erinys_memory.db import insert_observation_with_embedding  # noqa: E402
from erinys_memory.embedding import serialize_f32  # noqa: E402
from erinys_memory.search import rrf_hybrid_search, _parse_flexible_date  # noqa: E402


def _insert(db, engine, content, date, project="alpha"):
    payload = {
        "title": content[:40],
        "content": content,
        "project": project,
        "scope": "project",
        "metadata": json.dumps({"date": date}),
    }
    return insert_observation_with_embedding(db, payload, serialize_f32(engine.embed(content)))


# Same content, different dates → base retrieval score is identical, so the
# temporal date boost is the only thing that can reorder them.
_CONTENT = "Had lunch with Sam and talked about the museum plans"
_QUERY = "who did I have lunch with last Tuesday"
_AS_OF = "2023/04/18"          # a Tuesday → "last Tuesday" resolves to 2023/04/11
_NEAR_DATE = "2023/04/11"      # the resolved target
_FAR_DATE = "2023/02/01"


def test_flexible_date_parses_slash_format() -> None:
    assert _parse_flexible_date("2023/04/11 (Tue) 22:13").date().isoformat() == "2023-04-11"
    assert _parse_flexible_date("2023-04-11T10:00:00").date().isoformat() == "2023-04-11"
    assert _parse_flexible_date("not a date") is None


def test_as_of_promotes_date_matching_observation(db, engine) -> None:
    near = _insert(db, engine, _CONTENT + " (April)", _NEAR_DATE)
    far = _insert(db, engine, _CONTENT + " (February)", _FAR_DATE)
    emb = engine.embed(_QUERY)
    ranked = rrf_hybrid_search(db, _QUERY, emb, project="alpha", limit=5, as_of=_AS_OF)
    order = [int(r["id"]) for r in ranked]
    assert near in order and far in order
    assert order.index(near) < order.index(far), (
        "the observation dated on the resolved 'last Tuesday' must outrank the far-dated one"
    )
    near_row = next(r for r in ranked if int(r["id"]) == near)
    assert near_row.get("temporal_date_boost", 1.0) > 1.0


def test_without_as_of_no_temporal_date_boost(db, engine) -> None:
    _insert(db, engine, _CONTENT + " (April)", _NEAR_DATE)
    _insert(db, engine, _CONTENT + " (February)", _FAR_DATE)
    emb = engine.embed(_QUERY)
    ranked = rrf_hybrid_search(db, _QUERY, emb, project="alpha", limit=5)  # no as_of
    assert all("temporal_date_boost" not in r for r in ranked), (
        "temporal date boost must not activate unless the caller supplies as_of"
    )


def test_non_temporal_query_unaffected_by_as_of(db, engine) -> None:
    # A query with no relative-date expression must not trigger the boost even with as_of.
    _insert(db, engine, "The database uses PostgreSQL for storage", _NEAR_DATE)
    emb = engine.embed("which database is used")
    ranked = rrf_hybrid_search(db, "which database is used", emb, project="alpha", limit=5, as_of=_AS_OF)
    assert all("temporal_date_boost" not in r for r in ranked)
