from __future__ import annotations

import importlib.util
import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest

from erinys_memory.search import rrf_hybrid_search


@pytest.fixture(autouse=True)
def _default_server() -> None:
    return None


class FakeCursor:
    def __init__(self, rows: list[object]) -> None:
        self.rows = rows

    def fetchall(self) -> list[object]:
        return self.rows

    def fetchone(self) -> object | None:
        return self.rows[0] if self.rows else None


class SearchFakeDB:
    def __init__(
        self,
        created_at: object | None = None,
        last_accessed: object | None = None,
    ) -> None:
        self.project_batch_queries: list[tuple[str, list[object]]] = []
        self.project_single_queries: list[tuple[str, object]] = []
        self.created_at = created_at
        self.last_accessed = last_accessed

    def execute(self, sql: str, params: object = ()) -> FakeCursor:
        normalized = " ".join(sql.split())
        if "FROM observations_fts" in normalized:
            return FakeCursor([])
        if "FROM vec_observations" in normalized:
            return FakeCursor([(1, 1), (2, 2), (3, 3)])
        if normalized.startswith("SELECT COUNT(*) FROM observations WHERE project = ? AND id IN ("):
            self.project_batch_queries.append((normalized, list(params)))
            return FakeCursor([(1,)])
        if normalized == "SELECT 1 FROM observations WHERE id = ? AND project = ?":
            self.project_single_queries.append((normalized, params))
            return FakeCursor([(1,)])
        if normalized.startswith("SELECT o.* FROM observations o WHERE o.id IN (?,?,?) AND o.project = ?"):
            now = datetime.now(timezone.utc)
            row = {
                "id": 2,
                "project": "target",
                "content": "target memo",
                "base_strength": 1.0,
                "created_at": self.created_at or now,
                "last_accessed": self.last_accessed,
                "access_count": 0,
            }
            return FakeCursor([row])
        raise AssertionError(f"unexpected SQL: {normalized}")


class StubEmbeddingEngine:
    def __init__(self) -> None:
        self.batch_calls: list[list[str]] = []
        self.raise_on_batch = False

    def embed(self, text: str) -> list[float]:
        return [float(len(text))]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self.raise_on_batch:
            raise AssertionError("embed_batch should not be called")
        self.batch_calls.append(list(texts))
        return [[float(len(text))] for text in texts]


class InsertRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, object], list[float]]] = []

    def __call__(
        self,
        db: object,
        payload: dict[str, object],
        embedding: list[float],
    ) -> int:
        self.calls.append((payload, embedding))
        return len(self.calls)


def _load_locomo_module(
    monkeypatch: pytest.MonkeyPatch,
    engine: StubEmbeddingEngine,
    recorder: InsertRecorder,
):
    package = types.ModuleType("erinys_memory")
    package.__path__ = []
    config_mod = types.ModuleType("erinys_memory.config")
    db_mod = types.ModuleType("erinys_memory.db")
    embedding_mod = types.ModuleType("erinys_memory.embedding")
    search_mod = types.ModuleType("erinys_memory.search")
    pref_mod = types.ModuleType("erinys_memory.preference_extract")
    enhanced_mod = types.ModuleType("enhanced_search")

    class ErinysConfig:
        def __init__(self, db_path: str = ":memory:") -> None:
            self.db_path = db_path

    config_mod.ErinysConfig = ErinysConfig
    db_mod.init_db = lambda config: object()
    db_mod.insert_observation_with_embedding = recorder
    db_mod.embedding_engine = engine
    embedding_mod.serialize_f32 = lambda values: values
    search_mod.rrf_hybrid_search = lambda *args, **kwargs: []
    search_mod.focus_query_for_embedding = lambda query: query
    search_mod._is_temporal_query = lambda query: False
    search_mod.collapse_by_session = lambda results, limit: results
    pref_mod.extract_all = lambda text: []
    enhanced_mod.enhanced_hybrid_search_v2 = lambda *args, **kwargs: []

    monkeypatch.setitem(sys.modules, "erinys_memory", package)
    monkeypatch.setitem(sys.modules, "erinys_memory.config", config_mod)
    monkeypatch.setitem(sys.modules, "erinys_memory.db", db_mod)
    monkeypatch.setitem(sys.modules, "erinys_memory.embedding", embedding_mod)
    monkeypatch.setitem(sys.modules, "erinys_memory.search", search_mod)
    monkeypatch.setitem(sys.modules, "erinys_memory.preference_extract", pref_mod)
    monkeypatch.setitem(sys.modules, "enhanced_search", enhanced_mod)

    path = Path(__file__).resolve().parent.parent / "benchmarks" / "locomo_bench.py"
    spec = importlib.util.spec_from_file_location("locomo_bench_testmod", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_project_widening_uses_batched_existence_query() -> None:
    db = SearchFakeDB()
    results = rrf_hybrid_search(db, "query", [0.1], project="target", limit=2)
    assert db.project_single_queries == []
    assert len(db.project_batch_queries) == 1
    assert results[0]["project"] == "target"


def test_rrf_hybrid_search_parses_iso_datetime_fields() -> None:
    db = SearchFakeDB(
        created_at="2026-04-15T10:00:00Z",
        last_accessed="2026-04-16 09:00:00+00:00",
    )
    results = rrf_hybrid_search(db, "query", [0.1], project="target", limit=2)
    assert isinstance(results[0]["created_at"], datetime)
    assert isinstance(results[0]["last_accessed"], datetime)
    assert results[0]["created_at"].tzinfo is not None


class ClosingDB:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_locomo_precompute_embeddings_uses_json_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    engine = StubEmbeddingEngine()
    recorder = InsertRecorder()
    module = _load_locomo_module(monkeypatch, engine, recorder)
    sessions = {
        "conv": [
            {
                "session_id": "session_1",
                "dialogs": [{"speaker": "A", "text": "hello"}],
            }
        ]
    }

    module._EMBED_CACHE.clear()
    module.precompute_embeddings(sessions, show_progress=False, cache_dir=str(tmp_path))
    cache_path = tmp_path / module.EMBED_CACHE_FILENAME

    assert cache_path.exists()
    assert json.loads(cache_path.read_text(encoding="utf-8")) == {"conv_session_1": [8.0]}

    module._EMBED_CACHE.clear()
    engine.raise_on_batch = True
    module.precompute_embeddings(sessions, show_progress=False, cache_dir=str(tmp_path))
    assert engine.batch_calls == [["A: hello"]]


def test_locomo_ingest_sessions_backfills_missing_cache_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = StubEmbeddingEngine()
    recorder = InsertRecorder()
    module = _load_locomo_module(monkeypatch, engine, recorder)
    sessions = [
        {
            "session_id": "session_1",
            "session_num": 1,
            "dialogs": [{"speaker": "A", "text": "first"}],
            "date": "",
        },
        {
            "session_id": "session_2",
            "session_num": 2,
            "dialogs": [{"speaker": "A", "text": "second"}],
            "date": "",
        },
    ]

    module._EMBED_CACHE.clear()
    module._EMBED_CACHE["conv_session_1"] = [8.0]
    sid_to_obsid = module.ingest_sessions(object(), sessions, "conv")

    assert engine.batch_calls == [["A: second"]]
    assert module._EMBED_CACHE["conv_session_2"] == [9.0]
    assert sid_to_obsid == {"session_1": 1, "session_2": 2}
    assert [call[1] for call in recorder.calls[:2]] == [[8.0], [9.0]]


def test_locomo_run_benchmark_closes_db_on_search_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = StubEmbeddingEngine()
    recorder = InsertRecorder()
    module = _load_locomo_module(monkeypatch, engine, recorder)
    created_dbs: list[ClosingDB] = []

    def _create_db() -> ClosingDB:
        db = ClosingDB()
        created_dbs.append(db)
        return db

    monkeypatch.setattr(module, "create_bench_db", _create_db)
    monkeypatch.setattr(module, "precompute_embeddings", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "load_conversation_sessions",
        lambda conversation: [{"session_id": "session_1", "dialogs": []}],
    )
    monkeypatch.setattr(
        module,
        "ingest_sessions",
        lambda *args, **kwargs: {"session_1": 1},
    )
    module.SEARCH_FNS["rrf"] = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))

    data = [{
        "sample_id": "conv-1",
        "conversation": {},
        "qa": [{"question": "q", "category": 1, "evidence": ["D1:1"]}],
    }]

    with pytest.raises(RuntimeError, match="boom"):
        module.run_benchmark(data, mode="rrf", show_progress=False)

    assert created_dbs[0].closed is True


def test_locomo_run_benchmark_ignores_invalid_metadata_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = StubEmbeddingEngine()
    recorder = InsertRecorder()
    module = _load_locomo_module(monkeypatch, engine, recorder)

    monkeypatch.setattr(module, "create_bench_db", ClosingDB)
    monkeypatch.setattr(module, "precompute_embeddings", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "load_conversation_sessions",
        lambda conversation: [{"session_id": "session_1", "dialogs": []}],
    )
    monkeypatch.setattr(
        module,
        "ingest_sessions",
        lambda *args, **kwargs: {"session_1": 1},
    )
    module.SEARCH_FNS["rrf"] = lambda *args, **kwargs: [
        {"metadata": "not-json"},
        {"metadata": "[]"},
        {"metadata": "{\"session_id\": \"session_1\"}"},
    ]

    results, _summary = module.run_benchmark(
        [{
            "sample_id": "conv-1",
            "conversation": {},
            "qa": [{"question": "q", "category": 1, "evidence": ["D1:1"]}],
        }],
        mode="rrf",
        show_progress=False,
    )

    assert results[0]["retrieved_session_ids"] == ["session_1"]
