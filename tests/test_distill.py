"""distill.py の3粒度蒸留テスト。

LLM呼び出しは全てモックし、成功/失敗の全パスを検証する。
"""

from __future__ import annotations

import json
import sqlite3
from io import BytesIO
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import pytest

from erinys_memory.config import ErinysConfig
from erinys_memory.db import init_db, insert_observation_with_embedding
from erinys_memory.distill import (
    LEVELS,
    _build_prompt,
    _llm_generate,
    _parse_llm_response,
    _template_distillations,
    distill_observation,
)
from erinys_memory.embedding import serialize_f32

import erinys_memory.db as db_module
import erinys_memory.distill as distill_module


@pytest.fixture
def config() -> ErinysConfig:
    return ErinysConfig(
        db_path=":memory:",
        db_backup_on_init=False,
        auto_distill_on_save=False,
        distill_use_llm=True,
        distill_model="test-model",
        distill_endpoint="http://localhost:11434/api/generate",
        distill_timeout=5,
    )


@pytest.fixture
def engine():
    return db_module.embedding_engine


@pytest.fixture
def db(config: ErinysConfig, engine) -> Iterator[sqlite3.Connection]:
    connection = init_db(config)
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture(autouse=True)
def _patch_engine(db, engine, monkeypatch):
    monkeypatch.setattr(distill_module, "embedding_engine", engine)


def _insert_observation(
    db: sqlite3.Connection,
    engine,
    title: str = "Test observation",
    content: str = "Fixed CORS headers missing on /api/v2/users endpoint. Added Access-Control-Allow-Origin header.",
    obs_type: str = "bugfix",
    project: str = "test",
) -> int:
    payload = {
        "title": title,
        "content": content,
        "type": obs_type,
        "project": project,
        "scope": "project",
    }
    embedding = engine.embed(content)
    return insert_observation_with_embedding(db, payload, serialize_f32(embedding))


def _make_llm_response(concrete: str, abstract: str, meta: str) -> bytes:
    inner = json.dumps({"concrete": concrete, "abstract": abstract, "meta": meta})
    outer = json.dumps({"response": inner})
    return outer.encode("utf-8")


def _mock_urlopen_success(response_bytes: bytes):
    mock_resp = MagicMock()
    mock_resp.read.return_value = response_bytes
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestParseResponse:
    def test_valid_json(self):
        raw = json.dumps({"concrete": "fact", "abstract": "pattern", "meta": "lesson"})
        result = _parse_llm_response(raw)
        assert result is not None
        assert result["concrete"] == "fact"
        assert result["abstract"] == "pattern"
        assert result["meta"] == "lesson"

    def test_json_with_markdown_fences(self):
        raw = '```json\n{"concrete": "fact", "abstract": "pattern", "meta": "lesson"}\n```'
        result = _parse_llm_response(raw)
        assert result is not None
        assert result["concrete"] == "fact"

    def test_missing_key_returns_none(self):
        raw = json.dumps({"concrete": "fact", "abstract": "pattern"})
        assert _parse_llm_response(raw) is None

    def test_empty_value_returns_none(self):
        raw = json.dumps({"concrete": "", "abstract": "pattern", "meta": "lesson"})
        assert _parse_llm_response(raw) is None

    def test_invalid_json_returns_none(self):
        assert _parse_llm_response("not json at all") is None

    def test_non_dict_returns_none(self):
        assert _parse_llm_response("[1, 2, 3]") is None


class TestBuildPrompt:
    def test_base_prompt_includes_title_and_content(self):
        prompt = _build_prompt("My Title", "My Content", "manual")
        assert "My Title" in prompt
        assert "My Content" in prompt

    def test_bugfix_type_adds_guidance(self):
        prompt = _build_prompt("Fix", "Details", "bugfix")
        assert "root cause" in prompt

    def test_anti_pattern_type_adds_guidance(self):
        prompt = _build_prompt("Bad", "Details", "anti_pattern")
        assert "smell" in prompt

    def test_content_truncated_at_4000(self):
        long_content = "x" * 10000
        prompt = _build_prompt("T", long_content, "manual")
        assert "x" * 4000 in prompt
        assert "x" * 5000 not in prompt


class TestLlmGenerate:
    def test_success(self, config):
        response_bytes = _make_llm_response("fact", "pattern", "lesson")
        mock_resp = _mock_urlopen_success(response_bytes)
        with patch("erinys_memory.distill.urllib.request.urlopen", return_value=mock_resp):
            result = _llm_generate(config, "title", "content", "manual")
        assert result is not None
        assert result["concrete"] == "fact"
        assert result["abstract"] == "pattern"
        assert result["meta"] == "lesson"

    def test_connection_error_returns_none(self, config):
        import urllib.error
        with patch(
            "erinys_memory.distill.urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            result = _llm_generate(config, "title", "content", "manual")
        assert result is None

    def test_timeout_returns_none(self, config):
        with patch(
            "erinys_memory.distill.urllib.request.urlopen",
            side_effect=TimeoutError("timed out"),
        ):
            result = _llm_generate(config, "title", "content", "manual")
        assert result is None

    def test_invalid_response_returns_none(self, config):
        bad_response = json.dumps({"response": "not valid json"}).encode("utf-8")
        mock_resp = _mock_urlopen_success(bad_response)
        with patch("erinys_memory.distill.urllib.request.urlopen", return_value=mock_resp):
            result = _llm_generate(config, "title", "content", "manual")
        assert result is None

    def test_disabled_returns_none(self, config):
        config.distill_use_llm = False
        result = _llm_generate(config, "title", "content", "manual")
        assert result is None

    def test_retries_on_first_failure(self, config):
        import urllib.error
        response_bytes = _make_llm_response("fact", "pattern", "lesson")
        mock_resp = _mock_urlopen_success(response_bytes)
        with patch(
            "erinys_memory.distill.urllib.request.urlopen",
            side_effect=[urllib.error.URLError("first fail"), mock_resp],
        ):
            result = _llm_generate(config, "title", "content", "manual")
        assert result is not None
        assert result["concrete"] == "fact"


class TestTemplateDistillations:
    def test_produces_all_three_levels(self):
        source = {
            "title": "Test",
            "content": "Some content about authentication",
            "is_anti_pattern": False,
        }
        result = _template_distillations(source)
        assert "concrete" in result
        assert "abstract" in result
        assert "meta" in result
        assert all(isinstance(v, str) and v for v in result.values())


class TestDistillObservation:
    def test_llm_success_creates_three_levels(self, db, engine, config):
        obs_id = _insert_observation(db, engine)
        response_bytes = _make_llm_response(
            "CORS was missing on the users endpoint.",
            "New endpoints need a CORS review checklist.",
            "Security concerns should be opt-out, not opt-in.",
        )
        mock_resp = _mock_urlopen_success(response_bytes)
        with patch("erinys_memory.distill.urllib.request.urlopen", return_value=mock_resp):
            result = distill_observation(db, obs_id, "meta", config)
        assert len(result["created"]) == 3
        for record in result["created"]:
            meta = record.get("metadata") or {}
            assert meta.get("distill_method") == "llm"
            assert "distill_error" not in meta

    def test_llm_failure_falls_back_to_template(self, db, engine, config):
        obs_id = _insert_observation(db, engine)
        import urllib.error
        with patch(
            "erinys_memory.distill.urllib.request.urlopen",
            side_effect=urllib.error.URLError("refused"),
        ):
            result = distill_observation(db, obs_id, "meta", config)
        assert len(result["created"]) == 3
        for record in result["created"]:
            meta = record.get("metadata") or {}
            assert meta.get("distill_method") == "template_fallback"
            assert meta.get("distill_error") == "llm_unavailable"

    def test_llm_disabled_uses_template(self, db, engine, config):
        config.distill_use_llm = False
        obs_id = _insert_observation(db, engine)
        result = distill_observation(db, obs_id, "meta", config)
        assert len(result["created"]) == 3
        for record in result["created"]:
            meta = record.get("metadata") or {}
            assert meta.get("distill_method") == "template_fallback"

    def test_save_api_doesnt_fail_on_llm_error(self, db, engine, config):
        obs_id = _insert_observation(db, engine)
        with patch(
            "erinys_memory.distill.urllib.request.urlopen",
            side_effect=Exception("catastrophic failure"),
        ):
            result = distill_observation(db, obs_id, "meta", config)
        assert result["source"]["id"] == obs_id
        assert len(result["created"]) == 3

    def test_distillation_metadata_records_method(self, db, engine, config):
        obs_id = _insert_observation(db, engine)
        response_bytes = _make_llm_response("fact", "pattern", "lesson")
        mock_resp = _mock_urlopen_success(response_bytes)
        with patch("erinys_memory.distill.urllib.request.urlopen", return_value=mock_resp):
            result = distill_observation(db, obs_id, "meta", config)
        for record in result["created"]:
            meta = record.get("metadata") or {}
            assert "distill_method" in meta
            assert meta["distill_method"] in ("llm", "template_fallback")

    def test_edges_created_for_all_levels(self, db, engine, config):
        obs_id = _insert_observation(db, engine)
        config.distill_use_llm = False
        result = distill_observation(db, obs_id, "meta", config)
        for record in result["created"]:
            edges = db.execute(
                "SELECT * FROM edges WHERE source_id = ? AND relation = 'distilled_from'",
                [record["id"]],
            ).fetchall()
            assert len(edges) == 1

    def test_already_distilled_is_noop(self, db, engine, config):
        obs_id = _insert_observation(db, engine)
        config.distill_use_llm = False
        first = distill_observation(db, obs_id, "meta", config)
        concrete_id = first["created"][0]["id"]
        second = distill_observation(db, concrete_id, "concrete", config)
        assert len(second["created"]) == 0

    def test_partial_distillation_continues(self, db, engine, config):
        obs_id = _insert_observation(db, engine)
        config.distill_use_llm = False
        first = distill_observation(db, obs_id, "concrete", config)
        assert len(first["created"]) == 1
        concrete_id = first["created"][0]["id"]
        second = distill_observation(db, concrete_id, "meta", config)
        assert len(second["created"]) == 2

    def test_llm_content_is_used_not_template(self, db, engine, config):
        obs_id = _insert_observation(db, engine)
        llm_concrete = "LLM generated this unique concrete content."
        llm_abstract = "LLM generated this unique abstract content."
        llm_meta = "LLM generated this unique meta content."
        response_bytes = _make_llm_response(llm_concrete, llm_abstract, llm_meta)
        mock_resp = _mock_urlopen_success(response_bytes)
        with patch("erinys_memory.distill.urllib.request.urlopen", return_value=mock_resp):
            result = distill_observation(db, obs_id, "meta", config)
        contents = [r["content"] for r in result["created"]]
        assert llm_concrete in contents
        assert llm_abstract in contents
        assert llm_meta in contents

    def test_provenance_points_to_raw_observation(self, db, engine, config):
        """Codex Warning fix: partial distillation provenance always points to raw observation."""
        config.distill_use_llm = False
        obs_id = _insert_observation(db, engine)
        first = distill_observation(db, obs_id, "concrete", config)
        concrete_id = first["created"][0]["id"]
        second = distill_observation(db, concrete_id, "meta", config)
        for record in second["created"]:
            assert record["distilled_from"] == obs_id, (
                f"distilled_from should point to raw observation {obs_id}, "
                f"got {record['distilled_from']}"
            )
            edges = db.execute(
                "SELECT target_id FROM edges WHERE source_id = ? AND relation = 'distilled_from'",
                [record["id"]],
            ).fetchall()
            assert len(edges) == 1
            assert edges[0][0] == obs_id

    def test_stale_distill_error_cleaned_on_llm_success(self, db, engine, config):
        """Codex Warning fix: old distill_error is cleaned when LLM succeeds on re-distillation."""
        import urllib.error
        obs_id = _insert_observation(db, engine)
        with patch(
            "erinys_memory.distill.urllib.request.urlopen",
            side_effect=urllib.error.URLError("refused"),
        ):
            first = distill_observation(db, obs_id, "concrete", config)
        concrete = first["created"][0]
        concrete_meta = concrete.get("metadata") or {}
        assert concrete_meta.get("distill_error") == "llm_unavailable"

        response_bytes = _make_llm_response("fact2", "pattern2", "lesson2")
        mock_resp = _mock_urlopen_success(response_bytes)
        with patch("erinys_memory.distill.urllib.request.urlopen", return_value=mock_resp):
            second = distill_observation(db, concrete["id"], "meta", config)
        for record in second["created"]:
            meta = record.get("metadata") or {}
            assert meta.get("distill_method") == "llm"
            assert "distill_error" not in meta, (
                f"stale distill_error should be cleaned, got metadata: {meta}"
            )


class TestConfig:
    def test_default_config_has_distill_fields(self):
        config = ErinysConfig(db_path=":memory:", db_backup_on_init=False)
        assert config.distill_model == "gemma3:4b"
        assert "localhost" in config.distill_endpoint
        assert config.distill_timeout > 0
        assert isinstance(config.distill_use_llm, bool)

    def test_negative_timeout_corrected(self):
        config = ErinysConfig(
            db_path=":memory:", db_backup_on_init=False, distill_timeout=-5
        )
        assert config.distill_timeout == 20

    def test_remote_endpoint_warns(self):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = ErinysConfig(
                db_path=":memory:",
                db_backup_on_init=False,
                distill_endpoint="http://remote-server.com/api/generate",
            )
            assert any("never leaves your machine" in str(warning.message) for warning in w)

