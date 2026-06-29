"""#151 VMG Provenance Visibility の回帰テスト。

build_provenance(純関数)・_lineage_node(抽出)・erinys_lineage(系譜walk)を検証。
vec0 不要(server._fetch_observation を monkeypatch して DB を回避)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from erinys_memory import provenance, server  # noqa: E402


# ---- build_provenance (pure) ----

def test_provenance_schema_and_fields():
    p = provenance.build_provenance("user", "claude-opus-4-8", "save", [1, 2])
    assert p["schema"] == provenance.PROVENANCE_SCHEMA
    assert p["principal"] == "claude-opus-4-8"
    assert p["source"] == "user"
    assert p["derived_via"] == "save"
    assert p["parents"] == [1, 2]
    assert p["recorded_at"]  # ISO ts present


def test_principal_falls_back_to_env_then_unknown(monkeypatch):
    monkeypatch.delenv("ERINYS_PRINCIPAL", raising=False)
    assert provenance.build_provenance("user", None, "save", None)["principal"] == "unknown"
    monkeypatch.setenv("ERINYS_PRINCIPAL", "codex")
    assert provenance.build_provenance("user", None, "save", None)["principal"] == "codex"
    # 明示引数は env より優先
    assert provenance.build_provenance("user", "explicit", "save", None)["principal"] == "explicit"


def test_derived_via_clamped_to_known_set():
    assert provenance.build_provenance("user", "x", "save", None)["derived_via"] == "save"
    assert provenance.build_provenance("user", "x", "distill", None)["derived_via"] == "distill"
    # 未知の値は "save" に丸める(不正な出自ラベルを残さない)
    assert provenance.build_provenance("user", "x", "HACKED", None)["derived_via"] == "save"


def test_parents_coerced_to_int():
    p = provenance.build_provenance("user", "x", "save", ["3", 4])
    assert p["parents"] == [3, 4]


def test_recorded_at_override_for_backfill():
    p = provenance.build_provenance("user", "x", "backfill", None, recorded_at="2020-01-01T00:00:00+00:00")
    assert p["recorded_at"] == "2020-01-01T00:00:00+00:00"


# ---- _lineage_node (pure extraction) ----

def test_lineage_node_uses_provenance_parents():
    rec = {"id": 9, "title": "t", "source": "distill",
           "metadata": {"provenance": {"derived_via": "distill", "principal": "p",
                                       "source": "distill", "parents": [5, 6],
                                       "recorded_at": "ts"}}}
    node = server._lineage_node(rec)
    assert node["parents"] == [5, 6]
    assert node["derived_via"] == "distill"
    assert node["principal"] == "p"


def test_lineage_node_legacy_distilled_from_fallback():
    """provenance 無し(旧記憶)は distilled_from を親として使う。"""
    rec = {"id": 9, "title": "t", "source": "distill", "distilled_from": 4, "metadata": {}}
    node = server._lineage_node(rec)
    assert node["parents"] == [4]
    assert node["source"] == "distill"


# ---- erinys_lineage (walk) ----

def test_lineage_walk_chains_ancestors(monkeypatch):
    db = {
        3: {"id": 3, "title": "child", "metadata": {"provenance": {"derived_via": "distill", "parents": [2]}}},
        2: {"id": 2, "title": "mid", "metadata": {"provenance": {"derived_via": "distill", "parents": [1]}}},
        1: {"id": 1, "title": "root", "metadata": {"provenance": {"derived_via": "save", "parents": []}}},
    }

    def fake_fetch(oid):
        if oid not in db:
            raise LookupError(f"not found: {oid}")
        return db[oid]

    monkeypatch.setattr(server, "_fetch_observation", fake_fetch)
    res = server.erinys_lineage(id=3)
    assert res["ok"] is True
    chain = res["data"]["lineage"]
    ids = [n["id"] for n in chain]
    assert ids == [3, 2, 1]
    assert chain[0]["depth"] == 0 and chain[2]["depth"] == 2


def test_lineage_walk_handles_missing_parent(monkeypatch):
    db = {3: {"id": 3, "title": "c", "metadata": {"provenance": {"parents": [99]}}}}
    monkeypatch.setattr(server, "_fetch_observation", lambda oid: db[oid] if oid in db else (_ for _ in ()).throw(LookupError()))
    res = server.erinys_lineage(id=3)
    chain = res["data"]["lineage"]
    assert any(n.get("missing") for n in chain), "存在しない親は missing として記録"


def test_lineage_walk_no_infinite_loop_on_cycle(monkeypatch):
    db = {
        1: {"id": 1, "title": "a", "metadata": {"provenance": {"parents": [2]}}},
        2: {"id": 2, "title": "b", "metadata": {"provenance": {"parents": [1]}}},  # cycle
    }
    monkeypatch.setattr(server, "_fetch_observation", lambda oid: db[oid])
    res = server.erinys_lineage(id=1)
    ids = sorted(n["id"] for n in res["data"]["lineage"])
    assert ids == [1, 2], "循環でも seen で1回ずつに収束"


# ---- _observation_payload stamping (save path wiring, no DB/vec0) ----

class _FakeEmbed:
    model_name = "fake-model"


def test_observation_payload_stamps_provenance(monkeypatch):
    """save パスの _observation_payload が metadata.provenance を必ず付与する。"""
    monkeypatch.setattr(server, "_embedding", lambda: _FakeEmbed())
    monkeypatch.setenv("ERINYS_PRINCIPAL", "claude-opus-4-8")
    payload = server._observation_payload(
        "t", "c", "manual", "proj", "project", None, None, {"k": "v"},
        principal="codex", derived_via="save",
    )
    prov = payload["metadata"]["provenance"]
    assert prov["principal"] == "codex"        # 明示引数優先
    assert prov["derived_via"] == "save"
    assert payload["metadata"]["k"] == "v"      # 既存 metadata は保持


def test_observation_payload_overwrites_spoofed_provenance(monkeypatch):
    """呼び出し側が偽装した provenance はサーバが上書きする(出自はサーバ管理)。"""
    monkeypatch.setattr(server, "_embedding", lambda: _FakeEmbed())
    payload = server._observation_payload(
        "t", "c", "manual", None, "project", None, None,
        {"provenance": {"principal": "ATTACKER", "derived_via": "HACK"}},
        principal="real",
    )
    prov = payload["metadata"]["provenance"]
    assert prov["principal"] == "real"
    assert prov["derived_via"] == "save"
    assert prov["schema"] == provenance.PROVENANCE_SCHEMA


def test_observation_payload_none_metadata(monkeypatch):
    """metadata=None でも provenance が付く。"""
    monkeypatch.setattr(server, "_embedding", lambda: _FakeEmbed())
    payload = server._observation_payload(
        "t", "c", "manual", None, "project", None, None, None,
    )
    assert "provenance" in payload["metadata"]


# ---- erinys_update provenance preservation (Codex P2) ----

def test_update_preserves_provenance(monkeypatch):
    """metadata 全置換でもサーバ管理の provenance を消させない。"""
    current = {"id": 7, "title": "t", "metadata": {"provenance": {"schema": 1, "principal": "orig"}}}
    captured = {}
    monkeypatch.setattr(server, "_fetch_observation", lambda oid: current)
    monkeypatch.setattr(server, "update_observation", lambda db, oid, fields: captured.update(fields))
    monkeypatch.setattr(server, "_db", lambda: None)
    monkeypatch.setattr(server, "_audit", lambda *a, **k: None)
    res = server.erinys_update(id=7, metadata={"new": "data"})
    assert res["ok"] is True
    assert captured["metadata"]["new"] == "data"
    assert captured["metadata"]["provenance"]["principal"] == "orig", "provenance 引き継ぎ"


# ---- erinys_lineage edge cases (Codex P3) ----

def test_lineage_missing_root_is_not_found(monkeypatch):
    def boom(oid):
        raise LookupError("nope")
    monkeypatch.setattr(server, "_fetch_observation", boom)
    res = server.erinys_lineage(id=999)
    assert res["ok"] is False
    assert res["error"]["code"] == "NOT_FOUND"


def test_lineage_caps_max_depth(monkeypatch):
    # 自分を親に持つ連鎖でも cap で停止(seen でも止まるが cap の独立確認)
    monkeypatch.setattr(server, "_fetch_observation",
                        lambda oid: {"id": oid, "title": "x",
                                     "metadata": {"provenance": {"parents": [oid + 1]}}})
    res = server.erinys_lineage(id=1, max_depth=1000)
    assert res["ok"] is True
    assert res["data"]["depth_reached"] <= 100, "100 で上限"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
