"""Tests for SSGM pre-consolidation conflict flagging (A1, 2026-06-22).

erinys_save が保護corefact(decision/anti_pattern)の新規保存時のみ conflict_check を走らせ、
矛盾を metadata['conflicts_with'] に可逆フラグすることを保証する。
配線漏れ防止: 多数派type(discovery等)では conflict_check が一切呼ばれない(ノーコスト)。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import erinys_memory.server as srv


def test_non_protected_type_skips_check(monkeypatch):
    calls = []
    monkeypatch.setattr(srv, "conflict_check", lambda *a, **k: calls.append(1) or [])
    out = srv._flag_consolidation_conflicts({"id": 1, "metadata": {}}, "created", "discovery")
    assert out == []
    assert calls == []  # 多数派saveはノーコスト


def test_updated_status_skips_check(monkeypatch):
    calls = []
    monkeypatch.setattr(srv, "conflict_check", lambda *a, **k: calls.append(1) or [])
    out = srv._flag_consolidation_conflicts({"id": 1}, "updated", "decision")
    assert out == []
    assert calls == []


def test_protected_conflict_flags_metadata(monkeypatch):
    monkeypatch.setattr(srv, "conflict_check",
                        lambda db, oid, limit=3: [{"observation": {"id": 99}, "similarity": 0.9}])
    monkeypatch.setattr(srv, "_db", lambda: None)
    monkeypatch.setattr(srv, "_audit", lambda *a, **k: None)
    updates: dict = {}
    monkeypatch.setattr(srv, "update_observation",
                        lambda db, oid, fields: updates.setdefault(oid, fields))
    record = {"id": 7, "metadata": {"x": 1}}
    out = srv._flag_consolidation_conflicts(record, "created", "decision")
    assert out == [99]
    assert updates[7]["metadata"]["conflicts_with"] == [99]
    assert updates[7]["metadata"]["x"] == 1  # 既存metadataを保持
    # P3: 返却される record にも反映され、レスポンスが stale にならない
    assert record["metadata"]["conflicts_with"] == [99]


def test_flagging_failure_never_breaks_save(monkeypatch):
    """Codex P2a: update_observation が失敗してもsaveを壊さない(best-effort)。"""
    monkeypatch.setattr(srv, "conflict_check",
                        lambda *a, **k: [{"observation": {"id": 99}, "similarity": 0.9}])
    monkeypatch.setattr(srv, "_db", lambda: None)
    monkeypatch.setattr(srv, "_audit", lambda *a, **k: None)

    def boom(*a, **k):
        raise RuntimeError("db write failed")
    monkeypatch.setattr(srv, "update_observation", boom)
    out = srv._flag_consolidation_conflicts({"id": 7, "metadata": {}}, "created", "decision")
    assert out == []  # 例外を飲み込み、saveは継続


def test_wired_into_batch_save():
    """Codex P2b: erinys_batch_save も同じフラグ関数を通す(部分配線にしない)。"""
    src = Path(srv.__file__).read_text()
    assert "_flag_consolidation_conflicts(record, status," in src
    # batch_save のループ内に配線されていること
    assert src.count("_flag_consolidation_conflicts(record, status") >= 2


def test_protected_no_conflict_returns_empty(monkeypatch):
    monkeypatch.setattr(srv, "conflict_check", lambda *a, **k: [])
    monkeypatch.setattr(srv, "_db", lambda: None)
    out = srv._flag_consolidation_conflicts({"id": 7, "metadata": {}}, "created", "decision")
    assert out == []


def test_conflict_check_failure_does_not_crash(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("vec down")
    monkeypatch.setattr(srv, "conflict_check", boom)
    monkeypatch.setattr(srv, "_db", lambda: None)
    out = srv._flag_consolidation_conflicts({"id": 7}, "created", "decision")
    assert out == []  # 失敗してもsaveを壊さない


def test_wired_into_erinys_save():
    """配線漏れ防止: erinys_save が実際にフラグ関数を呼ぶ。"""
    src = Path(srv.__file__).read_text()
    assert "_flag_consolidation_conflicts(record, status, type)" in src
