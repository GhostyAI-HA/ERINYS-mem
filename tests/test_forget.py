"""#151 VMG Verified Forgetting の回帰テスト。

実 substrate 名(observations/vec_observations/observations_fts/edges/collisions)+
FTS削除トリガ + CASCADE FK を持つ test DB を組み、closure 削除と membership test を
検証する(vec0 不要 — vec_observations は通常テーブルで代用)。
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from erinys_memory import server  # noqa: E402

_SCHEMA = """
CREATE TABLE observations (
    id INTEGER PRIMARY KEY, title TEXT, content TEXT, type TEXT, project TEXT,
    distilled_from INTEGER REFERENCES observations(id),
    superseded_by INTEGER REFERENCES observations(id), metadata TEXT
);
CREATE TABLE vec_observations (rowid INTEGER PRIMARY KEY, embedding BLOB);
CREATE TABLE observations_fts (rowid INTEGER PRIMARY KEY, title TEXT, content TEXT);
CREATE TABLE edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER REFERENCES observations(id) ON DELETE CASCADE,
    target_id INTEGER REFERENCES observations(id) ON DELETE CASCADE,
    relation TEXT
);
CREATE TABLE collisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_a INTEGER REFERENCES observations(id) ON DELETE CASCADE,
    source_b INTEGER REFERENCES observations(id) ON DELETE CASCADE,
    insight TEXT
);
CREATE TABLE audit_log (id INTEGER PRIMARY KEY AUTOINCREMENT, operation TEXT,
    target_type TEXT, target_id INTEGER, detail TEXT, created_at TEXT DEFAULT (datetime('now')));
CREATE TRIGGER fts_del AFTER DELETE ON observations BEGIN
  DELETE FROM observations_fts WHERE rowid = old.id;
END;
"""


@pytest.fixture
def db(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA)
    monkeypatch.setattr(server, "_db", lambda: conn)
    monkeypatch.setattr(server, "_fetch_observation",
                        lambda oid: dict(conn.execute("SELECT * FROM observations WHERE id=?", [oid]).fetchone()
                                         or _raise_notfound(oid)))
    monkeypatch.setattr(server, "_audit", lambda *a, **k: None)
    return conn


def _raise_notfound(oid):
    raise LookupError(f"not found: {oid}")


def _add(conn, oid, distilled_from=None, superseded_by=None):
    conn.execute("INSERT INTO observations(id,title,content,type,project,distilled_from,superseded_by) "
                 "VALUES (?,?,?,?,?,?,?)", [oid, f"t{oid}", f"c{oid}", "manual", "P", distilled_from, superseded_by])
    conn.execute("INSERT INTO vec_observations(rowid,embedding) VALUES (?,?)", [oid, b"x"])
    conn.execute("INSERT INTO observations_fts(rowid,title,content) VALUES (?,?,?)", [oid, f"t{oid}", f"c{oid}"])
    conn.commit()


def test_closure_is_leaf_first(db):
    _add(db, 1)
    _add(db, 2, distilled_from=1)
    _add(db, 3, distilled_from=2)  # grandchild
    closure = server._forget_closure(1)
    assert closure[-1] == 1, "親は最後"
    assert closure.index(3) < closure.index(2) < closure.index(1), "子孫が先"


def test_closure_cycle_safe(db, monkeypatch):
    _add(db, 1)
    _add(db, 2, distilled_from=1)
    db.execute("UPDATE observations SET distilled_from=2 WHERE id=1")  # 1<->2 cycle
    db.commit()
    closure = server._forget_closure(1)
    assert sorted(closure) == [1, 2]


def test_forget_dry_run_does_not_delete(db):
    _add(db, 1)
    _add(db, 2, distilled_from=1)
    res = server.erinys_forget(id=1, dry_run=True)
    assert res["ok"] and res["data"]["dry_run"] is True
    assert sorted(res["data"]["closure"]) == [1, 2]
    assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 2  # 未削除


def test_forget_deletes_closure_and_proves_absence(db):
    _add(db, 1)
    _add(db, 2, distilled_from=1)
    _add(db, 3, distilled_from=1)
    db.execute("INSERT INTO edges(source_id,target_id,relation) VALUES (2,1,'distilled_from')")
    db.execute("INSERT INTO collisions(source_a,source_b,insight) VALUES (1,2,'x')")
    db.commit()
    res = server.erinys_forget(id=1, dry_run=False)["data"]
    assert res["complete"] is True
    assert all(v == 0 for v in res["residual"].values())
    # 全substrate 実消去
    for t in ("observations", "vec_observations", "observations_fts", "edges", "collisions"):
        assert db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] == 0, t


def test_forget_child_only_keeps_parent(db):
    _add(db, 1)
    _add(db, 2, distilled_from=1)
    res = server.erinys_forget(id=2, dry_run=False)["data"]
    assert res["closure"] == [2]
    assert db.execute("SELECT COUNT(*) FROM observations WHERE id=1").fetchone()[0] == 1


def test_forget_nulls_external_superseded_by(db):
    """closure 外の行が closure を superseded_by で指す場合 NULL 化して FK 違反を防ぐ。"""
    _add(db, 1)
    _add(db, 2)
    db.execute("UPDATE observations SET superseded_by=1 WHERE id=2")  # 2 -> superseded_by 1
    db.commit()
    res = server.erinys_forget(id=1, dry_run=False)["data"]
    assert res["complete"] is True
    assert db.execute("SELECT superseded_by FROM observations WHERE id=2").fetchone()[0] is None


def test_forget_missing_is_not_found(db):
    res = server.erinys_forget(id=999, dry_run=False)
    assert res["ok"] is False and res["error"]["code"] == "NOT_FOUND"


def test_closure_follows_provenance_parents_not_just_distilled_from(db):
    """distilled_from 列に無くても provenance.parents で親を指す子は closure に入る(Codex P2)。"""
    _add(db, 1)
    # 子: distilled_from は NULL だが provenance.parents=[1]（supersede/collide 相当）
    db.execute("INSERT INTO observations(id,title,content,type,project,metadata) "
               "VALUES (2,'t2','c2','manual','P','{\"provenance\":{\"parents\":[1]}}')")
    db.execute("INSERT INTO vec_observations(rowid,embedding) VALUES (2,?)", [b"x"])
    db.execute("INSERT INTO observations_fts(rowid,title,content) VALUES (2,'t2','c2')")
    db.commit()
    closure = server._forget_closure(1)
    assert 2 in closure, "provenance.parents 由来の子も忘却対象"


def test_closure_deep_chain_no_recursion_error(db):
    """深い distilled 連鎖でも反復DFSで RecursionError しない。"""
    _add(db, 1)
    for i in range(2, 1600):  # Python 既定再帰上限(~1000)を超える深さ
        _add(db, i, distilled_from=i - 1)
    closure = server._forget_closure(1)
    assert len(closure) == 1599
    assert closure[-1] == 1 and closure[0] == 1599  # leaf first, root last


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
