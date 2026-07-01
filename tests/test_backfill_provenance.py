"""#151 VMG backfill_provenance の回帰テスト（plain sqlite・vec0不要）。"""
from __future__ import annotations

import importlib.util
import json
from erinys_memory._sqlite import sqlite3
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parent.parent / "scripts" / "backfill_provenance.py"
_spec = importlib.util.spec_from_file_location("backfill_provenance", _SCRIPT)
backfill = importlib.util.module_from_spec(_spec)
sys.modules["backfill_provenance"] = backfill
_spec.loader.exec_module(backfill)


def _row(metadata=None, source="user", distilled_from=None, created_at="2020-01-01 00:00:00"):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE t(id INTEGER, metadata TEXT, source TEXT, distilled_from INTEGER, created_at TEXT)")
    conn.execute("INSERT INTO t VALUES (1,?,?,?,?)", [metadata, source, distilled_from, created_at])
    return conn.execute("SELECT * FROM t").fetchone()


def test_needs_backfill_detects_missing():
    assert backfill._needs_backfill(None) == {}
    assert backfill._needs_backfill('{"k":"v"}') == {"k": "v"}


def test_needs_backfill_skips_existing():
    assert backfill._needs_backfill('{"provenance":{"schema":1}}') is None


def test_needs_backfill_handles_corrupt_metadata():
    assert backfill._needs_backfill("{not json") == {}


def test_derive_save_provenance_preserves_created_at():
    row = _row(source="user", created_at="2019-05-05 10:00:00")
    meta = backfill._derive_provenance(row, {})
    p = meta["provenance"]
    assert p["derived_via"] == "save"
    assert p["parents"] == []
    assert p["principal"] == "unknown(backfill)"
    assert p["recorded_at"] == "2019-05-05 10:00:00"   # 元 created_at を保持


def test_derive_distill_provenance_links_parent():
    row = _row(source="distill", distilled_from=42)
    p = backfill._derive_provenance(row, {})["provenance"]
    assert p["derived_via"] == "distill"
    assert p["parents"] == [42]


def test_derive_preserves_existing_metadata_keys():
    row = _row()
    meta = backfill._derive_provenance(row, {"existing": "kept"})
    assert meta["existing"] == "kept"
    assert "provenance" in meta


def test_backfill_idempotent_end_to_end(tmp_path, monkeypatch, capsys):
    """--apply→再--apply で2回目は対象0（冪等）。"""
    db = tmp_path / "m.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE observations(id INTEGER PRIMARY KEY, metadata TEXT, source TEXT, "
                 "distilled_from INTEGER, created_at TEXT)")
    conn.execute("INSERT INTO observations VALUES (1, NULL, 'user', NULL, '2020-01-01 00:00:00')")
    conn.execute("INSERT INTO observations VALUES (2, NULL, 'distill', 1, '2020-01-02 00:00:00')")
    conn.commit()
    conn.close()

    monkeypatch.setattr(sys, "argv", ["backfill", "--db", str(db), "--apply"])
    assert backfill.main() == 0
    # 検証: 両行に provenance、子は distill+parent
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    metas = {r["id"]: json.loads(r["metadata"]) for r in conn.execute("SELECT id, metadata FROM observations")}
    assert metas[1]["provenance"]["derived_via"] == "save"
    assert metas[2]["provenance"]["derived_via"] == "distill"
    assert metas[2]["provenance"]["parents"] == [1]
    conn.close()

    # 2回目: 冪等（対象0）。前回バックアップは退避してから再実行。
    Path(str(db) + ".prebackfill.bak").unlink()
    capsys.readouterr()
    assert backfill.main() == 0
    out = capsys.readouterr().out
    assert "対象なし" in out or "対象: 0" in out


def test_backfill_refuses_to_clobber_existing_backup(tmp_path, monkeypatch):
    """既存バックアップ(過去の復元点)を上書きしない。"""
    db = tmp_path / "m.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE observations(id INTEGER PRIMARY KEY, metadata TEXT, source TEXT, "
                 "distilled_from INTEGER, created_at TEXT)")
    conn.execute("INSERT INTO observations VALUES (1, NULL, 'user', NULL, '2020-01-01 00:00:00')")
    conn.commit()
    conn.close()
    (tmp_path / "m.db.prebackfill.bak").write_text("OLD RECOVERY POINT")
    monkeypatch.setattr(sys, "argv", ["backfill", "--db", str(db), "--apply"])
    assert backfill.main() == 1, "既存バックアップがあれば中止"
    assert (tmp_path / "m.db.prebackfill.bak").read_text() == "OLD RECOVERY POINT"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
