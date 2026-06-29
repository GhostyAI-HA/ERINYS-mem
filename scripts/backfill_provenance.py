#!/usr/bin/env python3
"""#151 VMG: 既存 observation に provenance を後付けする一回限りの移行スクリプト。

実行例（memory/ ディレクトリから）:
  PYTHONPATH=src python3 scripts/backfill_provenance.py            # dry-run(件数のみ)
  PYTHONPATH=src python3 scripts/backfill_provenance.py --apply    # 実行(事前バックアップ)

- 既存カラム(source / distilled_from / created_at)から provenance を導出する。
  principal は遡及不能なので "unknown(backfill)"、derived_via は distilled_from が
  あれば "distill"、なければ "save"。recorded_at は元の created_at を保持。
- metadata に既に provenance がある行はスキップ（冪等）。
- metadata のみ更新するため vec0 は不要。title/content 不変。FTS トリガ発火有無は
  スキーマ世代で異なる(WHEN guard 付き trigger なら不発、schema.sql の obs_au は発火)。
  どちらでも FTS は整合を保つが、発火する環境では 39k 行ぶん FTS/WAL が churn する。
- --apply 時のみ書き込み。直前に SQLite backup API で WAL 込みの一貫スナップショットを
  取る(shutil.copy では -wal を取りこぼすため不可)。対象の走査は BEGIN IMMEDIATE 下で
  行い、走査〜更新間の競合書き込み(TOCTOU)を防ぐ。
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from erinys_memory.provenance import build_provenance  # noqa: E402

DEFAULT_DB = os.environ.get("ERINYS_DB_PATH", "~/.erinys/memory.db")
BATCH = 500


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=DEFAULT_DB, help="ERINYS DB パス")
    p.add_argument("--apply", action="store_true", help="実際に書き込む(既定は dry-run)")
    return p.parse_args()


def _needs_backfill(metadata: str | None) -> dict | None:
    """provenance を持たない行は (既存metadata dict) を、持つ行は None を返す。"""
    try:
        d = json.loads(metadata) if metadata else {}
    except (json.JSONDecodeError, TypeError):
        d = {}
    if not isinstance(d, dict):
        d = {}
    if "provenance" in d:
        return None
    return d


def _derive_provenance(row: sqlite3.Row, base: dict) -> dict:
    distilled_from = row["distilled_from"]
    via = "distill" if distilled_from is not None else "save"
    parents = [int(distilled_from)] if distilled_from is not None else []
    base = dict(base)
    base["provenance"] = build_provenance(
        source=row["source"] or "user",
        principal="unknown(backfill)",
        derived_via=via,
        parents=parents,
        recorded_at=row["created_at"],
    )
    return base


def _scan(conn: sqlite3.Connection) -> tuple[list[tuple[int, str]], int]:
    """provenance 欠落行の (id, 新metadata JSON) と全行数を返す。"""
    rows = conn.execute(
        "SELECT id, metadata, source, distilled_from, created_at FROM observations"
    ).fetchall()
    targets: list[tuple[int, str]] = []
    for row in rows:
        base = _needs_backfill(row["metadata"])
        if base is None:
            continue
        new_meta = _derive_provenance(row, base)
        targets.append((int(row["id"]), json.dumps(new_meta, ensure_ascii=False)))
    return targets, len(rows)


def _wal_safe_backup(conn: sqlite3.Connection, backup: str) -> None:
    """SQLite backup API で WAL 込みの一貫スナップショットを取る。"""
    dest = sqlite3.connect(backup)
    try:
        conn.backup(dest)
    finally:
        dest.close()


def _apply(conn: sqlite3.Connection, db_path: str) -> int:
    backup = db_path + ".prebackfill.bak"
    if Path(backup).exists():
        # 既存バックアップ(過去の復元点)を上書きしない。手動で退避してから再実行。
        print(f"❌ バックアップが既に存在: {backup}。退避/削除してから再実行", file=sys.stderr)
        return 1
    _wal_safe_backup(conn, backup)
    print(f"バックアップ作成(WAL込み): {backup}")
    try:
        conn.execute("BEGIN IMMEDIATE")  # ロック下で走査→更新(TOCTOU回避)
        targets, total = _scan(conn)
        print(f"対象: {len(targets)} / 全 {total} 行")
        if not targets:
            conn.rollback()
            print("backfill 対象なし（全行 provenance 済み）")
            return 0
        for i in range(0, len(targets), BATCH):
            conn.executemany(
                "UPDATE observations SET metadata = ? WHERE id = ?",
                [(meta, oid) for oid, meta in targets[i:i + BATCH]],
            )
            print(f"  updated {min(i + BATCH, len(targets))}/{len(targets)}", flush=True)
        conn.commit()
    except Exception:
        conn.rollback()
        print("失敗。ロールバック済み（バックアップは保持）", file=sys.stderr)
        raise
    print(f"✅ backfill 完了: {len(targets)} 行に provenance 付与")
    return 0


def main() -> int:
    args = parse_args()
    db_path = os.path.expanduser(args.db)
    if not Path(db_path).exists():
        print(f"DB が見つからない: {db_path}", file=sys.stderr)
        return 1
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    if not args.apply:
        targets, total = _scan(conn)
        print(f"対象: {len(targets)} / 全 {total} 行（provenance 欠落分）")
        print("dry-run（--apply で書き込み）")
        if targets:
            print(f"  例 id={targets[0][0]}: {json.loads(targets[0][1])['provenance']}")
        return 0
    return _apply(conn, db_path)


if __name__ == "__main__":
    sys.exit(main())
