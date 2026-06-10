#!/usr/bin/env python3
"""embedding モデル移行: 全 observations を新モデルで再埋め込みする。

実行例（memory/ ディレクトリから）:
  PYTHONPATH=src python3 scripts/migrate_embedding_model.py \
      --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --dim 384

- src/sitecustomize.py が sqlite3 を pysqlite3 に差し替えるため、
  pyenv python でも vec0 拡張をロードできる（PYTHONPATH=src 必須）。
- 既定はフル再構築（vec_observations DROP→CREATE→全件INSERT）。
- --stale-only: embedding_model が移行先と異なる行だけ再埋め込み（冪等な追い込み用。
  移行後に旧モデルのままのプロセスが書いた行の修復に使う）。
- ロック時間最小化のため、埋め込み計算はロック外で行い、書き込みのみ短時間ロック。
"""

from __future__ import annotations

import argparse
import sys
import time

from erinys_memory.db import get_db
from erinys_memory.embedding import EmbeddingEngine, serialize_f32

BATCH = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="移行先モデル名 (fastembed対応)")
    parser.add_argument("--dim", type=int, required=True, help="移行先モデルの次元数")
    parser.add_argument("--stale-only", action="store_true",
                        help="embedding_model が移行先以外の行のみ再埋め込み")
    parser.add_argument("--dry-run", action="store_true", help="対象件数の表示のみ")
    return parser.parse_args()


def fetch_targets(db, model: str, stale_only: bool) -> list[tuple[int, str]]:
    sql = "SELECT id, content FROM observations"
    if stale_only:
        # vec行欠損も対象に含める（移行中に並行プロセスが挿入した行の回収）
        sql += (" WHERE embedding_model IS NULL OR embedding_model != ?"
                " OR id NOT IN (SELECT rowid FROM vec_observations)")
        rows = db.execute(sql, [model]).fetchall()
    else:
        rows = db.execute(sql).fetchall()
    return [(int(r["id"]), str(r["content"])) for r in rows]


def embed_all(engine: EmbeddingEngine, targets: list[tuple[int, str]]) -> list[tuple[int, bytes]]:
    blobs: list[tuple[int, bytes]] = []
    start = time.perf_counter()
    for i in range(0, len(targets), BATCH):
        chunk = targets[i:i + BATCH]
        vectors = engine.embed_batch([content for _, content in chunk])
        blobs.extend((obs_id, serialize_f32(vec)) for (obs_id, _), vec in zip(chunk, vectors))
        if (i // BATCH) % 20 == 0:
            print(f"  embedded {min(i + BATCH, len(targets))}/{len(targets)} "
                  f"({time.perf_counter() - start:.1f}s)", flush=True)
    return blobs


def _update_metadata(db, model: str, dim: int) -> None:
    db.execute(
        "UPDATE db_metadata SET embedding_model = ?, embedding_dim = ?, "
        "updated_at = datetime('now') WHERE id = 1",
        [model, dim],
    )


def write_full(db, blobs: list[tuple[int, bytes]], model: str, dim: int) -> None:
    """vec_observations を作り直して全件投入。db_metadata も更新する。"""
    db.execute("BEGIN IMMEDIATE")
    db.execute("DROP TABLE IF EXISTS vec_observations")
    db.execute(f"CREATE VIRTUAL TABLE vec_observations USING vec0(embedding float[{dim}])")
    db.executemany("INSERT INTO vec_observations(rowid, embedding) VALUES (?, ?)", blobs)
    # 再埋め込みした行だけ更新する（移行中に並行挿入された行を偽装しない）
    db.executemany(
        "UPDATE observations SET embedding_model = ? WHERE id = ?",
        [(model, obs_id) for obs_id, _ in blobs],
    )
    _update_metadata(db, model, dim)
    db.commit()


def write_stale(db, blobs: list[tuple[int, bytes]], model: str) -> None:
    """対象行のみ DELETE→INSERT で差し替える。"""
    db.execute("BEGIN IMMEDIATE")
    for obs_id, blob in blobs:
        db.execute("DELETE FROM vec_observations WHERE rowid = ?", [obs_id])
        db.execute("INSERT INTO vec_observations(rowid, embedding) VALUES (?, ?)", [obs_id, blob])
        db.execute("UPDATE observations SET embedding_model = ? WHERE id = ?", [model, obs_id])
    db.commit()


def verify(db, model: str) -> bool:
    obs = db.execute("SELECT COUNT(*) AS c FROM observations").fetchone()["c"]
    vec = db.execute("SELECT COUNT(*) AS c FROM vec_observations").fetchone()["c"]
    stale = db.execute(
        "SELECT COUNT(*) AS c FROM observations WHERE embedding_model IS NULL "
        "OR embedding_model != ?", [model]).fetchone()["c"]
    meta = db.execute("SELECT embedding_model, embedding_dim FROM db_metadata WHERE id=1").fetchone()
    print(f"verify: obs={obs} vec={vec} stale={stale} "
          f"db_metadata=({meta['embedding_model']}, {meta['embedding_dim']})")
    return obs == vec and stale == 0 and meta["embedding_model"] == model


def migrate(db, args: argparse.Namespace, targets: list[tuple[int, str]]) -> int:
    engine = EmbeddingEngine(model_name=args.model)
    blobs = embed_all(engine, targets)
    if args.stale_only:
        write_stale(db, blobs, args.model)
    else:
        write_full(db, blobs, args.model, args.dim)
    ok = verify(db, args.model)
    print("migration:", "OK" if ok else "FAILED")
    return 0 if ok else 1


def main() -> int:
    args = parse_args()
    db = get_db()
    targets = fetch_targets(db, args.model, args.stale_only)
    print(f"targets: {len(targets)} observations (stale_only={args.stale_only})")
    if args.dry_run or not targets:
        return 0
    return migrate(db, args, targets)


if __name__ == "__main__":
    sys.exit(main())
