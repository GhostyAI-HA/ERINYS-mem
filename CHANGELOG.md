# ERINYS Memory — Changelog

## [Unreleased]

### Added — JSON CLI（CLI-first / MCP-thin-adapter 移行 第一フェーズ + hardening）

- `erinys_memory/cli.py`: 自動化向け JSON CLI（`save` / `summary` / `get` / `search` / `recall` / `context` / `stats` / `undistilled` / `distill` / `dream` / `prune` / `health`）。プロジェクト側ラッパーは `.agent/scripts/erinys_cli.py`
- `--readonly`（search / recall / context）: SQLite `mode=ro` 直読み。server import・migration・audit log 書き込みなし・venv 不要。search は FTS5 キーワード（LIKE フォールバック）
- `undistilled --limit N`: 最古の未蒸留 observation ID リストを返す（/morning Phase 2.5 用）
- `health`: vector 検証不可/劣化時は `status: degraded`・`ok: false` を返す。`--deep` で server import + search smoke test まで実施（正式判定）
- usage error も JSON 契約に統一（`error.code: "USAGE"`、exit 2）
- `prune --execute` は `--confirm-global` 必須化（dream / prune は全プロジェクト横断の GLOBAL operation であることを明示）
- ラッパーの venv 探索を `.venv` / `.erinys-venv`（erinys root・project root）の多段フォールバックに拡張

## [0.3.0] — 2026-06-10

### Changed — 多言語埋め込みモデル移行

- 埋め込みモデルを `BAAI/bge-small-en-v1.5`（英語専用）から `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`（多言語・384次元・0.22GB）へ移行
  - 動機: 日本語クエリの直接意味検索が英語専用モデルでは品質不合格（実測 MRR 0.402）。多言語MiniLMは日→英クロスリンガル検索で MRR 1.0（5問全問rank 1、ベンチ: `outputs/2026-06-10_erinys_multilingual_embedding/`）
  - 次元数384は旧モデルと同一のため vec_observations のスキーマ次元変更なし
  - bge-m3 / multilingual-e5-small は fastembed 0.8.0 非対応のため候補外。multilingual-e5-large (1024d/2.24GB) と mpnet-base-v2 (768d/1.0GB) は同等品質だがサイズ・ロード時間で不採用
- `embedding.py` / `config.py` のデフォルトモデル名を更新。`.mcp.json` / `ha_setup.py` / `erinys_cli.py` / `batch_distill.py` の参照も同期
- `db_metadata` (embedding_model/embedding_dim) を更新済み。全9,744観測を新モデルで再埋め込み済み（embedding計算 約85s）

### Added

- `scripts/migrate_embedding_model.py`: 再埋め込みマイグレーションスクリプト
  - フル再構築（vec_observations DROP→CREATE→全件INSERT、ロック時間最小化のため埋め込み計算はロック外）
  - `--stale-only`: embedding_model が移行先以外の行 + vec欠損行のみ回収（冪等。移行中に並行プロセスが挿入した行や、旧モデルを保持したままの稼働中サーバーが保存した行の修復用）
  - `--dry-run`: 対象件数のみ表示

### Migration Note

- 移行前から稼働中の MCP サーバープロセスは旧モデルをメモリに保持。セッション再起動後に `--stale-only` を一度実行して回収すること
- 移行前バックアップ: `~/.erinys/memory.db.bak-20260610-pre-multilingual`（sqlite backup API・WAL反映済み）

## [0.2.1] — 2026-06-10

### Performance

**Collider**
- `find_collision_candidates` を numpy ベクトル化: O(N²) 純Python cosine + ペア毎の `get_collision` DBクエリを、正規化行列の行ベクトル積 + 既存ペア1クエリ一括ロードに置換
  - 実測: N=400 で 4.02s → 2.5ms（1591倍、結果完全一致）。本番DB (N=9,736) で 16.9s（旧実装は推定数時間 → `erinys_dream` が実用化）
- `_fetch_observations_with_embeddings` は `np.frombuffer` で正規化済み行列を返す形に変更（モジュール内専用API）

**Distill**
- `distill_observation`: source embedding を per-level 再計算（3回）から1回に削減。さらに vec_observations 保存済み embedding を再利用（`_source_embedding`）し、auto-distill 付き保存1回あたりの embedding 計算を 7回 → 4回に削減
- `_create_distillation_record` に `source_embedding` パラメータ追加（None なら従来挙動）

**DB**
- `get_db` に `PRAGMA journal_mode=WAL` / `synchronous=NORMAL` / `cache_size=-16000` を追加（WAL下のcommit fsync削減 + 16MB page cache）

**Temporal**
- `conflict_check` の N+1 クエリ解消: 候補 observation + embedding を `_fetch_candidates_batch` で1クエリ取得（旧: 候補毎に2クエリ）

### Tests
- `test_collider_candidates.py` 追加: band/コンテキスト/既存ペア除外/空DB/ゼロベクトルの回帰テスト4件
- Total: 66 tests (all passing)

## [0.2.0] — 2026-06-04

### Added — MAGMA Improvements (Growth Radar #110, #105, #131, #128, #108)

**Search Pipeline**
- Adaptive retrieval: query complexity classification (L1/L2/L3) with automatic FTS/vec weight adjustment
- CJK/日本語クエリのL2デフォルト化（FTS5 porter tokenizerのCJK弱点を回避）
- CJK+ASCII混在クエリに+2 complexity bonus
- Intent-aware query router (WHY/WHO/WHEN/WHAT/GENERAL) with per-intent boost overrides
- Graph knowledge reranking: graph-reachable nodes get 1.15x score boost after RRF fusion
- FTS5 NEAR句生成による名詞句展開
- Vec search graceful degradation: sqlite-vec未ロード環境でもクラッシュしない

**Distillation**
- Level-aware quality scoring (concrete: keyword重視, meta: semantic重視)
- Compression ratio scoring with ideal ratios per level (concrete=0.7, abstract=0.4, meta=0.2)
- Edge creation の graceful degradation (try/except)

**Collider**
- Dream cycle outcome scoring: novelty/relevance/serendipity_score 自動計算
- Score値の[0,1] clamping
- Collision score の metadata JSON 永続化
- `get_collision` で collision_score をデコードして返却

**Schema & Migration**
- Schema v2.0: edges CHECK制約に `causal`, `entity`, `temporal` を追加
- Collisions テーブルに `metadata` カラム追加 (JSON CHECK制約付き)
- 初期 `schema_version` を 2 に設定
- `_migrate()` 関数: v1→v2 自動マイグレーション (edges 再作成 + collisions ALTER TABLE)

**Graph**
- `graph_search` 関数: intent→edge type mapping + BFS展開

### Added — Tests
- `test_migration.py`: v1→v2 マイグレーション + collision metadata round-trip (8 tests)
- `test_search_regression.py`: rrf_hybrid_search パイプライン回帰テスト (12 tests)
- Total: 55 tests (all passing)

### Changed
- `rrf_hybrid_search` のデフォルト weights がクエリ複雑度に応じて自動調整されるようになった
  - 明示的に weights を指定した場合は従来通り尊重される
- Intent overrides はcallerがデフォルト値を使用している場合のみ適用

---

## [0.1.0] — 2026-04-24

### Added
- Initial release
- SQLite + FTS5 + sqlite-vec hybrid search (RRF fusion)
- Observation CRUD with embedding
- Knowledge distillation (concrete → abstract → meta)
- Dream cycle collision detection
- Graph edge management
- Session management
- Temporal decay
- 25 MCP tools via FastMCP
