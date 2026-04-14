# ERINYS — 技術設計書 & テスト計画

> Reflexive Memory for AI Agents — 実装仕様

---

## 1. ディレクトリ構造

```
erinys-memory/
├── pyproject.toml
├── README.md
├── LICENSE                    # MIT
├── src/
│   └── erinys_memory/
│       ├── __init__.py        # version, public API
│       ├── server.py          # FastMCP server entry point
│       ├── db.py              # SQLite connection, schema init, migrations
│       ├── schema.sql         # DDL (tables, FTS5, triggers)
│       ├── search.py          # RRF hybrid search engine
│       ├── embedding.py       # fastembed wrapper (local ONNX)
│       ├── decay.py           # e^(-λt) strength decay + reinforce
│       ├── graph.py           # typed edges, traversal
│       ├── collider.py        # Memory Collider engine
│       ├── distill.py         # 3-granularity distillation
│       ├── temporal.py        # valid_from/until, supersede chain
│       ├── session.py         # session lifecycle management
│       ├── migrate.py         # engram DB → ERINYS migration
│       └── config.py          # default settings, env overrides
├── tests/
│   ├── conftest.py            # fixtures: in-memory DB, sample data
│   ├── test_db.py             # schema creation, CRUD
│   ├── test_search.py         # FTS5, vector, RRF hybrid
│   ├── test_decay.py          # strength decay, reinforcement
│   ├── test_graph.py          # edges, traversal
│   ├── test_collider.py       # memory collision
│   ├── test_distill.py        # 3-granularity distillation
│   ├── test_temporal.py       # supersede, timeline queries
│   ├── test_session.py        # session lifecycle
│   ├── test_migration.py      # engram → ERINYS migration
│   ├── test_server.py         # FastMCP tool integration
│   └── test_e2e.py            # end-to-end scenarios
└── scripts/
    └── migrate_engram.py      # standalone migration CLI
```

---

## 2. SQLite Schema（詳細）

```sql
-- ============================================================
-- ERINYS Schema v1.0
-- Single-file SQLite database for reflexive AI memory
-- ============================================================

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ────────────────────────────────────────
-- Sessions (MUST be created before observations due to FK)
-- ────────────────────────────────────────
CREATE TABLE sessions (
  id          TEXT PRIMARY KEY,
  project     TEXT,
  directory   TEXT,
  started_at  DATETIME NOT NULL DEFAULT (datetime('now')),
  ended_at    DATETIME,
  summary     TEXT,
  metadata    TEXT CHECK(metadata IS NULL OR json_valid(metadata))  -- JSON
);

-- ────────────────────────────────────────
-- Core: Observations (memory units)
-- ────────────────────────────────────────
CREATE TABLE observations (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  title         TEXT    NOT NULL,
  content       TEXT    NOT NULL,
  type          TEXT    NOT NULL DEFAULT 'manual'
                        CHECK(type IN (
                          'manual','decision','architecture','bugfix',
                          'pattern','config','discovery','learning',
                          'anti_pattern','meta_knowledge'
                        )),
  project       TEXT,
  scope         TEXT    NOT NULL DEFAULT 'project'
                        CHECK(scope IN ('project','personal','global')),

  -- Reflexive Memory classification
  is_anti_pattern      BOOLEAN NOT NULL DEFAULT 0,
  is_pattern           BOOLEAN NOT NULL DEFAULT 0,
  distillation_level   TEXT CHECK(distillation_level IN (
                          'concrete','abstract','meta'
                        )),
  distilled_from       INTEGER REFERENCES observations(id),

  -- Temporal validity (Graphiti-inspired)
  valid_from    DATETIME NOT NULL DEFAULT (datetime('now')),
  valid_until   DATETIME,            -- NULL = currently valid
  superseded_by INTEGER REFERENCES observations(id),

  -- Decay & Reinforcement (Ebbinghaus-inspired)
  base_strength REAL    NOT NULL DEFAULT 1.0,
  access_count  INTEGER NOT NULL DEFAULT 0,
  last_accessed DATETIME,

  -- Provenance tracking
  source        TEXT    NOT NULL DEFAULT 'user'
                        CHECK(source IN (
                          'user','agent','collider','distill','migration','prompt'
                        )),
  embedding_model TEXT,  -- e.g. 'BAAI/bge-small-en-v1.5'

  -- Topic key for upsert (engram-compatible)
  topic_key     TEXT,

  -- Caller-defined metadata (JSON)
  -- HA stores State Vector here: {"mode":"D2","skills":["mcp-builder"],"risk":"L"}
  -- Generic users store any key-value pairs
  metadata      TEXT CHECK(metadata IS NULL OR json_valid(metadata)),  -- JSON, queryable via json_extract()

  -- Metadata
  created_at    DATETIME NOT NULL DEFAULT (datetime('now')),
  updated_at    DATETIME NOT NULL DEFAULT (datetime('now')),  -- updated_at is set by application layer in UPDATE SET clause, not by trigger
  session_id    TEXT     REFERENCES sessions(id)
);

CREATE INDEX idx_obs_project      ON observations(project);
CREATE INDEX idx_obs_type         ON observations(type);
CREATE INDEX idx_obs_scope        ON observations(scope);
-- DEPRECATED: plain topic_key index is replaced by partial UNIQUE upsert index.
-- CREATE INDEX idx_obs_topic_key    ON observations(topic_key);
-- NULL-safe UNIQUE: SQLite treats NULL as distinct in UNIQUE indexes.
-- To enforce uniqueness even when project IS NULL, use COALESCE sentinel.
CREATE UNIQUE INDEX idx_obs_topic_key_upsert
  ON observations(COALESCE(project, '__NULL__'), scope, topic_key)
  WHERE topic_key IS NOT NULL;
CREATE INDEX idx_obs_base_strength ON observations(base_strength);
CREATE INDEX idx_obs_valid        ON observations(valid_from, valid_until);
CREATE INDEX idx_obs_anti_pattern ON observations(is_anti_pattern)
                                  WHERE is_anti_pattern = 1;

-- ────────────────────────────────────────
-- Full-Text Search (FTS5)
-- ────────────────────────────────────────
-- porter unicode61 provides basic CJK support via unicode61.
-- For Japanese-heavy workloads, vector search (sqlite-vec) is the primary recall mechanism.
-- FTS5 serves as a keyword boost signal.
CREATE VIRTUAL TABLE observations_fts USING fts5(
  title,
  content,
  project,
  type,
  content=observations,
  content_rowid=id,
  tokenize='porter unicode61'
);

-- Auto-sync FTS on INSERT/UPDATE/DELETE
CREATE TRIGGER obs_ai AFTER INSERT ON observations BEGIN
  INSERT INTO observations_fts(rowid, title, content, project, type)
  VALUES (new.id, new.title, new.content, new.project, new.type);
END;

CREATE TRIGGER obs_ad AFTER DELETE ON observations BEGIN
  INSERT INTO observations_fts(observations_fts, rowid, title, content, project, type)
  VALUES ('delete', old.id, old.title, old.content, old.project, old.type);
END;

CREATE TRIGGER obs_au AFTER UPDATE ON observations BEGIN
  INSERT INTO observations_fts(observations_fts, rowid, title, content, project, type)
  VALUES ('delete', old.id, old.title, old.content, old.project, old.type);
  INSERT INTO observations_fts(rowid, title, content, project, type)
  VALUES (new.id, new.title, new.content, new.project, new.type);
END;

-- updated_at is set by application layer in UPDATE SET clause, not by trigger.
-- DEPRECATED: recursive AFTER UPDATE trigger is intentionally not created.
-- CREATE TRIGGER obs_updated_at AFTER UPDATE ON observations BEGIN
--   UPDATE observations SET updated_at = datetime('now') WHERE id = new.id;
-- END;

-- ────────────────────────────────────────
-- Vector Search (sqlite-vec)
-- ────────────────────────────────────────
-- Embedding dimension: 384 (BAAI/bge-small-en-v1.5 via fastembed)
CREATE TABLE db_metadata (
  id              INTEGER PRIMARY KEY CHECK(id = 1),
  embedding_model TEXT    NOT NULL,
  embedding_dim   INTEGER NOT NULL CHECK(embedding_dim > 0),
  created_at      DATETIME NOT NULL DEFAULT (datetime('now')),
  updated_at      DATETIME NOT NULL DEFAULT (datetime('now'))
);

-- NOTE: vec0 virtual table does NOT support triggers.
-- INSERT/DELETE sync is handled in db.py application layer
-- (insert_observation / delete_observation functions).
-- CRITICAL: db.py MUST insert into observations first, capture observations.id,
-- then insert into vec_observations with explicit rowid = observations.id
-- inside the same single transaction.
-- Reconciliation note: db.py startup should detect and repair/report drift
-- between observations.id and vec_observations.rowid, including orphan rows.
CREATE VIRTUAL TABLE vec_observations USING vec0(
  embedding float[384]
);

-- ────────────────────────────────────────
-- Graph Edges (contextplus-inspired)
-- ────────────────────────────────────────
CREATE TABLE edges (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  source_id   INTEGER NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
  target_id   INTEGER NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
  relation    TEXT    NOT NULL
              CHECK(relation IN (
                'relates_to','depends_on','implements',
                'references','similar_to','contains',
                'contradicts','supersedes','distilled_from'
              )),
  weight      REAL    NOT NULL DEFAULT 1.0
              CHECK(weight >= 0.0 AND weight <= 1.0),
  metadata    TEXT CHECK(metadata IS NULL OR json_valid(metadata)),  -- JSON
  created_at  DATETIME NOT NULL DEFAULT (datetime('now')),

  UNIQUE(source_id, target_id, relation)
);

CREATE INDEX idx_edges_source   ON edges(source_id);
CREATE INDEX idx_edges_target   ON edges(target_id);
CREATE INDEX idx_edges_relation ON edges(relation);

-- ────────────────────────────────────────
-- Memory Collider: collision outputs
-- ────────────────────────────────────────
CREATE TABLE collisions (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  source_a    INTEGER NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
  source_b    INTEGER NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
  insight     TEXT    NOT NULL,
  confidence  REAL,
  accepted    BOOLEAN,   -- NULL=未確認, 1=採用, 0=棄却
  created_at  DATETIME NOT NULL DEFAULT (datetime('now')),

  UNIQUE(source_a, source_b)  -- 同じペアは1度だけ衝突
  -- NOTE: Application layer normalizes: source_a = min(a,b), source_b = max(a,b)
  CHECK(source_a < source_b)
);

-- ────────────────────────────────────────
-- User prompts (intent tracking)
-- ────────────────────────────────────────
CREATE TABLE prompts (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  content     TEXT    NOT NULL,
  project     TEXT,
  session_id  TEXT    REFERENCES sessions(id),
  created_at  DATETIME NOT NULL DEFAULT (datetime('now'))
);

-- ────────────────────────────────────────
-- Audit log (operation tracking)
-- ────────────────────────────────────────
CREATE TABLE audit_log (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  operation   TEXT    NOT NULL,  -- save/search/update/delete/prune/collide
  target_type TEXT,              -- observation/edge/session
  target_id   INTEGER,
  detail      TEXT CHECK(detail IS NULL OR json_valid(detail)),  -- JSON: query, result_count, etc.
  created_at  DATETIME NOT NULL DEFAULT (datetime('now'))
);

-- ────────────────────────────────────────
-- Schema version tracking
-- ────────────────────────────────────────
CREATE TABLE schema_version (
  version     INTEGER PRIMARY KEY,
  applied_at  DATETIME NOT NULL DEFAULT (datetime('now'))
);
INSERT INTO schema_version(version) VALUES (1);
```

### 2.1 db.py 実装ノート

```python
# db.py

def validate_db_metadata(db: sqlite3.Connection, config: ErinysConfig) -> None:
    """
    初回起動時は db_metadata を作成し、
    以後は config の embedding_model / embedding_dim と一致確認する。
    """
    row = db.execute(
        "SELECT embedding_model, embedding_dim FROM db_metadata WHERE id = 1"
    ).fetchone()

    if row is None:
        db.execute(
            """
            INSERT INTO db_metadata(id, embedding_model, embedding_dim, updated_at)
            VALUES (1, ?, ?, datetime('now'))
            """,
            [config.embedding_model, config.embedding_dim],
        )
        return

    if (
        row["embedding_model"] != config.embedding_model
        or row["embedding_dim"] != config.embedding_dim
    ):
        raise RuntimeError("embedding model/dim mismatch: DB metadata vs config")


def insert_observation_with_embedding(db, obs_payload, embedding_blob) -> int:
    """
    observations と vec_observations は rowid を明示一致させる。
    drift 防止のため、単一トランザクションで両方を書き込む。
    """
    db.execute("BEGIN IMMEDIATE")
    try:
        cursor = db.execute(OBS_INSERT_SQL, obs_payload)
        obs_id = cursor.lastrowid
        db.execute(
            "INSERT INTO vec_observations(rowid, embedding) VALUES (?, ?)",
            [obs_id, embedding_blob],
        )
        db.commit()
        return obs_id
    except Exception:
        db.rollback()
        raise


def delete_observation_with_embedding(db, obs_id: int) -> None:
    """
    observations と vec_observations を単一トランザクションで同時削除。
    edges/collisions は ON DELETE CASCADE で自動削除。
    """
    db.execute("BEGIN IMMEDIATE")
    try:
        db.execute("DELETE FROM vec_observations WHERE rowid = ?", [obs_id])
        db.execute("DELETE FROM observations WHERE id = ?", [obs_id])
        db.commit()
    except Exception:
        db.rollback()
        raise


def reconcile_vec_observations(db) -> dict:
    """
    Startup reconciliation: observations ↔ vec_observations の整合性検査。
    1. orphan vec rows (vec にあるが obs にない) → DELETE
    2. missing vec rows (obs にあるが vec にない) → 再embedding + INSERT
    Returns: {"orphans_removed": int, "missing_reembedded": int}
    """
    orphans = db.execute(
        "SELECT rowid FROM vec_observations WHERE rowid NOT IN (SELECT id FROM observations)"
    ).fetchall()
    for row in orphans:
        db.execute("DELETE FROM vec_observations WHERE rowid = ?", [row[0]])

    missing = db.execute(
        "SELECT id, content FROM observations WHERE id NOT IN (SELECT rowid FROM vec_observations)"
    ).fetchall()
    for row in missing:
        embedding = embedding_engine.embed(row[1])
        db.execute(
            "INSERT INTO vec_observations(rowid, embedding) VALUES (?, ?)",
            [row[0], serialize_f32(embedding)],
        )

    return {"orphans_removed": len(orphans), "missing_reembedded": len(missing)}


def update_observation(db, obs_id: int, fields: dict) -> None:
    """updated_at は UPDATE SET で明示更新する。"""
    ...
```

---

## 3. RRF Hybrid Search アルゴリズム

Reciprocal Rank Fusion: 2つの検索結果リストを統合するランキング手法。

```python
# search.py の核心ロジック

def rrf_hybrid_search(
    db: sqlite3.Connection,
    query: str,
    query_embedding: list[float],
    project: str | None = None,
    metadata_filter: dict | None = None,
    limit: int = 10,
    k: int = 60,          # RRF定数 (標準値)
    fts_weight: float = 0.4,
    vec_weight: float = 0.6,
) -> list[dict]:
    """
    FTS5 keyword search + sqlite-vec vector search → RRF fusion.

    RRF score = Σ (weight / (k + rank))

    Project filter — adaptive widening:
    - FTS5: project条件をMATCH句に含めるため、候補段階でproject絞り込み済み
    - sqlite-vec: project列を持たないため、iterative widening で対応
      初期k=limit*5 → in-project候補不足ならk*2に拡大 → 繰り返し
      終了条件: in_project >= limit OR vec候補が枯渇 (returned < requested)
      上限: VEC_MAX_K (10000) で安全停止
    """
    MAX_SEARCH_LIMIT = 500  # SQLite bind parameter safety: prevents VALUES explosion
    VEC_INITIAL_MULTIPLIER = 5
    VEC_MAX_K = 10000  # safety cap to avoid scanning entire table

    # Clamp limit to prevent bind parameter overflow in join-back
    limit = min(limit, MAX_SEARCH_LIMIT)

    # Step 1: FTS5 keyword search (project-aware via MATCH)
    fts_match = sanitize_fts(query)
    if project:
        safe_project = project.replace('"', '').strip()
        if safe_project:
            fts_match += f' AND project:"{safe_project}"'
    fts_sql = """
        SELECT rowid,
               row_number() OVER (ORDER BY rank) as rank_num
        FROM observations_fts
        WHERE observations_fts MATCH ?
        LIMIT ?
    """
    fts_rows = db.execute(
        fts_sql, [fts_match, min(limit * VEC_INITIAL_MULTIPLIER, VEC_MAX_K)]
    ).fetchall()

    # Step 2: sqlite-vec vector search (iterative widening for project coverage)
    def fetch_vec_rows(vec_limit: int) -> list[tuple]:
        vec_sql = """
            SELECT rowid,
                   row_number() OVER (ORDER BY distance) as rank_num
            FROM vec_observations
            WHERE embedding MATCH ?
              AND k = ?
        """
        return db.execute(
            vec_sql, [serialize_f32(query_embedding), vec_limit]
        ).fetchall()

    # Step 3: Iterative RRF fusion with adaptive vec widening
    def compute_rrf(fts: list, vec: list) -> dict[int, float]:
        scores: dict[int, float] = {}
        for rowid, rank_num in fts:
            scores[rowid] = scores.get(rowid, 0) + fts_weight / (k + rank_num)
        for rowid, rank_num in vec:
            scores[rowid] = scores.get(rowid, 0) + vec_weight / (k + rank_num)
        return scores

    vec_k = min(limit * VEC_INITIAL_MULTIPLIER, VEC_MAX_K)
    vec_rows = fetch_vec_rows(vec_k)
    scores = compute_rrf(fts_rows, vec_rows)

    # Step 4: Project-aware adaptive widening loop
    # No pre-truncation: ALL scored candidates participate in project check.
    if project:
        while vec_k < VEC_MAX_K:
            ranked_ids = sorted(scores, key=scores.__getitem__, reverse=True)
            in_project_count = sum(
                1 for rid in ranked_ids
                if db.execute(
                    "SELECT 1 FROM observations WHERE id = ? AND project = ?",
                    [rid, project],
                ).fetchone()
            )
            if in_project_count >= limit:
                break
            # vec exhausted: returned fewer rows than requested
            if len(vec_rows) < vec_k:
                break
            # widen: double k
            vec_k = min(vec_k * 2, VEC_MAX_K)
            vec_rows = fetch_vec_rows(vec_k)
            scores = compute_rrf(fts_rows, vec_rows)
    
    ranked_ids = sorted(scores, key=scores.__getitem__, reverse=True)

    if not ranked_ids:
        return []

    # Step 5: Join back to observations in fused rank order.
    # Use temp table to avoid SQLite bind parameter limits (max ~32766).
    db.execute("CREATE TEMP TABLE IF NOT EXISTS _rrf_ranked(id INTEGER, ord INTEGER)")
    db.execute("DELETE FROM _rrf_ranked")
    db.executemany(
        "INSERT INTO _rrf_ranked(id, ord) VALUES (?, ?)",
        [(obs_id, ord_num) for ord_num, obs_id in enumerate(ranked_ids)],
    )

    where_clauses = ["1 = 1"]
    params: list[object] = []
    if project:
        where_clauses.append("o.project = ?")
        params.append(project)

    # Step 5.5: metadata_filter is applied via parameterized json_extract()
    # predicates during join-back, never by string interpolation.
    if metadata_filter:
        for key, value in metadata_filter.items():
            where_clauses.append("json_extract(o.metadata, ?) = ?")
            params.extend([f"$.{key}", value])

    join_sql = f"""
        SELECT o.*, r.ord
        FROM _rrf_ranked r
        JOIN observations o ON o.id = r.id
        WHERE {" AND ".join(where_clauses)}
        ORDER BY r.ord
    """
    fused_rows = db.execute(join_sql, params).fetchall()

    results = []
    for obs in fused_rows:
        effective_strength = current_strength(
            obs["base_strength"],
            obs["created_at"],
            obs["last_accessed"],
            obs["access_count"],
        )
        obs["rrf_score"] = scores[obs["id"]]
        obs["effective_strength"] = effective_strength
        obs["effective_score"] = obs["rrf_score"] * effective_strength
        results.append(obs)

    results.sort(key=lambda x: x["effective_score"], reverse=True)
    return results[:limit]
```

### FTS5 クエリサニタイズ

```python
def sanitize_fts(query: str) -> str:
    """
    FTS5 MATCH構文のために入力を正規化する。
    空文字は reject し、各トークンからダブルクォートを除去する。
    """
    raw_tokens = query.strip().split()
    tokens = [token.replace('"', "").strip() for token in raw_tokens]
    tokens = [token for token in tokens if token]

    if not tokens:
        raise ValueError("FTS query must not be empty")

    return " ".join(f'"{token}"' for token in tokens)
```

---

## 4. Embedding 戦略

```python
# embedding.py

from fastembed import TextEmbedding

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # 384次元, 多言語対応
EMBEDDING_DIM = 384

class EmbeddingEngine:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self._model = TextEmbedding(model_name=model_name)

    def embed(self, text: str) -> list[float]:
        """単一テキストをembedding。"""
        results = list(self._model.embed([text]))
        return results[0].tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """バッチembedding。"""
        return [r.tolist() for r in self._model.embed(texts)]
```

> **選定理由**: fastembed は ONNX Runtime ベースでCPUのみ動作。`pip install fastembed` で完結、PyTorch/TensorFlow不要。bge-small-en-v1.5 は384次元で軽量ながらMTEBスコアが高い。日本語は bge-m3 (1024次元) へのオプション切り替えで対応。
>
> **オフラインモデル管理**: fastembed は初回実行時に HuggingFace Hub からモデルをダウンロードし `~/.cache/fastembed/` にキャッシュする。「no network access」はランタイム動作の特性であり、初回セットアップ時にはネットワーク接続が必要。エアギャップ環境では `FASTEMBED_CACHE_PATH` 環境変数で事前配置済みキャッシュディレクトリを指定する。`erinys-migrate` コマンドにモデルプリフェッチオプション (`--prefetch-model`) を用意し、デプロイ前にモデルを取得可能にする。

---

## 5. Decay & Reinforcement

```python
# decay.py

import math
from datetime import datetime, timezone

LAMBDA = 0.01  # decay rate (adjustable)
REINFORCE_BOOST = 0.15  # access時の強化量
PRUNE_THRESHOLD = 0.1   # この閾値以下で自動削除候補

def current_strength(
    base_strength: float,
    created_at: datetime,
    last_accessed: datetime | None,
    access_count: int,
    now: datetime | None = None,
) -> float:
    """
    Ebbinghaus-inspired decay with reinforcement.

    effective_strength = reinforced_base_strength × e^(-λt)
    reinforced_base_strength = base_strength + (access_count × REINFORCE_BOOST)

    effective_strength は常に current_strength() で read 時に計算する。
    erinys_reinforce は access_count と last_accessed のみ更新し、
    base_strength は直接更新しない。
    """
    now = now or datetime.now(timezone.utc)
    reference_time = last_accessed or created_at
    elapsed_days = (now - reference_time).total_seconds() / 86400

    reinforced_base = min(base_strength + access_count * REINFORCE_BOOST, 2.0)
    return reinforced_base * math.exp(-LAMBDA * elapsed_days)

def should_prune(strength: float) -> bool:
    return strength < PRUNE_THRESHOLD
```

---

## 6. Memory Collider エンジン

```python
# collider.py — 設計概要

class MemoryCollider:
    """
    異なる文脈の記憶をぶつけて新しい洞察を生成する。
    GHOSTY COLLIDER の記憶版。

    衝突条件:
    1. 意味的近接: cosine similarity > 0.65 AND < 0.90
       (近すぎると重複、遠すぎると無関係)
    2. 文脈的差異: 異なるproject OR 異なるsession
    3. 未衝突: collisions テーブルにペアが存在しない
    """

    def find_collision_candidates(
        self, db, limit: int = 20
    ) -> list[tuple[int, int, float]]:
        """衝突候補ペアを発見する。"""
        # vec_observations からcosine similarityが0.65-0.90の
        # 異なるプロジェクトのペアを検索
        # collisions テーブルで既出ペアを除外
        ...

    def collide(
        self, db, obs_a: dict, obs_b: dict
    ) -> str | None:
        """
        2つの記憶を衝突させて洞察を生成。

        LLMを使わずに実行する軽量版:
        1. 共通キーワード抽出
        2. anti_pattern + pattern の組み合わせ → ルール生成
        3. 同一エンティティへの異なる知見 → 統合

        LLM版（オプション）:
        - 2つのobservationをプロンプトに入れて
          「この2つから導ける新しい知見は？」と問う
        """
        ...

    def dream_cycle(self, db, max_collisions: int = 10):
        """
        Dream Cycle: バッチ衝突実行。
        cron or session_end 時に呼ばれる。
        """
        candidates = self.find_collision_candidates(db, limit=max_collisions)
        for obs_a_id, obs_b_id, similarity in candidates:
            obs_a = fetch_observation(db, obs_a_id)
            obs_b = fetch_observation(db, obs_b_id)
            insight = self.collide(db, obs_a, obs_b)
            if insight:
                save_collision(db, obs_a_id, obs_b_id, insight, similarity)
```

---

## 7. FastMCP Server 構成

```python
# server.py

from fastmcp import FastMCP
from .db import get_db, init_db
from .search import rrf_hybrid_search
from .embedding import EmbeddingEngine
from .session import SessionManager
from .graph import GraphEngine
from .decay import current_strength
from .collider import MemoryCollider

mcp = FastMCP("ERINYS")
embedding = EmbeddingEngine()

# NOTE: All tool returns use a unified envelope:
# {"ok": bool, "data": ..., "error": ...}

# ─── P0: Core (14 tools) ───

@mcp.tool
def erinys_save(
    title: str,
    content: str,
    type: str = "manual",
    project: str | None = None,
    scope: str = "project",
    topic_key: str | None = None,
    session_id: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """Save a structured observation to ERINYS memory."""
    ...

@mcp.tool
def erinys_get(
    id: int,
) -> dict:
    """Get a single observation by ID (full content, untruncated)."""
    ...

@mcp.tool
def erinys_update(
    id: int,
    title: str | None = None,
    content: str | None = None,
    type: str | None = None,
    project: str | None = None,
    scope: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """Update an existing observation. Only provided fields are changed."""
    ...

@mcp.tool
def erinys_delete(
    id: int,
) -> dict:
    """Delete an observation and cascade dependent rows via FK rules."""
    ...

@mcp.tool
def erinys_search(
    query: str,
    project: str | None = None,
    limit: int = 10,
    include_anti_patterns: bool = True,
    include_distilled: bool = True,
    metadata_filter: dict | None = None,
) -> dict:
    """RRF hybrid search (FTS5 keyword + vector similarity).
    metadata_filter: e.g. {"mode": "D2"} → json_extract filtering."""
    ...

@mcp.tool
def erinys_save_prompt(
    content: str,
    project: str | None = None,
    session_id: str | None = None,
) -> dict:
    """Save a user prompt to track intent and goals."""
    ...

@mcp.tool
def erinys_session_start(
    id: str,
    project: str,
    directory: str | None = None,
) -> dict:
    """Start a new session."""
    ...

@mcp.tool
def erinys_session_end(
    id: str,
    summary: str | None = None,
) -> dict:
    """End a session with optional summary."""
    ...

@mcp.tool
def erinys_session_summary(
    content: str,
    project: str,
    session_id: str | None = None,
) -> dict:
    """Save comprehensive end-of-session summary (Goal/Discoveries/Accomplished)."""
    ...

@mcp.tool
def erinys_recall(
    project: str | None = None,
    limit: int = 10,
) -> dict:
    """Recall recent observations for context."""
    ...

@mcp.tool
def erinys_context(
    project: str,
    limit: int = 20,
) -> dict:
    """Get recent session context for a project."""
    ...

@mcp.tool
def erinys_export(
    project: str | None = None,
    format: str = "markdown",
) -> dict:
    """Export observations as markdown payload (Obsidian-compatible [[wikilinks]])."""
    ...

@mcp.tool
def erinys_backup(
    path: str | None = None,
) -> dict:
    """Create a consistent SQLite backup and return backup metadata."""
    ...

@mcp.tool
def erinys_stats(
    project: str | None = None,
) -> dict:
    """Database statistics: observation count, project breakdown, health."""
    ...

# ─── P1: Graph + Decay ───

@mcp.tool
def erinys_link(
    source_id: int,
    target_id: int,
    relation: str,
    weight: float = 1.0,
) -> dict:
    """Create typed edge between observations."""
    ...

@mcp.tool
def erinys_traverse(
    start_id: int,
    max_depth: int = 2,
    relation_filter: list[str] | None = None,
) -> dict:
    """Traverse graph from a starting observation."""
    ...

@mcp.tool
def erinys_prune(
    threshold: float = 0.1,
    dry_run: bool = True,
) -> dict:
    """Prune decayed observations below threshold."""
    ...

@mcp.tool
def erinys_batch_save(
    items: list[dict],
    auto_link: bool = True,
) -> dict:
    """Bulk-add observations with automatic similarity linking."""
    ...

@mcp.tool
def erinys_reinforce(
    observation_id: int,
) -> dict:
    """Reinforce an observation by updating access_count and last_accessed."""
    ...

# ─── P2: Self-Managing + Temporal ───

@mcp.tool
def erinys_conflict_check(
    observation_id: int,
) -> dict:
    """Detect contradicting observations."""
    ...

@mcp.tool
def erinys_supersede(
    old_id: int,
    new_content: str,
    reason: str,
) -> dict:
    """Supersede an old observation with updated fact."""
    ...

@mcp.tool
def erinys_timeline(
    query: str,
    as_of: str | None = None,
) -> dict:
    """Query facts valid at a specific point in time."""
    ...

# ─── P3: Collider + Distillation ───

@mcp.tool
def erinys_collide(
    obs_a_id: int,
    obs_b_id: int,
) -> dict:
    """Manually collide two observations to generate insight."""
    ...

@mcp.tool
def erinys_dream(
    max_collisions: int = 10,
) -> dict:
    """Run Dream Cycle: batch collision of candidate memory pairs."""
    ...

@mcp.tool
def erinys_distill(
    observation_id: int,
    level: str = "abstract",
) -> dict:
    """Distill observation to higher abstraction level."""
    ...

@mcp.tool
def erinys_eval(
    project: str | None = None,
) -> dict:
    """Self-evaluate memory quality (LOCOMO-inspired metrics)."""
    ...

if __name__ == "__main__":
    mcp.run()
```

---

## 8. engram → ERINYS 移行設計

```python
# migrate.py — 設計概要

"""
engram SQLite DB → ERINYS DB migration.

engram schema (confirmed from GP source):
  - observations: id, title, content, type, project, scope,
                  topic_key, session_id, created_at, updated_at
  - sessions: id, project, started_at, ended_at, summary

ERINYS additions:
  - base_strength, access_count, last_accessed (default values)
  - valid_from (= created_at), valid_until (NULL)
  - is_anti_pattern, is_pattern (inferred from type/content)
  - distillation_level (NULL for raw observations)
  - embedding (generated during migration)

episodic-memory JSONL → ERINYS sessions:
  - session_summaries.jsonl entries → sessions table
  - decisions/patterns → observations with type inference
"""

def migrate_engram_db(engram_path: str, erinys_path: str):
    """Full migration pipeline."""
    ...

def migrate_episodic_jsonl(jsonl_path: str, erinys_db):
    """Import session_summaries.jsonl into ERINYS."""
    ...

def infer_anti_pattern(content: str) -> bool:
    """Keyword + pattern matching to detect anti-patterns."""
    keywords = ['失敗', 'error', 'bug', 'mistake', 'wrong',
                'broken', 'crashed', 'fix', '壊れ', 'ハング']
    return any(kw in content.lower() for kw in keywords)
```

---

## 9. テスト計画

### 9.1 テストフレームワーク

```
pytest + pytest-asyncio
Coverage target: 85%+
CI: GitHub Actions (Python 3.10, 3.11, 3.12)
```

### 9.2 テストケース一覧（79ケース）

#### DB & Schema (8ケース)
| # | テスト | 検証内容 |
|:--|:--|:--|
| D1 | `test_schema_creation` | 全テーブル・インデックス・トリガーが正常に作成される |
| D2 | `test_schema_version` | schema_version テーブルに v1 が記録される |
| D3 | `test_observation_crud` | INSERT/SELECT/UPDATE/DELETE が動作する |
| D4 | `test_observation_type_check` | 無効な type で CHECK制約エラー |
| D5 | `test_scope_check` | 無効な scope で CHECK制約エラー |
| D6 | `test_fts_trigger_insert` | INSERT 後に FTS5 テーブルに自動同期 |
| D7 | `test_fts_trigger_update` | UPDATE 後に FTS5 が再同期 |
| D8 | `test_fts_trigger_delete` | DELETE 後に FTS5 から削除 |

#### Search (10ケース)
| # | テスト | 検証内容 |
|:--|:--|:--|
| S1 | `test_fts_basic_search` | FTS5 キーワード検索が結果を返す |
| S2 | `test_fts_japanese` | 日本語トークンで検索が動作する |
| S3 | `test_fts_sanitize` | 特殊文字 (`(`, `)`, `*`) がエスケープされる |
| S4 | `test_vec_basic_search` | sqlite-vec でKNN検索が動作する |
| S5 | `test_vec_similarity_order` | 類似度順にソートされる |
| S6 | `test_rrf_fusion` | FTS + vec の結果がRRFで統合される |
| S7 | `test_rrf_both_lists_boost` | 両方のリストに出現する結果がブーストされる |
| S8 | `test_rrf_weight_tuning` | fts_weight/vec_weight 変更で順位が変わる |
| S9 | `test_rrf_strength_factor` | effective_strength が低い記憶のランクが下がる |
| S10 | `test_search_project_filter` | project フィルタが正しく動作する |

#### Decay & Reinforcement (8ケース)
| # | テスト | 検証内容 |
|:--|:--|:--|
| R1 | `test_initial_strength` | 新規observationのbase_strengthが1.0 |
| R2 | `test_decay_over_time` | 時間経過で effective_strength が減少する |
| R3 | `test_reinforce_on_access` | erinys_reinforce 呼び出しで access_count と last_accessed が更新される |
| R4 | `test_frequently_accessed_survives` | 高頻度アクセスの記憶は decay に耐える |
| R5 | `test_prune_threshold` | 閾値以下の記憶が prune 候補になる |
| R6 | `test_prune_dry_run` | dry_run=True で実際には削除されない |
| R7 | `test_prune_execute` | dry_run=False で実際に削除される |
| R8 | `test_strength_cap` | effective_strength の上限が 2.0 を超えない |

#### Graph (8ケース)
| # | テスト | 検証内容 |
|:--|:--|:--|
| G1 | `test_create_edge` | edge 作成が成功する |
| G2 | `test_duplicate_edge_update` | 同一ペア+relation で weight が更新される |
| G3 | `test_edge_cascade_delete` | observation 削除時に edge も削除される |
| G4 | `test_traverse_depth_1` | 1ホップの隣接ノードが返る |
| G5 | `test_traverse_depth_2` | 2ホップで間接隣接ノードも返る |
| G6 | `test_traverse_relation_filter` | relation_filter で特定の関係のみ辿る |
| G7 | `test_contradicts_edge` | `contradicts` relation が作成できる |
| G8 | `test_supersedes_edge` | `supersedes` relation が自動作成される |

#### Session (6ケース)
| # | テスト | 検証内容 |
|:--|:--|:--|
| SS1 | `test_session_start` | セッション作成、started_at が記録される |
| SS2 | `test_session_end` | ended_at と summary が記録される |
| SS3 | `test_session_recall` | 直近N件のセッション+observationsが返る |
| SS4 | `test_session_context` | プロジェクト別のコンテキストが正しく構成される |
| SS5 | `test_duplicate_session_id` | 同一ID で開始するとエラー |
| SS6 | `test_observation_session_link` | observation.session_id でセッションに紐付く |

#### Temporal (4ケース)
| # | テスト | 検証内容 |
|:--|:--|:--|
| T1 | `test_supersede` | old observation に superseded_by + valid_until が設定される |
| T2 | `test_timeline_current` | as_of=None で現在有効な事実のみ返る |
| T3 | `test_timeline_historical` | as_of=過去日 でその時点の事実が返る |
| T4 | `test_supersede_chain` | A→B→C の上書きチェーンが辿れる |

#### Collider (6ケース)
| # | テスト | 検証内容 |
|:--|:--|:--|
| C1 | `test_find_candidates` | similarity 0.65-0.90 のペアが発見される |
| C2 | `test_exclude_same_project` | 同一プロジェクト内のペアは候補外 |
| C3 | `test_exclude_already_collided` | 既出ペアは候補外 |
| C4 | `test_collide_generates_insight` | 衝突結果が collisions テーブルに保存される |
| C5 | `test_dream_cycle_batch` | バッチ衝突が指定件数まで実行される |
| C6 | `test_collision_accept_reject` | accepted フラグで採用/棄却が記録される |

#### Migration (4ケース)
| # | テスト | 検証内容 |
|:--|:--|:--|
| M1 | `test_engram_observation_migration` | engram observations が全件移行される |
| M2 | `test_engram_session_migration` | engram sessions が全件移行される |
| M3 | `test_episodic_jsonl_import` | JSONL エントリが sessions に変換される |
| M4 | `test_anti_pattern_inference` | 移行時に失敗パターンが自動分類される |

#### E2E (2ケース)
| # | テスト | 検証内容 |
|:--|:--|:--|
| E1 | `test_save_search_cycle` | save → search → 結果にsavedが含まれる |
| E2 | `test_full_lifecycle` | session_start → save × 3 → search → reinforce → session_end → recall |

#### Reliability & Safety 追加回帰 (23ケース)
| # | テスト | 検証内容 |
|:--|:--|:--|
| RS1 | `test_vec_rowid_matches_observation_id` | vec_observations.rowid と observations.id が一致する |
| RS2 | `test_vec_observation_drift_reconciliation_reports_or_repairs` | vec/obs drift を起動時reconciliationで検出・修復またはエラー化する |
| RS3 | `test_topic_key_upsert_unique_per_project_scope` | `(project, scope, topic_key)` の部分UNIQUE制約が効く |
| RS4 | `test_topic_key_null_allows_multiple_rows` | `topic_key IS NULL` は部分UNIQUE制約の対象外 |
| RS5 | `test_invalid_json_observation_metadata_rejected` | observations.metadata の不正JSONが拒否される |
| RS6 | `test_invalid_json_sessions_edges_audit_rejected` | sessions/edges/audit_log の不正JSONが拒否される |
| RS7 | `test_empty_fts_query_rejected` | 空文字または引用符だけのFTS queryで ValueError |
| RS8 | `test_sanitize_fts_strips_double_quotes` | sanitize_fts がトークン内の `\"` を除去する |
| RS9 | `test_project_filter_adaptive_widening` | project指定時にinitial k内にin-project候補が不足する場合、widening loopが発火してlimit件を返す |
| RS9b | `test_project_filter_vec_exhaustion_stops_widening` | vec候補が枯渇した場合(returned < requested)、widening loopが安全停止し、取得可能な全in-project件を返す |
| RS9c | `test_vec_max_k_caps_initial_fetch` | limit >= 2000 でも初回vec fetchがVEC_MAX_Kで制限される |
| RS9d | `test_large_limit_clamped_to_max_search_limit` | limit=5000 を渡しても MAX_SEARCH_LIMIT=500 にクランプされ、bind parameter超過なく結果を返す |
| RS10 | `test_metadata_filter_json_extract_single_key` | metadata_filter の単一キー条件が json_extract で効く |
| RS11 | `test_metadata_filter_json_extract_multi_key` | metadata_filter の複数キー条件が AND 条件で効く |
| RS12 | `test_embedding_dim_mismatch_fails_startup` | db_metadata.embedding_dim と config 不一致で起動失敗 |
| RS13 | `test_embedding_model_mismatch_fails_startup` | db_metadata.embedding_model と config 不一致で起動失敗 |
| RS14 | `test_concurrent_read_write_wal_consistency` | reader/writer 並行時も busy_timeout 内で整合性が保たれる |
| RS15 | `test_begin_immediate_serializes_writers` | `BEGIN IMMEDIATE` により write-write 競合が直列化される |
| RS16 | `test_backup_restore_roundtrip` | erinys_backup で取得したDBが restore 後に一致する |
| RS17 | `test_migration_rollback_on_failure` | migrate 途中失敗で全変更がロールバックされる |
| RS18 | `test_erinys_backup_envelope` | erinys_backup が `{ok,data,error}` 形式で返る |
| RS19 | `test_erinys_delete_cascades_and_envelope` | erinys_delete が関連行削除と統一エンベロープを満たす |
| RS20 | `test_restore_recovers_db_metadata_and_vectors` | backup/restore 後も db_metadata と vector index が整合する |

---

## 10. 依存関係（最小構成）

```toml
# pyproject.toml

[project]
name = "erinys-memory"
version = "0.1.0"
description = "Reflexive Memory for AI Agents — MCP Server"
requires-python = ">=3.10"
license = "MIT"
dependencies = [
    "fastmcp>=3.2.0",
    "sqlite-vec>=0.1.6",
    "fastembed>=0.4.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "pytest-cov>=5.0",
    "ruff>=0.5",
]

[project.scripts]
erinys = "erinys_memory.server:main"
erinys-migrate = "erinys_memory.migrate:main"
```

---

## 11. Phase別 開発ロードマップ

```
Phase 1 (Week 1-2): P0 Core
├── [ ] pyproject.toml + ディレクトリ構築
├── [ ] schema.sql → db.py (init_db)
├── [ ] embedding.py (fastembed wrapper)
├── [ ] search.py (FTS5 + sqlite-vec + RRF)
├── [ ] session.py (start/end/recall/context)
├── [ ] server.py (14 P0 tools)
├── [ ] tests: D1-D8, S1-S10, SS1-SS6
└── [ ] smoke test: MCP client → save → search

Phase 2 (Week 3): P1 Graph + Decay  
├── [ ] graph.py (edges, traversal)
├── [ ] decay.py (base_strength, reinforce, prune)
├── [ ] server.py (4 P1 tools追加)
├── [ ] tests: G1-G8, R1-R8
└── [ ] integration: search結果にeffective_strength反映

Phase 3 (Week 4): P2 Self-Managing + Temporal
├── [ ] temporal.py (supersede, timeline)
├── [ ] conflict detection (embedding similarity)
├── [ ] migrate.py (engram → ERINYS)
├── [ ] server.py (3 P2 tools追加)
├── [ ] tests: T1-T4, M1-M4
└── [ ] HA integration: AGENTS.md routing更新

Phase 4 (Week 5-6): P3 Collider + Polish
├── [ ] collider.py (candidate discovery, collision)
├── [ ] distill.py (3-granularity distillation)
├── [ ] Dream Cycle daemon
├── [ ] eval.py (LOCOMO-like metrics)
├── [ ] server.py (4 P3 tools追加)
├── [ ] tests: C1-C6, E1-E2
├── [ ] README.md (競合比較, デモ)
└── [ ] GitHub OSS release
```

---

## 12. Configuration (config.py)

```python
# config.py — 全設定項目

from dataclasses import dataclass, field
import os

@dataclass
class ErinysConfig:
    # Database
    db_path: str = os.environ.get(
        "ERINYS_DB_PATH", "~/.erinys/memory.db"
    )
    db_backup_on_init: bool = True      # 起動時に自動バックアップ
    db_max_size_mb: int = 500           # 警告閾値
    db_reader_pool_size: int = 4        # per-request read connections

    # Embedding
    embedding_model: str = os.environ.get(
        "ERINYS_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
    )
    embedding_dim: int = 384            # モデルに合わせて変更

    # Search
    rrf_k: int = 60                     # RRF定数
    fts_weight: float = 0.4
    vec_weight: float = 0.6
    default_search_limit: int = 10
    max_search_limit: int = 500         # SQLite bind parameter safety cap

    # Decay
    decay_lambda: float = 0.01          # decay rate
    reinforce_boost: float = 0.15       # access時の強化量
    prune_threshold: float = 0.1        # 自動削除閾値
    strength_cap: float = 2.0           # 最大strength

    # Collider
    collider_sim_min: float = 0.65      # 衝突候補の最小類似度
    collider_sim_max: float = 0.90      # 衝突候補の最大類似度
    dream_max_collisions: int = 10      # Dream Cycle 1回の最大衝突数

    # Content limits
    max_content_length: int = 50_000    # observation content最大文字数
    max_title_length: int = 500

    # Audit
    enable_audit_log: bool = True       # 操作ログ記録
    prompt_retention_days: int = 30
    audit_retention_days: int = 90
    redact_secret_patterns: bool = True
```

---

## 13. Non-Functional Requirements

### 13.1 Concurrency
```
SQLite WAL mode + PRAGMA busy_timeout = 5000
専用writer connectionを1本持ち、write系操作はその接続に集約する
read系は per-request read connection を払い出す
接続プール: 1 writer + N readers (sqlite3 check_same_thread=False)
write開始時は毎回 BEGIN IMMEDIATE を明示して writer lock を早期取得する
connection pool size は config (`db_reader_pool_size`) で調整可能
MCPサーバーは単一プロセス → 内部でwriter lockを管理
```

### 13.2 Backup & Recovery
```
1. 起動時自動バックアップ: memory.db → memory.db.bak
2. erinys_backup() ツール: 任意タイミングでバックアップ作成
3. 移行前バックアップ: migrate実行前に元DBを自動コピー
4. WAL checkpointing: session_end時にWAL→main DBをフラッシュ
```

### 13.3 Error Handling
```python
# 全ツールの戻り値は統一フォーマット

# 成功時
{"ok": True, "data": {"id": 42}, "error": None}

# エラー時
{"ok": False, "data": None, "error": {"message": "observation not found", "code": "NOT_FOUND"}}

# エラーコード一覧
# NOT_FOUND:    指定IDが存在しない
# DUPLICATE:    unique制約違反
# VALIDATION:   入力値が不正 (type/scope/relation)
# DB_ERROR:     SQLite内部エラー
# CONTENT_TOO_LONG: max_content_length超過
# EMBEDDING_ERROR: embedding生成失敗
```

### 13.4 Logging
```
Python logging モジュール使用
Level: INFO (操作ログ) / WARNING (DB size警告) / ERROR (失敗)
audit_log テーブルに全操作を記録 (configで無効化可能)
```

### 13.5 Migration Safety
```
1. migrate実行前に元DB自動バックアップ
2. トランザクション内で実行 (失敗時は自動ロールバック)
3. 移行後の検証: 件数一致チェック + ランダム10件の内容照合
4. ロールバック: バックアップファイルからの復元コマンド
```

### 13.6 Security
```
1. local-first trust model: すべての保存・検索・バックアップはローカルSQLite内で完結
2. secret pattern redaction before save: API key / token / secret らしき文字列は保存前にマスク
3. prompts / audit_log は configurable retention policy で定期削除可能
4. no network access (runtime): embedding推論・search・backup・migration は外部送信なしで動作
   - 例外: fastembed 初回モデルDL (セットアップ時のみ、§4 参照)
   - エアギャップ対応: FASTEMBED_CACHE_PATH で事前配置済みモデルを使用
```

---

## 14. MECE Completeness Checklist

### Data Lifecycle Coverage
| 操作 | ツール | Phase |
|:--|:--|:--|
| **Create** | erinys_save, erinys_save_prompt, erinys_batch_save | P0/P1 |
| **Read** | erinys_get, erinys_search, erinys_recall, erinys_context | P0 |
| **Update** | erinys_update, erinys_reinforce, erinys_supersede | P0/P2 |
| **Delete** | erinys_delete, erinys_prune | P0/P1 |
| **Backup** | erinys_backup | P0 |
| **Export** | erinys_export | P0 |
| **Stats** | erinys_stats, erinys_eval | P0/P3 |

### engram 互換性マッピング
| engram | ERINYS | 備考 |
|:--|:--|:--|
| mem_save | erinys_save | metadata追加 |
| mem_search | erinys_search | RRF hybrid化 |
| mem_context | erinys_context | ✅ |
| mem_get_observation | erinys_get | ✅ 追加済み |
| mem_update | erinys_update | ✅ 追加済み |
| mem_save_prompt | erinys_save_prompt | ✅ 追加済み |
| mem_session_start | erinys_session_start | ✅ |
| mem_session_end | erinys_session_end | ✅ |
| mem_session_summary | erinys_session_summary | ✅ 追加済み |
| mem_suggest_topic_key | erinys_suggest_topic (P2) | 後回し |
| mem_capture_passive | erinys_save + parsing (HA側) | HA skillで処理 |

### contextplus 互換性マッピング
| contextplus | ERINYS | 備考 |
|:--|:--|:--|
| upsert_memory_node | erinys_save (topic_key) | ✅ |
| search_memory_graph | erinys_search | RRF化 |
| create_relation | erinys_link | ✅ |
| retrieve_with_traversal | erinys_traverse | ✅ |
| prune_stale_links | erinys_prune | ✅ |
| add_interlinked_context | erinys_batch_save | ✅ 追加済み |

---

## Open Questions

> [!IMPORTANT]
> **Q1: 名前「ERINYS」で行くか？**
> PyPI に `erinys` は既存パッケージあり。`erinys-memory` なら空いている可能性が高い。事前にPyPI/npm/GitHub での名前衝突を確認する必要がある。

> [!IMPORTANT]
> **Q2: Embedding モデルのデフォルト**
> `bge-small-en-v1.5` (384次元) は英語メイン。日本語メインなら `bge-m3` (1024次元) にすべきか？ 両対応にするなら config で切り替え可能にする。vec0 テーブルの次元が固定なので、モデル変更=DB再構築になる点に注意。

> [!WARNING]
> **Q3: Memory Collider のLLM依存**
> 軽量版（キーワード抽出+ルールベース）で十分か、LLM呼び出し版（高品質だがAPI依存）も用意するか。local-first の原則とのトレードオフ。
