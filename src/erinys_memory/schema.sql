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
