<p align="center">
  <img src="assets/logo.jpg" alt="ERINYS" width="200">
</p>

# ERINYS — Reflexive Memory for AI Agents

[🇯🇵 日本語版 / Japanese](README_ja.md)

> **From memories that existed, it even creates memories that never did.**

AI agent memory systems have always mimicked human memory. Short-term, long-term, episodic, semantic — textbook categories bolted straight onto implementations.

Something always felt off.

Humans forget. But existing memory systems don't. They grow endlessly, serving stale facts with the same weight as fresh ones. Humans notice "wait, didn't you say something different before?" But memory systems silently overwrite. Humans connect two unrelated experiences and think "oh, I can use that here." But memory systems just store and retrieve.

What needed to be mimicked wasn't the taxonomy of memory. It was the behavior.

That discomfort is what summoned ERINYS.

ERINYS is a guard dog. It remembers, forgets, questions, and bites.

## What Makes ERINYS Different

**Forgetting.** Most memory systems only accumulate. ERINYS decays memories over time following the Ebbinghaus forgetting curve. Old noise sinks. Frequently accessed knowledge floats. Search results stay relevant without manual curation.

**Distillation.** A specific bugfix ("JWT httpOnly flag was missing") automatically generates three layers: the concrete fact → a reusable pattern ("new endpoints need a security checklist") → a universal principle ("security defaults should be safe without opt-in"). No other memory system does this.

**Dream Cycle.** Two memories are fed to an LLM: "is there a connection?" Candidate pairs are selected by semantic similarity — close enough to be related (cosine > 0.65), far enough to not be redundant (< 0.90). Scheduled overnight via cron, it finds connections you'd never think to look for. No magic — just automated note comparison at scale.

## Design Philosophy

### Memory has layers

Not all memory is equal. ERINYS organizes knowledge by abstraction level:

- **Concrete** — what happened. "The JWT httpOnly flag was missing on `/api/auth`."
- **Abstract** — patterns from facts. "New API endpoints need a security header checklist."
- **Meta** — principles from patterns. "Security defaults should be safe without manual opt-in."

A single bugfix generates all three through distillation. The meta layer accumulates principles that transfer across projects and tech stacks.

### Forgetting is a feature

Every memory has a strength score that decays over time. A memory saved 6 months ago ranks lower than one saved yesterday. Memories accessed frequently resist decay — repeated retrieval reinforces them.

When strength drops below a threshold, the memory becomes a pruning candidate. The database stays lean. Search stays relevant.

### Facts change. History shouldn't disappear

When information updates — "we moved from AWS to GCP" — ERINYS doesn't overwrite. It creates a supersede chain: the old fact is marked as replaced but preserved. You can ask "what did we believe in March?" and get the answer that was true then.

### Contradictions should be caught

If memory contains both "use PostgreSQL" and "use SQLite", ERINYS detects the conflict. Instead of silently switching, the agent asks: "you previously chose PostgreSQL — has the requirement changed?"

### Search finds meaning, not just keywords

Two searches run simultaneously and fuse results:

- **Keyword search** (FTS5) — exact term matching.
- **Vector search** (sqlite-vec) — semantic similarity. "authentication" finds "login", "JWT", "session tokens".

Results merge via Reciprocal Rank Fusion (RRF). High in both = highest score.

### Everything stays local

Single SQLite file. No cloud APIs. No API keys. No subscriptions. Offline-capable. Your agent's memory never leaves your machine.

## Use Cases

### 1. Cross-Session Memory for Coding Agents

```python
# Agent saves a learning after fixing a bug
erinys_save(
  title="Fixed JWT httpOnly flag missing",
  content="Cookie was accessible via JS. Added httpOnly: true, secure: true, sameSite: strict.",
  type="bugfix",
  project="my-app"
)

# Next week, similar task — agent searches memory
erinys_search(query="authentication cookie security", project="my-app")
# → Returns the JWT fix with relevance score
```

### 2. Contradiction Detection

```python
erinys_save(title="Database choice", content="Using SQLite for simplicity", project="my-app")
erinys_conflict_check(observation_id=42)
# → "⚠️ Conflicts with #18: 'Using PostgreSQL for production reliability'"
```

### 3. Dream Cycle — Overnight Knowledge Synthesis

```python
erinys_dream(max_collisions=10)
# Picks memory pairs in the "sweet spot" (cosine 0.65–0.90)
# Memory A: "RTK reduces token usage by 60-90%"
# Memory B: "Bootstrap Gate takes 3 seconds due to multiple script calls"
# → Insight: "Apply RTK prefix to Bootstrap Gate scripts to reduce overhead"
```

### 4. Temporal Queries

```python
erinys_timeline(query="deployment target", as_of="2026-03-01")
# → "AWS EC2 (decided 2026-02-15)"

erinys_timeline(query="deployment target", as_of="2026-04-01")
# → "GCP Cloud Run (superseded AWS on 2026-03-20)"
```

### 5. Knowledge Distillation

```python
erinys_save(title="Forgot CORS headers on new endpoint", type="bugfix", ...)
erinys_distill(observation_id=50, level="meta")
# → concrete: "CORS headers missing on /api/v2/users endpoint"
# → abstract: "New API endpoints need a CORS review checklist"
# → meta:     "Security concerns should be opt-out, not opt-in"
```

### 6. Obsidian Export

```python
erinys_export(format="markdown")
# → Generates .md files with [[wikilinks]]
# Drop into Obsidian → instant knowledge graph
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run as MCP server (stdio)
python -m erinys_memory.server

# Run tests
PYTHONPATH=src pytest tests/ -v
```

## MCP Configuration

### Claude Desktop / Claude Code

```json
{
  "mcpServers": {
    "erinys": {
      "command": "/path/to/ERINYS-mem/.venv/bin/python3",
      "args": ["-m", "erinys_memory.server"],
      "env": {
        "ERINYS_DB_PATH": "~/.erinys/memory.db"
      }
    }
  }
}
```

### Gemini (Antigravity)

Add to `~/.gemini/antigravity/settings.json` under `mcpServers`:

```json
{
  "erinys": {
    "command": "/path/to/ERINYS-mem/.venv/bin/python3",
    "args": ["-m", "erinys_memory.server"],
    "env": {
      "ERINYS_DB_PATH": "~/.erinys/memory.db"
    }
  }
}
```

## Environment Variables

| Variable | Default | Description |
|:--|:--|:--|
| `ERINYS_DB_PATH` | `~/.erinys/memory.db` | SQLite database path |
| `ERINYS_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | fastembed model |

## Tools (25)

### Core
- `erinys_save` — Save observation (with topic_key upsert)
- `erinys_get` — Get by ID (full content, untruncated)
- `erinys_update` — Partial update
- `erinys_delete` — Delete with FK cascade
- `erinys_search` — RRF hybrid search (FTS5 + vector)
- `erinys_save_prompt` — Save user prompt
- `erinys_recall` — Recent observations
- `erinys_context` — Session context recall
- `erinys_export` — Obsidian-compatible markdown export
- `erinys_backup` — SQLite backup
- `erinys_stats` — Database statistics

### Graph
- `erinys_link` — Create typed edge
- `erinys_traverse` — BFS graph traversal
- `erinys_prune` — Prune weak/decayed edges

### Temporal
- `erinys_reinforce` — Boost observation strength
- `erinys_supersede` — Version an observation
- `erinys_timeline` — Query as-of timestamp
- `erinys_conflict_check` — Detect contradictions

### Dream Cycle
- `erinys_collide` — Collide two observations via LLM
- `erinys_dream` — Batch collision cycle

### Distillation
- `erinys_distill` — 3-granularity abstraction (concrete → abstract → meta)

### Batch & Eval
- `erinys_batch_save` — Bulk save with auto-linking
- `erinys_eval` — LOCOMO-inspired quality metrics

### Session
- `erinys_session_start` — Start session
- `erinys_session_end` — End session with summary
- `erinys_session_summary` — Save structured summary

## How ERINYS Compares

| Feature | ERINYS | Mem0 | Official MCP Memory |
|:--|:--|:--|:--|
| **Hybrid search** (keyword + vector) | ✅ FTS5 + sqlite-vec RRF | ✅ Vector + graph | ❌ Knowledge graph only |
| **Time-decay forgetting** | ✅ Ebbinghaus curve | ⚠️ Priority scoring | ❌ |
| **3-level distillation** (concrete → abstract → meta) | ✅ | ❌ | ❌ |
| **Dream Cycle** (collision-based insight) | ✅ | ❌ | ❌ |
| **Contradiction detection** | ✅ | ⚠️ Overwrites via resolver | ❌ |
| **Temporal queries** ("what did we believe in March?") | ✅ Supersede chain | ⚠️ Graph invalidation | ❌ |
| **Local-first** (no cloud API) | ✅ SQLite single file | ❌ Cloud default | ✅ |
| **Obsidian export** | ✅ [[wikilinks]] | ❌ | ❌ |
| **Auto-distill on save** | ✅ | ❌ | ❌ |
| **MCP native** | ✅ 25 tools | ✅ | ✅ |
| **Self-evaluation** (LOCOMO metrics) | ✅ | ❌ | ❌ |

> **TL;DR** — Most memory servers store and retrieve. ERINYS also forgets, distills, and dreams.

## Architecture

```
┌──────────────────────────┐
│     FastMCP Server       │  25 tools, unified envelope
├──────────────────────────┤
│  search.py  │ graph.py   │  RRF hybrid │ typed edges
│  decay.py   │ session.py │  Ebbinghaus │ lifecycle
│  temporal.py│collider.py │  versioning │ cross-pollination
│  distill.py │ db.py      │  abstraction│ SQLite + vec
├──────────────────────────┤
│  embedding.py            │  fastembed (BAAI/bge-small-en-v1.5)
├──────────────────────────┤
│  SQLite + FTS5 + vec0    │  Local-first, no network at runtime
└──────────────────────────┘
```

## Roadmap

- [ ] Dream Daemon — Background auto-execution of Dream Cycle
- [x] Auto-Distill on Save — Trigger 3-granularity distillation on every save
- [ ] Auto-Prune — GC decayed observations when DB exceeds size threshold
- [ ] Cron-ready CLI — `erinys dream --max 10` for scheduled overnight synthesis
- [ ] PyPI package — `pip install erinys-memory`
- [ ] Multi-agent support — Scoped memory per agent identity

## License

MIT
