<p align="center">
  <img src="assets/logo.png" alt="ERINYS" width="600">
</p>

# ERINYS — Verifiable local memory for AI agents

**v0.4.1** · **10ms local retrieval. No API key. No token cost.**

Retrieval recall 100% on LongMemEval-S (`_s` split); end-to-end QA accuracy pending — see [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

[🇯🇵 日本語版 / Japanese](README_ja.md) · [Limitations](docs/LIMITATIONS.md) · [Comparison](./docs/COMPARISON.md)

Stores facts, preserves history, catches contradictions, and proves deletion. A local trust layer for agent memory — one SQLite file, zero LLM calls in retrieval.

AI agent memory systems have always mimicked human memory. Short-term, long-term, episodic, semantic — textbook categories bolted straight onto implementations.

Something always felt off.

Humans forget. But existing memory systems don't. They grow endlessly, serving stale facts with the same weight as fresh ones. Humans notice "wait, didn't you say something different before?" But memory systems silently overwrite. Humans connect two unrelated experiences and think "oh, I can use that here." But memory systems just store and retrieve.

What needed to be mimicked wasn't the taxonomy of memory. It was the behavior.

That discomfort is what summoned ERINYS.

ERINYS is a guard dog. It stores facts, preserves history, catches contradictions, and proves deletion.

> **Origin:** ERINYS was built as the retrieval layer for [HyperAION](https://aionexo.com/hyperaion/), an AI agent self-improvement framework. It is released as a standalone MCP server so any agent stack can use it independently.

## Quickstart (30 seconds)

**1. Install.**

```bash
pip install erinys-memory
```

**2. Verify your environment.** One command checks Python, SQLite + extension support, sqlite-vec, embeddings, deps, and the DB — each failing check prints a `fix`.

```bash
erinys doctor
```

**3. Register the MCP server** with your client (Claude Desktop / Claude Code / any stdio MCP host):

```json
{
  "mcpServers": {
    "erinys": {
      "command": "erinys-memory",
      "env": {
        "ERINYS_DB_PATH": "~/.erinys/memory.db"
      }
    }
  }
}
```

**4. Save and search** from the JSON CLI (no LLM, no network):

```bash
erinys save --title "JWT httpOnly flag was missing" \
  --content "Cookie was JS-accessible; added httpOnly, secure, sameSite=strict." \
  --type bugfix --project demo

erinys search "auth cookie security" --project demo
```

Retrieval runs in ~7–10ms against a single local SQLite file, with zero LLM calls. Next steps: [Limitations](docs/LIMITATIONS.md) · [Comparison](./docs/COMPARISON.md).

## CLI-First Operations

ERINYS automation should use the JSON CLI as the primary control surface. MCP remains available as an agent-facing adapter, but scheduled jobs, CI, recovery, and manual verification should call the CLI so failures have stable exit codes and machine-readable output. This first CLI phase bypasses the MCP protocol while reusing the existing `erinys_memory.server` functions; the next refactor should split those functions into a protocol-neutral service layer.

From your workspace:

```bash
python3 .agent/scripts/erinys_cli.py health --project my-project --json          # light: no venv needed; cannot verify vectors (may report degraded)
python3 .agent/scripts/erinys_cli.py health --project my-project --deep --json   # authoritative: server import + search smoke test
python3 .agent/scripts/erinys_cli.py context --project my-project --limit 10 --readonly --json
python3 .agent/scripts/erinys_cli.py search "Buffer DNS" --project my-project --limit 5 --readonly --json
python3 .agent/scripts/erinys_cli.py undistilled --project my-project --limit 10 --json
python3 .agent/scripts/erinys_cli.py save --title "Decision" --content "What: ..." --type decision --project my-project --json
python3 .agent/scripts/erinys_cli.py distill 123 --level meta --json
```

`--readonly` reads via SQLite `mode=ro` (keyword search only, no migration / audit-log writes, no venv required). Drop it when semantic search is needed. `dream` / `prune` operate on the whole database across ALL projects; `prune --execute` additionally requires `--confirm-global`. Usage errors also emit JSON (`error.code: "USAGE"`, exit code 2).

When running inside the ERINYS package directly, use:

```bash
python -m erinys_memory.cli health --project my-project --json
```

## What's New in v0.4.0 — VMG (Verifiable Memory Governance)

ERINYS now governs the *provenance* and *forgetting* of every memory, mapping to the Verifiable Memory Governance framework for long-term agent memory.

**Provenance Visibility.** Every observation carries a server-controlled `metadata.provenance` block — `principal` (who wrote it), `source`, `derived_via` (`save`/`batch_save`/`distill`/`session_summary`/`supersede`), `parents` (lineage), and `recorded_at`. It is stamped on *all* write paths and cannot be spoofed by the caller. The new **`erinys_lineage`** tool walks a memory's ancestry to a lineage-complete chain (with a graceful `distilled_from` fallback for pre-0.4 memories).

**Verified Forgetting.** The new **`erinys_forget`** tool deletes a memory *and its derived closure* (distilled descendants, found via `distilled_from` or `provenance.parents`) in a single transaction, then runs a **membership test** proving zero residue across every DB substrate (`observations` / `vec_observations` / FTS / `edges` / `collisions`). Unlike `erinys_delete`, it can forget a parent that still has children (the closure resolves the `NO ACTION` FK). External substrates (the Obsidian vault clears on the next export sweep; `.bak` backups are retained by design) are explicitly reported as out of DB-verification scope.

**Migration.** `scripts/backfill_provenance.py` retro-stamps provenance onto pre-0.4 rows from existing columns (idempotent, dry-run by default, WAL-safe backup). Optional — lineage and forgetting already work on legacy rows via the `distilled_from` fallback. The DB `schema_version` is unchanged (provenance lives inside the existing `metadata` column).

## What's New in v0.2.0

**MAGMA improvements** — five enhancements from Growth Radar analysis:

- **Adaptive Retrieval** — Query complexity classification (L1/L2/L3) automatically adjusts FTS/vec search weights. CJK queries default to vector-heavy mode, bypassing FTS5 porter tokenizer limitations.
- **Intent-Aware Router** — Classifies queries as WHY/WHO/WHEN/WHAT/GENERAL and adjusts boost parameters per intent. "Why did X fail?" triggers causal graph edges and higher vec weight.
- **Graph Knowledge Reranking** — After RRF fusion, graph-reachable nodes receive a 1.15× score boost. Supports `causal`, `entity`, `temporal` edge types alongside existing relation types.
- **Distillation Quality Scoring** — Level-aware quality scores (concrete: keyword-heavy, meta: semantic-heavy) with compression ratio scoring. Results stored as metadata on distilled observations.
- **Dream Cycle Outcome Scoring** — Automatic novelty/relevance/serendipity scoring for collisions. Scores persisted as JSON metadata for post-hoc analysis.

Schema upgraded to v2 with automatic migration from v1.

## Benchmarks

These are **retrieval recall** numbers (is the correct session in the Top-K?), not end-to-end QA accuracy. Retrieval recall 100% on LongMemEval-S (`_s` split); end-to-end QA accuracy pending — see [docs/LIMITATIONS.md](docs/LIMITATIONS.md). All results use the same mode (`enhanced_v2_boost`) with **zero LLM calls** in the retrieval pipeline. Note: higher-level features (Dream Cycle, Distillation) do use an LLM — see below.

| Benchmark | N | R@5 | R@10 | Avg Latency |
|:--|:--|:--|:--|:--|
| **LongMemEval-S** | 500 | **100.0%** | **100.0%** | 10.3 ms |
| **LoCoMo** | 1,982 | **94.0%** | **98.1%** | 6.9 ms |
| **ConvoMem** | 250 | **97.6%** | — | — |

> **Why this matters:** No API keys. No network. No tokens burned for retrieval. ERINYS achieves these results with FTS5 + sqlite-vec + algorithmic boosting alone. Your agent's memory searches at the speed of SQLite.
>
> LongMemEval evaluated on `longmemeval_s` split (~20 sessions/question). **Results on the harder `_m` split have not yet been evaluated.** Full methodology, per-category breakdown, and reproduction commands → [benchmarks/BENCHMARKS.md](benchmarks/BENCHMARKS.md)

The story of how we got to 100% → [🇯🇵 Japanese](docs/benchmark_story_ja.md) / [🇺🇸 English](docs/benchmark_story_en.md)

## What Makes ERINYS Different

**Forgetting.** Most memory systems only accumulate. ERINYS decays memories over time following the Ebbinghaus forgetting curve. Old noise sinks. Frequently accessed knowledge floats. Search results stay relevant without manual curation. Decay runs automatically — no LLM needed.

**Distillation.** A specific bugfix ("JWT httpOnly flag was missing") automatically generates three layers: the concrete fact → a reusable pattern ("new endpoints need a security checklist") → a universal principle ("security defaults should be safe without opt-in"). No other memory system does this. ⚠️ *Distillation requires an LLM call to generate the abstract/meta layers. v0.2.0 adds quality scoring per distillation level.*

**Dream Cycle.** Two memories are fed to an LLM: "is there a connection?" Candidate pairs are selected by semantic similarity — close enough to be related (cosine > 0.65), far enough to not be redundant (< 0.90). Currently triggered manually via `erinys_dream`. ⚠️ *Dream Cycle requires LLM calls — it is not part of the zero-LLM retrieval pipeline. v0.2.0 adds automatic outcome scoring (novelty/relevance/serendipity).*

> Distillation and the Dream Cycle are the generative edge of ERINYS: from memories that existed, it even synthesizes memories that never did. Both are LLM-backed and live outside the zero-LLM retrieval path.

**Adaptive Search (v0.2.0).** Query complexity is classified automatically. Simple keyword lookups stay FTS-heavy. Complex multi-hop questions shift to vector-heavy retrieval. CJK queries and mixed CJK+ASCII queries are routed to vector search by default, where embedding models outperform FTS5's porter tokenizer on non-Latin scripts.

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

- **Keyword search** (FTS5) — exact term matching with NEAR phrase expansion.
- **Vector search** (sqlite-vec) — semantic similarity via BGE embeddings.
- **RRF fusion** — Reciprocal Rank Fusion combines both rankings with adaptive weights.
- **Intent routing** (v0.2.0) — WHY/WHEN/WHO queries adjust boost parameters and edge types.
- **Graph reranking** (v0.2.0) — Knowledge graph neighbors boost fusion scores.

No LLM in the loop. Retrieval latency stays under 15ms.
