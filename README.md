<p align="center">
  <img src="assets/logo.png" alt="ERINYS" width="600">
</p>

# ERINYS — Verifiable local memory for AI agents

<p>
  <a href="https://pypi.org/project/erinys-memory/"><img src="https://img.shields.io/pypi/v/erinys-memory" alt="PyPI"></a>
  <a href="https://github.com/GhostyAI-HA/ERINYS-mem/actions/workflows/test.yml"><img src="https://github.com/GhostyAI-HA/ERINYS-mem/actions/workflows/test.yml/badge.svg" alt="Tests"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License"></a>
</p>

**~10 ms local retrieval. No API key. No token cost.**

A local trust layer for agent memory: one SQLite file, zero LLM calls in the retrieval path. Retrieval recall 99.4% R@5 / 100% R@10 on LongMemEval-S, 96.8% R@5 on the harder `_m` split — see [Benchmarks](#benchmarks).

[🇯🇵 日本語](README_ja.md) · [Limitations](docs/LIMITATIONS.md) · [Comparison](docs/COMPARISON.md) · [Tool reference](docs/TOOLS.md) · [Changelog](CHANGELOG.md)

---

AI agent memory systems have always mimicked human memory. Short-term, long-term, episodic, semantic — textbook categories bolted straight onto implementations.

Something always felt off.

Humans forget. But existing memory systems don't — they grow endlessly, serving stale facts with the same weight as fresh ones. Humans notice "wait, didn't you say something different before?" But memory systems silently overwrite. Humans connect two unrelated experiences and think "oh, I can use that here." Memory systems just store and retrieve.

What needed to be mimicked wasn't the taxonomy of memory. It was the behavior. That discomfort is what summoned ERINYS.

ERINYS is a guard dog: it stores facts, preserves history, catches contradictions, and proves deletion.

> **Origin:** ERINYS was built as the retrieval layer for [HyperAION](https://aionexo.com/hyperaion/), an AI agent self-improvement framework. It is released as a standalone MCP server so any agent stack can use it independently.

## Quickstart (30 seconds)

**1. Install.**

```bash
pip install erinys-memory
```

**2. Verify your environment.** One command checks Python, SQLite extension support, sqlite-vec, embeddings, dependencies, and the DB — each failing check prints a `fix`.

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

Retrieval runs in ~7–10 ms against a single local SQLite file, with zero LLM calls.

## Benchmarks

These are **retrieval recall** numbers ("is the correct session in the top-K?"), not end-to-end QA accuracy — QA and answerability harnesses shipped in v0.5.1, real runs pending ([LIMITATIONS.md](docs/LIMITATIONS.md)). All results use the same mode (`enhanced_v2_boost`) with **zero LLM calls** in the retrieval pipeline, reproduced on current dependencies (2026-07).

| Benchmark | N | R@5 | R@10 |
|:--|:--|:--|:--|
| **LongMemEval-S** | 500 | **99.4%** | **100.0%** |
| **LongMemEval-M** (~476-session haystack) | 500 | **96.8%** | 98.0% |
| **LoCoMo** | 1,982 | **92.7%** (fair ≈ 95.7%¹) | 97.2% |
| **ConvoMem** | 250 | 97.6%² | — |

¹ A miss audit found 42% of LoCoMo R@5 misses are benchmark-label defects (single-gold labels for multi-session answers, unanswerable adversarial questions), not retrieval failures.
² April 2026 configuration; re-run on current dependencies pending.

> **Why this matters:** no API keys, no network, no tokens burned for retrieval. ERINYS reaches these numbers with FTS5 + sqlite-vec + algorithmic boosting alone — your agent's memory searches at the speed of SQLite. Full methodology, per-category breakdowns, miss analyses, and reproduction commands → [benchmarks/BENCHMARKS.md](benchmarks/BENCHMARKS.md)

The story of how we got here → [🇺🇸 English](docs/benchmark_story_en.md) / [🇯🇵 日本語](docs/benchmark_story_ja.md)

## What makes ERINYS different

**Forgetting.** Most memory systems only accumulate. ERINYS decays memories over time following the Ebbinghaus forgetting curve: old noise sinks, frequently accessed knowledge floats. Search results stay relevant without manual curation — decay runs automatically, no LLM needed.

**Verified forgetting.** `erinys_forget` deletes a memory *and its derived closure* (distilled descendants) in a single transaction, then runs a membership test proving zero residue across every DB substrate (observations / vectors / FTS / edges / collisions). Deletion you can prove, not just request.

**Provenance.** Every observation carries a server-stamped provenance block — who wrote it, via which write path, derived from which parents. `erinys_lineage` walks any memory back to its origins. Callers cannot spoof it.

**Distillation.** A specific bugfix ("JWT httpOnly flag was missing") automatically generates three layers: the concrete fact → a reusable pattern ("new endpoints need a security checklist") → a universal principle ("security defaults should be safe without opt-in"). ⚠️ *Requires an LLM call (local Ollama by default).*

**Dream Cycle.** Two memories are fed to an LLM: "is there a connection?" Candidate pairs are selected by semantic similarity — close enough to be related (cosine > 0.65), far enough to not be redundant (< 0.90). ⚠️ *Requires LLM calls; not part of the zero-LLM retrieval path.*

> Distillation and the Dream Cycle are the generative edge of ERINYS: from memories that existed, it even synthesizes memories that never did. Both are LLM-backed and live outside the zero-LLM retrieval path.

**Adaptive search.** Query complexity is classified automatically (L1/L2/L3): simple keyword lookups stay FTS-heavy, complex multi-hop questions shift to vector-heavy retrieval. WHY/WHO/WHEN intents adjust boost parameters and graph edge types, and graph-reachable neighbors are reranked upward. CJK queries route to vector search by default, where embedding models outperform FTS5's porter tokenizer.

## Design philosophy

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
- **Vector search** (sqlite-vec) — semantic similarity via local embeddings.
- **RRF fusion** — Reciprocal Rank Fusion combines both rankings with adaptive weights.
- **Intent routing** — WHY/WHEN/WHO queries adjust boost parameters and edge types.
- **Graph reranking** — knowledge-graph neighbors boost fusion scores.

No LLM in the loop. Retrieval latency stays under 15 ms.

### Everything stays local

One SQLite file. No cloud API, no subscription, works offline. Your agent's memory never leaves the machine.

## Use cases

**Memory across sessions.** An agent saves what it learned; next week's agent finds it.

```python
erinys_save(title="Fixed JWT httpOnly flag",
            content="Cookie was JS-accessible; added httpOnly, secure, sameSite=strict.",
            type="bugfix", project="my-app")
erinys_search(query="auth cookie security", project="my-app")
# → the JWT fix comes back, scored. The same mistake is not repeated.
```

**Contradiction detection.**

```python
erinys_save(title="DB choice", content="Use SQLite for simplicity", project="my-app")
erinys_conflict_check(observation_id=42)
# → "⚠️ Conflicts with observation #18: 'Use PostgreSQL for production reliability'"
```

**Time-travel queries.**

```python
erinys_timeline(query="deployment target", as_of="2026-03-01")  # → "AWS EC2 (decided 2026-02-15)"
erinys_timeline(query="deployment target", as_of="2026-04-01")  # → "GCP Cloud Run (switched 2026-03-20)"
```

**Verified forgetting.**

```python
erinys_forget(id=42)                  # dry run: shows the derived closure that would be deleted
erinys_forget(id=42, dry_run=False)   # deletes it, then proves zero residue across all DB substrates
```

## CLI

The MCP server is the agent-facing adapter; the same operations are available from a JSON CLI for scheduled jobs, CI, recovery, and manual verification — stable exit codes, machine-readable output.

```bash
erinys health --project my-app --deep --json     # authoritative: server import + search smoke test
erinys search "Buffer DNS" --project my-app --limit 5 --readonly --json
erinys context --project my-app --limit 10 --readonly --json
erinys save --title "Decision" --content "What: ..." --type decision --project my-app --json
erinys undistilled --project my-app --limit 10 --json
erinys distill 123 --level meta --json
```

- `--readonly` reads via SQLite `mode=ro` — keyword search only, no migration or audit-log writes. Drop it when semantic search is needed.
- `dream` / `prune` operate on the whole database across **all** projects; `prune --execute` additionally requires `--confirm-global`.
- Usage errors also emit JSON (`error.code: "USAGE"`, exit code 2).
- Module form: `python -m erinys_memory.cli <command>`.

## MCP tool surface (28 tools)

| Tier | Tools | What | LLM |
|:--|:--|:--|:--|
| **Basic** | 17 | save / search / recall / sessions — the stable core | ❌ none |
| **Governance** | 7 | lineage, supersede, timeline, conflict check, verified forgetting | ❌ none |
| **Experimental** | 4 | distill, dream, collide, eval — research features | ⚠️ distill / dream / collide call an LLM |

Every tool returns the same `{ok, data, error}` envelope. Full per-tool reference → [docs/TOOLS.md](docs/TOOLS.md)

## Configuration

| Variable | Default | Description |
|:--|:--|:--|
| `ERINYS_DB_PATH` | `~/.erinys/memory.db` | SQLite database path |
| `ERINYS_EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | fastembed embedding model |
| `ERINYS_AUTO_DISTILL` | `1` | Auto-distill on save (set `0` to disable) |
| `ERINYS_DISTILL_MODEL` | `gemma4:e4b` | Local Ollama model for distillation |
| `ERINYS_DISTILL_ENDPOINT` | `http://localhost:11434/api/generate` | Ollama generate endpoint |

## Architecture

```
┌─────────────────────────────┐
│       FastMCP Server        │  28 tools, uniform {ok, data, error} envelope
├─────────────────────────────┤
│ search.py     │ graph.py    │  RRF hybrid search │ typed edges
│ decay.py      │ session.py  │  Ebbinghaus decay  │ lifecycle
│ temporal.py   │ collider.py │  versioning        │ dream cycle
│ distill.py    │ policy.py   │  3-level distill   │ access policy
│ provenance.py │ db.py       │  lineage stamps    │ SQLite + vec
├─────────────────────────────┤
│ embedding.py                │  fastembed (multilingual-MiniLM, 384d)
├─────────────────────────────┤
│ SQLite + FTS5 + sqlite-vec  │  fully local, no network at runtime
└─────────────────────────────┘
```

## Development

```bash
git clone https://github.com/GhostyAI-HA/ERINYS-mem && cd ERINYS-mem
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
PYTHONPATH=src pytest tests/ -v          # run tests
python -m erinys_memory.server           # run the MCP server from source (stdio)
ollama pull gemma4:e4b                   # only for distill / dream (LLM features)
```

## Release highlights

- **v0.5** — benchmark truth (reproduced numbers, first `_m` split evaluation), temporal grounding, answerability, opt-in memory access policy, QA / answerability eval harnesses
- **v0.4** — VMG: server-stamped provenance, `erinys_lineage`, verified forgetting (`erinys_forget`); MAGMA retrieval (adaptive weights, intent routing, graph reranking)
- **v0.2** — adaptive retrieval, intent-aware router, distillation quality scoring, dream outcome scoring

Full details → [CHANGELOG.md](CHANGELOG.md)

## Roadmap

- [ ] Dream Daemon — run the Dream Cycle in the background (currently manual)
- [ ] Auto-prune when the DB crosses a size threshold
- [ ] Multi-agent scoping — per-agent memory isolation
- [ ] ConvoMem re-run + real end-to-end QA evaluation on current dependencies

## License

MIT © 2026 SHUN FUJIYOSHI (GhostyAI-HA) — see [LICENSE](LICENSE)
