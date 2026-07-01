# MCP Tool Surface

The `erinys-memory` MCP server (`erinys_memory.server`) exposes 28 tools, all
prefixed `erinys_`. They fall into three tiers by stability and by what they
depend on:

- **Basic** — the stable core you reach for every session: writing memories,
  retrieving them, and managing sessions. All local, zero LLM calls.
- **Governance** — the trust layer that makes memory *verifiable*: provenance,
  lineage, contradiction detection, time-travel queries, and deletion you can
  prove. Also local and LLM-free.
- **Experimental** — research / higher-abstraction features. **`distill`,
  `dream`, and `collide` call an LLM** (a local Ollama endpoint by default) and
  are **not part of the zero-LLM retrieval path** described in the README. Treat
  their output as generated, not retrieved. `eval` is LLM-free but reports
  heuristic quality metrics and is subject to change.

Tier assignment does not change how you call a tool — it tells you how much to
lean on it. Every tool returns the same `{ok, data, error}` envelope.

For where these guarantees end (verified forgetting scope, CJK, temporal
inference, the LLM-dependent features above), see
[LIMITATIONS.md](./LIMITATIONS.md).

---

## Basic — stable core (17 tools)

Local, zero LLM calls. Retrieval (`search` / `recall` / `context`) runs the
FTS5 + sqlite-vec + RRF pipeline in ~7–10 ms with no model inference.

| Tool | Purpose |
|:--|:--|
| `erinys_save` | Save one structured observation (title, content, type, scope, provenance). |
| `erinys_batch_save` | Bulk-insert observations and auto-link similar ones into edges. |
| `erinys_get` | Fetch a single observation by ID with full, untruncated content. |
| `erinys_search` | RRF hybrid search (FTS5 keyword + vector similarity) — the core retrieval path. |
| `erinys_recall` | Return the most recent active observations for quick context. |
| `erinys_context` | Recent sessions plus recent observations for a given project. |
| `erinys_stats` | Database statistics: counts, per-project breakdown, vector health. |
| `erinys_update` | Update selected fields of an existing observation (provenance preserved). |
| `erinys_link` | Create a typed, weighted edge between two observations. |
| `erinys_traverse` | Walk the graph from a starting observation, optionally filtered by relation. |
| `erinys_reinforce` | Bump access count / last-accessed to slow decay and boost effective strength. |
| `erinys_export` | Export observations as Obsidian-compatible markdown with `[[wikilinks]]`. |
| `erinys_backup` | Create a consistent SQLite backup and return its metadata. |
| `erinys_save_prompt` | Record a user prompt to track intent and goals. |
| `erinys_session_start` | Open a session for a project (optional working directory). |
| `erinys_session_end` | Close a session, optionally attaching a summary. |
| `erinys_session_summary` | Save a comprehensive end-of-session summary as an observation. |

## Governance — the trust layer (7 tools)

Local, zero LLM calls. These make memory auditable and correctable over time:
where a fact came from, what it superseded, what contradicts it, what was true
at a given moment, and proof that a deletion actually happened.

| Tool | Purpose |
|:--|:--|
| `erinys_lineage` | Trace an observation's provenance ancestry (parents / `distilled_from`) into a derivation chain. |
| `erinys_supersede` | Replace an old fact with an updated one, closing the old fact's validity and recording the reason. |
| `erinys_conflict_check` | Detect observations that contradict a given one. |
| `erinys_timeline` | Time-travel query: which facts were valid as of a specific point in time. |
| `erinys_delete` | Delete a single observation, cascading dependent rows via FK rules. |
| `erinys_forget` | Verified forgetting: delete an observation and its derived closure in one transaction, then prove zero residue with a membership test across every in-database substrate. Defaults to `dry_run`. See LIMITATIONS.md for external-substrate caveats. |
| `erinys_prune` | Remove decayed observations below a strength threshold. Defaults to `dry_run`. |

## Experimental — research / LLM-dependent (4 tools)

| Tool | Purpose |
|:--|:--|
| `erinys_distill` | **LLM-dependent.** Generate a higher-abstraction layer (abstract / meta) from a concrete observation. |
| `erinys_dream` | **LLM-dependent.** Dream Cycle: batch-collide candidate memory pairs to propose new connections. |
| `erinys_collide` | **LLM-dependent.** Manually collide two observations to synthesize an insight. |
| `erinys_eval` | Self-evaluate memory quality with LoCoMo-inspired heuristic metrics (LLM-free, but the scoring is experimental and may change). |

> **`distill` / `dream` / `collide` call an LLM** — a local Ollama endpoint by
> default. They are **not** part of the zero-LLM retrieval pipeline. If you point
> their endpoint off-machine, memory content leaves your machine; ERINYS warns
> when you configure that. See
> [LIMITATIONS.md § LLM-dependent features](./LIMITATIONS.md).

---

**Tier counts:** Basic 17 · Governance 7 · Experimental 4 · **Total 28.**
