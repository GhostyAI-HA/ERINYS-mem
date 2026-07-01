# ERINYS vs. Other Agent Memory Systems

This page is an honest, best-effort comparison of ERINYS against the memory
systems agent builders most often weigh it against. The goal is to help you pick
the *right* tool, not to win an argument. Where ERINYS is behind, this page says
so plainly.

> **One thing to get straight first: recall is not QA.** ERINYS publishes
> **retrieval recall** ("is the correct evidence in the top-K?"), not end-to-end
> question-answering accuracy ("did the agent produce the right final answer?").
> These are different metrics and must not be compared directly. Several systems
> below publish QA accuracy instead — those figures are labelled *(QA acc)* and
> are not apples-to-apples with ERINYS's R@5. See
> [docs/LIMITATIONS.md](LIMITATIONS.md) for the full caveats, and
> [benchmarks/BENCHMARKS.md](../benchmarks/BENCHMARKS.md) for exact conditions.

**What ERINYS is:** a local, embeddable memory engine — one SQLite file, FTS5 +
sqlite-vec + RRF, with zero LLM calls in the retrieval path, plus a memory
*governance* layer (provenance, verified forgetting, lineage, contradiction
detection). Positioning: **verifiable local memory for AI agents** — a *local
trust layer*, not a managed platform.

**What ERINYS is not:** a hosted service. There is no cloud API, dashboard,
multi-tenant control plane, SLA, or SDK spread across five languages. If you want
managed convenience and scale-on-someone-else's-machine, the hosted products
below are built for exactly that, and ERINYS is not trying to replace them.

---

## The contenders

| System | What it is | Deployment |
|:--|:--|:--|
| **ERINYS** | Local memory engine on one SQLite file; MCP server + JSON CLI | Local / embedded |
| **Mem0** | Memory layer that LLM-extracts structured facts; OSS core + hosted platform | Self-host or managed |
| **Zep / Graphiti** | Temporal knowledge-graph memory (Graphiti is the OSS engine under Zep) | Self-host (Graphiti) or managed (Zep Cloud) |
| **MemPalace** | OSS retrieval-focused memory with optional LLM reranking | Self-host |
| **Letta** (formerly MemGPT) | Stateful agent framework with self-editing memory + agent server | Self-host or managed |
| **LangGraph Memory / Store** | Memory primitives inside the LangGraph/LangChain stack (`BaseStore`, checkpointers) | Self-host or LangGraph Platform |
| **Vector DB / RAG baseline** | Roll-your-own: an embedding model + a vector store (pgvector, Chroma, Pinecone, …) | Depends on the store |

---

## Comparison matrix

Legend: ✅ strong / built-in · 🟡 partial or via extra work · ❌ not a goal / absent · "—" = no comparable public figure. Descriptions are qualitative where numbers are not independently available; **do not read a blank as zero.**

| Axis | ERINYS | Mem0 | Zep / Graphiti | MemPalace | Letta | LangGraph Memory/Store | Vector DB / RAG |
|:--|:--|:--|:--|:--|:--|:--|:--|
| **Deployment model** | Local / embedded (one SQLite file) | Self-host OSS or managed cloud | Graphiti self-host; Zep managed | Self-host OSS | Self-host or managed | Library in your stack; optional managed platform | Depends on the store (local → fully managed) |
| **Retrieval recall (measured)** | R@5 100% LongMemEval-**S** (`_s` only), 94% LoCoMo (98.1% R@10), 97.6% ConvoMem — *recall, not QA* | Not published as recall on these splits (publishes QA-style results) | Not published as recall on these splits (publishes QA-style results) | R@5 96.6% LongMemEval-S raw; higher with LLM rerank (see BENCHMARKS.md) | Not published as retrieval recall | No first-party benchmark (depends on your retriever) | Entirely dependent on your embedding + store choice |
| **End-to-end QA accuracy** | **Not independently measured** (harness in progress; recall is a *ceiling*, not a QA number) | Reports QA-style accuracy in its own publications | Reports QA-style accuracy in its own publications | — | — | — | — |
| **Retrieval latency** | ~7–10 ms, in-process, no network (benchmark-sized corpora) | Network round-trip if managed; self-host varies | Graph traversal; self-host varies; managed adds network | Fast without rerank; **seconds** when LLM rerank is on | Depends on backend + agent loop | Depends on the store | Depends on the store (local ms → managed network hop) |
| **Cost (API / tokens)** | **$0 for retrieval** — no API key, no tokens burned to search | LLM calls to extract facts on write; managed tier billed | LLM calls to build/maintain the graph; managed tier billed | Free without rerank; rerank spends LLM tokens | LLM-driven memory edits cost tokens | You pay for whatever LLM/store you wire in | Embedding calls + store hosting |
| **Write-time LLM calls** | ❌ None for `save`/`batch_save`/`supersede` (deterministic). Distillation & Dream *do* call an LLM but are **opt-in, manual, default-local** | ✅ Uses an LLM to extract structured facts on write | ✅ Uses an LLM to build the temporal graph | ❌ for storage (LLM only at optional rerank) | ✅ Self-editing memory is LLM-driven | 🟡 You choose (raw store = none; summarizers = yes) | ❌ raw store; 🟡 if you add an LLM extractor |
| **Privacy / data-leaves-machine** | ✅ Stays on your machine by default; retrieval never phones home. Off-machine only if *you* point Distill/Dream at a remote LLM (ERINYS warns) | 🟡 Self-host OSS keeps data local; managed sends it to their cloud | 🟡 Same: local with Graphiti, off-machine with Zep Cloud | ✅ Self-host; off-machine only at optional LLM rerank | 🟡 Self-host local; managed off-machine | 🟡 Depends entirely on which LLM/store you wire in | 🟡 Local store = local; managed store = off-machine |
| **Memory governance** (provenance / verified-forget / lineage / contradiction) | ✅ First-class: spoof-proof `provenance` on every write, `erinys_forget` with a zero-residue membership proof across DB substrates, `erinys_lineage` ancestry walk, supersede chains + contradiction detection | 🟡 Has memory update/deletion & history APIs; not a cryptographic-style deletion proof | 🟡 Temporal graph gives strong *history/validity* modelling; deletion-proof not the framing | ❌ Retrieval-focused; governance not the goal | 🟡 Memory is editable/inspectable but no verified-forget proof | ❌ Store primitives; governance is your job | ❌ Not provided — you build it |
| **Temporal-KG expressiveness** | 🟡 supersede chains + `valid_from/valid_until` + temporal boosts, but **temporal-inference is the weakest category (~76% R@5 on LoCoMo)**; not a full bi-temporal graph | 🟡 Fact-level history | ✅ **Strongest here** — Graphiti is a purpose-built bi-temporal knowledge graph | 🟡 Retrieval-time signals | 🟡 Agent-managed | 🟡 You model it | ❌ None by default |
| **Adoption / ecosystem maturity** | ❌ **Weakest here** — alpha (`Development Status :: 3 - Alpha`), ~5 GitHub stars, single MCP + CLI surface, no SDK spread | ✅ Large following, multi-language SDKs, integrations, managed platform | ✅ Established project, docs, managed offering, integrations | 🟡 Niche OSS, benchmark-oriented | ✅ Well-known (ex-MemGPT), active community, framework + platform | ✅ Rides the large LangChain/LangGraph ecosystem | ✅ Vector-DB ecosystems (pgvector/Chroma/Pinecone/…) are huge and mature |

Notes on the numbers:

- ERINYS's recall figures are on the LongMemEval **`_s`** split only; the harder
  `_m` split has **not** been run. See
  [BENCHMARKS.md](../benchmarks/BENCHMARKS.md) and
  [LIMITATIONS.md](LIMITATIONS.md).
- MemPalace figures are self-reported in its repository; the raw (no-rerank)
  numbers are the fair comparison to ERINYS's zero-LLM path. Its 100% LongMemEval-S
  and LoCoMo results involve LLM reranking (and, for LoCoMo, a `top_k` that
  exceeds the session count). Details in
  [BENCHMARKS.md → Landscape](../benchmarks/BENCHMARKS.md#landscape).
- For Mem0, Zep/Graphiti, Letta, LangGraph, and the RAG baseline, we deliberately
  do **not** print recall numbers on these splits, because we have no
  independently-run figures under identical conditions. Where those projects
  publish results, they are typically QA-style accuracy — a different metric.
  Inventing comparable recall numbers for them would be dishonest, so we describe
  them qualitatively instead.

---

## How to read this

The honest short version:

- If your bottleneck is **"where does this stuff run, what does it cost per
  query, and can I prove what I forgot?"** — ERINYS is built for that.
- If your bottleneck is **"give me a hosted service, SDKs, a dashboard, a
  temporal knowledge graph, and a community"** — the managed/graph systems are
  built for that, and ERINYS is not.

ERINYS is not trying to out-feature Mem0/Zep on hosted convenience or
out-express Graphiti on temporal graphs. It occupies a different point: the
**local, deterministic, verifiable** corner.

---

## Where ERINYS genuinely wins

- **Local-first, no API key.** The retrieval pipeline runs in-process on one
  SQLite file. Nothing leaves the machine to answer a query.
- **Zero-LLM deterministic retrieval.** FTS5 + sqlite-vec + RRF + algorithmic
  boosts. Same input → same output. No token cost, no rate limits, no
  provider outage in the read path. ~7–10 ms on benchmark-sized corpora.
- **Operational simplicity.** One file to back up, move, or delete. No server to
  operate, no vector-DB cluster, no control plane. `erinys doctor` diagnoses the
  environment and hands you a `fix` for each failing check.
- **The governance / trust layer.** Spoof-proof provenance on every write;
  `erinys_forget` deletes a memory *and its derived closure* and then proves zero
  residue across DB substrates; `erinys_lineage` walks ancestry; supersede chains
  preserve history; contradiction detection surfaces conflicts instead of
  silently overwriting. This is the part most memory systems don't try to do.

## Where ERINYS genuinely loses

- **Managed UX & scale.** No hosted service, no autoscaling, no multi-tenant
  control plane, no SLA. If you want someone else to run it, this is the wrong
  tool.
- **SDKs & dashboards.** The surface is an MCP server + a JSON CLI. There is no
  polished multi-language SDK suite or web dashboard like the larger platforms
  offer.
- **Ecosystem & adoption.** Alpha software, ~5 GitHub stars. Mem0, Letta, and the
  LangGraph ecosystem have far more integrations, docs, and community mileage.
- **Temporal-KG expressiveness.** Zep/Graphiti's purpose-built bi-temporal
  knowledge graph is more expressive than ERINYS's supersede chains +
  `valid_from/valid_until`. And temporal inference is ERINYS's *weakest*
  retrieval category (~76% R@5 on LoCoMo — see
  [LIMITATIONS.md](LIMITATIONS.md)).
- **End-to-end QA is unproven.** ERINYS reports recall, not answer accuracy. A
  QA harness is in progress; until it lands, treat recall as a ceiling.
- **Scale is unbenchmarked.** Published latencies are on benchmark-sized corpora;
  behaviour at 1M+ memories and under heavy write contention is not yet measured.

---

## When to choose ERINYS

- You want memory to run **locally / on-device / air-gapped**, with **no API key**
  and **no per-query token cost**.
- You need **fast, deterministic retrieval** and can't tolerate an LLM (latency,
  cost, non-determinism, or an outage) in the read path.
- You care about **memory governance**: knowing *who wrote* a memory, *what it was
  derived from*, being able to **prove** a memory (and its descendants) was
  deleted, and catching contradictions instead of silently overwriting.
- You value **one-file simplicity** — easy to embed, back up, ship, and reason
  about — over a managed control plane.
- You're embedding memory into your **own** stack and want a *trust layer*, not a
  platform to depend on.

## When *not* to choose ERINYS

- You want a **managed service** with a dashboard, SDKs, autoscaling, RBAC, and an
  SLA — pick a hosted platform (e.g. Mem0's or Zep's managed tier).
- You need a **rich temporal knowledge graph** with strong multi-hop time
  reasoning — **Zep/Graphiti** is purpose-built for that.
- Your team is already **standardized on the LangChain/LangGraph** stack and just
  wants memory primitives that drop into it — LangGraph's `Store`/checkpointers
  fit that path.
- You want a **framework that manages the whole agent loop** (self-editing memory,
  agent server, tools) rather than a focused memory engine — **Letta** is that.
- You need a **proven, high-adoption** system with a large community *today* —
  ERINYS is alpha; that maturity lives elsewhere.
- Your use case is **CJK-heavy and you require morphological analysis** — ERINYS
  relies on vector search for CJK (no morphological analyzer, no independent CJK
  benchmark yet; see [LIMITATIONS.md](LIMITATIONS.md)).

---

## Fair-comparison caveats

- **Recall ≠ QA.** Restating the top warning because it's the one people get
  wrong: ERINYS's headline numbers are retrieval recall. Do not compare them to
  another system's QA accuracy. See [LIMITATIONS.md](LIMITATIONS.md).
- **No invented competitor numbers.** Where we lack an independently-run figure
  under identical conditions, we describe the competitor qualitatively rather than
  print a number we can't stand behind.
- **Split matters.** ERINYS's LongMemEval result is `_s`-only. The `_m` split is
  not evaluated.
- **This is a snapshot.** These projects move fast. Check each project's own docs
  for current, first-party claims before making a decision.
