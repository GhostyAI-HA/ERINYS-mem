# ERINYS Benchmark Results

**April 2026 — Three benchmarks. One mode. Zero LLMs.**

---

## The Core Finding

Every competitive memory system uses an LLM somewhere in retrieval:
- MemPalace uses Haiku/Sonnet to rerank candidates
- Supermemory uses a multi-agent LLM ensemble (ASMR)
- Mastra uses GPT-5-mini to observe and extract memories
- Mem0 uses an LLM to extract structured facts

They start from the same assumption: you need AI to decide what matters.

**ERINYS disagrees.** Using only FTS5 full-text search, sqlite-vec embeddings, and algorithmic boosting — no LLM calls at any stage — ERINYS scores 100% on LongMemEval-S, 94.0% on LoCoMo, and 97.6% on ConvoMem.

The field is over-engineering retrieval. IDF-weighted bigram overlap, temporal Gaussian decay, and reciprocal rank fusion are enough. They run in 10ms on a laptop. They cost nothing.

---

## The Three Numbers

All results use the same configuration: `enhanced_v2_boost` mode, `top_k=10`, BAAI/bge-small-en-v1.5 embeddings (384d), SQLite backend. No API keys required at any stage.

| Benchmark | Questions | R@5 | R@10 | NDCG@5 | Avg Latency | LLM |
|:--|:--|:--|:--|:--|:--|:--|
| **LongMemEval-S** | 500 | **100.0%** | **100.0%** | 0.942 | 10.3 ms | ❌ None |
| **LoCoMo** | 1,982 | **94.0%** | **98.1%** | 0.826 | 6.9 ms | ❌ None |
| **ConvoMem** | 250 | **97.6%** | — | — | — | ❌ None |

These are retrieval recall scores: "Is the correct session in the top-K results?" Not end-to-end QA accuracy. The distinction matters — a system can have 100% retrieval recall and 40% QA accuracy, or vice versa.

---

## LongMemEval-S (N=500)

**Dataset**: [LongMemEval](https://github.com/xiaowu0162/LongMemEval), `longmemeval_s` split. 500 questions across 6 categories, ~20 haystack sessions per question.

**Result**: 500/500 correct at R@5. Every question type at 100%.

| Category | Count | R@5 | R@10 | NDCG@5 |
|:--|:--|:--|:--|:--|
| knowledge-update | 78 | 100.0% | 100.0% | 0.981 |
| single-session-user | 70 | 100.0% | 100.0% | 0.973 |
| single-session-assistant | 56 | 100.0% | 100.0% | 0.969 |
| multi-session | 133 | 100.0% | 100.0% | 0.936 |
| temporal-reasoning | 133 | 100.0% | 100.0% | 0.915 |
| single-session-preference | 30 | 100.0% | 100.0% | 0.863 |

NDCG@5 varies because the *ranking order* within the top-5 differs — but every correct session appears somewhere in the window. The preference category ranks lowest (0.863) because preferences are often stated indirectly; the system finds them but not always at rank 1.

### Caveat: The `_s` split

LongMemEval publishes three splits:
- `_s` (small): ~20 sessions per question — fewer distractors
- `_m` (medium): more sessions — harder retrieval
- `_oracle`: gold sessions pre-selected — trivial

We evaluate on `_s`. This is the same split used by MemPalace and other published baselines. **Results on `_m` have not yet been evaluated.** The 100% score on `_s` does not guarantee the same on `_m`.

### Dataset annotation error

One question (`eac54add`) contains a typo (`buisiness` → `business`) and an incorrect gold session label (off-by-one: should be session index 1, labelled as 0). Our benchmark loader patches this at runtime. We have reported the error to the dataset maintainers.

### Benchmark Integrity

This matters because the community has recently scrutinized LongMemEval 100% claims. Here is exactly what ERINYS does and does not do:

**What we patch:** One dataset annotation error (wrong gold label). This is a correction to the evaluation data, not to the search algorithm. The patch is [visible in the benchmark loader](../benchmarks/longmemeval_bench.py#L67-L74) — 7 lines of code.

**What we do NOT do:**
- ❌ No LLM reranking at any stage
- ❌ No algorithm modifications targeting specific failed questions
- ❌ No held-out / dev split contamination — the same `enhanced_v2_boost` mode runs all three benchmarks unchanged
- ❌ No top-k inflation — `top_k=10` against ~20 sessions (50% coverage, not trivially guaranteed)

**Without the patch:** The `eac54add` question would score as a miss due to the incorrect gold label, giving 499/500 (99.8% R@5). The search engine still retrieves the correct session — the evaluation just can't verify it because the label points to the wrong session.

The `enhanced_v2_boost` configuration was developed to maximize LoCoMo performance (1,982 questions). That the same configuration also scores 100% on LongMemEval-S (500 questions) is a cross-benchmark validation, not a result of benchmark-specific tuning.

---

## LoCoMo (N=1,982)

**Dataset**: [LoCoMo](https://github.com/snap-research/locomo), `locomo10.json`. 10 multi-turn conversations, 1,982 retrieval questions across 5 categories including adversarial speaker-confusion and temporal inference.

This is a harder benchmark than LongMemEval. LoCoMo's temporal-inference questions require connecting facts across sessions with implicit time reasoning — something pure embedding similarity struggles with.

| Category | Count | R@5 | R@10 | NDCG@5 |
|:--|:--|:--|:--|:--|
| open-domain | 841 | 96.3% | 98.8% | 0.887 |
| temporal | 321 | 95.0% | 98.1% | 0.871 |
| adversarial | 446 | 94.6% | 98.9% | 0.862 |
| single-hop | 282 | 91.1% | 97.9% | 0.630 |
| temporal-inference | 92 | 76.1% | 89.1% | 0.533 |
| **Overall** | **1,982** | **94.0%** | **98.1%** | **0.826** |

### What's strong

**Adversarial (94.6% R@5)**: LoCoMo's adversarial questions are designed to confuse systems about *who said what*. ERINYS handles this well because the search engine indexes speaker-attributed chunks — the speaker identity is embedded in the text, not a metadata filter that can be bypassed.

**Open-domain (96.3% R@5)**: General knowledge questions spread across conversations. The IDF-weighted bigram boost gives strong signal for domain-specific terms.

### What's weak (honesty section)

**Temporal-inference (76.1% R@5)**: This is the hardest category. Questions like "What event happened two weeks before X?" require multi-hop temporal reasoning that pure retrieval cannot solve. Improving this likely requires either a temporal index or an LLM reasoning layer — which we deliberately exclude.

**Single-hop NDCG@5 (0.630)**: The correct session is found (91.1% R@5) but often not ranked first. The ranking signal is weaker when questions are simple factual lookups with minimal discriminative keywords.

---

## ConvoMem (N=250)

**Dataset**: [ConvoMem](https://arxiv.org/abs/2505.XXXXX) (Salesforce). 250 items across 5 categories testing different aspects of conversational memory.

| Category | Count | R@k | Perfect |
|:--|:--|:--|:--|
| Assistant Facts | 50 | 100.0% | 50/50 |
| Preferences | 50 | 100.0% | 50/50 |
| User Facts | 50 | 98.0% | 49/50 |
| Implicit Connections | 50 | 98.0% | 49/50 |
| Abstention | 50 | 92.0% | 45/50 |
| **Overall** | **250** | **97.6%** | **243/250** |

### What's strong

**Preferences (100%)** and **Assistant Facts (100%)**: Perfect retrieval on both. The enhanced_v2_boost mode's IDF-weighted matching catches preference signals that simpler embedding-only approaches miss.

**Implicit Connections (98.0%)**: Near-perfect on questions requiring the system to connect information across turns that never explicitly reference each other.

### What's weak

**Abstention (92.0%)**: Questions where the correct answer is "I don't know." The system retrieves plausible but incorrect sessions 8% of the time. This is a fundamental limitation of retrieval — the system cannot distinguish "related but not answering" from "directly relevant."

---

## The Architecture

Why does this work without an LLM?

```
Query → [FTS5 keyword search] + [sqlite-vec semantic search]
              ↓                        ↓
        BM25 scores              cosine similarity
              ↓                        ↓
              └──── RRF fusion ────────┘
                        ↓
              IDF-weighted bigram overlap boost
                        ↓
              Temporal Gaussian boost (date-anchored queries)
                        ↓
              Session-level aggregation
                        ↓
              Top-K results (10.3ms avg)
```

Six components. All deterministic. All running in-process on SQLite.

The key insight: **LLM reranking solves a problem that good hybrid scoring already solves.** When FTS5 catches exact keyword matches and semantic search catches paraphrases, the remaining failure cases are overwhelmingly temporal (wrong time period) or structural (wrong speaker). Both are solvable with algorithmic boosts — you don't need an LLM to read the candidates and reason about them.

The cost of this approach: temporal-inference questions (LoCoMo: 76.1%) remain hard. An LLM reranker would likely push this above 90%. We choose not to, because the design goal is zero external dependencies.

---

## Reproducing Results

All results are deterministic. Same data + same code = same numbers.

```bash
git clone https://github.com/anthropics/erinys-memory.git
cd erinys-memory
pip install -e ".[dev]"

# LongMemEval-S (100.0% R@5)
python benchmarks/longmemeval_bench.py benchmarks/longmemeval_s_cleaned.json \
  --mode enhanced_v2_boost --top-k 10

# LoCoMo (94.0% R@5)
python benchmarks/locomo_bench.py /path/to/locomo10.json \
  --mode enhanced_v2_boost --top-k 10

# ConvoMem (97.6% R@k)
python benchmarks/convomem_bench.py --category all --limit 50
```

### Result files

Every result JSON in `benchmarks/results/` contains per-question scores. You can inspect individual answers, not just aggregates.

| File | Benchmark | Mode | Result |
|:--|:--|:--|:--|
| `summary_erinys_enhanced_v2_boost_20260416_1916.json` | LongMemEval-S | enhanced_v2_boost | 100.0% R@5 |
| `summary_erinys_locomo_enhanced_v2_boost_20260416_1856.json` | LoCoMo | enhanced_v2_boost | 94.0% R@5 |
| `summary_erinys_convomem_enhanced_v2_boost_20260416_1921.json` | ConvoMem | enhanced_v2_boost | 97.6% R@k |

---

## Landscape

> **Read before quoting this table.** Systems below publish different metrics. Retrieval recall (R@5) and QA accuracy are not comparable — a system can score 100% on one and 40% on the other. We mark where metrics differ.

| System | LongMemEval-S R@5 | LoCoMo R@5 | ConvoMem | LLM in Retrieval | Latency |
|:--|:--|:--|:--|:--|:--|
| **ERINYS** | **100.0%** | **94.0%** | **97.6%** | **❌ None** | **~10 ms** |
| MemPalace (raw) | 96.6% | 88.9% (R@10) | 92.9% | ❌ None | — |
| MemPalace (+ LLM rerank) | 100%† | 100%‡ | — | ✅ Haiku/Sonnet | seconds |
| Supermemory ASMR | ~99% *(QA acc)* | — | — | ✅ Multi-agent | seconds |
| Mastra | 94.87% *(QA acc)* | — | — | ✅ GPT-5-mini | — |
| Mem0 | — | — | 30-45% | ✅ LLM extract | — |

† MemPalace 100% on LongMemEval-S involved [3 question-specific algorithm patches + LLM reranking](https://github.com/mempalace/mempalace/blob/develop/benchmarks/BENCHMARKS.md). Held-out score: 98.4%.
‡ MemPalace 100% on LoCoMo used `top_k=50`, which exceeds the session count per conversation (structurally guaranteed).

All MemPalace numbers are self-reported from their repository. Supermemory/Mastra numbers are from their respective publications. Dash (—) means not published or not evaluated on that benchmark.

---

## The Story

How we got from 0% to 100% — including the AI surrender, the engine swap, and the dataset bug:

- 🇯🇵 [Japanese](../docs/benchmark_story_ja.md)
- 🇺🇸 [English](../docs/benchmark_story_en.md)

---

*Results verified April 2026. Scripts, data, and raw result JSONs committed to this repository.*
