# Known Limitations & Scope

ERINYS is **alpha** software (`Development Status :: 3 - Alpha`). This page states
plainly what it does *not* yet do, and where its guarantees end. We would rather
you know the edges up front than discover them in production.

Run `erinys doctor` at any time to check your environment against the runtime
requirements below.

---

## Python requirements (loadable SQLite extensions)

ERINYS's vector index is [`sqlite-vec`](https://github.com/asg017/sqlite-vec), a
**loadable SQLite extension**. Loading it requires a Python whose `sqlite3`
module was built with `--enable-loadable-sqlite-extensions`.

| Python build | Extension support | Works out of the box |
|:--|:--|:--|
| python.org installers | ✅ | Yes |
| Homebrew `python@3.x` | ✅ | Yes |
| conda / miniconda | ✅ | Yes |
| GitHub Actions `setup-python` | ✅ | Yes |
| macOS system Python (`/usr/bin/python3`) | ❌ | No — see fallback |
| pyenv built without the flag | ❌ | No — see fallback |

**If your interpreter lacks support**, ERINYS raises a clear, actionable error
(not a cryptic `AttributeError`) and `erinys doctor` reports exactly what to fix.
Two remedies:

1. **Use a capable Python** — python.org, Homebrew, or conda builds all work.
2. **Install a fallback SQLite**: `pip install "erinys-memory[fallback]"`. This
   pulls `pysqlite3-binary`, which ERINYS auto-detects and uses in place of the
   stdlib. Prebuilt wheels are published mainly for Linux; on macOS/other
   platforms you may need to build `pysqlite3` from source, or simply use a
   capable Python (option 1).

Internally, one selector module (`erinys_memory._sqlite`) chooses the
implementation once so exception classes stay identical across the codebase.

---

## Benchmarks measure retrieval recall, not end-to-end QA

The headline numbers (LongMemEval-S R@5 100%, LoCoMo 94%, ConvoMem 97.6%) are
**retrieval recall** — did the correct evidence appear in the top-k results.
They are **not** end-to-end question-answering accuracy, which additionally
depends on the downstream LLM's reasoning over what was retrieved.

- **`_m` split not evaluated.** LongMemEval-S numbers are on the `longmemeval_s`
  split (~20 sessions/question). The harder `longmemeval_m` split has not been
  run. Do not read `_s` results as `_m` results.
- **E2E QA is in progress.** A fixed-LLM + fixed-retriever QA harness (accuracy /
  abstention / hallucination) is being added so recall and answer quality are
  reported separately. Until then, treat recall as a *ceiling* on answer quality,
  not a measure of it.
- See [benchmarks/BENCHMARKS.md](../benchmarks/BENCHMARKS.md) for exact
  conditions (top-k, embedding model, hardware, with/without adaptive boosting).

## Temporal inference is the weakest retrieval category

On LoCoMo, temporal-inference questions ("what happened *before* X?", ordering
across sessions) score materially lower than other categories (~76% R@5 vs 94%
overall). Relative-date parsing and `valid_from/valid_until` filtering help, but
multi-hop temporal ordering is still a known gap. A temporal reranker is planned.

## Abstention: "related but not the answer"

On ConvoMem, abstention is ~92%, meaning ~8% of unanswerable/only-loosely-related
questions still surface a plausible-but-incorrect memory. ERINYS does not yet
compute an answerability score, so a retrieved memory being *similar* is not
proof it *answers* the query. Downstream agents should not treat top-1 as
ground truth for questions that may have no answer.

---

## Verified forgetting: what is (and isn't) in scope

`erinys_forget` deletes a memory **and its derived closure** in one transaction,
then runs a membership test proving zero residue across every **in-database**
substrate: `observations`, `vec_observations`, FTS, `edges`, `collisions`.

**Out of scope of the deletion proof** (reported explicitly, not silently):

- **Obsidian vault exports** — cleared on the *next* export sweep, not inside the
  forget transaction. There is a window where an exported note still exists.
- **`.bak` / backup files** — retained by design (that is the point of a backup).
  A forgotten memory may still exist in an older backup you hold.
- **Anything you copied elsewhere** — logs, other databases, an LLM provider's
  training/caching if a memory was ever sent off-machine.

If you need a hard deletion guarantee for compliance, scope it to the ERINYS DB
and treat external copies (backups, exports) with your own retention policy.

## CJK / Japanese

FTS5's `porter` tokenizer does not segment CJK text well. ERINYS compensates by
routing CJK and mixed CJK+ASCII queries to **vector-heavy** retrieval, where the
multilingual embedding model outperforms keyword matching. This works in practice
(the default model is multilingual MiniLM), but there is **no independent
Japanese/CJK benchmark** yet, and no morphological analyzer — Japanese
named-entity and inflection handling rely entirely on the embedding model.

## LLM-dependent features are not "zero-LLM"

The **zero-LLM** claim applies to the **retrieval pipeline** (search). Two
higher-level features *do* call an LLM and are opt-in / manual:

- **Distillation** — generating abstract/meta layers from concrete facts.
- **Dream Cycle** — proposing connections between memory pairs.

By default these point at a **local** endpoint (Ollama). Pointing them at a
non-localhost endpoint means memory content leaves your machine — ERINYS warns
when you configure that, because it breaks the "memory never leaves your machine"
contract.

## Scale is not yet benchmarked

Published latencies (~7–10 ms) are on benchmark-sized corpora. Behaviour at
1M+ memories, cold start, WAL under many parallel readers, and write contention
has **not** been formally measured. A scale benchmark is planned. For very large
deployments, expect to tune `db_reader_pool_size`, pruning, and WAL checkpointing.

## Not a managed platform

ERINYS is a **local memory engine**, not a hosted service. There is no cloud API,
dashboard, multi-tenant control plane, RBAC, or SLA. `principal` / `project` /
`scope` give you namespacing primitives, but org/user/agent isolation, encryption
at rest, and retention policies are your responsibility to build around it. If you
need a managed platform, ERINYS is the wrong tool — it is the *local trust layer*
you embed in your own stack.
