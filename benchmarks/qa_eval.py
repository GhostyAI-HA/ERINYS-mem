#!/usr/bin/env python3
"""
ERINYS end-to-end QA evaluation harness (P0-4)
==============================================

Existing benchmarks (``longmemeval_bench.py``, ``locomo_bench.py``) measure
retrieval **recall**: did the right memory land in the top-K? They do NOT
measure whether an agent, handed that retrieved context, actually *answers the
question correctly* — or whether it hallucinates when the context does not
contain the answer.

This harness closes that gap. Given a dataset of

    {question, gold_answer, retrieved_context, unanswerable?}

it drives a **pluggable** answerer (``answer_fn``) over each item, then scores
the produced answers with a **pluggable** judge and reports three rates:

* **accuracy**       — over *answerable* items, fraction judged correct.
* **abstention_rate**— over *unanswerable* items, fraction where the agent
                       correctly abstained ("I don't know"). This is the
                       *good* behaviour on unanswerable questions.
* **hallucination_rate** — over *unanswerable* items, fraction where the agent
                       asserted a concrete answer instead of abstaining. This
                       is the *bad* behaviour on unanswerable questions.

Both the answerer and the judge are injected as parameters, so the whole
pipeline is unit-testable **without any real LLM** (see ``tests/test_qa_eval.py``).
The default answerer / judge talk to a local Ollama endpoint, matching the rest
of ERINYS (``ERINYS_DISTILL_ENDPOINT`` / ``ERINYS_DISTILL_MODEL``).

--------------------------------------------------------------------------
Dataset schema (JSON list, or JSONL — one object per line)
--------------------------------------------------------------------------
Each item is an object::

    {
      "question": "What is my dog's name?",          # required, str
      "gold_answer": "Rex",                            # required for answerable
      "retrieved_context": ["My dog Rex loves walks"], # required; str or list[str]
      "unanswerable": false,                           # optional, default false
      "id": "q_0001"                                   # optional, str
    }

* ``retrieved_context`` may be a single string or a list of context chunks
  (as produced by ERINYS retrieval). Lists are joined with blank lines.
* ``unanswerable: true`` marks an item whose answer is NOT in the context —
  the agent *should* abstain. ``gold_answer`` is ignored for these.
* If ``unanswerable`` is absent it is inferred ``true`` when ``gold_answer`` is
  empty/None, otherwise ``false``.

--------------------------------------------------------------------------
Result schema (returned by ``evaluate`` / written by ``--output``)
--------------------------------------------------------------------------
::

    {
      "summary": {
        "n_total": int,
        "n_answerable": int,
        "n_unanswerable": int,
        "accuracy": float | None,          # None if n_answerable == 0
        "abstention_rate": float | None,   # None if n_unanswerable == 0
        "hallucination_rate": float | None,# None if n_unanswerable == 0
        "n_correct": int,                  # answerable & judged correct
        "n_abstained_correctly": int,      # unanswerable & abstained
        "n_hallucinated": int              # unanswerable & asserted an answer
      },
      "items": [
        {
          "id": str,
          "question": str,
          "gold_answer": str | None,
          "unanswerable": bool,
          "answer": str,                   # what answer_fn produced
          "abstained": bool,               # judge said the answer is an abstention
          "correct": bool | None,          # answerable only; None otherwise
          "hallucinated": bool | None,     # unanswerable only; None otherwise
          "judge_reason": str | None       # optional judge rationale
        },
        ...
      ]
    }

--------------------------------------------------------------------------
Running for real against LongMemEval with Ollama
--------------------------------------------------------------------------
1. Start Ollama and pull a model (e.g. a small instruct model)::

       ollama serve &
       ollama pull llama3.1:8b

2. Point ERINYS at it (defaults already target localhost:11434)::

       export ERINYS_DISTILL_ENDPOINT=http://localhost:11434/api/generate
       export ERINYS_DISTILL_MODEL=llama3.1:8b

3. Produce a QA dataset. The retrieval benchmarks emit per-question
   ``retrieved_session_ids``; convert those into ``retrieved_context`` chunks
   plus the LongMemEval ``answer`` field as ``gold_answer``. LongMemEval marks
   unanswerable ("abstention") questions with a ``question_id`` ending in
   ``_abs`` — set ``unanswerable: true`` for those and leave ``gold_answer``
   empty. A minimal converter::

       import json
       rows = json.load(open("longmemeval_s.json"))
       out = []
       for r in rows:
           abs_q = str(r["question_id"]).endswith("_abs")
           out.append({
               "id": r["question_id"],
               "question": r["question"],
               "gold_answer": "" if abs_q else r.get("answer", ""),
               # In practice: replace with the sessions ERINYS actually
               # retrieved for this question (top-K), rendered to text.
               "retrieved_context": render_retrieved_sessions(r),
               "unanswerable": abs_q,
           })
       json.dump(out, open("/tmp/qa_dataset.json", "w"))

4. Run the harness (defaults use the Ollama answerer + LLM judge)::

       python benchmarks/qa_eval.py /tmp/qa_dataset.json \
           --output benchmarks/results

   Add ``--judge string`` to use the offline substring/token-overlap judge
   instead of a second LLM call (faster, no judge model needed).

The harness is intentionally decoupled from ``server.py`` / ``search.py``: it
consumes an already-retrieved context so it can score *any* retrieval strategy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

# --------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------

# answer_fn(question, context) -> answer string
AnswerFn = Callable[[str, str], str]


@dataclass
class JudgeResult:
    """Verdict for a single (question, answer) pair.

    * ``abstained``  — the answer is an "I don't know" style refusal.
    * ``correct``    — the answer matches the gold answer (only meaningful for
                       answerable items; ignored for unanswerable ones).
    * ``reason``     — optional free-text rationale (LLM judge fills this in).
    """

    abstained: bool
    correct: bool
    reason: Optional[str] = None


# judge_fn(question, answer, gold_answer, unanswerable) -> JudgeResult
JudgeFn = Callable[[str, str, Optional[str], bool], JudgeResult]


# --------------------------------------------------------------------------
# Context rendering + dataset loading
# --------------------------------------------------------------------------

def render_context(retrieved_context: object) -> str:
    """Normalise ``retrieved_context`` (str or list of str) into one string."""
    if retrieved_context is None:
        return ""
    if isinstance(retrieved_context, str):
        return retrieved_context.strip()
    if isinstance(retrieved_context, (list, tuple)):
        parts = [str(chunk).strip() for chunk in retrieved_context if str(chunk).strip()]
        return "\n\n".join(parts)
    return str(retrieved_context).strip()


def normalize_item(raw: dict) -> dict:
    """Validate + normalise one dataset row into the internal item shape.

    Raises ``ValueError`` on a missing question so bad datasets fail loudly
    rather than silently scoring garbage.
    """
    question = raw.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"dataset item missing non-empty 'question': {raw!r}")

    gold_answer = raw.get("gold_answer")
    if gold_answer is not None:
        gold_answer = str(gold_answer)

    # Infer unanswerable when not explicitly provided: empty/None gold_answer.
    if "unanswerable" in raw:
        unanswerable = bool(raw["unanswerable"])
    else:
        unanswerable = not (gold_answer and gold_answer.strip())

    return {
        "id": str(raw.get("id") or raw.get("question_id") or ""),
        "question": question.strip(),
        "gold_answer": gold_answer,
        "context": render_context(raw.get("retrieved_context")),
        "unanswerable": unanswerable,
    }


def load_dataset(path: str) -> list[dict]:
    """Load a QA dataset from a ``.json`` (list) or ``.jsonl`` file."""
    text = Path(path).read_text(encoding="utf-8")
    stripped = text.lstrip()
    rows: list[dict]
    if stripped.startswith("["):
        rows = json.loads(text)
    else:
        # JSONL: one object per non-empty line.
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list):
        raise ValueError(f"dataset at {path} must be a JSON list or JSONL of objects")
    return [normalize_item(r) for r in rows]


# --------------------------------------------------------------------------
# Abstention detection (used by the offline judge and by hallucination scoring)
# --------------------------------------------------------------------------

_ABSTENTION_PHRASES = (
    "i don't know",
    "i do not know",
    "i dont know",
    "not sure",
    "no information",
    "not enough information",
    "insufficient information",
    "cannot answer",
    "can't answer",
    "cannot determine",
    "unable to answer",
    "unable to determine",
    "the context does not",
    "context doesn't",
    "not mentioned",
    "not provided",
    "no relevant",
    "unknown",
)


def looks_like_abstention(answer: str) -> bool:
    """Heuristic: does ``answer`` read as an "I don't know" refusal?

    Deliberately conservative — it matches explicit refusal phrasing rather
    than trying to judge correctness. Used as the default abstention signal
    when a judge does not report one itself.
    """
    if answer is None:
        return True
    text = answer.strip().lower()
    if not text:
        return True
    return any(phrase in text for phrase in _ABSTENTION_PHRASES)


# --------------------------------------------------------------------------
# Default answerer: local Ollama
# --------------------------------------------------------------------------

_ANSWER_PROMPT = """You are a precise question-answering assistant. Answer the \
question using ONLY the context below. If the context does not contain the \
answer, reply exactly with: I don't know.

Context:
{context}

Question: {question}

Answer:"""


def _ollama_generate(prompt: str, *, endpoint: str, model: str, timeout: int) -> str:
    """Single-shot generation against an Ollama ``/api/generate`` endpoint.

    Mirrors ``erinys_memory.distill._llm_generate`` (same request shape,
    ``stream: false``, urllib, one retry) so the harness inherits ERINYS's
    "stays on localhost" posture.
    """
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
    }).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    last_err: Optional[Exception] = None
    for _ in range(2):
        try:
            resp = urllib.request.urlopen(request, timeout=timeout)
            payload = json.loads(resp.read().decode("utf-8"))
            return str(payload.get("response", "")).strip()
        except (urllib.error.URLError, json.JSONDecodeError, OSError, KeyError) as exc:
            last_err = exc
    raise RuntimeError(f"Ollama generation failed: {last_err}")


def make_ollama_answer_fn(
    *,
    endpoint: Optional[str] = None,
    model: Optional[str] = None,
    timeout: int = 60,
) -> AnswerFn:
    """Build the default context-grounded answerer backed by local Ollama."""
    endpoint = endpoint or os.environ.get(
        "ERINYS_DISTILL_ENDPOINT", "http://localhost:11434/api/generate"
    )
    model = model or os.environ.get("ERINYS_DISTILL_MODEL", "gemma4:e4b")

    def answer_fn(question: str, context: str) -> str:
        prompt = _ANSWER_PROMPT.format(context=context or "(no context)", question=question)
        return _ollama_generate(prompt, endpoint=endpoint, model=model, timeout=timeout)

    return answer_fn


# --------------------------------------------------------------------------
# Judges
# --------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    return {t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t}


def string_match_judge(
    question: str,
    answer: str,
    gold_answer: Optional[str],
    unanswerable: bool,
) -> JudgeResult:
    """Offline judge — no LLM. Substring + token-overlap match against gold.

    * ``abstained`` uses :func:`looks_like_abstention`.
    * ``correct`` (answerable only) is true when the gold answer is contained
      in the answer (case-insensitive) OR every gold token appears in the
      answer's tokens. This is a *containment* check, robust to the model
      wrapping the fact in a sentence ("Your dog's name is Rex." vs "Rex").
    """
    abstained = looks_like_abstention(answer)
    if unanswerable or not gold_answer or not gold_answer.strip():
        # Correctness is undefined for unanswerable items.
        return JudgeResult(abstained=abstained, correct=False, reason="unanswerable/no-gold")
    if abstained:
        return JudgeResult(abstained=True, correct=False, reason="abstained on answerable")

    ans_lower = answer.lower()
    gold_lower = gold_answer.strip().lower()
    if gold_lower and gold_lower in ans_lower:
        return JudgeResult(abstained=False, correct=True, reason="substring match")

    gold_tokens = _tokenize(gold_answer)
    ans_tokens = _tokenize(answer)
    if gold_tokens and gold_tokens.issubset(ans_tokens):
        return JudgeResult(abstained=False, correct=True, reason="token-subset match")

    return JudgeResult(abstained=False, correct=False, reason="no match")


_JUDGE_PROMPT = """You are grading a question-answering system. Decide two \
things about the SYSTEM ANSWER.

Question: {question}
Gold answer: {gold}
System answer: {answer}

Reply with a JSON object with exactly these keys:
- "abstained": true if the system answer is an "I don't know" style refusal, \
otherwise false.
- "correct": true if the system answer conveys the same fact as the gold \
answer, otherwise false. If the system abstained, "correct" is false.

JSON:"""


def make_ollama_judge_fn(
    *,
    endpoint: Optional[str] = None,
    model: Optional[str] = None,
    timeout: int = 60,
) -> JudgeFn:
    """Build the default LLM judge backed by local Ollama (JSON output)."""
    endpoint = endpoint or os.environ.get(
        "ERINYS_DISTILL_ENDPOINT", "http://localhost:11434/api/generate"
    )
    model = model or os.environ.get("ERINYS_DISTILL_MODEL", "gemma4:e4b")

    def judge_fn(
        question: str,
        answer: str,
        gold_answer: Optional[str],
        unanswerable: bool,
    ) -> JudgeResult:
        # For unanswerable items there is no gold fact to match against; the
        # only thing that matters is whether the model abstained, which the
        # offline detector already handles reliably and cheaply.
        if unanswerable:
            return JudgeResult(
                abstained=looks_like_abstention(answer),
                correct=False,
                reason="unanswerable (abstention checked offline)",
            )
        prompt = _JUDGE_PROMPT.format(
            question=question, gold=gold_answer or "", answer=answer
        )
        body = json.dumps({
            "model": model, "prompt": prompt, "stream": False, "format": "json",
        }).encode("utf-8")
        request = urllib.request.Request(
            endpoint, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            resp = urllib.request.urlopen(request, timeout=timeout)
            payload = json.loads(resp.read().decode("utf-8"))
            parsed = json.loads(payload.get("response", "{}"))
            abstained = bool(parsed.get("abstained", False))
            correct = bool(parsed.get("correct", False)) and not abstained
            return JudgeResult(abstained=abstained, correct=correct, reason="llm-judge")
        except (urllib.error.URLError, json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
            # Fall back to the offline judge so a flaky judge model does not
            # crash the whole run.
            return string_match_judge(question, answer, gold_answer, unanswerable)

    return judge_fn


# --------------------------------------------------------------------------
# Core evaluation
# --------------------------------------------------------------------------

@dataclass
class QAEvalConfig:
    """Knobs for a run. All optional; sensible defaults for offline scoring."""

    answer_fn: Optional[AnswerFn] = None
    judge_fn: Optional[JudgeFn] = None
    progress: bool = False


def evaluate(
    dataset: Sequence[dict],
    answer_fn: AnswerFn,
    judge_fn: JudgeFn,
    *,
    progress: bool = False,
) -> dict:
    """Run the full QA eval over ``dataset``.

    ``dataset`` items are the *normalised* dicts produced by
    :func:`normalize_item` / :func:`load_dataset`, i.e. each has keys
    ``id, question, gold_answer, context, unanswerable``.

    Returns the result dict documented in the module docstring.
    """
    items: list[dict] = []
    n_correct = 0
    n_answerable = 0
    n_unanswerable = 0
    n_abstained_correctly = 0
    n_hallucinated = 0

    for idx, item in enumerate(dataset):
        question = item["question"]
        context = item["context"]
        gold = item["gold_answer"]
        unanswerable = bool(item["unanswerable"])

        answer = answer_fn(question, context)
        verdict = judge_fn(question, answer, gold, unanswerable)

        record: dict = {
            "id": item.get("id") or f"item_{idx}",
            "question": question,
            "gold_answer": gold,
            "unanswerable": unanswerable,
            "answer": answer,
            "abstained": bool(verdict.abstained),
            "judge_reason": verdict.reason,
        }

        if unanswerable:
            n_unanswerable += 1
            if verdict.abstained:
                n_abstained_correctly += 1
                record["hallucinated"] = False
            else:
                n_hallucinated += 1
                record["hallucinated"] = True
            record["correct"] = None
        else:
            n_answerable += 1
            correct = bool(verdict.correct) and not verdict.abstained
            if correct:
                n_correct += 1
            record["correct"] = correct
            record["hallucinated"] = None

        items.append(record)

        if progress:
            print(f"  [{idx + 1}/{len(dataset)}] {record['id']}", file=sys.stderr)

    summary = {
        "n_total": len(dataset),
        "n_answerable": n_answerable,
        "n_unanswerable": n_unanswerable,
        "accuracy": (n_correct / n_answerable) if n_answerable else None,
        "abstention_rate": (n_abstained_correctly / n_unanswerable) if n_unanswerable else None,
        "hallucination_rate": (n_hallucinated / n_unanswerable) if n_unanswerable else None,
        "n_correct": n_correct,
        "n_abstained_correctly": n_abstained_correctly,
        "n_hallucinated": n_hallucinated,
    }
    return {"summary": summary, "items": items}


def run_file(
    path: str,
    *,
    answer_fn: Optional[AnswerFn] = None,
    judge_fn: Optional[JudgeFn] = None,
    limit: Optional[int] = None,
    progress: bool = False,
) -> dict:
    """Load a dataset file and evaluate it. Convenience wrapper for the CLI."""
    dataset = load_dataset(path)
    if limit is not None:
        dataset = dataset[:limit]
    if answer_fn is None:
        answer_fn = make_ollama_answer_fn()
    if judge_fn is None:
        judge_fn = make_ollama_judge_fn()
    return evaluate(dataset, answer_fn, judge_fn, progress=progress)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ERINYS end-to-end QA evaluation harness (accuracy / abstention / hallucination).",
    )
    parser.add_argument("dataset", help="Path to a .json (list) or .jsonl QA dataset.")
    parser.add_argument(
        "--judge",
        choices=["llm", "string"],
        default="llm",
        help="Judge: 'llm' (default, Ollama) or 'string' (offline substring/token match).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N items.")
    parser.add_argument(
        "--output",
        default=None,
        help="Directory to write the JSON result to (default: print summary only).",
    )
    parser.add_argument(
        "--endpoint", default=None, help="Override Ollama endpoint (else ERINYS_DISTILL_ENDPOINT)."
    )
    parser.add_argument(
        "--model", default=None, help="Override Ollama model (else ERINYS_DISTILL_MODEL)."
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable per-item progress.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    answer_fn = make_ollama_answer_fn(endpoint=args.endpoint, model=args.model)
    if args.judge == "string":
        judge_fn: JudgeFn = string_match_judge
    else:
        judge_fn = make_ollama_judge_fn(endpoint=args.endpoint, model=args.model)

    result = run_file(
        args.dataset,
        answer_fn=answer_fn,
        judge_fn=judge_fn,
        limit=args.limit,
        progress=not args.no_progress,
    )

    summary = result["summary"]
    print(json.dumps(summary, indent=2))

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"qa_eval_{stamp}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nWrote full result to {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
