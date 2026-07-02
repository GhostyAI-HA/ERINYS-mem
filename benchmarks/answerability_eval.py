#!/usr/bin/env python3
"""
ERINYS answerability-signal validation harness
===============================================

``erinys_memory.search.assess_answerability(query, results)`` is a **zero-LLM**
"should I answer this, or abstain?" signal. Retrieval can always return *a*
nearest neighbour; that neighbour being close does not mean it *answers* the
question ("what is my hamster's name?" happily retrieves a memory about the
cat). ``assess_answerability`` looks at how much of the query's content is
actually *grounded* in the top result and returns ``answerable: False`` when the
answer is probably not in memory — a cue for the agent to abstain instead of
asserting a plausible-but-wrong fact.

The end-to-end ``qa_eval.py`` harness measures what an *LLM answerer* does with
retrieved context. This harness is narrower and needs **no LLM at all**: it
measures the retrieval-internal ``assess_answerability`` signal *in isolation*.
Given a labelled set of questions tagged answerable vs. unanswerable, it asks:
*how well does the abstain decision separate the two classes?*

--------------------------------------------------------------------------
What it reports
--------------------------------------------------------------------------
The decision under test is **"abstain"** (i.e. ``answerable == False``). We treat
*abstain* as the **positive** class, because on an abstention benchmark the
thing we want to detect is "this question has no answer in memory". So:

* **TP** — item is unanswerable (``should_abstain=True``) AND the signal abstains.
* **FP** — item is answerable but the signal abstains (over-cautious: it refused
  a question it could have answered).
* **FN** — item is unanswerable but the signal says answerable (dangerous: it
  would let the agent hallucinate an answer).
* **TN** — item is answerable AND the signal says answerable.

From the confusion matrix it computes **precision / recall / F1** of the abstain
decision plus **accuracy** and **abstain-recall** (a.k.a. sensitivity — the
fraction of truly-unanswerable questions the signal caught).

Because ``assess_answerability`` decides ``answerable`` by comparing its
``grounding`` cue against a single floor
(``search.ANSWERABILITY_MIN_GROUNDING``), we can **sweep** that floor for free:
we call ``assess_answerability`` once per item to obtain its ``grounding``, then
re-derive ``abstain = grounding < threshold`` for each candidate threshold. No
monkeypatching of the module constant is needed, and the sweep shows the
precision/recall trade-off and which threshold maximises F1.

--------------------------------------------------------------------------
Dataset schema (JSON list, or JSONL — one object per line)
--------------------------------------------------------------------------
Each item is an object::

    {
      "query": "what is my hamster called",         # required, str
      "should_abstain": true,                        # required, bool
      # Provide the retrieved candidates in ONE of two ways:
      "results": [                                   # (a) real result dicts
        {"content": "My cat Luna sleeps all day", "effective_score": 0.61},
        {"content": "I enjoy hiking",              "effective_score": 0.44}
      ],
      # ...or...
      "memories": ["My cat Luna sleeps all day", "I enjoy hiking"],  # (b) plain
      "id": "q_0001"                                 # optional, str
    }

* ``results`` — the exact list ERINYS retrieval returns (dicts with at least
  ``content``; ``effective_score`` feeds the margin cue). Pass this when you
  have run real retrieval and want to validate the signal on real neighbours.
* ``memories`` — a convenience for hand-written fixtures: a list of strings (or
  ``{"content": ..., "score": ...}`` dicts). They are wrapped into the result
  shape ``assess_answerability`` expects, with descending synthetic scores so
  the margin cue is well-defined.
* ``should_abstain`` — ground truth. ``True`` == the answer is NOT in the
  retrieved candidates (an "abstention" / unanswerable question). If omitted it
  is inferred ``True`` when there are no candidates.

--------------------------------------------------------------------------
Result schema (returned by :func:`evaluate`, written by ``--output``)
--------------------------------------------------------------------------
::

    {
      "summary": {
        "n_total": int,
        "n_should_abstain": int,          # ground-truth unanswerable count
        "n_should_answer": int,           # ground-truth answerable count
        "default_threshold": float,       # search.ANSWERABILITY_MIN_GROUNDING
        "at_default": {                   # metrics using the signal's own decision
          "threshold": float,
          "precision": float | None,      # None when TP+FP == 0
          "recall": float | None,         # abstain-recall; None when no positives
          "f1": float | None,
          "accuracy": float,
          "confusion": {"tp": int, "fp": int, "fn": int, "tn": int}
        },
        "best_f1": { ... same shape ... }  # the swept threshold with max F1
      },
      "sweep": [ { "threshold": float, "precision": ..., ... }, ... ],
      "items": [
        {
          "id": str, "query": str, "should_abstain": bool,
          "grounding": float,             # assess_answerability's grounding cue
          "score": float, "margin": float,
          "signal_answerable": bool,      # the function's own answerable verdict
          "signal_abstain": bool,         # == not signal_answerable (default thr)
          "reason": str
        }, ...
      ]
    }

--------------------------------------------------------------------------
Running for real against ConvoMem / LongMemEval-M abstention questions
--------------------------------------------------------------------------
This harness needs no dataset download for its unit test, but for a real run
point it at the abstention subset of a public benchmark:

* **LongMemEval / LongMemEval-M** marks abstention questions with a
  ``question_id`` ending in ``_abs``; their gold answer is a refusal of the form
  *"you did not mention ..."*. Convert each row to
  ``{"query": r["question"], "should_abstain": r["question_id"].endswith("_abs")}``
  and attach ``results`` = the top-K ERINYS actually retrieved for that question
  (run ``rrf_hybrid_search`` and pass the returned list straight through).
* **ConvoMem** ships explicit unanswerable / "not-in-memory" questions whose
  gold answer is *"you did not mention..."*; label those ``should_abstain=True``
  and the answerable questions ``False``, then attach the retrieved candidates.

A minimal converter (LongMemEval-M shown; ConvoMem is analogous)::

    import json
    from erinys_memory.search import rrf_hybrid_search
    from erinys_memory.embedding import embedding_engine as eng
    rows = json.load(open("longmemeval_m.json"))
    out = []
    for r in rows:
        q = r["question"]
        res = rrf_hybrid_search(db, q, eng.embed(q), project=proj, limit=10)
        out.append({
            "id": r["question_id"],
            "query": q,
            "should_abstain": str(r["question_id"]).endswith("_abs"),
            "results": res,   # dicts already carry content + effective_score
        })
    json.dump(out, open("/tmp/answerability_dataset.json", "w"))

then::

    python benchmarks/answerability_eval.py /tmp/answerability_dataset.json \
        --output benchmarks/results

The harness is decoupled from ``server.py``: it consumes already-retrieved
candidates, so it can validate the abstain signal over *any* retrieval strategy.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

# The signal under test lives in the installed package.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from erinys_memory.search import (  # noqa: E402
    ANSWERABILITY_MIN_GROUNDING,
    assess_answerability,
)

# --------------------------------------------------------------------------
# Dataset loading / normalisation
# --------------------------------------------------------------------------


def _coerce_results(raw: dict) -> list[dict[str, Any]]:
    """Turn a dataset item's candidate list into the result-dict shape that
    :func:`assess_answerability` consumes (each needs ``content`` and, ideally,
    ``effective_score``).

    Accepts either ``results`` (already result-shaped dicts, passed straight
    through) or ``memories`` (plain strings / ``{content, score}`` dicts, wrapped
    with descending synthetic scores so the margin cue is well-defined).
    """
    if raw.get("results") is not None:
        results = raw["results"]
        if not isinstance(results, (list, tuple)):
            raise ValueError(f"'results' must be a list: {raw!r}")
        out: list[dict[str, Any]] = []
        for r in results:
            if isinstance(r, dict):
                out.append(dict(r))
            else:
                out.append({"content": str(r)})
        return out

    memories = raw.get("memories")
    if memories is None:
        return []
    if not isinstance(memories, (list, tuple)):
        raise ValueError(f"'memories' must be a list: {raw!r}")

    out = []
    n = len(memories)
    for i, m in enumerate(memories):
        if isinstance(m, dict):
            item = dict(m)
            item.setdefault("content", "")
            # Keep an explicit score if given; else assign a descending default.
            if "effective_score" not in item:
                item["effective_score"] = float(item.get("score", n - i))
            out.append(item)
        else:
            # Descending scores (n, n-1, ...) give the top item a positive margin.
            out.append({"content": str(m), "effective_score": float(n - i)})
    return out


def normalize_item(raw: dict) -> dict:
    """Validate + normalise one dataset row.

    Raises ``ValueError`` on a missing/empty query so bad datasets fail loudly.
    ``should_abstain`` is inferred ``True`` when no candidates are supplied.
    """
    query = raw.get("query") or raw.get("question")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"dataset item missing non-empty 'query': {raw!r}")

    results = _coerce_results(raw)

    if "should_abstain" in raw:
        should_abstain = bool(raw["should_abstain"])
    elif "unanswerable" in raw:  # tolerate the qa_eval-style label
        should_abstain = bool(raw["unanswerable"])
    else:
        should_abstain = len(results) == 0

    return {
        "id": str(raw.get("id") or raw.get("question_id") or ""),
        "query": query.strip(),
        "results": results,
        "should_abstain": should_abstain,
    }


def load_dataset(path: str) -> list[dict]:
    """Load a labelled dataset from a ``.json`` (list) or ``.jsonl`` file."""
    text = Path(path).read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list):
        raise ValueError(f"dataset at {path} must be a JSON list or JSONL of objects")
    return [normalize_item(r) for r in rows]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


@dataclass
class ThresholdMetrics:
    """Precision/recall/F1/accuracy of the *abstain* decision at one threshold.

    Positive class == "abstain" (the signal says the answer is not in memory).
    ``precision`` / ``recall`` / ``f1`` are ``None`` when their denominator is
    zero (no predicted positives / no actual positives), never a crashy 0/0.
    """

    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> Optional[float]:
        denom = self.tp + self.fp
        return (self.tp / denom) if denom else None

    @property
    def recall(self) -> Optional[float]:
        denom = self.tp + self.fn
        return (self.tp / denom) if denom else None

    @property
    def f1(self) -> Optional[float]:
        p, r = self.precision, self.recall
        if p is None or r is None or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.fn + self.tn
        return ((self.tp + self.tn) / total) if total else 0.0

    def to_dict(self) -> dict:
        return {
            "threshold": round(self.threshold, 4),
            "precision": _round_opt(self.precision),
            "recall": _round_opt(self.recall),
            "f1": _round_opt(self.f1),
            "accuracy": round(self.accuracy, 4),
            "confusion": {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn},
        }


def _round_opt(value: Optional[float]) -> Optional[float]:
    return round(value, 4) if value is not None else None


def confusion_at_threshold(
    groundings: Sequence[float],
    labels: Sequence[bool],
    threshold: float,
) -> ThresholdMetrics:
    """Build the abstain-decision confusion matrix at one grounding ``threshold``.

    ``abstain`` is predicted when ``grounding < threshold`` (mirroring
    ``assess_answerability``, which sets ``answerable = grounding >= floor``).
    ``labels[i]`` is the ground-truth ``should_abstain`` for item ``i``.
    """
    tp = fp = fn = tn = 0
    for grounding, should_abstain in zip(groundings, labels):
        predict_abstain = grounding < threshold
        if should_abstain and predict_abstain:
            tp += 1
        elif not should_abstain and predict_abstain:
            fp += 1
        elif should_abstain and not predict_abstain:
            fn += 1
        else:
            tn += 1
    return ThresholdMetrics(threshold=threshold, tp=tp, fp=fp, fn=fn, tn=tn)


def _default_sweep_thresholds() -> list[float]:
    """Thresholds 0.0, 0.1, ... 1.0 (grounding is a fraction in [0, 1])."""
    return [round(i / 10, 4) for i in range(0, 11)]


# --------------------------------------------------------------------------
# Core evaluation
# --------------------------------------------------------------------------


def evaluate(
    dataset: Sequence[dict],
    *,
    thresholds: Optional[Sequence[float]] = None,
    assess_fn=assess_answerability,
) -> dict:
    """Score ``assess_answerability`` over a labelled dataset.

    ``dataset`` items are the normalised dicts from :func:`normalize_item` /
    :func:`load_dataset` (keys ``id, query, results, should_abstain``).

    ``assess_fn`` is injectable so tests can stub the signal; it defaults to the
    real :func:`erinys_memory.search.assess_answerability`. Returns the result
    dict documented in the module docstring.
    """
    if thresholds is None:
        thresholds = _default_sweep_thresholds()

    items: list[dict] = []
    groundings: list[float] = []
    labels: list[bool] = []
    n_should_abstain = 0

    for idx, item in enumerate(dataset):
        query = item["query"]
        results = item["results"]
        should_abstain = bool(item["should_abstain"])

        signal = assess_fn(query, results)
        grounding = float(signal.get("grounding", 0.0))
        signal_answerable = bool(signal.get("answerable", False))

        groundings.append(grounding)
        labels.append(should_abstain)
        if should_abstain:
            n_should_abstain += 1

        items.append({
            "id": item.get("id") or f"item_{idx}",
            "query": query,
            "should_abstain": should_abstain,
            "grounding": round(grounding, 4),
            "score": signal.get("score"),
            "margin": signal.get("margin"),
            "signal_answerable": signal_answerable,
            "signal_abstain": not signal_answerable,
            "reason": signal.get("reason"),
        })

    # Metrics at the signal's own operating point (its built-in floor). We derive
    # the confusion from the SIGNAL's actual answerable verdict, not a re-thresholded
    # grounding, so this row reflects exactly what the deployed signal decides.
    default_metrics = _metrics_from_signal_decisions(items, ANSWERABILITY_MIN_GROUNDING)

    # Threshold sweep over the grounding cue.
    sweep = [confusion_at_threshold(groundings, labels, t) for t in thresholds]

    # Pick the best sweep point by F1 (ties broken by higher recall, then acc).
    def _f1_key(m: ThresholdMetrics) -> tuple:
        return (m.f1 or -1.0, m.recall or -1.0, m.accuracy)

    best = max(sweep, key=_f1_key) if sweep else default_metrics

    summary = {
        "n_total": len(dataset),
        "n_should_abstain": n_should_abstain,
        "n_should_answer": len(dataset) - n_should_abstain,
        "default_threshold": ANSWERABILITY_MIN_GROUNDING,
        "at_default": default_metrics.to_dict(),
        "best_f1": best.to_dict(),
    }
    return {
        "summary": summary,
        "sweep": [m.to_dict() for m in sweep],
        "items": items,
    }


def _metrics_from_signal_decisions(items: Sequence[dict], threshold: float) -> ThresholdMetrics:
    """Confusion matrix built from each item's ``signal_abstain`` verdict.

    Unlike :func:`confusion_at_threshold` (which re-thresholds the raw grounding),
    this trusts the signal's own ``answerable`` decision, so ``at_default``
    reflects precisely what ``assess_answerability`` deployed today would do.
    """
    tp = fp = fn = tn = 0
    for it in items:
        should_abstain = bool(it["should_abstain"])
        predict_abstain = bool(it["signal_abstain"])
        if should_abstain and predict_abstain:
            tp += 1
        elif not should_abstain and predict_abstain:
            fp += 1
        elif should_abstain and not predict_abstain:
            fn += 1
        else:
            tn += 1
    return ThresholdMetrics(threshold=threshold, tp=tp, fp=fp, fn=fn, tn=tn)


def run_file(
    path: str,
    *,
    thresholds: Optional[Sequence[float]] = None,
) -> dict:
    """Load a dataset file and evaluate it. Convenience wrapper for the CLI."""
    dataset = load_dataset(path)
    return evaluate(dataset, thresholds=thresholds)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "ERINYS answerability-signal validation harness "
            "(precision/recall/F1 of the abstain decision, with a grounding sweep)."
        ),
    )
    parser.add_argument("dataset", help="Path to a .json (list) or .jsonl labelled dataset.")
    parser.add_argument(
        "--thresholds",
        default=None,
        help="Comma-separated grounding thresholds to sweep (default: 0.0..1.0 by 0.1).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Directory to write the full JSON result to (default: print summary only).",
    )
    return parser


def _parse_thresholds(spec: Optional[str]) -> Optional[list[float]]:
    if not spec:
        return None
    return [float(x) for x in spec.split(",") if x.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    thresholds = _parse_thresholds(args.thresholds)

    result = run_file(args.dataset, thresholds=thresholds)
    print(json.dumps(result["summary"], indent=2))

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"answerability_eval_{stamp}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nWrote full result to {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
