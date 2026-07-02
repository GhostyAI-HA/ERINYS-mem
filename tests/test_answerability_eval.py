"""Unit tests for the answerability-signal validation harness
(``benchmarks/answerability_eval.py``).

These tests prove — WITHOUT any real LLM, retrieval, or dataset download — that:

* the harness computes precision / recall / F1 / accuracy of the *abstain*
  decision correctly from a known confusion matrix;
* the real ``assess_answerability`` signal actually *separates* answerable from
  unanswerable questions on a tiny in-code fixture (abstain-recall > 0 on the
  unanswerable items, and it does not abstain on the answerable ones);
* the grounding-threshold sweep is well-formed and rate denominators never
  divide by zero (``None`` instead of a crash when a class is empty).

The fixture supplies retrieved candidates inline (as ``memories`` / ``results``),
so no ConvoMem / LongMemEval download is required — see the module docstring of
``answerability_eval.py`` for how to point it at those for a real run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# The harness lives under benchmarks/, not the installed package.
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))

import answerability_eval  # noqa: E402
from answerability_eval import (  # noqa: E402
    ThresholdMetrics,
    confusion_at_threshold,
    evaluate,
    load_dataset,
    normalize_item,
    run_file,
)


# --------------------------------------------------------------------------
# Fixture: 2 answerable + 2 unanswerable questions with inline candidates.
#
# The query asks about a hamster. On the answerable items the top candidate
# grounds the query's content keywords ("hamster" appears); on the unanswerable
# items the memory is topically near (a pet) but mentions no hamster, so the
# grounding cue is low and the signal should abstain.
# --------------------------------------------------------------------------

def _fixture_dataset() -> list[dict]:
    return [
        normalize_item({
            "id": "ans_1",
            "query": "what is my hamster called",
            "should_abstain": False,
            "memories": [
                "My hamster is called Nibbles and loves his exercise wheel",
                "I went hiking at Muir Woods last weekend",
            ],
        }),
        normalize_item({
            "id": "ans_2",
            "query": "where does my hamster sleep",
            "should_abstain": False,
            "memories": [
                "My hamster sleeps in a cozy nest inside the plastic tube",
                "The weather was rainy on Tuesday",
            ],
        }),
        normalize_item({
            "id": "abs_1",
            "query": "what is my hamster called",
            "should_abstain": True,
            "memories": [
                "My cat Luna sleeps all day on the couch",
                "I enjoy hiking on the weekends",
            ],
        }),
        normalize_item({
            "id": "abs_2",
            "query": "what breed is my hamster",
            "should_abstain": True,
            "memories": [
                "My dog Rex loves long walks in the park",
                "I bought a new coffee machine last month",
            ],
        }),
    ]


# --------------------------------------------------------------------------
# normalize_item / _coerce_results / load_dataset
# --------------------------------------------------------------------------

def test_normalize_item_wraps_memories_with_descending_scores() -> None:
    item = normalize_item({
        "query": "q?", "should_abstain": False,
        "memories": ["first", "second", "third"],
    })
    scores = [r["effective_score"] for r in item["results"]]
    assert scores == [3.0, 2.0, 1.0]  # descending -> top item has positive margin
    assert item["results"][0]["content"] == "first"


def test_normalize_item_passes_results_through() -> None:
    item = normalize_item({
        "query": "q?", "should_abstain": True,
        "results": [{"content": "c", "effective_score": 0.9}],
    })
    assert item["results"] == [{"content": "c", "effective_score": 0.9}]


def test_normalize_item_infers_should_abstain_when_no_candidates() -> None:
    item = normalize_item({"query": "q?"})
    assert item["results"] == []
    assert item["should_abstain"] is True


def test_normalize_item_requires_query() -> None:
    with pytest.raises(ValueError):
        normalize_item({"should_abstain": True, "memories": ["x"]})


def test_load_dataset_json_and_jsonl(tmp_path: Path) -> None:
    rows = [
        {"query": "q1?", "should_abstain": False, "memories": ["a1"]},
        {"query": "q2?", "should_abstain": True, "memories": ["a2"]},
    ]
    json_path = tmp_path / "data.json"
    json_path.write_text(json.dumps(rows), encoding="utf-8")
    loaded = load_dataset(str(json_path))
    assert len(loaded) == 2
    assert loaded[1]["should_abstain"] is True

    jsonl_path = tmp_path / "data.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    loaded_l = load_dataset(str(jsonl_path))
    assert len(loaded_l) == 2
    assert loaded_l[0]["query"] == "q1?"


# --------------------------------------------------------------------------
# Metrics: precision / recall / F1 / accuracy from a known confusion matrix
# --------------------------------------------------------------------------

def test_threshold_metrics_precision_recall_f1() -> None:
    # tp=3, fp=1, fn=1, tn=5
    m = ThresholdMetrics(threshold=0.5, tp=3, fp=1, fn=1, tn=5)
    assert m.precision == pytest.approx(3 / 4)
    assert m.recall == pytest.approx(3 / 4)
    assert m.f1 == pytest.approx(0.75)
    assert m.accuracy == pytest.approx(8 / 10)


def test_threshold_metrics_none_when_denominator_zero() -> None:
    # No predicted positives -> precision undefined; no actual positives -> recall undefined.
    m = ThresholdMetrics(threshold=1.0, tp=0, fp=0, fn=0, tn=4)
    assert m.precision is None
    assert m.recall is None
    assert m.f1 is None
    assert m.accuracy == 1.0


def test_confusion_at_threshold_counts_correctly() -> None:
    # grounding: high = answerable, low = abstain. threshold 0.5 => abstain if <0.5.
    groundings = [0.9, 0.2, 0.1, 0.8]
    labels =     [False, True, True, False]  # noqa: E222  ground-truth should_abstain
    m = confusion_at_threshold(groundings, labels, 0.5)
    # 0.9 answerable & not-abstain -> tn; 0.2 abstain-labeled & predict abstain -> tp
    # 0.1 abstain-labeled & predict abstain -> tp; 0.8 answerable & not-abstain -> tn
    assert (m.tp, m.fp, m.fn, m.tn) == (2, 0, 0, 2)
    assert m.precision == 1.0
    assert m.recall == 1.0


# --------------------------------------------------------------------------
# End-to-end: the REAL assess_answerability separates the two classes
# --------------------------------------------------------------------------

def test_signal_separates_answerable_from_unanswerable() -> None:
    result = evaluate(_fixture_dataset())
    s = result["summary"]

    assert s["n_total"] == 4
    assert s["n_should_abstain"] == 2
    assert s["n_should_answer"] == 2

    # The signal's own operating point must actually catch the unanswerable ones
    # (abstain-recall > 0) — this is the core separation claim.
    default = s["at_default"]
    assert default["recall"] is not None
    assert default["recall"] > 0, default

    # And it should not be abstaining on the answerable items it grounds well.
    ans_items = [it for it in result["items"] if not it["should_abstain"]]
    assert all(it["signal_answerable"] for it in ans_items), ans_items

    # The unanswerable items should be flagged abstain.
    abs_items = [it for it in result["items"] if it["should_abstain"]]
    assert all(it["signal_abstain"] for it in abs_items), abs_items

    # Perfect separation on this clean fixture -> confusion has no errors.
    conf = default["confusion"]
    assert conf["fp"] == 0 and conf["fn"] == 0, conf
    assert default["precision"] == 1.0
    assert default["recall"] == 1.0
    assert default["f1"] == 1.0
    assert default["accuracy"] == 1.0


def test_grounding_is_lower_on_unanswerable_items() -> None:
    """The mechanism behind separation: grounding is high on answerable items
    and low on unanswerable ones."""
    result = evaluate(_fixture_dataset())
    by_id = {it["id"]: it for it in result["items"]}
    assert by_id["ans_1"]["grounding"] >= 0.5
    assert by_id["ans_2"]["grounding"] >= 0.5
    assert by_id["abs_1"]["grounding"] < 0.5
    assert by_id["abs_2"]["grounding"] < 0.5


# --------------------------------------------------------------------------
# Sweep: well-formed, covers default threshold, picks a best-F1 point
# --------------------------------------------------------------------------

def test_sweep_is_well_formed_and_finds_best_f1() -> None:
    result = evaluate(_fixture_dataset())
    sweep = result["sweep"]
    # Default sweep is 0.0..1.0 by 0.1 => 11 points.
    assert len(sweep) == 11
    assert sweep[0]["threshold"] == 0.0
    assert sweep[-1]["threshold"] == 1.0

    # At threshold 0.0 nothing is predicted abstain: no predicted positives, so
    # precision is undefined (None), while recall is 0.0 (every truly-abstain
    # item is missed -> fn>0) and answerable items are true negatives.
    assert sweep[0]["confusion"]["tp"] == 0
    assert sweep[0]["confusion"]["fp"] == 0
    assert sweep[0]["precision"] is None
    assert sweep[0]["recall"] == 0.0

    # best_f1 on this cleanly-separable fixture reaches a perfect F1.
    assert result["summary"]["best_f1"]["f1"] == 1.0


def test_custom_thresholds_are_respected() -> None:
    result = evaluate(_fixture_dataset(), thresholds=[0.3, 0.6])
    assert [row["threshold"] for row in result["sweep"]] == [0.3, 0.6]


# --------------------------------------------------------------------------
# Empty-class safety: all-answerable dataset -> abstain metrics are None, not 0/0
# --------------------------------------------------------------------------

def test_all_answerable_gives_none_recall_no_crash() -> None:
    dataset = [
        normalize_item({
            "query": "what is my hamster called", "should_abstain": False,
            "memories": ["My hamster is called Nibbles"],
        }),
    ]
    result = evaluate(dataset)
    default = result["summary"]["at_default"]
    # No positives (no should_abstain items) -> recall undefined.
    assert default["recall"] is None
    # The single answerable item is grounded, so no false positive -> precision None (no predicted pos).
    assert default["confusion"]["tp"] == 0
    assert default["confusion"]["fn"] == 0


# --------------------------------------------------------------------------
# Injectable assess_fn: the harness is decoupled from the real signal
# --------------------------------------------------------------------------

def test_evaluate_accepts_stub_assess_fn() -> None:
    """A perfect stub signal yields perfect metrics — proves the wiring, not the model."""
    dataset = _fixture_dataset()

    def perfect_stub(query, results):
        # Ground truth is encoded in the fixture; mimic a signal that nails it:
        # abstain (grounding 0) when the top content has no 'hamster'.
        content = (results[0]["content"].lower() if results else "")
        grounds = "hamster" in content
        return {
            "answerable": grounds,
            "grounding": 1.0 if grounds else 0.0,
            "score": 1.0 if grounds else 0.0,
            "margin": 1.0,
            "reason": "stub",
        }

    result = evaluate(dataset, assess_fn=perfect_stub)
    default = result["summary"]["at_default"]
    assert default["precision"] == 1.0
    assert default["recall"] == 1.0
    assert default["f1"] == 1.0


# --------------------------------------------------------------------------
# End-to-end via run_file (JSON on disk)
# --------------------------------------------------------------------------

def test_run_file_end_to_end(tmp_path: Path) -> None:
    rows = [
        {"id": "a", "query": "what is my hamster called", "should_abstain": False,
         "memories": ["My hamster is called Nibbles and loves his wheel"]},
        {"id": "u", "query": "what is my hamster called", "should_abstain": True,
         "memories": ["My cat Luna sleeps all day on the couch"]},
    ]
    path = tmp_path / "answerability.json"
    path.write_text(json.dumps(rows), encoding="utf-8")

    result = run_file(str(path))
    s = result["summary"]
    assert s["n_total"] == 2
    assert s["n_should_abstain"] == 1
    # The abstain signal catches the unanswerable item.
    assert s["at_default"]["confusion"]["tp"] == 1
    assert s["at_default"]["recall"] == 1.0
