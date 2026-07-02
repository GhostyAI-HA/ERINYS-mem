"""Unit tests for the end-to-end QA evaluation harness (benchmarks/qa_eval.py).

These tests prove that accuracy / abstention / hallucination are computed
correctly WITHOUT any real LLM: a fake ``answer_fn`` returns canned answers per
question, and a fake judge (or the offline string judge) grades them. The tiny
fixture includes both an answerable item and an unanswerable one, plus a
hallucination case (asserting an answer on an unanswerable question).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# The harness lives under benchmarks/, not the installed package, so add it to
# the path the same way the other benchmark-adjacent tests add src/.
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))

import qa_eval  # noqa: E402
from qa_eval import (  # noqa: E402
    JudgeResult,
    evaluate,
    load_dataset,
    looks_like_abstention,
    normalize_item,
    render_context,
    run_file,
    string_match_judge,
)


# --------------------------------------------------------------------------
# Fixtures: a tiny dataset + a fake LLM answerer
# --------------------------------------------------------------------------

def _fixture_dataset() -> list[dict]:
    """Normalised items: 1 answerable, 1 unanswerable (used for abstain/halluc)."""
    return [
        normalize_item({
            "id": "answerable",
            "question": "What is my dog's name?",
            "gold_answer": "Rex",
            "retrieved_context": ["My dog Rex loves long walks in the park."],
            "unanswerable": False,
        }),
        normalize_item({
            "id": "unanswerable",
            "question": "What is my cat's name?",
            "gold_answer": "",
            "retrieved_context": ["My dog Rex loves long walks in the park."],
            "unanswerable": True,
        }),
    ]


def _fake_answer_fn(mapping: dict[str, str]):
    """Return a fake answerer that maps a question to a canned answer."""

    def answer_fn(question: str, context: str) -> str:
        return mapping[question]

    return answer_fn


# A fake judge that trusts the string judge but is fully controllable.
def _fake_judge(question, answer, gold_answer, unanswerable):
    abstained = looks_like_abstention(answer)
    if unanswerable:
        return JudgeResult(abstained=abstained, correct=False, reason="fake-unanswerable")
    correct = (gold_answer or "").lower() in answer.lower() and not abstained
    return JudgeResult(abstained=abstained, correct=correct, reason="fake-answerable")


# --------------------------------------------------------------------------
# render_context / normalize_item / load_dataset
# --------------------------------------------------------------------------

def test_render_context_joins_list_and_strips() -> None:
    assert render_context(["a", "  b  ", ""]) == "a\n\nb"
    assert render_context("  hello ") == "hello"
    assert render_context(None) == ""


def test_normalize_item_infers_unanswerable_from_empty_gold() -> None:
    item = normalize_item({
        "question": "Q?", "gold_answer": "", "retrieved_context": "ctx"
    })
    assert item["unanswerable"] is True

    item2 = normalize_item({
        "question": "Q?", "gold_answer": "A", "retrieved_context": "ctx"
    })
    assert item2["unanswerable"] is False


def test_normalize_item_requires_question() -> None:
    with pytest.raises(ValueError):
        normalize_item({"gold_answer": "A", "retrieved_context": "ctx"})


def test_load_dataset_json_and_jsonl(tmp_path: Path) -> None:
    rows = [
        {"question": "Q1?", "gold_answer": "A1", "retrieved_context": "c1"},
        {"question": "Q2?", "gold_answer": "", "retrieved_context": "c2", "unanswerable": True},
    ]
    json_path = tmp_path / "data.json"
    json_path.write_text(json.dumps(rows), encoding="utf-8")
    loaded = load_dataset(str(json_path))
    assert len(loaded) == 2
    assert loaded[0]["question"] == "Q1?"
    assert loaded[1]["unanswerable"] is True

    jsonl_path = tmp_path / "data.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    loaded_l = load_dataset(str(jsonl_path))
    assert len(loaded_l) == 2
    assert loaded_l[0]["context"] == "c1"


# --------------------------------------------------------------------------
# looks_like_abstention
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "I don't know.",
    "The context does not mention that.",
    "Not enough information to answer.",
    "",
    "   ",
])
def test_looks_like_abstention_positive(text: str) -> None:
    assert looks_like_abstention(text) is True


@pytest.mark.parametrize("text", ["Rex", "Your dog is named Rex."])
def test_looks_like_abstention_negative(text: str) -> None:
    assert looks_like_abstention(text) is False


# --------------------------------------------------------------------------
# Core: accuracy computed correctly (answerable item answered correctly)
# --------------------------------------------------------------------------

def test_accuracy_correct_answer() -> None:
    dataset = _fixture_dataset()
    answers = {
        "What is my dog's name?": "Your dog's name is Rex.",
        "What is my cat's name?": "I don't know.",
    }
    result = evaluate(dataset, _fake_answer_fn(answers), _fake_judge)
    s = result["summary"]

    assert s["n_total"] == 2
    assert s["n_answerable"] == 1
    assert s["n_unanswerable"] == 1
    # answerable item answered correctly -> accuracy 1.0
    assert s["accuracy"] == 1.0
    assert s["n_correct"] == 1
    # unanswerable item correctly abstained -> abstention 1.0, halluc 0.0
    assert s["abstention_rate"] == 1.0
    assert s["hallucination_rate"] == 0.0

    ans_item = next(i for i in result["items"] if i["id"] == "answerable")
    assert ans_item["correct"] is True
    assert ans_item["hallucinated"] is None

    unans_item = next(i for i in result["items"] if i["id"] == "unanswerable")
    assert unans_item["correct"] is None
    assert unans_item["abstained"] is True
    assert unans_item["hallucinated"] is False


# --------------------------------------------------------------------------
# Core: wrong answer lowers accuracy
# --------------------------------------------------------------------------

def test_accuracy_wrong_answer() -> None:
    dataset = _fixture_dataset()
    answers = {
        "What is my dog's name?": "Your dog's name is Fido.",  # wrong (gold=Rex)
        "What is my cat's name?": "I don't know.",
    }
    result = evaluate(dataset, _fake_answer_fn(answers), _fake_judge)
    s = result["summary"]
    assert s["accuracy"] == 0.0
    assert s["n_correct"] == 0
    # unanswerable still handled independently
    assert s["abstention_rate"] == 1.0
    assert s["hallucination_rate"] == 0.0


# --------------------------------------------------------------------------
# Core: hallucination computed correctly (asserted answer on unanswerable)
# --------------------------------------------------------------------------

def test_hallucination_on_unanswerable() -> None:
    dataset = _fixture_dataset()
    answers = {
        "What is my dog's name?": "Rex",
        # Model invents a cat name though context has no cat -> hallucination.
        "What is my cat's name?": "Your cat's name is Whiskers.",
    }
    result = evaluate(dataset, _fake_answer_fn(answers), _fake_judge)
    s = result["summary"]

    assert s["accuracy"] == 1.0  # dog still correct
    assert s["abstention_rate"] == 0.0
    assert s["hallucination_rate"] == 1.0
    assert s["n_hallucinated"] == 1
    assert s["n_abstained_correctly"] == 0

    unans_item = next(i for i in result["items"] if i["id"] == "unanswerable")
    assert unans_item["hallucinated"] is True
    assert unans_item["abstained"] is False


# --------------------------------------------------------------------------
# abstaining on an ANSWERABLE question is not "correct"
# --------------------------------------------------------------------------

def test_abstain_on_answerable_is_incorrect() -> None:
    dataset = _fixture_dataset()
    answers = {
        "What is my dog's name?": "I don't know.",  # should have answered Rex
        "What is my cat's name?": "I don't know.",
    }
    result = evaluate(dataset, _fake_answer_fn(answers), _fake_judge)
    s = result["summary"]
    assert s["accuracy"] == 0.0
    ans_item = next(i for i in result["items"] if i["id"] == "answerable")
    assert ans_item["correct"] is False
    assert ans_item["abstained"] is True


# --------------------------------------------------------------------------
# Rates are None when a class is empty (no division by zero)
# --------------------------------------------------------------------------

def test_rates_none_when_class_empty() -> None:
    only_answerable = [normalize_item({
        "question": "Q?", "gold_answer": "A", "retrieved_context": "A is the value"
    })]
    result = evaluate(
        only_answerable,
        _fake_answer_fn({"Q?": "A is the value"}),
        _fake_judge,
    )
    s = result["summary"]
    assert s["abstention_rate"] is None
    assert s["hallucination_rate"] is None
    assert s["accuracy"] == 1.0


# --------------------------------------------------------------------------
# The offline string_match_judge itself grades sanely (no LLM path)
# --------------------------------------------------------------------------

def test_string_match_judge_substring() -> None:
    v = string_match_judge("Q?", "Your dog is Rex.", "Rex", False)
    assert v.correct is True
    assert v.abstained is False


def test_string_match_judge_token_subset() -> None:
    v = string_match_judge("Q?", "The answer is New York City.", "new york", False)
    assert v.correct is True


def test_string_match_judge_no_match() -> None:
    v = string_match_judge("Q?", "It is Fido.", "Rex", False)
    assert v.correct is False


def test_string_match_judge_abstention_on_answerable() -> None:
    v = string_match_judge("Q?", "I don't know.", "Rex", False)
    assert v.abstained is True
    assert v.correct is False


def test_string_match_judge_unanswerable_correct_is_false() -> None:
    v = string_match_judge("Q?", "Something", None, True)
    assert v.correct is False


# --------------------------------------------------------------------------
# End-to-end via run_file using the offline string judge (no network)
# --------------------------------------------------------------------------

def test_run_file_with_string_judge(tmp_path: Path) -> None:
    rows = [
        {"id": "a", "question": "What is my dog's name?", "gold_answer": "Rex",
         "retrieved_context": ["My dog Rex loves walks."]},
        {"id": "u", "question": "What is my cat's name?", "gold_answer": "",
         "retrieved_context": ["My dog Rex loves walks."], "unanswerable": True},
    ]
    path = tmp_path / "qa.json"
    path.write_text(json.dumps(rows), encoding="utf-8")

    answers = {
        "What is my dog's name?": "Your dog's name is Rex.",
        "What is my cat's name?": "I don't know.",
    }
    result = run_file(
        str(path),
        answer_fn=_fake_answer_fn(answers),
        judge_fn=string_match_judge,
    )
    s = result["summary"]
    assert s["accuracy"] == 1.0
    assert s["abstention_rate"] == 1.0
    assert s["hallucination_rate"] == 0.0


# --------------------------------------------------------------------------
# Default Ollama answerer/judge builders don't hit the network at build time
# and the answerer's HTTP path is exercised via a stubbed urlopen.
# --------------------------------------------------------------------------

def test_ollama_answer_fn_uses_stubbed_http(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeResp:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def read(self) -> bytes:
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    captured: dict = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["data"] = json.loads(request.data.decode("utf-8"))
        return _FakeResp(json.dumps({"response": "  Rex  "}).encode("utf-8"))

    monkeypatch.setattr(qa_eval.urllib.request, "urlopen", fake_urlopen)

    answer_fn = qa_eval.make_ollama_answer_fn(
        endpoint="http://localhost:11434/api/generate", model="test-model"
    )
    out = answer_fn("What is my dog's name?", "My dog Rex.")
    assert out == "Rex"
    assert captured["data"]["model"] == "test-model"
    assert captured["data"]["stream"] is False
    assert "My dog Rex." in captured["data"]["prompt"]


def test_ollama_judge_fn_uses_stubbed_http(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeResp:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def read(self) -> bytes:
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout=None):
        return _FakeResp(json.dumps({
            "response": json.dumps({"abstained": False, "correct": True})
        }).encode("utf-8"))

    monkeypatch.setattr(qa_eval.urllib.request, "urlopen", fake_urlopen)

    judge_fn = qa_eval.make_ollama_judge_fn(model="test-model")
    verdict = judge_fn("Q?", "Rex", "Rex", False)
    assert verdict.correct is True
    assert verdict.abstained is False

    # Unanswerable path never calls the LLM; abstention decided offline.
    verdict2 = judge_fn("Q?", "I don't know.", None, True)
    assert verdict2.abstained is True
    assert verdict2.correct is False
