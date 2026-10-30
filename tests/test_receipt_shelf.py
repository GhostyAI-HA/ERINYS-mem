"""Receipts (machine-generated audit records) live on the evidence shelf and stay out of context recall."""
from __future__ import annotations

from types import SimpleNamespace

from pytest import MonkeyPatch

from erinys_memory import cli, sentry, server
from erinys_memory.sentry import IntendedUse, SourceClass, build_memory_safety


def _record(obs_id: int, source_class: str, title: str = "x") -> dict[str, object]:
    return {
        "id": obs_id, "title": title, "content": "body", "type": "decision",
        "is_anti_pattern": 0, "distillation_level": None,
        "metadata": {"memory_safety": build_memory_safety("user", "save", source_class)},
    }


def test_receipt_class_is_evidence_only() -> None:
    safety = sentry.MemorySafety.model_validate(build_memory_safety("agent", "save", "receipt"))
    assert safety.source_class is SourceClass.RECEIPT
    assert safety.trusted_for == [IntendedUse.EVIDENCE]
    assert sentry.excluded_for_use(_record(1, "receipt"), "context") is True
    assert sentry.excluded_for_use(_record(1, "receipt"), "preference") is True
    assert sentry.excluded_for_use(_record(1, "receipt"), "evidence") is False
    assert sentry.excluded_for_use(_record(2, "direct_user"), "context") is False


def test_search_drops_receipts_for_context_but_keeps_them_for_evidence(monkeypatch: MonkeyPatch) -> None:
    rows = [_record(1, "receipt"), _record(2, "direct_user"), _record(3, "receipt"), _record(4, "external")]
    monkeypatch.setattr(server, "_embedding", lambda: SimpleNamespace(embed=lambda text: [0.0]))
    monkeypatch.setattr(server, "_db", lambda: None)
    monkeypatch.setattr(server, "rrf_hybrid_search", lambda *a, **k: rows)
    monkeypatch.setattr(server, "_normalize_observation", lambda row: dict(row))

    context = server._search_results("q", None, 3, True, True, None, "context")
    evidence = server._search_results("q", None, 3, True, True, None, "evidence")

    assert [r["id"] for r in context] == [2, 4]
    assert [r["id"] for r in evidence] == [1, 2, 3]


def test_cli_accepts_receipt_source_class() -> None:
    args = cli.build_parser().parse_args(["save", "--title", "t", "--content", "x", "--source-class", "receipt"])
    assert args.source_class == "receipt"
