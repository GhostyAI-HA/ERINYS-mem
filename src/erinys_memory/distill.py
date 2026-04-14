"""Observation を concrete / abstract / meta に蒸留する。"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from .db import embedding_engine, insert_observation_with_embedding
from .embedding import serialize_f32
from .graph import create_edge

LEVELS = ("concrete", "abstract", "meta")


def _decode_json(value: object) -> object:
    if isinstance(value, str) and value:
        return json.loads(value)
    return value


def _observation_record(row: sqlite3.Row) -> dict[str, Any]:
    record = dict(row)
    record["metadata"] = _decode_json(record.get("metadata"))
    return record


def _fetch_observation(db: sqlite3.Connection, obs_id: int) -> dict[str, Any]:
    row = db.execute("SELECT * FROM observations WHERE id = ?", [obs_id]).fetchone()
    if row is None:
        raise LookupError(f"observation not found: {obs_id}")
    return _observation_record(row)


def _extract_keywords(text: str) -> list[str]:
    lowered = text.lower().replace("\n", " ")
    tokens = [token.strip(".,:;!?()[]{}\"'") for token in lowered.split()]
    keywords = [token for token in tokens if len(token) >= 4]
    seen: set[str] = set()
    ordered: list[str] = []
    for token in keywords:
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered[:5]


def _first_sentence(text: str) -> str:
    compact = " ".join(part.strip() for part in text.splitlines() if part.strip())
    for delimiter in (". ", "。", "! ", "? "):
        if delimiter in compact:
            return compact.split(delimiter, 1)[0].strip()
    return compact[:240].strip()


def _level_index(level: str | None) -> int:
    if level is None:
        return -1
    return LEVELS.index(level)


def _levels_to_create(current_level: str | None, target_level: str) -> list[str]:
    if target_level not in LEVELS:
        raise ValueError(f"invalid distillation level: {target_level}")
    current_index = _level_index(current_level)
    target_index = _level_index(target_level)
    if target_index <= current_index:
        return []
    return list(LEVELS[current_index + 1 : target_index + 1])


def _distilled_title(title: str, level: str) -> str:
    return f"{level.capitalize()} Distillation: {title}"


def _distilled_type(source: dict[str, Any], level: str) -> str:
    if level == "meta":
        return "meta_knowledge"
    if source.get("is_anti_pattern") or source.get("type") == "anti_pattern":
        return "anti_pattern"
    if level == "abstract":
        return "pattern"
    return str(source["type"])


def _concrete_content(source: dict[str, Any]) -> str:
    first_sentence = _first_sentence(str(source["content"]))
    keywords = ", ".join(_extract_keywords(f"{source['title']} {source['content']}"))
    return f"Concrete summary: {first_sentence}. Key details: {keywords}."


def _abstract_content(source: dict[str, Any]) -> str:
    keywords = _extract_keywords(f"{source['title']} {source['content']}")
    subject = ", ".join(keywords[:3]) or str(source["title"])
    verb = "avoid" if source.get("is_anti_pattern") else "prefer"
    return f"Abstract pattern: when handling {subject}, {verb} the repeatable approach implied by this memory."


def _meta_content(source: dict[str, Any]) -> str:
    keywords = _extract_keywords(f"{source['title']} {source['content']}")
    subject = ", ".join(keywords[:3]) or str(source["title"])
    return f"Meta lesson: turn observations about {subject} into reusable decision heuristics across projects and sessions."


def _distilled_content(source: dict[str, Any], level: str) -> str:
    if level == "concrete":
        return _concrete_content(source)
    if level == "abstract":
        return _abstract_content(source)
    return _meta_content(source)


def _distilled_flags(source: dict[str, Any], level: str) -> tuple[int, int]:
    is_anti_pattern = int(bool(source.get("is_anti_pattern")))
    is_pattern = int(level in {"abstract", "meta"} and not is_anti_pattern)
    return is_anti_pattern, is_pattern


def _create_distillation(
    db: sqlite3.Connection,
    source: dict[str, Any],
    level: str,
) -> dict[str, Any]:
    is_anti_pattern, is_pattern = _distilled_flags(source, level)
    payload = {
        "title": _distilled_title(str(source["title"]), level),
        "content": _distilled_content(source, level),
        "type": _distilled_type(source, level),
        "project": source["project"],
        "scope": source["scope"],
        "is_anti_pattern": is_anti_pattern,
        "is_pattern": is_pattern,
        "distillation_level": level,
        "distilled_from": source["id"],
        "source": "distill",
        "metadata": {**dict(source.get("metadata") or {}), "distilled": True, "level": level},
        "session_id": source["session_id"],
    }
    embedding = embedding_engine.embed(str(payload["content"]))
    new_id = insert_observation_with_embedding(db, payload, serialize_f32(embedding))
    create_edge(db, new_id, int(source["id"]), "distilled_from", 1.0, {"level": level})
    return _fetch_observation(db, new_id)


def distill_observation(
    db: sqlite3.Connection,
    observation_id: int,
    level: str = "abstract",
) -> dict[str, Any]:
    """必要な中間レベルを含めて distillation chain を作る。"""
    source = _fetch_observation(db, observation_id)
    created: list[dict[str, Any]] = []
    current = source
    for next_level in _levels_to_create(source.get("distillation_level"), level):
        current = _create_distillation(db, current, next_level)
        created.append(current)
    return {"source": source, "created": created, "final": current}
