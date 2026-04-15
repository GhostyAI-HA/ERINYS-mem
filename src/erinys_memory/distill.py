"""Observation を concrete / abstract / meta に蒸留する。

LLM (Ollama) が利用可能な場合は1回の呼び出しで3レベルを一括生成。
利用不可の場合はテンプレート fallback で後方互換を維持する。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import urllib.request
import urllib.error
from typing import Any

from .config import ErinysConfig
from .db import embedding_engine, insert_observation_with_embedding
from .embedding import serialize_f32
from .graph import create_edge

logger = logging.getLogger(__name__)

LEVELS = ("concrete", "abstract", "meta")

_DISTILL_PROMPT = """You are a knowledge distillation engine. Given an observation, produce 3 levels of distillation as JSON.

Rules:
- "concrete": 2-3 sentences. Summarize the specific facts (WHO, WHAT, WHERE, WHY).
- "abstract": 1 paragraph. Extract the reusable pattern or anti-pattern as a general principle.
- "meta": 1 sentence. State the meta-lesson about how to learn, decide, or improve processes.
- Return ONLY valid JSON with exactly these 3 keys. No markdown fences.

Observation Title: {title}
Observation Content:
{content}"""

_TYPE_AWARE_PROMPTS: dict[str, str] = {
    "bugfix": (
        "Focus concrete on root cause and fix. "
        "Abstract should capture the class of bugs this belongs to. "
        "Meta should teach how to prevent this category of bugs."
    ),
    "anti_pattern": (
        "Focus concrete on what went wrong and consequences. "
        "Abstract should define the anti-pattern as a named smell. "
        "Meta should teach how to recognize this smell early."
    ),
    "decision": (
        "Focus concrete on context, options considered, and choice made. "
        "Abstract should capture the decision framework used. "
        "Meta should teach when this framework applies."
    ),
}


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


def _resolve_raw_source(db: sqlite3.Connection, source: dict[str, Any], max_hops: int = 10) -> dict[str, Any]:
    current = source
    for _ in range(max_hops):
        parent_id = current.get("distilled_from")
        if parent_id is None:
            return current
        try:
            current = _fetch_observation(db, int(parent_id))
        except LookupError:
            return current
    return current


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


def _concrete_content_template(source: dict[str, Any]) -> str:
    first_sentence = _first_sentence(str(source["content"]))
    keywords = ", ".join(_extract_keywords(f"{source['title']} {source['content']}"))
    return f"Concrete summary: {first_sentence}. Key details: {keywords}."


def _abstract_content_template(source: dict[str, Any]) -> str:
    keywords = _extract_keywords(f"{source['title']} {source['content']}")
    subject = ", ".join(keywords[:3]) or str(source["title"])
    verb = "avoid" if source.get("is_anti_pattern") else "prefer"
    return f"Abstract pattern: when handling {subject}, {verb} the repeatable approach implied by this memory."


def _meta_content_template(source: dict[str, Any]) -> str:
    keywords = _extract_keywords(f"{source['title']} {source['content']}")
    subject = ", ".join(keywords[:3]) or str(source["title"])
    return f"Meta lesson: turn observations about {subject} into reusable decision heuristics across projects and sessions."


def _template_distillations(source: dict[str, Any]) -> dict[str, str]:
    return {
        "concrete": _concrete_content_template(source),
        "abstract": _abstract_content_template(source),
        "meta": _meta_content_template(source),
    }


def _build_prompt(title: str, content: str, obs_type: str) -> str:
    base = _DISTILL_PROMPT.format(title=title, content=content[:4000])
    type_hint = _TYPE_AWARE_PROMPTS.get(obs_type, "")
    if type_hint:
        return f"{base}\n\nType-specific guidance: {type_hint}"
    return base


def _parse_llm_response(raw: str) -> dict[str, str] | None:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    required = {"concrete", "abstract", "meta"}
    if not required.issubset(parsed.keys()):
        return None
    if not all(isinstance(parsed[k], str) and parsed[k].strip() for k in required):
        return None
    return {k: str(parsed[k]).strip() for k in required}


def _llm_generate(config: ErinysConfig, title: str, content: str, obs_type: str) -> dict[str, str] | None:
    if not config.distill_use_llm:
        return None
    prompt = _build_prompt(title, content, obs_type)
    body = json.dumps({
        "model": config.distill_model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    }).encode("utf-8")
    request = urllib.request.Request(
        config.distill_endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    for attempt in range(2):
        try:
            resp = urllib.request.urlopen(request, timeout=config.distill_timeout)
            resp_body = json.loads(resp.read().decode("utf-8"))
            result = _parse_llm_response(resp_body.get("response", ""))
            if result is not None:
                return result
            logger.warning("LLM returned unparseable response (attempt %d)", attempt + 1)
        except (urllib.error.URLError, json.JSONDecodeError, OSError, KeyError) as exc:
            logger.warning("LLM distillation attempt %d failed: %s", attempt + 1, exc)
        except Exception as exc:
            logger.warning("LLM distillation unexpected error (attempt %d): %s", attempt + 1, exc)
            break
    return None


def _distilled_flags(source: dict[str, Any], level: str) -> tuple[int, int]:
    is_anti_pattern = int(bool(source.get("is_anti_pattern")))
    is_pattern = int(level in {"abstract", "meta"} and not is_anti_pattern)
    return is_anti_pattern, is_pattern


def _build_distill_metadata(
    source: dict[str, Any],
    level: str,
    method: str,
    error: str | None = None,
) -> dict[str, object]:
    base = dict(source.get("metadata") or {})
    base.pop("distill_error", None)
    base["distilled"] = True
    base["level"] = level
    base["distill_method"] = method
    if error is not None:
        base["distill_error"] = error
    return base


def _create_distillation_record(
    db: sqlite3.Connection,
    source: dict[str, Any],
    raw_source: dict[str, Any],
    level: str,
    content: str,
    method: str,
    error: str | None = None,
) -> dict[str, Any]:
    is_anti_pattern, is_pattern = _distilled_flags(source, level)
    raw_id = int(raw_source["id"])
    payload = {
        "title": _distilled_title(str(raw_source["title"]), level),
        "content": content,
        "type": _distilled_type(source, level),
        "project": raw_source["project"],
        "scope": raw_source["scope"],
        "is_anti_pattern": is_anti_pattern,
        "is_pattern": is_pattern,
        "distillation_level": level,
        "distilled_from": raw_id,
        "source": "distill",
        "metadata": _build_distill_metadata(source, level, method, error),
        "session_id": raw_source["session_id"],
    }
    embedding = embedding_engine.embed(str(payload["content"]))
    new_id = insert_observation_with_embedding(db, payload, serialize_f32(embedding))
    create_edge(db, new_id, raw_id, "distilled_from", 1.0, {"level": level})
    return _fetch_observation(db, new_id)


def distill_observation(
    db: sqlite3.Connection,
    observation_id: int,
    level: str = "abstract",
    config: ErinysConfig | None = None,
) -> dict[str, Any]:
    """必要な中間レベルを含めて distillation chain を作る。

    LLM利用可能時: raw observation から1回で3レベルを一括生成。
    LLM失敗時: テンプレートで全レベルを生成（半端状態なし）。
    """
    config = config or ErinysConfig()
    source = _fetch_observation(db, observation_id)
    levels_needed = _levels_to_create(source.get("distillation_level"), level)
    if not levels_needed:
        return {"source": source, "created": [], "final": source}

    raw_source = _resolve_raw_source(db, source)

    llm_result = _llm_generate(
        config,
        str(raw_source["title"]),
        str(raw_source["content"]),
        str(raw_source.get("type", "manual")),
    )
    method = "llm" if llm_result is not None else "template_fallback"
    error = None if llm_result is not None else "llm_unavailable"

    if llm_result is None:
        llm_result = _template_distillations(raw_source)

    created: list[dict[str, Any]] = []
    for next_level in levels_needed:
        content = llm_result.get(next_level, _template_distillations(raw_source)[next_level])
        record = _create_distillation_record(db, source, raw_source, next_level, content, method, error)
        created.append(record)

    final = created[-1] if created else source
    return {"source": source, "created": created, "final": final}
