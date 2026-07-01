"""Observation を concrete / abstract / meta に蒸留する。

LLM (Ollama) が利用可能な場合は1回の呼び出しで3レベルを一括生成。
利用不可の場合はテンプレート fallback で後方互換を維持する。
"""

from __future__ import annotations

import json
import logging
import math
from ._sqlite import sqlite3
import struct
import urllib.request
import urllib.error
from typing import Any

from .config import ErinysConfig
from .db import embedding_engine, insert_observation_with_embedding, resolve_session_id
from .embedding import serialize_f32
from .graph import create_edge
from .provenance import build_provenance

logger = logging.getLogger(__name__)

LEVELS = ("concrete", "abstract", "meta")

# SSGM quality gate (growth-radar #147, 2026-06-22):
# _compute_distillation_quality は既存だが quality_score<0.4 が warning 止まりで、
# 低品質蒸留がそのまま consolidate されていた。ここで metadata に可逆フラグを立て、
# recall/再蒸留が識別できる「強制ゲート」に配線する（データは消さない）。
QUALITY_GATE_THRESHOLD = 0.4

_DISTILL_PROMPT = """You are a knowledge distillation engine. Given an observation, produce 3 levels of distillation as JSON.

Rules:
- "concrete": 2-3 sentences. Summarize the specific facts (WHO, WHAT, WHERE, WHY).
- "abstract": 1 paragraph. Extract the reusable pattern or anti-pattern as a general principle.
- "meta": 1 sentence. State the meta-lesson about how to learn, decide, or improve processes.
- "what_made_it_fail": 1 sentence. If the observation describes a failure or anti-pattern, state the specific causal factor that turned a near-miss into an actual failure. If not applicable, return empty string.
- "what_made_it_work": 1 sentence. If the observation describes a success or good pattern, state the specific factor that made it succeed where alternatives failed. If not applicable, return empty string.
- Return ONLY valid JSON with exactly these 5 keys. No markdown fences.

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


# raw source の embedding は vec_observations に保存済み。再計算せず取り出す。
# 取得できない場合（テスト用 DB 等）は embed にフォールバックする。
def _source_embedding(db: sqlite3.Connection, raw_source: dict[str, Any]) -> list[float]:
    try:
        row = db.execute(
            "SELECT embedding FROM vec_observations WHERE rowid = ?",
            [int(raw_source["id"])],
        ).fetchone()
    except sqlite3.OperationalError:
        row = None
    if row is not None:
        blob = bytes(row[0])
        return list(struct.unpack(f"<{len(blob) // 4}f", blob))
    return embedding_engine.embed(str(raw_source["content"]))


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


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(v * v for v in left))
    right_norm = math.sqrt(sum(v * v for v in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _compute_distillation_quality(
    source_content: str,
    distilled_content: str,
    source_embedding: list[float],
    distilled_embedding: list[float],
    level: str = "concrete",
) -> dict[str, float]:
    """Compute quality metrics for a distillation.

    Returns dict with:
    - semantic_preservation: cosine similarity between source and distilled embeddings (0-1)
    - keyword_retention: fraction of source keywords preserved in distilled content (0-1)
    - compression_ratio: len(distilled) / len(source)
    - quality_score: weighted average (0-1, >= 0.7 is good)

    W6: Level-aware weights — meta emphasizes semantic, concrete emphasizes keywords.
    W5: Compression is scored as distance from ideal ratio per level.
    """
    semantic_preservation = max(0.0, _cosine_similarity(source_embedding, distilled_embedding))

    source_keywords = _extract_keywords(source_content)
    if source_keywords:
        distilled_lower = distilled_content.lower()
        retained = sum(1 for kw in source_keywords if kw in distilled_lower)
        keyword_retention = retained / len(source_keywords)
    else:
        keyword_retention = 1.0

    source_len = len(source_content)
    compression_ratio = len(distilled_content) / source_len if source_len > 0 else 0.0

    # W5: Score compression as closeness to ideal ratio per level
    # Ideal: concrete ≈ 0.6-0.8, abstract ≈ 0.3-0.5, meta ≈ 0.1-0.3
    _IDEAL_RATIOS = {"concrete": 0.7, "abstract": 0.4, "meta": 0.2}
    ideal = _IDEAL_RATIOS.get(level, 0.5)
    compression_score = max(0.0, 1.0 - abs(compression_ratio - ideal) / ideal)

    # W6: Level-aware quality weights
    _LEVEL_WEIGHTS = {
        "concrete": {"semantic": 0.40, "keyword": 0.40, "compression": 0.20},
        "abstract": {"semantic": 0.50, "keyword": 0.30, "compression": 0.20},
        "meta":     {"semantic": 0.60, "keyword": 0.20, "compression": 0.20},
    }
    weights = _LEVEL_WEIGHTS.get(level, {"semantic": 0.50, "keyword": 0.30, "compression": 0.20})

    quality_score = (
        weights["semantic"] * semantic_preservation
        + weights["keyword"] * keyword_retention
        + weights["compression"] * compression_score
    )

    return {
        "semantic_preservation": round(semantic_preservation, 4),
        "keyword_retention": round(keyword_retention, 4),
        "compression_ratio": round(compression_ratio, 4),
        "compression_score": round(compression_score, 4),
        "quality_score": round(quality_score, 4),
    }


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


def _exploitability_template(source: dict[str, Any]) -> tuple[str, str]:
    is_anti = source.get("is_anti_pattern") or source.get("type") == "anti_pattern"
    keywords = _extract_keywords(f"{source['title']} {source['content']}")
    subject = ", ".join(keywords[:3]) or str(source["title"])
    if is_anti:
        return (f"The failure in {subject} became actual because no structural guard prevented it.", "")
    return ("", f"The approach for {subject} succeeded because a structural mechanism enforced the correct path.")


def _template_distillations(source: dict[str, Any]) -> dict[str, str]:
    fail, work = _exploitability_template(source)
    return {
        "concrete": _concrete_content_template(source),
        "abstract": _abstract_content_template(source),
        "meta": _meta_content_template(source),
        "what_made_it_fail": fail,
        "what_made_it_work": work,
    }


def _causal_factors_with_fallback(
    llm_result: dict[str, str],
    template_result: dict[str, str],
) -> dict[str, str]:
    return {
        "what_made_it_fail": llm_result.get(
            "what_made_it_fail", template_result["what_made_it_fail"]
        ),
        "what_made_it_work": llm_result.get(
            "what_made_it_work", template_result["what_made_it_work"]
        ),
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
    result = {k: str(parsed[k]).strip() for k in required}
    for causal_key in ("what_made_it_fail", "what_made_it_work"):
        if causal_key in parsed and isinstance(parsed[causal_key], str):
            result[causal_key] = parsed[causal_key].strip()
    return result


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
    causal_factors: dict[str, str] | None = None,
) -> dict[str, object]:
    base = dict(source.get("metadata") or {})
    base.pop("distill_error", None)
    base["distilled"] = True
    base["level"] = level
    base["distill_method"] = method
    if error is not None:
        base["distill_error"] = error
    if causal_factors:
        for k, v in causal_factors.items():
            if v:
                base[k] = v
    return base


def _apply_quality_gate(
    raw_source: dict[str, Any],
    level: str,
    method: str,
    quality: dict[str, Any],
    metadata: dict[str, object],
) -> None:
    """低品質蒸留を消さずに metadata へ可逆フラグする（SSGM quality gate の配線）。"""
    # 源泉metadataから継承した古いフラグを必ず除去（今回の品質を反映させる。Codex P2）。
    metadata.pop("quality_gate", None)
    if quality["quality_score"] >= QUALITY_GATE_THRESHOLD:
        return
    metadata["quality_gate"] = {"status": "low", "score": quality["quality_score"]}
    logger.warning(
        "Low distillation quality %.2f for '%s' level=%s (method=%s)",
        quality["quality_score"], raw_source.get("title", "?"), level, method,
    )


def _create_distillation_record(
    db: sqlite3.Connection,
    source: dict[str, Any],
    raw_source: dict[str, Any],
    level: str,
    content: str,
    method: str,
    error: str | None = None,
    causal_factors: dict[str, str] | None = None,
    source_embedding: list[float] | None = None,
) -> dict[str, Any]:
    is_anti_pattern, is_pattern = _distilled_flags(source, level)
    raw_id = int(raw_source["id"])
    metadata = _build_distill_metadata(source, level, method, error, causal_factors)

    source_content = str(raw_source["content"])
    if source_embedding is None:
        source_embedding = _source_embedding(db, raw_source)
    distilled_embedding = embedding_engine.embed(content)

    quality = _compute_distillation_quality(
        source_content, content, source_embedding, distilled_embedding,
        level=level,
    )
    metadata["distillation_quality"] = quality
    _apply_quality_gate(raw_source, level, method, quality, metadata)
    # VMG provenance: 蒸留物の出自を raw source に紐づけて記録(parents=[raw_id])。
    metadata["provenance"] = build_provenance(
        "distill", None, "distill", [raw_id]
    )

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
        "metadata": metadata,
        "session_id": resolve_session_id(db, raw_source["session_id"]),
    }
    # W7: Wrap insert + edge creation in single transaction via insert_observation_with_embedding
    # (which already uses BEGIN IMMEDIATE). create_edge commits separately, so we catch failures.
    new_id = insert_observation_with_embedding(db, payload, serialize_f32(distilled_embedding))
    try:
        create_edge(db, new_id, raw_id, "distilled_from", 1.0, {"level": level})
    except Exception as exc:
        logger.warning(
            "Failed to create distilled_from edge %d→%d: %s; observation saved.",
            new_id, raw_id, exc,
        )
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

    template_result = _template_distillations(raw_source)
    if llm_result is None:
        llm_result = template_result

    causal_factors = _causal_factors_with_fallback(llm_result, template_result)

    source_embedding = _source_embedding(db, raw_source)
    created: list[dict[str, Any]] = []
    for next_level in levels_needed:
        content = llm_result.get(next_level, template_result[next_level])
        record = _create_distillation_record(
            db, source, raw_source, next_level, content, method, error,
            causal_factors=causal_factors,
            source_embedding=source_embedding,
        )
        created.append(record)

    final = created[-1] if created else source
    return {"source": source, "created": created, "final": final}
