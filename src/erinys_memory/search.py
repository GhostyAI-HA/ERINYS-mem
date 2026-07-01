"""FTS5 と sqlite-vec を RRF で統合する検索ロジック。

Enhanced v2: OR-mode FTS5, IDF-weighted keyword overlap,
bigram/proper-noun/temporal boosting を統合。
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timedelta, timezone
from ._sqlite import sqlite3
from typing import Any, Sequence

from .decay import current_strength
from .embedding import serialize_f32


# -- Query Complexity Classification (Growth Radar #110: SimpleMem-inspired) --

COMPLEXITY_L1 = "L1"  # Simple: single keyword or short phrase
COMPLEXITY_L2 = "L2"  # Medium: 2-3 keywords with context
COMPLEXITY_L3 = "L3"  # Complex: long text or compound intent

# Weight presets per complexity level
_COMPLEXITY_WEIGHTS: dict[str, dict[str, float]] = {
    COMPLEXITY_L1: {"fts_weight": 0.55, "vec_weight": 0.45},
    COMPLEXITY_L2: {"fts_weight": 0.40, "vec_weight": 0.60},
    COMPLEXITY_L3: {"fts_weight": 0.30, "vec_weight": 0.70},
}


def classify_query_complexity(query: str) -> str:
    """Classify query complexity into L1/L2/L3 for adaptive retrieval.

    L1 (Simple): short queries, single concepts → FTS5 dominant
    L2 (Medium): moderate queries, some context → balanced
    L3 (Complex): long queries, multiple concepts → vec dominant

    Criteria:
    - Token count
    - Presence of proper nouns
    - Temporal expressions
    - Quoted phrases
    - Question complexity (compound questions)
    """
    tokens = re.findall(r"[a-zA-Z0-9']+", query)
    content_tokens = [t for t in tokens if t.lower() not in _CLASSIFY_STOPWORDS and len(t) > 1]
    token_count = len(content_tokens)

    # W2: CJK/non-Latin queries get zero ASCII tokens → default to L2 (vec-heavy)
    # since FTS5 porter tokenizer has poor CJK recall
    has_cjk = bool(re.search(r'[\u3000-\u9fff\uac00-\ud7af]', query))
    if token_count == 0 and has_cjk:
        return COMPLEXITY_L2
    # Mixed CJK+ASCII: CJK presence pushes toward vec-heavy
    if has_cjk:
        complexity_score_cjk_bonus = 2
    else:
        complexity_score_cjk_bonus = 0

    has_quotes = bool(re.search(r"""['"][^'"]+['"]""", query))
    has_temporal = bool(TEMPORAL_PHRASES.search(query))
    proper_noun_count = sum(
        1 for i, w in enumerate(query.split())
        if i > 0 and w[0:1].isupper() and w.lower() not in _CLASSIFY_STOPWORDS
    )

    complexity_score = token_count + complexity_score_cjk_bonus
    if has_quotes:
        complexity_score += 2
    if has_temporal:
        complexity_score += 2
    complexity_score += proper_noun_count

    if complexity_score <= 2:
        return COMPLEXITY_L1
    if complexity_score <= 6:
        return COMPLEXITY_L2
    return COMPLEXITY_L3


# Minimal stopwords for complexity classification (avoid importing full set before definition)
_CLASSIFY_STOPWORDS = frozenset({
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
    "a", "an", "the", "and", "but", "or", "if", "is", "are", "was", "were",
    "be", "been", "have", "has", "had", "do", "does", "did", "of", "at",
    "by", "for", "with", "to", "from", "in", "on", "what", "which", "who",
    "how", "when", "where", "why", "this", "that", "about",
})


# -- Intent-Aware Router (Growth Radar #131: MAGMA-inspired) --

INTENT_WHAT = "WHAT"  # Factual retrieval
INTENT_WHEN = "WHEN"  # Temporal retrieval
INTENT_WHY = "WHY"    # Causal reasoning
INTENT_WHO = "WHO"    # Entity/person retrieval
INTENT_GENERAL = "GENERAL"  # No specific intent detected

# Boost overrides per intent (applied on top of complexity weights)
_INTENT_BOOST_OVERRIDES: dict[str, dict[str, float]] = {
    INTENT_WHAT: {
        "keyword_boost": 0.5,     # ↑ factual keywords matter more
        "bigram_boost": 0.7,      # ↑ exact phrases matter
    },
    INTENT_WHEN: {
        "temporal_content_boost": 0.6,  # ↑ temporal context
    },
    INTENT_WHY: {
        "vec_weight_delta": 0.10,  # ↑ semantic similarity for causality
        "keyword_boost": 0.2,      # ↓ keywords less relevant
    },
    INTENT_WHO: {
        "proper_noun_boost": 0.6,  # ↑ names matter
    },
    INTENT_GENERAL: {},
}

_INTENT_PATTERNS: dict[str, re.Pattern[str]] = {
    INTENT_WHEN: re.compile(
        r"(?:\b(?:when|date|timeline)\b|いつ|何時|時期|日付)", re.IGNORECASE
    ),
    INTENT_WHY: re.compile(
        r"(?:\b(?:why|because|reason|cause)\b|なぜ|どうして|理由|原因)", re.IGNORECASE
    ),
    INTENT_WHO: re.compile(
        r"(?:\b(?:who|author|person|name)\b|誰|だれ|名前)", re.IGNORECASE
    ),
    INTENT_WHAT: re.compile(
        r"(?:\b(?:what|define|definition|describe)\b|何|なに|概要|説明)", re.IGNORECASE
    ),
}


def classify_query_intent(query: str) -> str:
    """Classify query intent into WHAT/WHEN/WHY/WHO/GENERAL.

    Uses question word patterns and keyword heuristics.
    Priority: WHY > WHEN > WHO > WHAT > GENERAL
    (WHY is highest because causal queries need the most routing adjustment)
    """
    for intent in (INTENT_WHY, INTENT_WHEN, INTENT_WHO, INTENT_WHAT):
        if _INTENT_PATTERNS[intent].search(query):
            return intent
    return INTENT_GENERAL



STOPWORDS = frozenset({
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as",
    "until", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how",
    "all", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "s", "t", "can", "will", "just",
    "don", "should", "now", "d", "ll", "m", "o", "re", "ve",
    "y", "ain", "aren", "couldn", "didn", "doesn", "hadn",
    "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn",
    "shan", "shouldn", "wasn", "weren", "won", "wouldn",
    "think", "would", "could", "also", "like", "get", "got",
    "go", "went", "going", "come", "came", "make", "made",
    "know", "knew", "see", "saw", "seem", "take", "took",
    "give", "gave", "tell", "told", "find", "found",
    "want", "need", "use", "used", "try", "tried",
    "look", "looking", "well", "back", "even", "still",
    "let", "may", "might", "much", "many", "since",
    "last", "first", "new", "old", "good", "great",
    "long", "little", "right", "big", "high", "small",
    "really", "always", "never", "often", "already",
    "ago", "day", "days", "week", "weeks", "month", "months",
    "year", "years", "time", "today", "yesterday", "tomorrow",
    "recently", "earlier", "later", "mentioned", "discussed",
    "remind", "remember", "noticed", "feeling", "lately",
})

TEMPORAL_WORDS = frozenset({
    "ago", "last", "yesterday", "recently", "earlier",
    "week", "weeks", "month", "months", "day", "days",
    "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "monday",
    "four", "two", "three", "ten", "couple",
})

TEMPORAL_PHRASES = re.compile(
    r"\b(?:"
    r"\d+\s+(?:days?|weeks?|months?|years?)\s+ago"
    r"|last\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"|yesterday|today|tomorrow|recently|earlier|lately"
    r"|a\s+(?:week|month|year|couple\s+of\s+(?:days?|weeks?|months?|years?))\s+ago"
    r"|(?:four|three|two|ten|five|six|seven|eight|nine)\s+(?:days?|weeks?|months?|years?)\s+ago"
    r")\b",
    re.IGNORECASE,
)

CONVERSATIONAL_FILLER = frozenset({
    "mentioned", "discussed", "talked", "told", "said", "shared",
    "remember", "recall", "remind", "noticed", "feeling",
    "think", "believe", "idea", "opinion",
    "lately", "recently", "currently",
})
SQLITE_IN_CHUNK_SIZE = 500


def _parse_datetime_value(value: object, field_name: str) -> datetime:
    """日時境界で created_at/last_accessed の型を確定する。"""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = value.decode("utf-8") if isinstance(value, bytes) else value
    if isinstance(text, str):
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    raise TypeError(f"{field_name} must be datetime or ISO string")


def _parse_optional_datetime_value(value: object, field_name: str) -> datetime | None:
    """NULL 許容の日時フィールドを境界で正規化する。"""
    if value in (None, ""):
        return None
    return _parse_datetime_value(value, field_name)


def _count_project_matches(
    db: sqlite3.Connection,
    observation_ids: Sequence[int],
    project: str,
) -> int:
    total = 0
    for start in range(0, len(observation_ids), SQLITE_IN_CHUNK_SIZE):
        chunk = list(observation_ids[start:start + SQLITE_IN_CHUNK_SIZE])
        placeholders = ",".join("?" for _ in chunk)
        row = db.execute(
            f"SELECT COUNT(*) FROM observations WHERE project = ? AND id IN ({placeholders})",
            [project, *chunk],
        ).fetchone()
        total += int(row[0]) if row else 0
    return total


def strip_temporal_from_query(query: str) -> str:
    """Remove temporal expressions from query text for cleaner embedding."""
    stripped = TEMPORAL_PHRASES.sub("", query)
    tokens = stripped.split()
    tokens = [t for t in tokens if t.lower().rstrip(".,?!") not in TEMPORAL_WORDS]
    result = " ".join(tokens).strip()
    result = re.sub(r"\s+", " ", result)
    return result if len(result.split()) >= 1 else query


def focus_query_for_embedding(query: str) -> str:
    """Strip temporal expressions and conversational filler for focused embedding."""
    cleaned = strip_temporal_from_query(query)
    tokens = cleaned.split()
    focused = [t for t in tokens if t.lower().rstrip(".,?!") not in CONVERSATIONAL_FILLER]
    result = " ".join(focused).strip()
    return result if len(result.split()) >= 1 else cleaned


_WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "a": 1, "couple": 2,
}

_DAY_NAMES = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

_RELATIVE_DATE_PATTERNS = [
    re.compile(r"(\d+)\s+(days?|weeks?|months?|years?)\s+ago", re.I),
    re.compile(
        r"(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|a|couple)"
        r"\s+(?:of\s+)?(days?|weeks?|months?|years?)\s+ago",
        re.I,
    ),
    re.compile(
        r"last\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday"
        r"|week|month|year)",
        re.I,
    ),
    re.compile(r"\byesterday\b", re.I),
]


def parse_relative_date(query: str, anchor: datetime) -> datetime | None:
    """Parse relative date expressions and return the target date."""
    q_lower = query.lower()

    m = _RELATIVE_DATE_PATTERNS[0].search(q_lower)
    if m:
        n = int(m.group(1))
        unit = m.group(2).rstrip("s")
        if unit == "day":
            return anchor - timedelta(days=n)
        if unit == "week":
            return anchor - timedelta(weeks=n)
        if unit == "month":
            return anchor - timedelta(days=n * 30)
        if unit == "year":
            return anchor - timedelta(days=n * 365)

    m = _RELATIVE_DATE_PATTERNS[1].search(q_lower)
    if m:
        n = _WORD_TO_NUM.get(m.group(1).lower(), 1)
        unit = m.group(2).rstrip("s")
        if unit == "day":
            return anchor - timedelta(days=n)
        if unit == "week":
            return anchor - timedelta(weeks=n)
        if unit == "month":
            return anchor - timedelta(days=n * 30)
        if unit == "year":
            return anchor - timedelta(days=n * 365)

    m = _RELATIVE_DATE_PATTERNS[2].search(q_lower)
    if m:
        target = m.group(1).lower()
        if target in _DAY_NAMES:
            target_weekday = _DAY_NAMES[target]
            days_back = (anchor.weekday() - target_weekday) % 7
            if days_back == 0:
                days_back = 7
            return anchor - timedelta(days=days_back)
        if target == "week":
            return anchor - timedelta(weeks=1)
        if target == "month":
            return anchor - timedelta(days=30)
        if target == "year":
            return anchor - timedelta(days=365)

    if _RELATIVE_DATE_PATTERNS[3].search(q_lower):
        return anchor - timedelta(days=1)

    return None


def apply_temporal_boost(
    results: list[dict[str, Any]],
    query: str,
    anchor: datetime,
    weight: float = 0.45,
    sigma: float = 2.0,
) -> list[dict[str, Any]]:
    """Apply Gaussian temporal proximity boost to results.
    Results with created_at close to the parsed target date get score boosts."""
    target = parse_relative_date(query, anchor)
    if target is None:
        return results

    target_date = target.date()
    for r in results:
        created_str = r.get("created_at", "")
        if not created_str:
            continue
        try:
            created_dt = datetime.fromisoformat(str(created_str).replace("Z", "+00:00"))
            delta_days = abs((created_dt.date() - target_date).days)
            boost = 1.0 + weight * math.exp(-(delta_days ** 2) / (2 * sigma ** 2))
            old_score = float(r.get("effective_score", 0))
            r["temporal_boost"] = boost
            r["effective_score"] = old_score * boost
        except (ValueError, TypeError):
            continue

    return results

COMMON_QUERY_WORDS = frozenset({
    "recommend", "suggest", "tell", "help", "explain",
    "describe", "list", "show", "give", "provide",
    "share", "anything", "something", "everything",
    "interesting", "favorite", "best", "good", "nice",
    "important", "significant", "special", "different",
    "recent", "current", "typical", "usual",
    "regarding", "related", "relevant", "specific",
    "particular", "certain", "general", "common",
    "activities", "things", "stuff", "ideas",
    "information", "details", "reason", "reasons",
})


def _extract_content_keywords(query: str) -> list[str]:
    """Extract meaningful keywords from query, removing stopwords."""
    tokens = re.findall(r"[a-zA-Z0-9']+", query.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def _keyword_idf_weight(keyword: str) -> float:
    """IDF-like weight: rare/specific words get higher weight."""
    if keyword in COMMON_QUERY_WORDS:
        return 0.3
    if len(keyword) >= 6:
        return 1.5
    if len(keyword) >= 4:
        return 1.0
    return 0.7


def _compute_idf_keyword_overlap(query_keywords: list[str], content: str) -> float:
    """IDF-weighted keyword overlap: specific terms count more."""
    if not query_keywords:
        return 0.0
    content_lower = content.lower()
    total_weight = sum(_keyword_idf_weight(kw) for kw in query_keywords)
    if total_weight == 0:
        return 0.0
    matched_weight = sum(
        _keyword_idf_weight(kw) for kw in query_keywords if kw in content_lower
    )
    return matched_weight / total_weight


def _extract_bigrams(query: str) -> list[str]:
    """Extract consecutive bigrams from non-stopword tokens."""
    tokens = re.findall(r"[a-zA-Z0-9']+", query.lower())
    content_tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    bigrams = []
    for i in range(len(content_tokens) - 1):
        bigrams.append(f"{content_tokens[i]} {content_tokens[i+1]}")
    all_tokens = [t for t in tokens if len(t) > 1]
    for i in range(len(all_tokens) - 1):
        bg = f"{all_tokens[i]} {all_tokens[i+1]}"
        if bg not in bigrams:
            bigrams.append(bg)
    return bigrams


def _compute_bigram_overlap(bigrams: list[str], content: str) -> float:
    """IDF-weighted bigram overlap: content-rich bigrams count more."""
    if not bigrams:
        return 0.0
    content_lower = content.lower()
    total_weight = 0.0
    matched_weight = 0.0
    for bg in bigrams:
        parts = bg.split()
        is_specific = (
            len(parts) == 2
            and all(len(w) >= 4 for w in parts)
            and not any(w in COMMON_QUERY_WORDS for w in parts)
        )
        w = 3.0 if is_specific else 1.0
        total_weight += w
        if bg in content_lower:
            matched_weight += w
    return matched_weight / total_weight if total_weight > 0 else 0.0


def _compute_keyword_overlap(query_keywords: list[str], content: str) -> float:
    """Fraction of query keywords found in content text."""
    if not query_keywords:
        return 0.0
    content_lower = content.lower()
    matches = sum(1 for kw in query_keywords if kw in content_lower)
    return matches / len(query_keywords)


def _extract_quoted_phrases(query: str) -> list[str]:
    """Extract quoted phrases."""
    return re.findall(r"['\"]([^'\"]+)['\"]", query)


def _extract_noun_phrases(query: str) -> list[str]:
    """Extract 2-3 word noun phrases with stopword preservation.

    Sliding window over all tokens so 'high school' is captured
    even though 'high' is a stopword. Requires both words >= 4 chars
    and at least one high-IDF word (non-stopword, >= 5 chars).
    """
    tokens = re.findall(r"[a-zA-Z']+", query.lower())
    all_tokens = [t for t in tokens if len(t) >= 4]
    phrases = []
    _low_idf = STOPWORDS | COMMON_QUERY_WORDS
    for i in range(len(all_tokens) - 1):
        w1, w2 = all_tokens[i], all_tokens[i + 1]
        has_high_idf = (
            (w1 not in _low_idf and len(w1) >= 5)
            or (w2 not in _low_idf and len(w2) >= 5)
        )
        if has_high_idf:
            phrases.append(f"{w1} {w2}")
    for i in range(len(all_tokens) - 2):
        w1, w2, w3 = all_tokens[i], all_tokens[i + 1], all_tokens[i + 2]
        high_idf_count = sum(
            1
            for w in (w1, w2, w3)
            if w not in _low_idf and len(w) >= 5
        )
        if high_idf_count >= 2:
            phrases.append(f"{w1} {w2} {w3}")
    return phrases


def _extract_proper_nouns(query: str) -> list[str]:
    """Extract capitalized words that look like proper nouns."""
    words = query.split()
    nouns = []
    for i, w in enumerate(words):
        cleaned = re.sub(r"[^a-zA-Z]", "", w)
        if cleaned and cleaned[0].isupper() and i > 0 and len(cleaned) > 1:
            if cleaned.lower() not in STOPWORDS:
                nouns.append(cleaned.lower())
    return nouns


def _is_temporal_query(query: str) -> bool:
    """Check if query involves temporal reasoning."""
    tokens = set(re.findall(r"[a-zA-Z]+", query.lower()))
    return bool(tokens & TEMPORAL_WORDS)


def _extract_temporal_context_keywords(query: str) -> list[str]:
    """Extract non-temporal content keywords for temporal queries."""
    tokens = re.findall(r"[a-zA-Z0-9']+", query.lower())
    exclude = STOPWORDS | TEMPORAL_WORDS | {
        "how", "many", "long", "passed", "since", "before",
    }
    return [t for t in tokens if t not in exclude and len(t) > 2]


def sanitize_fts(query: str) -> str:
    """
    FTS5 MATCH構文のために入力を正規化する。
    空文字は reject し、各トークンからダブルクォートを除去する。
    """
    raw_tokens = query.strip().split()
    tokens = [token.replace('"', "").strip() for token in raw_tokens]
    tokens = [token for token in tokens if token]

    if not tokens:
        raise ValueError("FTS query must not be empty")

    return " ".join(f'"{token}"' for token in tokens)


def _sanitize_fts_or(query: str) -> str:
    """FTS5 query using OR mode with stopword removal and noun phrase expansion.

    Phase 2 enhancement (#105 TrueMemory-inspired):
    Adds noun phrases to the FTS5 query for improved recall without LLM dependency.
    """
    keywords = _extract_content_keywords(query)
    if not keywords:
        raw_tokens = query.strip().split()
        tokens = [t.replace('"', '').strip() for t in raw_tokens if t.strip()]
        if not tokens:
            raise ValueError("FTS query must not be empty")
        keywords = tokens[:3]

    # Phase 2: Noun phrase expansion for FTS5 recall improvement
    noun_phrases = _extract_noun_phrases(query)
    keyword_set = set(keywords)
    expanded_phrases: list[str] = []
    for np in noun_phrases:
        np_words = set(np.split())
        # Skip if all words already covered by individual keywords
        if not np_words.issubset(keyword_set):
            expanded_phrases.append(np)

    terms = [f'"{kw}"' for kw in keywords]
    for phrase in expanded_phrases[:3]:  # Cap at 3 noun phrases to avoid FTS5 overload
        # Noun phrases as NEAR queries for phrase proximity matching
        words = phrase.split()
        if len(words) == 2:
            terms.append(f'NEAR("{words[0]}" "{words[1]}", 2)')
        elif len(words) == 3:
            terms.append(f'NEAR("{words[0]}" "{words[1]}" "{words[2]}", 3)')

    return " OR ".join(terms)


def _build_why_included(
    obs: dict[str, Any],
    fts_rank_by_id: dict[int, int],
    vec_rank_by_id: dict[int, int],
) -> dict[str, Any]:
    """Explain why this memory was retrieved.

    Governance/explainability: every result carries the channels that surfaced
    it (keyword vs semantic, with rank), the boost signals that fired, its score
    breakdown, and its provenance — so an agent (or auditor) can see *why* a
    memory was included, not just that it was.
    """
    obs_id = int(obs["id"])
    channels: list[dict[str, Any]] = []
    fts_rank = fts_rank_by_id.get(obs_id)
    vec_rank = vec_rank_by_id.get(obs_id)
    if fts_rank is not None:
        channels.append({"channel": "keyword", "rank": fts_rank})
    if vec_rank is not None:
        channels.append({"channel": "semantic", "rank": vec_rank})

    signals = list(obs.get("_why_signals", []))
    if bool(obs.get("graph_boosted")):
        signals.append(f"graph-reachable (intent={obs.get('query_intent')})")

    # Provenance is out-of-band trust context; tolerate str or dict metadata.
    provenance = None
    metadata = obs.get("metadata")
    if isinstance(metadata, str) and metadata:
        try:
            metadata = json.loads(metadata)
        except (ValueError, TypeError):
            metadata = None
    if isinstance(metadata, dict):
        provenance = metadata.get("provenance")

    summary_bits: list[str] = []
    if channels:
        summary_bits.append(" + ".join(f"{c['channel']} rank {c['rank']}" for c in channels))
    if signals:
        summary_bits.append(", ".join(signals))
    summary = ("matched via " + "; ".join(summary_bits)) if summary_bits else "ranked by hybrid fusion"

    return {
        "summary": summary,
        "channels": channels,
        "signals": signals,
        "score": {
            "rrf_score": round(float(obs.get("rrf_score", 0.0)), 6),
            "boost": round(float(obs.get("boost", 1.0)), 4),
            "effective_strength": round(float(obs.get("effective_strength", 1.0)), 4),
            "effective_score": round(float(obs.get("effective_score", 0.0)), 6),
            "graph_boosted": bool(obs.get("graph_boosted", False)),
        },
        "query": {
            "complexity": obs.get("query_complexity"),
            "intent": obs.get("query_intent"),
        },
        "provenance": provenance,
    }


def rrf_hybrid_search(
    db: sqlite3.Connection,
    query: str,
    query_embedding: list[float],
    project: str | None = None,
    metadata_filter: dict[str, object] | None = None,
    limit: int = 10,
    k: int = 30,
    fts_weight: float = 0.40,
    vec_weight: float = 0.60,
    keyword_boost: float = 0.3,
    bigram_boost: float = 0.5,
    proper_noun_boost: float = 0.3,
    quoted_phrase_boost: float = 0.3,
    temporal_content_boost: float = 0.3,
    focused_embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    """
    Enhanced RRF hybrid search: FTS5 OR-mode + sqlite-vec + multi-signal boosting.

    Pipeline:
    0. Query complexity classification (L1/L2/L3) → adaptive weight selection
    1. FTS5 OR-mode → broad keyword candidates
    2. sqlite-vec → semantic similarity candidates
    3. RRF score fusion (k=30, adaptive weights)
    4. IDF-weighted keyword overlap boost
    5. Bigram / proper noun / temporal context boost
    6. Decay-adjusted effective strength

    Project filter uses adaptive widening on vec side.
    """
    MAX_SEARCH_LIMIT = 500
    VEC_INITIAL_MULTIPLIER = 10
    VEC_MAX_K = 4096
    limit = min(limit, MAX_SEARCH_LIMIT)

    # Phase 1: Adaptive Query-Aware Retrieval (#110 SimpleMem)
    # Only override weights when caller used defaults (0.40/0.60)
    query_complexity = classify_query_complexity(query)
    caller_used_defaults = (
        abs(fts_weight - 0.40) < 1e-9 and abs(vec_weight - 0.60) < 1e-9
    )
    if caller_used_defaults:
        adaptive = _COMPLEXITY_WEIGHTS[query_complexity]
        fts_weight = adaptive["fts_weight"]
        vec_weight = adaptive["vec_weight"]

    # Phase 3a: Intent-Aware Router (#131 MAGMA)
    # Only apply intent overrides when caller used default boost values
    _DEFAULT_BOOSTS = {
        "keyword_boost": 0.3,
        "bigram_boost": 0.5,
        "proper_noun_boost": 0.3,
        "temporal_content_boost": 0.3,
    }
    query_intent = classify_query_intent(query)
    intent_overrides = _INTENT_BOOST_OVERRIDES.get(query_intent, {})
    if "keyword_boost" in intent_overrides and abs(keyword_boost - _DEFAULT_BOOSTS["keyword_boost"]) < 1e-9:
        keyword_boost = intent_overrides["keyword_boost"]
    if "bigram_boost" in intent_overrides and abs(bigram_boost - _DEFAULT_BOOSTS["bigram_boost"]) < 1e-9:
        bigram_boost = intent_overrides["bigram_boost"]
    if "proper_noun_boost" in intent_overrides and abs(proper_noun_boost - _DEFAULT_BOOSTS["proper_noun_boost"]) < 1e-9:
        proper_noun_boost = intent_overrides["proper_noun_boost"]
    if "temporal_content_boost" in intent_overrides and abs(temporal_content_boost - _DEFAULT_BOOSTS["temporal_content_boost"]) < 1e-9:
        temporal_content_boost = intent_overrides["temporal_content_boost"]
    if "vec_weight_delta" in intent_overrides and caller_used_defaults:
        delta = intent_overrides["vec_weight_delta"]
        vec_weight = min(vec_weight + delta, 0.85)
        fts_weight = max(fts_weight - delta, 0.15)

    fts_match = _sanitize_fts_or(query)
    if project:
        safe_project = project.replace('"', "").strip()
        if safe_project:
            fts_match = f'({fts_match}) AND project:"{safe_project}"'

    fts_sql = """
        SELECT rowid,
               row_number() OVER (ORDER BY rank) as rank_num
        FROM observations_fts
        WHERE observations_fts MATCH ?
        LIMIT ?
    """
    try:
        fts_rows = db.execute(
            fts_sql, [fts_match, min(limit * VEC_INITIAL_MULTIPLIER, VEC_MAX_K)]
        ).fetchall()
    except sqlite3.OperationalError as fts_err:
        import logging as _log
        _log.getLogger(__name__).warning("FTS query failed: %s", fts_err)
        fts_rows = []

    vec_emb = focused_embedding if focused_embedding else query_embedding

    def fetch_vec_rows(vec_limit: int) -> list[sqlite3.Row]:
        vec_sql = """
            SELECT rowid,
                   row_number() OVER (ORDER BY distance) as rank_num
            FROM vec_observations
            WHERE embedding MATCH ?
              AND k = ?
        """
        return db.execute(
            vec_sql, [serialize_f32(vec_emb), vec_limit]
        ).fetchall()

    def compute_rrf(
        fts: Sequence[sqlite3.Row], vec: Sequence[sqlite3.Row]
    ) -> dict[int, float]:
        scores: dict[int, float] = {}
        for rowid, rank_num in fts:
            scores[rowid] = scores.get(rowid, 0.0) + fts_weight / (k + rank_num)
        for rowid, rank_num in vec:
            scores[rowid] = scores.get(rowid, 0.0) + vec_weight / (k + rank_num)
        return scores

    vec_k = min(limit * VEC_INITIAL_MULTIPLIER, VEC_MAX_K)
    try:
        vec_rows = fetch_vec_rows(vec_k)
    except sqlite3.OperationalError as vec_err:
        import logging as _log
        _log.getLogger(__name__).warning("Vec query failed: %s", vec_err)
        vec_rows = []
    scores = compute_rrf(fts_rows, vec_rows)

    if project:
        while vec_k < VEC_MAX_K:
            ranked_ids = sorted(scores, key=scores.__getitem__, reverse=True)
            in_project_count = _count_project_matches(db, ranked_ids, project)
            if in_project_count >= limit:
                break
            if len(vec_rows) < vec_k:
                break
            vec_k = min(vec_k * 2, VEC_MAX_K)
            try:
                vec_rows = fetch_vec_rows(vec_k)
            except sqlite3.OperationalError:
                break
            scores = compute_rrf(fts_rows, vec_rows)

    ranked_ids = sorted(scores, key=scores.__getitem__, reverse=True)
    if not ranked_ids:
        return []

    # Rank of each candidate within each channel — the raw evidence for
    # `why_included` (which channel surfaced this memory, and how strongly).
    fts_rank_by_id = {int(r[0]): int(r[1]) for r in fts_rows}
    vec_rank_by_id = {int(r[0]): int(r[1]) for r in vec_rows}

    query_keywords = _extract_content_keywords(query)
    query_bigrams = _extract_bigrams(query)
    quoted_phrases = _extract_quoted_phrases(query)
    proper_nouns = _extract_proper_nouns(query)
    noun_phrases = _extract_noun_phrases(query)
    temporal = _is_temporal_query(query)
    temporal_keywords = _extract_temporal_context_keywords(query) if temporal else []

    candidate_ids = ranked_ids[:limit * 8]

    where_clauses = ["o.id IN ({})".format(",".join("?" * len(candidate_ids)))]
    params: list[object] = list(candidate_ids)
    if project:
        where_clauses.append("o.project = ?")
        params.append(project)
    if metadata_filter:
        for mkey, mvalue in metadata_filter.items():
            where_clauses.append("json_extract(o.metadata, ?) = ?")
            params.extend([f"$.{mkey}", mvalue])

    fetch_sql = f"""
        SELECT o.*
        FROM observations o
        WHERE {" AND ".join(where_clauses)}
    """
    rows = db.execute(fetch_sql, params).fetchall()

    results: list[dict[str, Any]] = []
    for row in rows:
        obs = dict(row)
        obs_id = int(obs["id"])
        base_score = scores.get(obs_id, 0.0)
        created_at = _parse_datetime_value(obs["created_at"], "created_at")
        last_accessed = _parse_optional_datetime_value(
            obs.get("last_accessed"),
            "last_accessed",
        )
        obs["created_at"] = created_at
        obs["last_accessed"] = last_accessed

        content = str(obs.get("content", ""))
        content_lower = content.lower()

        # `signals` records which boost sources fired, for `why_included`.
        signals: list[str] = []
        kw_overlap = _compute_idf_keyword_overlap(query_keywords, content)
        bg_overlap = _compute_bigram_overlap(query_bigrams, content)
        boost = 1.0 + keyword_boost * kw_overlap + bigram_boost * bg_overlap
        if kw_overlap > 0:
            signals.append(f"keyword overlap ({kw_overlap:.2f})")
        if bg_overlap > 0:
            signals.append(f"bigram overlap ({bg_overlap:.2f})")

        for phrase in quoted_phrases:
            if phrase.lower() in content_lower:
                boost += quoted_phrase_boost
                signals.append(f'quoted phrase "{phrase}"')

        for noun in proper_nouns:
            if noun in content_lower:
                boost += proper_noun_boost
                signals.append(f'proper noun "{noun}"')

        np_boost_total = 0.0
        for np in noun_phrases:
            if np in content_lower:
                np_boost_total += 1.0
        boost += min(np_boost_total, 1.0)
        if np_boost_total > 0:
            signals.append("noun-phrase match")

        if temporal and temporal_keywords:
            t_overlap = _compute_keyword_overlap(temporal_keywords, content)
            boost += (temporal_content_boost * 2.0) * t_overlap
            if t_overlap > 0:
                signals.append("temporal context")

        effective_strength = current_strength(
            float(obs["base_strength"]),
            created_at,
            last_accessed,
            int(obs["access_count"]),
        )

        obs["rrf_score"] = base_score
        obs["boost"] = boost
        obs["effective_strength"] = effective_strength
        obs["effective_score"] = base_score * boost * effective_strength
        obs["query_complexity"] = query_complexity
        obs["query_intent"] = query_intent
        obs["_why_signals"] = signals
        results.append(obs)

    # Phase 3b: Graph-boosted reranking (#131 MAGMA)
    # Boost results reachable via intent-relevant graph edges
    if query_intent != INTENT_GENERAL and results:
        try:
            from .graph import graph_search as _graph_search
            top_ids = [int(r["id"]) for r in sorted(
                results, key=lambda x: float(x["effective_score"]), reverse=True
            )[:3]]
            graph_ids = set(_graph_search(db, query_intent, top_ids, max_depth=1))
            if graph_ids:
                _GRAPH_BOOST = 1.15
                for obs in results:
                    if int(obs["id"]) in graph_ids:
                        obs["effective_score"] = float(obs["effective_score"]) * _GRAPH_BOOST
                        obs["graph_boosted"] = True
        except Exception:
            pass  # graph layer is optional; degrade gracefully

    results.sort(key=lambda item: float(item["effective_score"]), reverse=True)
    top = results[:limit]
    for obs in top:
        obs["why_included"] = _build_why_included(obs, fts_rank_by_id, vec_rank_by_id)
        obs.pop("_why_signals", None)
    return top


def collapse_by_session(
    results: list[dict[str, Any]],
    limit: int = 10,
    agg_mode: str = "sum",
) -> list[dict[str, Any]]:
    """Collapse multi-chunk results by session_id.
    Each session gets scored by top1 + 0.2*top2 + 0.1*top3 + 0.05*rest.
    Returns the highest-scoring observation per session."""
    if not results:
        return []

    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        meta = row.get("metadata")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                meta = {}
        sid = ""
        if isinstance(meta, dict):
            sid = str(meta.get("session_id", ""))
        if not sid:
            sid = str(row.get("session_id", "")) or f"obs:{row.get('id', id(row))}"
        buckets.setdefault(sid, []).append(row)

    ranked: list[tuple[float, dict[str, Any]]] = []
    for group in buckets.values():
        group.sort(key=lambda r: float(r.get("effective_score", 0)), reverse=True)
        score = float(group[0].get("effective_score", 0))
        if agg_mode == "sum":
            if len(group) > 1:
                score += 0.2 * float(group[1].get("effective_score", 0))
            if len(group) > 2:
                score += 0.1 * float(group[2].get("effective_score", 0))
            for g in group[3:]:
                score += 0.05 * float(g.get("effective_score", 0))
        representative = group[0]
        representative["session_score"] = score
        ranked.append((score, representative))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [obs for _, obs in ranked[:limit]]
