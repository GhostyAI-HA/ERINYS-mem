"""FTS5 と sqlite-vec を RRF で統合する検索ロジック。

Enhanced v2: OR-mode FTS5, IDF-weighted keyword overlap,
bigram/proper-noun/temporal boosting を統合。
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timedelta, timezone
import sqlite3
from typing import Any, Sequence

from .decay import current_strength
from .embedding import serialize_f32


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
    """FTS5 query using OR mode with stopword removal for broader recall."""
    keywords = _extract_content_keywords(query)
    if not keywords:
        raw_tokens = query.strip().split()
        tokens = [t.replace('"', '').strip() for t in raw_tokens if t.strip()]
        if not tokens:
            raise ValueError("FTS query must not be empty")
        keywords = tokens[:3]
    return " OR ".join(f'"{kw}"' for kw in keywords)


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
    1. FTS5 OR-mode → broad keyword candidates
    2. sqlite-vec → semantic similarity candidates
    3. RRF score fusion (k=30, w=0.40/0.60)
    4. IDF-weighted keyword overlap boost
    5. Bigram / proper noun / temporal context boost
    6. Decay-adjusted effective strength

    Project filter uses adaptive widening on vec side.
    """
    MAX_SEARCH_LIMIT = 500
    VEC_INITIAL_MULTIPLIER = 10
    VEC_MAX_K = 4096
    limit = min(limit, MAX_SEARCH_LIMIT)

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
    vec_rows = fetch_vec_rows(vec_k)
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
            vec_rows = fetch_vec_rows(vec_k)
            scores = compute_rrf(fts_rows, vec_rows)

    ranked_ids = sorted(scores, key=scores.__getitem__, reverse=True)
    if not ranked_ids:
        return []

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

        kw_overlap = _compute_idf_keyword_overlap(query_keywords, content)
        bg_overlap = _compute_bigram_overlap(query_bigrams, content)
        boost = 1.0 + keyword_boost * kw_overlap + bigram_boost * bg_overlap

        for phrase in quoted_phrases:
            if phrase.lower() in content_lower:
                boost += quoted_phrase_boost

        for noun in proper_nouns:
            if noun in content_lower:
                boost += proper_noun_boost

        np_boost_total = 0.0
        for np in noun_phrases:
            if np in content_lower:
                np_boost_total += 1.0
        boost += min(np_boost_total, 1.0)

        if temporal and temporal_keywords:
            t_overlap = _compute_keyword_overlap(temporal_keywords, content)
            boost += (temporal_content_boost * 2.0) * t_overlap

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
        results.append(obs)

    results.sort(key=lambda item: float(item["effective_score"]), reverse=True)
    return results[:limit]


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
