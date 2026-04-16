"""
Enhanced search v2 for LongMemEval benchmark.
Improvements over v1:
  1. FTS5 OR mode with stopword filtering
  2. Keyword overlap boosting with bigram matching
  3. Wider candidate pool (limit*5 instead of limit*3)
  4. Content-length-aware normalization
  5. Contextual bigram/trigram phrase boosting
  6. Higher vec initial multiplier for better recall
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any, Sequence

from erinys_memory.embedding import serialize_f32


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


def extract_content_keywords(query: str) -> list[str]:
    """Extract meaningful keywords from query, removing stopwords."""
    tokens = re.findall(r"[a-zA-Z0-9']+", query.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


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


def keyword_idf_weight(keyword: str) -> float:
    """Assign IDF-like weight: rare/specific words get higher weight."""
    if keyword in COMMON_QUERY_WORDS:
        return 0.3
    if len(keyword) >= 6:
        return 1.5
    if len(keyword) >= 4:
        return 1.0
    return 0.7


def compute_idf_keyword_overlap(query_keywords: list[str], content: str) -> float:
    """IDF-weighted keyword overlap: specific terms count more."""
    if not query_keywords:
        return 0.0
    content_lower = content.lower()
    total_weight = sum(keyword_idf_weight(kw) for kw in query_keywords)
    if total_weight == 0:
        return 0.0
    matched_weight = sum(
        keyword_idf_weight(kw) for kw in query_keywords if kw in content_lower
    )
    return matched_weight / total_weight


def extract_bigrams(query: str) -> list[str]:
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


def compute_keyword_overlap(query_keywords: list[str], content: str) -> float:
    """Fraction of query keywords found in content text."""
    if not query_keywords:
        return 0.0
    content_lower = content.lower()
    matches = sum(1 for kw in query_keywords if kw in content_lower)
    return matches / len(query_keywords)


def compute_bigram_overlap(bigrams: list[str], content: str) -> float:
    """Fraction of bigrams found in content text."""
    if not bigrams:
        return 0.0
    content_lower = content.lower()
    matches = sum(1 for bg in bigrams if bg in content_lower)
    return matches / len(bigrams)


def extract_quoted_phrases(query: str) -> list[str]:
    """Extract single or double quoted phrases from query."""
    return re.findall(r"['\"]([^'\"]+)['\"]", query)


def extract_proper_nouns(query: str) -> list[str]:
    """Extract capitalized words that look like proper nouns."""
    words = query.split()
    nouns = []
    for i, w in enumerate(words):
        cleaned = re.sub(r"[^a-zA-Z]", "", w)
        if cleaned and cleaned[0].isupper() and i > 0 and len(cleaned) > 1:
            if cleaned.lower() not in STOPWORDS:
                nouns.append(cleaned.lower())
    return nouns


def is_temporal_query(query: str) -> bool:
    """Check if query involves temporal reasoning."""
    tokens = set(re.findall(r"[a-zA-Z]+", query.lower()))
    return bool(tokens & TEMPORAL_WORDS)


def extract_temporal_context_keywords(query: str) -> list[str]:
    """Extract non-temporal content keywords for temporal queries."""
    tokens = re.findall(r"[a-zA-Z0-9']+", query.lower())
    temporal_extended = STOPWORDS | TEMPORAL_WORDS | {
        "how", "many", "long", "passed", "since", "before",
    }
    return [t for t in tokens if t not in temporal_extended and len(t) > 2]


def sanitize_fts_or(query: str) -> str:
    """FTS5 query using OR mode with stopword removal."""
    keywords = extract_content_keywords(query)
    if not keywords:
        raw_tokens = query.strip().split()
        tokens = [t.replace('"', '').strip() for t in raw_tokens if t.strip()]
        if not tokens:
            raise ValueError("FTS query must not be empty")
        keywords = tokens[:3]
    return " OR ".join(f'"{kw}"' for kw in keywords)


def enhanced_hybrid_search_v2(
    db: sqlite3.Connection,
    query: str,
    query_embedding: list[float],
    project: str | None = None,
    limit: int = 10,
    k: int = 30,
    fts_weight: float = 0.40,
    vec_weight: float = 0.60,
    keyword_boost: float = 0.3,
    bigram_boost: float = 0.3,
    proper_noun_boost: float = 0.3,
    quoted_phrase_boost: float = 0.3,
    temporal_content_boost: float = 0.3,
) -> list[dict[str, Any]]:
    """
    Enhanced RRF v2 hybrid search with:
    - OR-mode FTS5 (broader candidate retrieval)
    - Keyword overlap boosting with bigram matching
    - Proper noun boosting
    - Temporal context boosting
    - Wider candidate pool
    """
    MAX_SEARCH_LIMIT = 500
    VEC_INITIAL_MULTIPLIER = 10
    VEC_MAX_K = 10000
    limit = min(limit, MAX_SEARCH_LIMIT)

    fts_match = sanitize_fts_or(query)
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
    except Exception:
        fts_rows = []

    def fetch_vec_rows(vec_limit: int) -> list[sqlite3.Row]:
        vec_sql = """
            SELECT rowid,
                   row_number() OVER (ORDER BY distance) as rank_num
            FROM vec_observations
            WHERE embedding MATCH ?
              AND k = ?
        """
        return db.execute(
            vec_sql, [serialize_f32(query_embedding), vec_limit]
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
            in_project_count = sum(
                1 for rid in ranked_ids
                if db.execute(
                    "SELECT 1 FROM observations WHERE id = ? AND project = ?",
                    [rid, project],
                ).fetchone()
            )
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

    query_keywords = extract_content_keywords(query)
    query_bigrams = extract_bigrams(query)
    quoted_phrases = extract_quoted_phrases(query)
    proper_nouns = extract_proper_nouns(query)
    temporal = is_temporal_query(query)
    temporal_keywords = extract_temporal_context_keywords(query) if temporal else []

    candidate_ids = ranked_ids[:limit * 8]

    where_clauses = ["o.id IN ({})".format(",".join("?" * len(candidate_ids)))]
    params: list[object] = list(candidate_ids)
    if project:
        where_clauses.append("o.project = ?")
        params.append(project)

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

        content = str(obs.get("content", ""))
        content_lower = content.lower()

        kw_overlap = compute_idf_keyword_overlap(query_keywords, content)
        bg_overlap = compute_bigram_overlap(query_bigrams, content)
        boost = 1.0 + keyword_boost * kw_overlap + bigram_boost * bg_overlap

        for phrase in quoted_phrases:
            if phrase.lower() in content_lower:
                boost += quoted_phrase_boost

        for noun in proper_nouns:
            if noun in content_lower:
                boost += proper_noun_boost

        if temporal and temporal_keywords:
            t_overlap = compute_keyword_overlap(temporal_keywords, content)
            boost += temporal_content_boost * t_overlap

        obs["rrf_score"] = base_score
        obs["keyword_overlap"] = kw_overlap
        obs["bigram_overlap"] = bg_overlap
        obs["boost"] = boost
        obs["effective_score"] = base_score * boost

        results.append(obs)

    results.sort(key=lambda x: float(x["effective_score"]), reverse=True)
    return results[:limit]
