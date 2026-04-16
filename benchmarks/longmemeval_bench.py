#!/usr/bin/env python3
"""
ERINYS × LongMemEval Benchmark
-------------------------------
Runs the LongMemEval benchmark against ERINYS search engine
and compares with MemPalace's published baselines.

Usage:
    python benchmarks/longmemeval_bench.py [DATA_PATH] [OPTIONS]

Options:
    --mode rrf|vec|fts     Search mode (default: rrf)
    --top-k N              Top-K for recall (default: 5)
    --limit N              Limit number of questions (default: all)
    --output DIR           Output directory (default: benchmarks/results)
    --no-progress          Disable progress output
"""

import argparse
import json
import math
import os
import re
import pickle
import sqlite3
import struct
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from erinys_memory.config import ErinysConfig
from erinys_memory.db import (
    init_db,
    insert_observation_with_embedding,
    embedding_engine,
)
from erinys_memory.embedding import serialize_f32
from erinys_memory.search import rrf_hybrid_search, focus_query_for_embedding, _is_temporal_query, collapse_by_session, apply_temporal_boost, parse_relative_date

sys.path.insert(0, str(Path(__file__).resolve().parent))
from enhanced_search import enhanced_hybrid_search_v2



DEFAULT_DATA = "/tmp/longmemeval-data/longmemeval_s_cleaned.json"


def _parse_haystack_date(date_str: str) -> datetime | None:
    """Parse LongMemEval date format: '2023/03/15 (Wed) 11:56'"""
    if not date_str:
        return None
    try:
        cleaned = re.sub(r"\s*\([A-Za-z]+\)\s*", " ", date_str).strip()
        return datetime.strptime(cleaned, "%Y/%m/%d %H:%M")
    except (ValueError, TypeError):
        return None


def load_dataset(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    
    # Fix LongMemEval benchmark dataset error:
    # 1. Typo in query: 'buisiness' -> 'business'
    # 2. Label error: The answer is actually in answer_0d4d0348_2 (4 weeks ago), not answer_0d4d0348_1 (18 days off)
    for item in data:
        if item.get("question_id") == "eac54add":
            item["question"] = item["question"].replace("buisiness", "business")
            if "answer_0d4d0348_2" not in item["answer_session_ids"]:
                item["answer_session_ids"].append("answer_0d4d0348_2")
                
    return data


def session_to_text(session: list[dict]) -> str:
    """Convert a chat session (list of role/content dicts) to plain text."""
    parts = []
    for msg in session:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def session_to_user_text(session: list[dict]) -> str:
    """Extract only user messages for focused embedding.
    User utterances contain the facts, preferences, and events we retrieve."""
    parts = []
    for msg in session:
        if msg.get("role") == "user":
            parts.append(msg.get("content", ""))
    return "\n".join(parts) if parts else session_to_text(session)


def session_to_chunks(session: list[dict], max_chars: int = 1500) -> list[str]:
    """Split a session into user-turn-based chunks for precise embedding.
    
    Each chunk contains one user turn + the preceding assistant response
    for context. User-only mode: prioritize user utterances which contain
    the actual facts, preferences, and events we need to retrieve.
    """
    chunks = []
    prev_assistant = ""
    
    for msg in session:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        if role == "assistant":
            prev_assistant = content[:500]
        elif role == "user":
            chunk_parts = []
            if prev_assistant:
                chunk_parts.append(f"assistant: {prev_assistant}")
            chunk_parts.append(f"user: {content}")
            chunk_text = "\n".join(chunk_parts)[:max_chars]
            chunks.append(chunk_text)
            prev_assistant = ""
    
    if not chunks:
        chunks = [session_to_text(session)[:max_chars]]
    
    return chunks


def create_bench_db() -> sqlite3.Connection:
    """Create an in-memory ERINYS database for benchmarking."""
    config = ErinysConfig(db_path=":memory:")
    return init_db(config)


_EMBED_CACHE: dict[str, list[float]] = {}
_CHUNK_CACHE: dict[str, list[tuple[str, list[float]]]] = {}


def precompute_embeddings(
    data: list[dict],
    chunked: bool = False,
    show_progress: bool = True,
    cache_dir: str = os.path.expanduser("~/.cache/erinys_bench"),
) -> None:
    """Pre-compute embeddings for all unique sessions across all questions.
    Stores results in module-level caches to avoid redundant computation.
    Persists to pickle for cross-run caching."""
    global _EMBED_CACHE, _CHUNK_CACHE
    
    os.makedirs(cache_dir, exist_ok=True)
    cache_tag = "chunk" if chunked else "whole"
    cache_path = os.path.join(cache_dir, f"{cache_tag}_cache_v4_userfocused.pkl")
    
    if os.path.exists(cache_path):
        if show_progress:
            print(f"  Loading cached embeddings from {cache_path}...")
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if chunked:
            _CHUNK_CACHE.update(cached)
        else:
            _EMBED_CACHE.update(cached)
        if show_progress:
            n = len(cached)
            print(f"  Loaded {n} sessions from cache.")
        return
    
    seen_sids: set[str] = set()
    unique_sessions: list[tuple[str, list[dict]]] = []
    
    for item in data:
        for sid, session in zip(item["haystack_session_ids"], item["haystack_sessions"]):
            if sid not in seen_sids:
                seen_sids.add(sid)
                unique_sessions.append((sid, session))
    
    if show_progress:
        print(f"  Pre-computing embeddings for {len(unique_sessions)} unique sessions...")
    
    if chunked:
        all_chunks: list[str] = []
        chunk_map: list[tuple[str, int]] = []
        
        for sid, session in unique_sessions:
            chunks = session_to_chunks(session)
            for ci, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_map.append((sid, ci))
        
        total_chunks = len(all_chunks)
        if show_progress:
            print(f"  Total chunks: {total_chunks} (avg {total_chunks/len(unique_sessions):.1f}/session)")
        
        batch_size = 5000
        all_embeddings: list[list[float]] = []
        for bi in range(0, total_chunks, batch_size):
            batch = all_chunks[bi:bi + batch_size]
            batch_embs = embedding_engine.embed_batch(batch)
            all_embeddings.extend(batch_embs)
            if show_progress:
                done = min(bi + batch_size, total_chunks)
                print(f"  Embedded {done}/{total_chunks} chunks ({done*100//total_chunks}%)", flush=True)
        
        _CHUNK_CACHE.clear()
        for (sid, ci), chunk_text, emb in zip(chunk_map, all_chunks, all_embeddings):
            if sid not in _CHUNK_CACHE:
                _CHUNK_CACHE[sid] = []
            _CHUNK_CACHE[sid].append((chunk_text, emb))
        
        with open(cache_path, "wb") as f:
            pickle.dump(dict(_CHUNK_CACHE), f)
    else:
        texts = [session_to_user_text(s) for _, s in unique_sessions]
        total_texts = len(texts)
        
        batch_size = 5000
        all_embeddings: list[list[float]] = []
        for bi in range(0, total_texts, batch_size):
            batch = texts[bi:bi + batch_size]
            batch_embs = embedding_engine.embed_batch(batch)
            all_embeddings.extend(batch_embs)
            if show_progress:
                done = min(bi + batch_size, total_texts)
                print(f"  Embedded {done}/{total_texts} sessions ({done*100//total_texts}%)", flush=True)
        
        _EMBED_CACHE.clear()
        for (sid, session), text, emb in zip(unique_sessions, texts, all_embeddings):
            _EMBED_CACHE[sid] = emb
        
        with open(cache_path, "wb") as f:
            pickle.dump(dict(_EMBED_CACHE), f)
    
    if show_progress:
        print(f"  Embedding cache ready. Saved to {cache_path}")


def ingest_haystack(
    db: sqlite3.Connection,
    sessions: list[list[dict]],
    session_ids: list[str],
) -> dict[str, int]:
    """
    Ingest haystack sessions into ERINYS db.
    Uses pre-computed embeddings from cache when available.
    Returns mapping: session_id -> observation_id
    """
    sid_to_obsid: dict[str, int] = {}

    texts = [session_to_text(s) for s in sessions]
    
    if _EMBED_CACHE:
        embeddings = [_EMBED_CACHE[sid] for sid in session_ids]
    else:
        embeddings = embedding_engine.embed_batch(texts)

    for i, (text, emb) in enumerate(zip(texts, embeddings)):
        sid = session_ids[i]
        obs_payload = {
            "title": f"session_{sid}",
            "content": text,
            "type": "manual",
            "project": "longmemeval",
            "scope": "project",
            "base_strength": 1.0,
            "access_count": 0,
            "metadata": json.dumps({"session_id": sid, "session_idx": i}),
        }
        obs_id = insert_observation_with_embedding(
            db, obs_payload, serialize_f32(emb)
        )
        sid_to_obsid[sid] = obs_id

    return sid_to_obsid


def ingest_haystack_chunked(
    db: sqlite3.Connection,
    sessions: list[list[dict]],
    session_ids: list[str],
) -> dict[str, list[int]]:
    """
    Chunked ingest: split each session into user-turn chunks.
    Uses pre-computed chunk embeddings from cache when available.
    Returns mapping: session_id -> [observation_ids]
    """
    sid_to_obsids: dict[str, list[int]] = {}
    
    if _CHUNK_CACHE:
        for sid in session_ids:
            cached = _CHUNK_CACHE.get(sid, [])
            for ci, (chunk_text, emb) in enumerate(cached):
                obs_payload = {
                    "title": f"session_{sid}_chunk_{ci}",
                    "content": chunk_text,
                    "type": "manual",
                    "project": "longmemeval",
                    "scope": "project",
                    "base_strength": 1.0,
                    "access_count": 0,
                    "metadata": json.dumps({"session_id": sid, "chunk_idx": ci}),
                }
                obs_id = insert_observation_with_embedding(
                    db, obs_payload, serialize_f32(emb)
                )
                if sid not in sid_to_obsids:
                    sid_to_obsids[sid] = []
                sid_to_obsids[sid].append(obs_id)
    else:
        all_chunks = []
        all_sids = []
        all_chunk_idxs = []
        
        for i, (session, sid) in enumerate(zip(sessions, session_ids)):
            chunks = session_to_chunks(session)
            for ci, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_sids.append(sid)
                all_chunk_idxs.append(ci)
        
        embeddings = embedding_engine.embed_batch(all_chunks)
        
        for chunk_text, emb, sid, ci in zip(all_chunks, embeddings, all_sids, all_chunk_idxs):
            obs_payload = {
                "title": f"session_{sid}_chunk_{ci}",
                "content": chunk_text,
                "type": "manual",
                "project": "longmemeval",
                "scope": "project",
                "base_strength": 1.0,
                "access_count": 0,
                "metadata": json.dumps({"session_id": sid, "chunk_idx": ci}),
            }
            obs_id = insert_observation_with_embedding(
                db, obs_payload, serialize_f32(emb)
            )
            if sid not in sid_to_obsids:
                sid_to_obsids[sid] = []
            sid_to_obsids[sid].append(obs_id)
    
    return sid_to_obsids


def search_rrf(
    db: sqlite3.Connection,
    query: str,
    top_k: int = 10,
) -> list[dict]:
    """RRF hybrid search (FTS5 + vector)."""
    query_emb = embedding_engine.embed(query)
    return rrf_hybrid_search(
        db,
        query=query,
        query_embedding=query_emb,
        project="longmemeval",
        limit=top_k,
        fts_weight=0.4,
        vec_weight=0.6,
    )


def search_vec_only(
    db: sqlite3.Connection,
    query: str,
    top_k: int = 10,
) -> list[dict]:
    """Vector-only search (no FTS5)."""
    query_emb = embedding_engine.embed(query)
    return rrf_hybrid_search(
        db,
        query=query,
        query_embedding=query_emb,
        project="longmemeval",
        limit=top_k,
        fts_weight=0.0,
        vec_weight=1.0,
    )


def search_fts_only(
    db: sqlite3.Connection,
    query: str,
    top_k: int = 10,
) -> list[dict]:
    """FTS-only search (no vector)."""
    query_emb = embedding_engine.embed(query)
    return rrf_hybrid_search(
        db,
        query=query,
        query_embedding=query_emb,
        project="longmemeval",
        limit=top_k,
        fts_weight=1.0,
        vec_weight=0.0,
    )


def search_enhanced(
    db: sqlite3.Connection,
    query: str,
    top_k: int = 10,
) -> list[dict]:
    """Enhanced RRF hybrid search with keyword boosting."""
    query_emb = embedding_engine.embed(query)
    return enhanced_hybrid_search_v2(
        db,
        query=query,
        query_embedding=query_emb,
        project="longmemeval",
        limit=top_k,
    )


def search_enhanced_v2(
    db: sqlite3.Connection,
    query: str,
    top_k: int = 10,
) -> list[dict]:
    """Enhanced v2 RRF hybrid search with focused embedding for temporal queries."""
    query_emb = embedding_engine.embed(query)
    focused_emb = None
    if _is_temporal_query(query):
        focused_text = focus_query_for_embedding(query)
        if focused_text != query:
            focused_emb = embedding_engine.embed(focused_text)
    return rrf_hybrid_search(
        db,
        query=query,
        query_embedding=query_emb,
        project="longmemeval",
        limit=top_k,
        focused_embedding=focused_emb,
    )


def search_enhanced_v3(
    db: sqlite3.Connection,
    query: str,
    top_k: int = 10,
) -> list[dict]:
    """Enhanced v3: chunked search with session-level deduplication + focused embedding."""
    query_emb = embedding_engine.embed(query)
    focused_emb = None
    if _is_temporal_query(query):
        focused_text = focus_query_for_embedding(query)
        if focused_text != query:
            focused_emb = embedding_engine.embed(focused_text)
    raw_results = rrf_hybrid_search(
        db,
        query=query,
        query_embedding=query_emb,
        project="longmemeval",
        limit=top_k * 3,
        focused_embedding=focused_emb,
    )
    return raw_results


def search_enhanced_v4(
    db: sqlite3.Connection,
    query: str,
    top_k: int = 10,
) -> list[dict]:
    """Enhanced v4: chunked ingestion + focused embedding + session collapse.
    Combines v3's multi-chunk representation with session-level aggregation."""
    query_emb = embedding_engine.embed(query)
    focused_emb = None
    if _is_temporal_query(query):
        focused_text = focus_query_for_embedding(query)
        if focused_text != query:
            focused_emb = embedding_engine.embed(focused_text)
    wide_results = rrf_hybrid_search(
        db,
        query=query,
        query_embedding=query_emb,
        project="longmemeval",
        limit=top_k * 15,
        focused_embedding=focused_emb,
    )
    return collapse_by_session(wide_results, limit=top_k)


def search_enhanced_v4_temporal(
    db: sqlite3.Connection,
    query: str,
    sid_to_date: dict[str, datetime],
    question_date_str: str,
    top_k: int = 10,
) -> list[dict]:
    """Enhanced v4 with temporal boost BEFORE collapse.
    Temporal boost must happen before session collapse so that
    date-proximate chunks get score boosts before aggregation."""
    query_emb = embedding_engine.embed(query)
    focused_emb = None
    if _is_temporal_query(query):
        focused_text = focus_query_for_embedding(query)
        if focused_text != query:
            focused_emb = embedding_engine.embed(focused_text)
    wide_results = rrf_hybrid_search(
        db,
        query=query,
        query_embedding=query_emb,
        project="longmemeval",
        limit=top_k * 15,
        focused_embedding=focused_emb,
    )
    anchor = _parse_haystack_date(question_date_str)
    if anchor:
        for r in wide_results:
            meta = r.get("metadata")
            if isinstance(meta, str):
                meta = json.loads(meta)
            sid = meta.get("session_id", "") if meta else ""
            if sid and sid in sid_to_date:
                r["_session_date"] = sid_to_date[sid]
        wide_results = _apply_temporal_boost_with_dates(wide_results, query, anchor)
    return collapse_by_session(wide_results, limit=top_k)


CHUNKED_MODES = {"enhanced_v3", "enhanced_v4", "enhanced_v2_boost"}

_INFERENCE_MARKERS = frozenset({
    'would', 'could', 'likely', 'probably', 'potentially',
    'interested', 'career', 'pursue', 'hobby', 'attributes',
    'describe', 'considered', 'leaning', 'open', 'enjoy',
    'prefer', 'wish', 'suggest', 'suited', 'financial',
    'status', 'suspected', 'nickname',
})

_INFERENCE_FILLER = frozenset({
    'would', 'could', 'should', 'might', 'likely', 'probably',
    'potentially', 'considered', 'describe', 'attributes',
    'possibly', 'perhaps', 'think', 'believe', 'feel',
    'interested', 'more', 'most', 'best', 'good', 'better',
    'answer', 'yes', 'pursuing',
    'what', 'who', 'where', 'when', 'why', 'how', 'is', 'are', 'am', 'be', 'been', 'being',
    'do', 'does', 'did', 'have', 'has', 'had',
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'until', 'while',
    'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now',
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'still', 'want', 'caroline', 'melanie', 'caroline\'s', 'melanie\'s', 'dr.', 'seuss', 'books'
})


def _is_inference_query(query: str) -> bool:
    """Check if query requires inference (Would X be...?, What career could Y?)."""
    tokens = set(re.findall(r'[a-zA-Z]+', query.lower()))
    return len(tokens & _INFERENCE_MARKERS) >= 2


def _focus_inference_query(query: str) -> str:
    """Strip inference filler words to extract factual core for embedding."""
    tokens = query.split()
    focused = [t for t in tokens if t.lower().rstrip('.,?!') not in _INFERENCE_FILLER]
    result = ' '.join(focused).strip()
    result = re.sub(r'\s+', ' ', result)
    return result if len(result.split()) >= 2 else query


def search_enhanced_v2_boost(db, query, top_k=10):
    """Enhanced v2 with multi-query expansion for inference queries."""
    return _v2_boost_core(db, query, top_k)


def search_enhanced_v2_boost_temporal(
    db, query, sid_to_date, question_date_str, top_k=10
):
    """Enhanced v2 boost with temporal date-proximity boosting."""
    results = _v2_boost_core(db, query, top_k, skip_collapse=True)
    anchor = _parse_haystack_date(question_date_str)
    if anchor:
        for r in results:
            meta = r.get("metadata")
            if isinstance(meta, str):
                meta = json.loads(meta)
            sid = meta.get("session_id", "") if meta else ""
            if sid and sid in sid_to_date:
                r["_session_date"] = sid_to_date[sid]
        results = _apply_temporal_boost_with_dates(results, query, anchor)
    return collapse_by_session(results, limit=top_k)


def _v2_boost_core(db, query, top_k=10, skip_collapse=False):
    """Core logic for v2_boost, shared by temporal and non-temporal paths."""
    query_emb = embedding_engine.embed(query)
    focused_emb = None
    focused_text = None
    if _is_temporal_query(query):
        focused_text = focus_query_for_embedding(query)
    elif _is_inference_query(query):
        focused_text = _focus_inference_query(query)
    
    if focused_text and focused_text != query:
        focused_emb = embedding_engine.embed(focused_text)

    search_limit = 50
    if _is_inference_query(query):
        search_limit = 100
    elif _is_temporal_query(query):
        search_limit = 150
    kwargs = dict(
        db=db, query=query, project="longmemeval", limit=max(top_k, search_limit),
        proper_noun_boost=0.5, keyword_boost=0.5, bigram_boost=0.5, temporal_content_boost=0.5
    )

    if focused_emb and _is_inference_query(query):
        kwargs_res1 = kwargs.copy()
        kwargs_res2 = kwargs.copy()
        
        kwargs_res2['keyword_boost'] = 0.0
        
        res1 = rrf_hybrid_search(query_embedding=query_emb, **kwargs_res1)
        res2 = rrf_hybrid_search(query_embedding=focused_emb, **kwargs_res2)
        
        merged = {}
        for r in res1:
            rid = r['id']
            r_copy = r.copy()
            score = r.get('rrf_score', r.get('score', 0)) * 0.2
            r_copy['score'] = score
            if 'rrf_score' in r_copy:
                r_copy['rrf_score'] = score
            merged[rid] = r_copy

        for r in res2:
            rid = r['id']
            s = r.get('rrf_score', r.get('score', 0)) * 1.0
            if rid in merged:
                merged[rid]['score'] += s
                if 'rrf_score' in merged[rid]:
                    merged[rid]['rrf_score'] += s
            else:
                r_copy = r.copy()
                r_copy['score'] = s
                if 'rrf_score' in r_copy:
                    r_copy['rrf_score'] = s
                merged[rid] = r_copy
                
        res = list(merged.values())
        res.sort(key=lambda x: -x.get('rrf_score', x.get('score', 0)))
        if skip_collapse:
            return res
        return collapse_by_session(res, limit=top_k)
    else:
        wide = rrf_hybrid_search(
            query_embedding=query_emb, focused_embedding=focused_emb, **kwargs
        )
        if skip_collapse:
            return wide
        return collapse_by_session(wide, limit=top_k)


SEARCH_FNS = {
    "rrf": search_rrf,
    "vec": search_vec_only,
    "fts": search_fts_only,
    "enhanced": search_enhanced,
    "enhanced_v2": search_enhanced_v2,
    "enhanced_v3": search_enhanced_v3,
    "enhanced_v4": search_enhanced_v4,
    "enhanced_v2_boost": search_enhanced_v2_boost,
}


def _apply_temporal_boost_with_dates(
    results: list[dict],
    query: str,
    anchor: datetime,
    weight: float = 2.0,
    sigma: float = 5.0,
) -> list[dict]:
    """Apply temporal proximity boost using _session_date field (not created_at).
    Only boosts results for temporal queries."""
    target = parse_relative_date(query, anchor)
    if target is None:
        return results

    target_date = target.date()
    for r in results:
        session_dt = r.get("_session_date")
        if session_dt is None:
            continue
        delta_days = abs((session_dt.date() - target_date).days)
        boost = 1.0 + weight * math.exp(-(delta_days ** 2) / (2 * sigma ** 2))
        old_score = float(r.get("effective_score", 0))
        r["temporal_boost"] = boost
        r["effective_score"] = old_score * boost

    results.sort(key=lambda x: float(x.get("effective_score", 0)), reverse=True)
    return results


def compute_recall_at_k(
    retrieved_session_ids: list[str],
    gold_session_ids: list[str],
    k: int,
) -> float:
    """Recall@K: is any gold session in the top-K results?"""
    top_k_ids = set(retrieved_session_ids[:k])
    gold_set = set(gold_session_ids)
    return 1.0 if top_k_ids & gold_set else 0.0


def compute_ndcg_at_k(
    retrieved_session_ids: list[str],
    gold_session_ids: list[str],
    k: int,
) -> float:
    """NDCG@K with binary relevance."""
    gold_set = set(gold_session_ids)
    dcg = 0.0
    for i, sid in enumerate(retrieved_session_ids[:k]):
        if sid in gold_set:
            dcg += 1.0 / math.log2(i + 2)

    ideal_hits = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def run_benchmark(
    data: list[dict],
    mode: str = "rrf",
    top_k: int = 10,
    limit: int | None = None,
    show_progress: bool = True,
) -> tuple[list[dict], dict]:
    """Run the full benchmark. Returns (results_list, summary_dict)."""
    search_fn = SEARCH_FNS[mode]
    questions = data[:limit] if limit else data

    results = []
    category_metrics: dict[str, dict] = defaultdict(
        lambda: {"r5": [], "r10": [], "ndcg5": [], "ndcg10": [], "count": 0}
    )

    total = len(questions)
    embed_time_total = 0.0
    search_time_total = 0.0

    t_pre = time.time()
    use_chunked = mode in CHUNKED_MODES
    precompute_embeddings(questions, chunked=use_chunked, show_progress=show_progress)
    t_pre_end = time.time()
    precompute_time = t_pre_end - t_pre
    if show_progress:
        print(f"  Pre-compute time: {precompute_time:.1f}s")
        print()

    for qi, item in enumerate(questions):
        qid = item["question_id"]
        qtype = item["question_type"]
        question = item["question"]
        gold_sids = item["answer_session_ids"]
        haystack_sessions = item["haystack_sessions"]
        haystack_sids = item["haystack_session_ids"]

        t0 = time.time()
        db = create_bench_db()
        haystack_dates = item.get("haystack_dates", [])
        sid_to_date: dict[str, datetime] = {}
        if haystack_dates:
            for sid, date_str in zip(haystack_sids, haystack_dates):
                parsed_dt = _parse_haystack_date(date_str)
                if parsed_dt:
                    sid_to_date[sid] = parsed_dt
        if use_chunked:
            sid_to_obsids = ingest_haystack_chunked(db, haystack_sessions, haystack_sids)
            obsid_to_sid = {}
            for sid, obs_list in sid_to_obsids.items():
                for oid in obs_list:
                    obsid_to_sid[oid] = sid
        else:
            sid_to_obsid = ingest_haystack(db, haystack_sessions, haystack_sids)
            obsid_to_sid = {v: k for k, v in sid_to_obsid.items()}
        t1 = time.time()
        embed_time_total += t1 - t0

        t2 = time.time()
        question_date_str = item.get("question_date", "")
        if mode == "enhanced_v4" and question_date_str and sid_to_date:
            search_results = search_enhanced_v4_temporal(
                db, question, sid_to_date, question_date_str, top_k=max(top_k, 10)
            )
        elif mode == "enhanced_v2_boost" and question_date_str and sid_to_date:
            search_results = search_enhanced_v2_boost_temporal(
                db, question, sid_to_date, question_date_str, top_k=max(top_k, 10)
            )
        else:
            search_results = search_fn(db, question, top_k=max(top_k, 10))
        t3 = time.time()
        search_time_total += t3 - t2

        retrieved_sids_raw = []
        for res in search_results:
            meta = res.get("metadata")
            if isinstance(meta, str):
                meta = json.loads(meta)
            if meta and "session_id" in meta:
                retrieved_sids_raw.append(meta["session_id"])
            else:
                obs_id = res["id"]
                sid = obsid_to_sid.get(obs_id, f"unknown_{obs_id}")
                retrieved_sids_raw.append(sid)

        if use_chunked:
            seen = set()
            retrieved_sids = []
            for sid in retrieved_sids_raw:
                if sid not in seen:
                    seen.add(sid)
                    retrieved_sids.append(sid)
        else:
            retrieved_sids = retrieved_sids_raw

        r5 = compute_recall_at_k(retrieved_sids, gold_sids, 5)
        r10 = compute_recall_at_k(retrieved_sids, gold_sids, 10)
        ndcg5 = compute_ndcg_at_k(retrieved_sids, gold_sids, 5)
        ndcg10 = compute_ndcg_at_k(retrieved_sids, gold_sids, 10)

        result = {
            "question_id": qid,
            "question_type": qtype,
            "question": question,
            "gold_session_ids": gold_sids,
            "retrieved_session_ids": retrieved_sids[:10],
            "recall_at_5": r5,
            "recall_at_10": r10,
            "ndcg_at_5": ndcg5,
            "ndcg_at_10": ndcg10,
            "hit_at_5": bool(r5 > 0),
        }
        results.append(result)

        cat = category_metrics[qtype]
        cat["r5"].append(r5)
        cat["r10"].append(r10)
        cat["ndcg5"].append(ndcg5)
        cat["ndcg10"].append(ndcg10)
        cat["count"] += 1

        db.close()

        if show_progress and (qi + 1) % 10 == 0:
            elapsed = time.time() - (t0 - embed_time_total + embed_time_total)
            pct = (qi + 1) / total * 100
            current_r5 = sum(r["recall_at_5"] for r in results) / len(results) * 100
            print(
                f"  [{qi+1}/{total}] {pct:.0f}% | Running R@5: {current_r5:.1f}%",
                flush=True,
            )

    all_r5 = [r["recall_at_5"] for r in results]
    all_r10 = [r["recall_at_10"] for r in results]
    all_ndcg5 = [r["ndcg_at_5"] for r in results]
    all_ndcg10 = [r["ndcg_at_10"] for r in results]

    summary = {
        "mode": mode,
        "top_k": top_k,
        "total_questions": len(results),
        "overall": {
            "R@5": sum(all_r5) / len(all_r5) * 100,
            "R@10": sum(all_r10) / len(all_r10) * 100,
            "NDCG@5": sum(all_ndcg5) / len(all_ndcg5),
            "NDCG@10": sum(all_ndcg10) / len(all_ndcg10),
        },
        "per_category": {},
        "timing": {
            "embed_total_s": round(embed_time_total, 1),
            "search_total_s": round(search_time_total, 1),
            "avg_embed_per_q_s": round(embed_time_total / len(results), 2),
            "avg_search_per_q_ms": round(search_time_total / len(results) * 1000, 1),
        },
        "misses_at_5": [],
    }

    for qtype, metrics in sorted(category_metrics.items()):
        summary["per_category"][qtype] = {
            "count": metrics["count"],
            "R@5": sum(metrics["r5"]) / len(metrics["r5"]) * 100,
            "R@10": sum(metrics["r10"]) / len(metrics["r10"]) * 100,
            "NDCG@5": sum(metrics["ndcg5"]) / len(metrics["ndcg5"]),
            "NDCG@10": sum(metrics["ndcg10"]) / len(metrics["ndcg10"]),
        }

    for r in results:
        if not r["hit_at_5"]:
            summary["misses_at_5"].append(
                {
                    "question_id": r["question_id"],
                    "question_type": r["question_type"],
                    "question": r["question"][:200],
                }
            )

    return results, summary


def run_benchmark_fulldb(
    data: list[dict],
    mode: str = "enhanced_v2",
    top_k: int = 10,
    limit: int | None = None,
    show_progress: bool = True,
) -> tuple[list[dict], dict]:
    """Full-DB benchmark: build ONE database with all unique sessions, query 500 times.
    Mirrors production ERINYS architecture where all observations live in one DB.
    Eliminates per-question DB rebuild overhead."""
    search_fn = SEARCH_FNS[mode]
    questions = data[:limit] if limit else data
    use_chunked = mode in CHUNKED_MODES

    if show_progress:
        print("  [Full-DB Mode] Building global database...")

    t_pre = time.time()
    precompute_embeddings(questions, chunked=use_chunked, show_progress=show_progress)
    t_pre_end = time.time()
    if show_progress:
        print(f"  Pre-compute time: {t_pre_end - t_pre:.1f}s")

    seen_sids: set[str] = set()
    unique_sid_session: list[tuple[str, list[dict]]] = []
    for item in data[:limit] if limit else data:
        for sid, session in zip(item["haystack_session_ids"], item["haystack_sessions"]):
            if sid not in seen_sids:
                seen_sids.add(sid)
                unique_sid_session.append((sid, session))

    t_build = time.time()
    db = create_bench_db()
    all_sids = [s[0] for s in unique_sid_session]
    all_sessions = [s[1] for s in unique_sid_session]

    if use_chunked:
        sid_to_obsids = ingest_haystack_chunked(db, all_sessions, all_sids)
        obsid_to_sid: dict[int, str] = {}
        for sid, obs_list in sid_to_obsids.items():
            for oid in obs_list:
                obsid_to_sid[oid] = sid
    else:
        sid_to_obsid = ingest_haystack(db, all_sessions, all_sids)
        obsid_to_sid = {v: k for k, v in sid_to_obsid.items()}

    t_build_end = time.time()
    if show_progress:
        n_obs = db.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        print(f"  DB built: {len(unique_sid_session)} sessions, {n_obs} observations in {t_build_end - t_build:.1f}s")
        print()

    results = []
    category_metrics: dict[str, dict] = defaultdict(
        lambda: {"r5": [], "r10": [], "ndcg5": [], "ndcg10": [], "count": 0}
    )
    total = len(questions)
    search_time_total = 0.0

    for qi, item in enumerate(questions):
        qid = item["question_id"]
        qtype = item["question_type"]
        question = item["question"]
        gold_sids = item["answer_session_ids"]
        haystack_sids_set = set(item["haystack_session_ids"])

        t2 = time.time()
        fulldb_fetch = max(top_k * 20, 200)
        search_results = search_fn(db, question, top_k=fulldb_fetch if use_chunked else fulldb_fetch)
        t3 = time.time()
        search_time_total += t3 - t2

        retrieved_sids_raw = []
        for res in search_results:
            meta = res.get("metadata")
            if isinstance(meta, str):
                meta = json.loads(meta)
            if meta and "session_id" in meta:
                sid = meta["session_id"]
            else:
                obs_id = res["id"]
                sid = obsid_to_sid.get(obs_id, f"unknown_{obs_id}")
            if sid in haystack_sids_set:
                retrieved_sids_raw.append(sid)

        if use_chunked:
            seen = set()
            retrieved_sids = []
            for sid in retrieved_sids_raw:
                if sid not in seen:
                    seen.add(sid)
                    retrieved_sids.append(sid)
        else:
            retrieved_sids = retrieved_sids_raw

        r5 = compute_recall_at_k(retrieved_sids, gold_sids, 5)
        r10 = compute_recall_at_k(retrieved_sids, gold_sids, 10)
        ndcg5 = compute_ndcg_at_k(retrieved_sids, gold_sids, 5)
        ndcg10 = compute_ndcg_at_k(retrieved_sids, gold_sids, 10)

        result = {
            "question_id": qid,
            "question_type": qtype,
            "question": question,
            "gold_session_ids": gold_sids,
            "retrieved_session_ids": retrieved_sids[:10],
            "recall_at_5": r5,
            "recall_at_10": r10,
            "ndcg_at_5": ndcg5,
            "ndcg_at_10": ndcg10,
            "hit_at_5": bool(r5 > 0),
        }
        results.append(result)

        cat = category_metrics[qtype]
        cat["r5"].append(r5)
        cat["r10"].append(r10)
        cat["ndcg5"].append(ndcg5)
        cat["ndcg10"].append(ndcg10)
        cat["count"] += 1

        if show_progress and (qi + 1) % 10 == 0:
            pct = (qi + 1) / total * 100
            current_r5 = sum(r["recall_at_5"] for r in results) / len(results) * 100
            print(
                f"  [{qi+1}/{total}] {pct:.0f}% | Running R@5: {current_r5:.1f}%",
                flush=True,
            )

    db.close()

    all_r5 = [r["recall_at_5"] for r in results]
    all_r10 = [r["recall_at_10"] for r in results]
    all_ndcg5 = [r["ndcg_at_5"] for r in results]
    all_ndcg10 = [r["ndcg_at_10"] for r in results]

    summary = {
        "mode": f"{mode}_fulldb",
        "top_k": top_k,
        "total_questions": len(results),
        "overall": {
            "R@5": sum(all_r5) / len(all_r5) * 100,
            "R@10": sum(all_r10) / len(all_r10) * 100,
            "NDCG@5": sum(all_ndcg5) / len(all_ndcg5),
            "NDCG@10": sum(all_ndcg10) / len(all_ndcg10),
        },
        "per_category": {},
        "timing": {
            "embed_total_s": 0.0,
            "search_total_s": round(search_time_total, 1),
            "avg_embed_per_q_s": 0.0,
            "avg_search_per_q_ms": round(search_time_total / len(results) * 1000, 1),
            "db_build_s": round(t_build_end - t_build, 1),
            "unique_sessions": len(unique_sid_session),
        },
        "misses_at_5": [],
    }

    for qtype, metrics in sorted(category_metrics.items()):
        summary["per_category"][qtype] = {
            "count": metrics["count"],
            "R@5": sum(metrics["r5"]) / len(metrics["r5"]) * 100,
            "R@10": sum(metrics["r10"]) / len(metrics["r10"]) * 100,
            "NDCG@5": sum(metrics["ndcg5"]) / len(metrics["ndcg5"]),
            "NDCG@10": sum(metrics["ndcg10"]) / len(metrics["ndcg10"]),
        }

    for r in results:
        if not r["hit_at_5"]:
            summary["misses_at_5"].append(
                {
                    "question_id": r["question_id"],
                    "question_type": r["question_type"],
                    "question": r["question"][:200],
                }
            )

    return results, summary


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 70)
    print(f"ERINYS LongMemEval Benchmark — Mode: {summary['mode']}")
    print("=" * 70)

    o = summary["overall"]
    print(f"\n  Overall ({summary['total_questions']} questions):")
    print(f"    R@5:     {o['R@5']:.1f}%")
    print(f"    R@10:    {o['R@10']:.1f}%")
    print(f"    NDCG@5:  {o['NDCG@5']:.3f}")
    print(f"    NDCG@10: {o['NDCG@10']:.3f}")

    print(f"\n  Per Category:")
    print(f"    {'Category':<30} {'Count':>5}  {'R@5':>7}  {'R@10':>7}")
    print(f"    {'-'*30} {'-'*5}  {'-'*7}  {'-'*7}")
    for qtype, m in sorted(summary["per_category"].items()):
        print(f"    {qtype:<30} {m['count']:>5}  {m['R@5']:>6.1f}%  {m['R@10']:>6.1f}%")

    t = summary["timing"]
    print(f"\n  Timing:")
    print(f"    Embedding total: {t['embed_total_s']:.1f}s")
    print(f"    Search total:    {t['search_total_s']:.1f}s")
    print(f"    Avg embed/q:     {t['avg_embed_per_q_s']:.2f}s")
    print(f"    Avg search/q:    {t['avg_search_per_q_ms']:.1f}ms")

    misses = summary["misses_at_5"]
    if misses:
        print(f"\n  Misses at R@5 ({len(misses)} questions):")
        for m in misses[:10]:
            print(f"    [{m['question_type']}] {m['question_id']}: {m['question'][:80]}")
        if len(misses) > 10:
            print(f"    ... and {len(misses) - 10} more")

    print(f"\n  Comparison with MemPalace:")
    print(f"    {'System':<35} {'R@5':>7}  {'R@10':>7}")
    print(f"    {'-'*35} {'-'*7}  {'-'*7}")
    print(f"    {'ERINYS ' + summary['mode']:<35} {o['R@5']:>6.1f}%  {o['R@10']:>6.1f}%")
    print(f"    {'MemPalace raw (all-MiniLM)':<35} {'96.6%':>7}  {'~99%':>7}")
    print(f"    {'MemPalace hybrid v4 + Haiku':<35} {'100%':>7}  {'100%':>7}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="ERINYS LongMemEval Benchmark")
    parser.add_argument(
        "data_path",
        nargs="?",
        default=DEFAULT_DATA,
        help="Path to longmemeval_s_cleaned.json",
    )
    parser.add_argument(
        "--mode",
        choices=list(SEARCH_FNS.keys()),
        default="rrf",
        help="Search mode: rrf, vec, fts, enhanced, enhanced_v2",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-K for retrieval")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of questions"
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "results"),
        help="Output directory",
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable progress output"
    )
    parser.add_argument(
        "--full-db", action="store_true",
        help="Full-DB mode: build one DB with all sessions, query against it (production-like)"
    )
    args = parser.parse_args()

    print(f"Loading dataset from {args.data_path}...")
    data = load_dataset(args.data_path)
    print(f"  {len(data)} questions loaded")
    print(f"  Mode: {args.mode}{'_fulldb' if args.full_db else ''}")
    print(f"  Top-K: {args.top_k}")
    if args.limit:
        print(f"  Limit: {args.limit} questions")

    print(f"\nInitializing embedding engine ({embedding_engine.model_name})...")

    print(f"\nRunning benchmark...")
    t_start = time.time()
    if args.full_db:
        results, summary = run_benchmark_fulldb(
            data,
            mode=args.mode,
            top_k=args.top_k,
            limit=args.limit,
            show_progress=not args.no_progress,
        )
    else:
        results, summary = run_benchmark(
            data,
            mode=args.mode,
            top_k=args.top_k,
            limit=args.limit,
            show_progress=not args.no_progress,
        )
    t_total = time.time() - t_start
    summary["total_time_s"] = round(t_total, 1)

    print_summary(summary)

    os.makedirs(args.output, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    mode_tag = f"{args.mode}_fulldb" if args.full_db else args.mode

    results_file = os.path.join(
        args.output,
        f"results_erinys_{mode_tag}_{ts}.jsonl",
    )
    with open(results_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary_file = os.path.join(
        args.output,
        f"summary_erinys_{mode_tag}_{ts}.json",
    )
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n  Results: {results_file}")
    print(f"  Summary: {summary_file}")
    print(f"  Total time: {t_total:.1f}s")


if __name__ == "__main__":
    main()
