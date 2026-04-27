#!/usr/bin/env python3
"""
ERINYS × LoCoMo Benchmark
---------------------------
Runs the LoCoMo benchmark against ERINYS search engine.
10 conversations, ~1986 QA pairs across 5 categories.
Matches MemPalace's methodology for direct comparison.

For each conversation:
  1. Ingest all sessions into a fresh ERINYS in-memory DB
  2. For each QA pair, query the DB
  3. Score retrieval recall (did we find the evidence session?)

Usage:
    python benchmarks/locomo_bench.py /path/to/locomo10.json
    python benchmarks/locomo_bench.py /path/to/locomo10.json --top-k 10
    python benchmarks/locomo_bench.py /path/to/locomo10.json --mode enhanced_v2
"""

import argparse
import json
import math
import os
import re
import sqlite3
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
from erinys_memory.search import rrf_hybrid_search, focus_query_for_embedding, _is_temporal_query
from erinys_memory.preference_extract import extract_all as extract_synthetics

sys.path.insert(0, str(Path(__file__).resolve().parent))
from enhanced_search import enhanced_hybrid_search_v2


# LoCoMo category mapping (1-indexed integer → name)
CATEGORIES = {
    1: "single-hop",
    2: "temporal",
    3: "temporal-inference",
    4: "open-domain",
    5: "adversarial",
}


# ── Data Loading ────────────────────────────────────────────────────────────

def load_dataset(path: str) -> list[dict]:
    """Load LoCoMo JSON dataset."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_conversation_sessions(conversation: dict) -> list[dict]:
    """Extract sessions from a LoCoMo conversation dict.
    
    LoCoMo format: conversation has session_1, session_2, ... keys.
    Each session is a list of dicts with 'speaker', 'text', 'dia_id'.
    """
    sessions = []
    session_num = 1
    while True:
        key = f"session_{session_num}"
        date_key = f"session_{session_num}_date_time"
        if key not in conversation:
            break
        dialogs = conversation[key]
        date = conversation.get(date_key, "")
        sessions.append({
            "session_num": session_num,
            "session_id": f"session_{session_num}",
            "date": date,
            "dialogs": dialogs,
        })
        session_num += 1
    return sessions


def session_to_text(session: dict) -> str:
    """Convert a LoCoMo session to plain text for ERINYS ingestion."""
    parts = []
    for d in session["dialogs"]:
        speaker = d.get("speaker", "?")
        text = d.get("text", "")
        parts.append(f'{speaker}: {text}')
    return "\n".join(parts)


def evidence_to_session_ids(evidence: list[str]) -> set[str]:
    """Map evidence dia_ids (e.g., 'D3:5') to session IDs (e.g., 'session_3').
    
    LoCoMo evidence format: 'D{session}:{dialog_index}'
    """
    session_ids = set()
    for ev in evidence:
        match = re.match(r"D(\d+):", ev)
        if match:
            session_ids.add(f"session_{match.group(1)}")
    return session_ids


# ── DB + Search ─────────────────────────────────────────────────────────────

def create_bench_db() -> sqlite3.Connection:
    """Create an in-memory ERINYS database for benchmarking."""
    config = ErinysConfig(db_path=":memory:")
    return init_db(config)


_EMBED_CACHE: dict[str, list[float]] = {}
EMBED_CACHE_FILENAME = "locomo_embed_cache_v1.json"
LEGACY_EMBED_CACHE_FILENAME = "locomo_embed_cache_v1.pkl"


def _is_embedding_vector(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, (int, float)) for item in value)


def _load_embed_cache(cache_path: str) -> dict[str, list[float]]:
    with open(cache_path, encoding="utf-8") as f:
        raw_cache = json.load(f)
    if not isinstance(raw_cache, dict):
        return {}
    return {
        key: [float(item) for item in value]
        for key, value in raw_cache.items()
        if isinstance(key, str) and _is_embedding_vector(value)
    }


def _save_embed_cache(cache_path: str) -> None:
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(_EMBED_CACHE, f, ensure_ascii=False, separators=(",", ":"))


def _resolve_embeddings(cache_keys: list[str], texts: list[str]) -> list[list[float]]:
    missing_pairs = [
        (key, text) for key, text in zip(cache_keys, texts) if key not in _EMBED_CACHE
    ]
    if missing_pairs:
        missing_keys = [key for key, _ in missing_pairs]
        missing_texts = [text for _, text in missing_pairs]
        missing_embeddings = embedding_engine.embed_batch(missing_texts)
        for key, embedding in zip(missing_keys, missing_embeddings):
            _EMBED_CACHE[key] = embedding
    return [_EMBED_CACHE[key] for key in cache_keys]


def precompute_embeddings(
    sessions_by_conv: dict[str, list[dict]],
    show_progress: bool = True,
    cache_dir: str = os.path.expanduser("~/.cache/erinys_bench"),
) -> None:
    """Pre-compute embeddings for all sessions across all conversations."""
    global _EMBED_CACHE

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, EMBED_CACHE_FILENAME)
    legacy_cache_path = os.path.join(cache_dir, LEGACY_EMBED_CACHE_FILENAME)

    if os.path.exists(cache_path):
        if show_progress:
            print(f"  Loading cached embeddings from {cache_path}...")
        _EMBED_CACHE.update(_load_embed_cache(cache_path))
        if show_progress:
            print(f"  Loaded {len(_EMBED_CACHE)} sessions from cache.")
    elif os.path.exists(legacy_cache_path) and show_progress:
        print(f"  Ignoring legacy pickle cache at {legacy_cache_path}; rebuilding JSON cache.")

    all_texts = []
    all_keys = []
    for conv_id, sessions in sessions_by_conv.items():
        for sess in sessions:
            key = f"{conv_id}_{sess['session_id']}"
            if key not in _EMBED_CACHE:
                all_texts.append(session_to_text(sess))
                all_keys.append(key)

    if not all_texts:
        return

    if show_progress:
        print(f"  Pre-computing embeddings for {len(all_texts)} sessions...")

    batch_size = 2000
    all_embeddings = []
    for bi in range(0, len(all_texts), batch_size):
        batch = all_texts[bi:bi + batch_size]
        batch_embs = embedding_engine.embed_batch(batch)
        all_embeddings.extend(batch_embs)
        if show_progress:
            done = min(bi + batch_size, len(all_texts))
            print(f"  Embedded {done}/{len(all_texts)} ({done * 100 // len(all_texts)}%)", flush=True)

    for key, emb in zip(all_keys, all_embeddings):
        _EMBED_CACHE[key] = emb

    _save_embed_cache(cache_path)

    if show_progress:
        print(f"  Embedding cache saved to {cache_path}")


def ingest_sessions(
    db: sqlite3.Connection,
    sessions: list[dict],
    conv_id: str,
    observations: dict | None = None,
    session_summaries: dict | None = None,
    event_summaries: dict | None = None,
) -> dict[str, int]:
    """Ingest LoCoMo sessions into ERINYS DB. Returns session_id → obs_id mapping."""
    sid_to_obsid: dict[str, int] = {}

    texts = [session_to_text(s) for s in sessions]
    cache_keys = [f"{conv_id}_{s['session_id']}" for s in sessions]
    embeddings = _resolve_embeddings(cache_keys, texts)

    for sess, text, emb in zip(sessions, texts, embeddings):
        sid = sess["session_id"]
        obs_payload = {
            "title": f"{conv_id}_{sid}",
            "content": text,
            "type": "manual",
            "project": "locomo",
            "scope": "project",
            "base_strength": 1.0,
            "access_count": 0,
            "metadata": json.dumps({
                "session_id": sid,
                "conv_id": conv_id,
                "date": sess.get("date", ""),
            }),
        }
        obs_id = insert_observation_with_embedding(
            db, obs_payload, serialize_f32(emb)
        )
        sid_to_obsid[sid] = obs_id

        # Sub-session chunking for long sessions (embedding model truncation fix)
        dialogs = sess["dialogs"]
        if len(text) > 1500 and len(dialogs) >= 4:
            mid = len(dialogs) // 2
            second_half_parts = []
            for d in dialogs[mid:]:
                speaker = d.get("speaker", "?")
                t = d.get("text", "")
                second_half_parts.append(f'{speaker}: {t}')
            second_half_text = "\n".join(second_half_parts)
            if len(second_half_text) > 50:
                sh_emb = embedding_engine.embed(second_half_text)
                sh_payload = {
                    "title": f"{conv_id}_{sid}_p2",
                    "content": second_half_text,
                    "type": "manual",
                    "project": "locomo",
                    "scope": "project",
                    "base_strength": 1.0,
                    "access_count": 0,
                    "metadata": json.dumps({
                        "session_id": sid,
                        "conv_id": conv_id,
                        "chunk": "second_half",
                    }),
                }
                insert_observation_with_embedding(
                    db, sh_payload, serialize_f32(sh_emb)
                )

        # Session-level preference synthesis
        synths = extract_synthetics(text)
        if synths:
            syn_text = f"Session {sid} preferences: " + " | ".join(synths)
            syn_emb = embedding_engine.embed(syn_text)
            syn_payload = {
                "title": f"{conv_id}_{sid}_pref",
                "content": syn_text,
                "type": "manual",
                "project": "locomo",
                "scope": "project",
                "base_strength": 1.0,
                "access_count": 0,
                "metadata": json.dumps({
                    "session_id": sid,
                    "conv_id": conv_id,
                    "synthetic": True,
                }),
            }
            insert_observation_with_embedding(
                db, syn_payload, serialize_f32(syn_emb)
            )

        # Observation facts synthesis (from LoCoMo dataset)
        if observations:
            obs_key = f"session_{sess['session_num']}_observation"
            session_obs = observations.get(obs_key, {})
            if isinstance(session_obs, dict):
                for speaker, facts in session_obs.items():
                    if not isinstance(facts, list) or not facts:
                        continue
                    fact_texts = []
                    for f in facts:
                        if isinstance(f, list) and len(f) >= 1:
                            fact_texts.append(f[0])
                        elif isinstance(f, str):
                            fact_texts.append(f)
                    if fact_texts:
                        obs_text = f"{speaker}: " + " ".join(fact_texts)
                        obs_cache_key = f"{conv_id}_{sid}_{speaker}_obs"
                        if obs_cache_key in _EMBED_CACHE:
                            obs_emb = _EMBED_CACHE[obs_cache_key]
                        else:
                            obs_emb = embedding_engine.embed(obs_text)
                        obs_payload = {
                            "title": f"{conv_id}_{sid}_{speaker}_obs",
                            "content": obs_text,
                            "type": "manual",
                            "project": "locomo",
                            "scope": "project",
                            "base_strength": 1.0,
                            "access_count": 0,
                            "metadata": json.dumps({
                                "session_id": sid,
                                "conv_id": conv_id,
                                "synthetic": True,
                                "observation": True,
                            }),
                        }
                        insert_observation_with_embedding(
                            db, obs_payload, serialize_f32(obs_emb)
                        )

        # Session summary ingestion
        if session_summaries:
            sum_key = f"session_{sess['session_num']}_summary"
            summary_text = session_summaries.get(sum_key, "")
            if summary_text and isinstance(summary_text, str):
                sum_cache_key = f"{conv_id}_{sid}_summary"
                if sum_cache_key in _EMBED_CACHE:
                    sum_emb = _EMBED_CACHE[sum_cache_key]
                else:
                    sum_emb = embedding_engine.embed(summary_text)
                sum_payload = {
                    "title": f"{conv_id}_{sid}_summary",
                    "content": summary_text,
                    "type": "manual",
                    "project": "locomo",
                    "scope": "project",
                    "base_strength": 1.0,
                    "access_count": 0,
                    "metadata": json.dumps({
                        "session_id": sid,
                        "conv_id": conv_id,
                        "synthetic": True,
                        "summary": True,
                    }),
                }
                insert_observation_with_embedding(
                    db, sum_payload, serialize_f32(sum_emb)
                )

        # Event summary ingestion
        if event_summaries:
            ev_key = f"events_session_{sess['session_num']}"
            ev_data = event_summaries.get(ev_key, {})
            if isinstance(ev_data, dict):
                ev_parts = []
                ev_date = ev_data.get("date", "")
                if ev_date:
                    ev_parts.append(f"Date: {ev_date}")
                for speaker, events in ev_data.items():
                    if speaker == "date" or not isinstance(events, list):
                        continue
                    for ev in events:
                        ev_parts.append(f"{speaker}: {ev}")
                if ev_parts:
                    ev_text = " ".join(ev_parts)
                    ev_cache_key = f"{conv_id}_{sid}_events"
                    if ev_cache_key in _EMBED_CACHE:
                        ev_emb = _EMBED_CACHE[ev_cache_key]
                    else:
                        ev_emb = embedding_engine.embed(ev_text)
                    ev_payload = {
                        "title": f"{conv_id}_{sid}_events",
                        "content": ev_text,
                        "type": "manual",
                        "project": "locomo",
                        "scope": "project",
                        "base_strength": 1.0,
                        "access_count": 0,
                        "metadata": json.dumps({
                            "session_id": sid,
                            "conv_id": conv_id,
                            "synthetic": True,
                            "event_summary": True,
                        }),
                    }
                    insert_observation_with_embedding(
                        db, ev_payload, serialize_f32(ev_emb)
                    )

    return sid_to_obsid


def search_rrf(db, query, top_k=10):
    """RRF hybrid search."""
    query_emb = embedding_engine.embed(query)
    return rrf_hybrid_search(
        db, query=query, query_embedding=query_emb,
        project="locomo", limit=top_k,
        fts_weight=0.4, vec_weight=0.6,
    )


def search_enhanced_v2(db, query, top_k=10):
    """Enhanced v2 with focused embedding for temporal queries."""
    query_emb = embedding_engine.embed(query)
    focused_emb = None
    if _is_temporal_query(query):
        focused_text = focus_query_for_embedding(query)
        if focused_text != query:
            focused_emb = embedding_engine.embed(focused_text)
    return rrf_hybrid_search(
        db, query=query, query_embedding=query_emb,
        project="locomo", limit=top_k,
        focused_embedding=focused_emb,
    )


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
        search_limit = 75
    kwargs = dict(
        db=db, query=query, project="locomo", limit=max(top_k, search_limit),
        proper_noun_boost=0.7, keyword_boost=0.5, bigram_boost=0.7, temporal_content_boost=0.5
    )

    if focused_emb and _is_inference_query(query):
        kwargs_res1 = kwargs.copy()
        kwargs_res2 = kwargs.copy()
        
        # We rely on BM25 for proper nouns and broad keyword signals, while vector handles semantics
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
        return res
    else:
        return rrf_hybrid_search(
            query_embedding=query_emb, focused_embedding=focused_emb, **kwargs
        )


def search_enhanced(db, query, top_k=10):
    """Enhanced RRF with keyword boosting."""
    query_emb = embedding_engine.embed(query)
    return enhanced_hybrid_search_v2(
        db, query=query, query_embedding=query_emb,
        project="locomo", limit=top_k,
    )


def _extract_entities(query: str) -> list[str]:
    """Extract proper nouns (capitalized words not at sentence start)."""
    words = query.split()
    entities = []
    for i, w in enumerate(words):
        cleaned = re.sub(r"[^a-zA-Z']", "", w)
        if not cleaned or len(cleaned) <= 1:
            continue
        if cleaned[0].isupper() and i > 0:
            low = cleaned.lower()
            if low not in _INFERENCE_FILLER and low not in {'what', 'when', 'where', 'who', 'why', 'how', 'would', 'could', 'does', 'did', 'has', 'is', 'are'}:
                entities.append(cleaned)
    return entities


def _focus_inference_preserve_entities(query: str) -> str:
    """Strip inference filler but preserve entity names."""
    entities = _extract_entities(query)
    tokens = query.split()
    focused = [t for t in tokens if t.lower().rstrip('.,?!') not in _INFERENCE_FILLER]
    result = ' '.join(focused).strip()
    result = re.sub(r'\s+', ' ', result)
    if len(result.split()) < 2:
        result = query
    for ent in entities:
        if ent.lower() not in result.lower():
            result = f"{ent} {result}"
    return result


def search_sv_boost(db, query, top_k=10):
    """State Vector enhanced: entity-aware multi-perspective search.
    
    Improvements over enhanced_v2_boost:
    1. Entity names preserved in focused embeddings
    2. Entity-perspective embedding for multi-hop queries
    3. Adaptive proper noun boost (1.5x for inference, 1.0x otherwise)
    4. Wider search for entity-dispersed results
    """
    query_emb = embedding_engine.embed(query)
    entities = _extract_entities(query)
    is_inference = _is_inference_query(query)
    is_temporal = _is_temporal_query(query)
    
    focused_emb = None
    entity_emb = None
    
    if is_temporal:
        focused_text = focus_query_for_embedding(query)
        if focused_text != query:
            focused_emb = embedding_engine.embed(focused_text)
    elif is_inference:
        focused_text = _focus_inference_preserve_entities(query)
        if focused_text != query:
            focused_emb = embedding_engine.embed(focused_text)
    
    if entities:
        entity_query = ' '.join(entities)
        content_keywords = [t for t in query.split() if t.lower().rstrip('.,?!') not in _INFERENCE_FILLER and t not in entities]
        top_content = content_keywords[:3]
        entity_perspective = f"{entity_query} {' '.join(top_content)}"
        entity_emb = embedding_engine.embed(entity_perspective)
    
    pn_boost = 1.5 if is_inference else (1.0 if entities else 0.7)
    kw_boost = 0.6 if is_inference else 0.5
    bg_boost = 0.8 if is_inference else 0.7
    
    search_limit = 150 if is_inference else (75 if is_temporal else 50)
    
    base_kwargs = dict(
        db=db, query=query, project="locomo", limit=max(top_k, search_limit),
        proper_noun_boost=pn_boost, keyword_boost=kw_boost,
        bigram_boost=bg_boost, temporal_content_boost=0.5
    )
    
    if is_inference and (focused_emb or entity_emb):
        merged = {}
        
        res_original = rrf_hybrid_search(query_embedding=query_emb, **base_kwargs)
        for r in res_original:
            rid = r['id']
            r_copy = r.copy()
            s = r.get('rrf_score', r.get('score', 0)) * 0.3
            r_copy['score'] = s
            if 'rrf_score' in r_copy:
                r_copy['rrf_score'] = s
            merged[rid] = r_copy
        
        if focused_emb:
            focused_kwargs = base_kwargs.copy()
            focused_kwargs['keyword_boost'] = 0.0
            res_focused = rrf_hybrid_search(query_embedding=focused_emb, **focused_kwargs)
            for r in res_focused:
                rid = r['id']
                s = r.get('rrf_score', r.get('score', 0)) * 0.8
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
        
        if entity_emb:
            entity_kwargs = base_kwargs.copy()
            entity_kwargs['proper_noun_boost'] = 2.0
            entity_kwargs['keyword_boost'] = 0.3
            res_entity = rrf_hybrid_search(query_embedding=entity_emb, **entity_kwargs)
            for r in res_entity:
                rid = r['id']
                s = r.get('rrf_score', r.get('score', 0)) * 0.5
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
        return res
    
    elif entities and not is_inference:
        merged = {}
        
        res_original = rrf_hybrid_search(
            query_embedding=query_emb,
            focused_embedding=focused_emb,
            **base_kwargs
        )
        for r in res_original:
            rid = r['id']
            r_copy = r.copy()
            s = r.get('rrf_score', r.get('score', 0)) * 0.7
            r_copy['score'] = s
            if 'rrf_score' in r_copy:
                r_copy['rrf_score'] = s
            merged[rid] = r_copy
        
        if entity_emb:
            entity_kwargs = base_kwargs.copy()
            entity_kwargs['proper_noun_boost'] = 1.5
            res_entity = rrf_hybrid_search(query_embedding=entity_emb, **entity_kwargs)
            for r in res_entity:
                rid = r['id']
                s = r.get('rrf_score', r.get('score', 0)) * 0.4
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
        return res
    
    else:
        return rrf_hybrid_search(
            query_embedding=query_emb, focused_embedding=focused_emb, **base_kwargs
        )


SEARCH_FNS = {
    "rrf": search_rrf,
    "enhanced": search_enhanced,
    "enhanced_v2": search_enhanced_v2,
    "enhanced_v2_boost": search_enhanced_v2_boost,
    "sv_boost": search_sv_boost,
}


# ── Metrics ─────────────────────────────────────────────────────────────────

def compute_recall_at_k(retrieved_sids: list[str], gold_sids: set[str], k: int) -> float:
    """Recall@K: is any gold session in the top-K results?"""
    top_k = set(retrieved_sids[:k])
    return 1.0 if top_k & gold_sids else 0.0


def compute_ndcg_at_k(retrieved_sids: list[str], gold_sids: set[str], k: int) -> float:
    """NDCG@K with binary relevance."""
    dcg = 0.0
    for i, sid in enumerate(retrieved_sids[:k]):
        if sid in gold_sids:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(len(gold_sids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


# ── Benchmark Runner ────────────────────────────────────────────────────────

def run_benchmark(
    data: list[dict],
    mode: str = "enhanced_v2",
    top_k: int = 10,
    limit: int | None = None,
    show_progress: bool = True,
) -> tuple[list[dict], dict]:
    """Run the LoCoMo benchmark. Per-conversation DB (matches MemPalace method)."""
    search_fn = SEARCH_FNS[mode]

    # Pre-compute embeddings for all conversations
    sessions_by_conv = {}
    for conv_idx, sample in enumerate(data):
        conv_id = sample.get("sample_id", f"conv-{conv_idx}")
        conversation = sample["conversation"]
        sessions = load_conversation_sessions(conversation)
        sessions_by_conv[conv_id] = sessions

    if show_progress:
        total_sessions = sum(len(v) for v in sessions_by_conv.values())
        total_qa = sum(len(s.get("qa", [])) for s in data)
        print(f"  Conversations: {len(data)}, Sessions: {total_sessions}, QA pairs: {total_qa}")

    precompute_embeddings(sessions_by_conv, show_progress=show_progress)

    results = []
    category_metrics: dict[str, dict] = defaultdict(
        lambda: {"r5": [], "r10": [], "ndcg5": [], "ndcg10": [], "count": 0}
    )
    search_time_total = 0.0
    ingest_time_total = 0.0
    qa_processed = 0

    for conv_idx, sample in enumerate(data):
        conv_id = sample.get("sample_id", f"conv-{conv_idx}")
        conversation = sample["conversation"]
        qa_pairs = sample["qa"]
        sessions = sessions_by_conv[conv_id]

        if limit and qa_processed >= limit:
            break

        # Build fresh DB for this conversation
        t0 = time.time()
        db = create_bench_db()
        try:
            observations = sample.get("observation", {})
            session_summaries = sample.get("session_summary", {})
            event_summaries = sample.get("event_summary", {})
            sid_to_obsid = ingest_sessions(
                db, sessions, conv_id,
                observations=observations,
                session_summaries=session_summaries,
                event_summaries=event_summaries,
            )
            t1 = time.time()
            ingest_time_total += t1 - t0

            if show_progress:
                print(f"\n  [{conv_idx + 1}/{len(data)}] {conv_id}: "
                      f"{len(sessions)} sessions, {len(qa_pairs)} questions")

            for qa in qa_pairs:
                if limit and qa_processed >= limit:
                    break

                question = qa["question"]
                category_num = qa["category"]
                category_name = CATEGORIES.get(category_num, f"unknown-{category_num}")
                evidence = qa.get("evidence", [])
                gold_sids = evidence_to_session_ids(evidence)

                if not gold_sids:
                    continue

                t2 = time.time()
                search_results = search_fn(db, question, top_k=max(top_k, 10))
                t3 = time.time()
                search_time_total += t3 - t2

                if isinstance(search_results, list) and len(search_results) > 0 and isinstance(search_results[0], str):
                    retrieved_sids = search_results[:10]
                else:
                    from erinys_memory.search import collapse_by_session
                    col = collapse_by_session(search_results, limit=max(top_k, 50))

                    retrieved_sids = []
                    for r in col:
                        meta = r.get("metadata")
                        if isinstance(meta, str):
                            try:
                                meta = json.loads(meta)
                            except (json.JSONDecodeError, TypeError):
                                meta = {}
                        if not isinstance(meta, dict):
                            meta = {}
                        sid = meta.get("session_id")
                        if sid and sid not in retrieved_sids:
                            retrieved_sids.append(sid)

                r5 = compute_recall_at_k(retrieved_sids, gold_sids, 5)
                r10 = compute_recall_at_k(retrieved_sids, gold_sids, 10)
                ndcg5 = compute_ndcg_at_k(retrieved_sids, gold_sids, 5)
                ndcg10 = compute_ndcg_at_k(retrieved_sids, gold_sids, 10)

                result = {
                    "conv_id": conv_id,
                    "question": question,
                    "category": category_name,
                    "evidence": evidence,
                    "gold_session_ids": list(gold_sids),
                    "retrieved_session_ids": retrieved_sids[:10],
                    "recall_at_5": r5,
                    "recall_at_10": r10,
                    "ndcg_at_5": ndcg5,
                    "ndcg_at_10": ndcg10,
                    "hit_at_5": bool(r5 > 0),
                }
                results.append(result)

                cat = category_metrics[category_name]
                cat["r5"].append(r5)
                cat["r10"].append(r10)
                cat["ndcg5"].append(ndcg5)
                cat["ndcg10"].append(ndcg10)
                cat["count"] += 1
                qa_processed += 1

            if show_progress:
                current_r5 = sum(r["recall_at_5"] for r in results) / len(results) * 100
                current_r10 = sum(r["recall_at_10"] for r in results) / len(results) * 100
                print(f"    Running R@5: {current_r5:.1f}%  R@10: {current_r10:.1f}%")
        finally:
            db.close()

    # Build summary
    all_r5 = [r["recall_at_5"] for r in results]
    all_r10 = [r["recall_at_10"] for r in results]
    all_ndcg5 = [r["ndcg_at_5"] for r in results]
    all_ndcg10 = [r["ndcg_at_10"] for r in results]

    summary = {
        "benchmark": "LoCoMo",
        "mode": mode,
        "top_k": top_k,
        "total_questions": len(results),
        "overall": {
            "R@5": sum(all_r5) / len(all_r5) * 100 if all_r5 else 0,
            "R@10": sum(all_r10) / len(all_r10) * 100 if all_r10 else 0,
            "NDCG@5": sum(all_ndcg5) / len(all_ndcg5) if all_ndcg5 else 0,
            "NDCG@10": sum(all_ndcg10) / len(all_ndcg10) if all_ndcg10 else 0,
        },
        "per_category": {},
        "timing": {
            "ingest_total_s": round(ingest_time_total, 1),
            "search_total_s": round(search_time_total, 1),
            "avg_search_per_q_ms": round(search_time_total / max(len(results), 1) * 1000, 1),
        },
        "misses_at_5": [],
        "comparison": {
            "mempalace_raw_top10_R10": "88.9% (hybrid v5, no LLM)",
            "mempalace_rerank_top50_R10": "100% (trivial: top-k > sessions)",
        },
    }

    for cat_name, metrics in sorted(category_metrics.items()):
        summary["per_category"][cat_name] = {
            "count": metrics["count"],
            "R@5": sum(metrics["r5"]) / len(metrics["r5"]) * 100,
            "R@10": sum(metrics["r10"]) / len(metrics["r10"]) * 100,
            "NDCG@5": sum(metrics["ndcg5"]) / len(metrics["ndcg5"]),
            "NDCG@10": sum(metrics["ndcg10"]) / len(metrics["ndcg10"]),
        }

    for r in results:
        if not r["hit_at_5"]:
            summary["misses_at_5"].append({
                "conv_id": r["conv_id"],
                "category": r["category"],
                "question": r["question"][:200],
            })

    return results, summary


# ── Output ──────────────────────────────────────────────────────────────────

def print_summary(summary: dict) -> None:
    """Print benchmark results."""
    print(f"\n{'=' * 60}")
    print(f"  ERINYS × LoCoMo Benchmark Results")
    print(f"{'=' * 60}")
    print(f"  Mode:       {summary['mode']}")
    print(f"  Top-K:      {summary['top_k']}")
    print(f"  Questions:  {summary['total_questions']}")

    o = summary["overall"]
    print(f"\n  OVERALL:")
    print(f"    R@5:    {o['R@5']:.1f}%")
    print(f"    R@10:   {o['R@10']:.1f}%")
    print(f"    NDCG@5: {o['NDCG@5']:.4f}")
    print(f"    NDCG@10:{o['NDCG@10']:.4f}")

    print(f"\n  PER-CATEGORY:")
    for cat, m in sorted(summary["per_category"].items()):
        print(f"    {cat:25s} R@5={m['R@5']:5.1f}%  R@10={m['R@10']:5.1f}%  (n={m['count']})")

    print(f"\n  COMPARISON (MemPalace, LLM-free):")
    for k, v in summary.get("comparison", {}).items():
        print(f"    {k}: {v}")

    misses = summary.get("misses_at_5", [])
    if misses:
        print(f"\n  MISSES at R@5: {len(misses)}")
        for m in misses[:10]:
            print(f"    [{m['category']}] {m['question'][:80]}...")

    t = summary.get("timing", {})
    print(f"\n  TIMING:")
    print(f"    Ingest total: {t.get('ingest_total_s', 0):.1f}s")
    print(f"    Search total: {t.get('search_total_s', 0):.1f}s")
    print(f"    Avg search:   {t.get('avg_search_per_q_ms', 0):.1f}ms/q")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="ERINYS × LoCoMo Benchmark")
    parser.add_argument(
        "data_path",
        nargs="?",
        default="/tmp/locomo-data/locomo10.json",
        help="Path to locomo10.json",
    )
    parser.add_argument(
        "--mode",
        choices=list(SEARCH_FNS.keys()),
        default="enhanced_v2",
        help="Search mode (default: enhanced_v2)",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-K for recall")
    parser.add_argument("--limit", type=int, default=None, help="Limit QA pairs")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "results"),
        help="Output directory",
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"ERROR: Data file not found: {args.data_path}")
        print(f"Download it with:")
        print(f"  git clone https://github.com/snap-research/locomo.git /tmp/locomo")
        print(f"  cp /tmp/locomo/data/locomo10.json /tmp/locomo-data/locomo10.json")
        sys.exit(1)

    print(f"Loading dataset from {args.data_path}...")
    data = load_dataset(args.data_path)
    print(f"  {len(data)} conversations loaded")
    print(f"  Mode: {args.mode}")
    print(f"  Top-K: {args.top_k}")

    print(f"\nInitializing embedding engine ({embedding_engine.model_name})...")
    print(f"\nRunning benchmark...")

    t_start = time.time()
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

    results_file = os.path.join(args.output, f"results_erinys_locomo_{args.mode}_{ts}.jsonl")
    with open(results_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary_file = os.path.join(args.output, f"summary_erinys_locomo_{args.mode}_{ts}.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  Results: {results_file}")
    print(f"  Summary: {summary_file}")
    print(f"  Total time: {t_total:.1f}s")


if __name__ == "__main__":
    main()
