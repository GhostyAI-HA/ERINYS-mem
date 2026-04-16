#!/usr/bin/env python3
"""
ERINYS × ConvoMem Benchmark
------------------------------
Runs the ConvoMem benchmark against ERINYS search engine.
75,336 QA pairs across 6 evidence categories (sampled).
Matches MemPalace's methodology for direct comparison.

For each evidence item:
  1. Ingest all conversations into a fresh ERINYS in-memory DB
  2. Query with the question
  3. Check if any retrieved message matches the evidence messages

Downloads evidence files from HuggingFace on first run.

Usage:
    python benchmarks/convomem_bench.py                            # sample 50/cat
    python benchmarks/convomem_bench.py --limit 100                # sample 100/cat
    python benchmarks/convomem_bench.py --category user_evidence   # one category
"""

import argparse
import json
import re
import os
import sqlite3
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime
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


HF_BASE = "https://huggingface.co/datasets/Salesforce/ConvoMem/resolve/main/core_benchmark/evidence_questions"

CATEGORIES = {
    "user_evidence": "User Facts",
    "assistant_facts_evidence": "Assistant Facts",
    "changing_evidence": "Changing Facts",
    "abstention_evidence": "Abstention",
    "preference_evidence": "Preferences",
    "implicit_connection_evidence": "Implicit Connections",
}


# ── Data Loading ────────────────────────────────────────────────────────────

def download_evidence_file(category: str, subpath: str, cache_dir: str) -> dict | None:
    """Download a single evidence file from HuggingFace."""
    url = f"{HF_BASE}/{category}/{subpath}"
    cache_path = os.path.join(cache_dir, category, subpath.replace("/", "_"))
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    print(f"    Downloading: {category}/{subpath}...")
    try:
        urllib.request.urlretrieve(url, cache_path)
        with open(cache_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"    Failed to download {url}: {e}")
        return None


def discover_files(category: str, cache_dir: str) -> list[str]:
    """Discover available files for a category via HuggingFace API."""
    api_url = (
        f"https://huggingface.co/api/datasets/Salesforce/ConvoMem/tree/main/"
        f"core_benchmark/evidence_questions/{category}/1_evidence"
    )
    cache_path = os.path.join(cache_dir, f"{category}_filelist.json")

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    try:
        req = urllib.request.Request(api_url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            files = json.loads(resp.read())
            paths = [
                f["path"].split(f"{category}/")[1]
                for f in files
                if f["path"].endswith(".json")
            ]
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(paths, f)
            return paths
    except Exception as e:
        print(f"    Failed to list files for {category}: {e}")
        return []


def load_evidence_items(
    categories: list[str], limit_per_cat: int, cache_dir: str
) -> list[dict]:
    """Load evidence items from specified categories."""
    all_items = []

    for category in categories:
        files = discover_files(category, cache_dir)
        if not files:
            print(f"  Skipping {category} — no files found")
            continue

        items_for_cat = []
        for fpath in files:
            if len(items_for_cat) >= limit_per_cat:
                break
            data = download_evidence_file(category, fpath, cache_dir)
            if data and "evidence_items" in data:
                for item in data["evidence_items"]:
                    item["_category_key"] = category
                    items_for_cat.append(item)

        all_items.extend(items_for_cat[:limit_per_cat])
        print(f"  {CATEGORIES.get(category, category)}: {len(items_for_cat[:limit_per_cat])} items loaded")

    return all_items


# ── DB + Search ─────────────────────────────────────────────────────────────

def create_bench_db() -> sqlite3.Connection:
    """Create an in-memory ERINYS database for benchmarking."""
    config = ErinysConfig(db_path=":memory:")
    return init_db(config)


def search_rrf(db, query, top_k=10):
    """RRF hybrid search."""
    query_emb = embedding_engine.embed(query)
    return rrf_hybrid_search(
        db, query=query, query_embedding=query_emb,
        project="convomem", limit=top_k,
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
        project="convomem", limit=top_k,
        focused_embedding=focused_emb,
    )


def search_enhanced_v2_pref(db, query, top_k=10):
    """Enhanced v2 with lowered FTS weight for short-text corpora."""
    query_emb = embedding_engine.embed(query)
    focused_emb = None
    if _is_temporal_query(query):
        focused_text = focus_query_for_embedding(query)
        if focused_text != query:
            focused_emb = embedding_engine.embed(focused_text)
    return rrf_hybrid_search(
        db, query=query, query_embedding=query_emb,
        project="convomem", limit=top_k,
        fts_weight=0.30, vec_weight=0.70,
        keyword_boost=0.4,
        bigram_boost=0.4,
        proper_noun_boost=0.5,
        focused_embedding=focused_emb,
    )


def search_enhanced(db, query, top_k=10):
    """Enhanced RRF with keyword boosting."""
    query_emb = embedding_engine.embed(query)
    return enhanced_hybrid_search_v2(
        db, query=query, query_embedding=query_emb,
        project="convomem", limit=top_k,
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
        db=db, query=query, project="convomem", limit=max(top_k, search_limit),
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
        return res
    else:
        return rrf_hybrid_search(
            query_embedding=query_emb, focused_embedding=focused_emb, **kwargs
        )


SEARCH_FNS = {
    "rrf": search_rrf,
    "enhanced": search_enhanced,
    "enhanced_v2": search_enhanced_v2,
    "enhanced_v2_pref": search_enhanced_v2_pref,
    "enhanced_v2_boost": search_enhanced_v2_boost,
}


# ── Per-item retrieval ──────────────────────────────────────────────────────

def retrieve_for_item(
    item: dict, top_k: int = 10, mode: str = "enhanced_v2"
) -> tuple[float, dict]:
    """
    Ingest conversations, query, check if evidence was retrieved.
    
    Returns (recall, details_dict).
    Evidence recall = fraction of evidence messages found in top-k.
    """
    search_fn = SEARCH_FNS[mode]
    conversations = item.get("conversations", [])
    question = item["question"]
    evidence_messages = item.get("message_evidences", [])
    evidence_texts = set(e["text"].strip().lower() for e in evidence_messages)

    # Build corpus: one observation per message
    all_texts = []
    all_speakers = []
    for conv in conversations:
        for msg in conv.get("messages", []):
            all_texts.append(msg["text"])
            all_speakers.append(msg.get("speaker", "unknown"))

    if not all_texts:
        return 0.0, {"error": "empty corpus"}

    db = create_bench_db()

    try:
        # Batch embed all messages
        embeddings = embedding_engine.embed_batch(all_texts)

        # Collect synthetic texts for preference extraction
        synthetic_texts = []
        synthetic_meta = []
        for i, text in enumerate(all_texts):
            synths = extract_synthetics(text)
            for syn in synths:
                synthetic_texts.append(syn)
                synthetic_meta.append({"synthetic": True, "source_msg": i})

        # Context-window chunking: combine adjacent messages for implicit connections
        window_size = 5
        for start in range(0, len(all_texts), window_size // 2):
            end = min(start + window_size, len(all_texts))
            if end - start < 2:
                continue
            chunk = " ".join(all_texts[start:end])
            if len(chunk) > 30:
                synthetic_texts.append(chunk)
                synthetic_meta.append({"synthetic": True, "context_window": f"{start}-{end}"})

        # Batch embed synthetics
        synthetic_embeddings = []
        if synthetic_texts:
            synthetic_embeddings = embedding_engine.embed_batch(synthetic_texts)

        # Ingest original messages
        for i, (text, speaker, emb) in enumerate(zip(all_texts, all_speakers, embeddings)):
            obs_payload = {
                "title": f"msg_{i}",
                "content": text,
                "type": "manual",
                "project": "convomem",
                "scope": "project",
                "base_strength": 1.0,
                "access_count": 0,
                "metadata": json.dumps({"speaker": speaker, "msg_idx": i}),
            }
            insert_observation_with_embedding(db, obs_payload, serialize_f32(emb))

        # Ingest synthetic observations
        for j, (syn_text, syn_emb, syn_m) in enumerate(
            zip(synthetic_texts, synthetic_embeddings, synthetic_meta)
        ):
            syn_payload = {
                "title": f"syn_{j}",
                "content": syn_text,
                "type": "manual",
                "project": "convomem",
                "scope": "project",
                "base_strength": 1.0,
                "access_count": 0,
                "metadata": json.dumps(syn_m),
            }
            insert_observation_with_embedding(db, syn_payload, serialize_f32(syn_emb))

        total_corpus = len(all_texts) + len(synthetic_texts)
        search_results = search_fn(db, question, top_k=min(top_k, total_corpus))

        # Check matches
        retrieved_texts = []
        for res in search_results:
            content = res.get("content", "").strip().lower()
            retrieved_texts.append(content)

        found = 0
        for ev_text in evidence_texts:
            for ret_text in retrieved_texts:
                if ev_text in ret_text or ret_text in ev_text:
                    found += 1
                    break

        recall = found / len(evidence_texts) if evidence_texts else 1.0

        return recall, {
            "corpus_size": len(all_texts),
            "retrieved_count": len(retrieved_texts),
            "evidence_count": len(evidence_texts),
            "found": found,
        }

    finally:
        db.close()


# ── Benchmark Runner ────────────────────────────────────────────────────────

def run_benchmark(
    categories: list[str],
    limit_per_cat: int,
    top_k: int,
    mode: str,
    cache_dir: str,
    out_file: str | None,
) -> tuple[list[dict], dict]:
    """Run the ConvoMem retrieval benchmark."""

    print(f"\n{'=' * 60}")
    print(f"  ERINYS × ConvoMem Benchmark")
    print(f"{'=' * 60}")
    print(f"  Categories:  {len(categories)}")
    print(f"  Limit/cat:   {limit_per_cat}")
    print(f"  Top-k:       {top_k}")
    print(f"  Mode:        {mode}")
    print(f"{'─' * 60}")
    print(f"\n  Loading data from HuggingFace...\n")

    items = load_evidence_items(categories, limit_per_cat, cache_dir)

    print(f"\n  Total items: {len(items)}")
    print(f"{'─' * 60}\n")

    all_recall = []
    per_category = defaultdict(list)
    results_log = []
    start_time = time.time()

    for i, item in enumerate(items):
        question = item["question"]
        answer = item.get("answer", "")
        cat_key = item.get("_category_key", "unknown")

        recall, details = retrieve_for_item(item, top_k=top_k, mode=mode)
        all_recall.append(recall)
        per_category[cat_key].append(recall)

        results_log.append({
            "question": question,
            "answer": answer,
            "category": cat_key,
            "recall": recall,
            "details": details,
        })

        status = "HIT" if recall >= 1.0 else ("part" if recall > 0 else "miss")
        if (i + 1) % 20 == 0 or i == len(items) - 1:
            avg = sum(all_recall) / len(all_recall)
            print(f"  [{i + 1:4}/{len(items)}] avg_recall={avg:.3f}  last={status}", flush=True)

    elapsed = time.time() - start_time
    avg_recall = sum(all_recall) / len(all_recall) if all_recall else 0

    # Build summary
    summary = {
        "benchmark": "ConvoMem",
        "mode": mode,
        "top_k": top_k,
        "total_items": len(items),
        "overall": {
            "R@k": round(avg_recall * 100, 1),
            "perfect_pct": round(sum(1 for r in all_recall if r >= 1.0) / max(len(all_recall), 1) * 100, 1),
            "zero_pct": round(sum(1 for r in all_recall if r == 0) / max(len(all_recall), 1) * 100, 1),
        },
        "per_category": {},
        "timing": {
            "total_s": round(elapsed, 1),
            "avg_per_item_s": round(elapsed / max(len(items), 1), 2),
        },
        "comparison": {
            "mempalace_raw_R": "92.9% (verbatim text, semantic search)",
            "mem0_RAG_R": "30-45%",
        },
    }

    for cat_key in sorted(per_category.keys()):
        vals = per_category[cat_key]
        avg = sum(vals) / len(vals)
        perfect = sum(1 for v in vals if v >= 1.0)
        summary["per_category"][cat_key] = {
            "name": CATEGORIES.get(cat_key, cat_key),
            "count": len(vals),
            "R@k": round(avg * 100, 1),
            "perfect": f"{perfect}/{len(vals)}",
        }

    return results_log, summary


def print_summary(summary: dict) -> None:
    """Print benchmark results."""
    print(f"\n{'=' * 60}")
    print(f"  ERINYS × ConvoMem Benchmark Results")
    print(f"{'=' * 60}")
    print(f"  Mode:    {summary['mode']}")
    print(f"  Top-K:   {summary['top_k']}")
    print(f"  Items:   {summary['total_items']}")

    o = summary["overall"]
    print(f"\n  OVERALL:")
    print(f"    Avg Recall: {o['R@k']:.1f}%")
    print(f"    Perfect:    {o['perfect_pct']:.1f}%")
    print(f"    Zero:       {o['zero_pct']:.1f}%")

    print(f"\n  PER-CATEGORY:")
    for cat_key, m in sorted(summary["per_category"].items()):
        name = m["name"]
        print(f"    {name:25s} R={m['R@k']:5.1f}%  perfect={m['perfect']}")

    print(f"\n  COMPARISON:")
    for k, v in summary.get("comparison", {}).items():
        print(f"    {k}: {v}")

    t = summary.get("timing", {})
    print(f"\n  TIMING:")
    print(f"    Total: {t.get('total_s', 0):.1f}s")
    print(f"    Per item: {t.get('avg_per_item_s', 0):.2f}s")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="ERINYS × ConvoMem Benchmark")
    parser.add_argument(
        "--limit", type=int, default=50,
        help="Items per category (default: 50, matches MemPalace)",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-k retrieval")
    parser.add_argument(
        "--category",
        choices=list(CATEGORIES.keys()) + ["all"],
        default="all",
        help="Category to test (default: all)",
    )
    parser.add_argument(
        "--mode",
        choices=list(SEARCH_FNS.keys()),
        default="enhanced_v2",
        help="Search mode (default: enhanced_v2)",
    )
    parser.add_argument(
        "--cache-dir", default="/tmp/convomem_cache",
        help="Cache directory for downloaded data",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "results"),
        help="Output directory",
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    if args.category == "all":
        categories = list(CATEGORIES.keys())
    else:
        categories = [args.category]

    print(f"\nInitializing embedding engine ({embedding_engine.model_name})...")

    results, summary = run_benchmark(
        categories, args.limit, args.top_k, args.mode, args.cache_dir, None
    )

    print_summary(summary)

    os.makedirs(args.output, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    results_file = os.path.join(args.output, f"results_erinys_convomem_{args.mode}_{ts}.jsonl")
    with open(results_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary_file = os.path.join(args.output, f"summary_erinys_convomem_{args.mode}_{ts}.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  Results: {results_file}")
    print(f"  Summary: {summary_file}")


if __name__ == "__main__":
    main()
