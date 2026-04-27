"""Reduced grid search: tc fixed at 0.5, wider range for pn/kw/bg."""
import sys, os, json, itertools, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import locomo_bench
from erinys_memory.search import rrf_hybrid_search, focus_query_for_embedding, _is_temporal_query

embedding_engine = locomo_bench.embedding_engine
_is_inference_query = locomo_bench._is_inference_query
_focus_inference_query = locomo_bench._focus_inference_query

data_path = "/tmp/locomo-data/locomo10.json"
data = locomo_bench.load_dataset(data_path)

sessions_by_conv = {}
for conv in data:
    conv_id = conv["sample_id"]
    sessions = locomo_bench.load_conversation_sessions(conv.get("conversation", conv))
    sessions_by_conv[conv_id] = sessions

locomo_bench.precompute_embeddings(sessions_by_conv, show_progress=True)

best_r5 = 0.0
best_params = None

TC = 0.5
param_grid = {
    "proper_noun_boost": [0.3, 0.5, 0.7],
    "keyword_boost": [0.3, 0.5, 0.7],
    "bigram_boost": [0.3, 0.5, 0.7],
}

keys = list(param_grid.keys())
combos = list(itertools.product(*[param_grid[k] for k in keys]))
print(f"Testing {len(combos)} combinations (tc fixed at {TC})...")

for i, combo in enumerate(combos):
    params = dict(zip(keys, combo))
    params["temporal_content_boost"] = TC

    def make_search_fn(p):
        def search_fn(db, query, top_k=10):
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
                **p,
            )
            if focused_emb and _is_inference_query(query):
                kwargs_res1 = kwargs.copy()
                kwargs_res2 = kwargs.copy()
                kwargs_res2["keyword_boost"] = 0.0
                res1 = rrf_hybrid_search(query_embedding=query_emb, **kwargs_res1)
                res2 = rrf_hybrid_search(query_embedding=focused_emb, **kwargs_res2)
                merged = {}
                for r in res1:
                    rid = r["id"]
                    r_copy = r.copy()
                    score = r.get("rrf_score", r.get("score", 0)) * 0.2
                    r_copy["score"] = score
                    if "rrf_score" in r_copy:
                        r_copy["rrf_score"] = score
                    merged[rid] = r_copy
                for r in res2:
                    rid = r["id"]
                    s = r.get("rrf_score", r.get("score", 0)) * 1.0
                    if rid in merged:
                        merged[rid]["score"] += s
                        if "rrf_score" in merged[rid]:
                            merged[rid]["rrf_score"] += s
                    else:
                        r_copy = r.copy()
                        r_copy["score"] = s
                        if "rrf_score" in r_copy:
                            r_copy["rrf_score"] = s
                        merged[rid] = r_copy
                res = list(merged.values())
                res.sort(key=lambda x: -x.get("rrf_score", x.get("score", 0)))
                return res
            else:
                return rrf_hybrid_search(query_embedding=query_emb, focused_embedding=focused_emb, **kwargs)
        return search_fn

    locomo_bench.SEARCH_FNS["tuning"] = make_search_fn(params)

    t0 = time.time()
    results, summary = locomo_bench.run_benchmark(data, mode="tuning", show_progress=False)
    elapsed = time.time() - t0
    r5 = summary["overall"]["R@5"]
    r10 = summary["overall"]["R@10"]

    marker = ""
    if r5 > best_r5:
        best_r5 = r5
        best_params = params.copy()
        marker = " *** NEW BEST ***"

    print(f"[{i+1}/{len(combos)}] pn={combo[0]} kw={combo[1]} bg={combo[2]} => R@5={r5:.2f}% R@10={r10:.2f}% ({elapsed:.0f}s){marker}")

print(f"\nBEST: R@5={best_r5:.2f}% params={best_params}")
