import json
import glob
import os

benchmarks = {
    "locomo": "locomo",
    "convomem": "convomem",
    "longmemeval": ""
}
mode = "enhanced_v2_boost"

print("======================================================================")
print(f"  ERINYS Benchmark Integration Results (Mode: {mode})")
print("======================================================================\n")

for name, prefix in benchmarks.items():
    search_prefix = f"summary_erinys_{prefix}_{mode}_*.json" if prefix else f"summary_erinys_{mode}_*.json"
    files = glob.glob(f"benchmarks/results/{search_prefix}")
    if files:
        latest = max(files, key=os.path.getmtime)
        with open(latest) as f:
            data = json.load(f)
            
        print(f"[{name.upper()} Benchmark]")
        o = data.get("overall", {})
        
        if name == "locomo":
            print(f"  R@5:    {o.get('R@5', 0):.2f}%")
            print(f"  R@10:   {o.get('R@10', 0):.2f}%")
            print(f"  NDCG@5: {o.get('NDCG@5', 0):.3f}")
            print(f"  NDCG@10:{o.get('NDCG@10', 0):.3f}")
        elif name == "convomem":
            print(f"  R@k:    {o.get('R@k', 0):.2f}%")
            print(f"  (Note: ConvoMem evaluates dynamic-k where k=avg_targets)")
        elif name == "longmemeval":
            print(f"  R@5:    {o.get('R@5', 0):.2f}%")
            print(f"  R@10:   {o.get('R@10', 0):.2f}%")
        print()
    else:
        print(f"[{name.upper()}] No summary found.")

print("======================================================================")
