import subprocess
import glob
import os
import json

benchmarks = ["locomo_bench.py", "convomem_bench.py", "longmemeval_bench.py"]
modes = ["enhanced_v2_boost"]
results = {}

sys_python = "/Users/fujiyoshi/Library/CloudStorage/Dropbox/002_work/KG_Antigravity/projects/erinys-memory/.venv/bin/python3"

for bench in benchmarks[1:]:
    print(f"Running {bench}...")
    try:
        subprocess.run([
            sys_python, 
            f"benchmarks/{bench}", 
            "--mode", "enhanced_v2_boost"
        ], cwd="/Users/fujiyoshi/Library/CloudStorage/Dropbox/002_work/KG_Antigravity/projects/erinys-memory", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"Failed to run {bench}: {e}")

for bench in benchmarks:
    prefix = bench.replace("_bench.py", "")
    files = glob.glob(f"benchmarks/results/summary_erinys_{prefix}_enhanced_v2_boost_*.json")
    if files:
        latest_file = max(files, key=os.path.getmtime)
        with open(latest_file) as f:
            data = json.load(f)
            results[bench] = data
    else:
        print(f"Could not find summary for {bench}")

print("\n--- Final Comparison across 3 Benchmarks (Mode: enhanced_v2_boost) ---")
for bench in benchmarks:
    if bench in results:
        r = results[bench].get("overall", {})
        if "R@5" in r:
            print(f"[{bench}] R@5: {r['R@5']:.2f}% | R@10: {r['R@10']:.2f}%")
        elif "R@k" in r:
            print(f"[{bench}] R@k: {r['R@k']:.2f}%")
        if "NDCG@5" in r:
            print(f"          NDCG@5: {r['NDCG@5']:.3f} | NDCG@10: {r['NDCG@10']:.3f}")
