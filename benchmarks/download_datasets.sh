#!/bin/bash
# Download benchmark datasets for ERINYS evaluation
# LoCoMo: from snap-research/locomo GitHub repo
# ConvoMem: auto-downloaded from HuggingFace by convomem_bench.py

set -e

LOCOMO_DIR="/tmp/locomo-data"
LOCOMO_REPO="/tmp/locomo-repo"

echo "=== ERINYS Benchmark Dataset Downloader ==="
echo ""

# ── LoCoMo ──────────────────────────────────────────────────────────────────
echo "[1/2] LoCoMo dataset..."

if [ -f "${LOCOMO_DIR}/locomo10.json" ]; then
    echo "  Already exists: ${LOCOMO_DIR}/locomo10.json"
else
    mkdir -p "${LOCOMO_DIR}"
    
    if [ -d "${LOCOMO_REPO}" ]; then
        echo "  Using existing repo: ${LOCOMO_REPO}"
    else
        echo "  Cloning snap-research/locomo..."
        git clone --depth 1 https://github.com/snap-research/locomo.git "${LOCOMO_REPO}"
    fi
    
    if [ -f "${LOCOMO_REPO}/data/locomo10.json" ]; then
        cp "${LOCOMO_REPO}/data/locomo10.json" "${LOCOMO_DIR}/locomo10.json"
        echo "  Copied to: ${LOCOMO_DIR}/locomo10.json"
    else
        echo "  ERROR: locomo10.json not found in repo. Checking..."
        ls -la "${LOCOMO_REPO}/data/" 2>/dev/null || echo "  data/ dir not found"
        exit 1
    fi
fi

# Verify
if [ -f "${LOCOMO_DIR}/locomo10.json" ]; then
    SIZE=$(wc -c < "${LOCOMO_DIR}/locomo10.json" | tr -d ' ')
    echo "  Verified: ${SIZE} bytes"
fi

# ── ConvoMem ────────────────────────────────────────────────────────────────
echo ""
echo "[2/2] ConvoMem dataset..."
echo "  ConvoMem data is auto-downloaded from HuggingFace by convomem_bench.py"
echo "  Cache dir: /tmp/convomem_cache"
echo "  No manual download needed."

echo ""
echo "=== Done ==="
echo ""
echo "Run benchmarks:"
echo "  python benchmarks/locomo_bench.py ${LOCOMO_DIR}/locomo10.json --top-k 5"
echo "  python benchmarks/convomem_bench.py --limit 50 --category all"
