#!/usr/bin/env bash
set -euo pipefail

echo "=== SANE — Download Benchmark Data ==="

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
THIRD_PARTY="$PROJECT_ROOT/third_party"
mkdir -p "$THIRD_PARTY"

# ── FinanceBench third-party data (PDFs + JSONL) ────────────────────
# Required before running FinanceBench — the benchmark reads PDFs and
# question files from this directory.
echo ""
echo "--- FinanceBench (third-party data) ---"
FB_DIR="$THIRD_PARTY/financebench"
if [ -d "$FB_DIR" ]; then
    echo "[SKIP] FinanceBench already exists at third_party/financebench"
else
    echo "[CLONE] FinanceBench → third_party/financebench"
    git clone "https://github.com/patronus-ai/financebench.git" "$FB_DIR" --depth 1
fi

# ── HotpotQA (HuggingFace) ──────────────────────────────────────────
echo ""
echo "--- HotpotQA ---"
python -c "
from benchmarks.hotpotqa.data import load_hotpotqa_data
data = load_hotpotqa_data()
print(f'{len(data)} instances ready')
"

# ── MuSiQue (HuggingFace) ───────────────────────────────────────────
echo ""
echo "--- MuSiQue ---"
python -c "
from benchmarks.musique.data import load_musique_data
data = load_musique_data()
print(f'{len(data)} instances ready')
"

# ── LongMemEval (HuggingFace) ───────────────────────────────────────
echo ""
echo "--- LongMemEval ---"
python -c "
from benchmarks.longmemeval.data import load_longmemeval_data
for split in ('s_cleaned', 'm_cleaned'):
    data = load_longmemeval_data(split)
    print(f'{split}: {len(data)} instances ready')
"

# ── FinanceBench (PDF extraction) ───────────────────────────────────
echo ""
echo "--- FinanceBench (PDF extraction) ---"
python benchmarks/financebench/setup.py --workers 8

echo ""
echo "=== All data ready ==="
