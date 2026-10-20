#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== SANE — Full Sweep ==="
echo ""

# ── HotpotQA ──────────────────────────────────────────────────────
echo "--- HotpotQA (rag) ---"
python -m benchmarks.hotpotqa.run --method rag --config config/hotpotqa.yaml --workers 50
echo "--- HotpotQA (sane) ---"
python -m benchmarks.hotpotqa.run --method sane --config config/hotpotqa.yaml --workers 50

# ── MuSiQue ───────────────────────────────────────────────────────
echo "--- MuSiQue (rag) ---"
python -m benchmarks.musique.run --method rag --config config/musique.yaml --workers 50
echo "--- MuSiQue (sane) ---"
python -m benchmarks.musique.run --method sane --config config/musique.yaml --workers 50

# ── FinanceBench ──────────────────────────────────────────────────
echo "--- FinanceBench (rag) ---"
python -m benchmarks.financebench.run --method rag --config config/financebench.yaml --workers 10 --judge-workers 16
echo "--- FinanceBench (sane) ---"
python -m benchmarks.financebench.run --method sane --config config/financebench.yaml --workers 10 --judge-workers 16

# ── LongMemEval ───────────────────────────────────────────────────
echo "--- LongMemEval (rag) ---"
python -m benchmarks.longmemeval.run --split m_cleaned --method rag --config config/longmemeval.yaml --workers 25 --judge-workers 16
echo "--- LongMemEval (sane) ---"
python -m benchmarks.longmemeval.run --split m_cleaned --method sane --config config/longmemeval.yaml --workers 25 --judge-workers 16

echo ""
echo "=== Done ==="
