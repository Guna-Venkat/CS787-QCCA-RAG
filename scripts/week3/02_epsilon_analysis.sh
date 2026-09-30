#!/usr/bin/env bash
# 02_epsilon_analysis.sh - Week 3 Epsilon Oracle & Heterogeneity Analysis
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PROJECT_ROOT}/sara_env/bin/python"
LOG_DIR="${PROJECT_ROOT}/results/week3/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/02_epsilon_analysis_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

echo "==========================================================" | tee -a "${LOG_FILE}"
echo "START: Week 3 Epsilon & Heterogeneity Analysis" | tee -a "${LOG_FILE}"
echo "Timestamp: $(date -Iseconds)" | tee -a "${LOG_FILE}"
echo "Git Commit: $(git rev-parse HEAD)" | tee -a "${LOG_FILE}"
echo "Python: ${PYTHON}" | tee -a "${LOG_FILE}"
echo "==========================================================" | tee -a "${LOG_FILE}"

cd "${PROJECT_ROOT}"

echo ">>> Running Epsilon Oracle Analysis..." | tee -a "${LOG_FILE}"
"${PYTHON}" -m src.week3.epsilon_analysis \
    --raw_matrix "${PROJECT_ROOT}/results/week2/raw/dev_fixed_k_matrix.jsonl" \
    --output_dir "${PROJECT_ROOT}/results/week3/processed" 2>&1 | tee -a "${LOG_FILE}"

echo ">>> Running Per-Query Heterogeneity Analysis..." | tee -a "${LOG_FILE}"
"${PYTHON}" -m src.week3.heterogeneity_analysis \
    --raw_matrix "${PROJECT_ROOT}/results/week2/raw/dev_fixed_k_matrix.jsonl" \
    --best_static "${PROJECT_ROOT}/results/week2/processed/best_static_baseline.json" \
    --output_dir "${PROJECT_ROOT}/results/week3/processed" 2>&1 | tee -a "${LOG_FILE}"

echo ">>> Running Feature Signal Analysis..." | tee -a "${LOG_FILE}"
"${PYTHON}" -m src.week3.feature_signal_analysis \
    --features "${PROJECT_ROOT}/results/week2/processed/dev_retrieval_features.jsonl" \
    --oracle "${PROJECT_ROOT}/results/week3/processed/dev_quality_oracle.csv" \
    --epsilon "${PROJECT_ROOT}/results/week3/processed/epsilon_oracle_all.csv" \
    --het "${PROJECT_ROOT}/results/week3/processed/per_query_heterogeneity.csv" \
    --output_dir "${PROJECT_ROOT}/results/week3/processed" 2>&1 | tee -a "${LOG_FILE}"

echo "Epsilon & Heterogeneity Analysis Complete: $(date -Iseconds)" | tee -a "${LOG_FILE}"
