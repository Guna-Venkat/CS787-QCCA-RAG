#!/usr/bin/env bash
# 01_oracle_analysis.sh - Week 3 Quality Oracle & Headroom Analysis
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PROJECT_ROOT}/sara_env/bin/python"
LOG_DIR="${PROJECT_ROOT}/results/week3/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/01_oracle_analysis_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

echo "==========================================================" | tee -a "${LOG_FILE}"
echo "START: Week 3 Quality Oracle Analysis" | tee -a "${LOG_FILE}"
echo "Timestamp: $(date -Iseconds)" | tee -a "${LOG_FILE}"
echo "Git Commit: $(git rev-parse HEAD)" | tee -a "${LOG_FILE}"
echo "Python: ${PYTHON}" | tee -a "${LOG_FILE}"
echo "Log File: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "==========================================================" | tee -a "${LOG_FILE}"

cd "${PROJECT_ROOT}"

"${PYTHON}" -m src.week3.oracle_analysis \
    --raw_matrix "${PROJECT_ROOT}/results/week2/raw/dev_fixed_k_matrix.jsonl" \
    --best_static "${PROJECT_ROOT}/results/week2/processed/best_static_baseline.json" \
    --output_dir "${PROJECT_ROOT}/results/week3/processed" 2>&1 | tee -a "${LOG_FILE}"

echo "Quality Oracle Analysis Complete: $(date -Iseconds)" | tee -a "${LOG_FILE}"
