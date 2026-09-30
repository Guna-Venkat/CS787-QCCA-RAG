#!/usr/bin/env bash
# 03_human_audit.sh - Week 3 Blinded 50-Question Human Audit Pipeline
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PROJECT_ROOT}/sara_env/bin/python"
LOG_DIR="${PROJECT_ROOT}/results/week3/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/03_human_audit_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

echo "==========================================================" | tee -a "${LOG_FILE}"
echo "START: Week 3 Blinded Human Audit" | tee -a "${LOG_FILE}"
echo "Timestamp: $(date -Iseconds)" | tee -a "${LOG_FILE}"
echo "Git Commit: $(git rev-parse HEAD)" | tee -a "${LOG_FILE}"
echo "Python: ${PYTHON}" | tee -a "${LOG_FILE}"
echo "==========================================================" | tee -a "${LOG_FILE}"

cd "${PROJECT_ROOT}"

"${PYTHON}" -m src.week3.human_audit \
    --raw_matrix "${PROJECT_ROOT}/results/week2/raw/dev_fixed_k_matrix.jsonl" \
    --oracle "${PROJECT_ROOT}/results/week3/processed/dev_quality_oracle.csv" \
    --output_dir "${PROJECT_ROOT}/results/week3/human_audit" 2>&1 | tee -a "${LOG_FILE}"

echo "Human Audit Analysis Complete: $(date -Iseconds)" | tee -a "${LOG_FILE}"
