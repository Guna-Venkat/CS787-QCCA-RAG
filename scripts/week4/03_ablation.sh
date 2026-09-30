#!/usr/bin/env bash
# 03_ablation.sh - Week 4 Feature Ablation and Sensitivity Analysis
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PROJECT_ROOT}/sara_env/bin/python"
LOG_DIR="${PROJECT_ROOT}/results/week4/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/03_ablation_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

echo "==========================================================" | tee -a "${LOG_FILE}"
echo "START: Week 4 Feature Ablation Analysis" | tee -a "${LOG_FILE}"
echo "Timestamp: $(date -Iseconds)" | tee -a "${LOG_FILE}"
echo "Git Commit: $(git rev-parse HEAD)" | tee -a "${LOG_FILE}"
echo "Python: ${PYTHON}" | tee -a "${LOG_FILE}"
echo "==========================================================" | tee -a "${LOG_FILE}"

cd "${PROJECT_ROOT}"

# Ablation is executed as part of the master runner pipeline
"${PYTHON}" -m src.week4.train_qcca "$@" 2>&1 | tee -a "${LOG_FILE}"

echo "Feature ablation complete: $(date -Iseconds)" | tee -a "${LOG_FILE}"
