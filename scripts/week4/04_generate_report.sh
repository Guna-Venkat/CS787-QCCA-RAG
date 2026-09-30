#!/usr/bin/env bash
# 04_generate_report.sh - Week 4 Report and Plot Generation
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PROJECT_ROOT}/sara_env/bin/python"
LOG_DIR="${PROJECT_ROOT}/results/week4/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/04_generate_report_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

echo "==========================================================" | tee -a "${LOG_FILE}"
echo "START: Week 4 Report & Plot Regeneration" | tee -a "${LOG_FILE}"
echo "Timestamp: $(date -Iseconds)" | tee -a "${LOG_FILE}"
echo "Git Commit: $(git rev-parse HEAD)" | tee -a "${LOG_FILE}"
echo "Python: ${PYTHON}" | tee -a "${LOG_FILE}"
echo "==========================================================" | tee -a "${LOG_FILE}"

cd "${PROJECT_ROOT}"

"${PYTHON}" -m src.week4.train_qcca "$@" 2>&1 | tee -a "${LOG_FILE}"

echo "Report & plot generation complete: $(date -Iseconds)" | tee -a "${LOG_FILE}"
