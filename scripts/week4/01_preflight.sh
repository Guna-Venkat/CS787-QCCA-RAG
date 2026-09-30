#!/usr/bin/env bash
# 01_preflight.sh - Week 4 Pre-flight Verification Script
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PROJECT_ROOT}/sara_env/bin/python"
LOG_DIR="${PROJECT_ROOT}/results/week4/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/01_preflight_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

echo "==========================================================" | tee -a "${LOG_FILE}"
echo "START: Week 4 Pre-flight Verification" | tee -a "${LOG_FILE}"
echo "Timestamp: $(date -Iseconds)" | tee -a "${LOG_FILE}"
echo "Git Commit: $(git rev-parse HEAD)" | tee -a "${LOG_FILE}"
echo "Python: ${PYTHON}" | tee -a "${LOG_FILE}"
echo "==========================================================" | tee -a "${LOG_FILE}"

cd "${PROJECT_ROOT}"

echo ">>> Running Week 4 unit tests (features, targets, CV, models, evaluation)..." | tee -a "${LOG_FILE}"
"${PYTHON}" -m pytest tests/week4/ -v 2>&1 | tee -a "${LOG_FILE}"

echo ">>> Pre-flight verification completed successfully." | tee -a "${LOG_FILE}"
