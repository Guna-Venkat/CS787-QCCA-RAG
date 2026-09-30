#!/usr/bin/env bash
# 04_generate_report.sh - Week 3 Plot and Scientific Report Generation
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PROJECT_ROOT}/sara_env/bin/python"
LOG_DIR="${PROJECT_ROOT}/results/week3/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/04_generate_report_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

echo "==========================================================" | tee -a "${LOG_FILE}"
echo "START: Week 3 Report & Plot Generation" | tee -a "${LOG_FILE}"
echo "Timestamp: $(date -Iseconds)" | tee -a "${LOG_FILE}"
echo "Git Commit: $(git rev-parse HEAD)" | tee -a "${LOG_FILE}"
echo "Python: ${PYTHON}" | tee -a "${LOG_FILE}"
echo "==========================================================" | tee -a "${LOG_FILE}"

cd "${PROJECT_ROOT}"

echo ">>> Generating publication plots..." | tee -a "${LOG_FILE}"
"${PYTHON}" -m src.week3.generate_plots 2>&1 | tee -a "${LOG_FILE}"

echo ">>> Generating comprehensive scientific report, manifests, and gate..." | tee -a "${LOG_FILE}"
"${PYTHON}" -m src.week3.generate_scientific_report 2>&1 | tee -a "${LOG_FILE}"

echo "Report & Plot Generation Complete: $(date -Iseconds)" | tee -a "${LOG_FILE}"
