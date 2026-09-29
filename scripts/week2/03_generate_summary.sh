#!/usr/bin/env bash
# Stage 3: Aggregate metrics, determine k_dev*, and generate Pareto plots
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_EXEC="/home/gunavenkat/sara_env/bin/python"

CONFIG_FILE="${WORKSPACE_ROOT}/configs/week2/sweep_config.yaml"

echo "============================================================"
echo "STAGE 3: METRICS AGGREGATION & PARETO PLOTTING"
echo "Start Time   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Workspace    : ${WORKSPACE_ROOT}"
echo "Python       : ${PYTHON_EXEC}"
echo "Config       : ${CONFIG_FILE}"
echo "============================================================"

cd "${WORKSPACE_ROOT}"
"${PYTHON_EXEC}" -m src.week2.metrics_aggregator --config "${CONFIG_FILE}"

echo "============================================================"
echo "STAGE 3 COMPLETE: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
