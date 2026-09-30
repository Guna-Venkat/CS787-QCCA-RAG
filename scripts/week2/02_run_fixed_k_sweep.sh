#!/usr/bin/env bash
# Stage 2: Execute Fixed-k Development Baseline Sweep across k in {0,2,4,5,6,8,10}
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_EXEC="/home/gunavenkat/sara_env/bin/python"

CONFIG_FILE="${WORKSPACE_ROOT}/configs/week2/sweep_config.yaml"
LIMIT="${1:-}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "============================================================"
echo "STAGE 2: FIXED-K DEVELOPMENT RESPONSE SWEEP"
echo "Start Time   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Workspace    : ${WORKSPACE_ROOT}"
echo "Python       : ${PYTHON_EXEC}"
echo "Config       : ${CONFIG_FILE}"
echo "Question Limit: ${LIMIT:-ALL (231)}"
echo "============================================================"

CMD=(
    env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
    "${PYTHON_EXEC}" -m src.week2.sweep_runner --config "${CONFIG_FILE}"
)

if [[ -n "${LIMIT}" ]]; then
    CMD+=(--limit "${LIMIT}")
fi

cd "${WORKSPACE_ROOT}"
"${CMD[@]}"

echo "============================================================"
echo "STAGE 2 COMPLETE: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
