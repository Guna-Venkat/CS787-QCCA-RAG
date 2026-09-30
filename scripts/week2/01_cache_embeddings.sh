#!/usr/bin/env bash
# Stage 1: Precompute BM25 retrieval and SFR passage embeddings for Week 2
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_EXEC="/home/gunavenkat/sara_env/bin/python"

CONFIG_FILE="${WORKSPACE_ROOT}/configs/week2/sweep_config.yaml"
LIMIT="${1:-}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "============================================================"
echo "STAGE 1: BM25 RETRIEVAL + SFR EMBEDDING CACHE"
echo "Start Time   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Workspace    : ${WORKSPACE_ROOT}"
echo "Python       : ${PYTHON_EXEC}"
echo "Config       : ${CONFIG_FILE}"
echo "Question Limit: ${LIMIT:-ALL (231)}"
echo "============================================================"

CMD=("${PYTHON_EXEC}" -m src.week2.precompute_dev_embeddings --config "${CONFIG_FILE}")
if [[ -n "${LIMIT}" ]]; then
    CMD+=(--limit "${LIMIT}")
fi

cd "${WORKSPACE_ROOT}"
"${CMD[@]}"

# Also extract pre-generation retrieval features
echo "Extracting retrieval distribution features..."
"${PYTHON_EXEC}" -c "
from pathlib import Path
from src.week2.feature_extractor import extract_features_from_retrieval_records
extract_features_from_retrieval_records(
    retrieval_file='${WORKSPACE_ROOT}/results/week2/raw/dev_retrieval.jsonl',
    output_file='${WORKSPACE_ROOT}/results/week2/processed/dev_retrieval_features.jsonl'
)
print('Saved retrieval features to results/week2/processed/dev_retrieval_features.jsonl')
"

echo "============================================================"
echo "STAGE 1 COMPLETE: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
