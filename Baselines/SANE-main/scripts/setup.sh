#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== SANE Environment Setup ==="

# ── Conda environment ──────────────────────────────────────────────
if ! conda info --envs 2>/dev/null | grep -q "^rag "; then
    echo "Creating conda environment 'rag'..."
    conda create -n rag python=3.11 -y
fi

echo "Installing dependencies..."
conda run -n rag pip install -r requirements.txt

# ── .env template ──────────────────────────────────────────────────
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example — edit it with your API keys"
fi

echo ""
echo "Setup complete. Next steps:"
echo "  1. bash scripts/download_data.sh        # download benchmark data"
echo "  2. Edit .env with your API keys"
echo "  3. conda activate rag"
echo "  4. bash scripts/run_all.sh"
