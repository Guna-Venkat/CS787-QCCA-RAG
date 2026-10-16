from __future__ import annotations

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.week8.semantic import SemanticScorer
from src.week8.data import load_config, load_week8_split
from src.week8.marginal_information import (
    compute_candidate_marginal_information,
    tune_mig_hyperparameters,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="W8-D Feature Engineering Runner")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test on 5 queries")
    parser.add_argument("--full", action="store_true", help="Run full feature extraction")
    parser.add_argument("--split", choices=["fit", "internal_val"], default="internal_val")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke-limit", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config()
    split = args.split
    smoke_limit = args.smoke_limit if args.smoke else None
    tag = "_smoke" if args.smoke else ""

    logger.info("Starting W8-D Feature Engineering (split=%s, smoke=%s)", split, args.smoke)

    records, _, _ = load_week8_split(split, cfg)
    items = records[:smoke_limit] if smoke_limit else records

    # 1. Marginal Information Gain computation
    scorer = SemanticScorer()
    mig_rows = []
    full_indices = [0, 1, 2, 3]
    candidate_indices = [4, 5, 6, 7, 8, 9]

    for item in items:
        qid = str(item["question_id"])
        q_text = item["question"]
        passages = item.get("passages", [])[:10]
        draft_text = f"Draft answer for {q_text}"

        mig_feats = compute_candidate_marginal_information(
            query_text=q_text,
            draft_text=draft_text,
            passages=passages,
            full_indices=full_indices,
            candidate_indices=candidate_indices,
            alpha=0.5,
            beta=0.5,
            semantic_scorer=scorer,
        )

        for c_idx, feats in mig_feats.items():
            row = {"question_id": qid, "candidate_passage_index": c_idx}
            row.update(feats)
            mig_rows.append(row)

    df_mig = pd.DataFrame(mig_rows)
    logger.info("Extracted MIG features for %d candidates across %d queries.", len(df_mig), len(items))

    # Save features
    out_tables = REPO_ROOT / "results/week8/tables"
    out_tables.mkdir(parents=True, exist_ok=True)
    df_mig.to_csv(out_tables / f"w8d_mig_features_{split}{tag}.csv", index=False)
    logger.info("Saved W8-D feature matrix to %s", out_tables)


if __name__ == "__main__":
    main()
