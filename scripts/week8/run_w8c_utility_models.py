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

from src.week8.data import load_config
from src.week8.utility_dataset import (
    build_utility_dataset,
    get_non_leaking_feature_columns,
)
from src.week8.utility_models import (
    TabularUtilityModel_U0,
    format_u1_text,
    format_u2_text,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="W8-C Utility Models Runner")
    parser.add_argument("--smoke", action="store_true", help="Run smoke evaluation on tiny synthetic/subset data")
    parser.add_argument("--full", action="store_true", help="Run full training and evaluation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tag = "_smoke" if args.smoke else ""
    logger.info("Starting W8-C Utility Models Runner (smoke=%s, device=%s)", args.smoke, args.device)

    # 1. Load or synthesize dataset
    prom_path = REPO_ROOT / "results/week8/surfaces" / f"promotion_surface_internal_val{tag}.jsonl"
    draft_path = REPO_ROOT / "results/week8/drafts" / f"draft_diagnostics_internal_val{tag}.jsonl"

    if prom_path.is_file() and draft_path.is_file():
        df_prom = pd.read_json(prom_path, lines=True)
        df_draft = pd.read_json(draft_path, lines=True)
    else:
        logger.warning("Promotion/draft cache not found at %s. Creating synthetic demo data for smoke verification.", prom_path)
        # Create minimal synthetic data for pipeline validation
        df_prom = pd.DataFrame([
            {"question_id": f"q_{i}", "paper_id": f"p_{i//2}", "candidate_passage_index": 4 + (i % 6), "base_f1": 0.35, "promoted_f1": 0.40 if i % 2 == 0 else 0.30, "delta_f1": 0.05 if i % 2 == 0 else -0.05, "base_context_tokens": 900, "promoted_context_tokens": 1100, "semantic_rank": 5 + (i % 6)}
            for i in range(10)
        ])
        df_draft = pd.DataFrame([
            {"question_id": f"q_{i}", "draft_answer": f"Draft answer {i}", "context_tokens": 900}
            for i in range(10)
        ])

    df_util = build_utility_dataset(df_prom, df_draft)
    logger.info("Constructed utility dataset with %d rows.", len(df_util))

    # 2. Train U0 Baseline
    feat_cols = ["base_context_tokens", "promoted_context_tokens", "semantic_rank"]
    X = df_util[feat_cols].values
    y_help = df_util["target_help_binary"].values
    y_hurt = df_util["target_hurt_binary"].values
    y_delta = df_util["target_continuous_delta"].values
    y_3way = df_util["target_3way_label"].values

    u0 = TabularUtilityModel_U0()
    u0.fit(X, y_help, y_hurt, y_delta, y_3way)
    preds_u0 = u0.predict(X)

    logger.info("U0 Mean Predicted P(HELP): %.4f, P(HURT): %.4f, Pred Delta: %.4f",
                float(np.mean(preds_u0["prob_help"])),
                float(np.mean(preds_u0["prob_hurt"])),
                float(np.mean(preds_u0["pred_delta"])))

    # Save results
    out_tables = REPO_ROOT / "results/week8/tables"
    out_tables.mkdir(parents=True, exist_ok=True)
    df_eval = pd.DataFrame({
        "question_id": df_util["question_id"],
        "true_delta_f1": y_delta,
        "u0_pred_delta": preds_u0["pred_delta"],
        "u0_prob_help": preds_u0["prob_help"],
        "u0_prob_hurt": preds_u0["prob_hurt"],
    })
    df_eval.to_csv(out_tables / f"w8c_utility_predictions{tag}.csv", index=False)
    logger.info("W8-C runner completed successfully. Results saved to %s", out_tables)


if __name__ == "__main__":
    main()
