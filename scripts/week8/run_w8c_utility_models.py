"""CLI Runner for W8-C Passage Utility Modeling.

Evaluates passage selectors (S0..S4, Oracle, U0..U4) on query-grouped ranking:
- Top-1 Oracle Passage Accuracy
- Near-Oracle Selection Rate (Delta* - Delta_chosen <= 0.01)
- Mean & Median Selector Regret
- Downstream F1 with Oracle Gate
- Selector Headroom Recovery (SHR)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.week8.data import assert_paper_disjoint, load_config
from src.week8.evaluation import (
    compute_calibration_metrics,
    compute_query_grouped_selector_metrics,
)
from src.week8.utility_dataset import (
    assert_no_feature_leakage,
    build_utility_dataset,
    get_non_leaking_feature_columns,
)
from src.week8.utility_models import (
    OracleSelector,
    S0_Rank5Selector,
    S1_BM25Selector,
    S2_SemanticSelector,
    S3_DraftSimilaritySelector,
    S4_MarginalInfoSelector,
    UtilityModelU0,
    train_u0_paper_grouped_oof,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="W8-C Passage Utility Models CLI Runner")
    parser.add_argument("--smoke", action="store_true", help="Run smoke verification on tiny subset")
    parser.add_argument("--split", default="internal_val", choices=["fit", "internal_val"], help="Dataset split to evaluate")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tag = "_smoke" if args.smoke else ""
    logger.info("Starting W8-C Runner (split=%s, smoke=%s, device=%s)", args.split, args.smoke, args.device)

    prom_path = REPO_ROOT / "results/week8/surfaces" / f"promotion_surface_{args.split}{tag}.jsonl"
    draft_path = REPO_ROOT / "results/week8/drafts" / f"draft_diagnostics_{args.split}{tag}.jsonl"

    if not prom_path.is_file():
        raise FileNotFoundError(
            f"Promotion surface for {args.split} not found at {prom_path}. "
            "Please generate canonical counterfactual surface first."
        )

    df_prom = pd.read_json(prom_path, lines=True)
    df_draft = pd.read_json(draft_path, lines=True) if draft_path.is_file() else None

    # 1. Build utility dataset with targets & regret
    df_util = build_utility_dataset(df_prom, df_draft)
    non_leaking_cols = get_non_leaking_feature_columns(df_util)
    assert_no_feature_leakage(non_leaking_cols)

    logger.info("Constructed %s utility dataset with %d rows (%d queries).", args.split, len(df_util), df_util["question_id"].nunique())
    logger.info("Identified %d verified non-leaking feature columns.", len(non_leaking_cols))

    # 2. Evaluate Heuristic Selectors
    metrics_by_model = {}
    selectors = {
        "S0_Rank5": S0_Rank5Selector(),
        "S1_BM25": S1_BM25Selector(),
        "S2_Semantic": S2_SemanticSelector(),
        "S3_DraftSim": S3_DraftSimilaritySelector(),
        "S4_MIG": S4_MarginalInfoSelector(),
        "Oracle": OracleSelector(),
    }

    for name, sel in selectors.items():
        m = compute_query_grouped_selector_metrics(df_util, selector_obj=sel)
        metrics_by_model[name] = m
        logger.info(
            "%-12s | Top-1: %.3f | Near-Oracle: %.3f | Regret: %.4f | F1: %.4f | SHR: %5.1f%%",
            name,
            m["top1_oracle_accuracy"],
            m["near_oracle_selection_rate"],
            m["mean_regret"],
            m["final_downstream_f1"],
            m["selector_headroom_recovery"] * 100,
        )

    # 3. Train Tabular U0 using paper-grouped OOF
    logger.info("Training U0 Tabular model using paper-grouped cross-validation...")
    oof_preds, final_u0 = train_u0_paper_grouped_oof(df_util, non_leaking_cols, n_splits=5)
    df_util["score_u0"] = oof_preds["utility_score"]

    m_u0 = compute_query_grouped_selector_metrics(df_util, selector_scores_col="score_u0")
    metrics_by_model["U0_Tabular_OOF"] = m_u0
    logger.info(
        "%-12s | Top-1: %.3f | Near-Oracle: %.3f | Regret: %.4f | F1: %.4f | SHR: %5.1f%%",
        "U0_Tabular_OOF",
        m_u0["top1_oracle_accuracy"],
        m_u0["near_oracle_selection_rate"],
        m_u0["mean_regret"],
        m_u0["final_downstream_f1"],
        m_u0["selector_headroom_recovery"] * 100,
    )

    # 4. Save results table
    out_tables = REPO_ROOT / "results/week8/tables"
    out_tables.mkdir(parents=True, exist_ok=True)
    summary_rows = [
        {
            "Model": name,
            "Top-1_Accuracy": m["top1_oracle_accuracy"],
            "Near-Oracle_Rate": m["near_oracle_selection_rate"],
            "HELP_Precision": m["help_precision"],
            "HURT_Rate": m["hurt_rate"],
            "Mean_Regret": m["mean_regret"],
            "Median_Regret": m["median_regret"],
            "Downstream_F1": m["final_downstream_f1"],
            "SHR": m["selector_headroom_recovery"],
        }
        for name, m in metrics_by_model.items()
    ]
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(out_tables / f"w8c_selector_comparison_{args.split}{tag}.csv", index=False)
    logger.info("W8-C runner complete! Summary saved to %s", out_tables / f"w8c_selector_comparison_{args.split}{tag}.csv")


if __name__ == "__main__":
    main()
