"""Feature ablation and leave-one-feature-out experiments for Week 4."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.week4.cross_validation import run_grouped_cross_validation
from src.week4.evaluation import FrozenResponseSurfaceLookup, evaluate_allocator_predictions

logger = logging.getLogger(__name__)


def run_feature_set_ablation(
    df_data: pd.DataFrame,
    feature_sets: Dict[str, List[str]],
    lookup: FrozenResponseSurfaceLookup,
    target_col: str = "target_k",
    group_col: str = "paper_id",
    n_splits: int = 5,
    random_state: int = 42,
    logreg_C: float = 1.0
) -> pd.DataFrame:
    """Compare performance across declared feature subsets (e.g. Concentration-only vs Full)."""
    ablation_records = []
    
    for set_name, f_cols in feature_sets.items():
        logger.info("Running ablation for feature set '%s': %s", set_name, f_cols)
        df_oof, _ = run_grouped_cross_validation(
            df_data=df_data,
            feature_cols=f_cols,
            target_col=target_col,
            group_col=group_col,
            n_splits=n_splits,
            random_state=random_state,
            logreg_C=logreg_C
        )
        
        # Evaluate primary model (Logistic Regression)
        metrics = evaluate_allocator_predictions(
            df_preds=df_oof,
            pred_col="pred_k_logreg",
            lookup=lookup,
            target_col=target_col
        )
        
        ablation_records.append({
            "feature_set": set_name,
            "feature_count": len(f_cols),
            "features": ", ".join(f_cols),
            "mean_f1": metrics["mean_f1"],
            "mean_k": metrics["mean_k"],
            "mean_context_tokens": metrics["mean_context_tokens"],
            "context_reduction_pct_vs_k8": metrics["context_reduction_pct_vs_k8"],
            "target_accuracy": metrics["target_accuracy"],
            "mean_epsilon_regret": metrics["mean_epsilon_regret"],
            "oracle_headroom_recovered_pct": metrics["oracle_headroom_recovered_pct"]
        })
        
    return pd.DataFrame(ablation_records)


def run_leave_one_feature_out(
    df_data: pd.DataFrame,
    full_feature_cols: List[str],
    lookup: FrozenResponseSurfaceLookup,
    target_col: str = "target_k",
    group_col: str = "paper_id",
    n_splits: int = 5,
    random_state: int = 42,
    logreg_C: float = 1.0
) -> pd.DataFrame:
    """Perform leave-one-feature-out sensitivity analysis on primary Logistic Regression model."""
    # First, evaluate baseline with full feature set
    df_oof_full, _ = run_grouped_cross_validation(
        df_data=df_data,
        feature_cols=full_feature_cols,
        target_col=target_col,
        group_col=group_col,
        n_splits=n_splits,
        random_state=random_state,
        logreg_C=logreg_C
    )
    base_metrics = evaluate_allocator_predictions(
        df_preds=df_oof_full,
        pred_col="pred_k_logreg",
        lookup=lookup,
        target_col=target_col
    )
    
    lofo_records = []
    
    for feat in full_feature_cols:
        sub_features = [f for f in full_feature_cols if f != feat]
        logger.info("Leave-one-out: Removing '%s'", feat)
        
        df_oof_sub, _ = run_grouped_cross_validation(
            df_data=df_data,
            feature_cols=sub_features,
            target_col=target_col,
            group_col=group_col,
            n_splits=n_splits,
            random_state=random_state,
            logreg_C=logreg_C
        )
        
        sub_metrics = evaluate_allocator_predictions(
            df_preds=df_oof_sub,
            pred_col="pred_k_logreg",
            lookup=lookup,
            target_col=target_col
        )
        
        delta_f1 = sub_metrics["mean_f1"] - base_metrics["mean_f1"]
        delta_k = sub_metrics["mean_k"] - base_metrics["mean_k"]
        delta_tokens = sub_metrics["mean_context_tokens"] - base_metrics["mean_context_tokens"]
        delta_acc = sub_metrics["target_accuracy"] - base_metrics["target_accuracy"]
        
        lofo_records.append({
            "removed_feature": feat,
            "remaining_features_count": len(sub_features),
            "mean_f1": sub_metrics["mean_f1"],
            "mean_k": sub_metrics["mean_k"],
            "context_tokens": sub_metrics["mean_context_tokens"],
            "target_accuracy": sub_metrics["target_accuracy"],
            "delta_f1": round(delta_f1, 4),
            "delta_mean_k": round(delta_k, 2),
            "delta_context_tokens": round(delta_tokens, 1),
            "delta_target_accuracy": round(delta_acc, 4)
        })
        
    return pd.DataFrame(lofo_records)
