"""Target loading and training dataset construction for Week 4 QCCA."""

import json
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

K_ALLOC_FROZEN = [2, 4, 5, 6, 8]
EPSILON_STAR_FROZEN = 0.01


def load_epsilon_targets(
    targets_path: str = "results/week3/processed/epsilon_oracle_all.csv",
    epsilon_star: float = EPSILON_STAR_FROZEN,
    k_alloc: Optional[List[int]] = None
) -> pd.DataFrame:
    """Load the frozen Week-3 epsilon-oracle targets filtered to epsilon_star."""
    if k_alloc is None:
        k_alloc = K_ALLOC_FROZEN
        
    p = Path(targets_path)
    if not p.exists():
        raise FileNotFoundError(f"Target artifact not found: {p}")
        
    df = pd.read_csv(p)
    if "epsilon" not in df.columns or "selected_k_epsilon" not in df.columns:
        raise ValueError(f"Target file missing expected columns: {df.columns.tolist()}")
        
    # Filter to epsilon_star
    df_eps = df[df["epsilon"] == epsilon_star].copy()
    if len(df_eps) == 0:
        raise ValueError(f"No rows found in target file for epsilon = {epsilon_star}")
        
    # Check for duplicate question_ids
    if df_eps["question_id"].duplicated().any():
        raise ValueError(f"Found duplicate question_id rows for epsilon = {epsilon_star}")
        
    # Check that all targets belong to k_alloc
    invalid_targets = set(df_eps["selected_k_epsilon"]) - set(k_alloc)
    if invalid_targets:
        raise ValueError(f"Targets contain actions outside K_alloc: {invalid_targets}")
        
    logger.info("Loaded %d target records at epsilon* = %s", len(df_eps), epsilon_star)
    return df_eps


def prepare_training_dataset(
    df_features: pd.DataFrame,
    df_targets: pd.DataFrame
) -> pd.DataFrame:
    """Inner join features and targets by question_id and verify data integrity."""
    target_cols = ["question_id", "paper_id", "selected_k_epsilon"]
    for opt_col in ["selected_f1", "reference_f1_alloc", "f1_regret"]:
        if opt_col in df_targets.columns:
            target_cols.append(opt_col)
            
    merged = pd.merge(
        df_features,
        df_targets[target_cols],
        on="question_id",
        suffixes=("_feat", "_targ")
    )
    
    if len(merged) != len(df_features):
        logger.warning(
            "Feature/target count mismatch: features=%d, targets=%d, merged=%d",
            len(df_features), len(df_targets), len(merged)
        )
        
    # Verify paper_id alignment
    if "paper_id_feat" in merged.columns and "paper_id_targ" in merged.columns:
        if not (merged["paper_id_feat"] == merged["paper_id_targ"]).all():
            raise ValueError("paper_id mismatch between features and targets!")
        merged["paper_id"] = merged["paper_id_feat"]
        merged.drop(columns=["paper_id_feat", "paper_id_targ"], inplace=True)
        
    # Rename target column cleanly
    merged["target_k"] = merged["selected_k_epsilon"].astype(int)
    return merged
