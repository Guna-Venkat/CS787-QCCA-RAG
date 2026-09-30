"""Feature extraction, validation, and preparation module for Week 4 QCCA."""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FORBIDDEN_LEAKAGE_KEYWORDS = [
    "f1", "em", "exact_match", "rouge", "gold", "answer",
    "generated", "logits", "human_score", "oracle", "regret",
    "best_k", "selected_k", "target", "label", "loss"
]

APPROVED_FEATURE_WHITELIST = [
    "rho_1",
    "delta_12",
    "entropy",
    "entropy_normalized",
    "query_token_length",
    "n_high"
]

METADATA_COLUMNS = ["question_id", "paper_id"]


def audit_for_leakage(df: pd.DataFrame, allowed_cols: Optional[List[str]] = None) -> None:
    """Ensure no forbidden post-generation, oracle, or ground truth columns exist."""
    cols_to_check = [c for c in df.columns if c not in METADATA_COLUMNS]
    if allowed_cols is not None:
        cols_to_check = [c for c in cols_to_check if c not in allowed_cols]
        
    for col in cols_to_check:
        col_lower = col.lower()
        for kw in FORBIDDEN_LEAKAGE_KEYWORDS:
            if kw in col_lower:
                raise ValueError(
                    f"Data Leakage Detected: Column '{col}' matches forbidden keyword '{kw}'. "
                    f"Allocator input features must be strictly pre-generation retrieval features."
                )


def load_retrieval_features(
    features_path: str = "results/week2/processed/dev_retrieval_features.jsonl",
    smoke_test: bool = False,
    smoke_limit: int = 3
) -> pd.DataFrame:
    """Load and validate the frozen pre-generation retrieval features from Week 2."""
    p = Path(features_path)
    if not p.exists():
        raise FileNotFoundError(f"Feature artifact not found: {p}")
        
    df = pd.read_json(p, lines=True)
    
    # Verify metadata columns exist
    for meta_col in METADATA_COLUMNS:
        if meta_col not in df.columns:
            raise ValueError(f"Missing required metadata column: {meta_col}")
            
    # Check for duplicate questions
    dup_count = df["question_id"].duplicated().sum()
    if dup_count > 0:
        raise ValueError(f"Found {dup_count} duplicate question_id rows in features")
        
    # Audit for leakage
    audit_for_leakage(df)
    
    # Verify approved feature whitelist
    feature_cols = [c for c in df.columns if c not in METADATA_COLUMNS]
    for col in feature_cols:
        if col not in APPROVED_FEATURE_WHITELIST:
            logger.warning("Feature '%s' is not in standard APPROVED_FEATURE_WHITELIST", col)
            
    # Handle NaN / Inf deterministically
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            logger.warning("Filling %d NaNs in feature '%s' with median %f", df[col].isna().sum(), col, median_val)
            df[col] = df[col].fillna(median_val)
        if np.isinf(df[col]).any():
            finite_max = df[col][np.isfinite(df[col])].max()
            logger.warning("Replacing Inf in feature '%s' with finite max %f", col, finite_max)
            df[col] = df[col].replace([np.inf, -np.inf], finite_max)

    if smoke_test:
        logger.info("SMOKE TEST MODE ENABLED: Subsetting features to %d rows", smoke_limit)
        df = df.iloc[:smoke_limit].copy()
        
    return df
