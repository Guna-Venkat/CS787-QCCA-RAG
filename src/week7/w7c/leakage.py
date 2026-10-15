"""Leakage audit and dataset sanitization utilities for Week 7-C.

Ensures no target, downstream evaluation metric, or post-generation oracle labels
leak into model input features X.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FORBIDDEN_SUBSTRINGS: Set[str] = {
    "f1",
    "oracle",
    "regret",
    "answer",
    "gold",
    "label",
    "target",
}

WHITELISTED_METADATA_COLS: Set[str] = {
    "split",
    "question_id",
    "paper_id",
    "k",
    "target_f1",
    "target_tokens",
    "target_max_f1",
    "target_regret",
    "target_log_regret",
    "target_sufficient",
    "target_oracle_k",
}


def audit_feature_leakage(feature_names: Sequence[str]) -> Tuple[bool, List[str]]:
    """Audit a list of candidate model input feature names for leakage.

    Flags any feature whose name contains forbidden substrings:
    f1, oracle, regret, answer, gold, label, target.

    Args:
        feature_names: Sequence of feature column names.

    Returns:
        (is_clean, violating_names)
    """
    violating_names: List[str] = []
    for feat in feature_names:
        feat_lower = str(feat).lower()
        for forbidden in FORBIDDEN_SUBSTRINGS:
            if forbidden in feat_lower:
                violating_names.append(feat)
                break

    is_clean = len(violating_names) == 0
    return is_clean, violating_names


def assert_zero_feature_leakage(feature_names: Sequence[str]) -> None:
    """Raise ValueError if any feature name violates the leakage policy."""
    is_clean, violations = audit_feature_leakage(feature_names)
    if not is_clean:
        raise ValueError(
            f"DATA LEAKAGE DETECTED in feature names! {len(violations)} features contain forbidden tokens: {violations}"
        )
    logger.info("PASSED: Zero feature leakage verified for %d features.", len(feature_names))


def sanitize_feature_matrix(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """Extract and validate the feature matrix X for modeling.

    Args:
        df: Input DataFrame containing features and metadata.
        feature_cols: Sequence of desired feature names.

    Returns:
        Clean DataFrame containing only validated feature columns, guaranteed free of target leakage.
    """
    assert_zero_feature_leakage(feature_cols)
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Requested features missing from DataFrame: {missing_cols}")

    X = df[list(feature_cols)].copy()
    # Check for NaN / Inf
    if X.isna().any().any():
        nan_cols = X.columns[X.isna().any()].tolist()
        logger.warning("NaN values detected in features: %s. Filling with 0.0.", nan_cols)
        X = X.fillna(0.0)

    # Replace Inf with finite limits
    X = X.replace([np.inf, -np.inf], 0.0)
    return X
