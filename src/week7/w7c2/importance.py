"""Grouped permutation and family-level feature importance for Week 7-C2.

Avoids split-count bias by evaluating grouped permutation importance on INTERNAL-VAL.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def assign_feature_family(feat_name: str) -> str:
    """Assign feature to one of the canonical W7 feature families."""
    if feat_name.startswith("q_"):
        return "Family A (Query Complexity)"
    elif feat_name.startswith("sem_"):
        return "Family B (Semantic Geometry)"
    elif feat_name.startswith("bm25_"):
        return "Family C (BM25 & QPP)"
    elif feat_name.startswith("cross_"):
        return "Family D (Cross-Retriever)"
    elif feat_name.startswith("div_"):
        return "Family E (Set Diversity)"
    elif feat_name.startswith("tail_"):
        return "Family F (Tail Threat)"
    elif feat_name.startswith("cov_"):
        return "Family G (Query Coverage)"
    elif feat_name.startswith("cost_"):
        return "Family H (Length & Cost)"
    elif feat_name.startswith("w4_"):
        return "Family F0 (W4 Baseline)"
    elif feat_name.startswith("ce_"):
        return "Family I (Cross-Encoder)"
    else:
        return "Other"


def compute_grouped_permutation_importance(
    model: Any,
    X_val: pd.DataFrame,
    df_val_cand: pd.DataFrame,
    df_val_matrix: pd.DataFrame,
    policy_eval_fn: Callable[[np.ndarray], float],
    n_repeats: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute family-level grouped permutation importance on INTERNAL-VAL.

    For each feature family, shuffles all member features simultaneously across queries
    and measures the resulting degradation in downstream answer F1.
    """
    np.random.seed(seed)

    # Baseline performance without permutation
    base_preds = model.predict(X_val)
    base_metric = policy_eval_fn(base_preds)

    # Group features by family
    family_to_feats: Dict[str, List[str]] = {}
    for col in X_val.columns:
        fam = assign_feature_family(col)
        if fam not in family_to_feats:
            family_to_feats[fam] = []
        family_to_feats[fam].append(col)

    results: List[Dict[str, Any]] = []

    for fam, cols in family_to_feats.items():
        drops: List[float] = []
        for _ in range(n_repeats):
            X_perm = X_val.copy()
            # Permute rows of these columns jointly to preserve intra-family correlations
            perm_indices = np.random.permutation(len(X_val))
            X_perm[cols] = X_val[cols].iloc[perm_indices].values

            perm_preds = model.predict(X_perm)
            perm_metric = policy_eval_fn(perm_preds)
            drop = base_metric - perm_metric
            drops.append(float(drop))

        results.append({
            "feature_family": fam,
            "n_features": len(cols),
            "mean_f1_drop": float(np.mean(drops)),
            "std_f1_drop": float(np.std(drops)),
        })

    df_imp = pd.DataFrame(results).sort_values(by="mean_f1_drop", ascending=False).reset_index(drop=True)
    return df_imp
