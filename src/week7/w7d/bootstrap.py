"""Paper-clustered bootstrap comparisons for Week 7-D.

Performs 2,000 cluster-bootstrap resamples at paper level to test statistical
significance against static semantic baselines and historical allocators.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_paper_clustered_bootstrap(
    df_eval: pd.DataFrame,
    pred_col_policy: str,
    baseline_pred_col: str,
    budgets: Sequence[int] = (2, 4, 5, 6, 8),
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Run paper-clustered bootstrap comparing policy vs baseline."""
    rng = np.random.RandomState(random_seed)
    unique_papers = np.array(df_eval["paper_id"].unique())
    n_papers = len(unique_papers)

    paper_to_indices = {p: df_eval.index[df_eval["paper_id"] == p].to_numpy() for p in unique_papers}

    delta_f1s = np.zeros(n_resamples, dtype=float)
    delta_tokens = np.zeros(n_resamples, dtype=float)

    for b in range(n_resamples):
        sampled_papers = rng.choice(unique_papers, size=n_papers, replace=True)
        sampled_indices = np.concatenate([paper_to_indices[p] for p in sampled_papers])
        df_boot = df_eval.iloc[sampled_indices]

        # Policy F1 & Tokens
        f1_pol = [float(df_boot.iloc[i][f"f1_k{int(df_boot.iloc[i][pred_col_policy])}"]) for i in range(len(df_boot))]
        tok_pol = [float(df_boot.iloc[i][f"tokens_k{int(df_boot.iloc[i][pred_col_policy])}"]) for i in range(len(df_boot))]

        # Baseline F1 & Tokens
        f1_base = [float(df_boot.iloc[i][f"f1_k{int(df_boot.iloc[i][baseline_pred_col])}"]) for i in range(len(df_boot))]
        tok_base = [float(df_boot.iloc[i][f"tokens_k{int(df_boot.iloc[i][baseline_pred_col])}"]) for i in range(len(df_boot))]

        delta_f1s[b] = np.mean(f1_pol) - np.mean(f1_base)
        delta_tokens[b] = np.mean(tok_pol) - np.mean(tok_base)

    alpha = 1.0 - confidence_level
    f1_ci_lower = float(np.percentile(delta_f1s, 100.0 * (alpha / 2.0)))
    f1_ci_upper = float(np.percentile(delta_f1s, 100.0 * (1.0 - alpha / 2.0)))

    tok_ci_lower = float(np.percentile(delta_tokens, 100.0 * (alpha / 2.0)))
    tok_ci_upper = float(np.percentile(delta_tokens, 100.0 * (1.0 - alpha / 2.0)))

    mean_delta_f1 = float(np.mean(delta_f1s))
    if mean_delta_f1 >= 0:
        p_val_f1 = float(2.0 * np.mean(delta_f1s <= 0))
    else:
        p_val_f1 = float(2.0 * np.mean(delta_f1s >= 0))
    p_val_f1 = min(1.0, max(0.0, p_val_f1))

    return {
        "mean_delta_f1": mean_delta_f1,
        "f1_ci_lower": f1_ci_lower,
        "f1_ci_upper": f1_ci_upper,
        "p_value_f1": p_val_f1,
        "mean_delta_tokens": float(np.mean(delta_tokens)),
        "token_ci_lower": tok_ci_lower,
        "token_ci_upper": tok_ci_upper,
    }
