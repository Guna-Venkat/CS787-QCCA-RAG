"""Evaluation metrics, conditional error breakdown, and paper-clustered bootstrap for Week 7-C.

Implements:
- Recovered Semantic Oracle Headroom (ROH_sem)
- Detailed under/over-allocation breakdown and regret analysis
- 2,000-replicate paper-clustered bootstrap comparisons on DEV
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)


def compute_recovered_oracle_headroom(
    policy_f1: float,
    reference_f1: float = 0.4010,  # Best static semantic k=8
    oracle_f1: float = 0.4869,     # Semantic epsilon-oracle
) -> float:
    """Compute Recovered Oracle Headroom (ROH_sem) under semantic routing.

    Formula:
        ROH_sem = (F1_policy - F1_ref) / (F1_oracle - F1_ref)
    """
    denom = oracle_f1 - reference_f1
    if abs(denom) < 1e-12:
        return 0.0
    return float((policy_f1 - reference_f1) / denom)


def compute_under_overallocation_breakdown(
    df_eval_queries: pd.DataFrame,
    budgets: Sequence[int] = (2, 4, 5, 6, 8),
    epsilon: float = 0.01,
) -> pd.DataFrame:
    """Analyze under-allocation, admissible allocation, and over-allocation.

    Args:
        df_eval_queries: DataFrame with columns:
            question_id, paper_id, pred_k, oracle_k, actual_f1, max_f1, actual_tokens, oracle_tokens

    Returns:
        Summary DataFrame with counts, percentages, and impact metrics.
    """
    total_q = max(1, len(df_eval_queries))
    rows: List[Dict[str, Any]] = []

    for cat_name, mask in [
        ("Under-allocation (pred_k < oracle_k)", df_eval_queries["pred_k"] < df_eval_queries["oracle_k"]),
        ("Admissible / Optimal (pred_k == oracle_k)", df_eval_queries["pred_k"] == df_eval_queries["oracle_k"]),
        ("Over-allocation (pred_k > oracle_k)", df_eval_queries["pred_k"] > df_eval_queries["oracle_k"]),
    ]:
        sub_df = df_eval_queries[mask]
        count = len(sub_df)
        pct = float(count / total_q) * 100.0

        if count > 0:
            mean_f1 = float(sub_df["actual_f1"].mean())
            mean_tok = float(sub_df["actual_tokens"].mean())
            f1_loss = float((sub_df["max_f1"] - sub_df["actual_f1"]).mean())
            token_delta = float((sub_df["actual_tokens"] - sub_df["oracle_tokens"]).mean())
        else:
            mean_f1 = 0.0
            mean_tok = 0.0
            f1_loss = 0.0
            token_delta = 0.0

        rows.append({
            "category": cat_name,
            "count": count,
            "pct_queries": pct,
            "mean_f1": mean_f1,
            "mean_tokens": mean_tok,
            "mean_f1_regret": f1_loss,
            "mean_token_delta_vs_oracle": token_delta,
        })

    return pd.DataFrame(rows)


def run_paper_clustered_bootstrap(
    paper_ids: Sequence[str],
    metric_a: Sequence[float],
    metric_b: Sequence[float],
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Paper-clustered bootstrap hypothesis test for paired differences (A - B).

    Resamples papers with replacement, pooling all queries belonging to sampled papers.
    """
    np.random.seed(seed)
    df_pairs = pd.DataFrame({
        "paper_id": [str(p).strip() for p in paper_ids],
        "a": np.array(metric_a, dtype=np.float64),
        "b": np.array(metric_b, dtype=np.float64),
    })

    unique_papers = df_pairs["paper_id"].unique()
    n_papers = len(unique_papers)

    # Group by paper for fast cluster sampling
    paper_groups = {pid: grp[["a", "b"]].values for pid, grp in df_pairs.groupby("paper_id")}

    diffs: List[float] = []
    for _ in range(n_resamples):
        sampled_papers = np.random.choice(unique_papers, size=n_papers, replace=True)
        sampled_vals = np.vstack([paper_groups[pid] for pid in sampled_papers])
        diff = np.mean(sampled_vals[:, 0]) - np.mean(sampled_vals[:, 1])
        diffs.append(float(diff))

    diffs_arr = np.array(diffs)
    mean_diff = float(np.mean(metric_a) - np.mean(metric_b))
    alpha = (1.0 - confidence_level) / 2.0
    ci_lower = float(np.percentile(diffs_arr, alpha * 100))
    ci_upper = float(np.percentile(diffs_arr, (1.0 - alpha) * 100))

    # Two-sided empirical p-value
    if mean_diff >= 0:
        p_val = 2.0 * float(np.mean(diffs_arr <= 0))
    else:
        p_val = 2.0 * float(np.mean(diffs_arr >= 0))
    p_val = min(1.0, max(1e-5, p_val))

    return {
        "mean_diff": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_val,
        "std_err": float(np.std(diffs_arr)),
    }
