"""Evaluation metrics, correct error taxonomy (E1-E4), and bootstrap for Week 7-C2.

Implements:
- Correct 4-way Error Taxonomy (E1, E2, E3, E4) reflecting non-monotonicity.
- Recovered Semantic Oracle Headroom (ROH_sem).
- Paper-clustered bootstrap comparisons (2,000 replicates).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_corrected_error_taxonomy(
    df_eval_queries: pd.DataFrame,
    budgets: Sequence[int] = (2, 4, 5, 6, 8),
    epsilon: float = 0.01,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the corrected 4-way error taxonomy accounting for non-monotonicity.

    Taxonomy:
    - E1: Minimum-cost oracle exactly (pred_k == oracle_k)
    - E2: Epsilon-admissible but over-budget (pred_k in A_eps and pred_k > oracle_k)
    - E3: Under-budget and inadmissible (pred_k < oracle_k and pred_k not in A_eps)
    - E4: Higher-budget but STILL inadmissible (pred_k > oracle_k and pred_k not in A_eps)

    Args:
        df_eval_queries: DataFrame containing per-query evaluation details.
            Columns: question_id, paper_id, pred_k, f1_k2, f1_k4, f1_k5, f1_k6, f1_k8,
                     tokens_k2, tokens_k4, tokens_k5, tokens_k6, tokens_k8
    """
    total_q = max(1, len(df_eval_queries))
    assigned_categories: List[str] = []
    regrets: List[float] = []
    token_penalties: List[float] = []

    for _, row in df_eval_queries.iterrows():
        pred_k = int(row["pred_k"])
        f1_vals = {k: float(row[f"f1_k{k}"]) for k in budgets}
        tok_vals = {k: float(row[f"tokens_k{k}"]) for k in budgets}

        max_f1 = max(f1_vals.values())
        adm_ks = set(k for k in budgets if f1_vals[k] >= max_f1 - epsilon)
        min_oracle_k = min(adm_ks) if adm_ks else max(budgets)

        pred_f1 = f1_vals[pred_k]
        pred_tokens = tok_vals[pred_k]
        oracle_tokens = tok_vals[min_oracle_k]

        f1_regret = max(0.0, max_f1 - pred_f1)
        token_penalty = pred_tokens - oracle_tokens

        regrets.append(f1_regret)
        token_penalties.append(token_penalty)

        if pred_k == min_oracle_k:
            cat = "E1: Minimum-cost oracle exactly"
        elif pred_k in adm_ks and pred_k > min_oracle_k:
            cat = "E2: Admissible but over-budget"
        elif pred_k not in adm_ks and pred_k < min_oracle_k:
            cat = "E3: Under-budget & inadmissible"
        else:
            cat = "E4: Higher-budget & STILL inadmissible"

        assigned_categories.append(cat)

    df_detailed = df_eval_queries.copy()
    df_detailed["error_category"] = assigned_categories
    df_detailed["f1_regret"] = regrets
    df_detailed["token_penalty_vs_oracle"] = token_penalties

    summary_rows = []
    for cat_name in [
        "E1: Minimum-cost oracle exactly",
        "E2: Admissible but over-budget",
        "E3: Under-budget & inadmissible",
        "E4: Higher-budget & STILL inadmissible",
    ]:
        sub = df_detailed[df_detailed["error_category"] == cat_name]
        c = len(sub)
        summary_rows.append({
            "category": cat_name,
            "count": c,
            "pct_queries": float(c / total_q) * 100.0,
            "mean_f1_regret": float(sub["f1_regret"].mean()) if c > 0 else 0.0,
            "mean_token_penalty": float(sub["token_penalty_vs_oracle"].mean()) if c > 0 else 0.0,
        })

    df_summary = pd.DataFrame(summary_rows)
    return df_detailed, df_summary


def compute_recovered_oracle_headroom(
    policy_f1: float,
    reference_f1: float = 0.4010,  # Best static semantic k=8
    oracle_f1: float = 0.4869,     # Semantic epsilon-oracle
) -> float:
    """Compute Recovered Oracle Headroom (ROH_sem) under semantic routing."""
    denom = oracle_f1 - reference_f1
    if abs(denom) < 1e-12:
        return 0.0
    return float((policy_f1 - reference_f1) / denom)


def run_paper_clustered_bootstrap(
    paper_ids: Sequence[str],
    metric_a: Sequence[float],
    metric_b: Sequence[float],
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Paper-clustered bootstrap hypothesis test for paired differences (A - B)."""
    np.random.seed(seed)
    df_pairs = pd.DataFrame({
        "paper_id": [str(p).strip() for p in paper_ids],
        "a": np.array(metric_a, dtype=np.float64),
        "b": np.array(metric_b, dtype=np.float64),
    })

    unique_papers = df_pairs["paper_id"].unique()
    n_papers = len(unique_papers)
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
