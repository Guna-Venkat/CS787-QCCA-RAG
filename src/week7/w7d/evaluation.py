"""Evaluation metrics, extended error taxonomy (E1-E6), safety/precision metrics for Week 7-D.

Provides:
- 4-way Error Taxonomy (E1-E4) + Anchor Diagnostics (E5, E6).
- Downshift Safety Metrics (precision of k=2 admissibility, regret, tokens saved).
- Upshift Precision Metrics (fraction of justified escalations, gain per added token).
- Recovered Semantic Oracle Headroom (ROH_sem).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_w7d_error_taxonomy(
    df_eval_queries: pd.DataFrame,
    budgets: Sequence[int] = (2, 4, 5, 6, 8),
    anchor_k: int = 4,
    epsilon: float = 0.01,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute error taxonomy reflecting non-monotonicity and anchor retention.

    Taxonomy:
    - E1: Minimum-cost oracle exactly
    - E2: Epsilon-admissible but over-budget
    - E3: Under-budget & inadmissible
    - E4: Higher-budget & STILL inadmissible
    - E5: Anchor k=4 chosen & admissible
    - E6: Anchor k=4 chosen but inadmissible
    """
    total_q = max(1, len(df_eval_queries))
    assigned_cats: List[str] = []
    regrets: List[float] = []
    token_penalties: List[float] = []
    e5_flags: List[int] = []
    e6_flags: List[int] = []

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

        assigned_cats.append(cat)
        e5_flags.append(int(pred_k == anchor_k and anchor_k in adm_ks))
        e6_flags.append(int(pred_k == anchor_k and anchor_k not in adm_ks))

    df_detailed = df_eval_queries.copy()
    df_detailed["error_category"] = assigned_cats
    df_detailed["f1_regret"] = regrets
    df_detailed["token_penalty_vs_oracle"] = token_penalties
    df_detailed["is_e5_anchor_admissible"] = e5_flags
    df_detailed["is_e6_anchor_inadmissible"] = e6_flags

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

    # Supplementary Anchor rows
    sub_e5 = df_detailed[df_detailed["is_e5_anchor_admissible"] == 1]
    c_e5 = len(sub_e5)
    summary_rows.append({
        "category": "E5: Anchor k=4 chosen & admissible",
        "count": c_e5,
        "pct_queries": float(c_e5 / total_q) * 100.0,
        "mean_f1_regret": float(sub_e5["f1_regret"].mean()) if c_e5 > 0 else 0.0,
        "mean_token_penalty": float(sub_e5["token_penalty_vs_oracle"].mean()) if c_e5 > 0 else 0.0,
    })

    sub_e6 = df_detailed[df_detailed["is_e6_anchor_inadmissible"] == 1]
    c_e6 = len(sub_e6)
    summary_rows.append({
        "category": "E6: Anchor k=4 chosen & inadmissible",
        "count": c_e6,
        "pct_queries": float(c_e6 / total_q) * 100.0,
        "mean_f1_regret": float(sub_e6["f1_regret"].mean()) if c_e6 > 0 else 0.0,
        "mean_token_penalty": float(sub_e6["token_penalty_vs_oracle"].mean()) if c_e6 > 0 else 0.0,
    })

    df_summary = pd.DataFrame(summary_rows)
    return df_detailed, df_summary


def compute_downshift_safety_metrics(
    df_eval_queries: pd.DataFrame,
    budgets: Sequence[int] = (2, 4, 5, 6, 8),
    anchor_k: int = 4,
    epsilon: float = 0.01,
) -> Dict[str, Any]:
    """Compute downshift safety metrics for queries where k=2 was chosen."""
    total_q = len(df_eval_queries)

    all_k2_admissible = 0
    for _, row in df_eval_queries.iterrows():
        f1s = [float(row[f"f1_k{k}"]) for k in budgets]
        if float(row["f1_k2"]) >= max(f1s) - epsilon:
            all_k2_admissible += 1

    df_down = df_eval_queries[df_eval_queries["pred_k"] == 2].copy()
    n_down = len(df_down)

    if n_down == 0:
        return {
            "n_downshifted": 0,
            "downshift_rate_pct": 0.0,
            "precision_k2_admissible": 0.0,
            "recall_k2_admissible": 0.0,
            "mean_f1_at_k2": 0.0,
            "mean_regret_vs_oracle": 0.0,
            "mean_regret_vs_k4": 0.0,
            "mean_tokens_saved_vs_k4": 0.0,
        }

    correct_down = 0
    f1_k2_list = []
    regret_oracle_list = []
    regret_k4_list = []
    tokens_saved_list = []

    for _, row in df_down.iterrows():
        f1s = [float(row[f"f1_k{k}"]) for k in budgets]
        max_f1 = max(f1s)
        f1_2 = float(row["f1_k2"])
        f1_4 = float(row[f"f1_k{anchor_k}"])
        tok_2 = float(row["tokens_k2"])
        tok_4 = float(row[f"tokens_k{anchor_k}"])

        if f1_2 >= max_f1 - epsilon:
            correct_down += 1

        f1_k2_list.append(f1_2)
        regret_oracle_list.append(max(0.0, max_f1 - f1_2))
        regret_k4_list.append(f1_4 - f1_2)
        tokens_saved_list.append(tok_4 - tok_2)

    precision = correct_down / n_down
    recall = correct_down / all_k2_admissible if all_k2_admissible > 0 else 0.0

    return {
        "n_downshifted": n_down,
        "downshift_rate_pct": float(n_down / total_q) * 100.0,
        "precision_k2_admissible": float(precision),
        "recall_k2_admissible": float(recall),
        "mean_f1_at_k2": float(np.mean(f1_k2_list)),
        "mean_regret_vs_oracle": float(np.mean(regret_oracle_list)),
        "mean_regret_vs_k4": float(np.mean(regret_k4_list)),
        "mean_tokens_saved_vs_k4": float(np.mean(tokens_saved_list)),
    }


def compute_upshift_precision_metrics(
    df_eval_queries: pd.DataFrame,
    anchor_k: int = 4,
    epsilon: float = 0.01,
) -> Dict[str, Any]:
    """Compute upshift precision metrics for queries where k > 4 was chosen."""
    total_q = len(df_eval_queries)
    df_up = df_eval_queries[df_eval_queries["pred_k"] > anchor_k].copy()
    n_up = len(df_up)

    if n_up == 0:
        return {
            "n_upshifted": 0,
            "upshift_rate_pct": 0.0,
            "fraction_gain_gt_epsilon": 0.0,
            "mean_actual_gain_vs_k4": 0.0,
            "mean_tokens_added_vs_k4": 0.0,
            "gain_per_100_added_tokens": 0.0,
            "chosen_k_distribution": {},
        }

    justified_count = 0
    gain_list = []
    added_tokens_list = []
    k_counts: Dict[int, int] = {5: 0, 6: 0, 8: 0}

    for _, row in df_up.iterrows():
        pred_k = int(row["pred_k"])
        k_counts[pred_k] = k_counts.get(pred_k, 0) + 1

        f1_pred = float(row[f"f1_k{pred_k}"])
        f1_4 = float(row[f"f1_k{anchor_k}"])
        tok_pred = float(row[f"tokens_k{pred_k}"])
        tok_4 = float(row[f"tokens_k{anchor_k}"])

        gain = f1_pred - f1_4
        added_tok = tok_pred - tok_4

        if gain > epsilon:
            justified_count += 1

        gain_list.append(gain)
        added_tokens_list.append(added_tok)

    mean_gain = float(np.mean(gain_list))
    mean_added_tok = float(np.mean(added_tokens_list))
    gain_per_100 = (mean_gain / (mean_added_tok / 100.0)) if mean_added_tok > 0 else 0.0
    dist_pct = {k: float(v / n_up) * 100.0 for k, v in k_counts.items()}

    return {
        "n_upshifted": n_up,
        "upshift_rate_pct": float(n_up / total_q) * 100.0,
        "fraction_gain_gt_epsilon": float(justified_count / n_up),
        "mean_actual_gain_vs_k4": mean_gain,
        "mean_tokens_added_vs_k4": mean_added_tok,
        "gain_per_100_added_tokens": float(gain_per_100),
        "chosen_k_distribution": dist_pct,
    }


def compute_recovered_oracle_headroom(
    policy_f1: float,
    static_k8_f1: float,
    oracle_f1: float,
) -> float:
    """Compute Recovered Semantic Oracle Headroom (ROH_sem)."""
    denom = oracle_f1 - static_k8_f1
    if abs(denom) < 1e-6:
        return 0.0
    return float((policy_f1 - static_k8_f1) / denom)
