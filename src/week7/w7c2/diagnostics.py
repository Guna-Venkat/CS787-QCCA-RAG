"""Oracle stability, margin, learnability, and non-monotonicity diagnostics for Week 7-C2.

Implements:
- D1: Best-vs-second-best margin distribution
- D2: Epsilon-admissible set size distribution |A_0.01(q)|
- D3: Monotonicity analysis and step reversal counts
- D4: Adjacent marginal gain distributions
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_best_vs_second_margin(
    response_matrix_df: pd.DataFrame,
    budgets: Sequence[int] = (2, 4, 5, 6, 8),
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Compute D1: Best-vs-second-best margin M(q) = F_best(q) - F_second(q)."""
    margins: List[float] = []

    for _, row in response_matrix_df.iterrows():
        f1_vals = sorted([float(row[f"f1_k{k}"]) for k in budgets], reverse=True)
        m = f1_vals[0] - f1_vals[1]
        margins.append(float(max(0.0, m)))

    arr = np.array(margins, dtype=np.float64)
    summary = {
        "mean_margin": float(np.mean(arr)),
        "median_margin": float(np.median(arr)),
        "p25_margin": float(np.percentile(arr, 25)),
        "p75_margin": float(np.percentile(arr, 75)),
        "frac_below_001": float(np.mean(arr < 0.01)),
        "frac_below_002": float(np.mean(arr < 0.02)),
        "frac_below_005": float(np.mean(arr < 0.05)),
    }
    return arr, summary


def compute_admissible_set_distribution(
    response_matrix_df: pd.DataFrame,
    budgets: Sequence[int] = (2, 4, 5, 6, 8),
    epsilon: float = 0.01,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Compute D2: Epsilon-admissible set size distribution |A_epsilon(q)|."""
    sizes: List[int] = []

    for _, row in response_matrix_df.iterrows():
        f1_vals = [float(row[f"f1_k{k}"]) for k in budgets]
        max_f1 = max(f1_vals)
        adm_count = sum(1 for f in f1_vals if f >= max_f1 - epsilon)
        sizes.append(adm_count)

    arr = np.array(sizes, dtype=int)
    total_q = max(1, len(arr))

    rows = []
    for s_val in range(1, len(budgets) + 1):
        count = int(np.sum(arr == s_val))
        rows.append({
            "admissible_set_size": s_val,
            "query_count": count,
            "pct_queries": float(count / total_q) * 100.0,
        })
    df_dist = pd.DataFrame(rows)
    return arr, df_dist


def compute_monotonicity_analysis(
    response_matrix_df: pd.DataFrame,
    budgets: Sequence[int] = (2, 4, 5, 6, 8),
) -> Dict[str, Any]:
    """Compute D3: Monotonicity check (F2 <= F4 <= F5 <= F6 <= F8) and reversal counts."""
    total_queries = len(response_matrix_df)
    n_monotonic = 0
    reversals = {
        "F4_lt_F2": 0,
        "F5_lt_F4": 0,
        "F6_lt_F5": 0,
        "F8_lt_F6": 0,
    }

    for _, row in response_matrix_df.iterrows():
        f2 = float(row["f1_k2"])
        f4 = float(row["f1_k4"])
        f5 = float(row["f1_k5"])
        f6 = float(row["f1_k6"])
        f8 = float(row["f1_k8"])

        is_mono = (f2 <= f4 + 1e-6) and (f4 <= f5 + 1e-6) and (f5 <= f6 + 1e-6) and (f6 <= f8 + 1e-6)
        if is_mono:
            n_monotonic += 1

        if f4 < f2 - 1e-6:
            reversals["F4_lt_F2"] += 1
        if f5 < f4 - 1e-6:
            reversals["F5_lt_F4"] += 1
        if f6 < f5 - 1e-6:
            reversals["F6_lt_F5"] += 1
        if f8 < f6 - 1e-6:
            reversals["F8_lt_F6"] += 1

    pct_mono = float(n_monotonic / total_queries) * 100.0
    pct_non_mono = 100.0 - pct_mono

    return {
        "total_queries": total_queries,
        "monotonic_queries": n_monotonic,
        "monotonic_pct": pct_mono,
        "non_monotonic_queries": total_queries - n_monotonic,
        "non_monotonic_pct": pct_non_mono,
        "reversals": reversals,
    }


def compute_adjacent_marginal_gains(
    response_matrix_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute D4: Adjacent marginal quality gains between consecutive representation budgets."""
    steps = [
        ("Delta_2_to_4", "f1_k4", "f1_k2"),
        ("Delta_4_to_5", "f1_k5", "f1_k4"),
        ("Delta_5_to_6", "f1_k6", "f1_k5"),
        ("Delta_6_to_8", "f1_k8", "f1_k6"),
    ]

    rows = []
    for step_name, col_high, col_low in steps:
        gains = response_matrix_df[col_high].values - response_matrix_df[col_low].values
        n = max(1, len(gains))
        rows.append({
            "step": step_name,
            "mean_gain": float(np.mean(gains)),
            "median_gain": float(np.median(gains)),
            "positive_frac": float(np.mean(gains > 1e-6)),
            "negative_frac": float(np.mean(gains < -1e-6)),
            "zero_frac": float(np.mean(np.abs(gains) <= 1e-6)),
            "gain_gt_001_frac": float(np.mean(gains > 0.01)),
            "drop_lt_minus001_frac": float(np.mean(gains < -0.01)),
        })

    return pd.DataFrame(rows)
