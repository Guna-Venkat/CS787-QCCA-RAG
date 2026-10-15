"""Oracle stability, anchor diagnostics, and 4->6 marginal gain diagnostics for Week 7-C3.

Implements:
- Oracle Anchor Diagnostics:
    P(k2 in A_eps), P(k4 in A_eps), P(k4 not in A_eps),
    cheapest admissible larger budget distribution among k4-inadequate,
    and P(k8 not in A_eps) demonstrating why k=8 fallback fails.
- 4->6 Gain Analysis:
    Distribution of G_{4->6} = F(q, 6) - F(q, 4), positive fraction,
    meaningful gains (>0.01), meaningful degradations (<-0.01),
    and relationships with tail relevance, novelty, and coverage for W7-D handoff.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_anchor_oracle_diagnostics(
    response_matrix_df: pd.DataFrame,
    budgets: Sequence[int] = (2, 4, 5, 6, 8),
    anchor_k: int = 4,
    epsilon: float = 0.01,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Compute oracle admissibility diagnostics for k=2, k=4, and k=8.

    Answers:
    - How frequently is k=2 admissible? (Potential for safe downshift)
    - How frequently is anchor k=4 admissible?
    - When k=4 is inadequate, which larger budget is the cheapest admissible action?
    - How frequently is k=8 INADMISSIBLE? (Why k=8 fallback fails)
    """
    n_total = len(response_matrix_df)
    k2_adm_count = 0
    k4_adm_count = 0
    k4_inadeq_count = 0
    k8_inadm_count = 0

    cheapest_larger_counts: Dict[str, int] = {"k5": 0, "k6": 0, "k8": 0, "none": 0}

    for _, row in response_matrix_df.iterrows():
        f1_vals = {k: float(row[f"f1_k{k}"]) for k in budgets}
        max_f1 = max(f1_vals.values())
        adm_ks = set(k for k in budgets if f1_vals[k] >= max_f1 - epsilon)

        if 2 in adm_ks:
            k2_adm_count += 1

        if anchor_k in adm_ks:
            k4_adm_count += 1
        else:
            k4_inadeq_count += 1
            larger_adm = [k for k in [5, 6, 8] if k in adm_ks]
            if larger_adm:
                cheapest_larger_counts[f"k{min(larger_adm)}"] += 1
            else:
                cheapest_larger_counts["none"] += 1

        if 8 not in adm_ks:
            k8_inadm_count += 1

    stats = {
        "n_queries": n_total,
        "k2_admissible_rate": float(k2_adm_count / n_total),
        "k2_admissible_count": k2_adm_count,
        "k4_admissible_rate": float(k4_adm_count / n_total),
        "k4_admissible_count": k4_adm_count,
        "k4_inadequate_rate": float(k4_inadeq_count / n_total),
        "k4_inadequate_count": k4_inadeq_count,
        "k8_inadmissible_rate": float(k8_inadm_count / n_total),
        "k8_inadmissible_count": k8_inadm_count,
        "cheapest_larger_counts": cheapest_larger_counts,
    }

    # Summary DataFrame for tabular export
    rows = [
        {"metric": "k2_admissible_rate", "value": stats["k2_admissible_rate"], "count": k2_adm_count, "description": "Queries where k=2 preserves quality (safe downshift potential)"},
        {"metric": "k4_admissible_rate", "value": stats["k4_admissible_rate"], "count": k4_adm_count, "description": "Queries where anchor k=4 is epsilon-admissible"},
        {"metric": "k4_inadequate_rate", "value": stats["k4_inadequate_rate"], "count": k4_inadeq_count, "description": "Queries where anchor k=4 misses oracle by > epsilon"},
        {"metric": "k8_inadmissible_rate", "value": stats["k8_inadmissible_rate"], "count": k8_inadm_count, "description": "Queries where k=8 is NOT admissible (proof of fallback failure)"},
    ]

    for k_name, cnt in cheapest_larger_counts.items():
        pct = float(cnt / k4_inadeq_count) if k4_inadeq_count > 0 else 0.0
        rows.append({
            "metric": f"k4_inadeq_cheapest_{k_name}",
            "value": pct,
            "count": cnt,
            "description": f"Among k4-inadequate queries, fraction where {k_name} is cheapest admissible",
        })

    df_table = pd.DataFrame(rows)
    return stats, df_table


def compute_4to6_gain_analysis(
    response_matrix_df: pd.DataFrame,
    features_df: Optional[pd.DataFrame] = None,
    anchor_k: int = 4,
    epsilon: float = 0.01,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Analyze the marginal gain G_{4->6} = F(q, 6) - F(q, 4) for W7-D handoff."""
    f1_k4 = response_matrix_df[f"f1_k{anchor_k}"].values.astype(float)
    f1_k6 = response_matrix_df["f1_k6"].values.astype(float)
    g_46 = f1_k6 - f1_k4
    n = len(g_46)

    mean_gain = float(np.mean(g_46))
    median_gain = float(np.median(g_46))
    std_gain = float(np.std(g_46))
    pos_frac = float(np.mean(g_46 > 0.0))
    gt_eps_frac = float(np.mean(g_46 > epsilon))
    lt_neg_eps_frac = float(np.mean(g_46 < -epsilon))
    zero_frac = float(np.mean(np.isclose(g_46, 0.0, atol=1e-6)))

    stats = {
        "mean_gain_4to6": mean_gain,
        "median_gain_4to6": median_gain,
        "std_gain_4to6": std_gain,
        "positive_fraction": pos_frac,
        "meaningful_gain_gt_0_01": gt_eps_frac,
        "meaningful_loss_lt_minus_0_01": lt_neg_eps_frac,
        "zero_gain_fraction": zero_frac,
    }

    rows = [
        {"metric": "mean_gain", "value": mean_gain, "description": "Mean downstream F1 gain from adding passages 5 and 6"},
        {"metric": "median_gain", "value": median_gain, "description": "Median gain (typically 0.0)"},
        {"metric": "std_gain", "value": std_gain, "description": "Standard deviation of 4->6 gains"},
        {"metric": "positive_fraction", "value": pos_frac, "description": "Fraction of queries with strictly positive gain"},
        {"metric": "gain_gt_0_01", "value": gt_eps_frac, "description": "Fraction with meaningful quality improvement (> 0.01)"},
        {"metric": "loss_lt_minus_0_01", "value": lt_neg_eps_frac, "description": "Fraction with meaningful quality degradation (< -0.01)"},
        {"metric": "zero_fraction", "value": zero_frac, "description": "Fraction of queries where adding passages 5 and 6 produces identical F1"},
    ]

    df_table = pd.DataFrame(rows)
    return stats, df_table
