"""Decision policies and quality-constrained model selection for Week 7-D.

Provides:
- Anchor-and-Deviate Decision Policy (Anchor = k4; safe downshift to k2; justified upshift to k5/k6/k8; fallback = k4).
- Direct Multi-label Admissibility Policy (Cheapest predicted admissible k; fallback = k4).
- Quality-Constrained Cost Minimization Selection Rule on INTERNAL-VAL (reference = static k4, eta = 0.005).
- Downstream policy evaluation via semantic response surface lookup.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BUDGETS: List[int] = [2, 4, 5, 6, 8]
ANCHOR_K: int = 4


def apply_w7d_anchor_deviate_policy(
    p_down: float,
    p_up: float,
    pred_deltas: Dict[int, float],
    tau_down: float = 0.85,
    tau_up: float = 0.70,
    delta_up: float = 0.0,
    anchor_k: int = ANCHOR_K,
    upshift_budgets: Sequence[int] = (5, 6, 8),
) -> int:
    """Apply Anchor-and-Deviate policy:
    1. If p_down >= tau_down -> k=2
    2. Else if p_up >= tau_up -> cheapest budget in U_delta
    3. Else -> anchor_k (4).
    NO k8 fallback.
    """
    if p_down >= tau_down:
        return 2

    if p_up >= tau_up:
        sorted_up = sorted(upshift_budgets)
        max_delta = max(pred_deltas[k] for k in sorted_up)
        near_best = [k for k in sorted_up if pred_deltas[k] >= max_delta - delta_up]
        return min(near_best)

    return anchor_k


def apply_w7d_multilabel_policy(
    pred_admissibility: Dict[int, float],
    tau_adm: float = 0.50,
    budgets: Sequence[int] = (2, 4, 5, 6, 8),
    fallback_k: int = ANCHOR_K,
) -> int:
    """Apply direct multi-label admissibility policy:
    Chooses cheapest budget where p(admissible) >= tau_adm.
    If none pass, falls back to fallback_k (default k=4, NEVER k=8).
    """
    admissible_ks = [k for k in sorted(budgets) if pred_admissibility[k] >= tau_adm]
    if admissible_ks:
        return min(admissible_ks)
    return fallback_k


def evaluate_downstream_fidelity_policy(
    predicted_ks: Dict[str, int],
    response_matrix_df: pd.DataFrame,
    budgets: Sequence[int] = (2, 4, 5, 6, 8),
    anchor_k: int = ANCHOR_K,
    epsilon: float = 0.01,
) -> Dict[str, Any]:
    """Evaluate downstream policy metrics by looking up actual response surface."""
    f1_list: List[float] = []
    token_list: List[float] = []
    k_list: List[int] = []
    regret_list: List[float] = []
    admissible_list: List[int] = []
    exact_oracle_list: List[int] = []
    under_alloc_list: List[int] = []
    over_budget_list: List[int] = []
    higher_inadm_list: List[int] = []
    anchor_list: List[int] = []
    downshift_list: List[int] = []
    upshift_list: List[int] = []

    for _, row in response_matrix_df.iterrows():
        qid = str(row["question_id"]).strip()
        pred_k = int(predicted_ks[qid])

        actual_f1 = float(row[f"f1_k{pred_k}"])
        actual_tokens = float(row[f"tokens_k{pred_k}"])

        f1_vals = {k: float(row[f"f1_k{k}"]) for k in budgets}
        max_f1 = max(f1_vals.values())
        adm_ks = set(k for k in budgets if f1_vals[k] >= max_f1 - epsilon)
        min_oracle_k = min(adm_ks) if adm_ks else max(budgets)

        f1_regret = max(0.0, max_f1 - actual_f1)
        is_adm = int(pred_k in adm_ks)
        is_exact = int(pred_k == min_oracle_k)

        is_under = int(pred_k not in adm_ks and pred_k < min_oracle_k)
        is_over = int(pred_k in adm_ks and pred_k > min_oracle_k)
        is_higher_inadm = int(pred_k not in adm_ks and pred_k > min_oracle_k)

        f1_list.append(actual_f1)
        token_list.append(actual_tokens)
        k_list.append(pred_k)
        regret_list.append(f1_regret)
        admissible_list.append(is_adm)
        exact_oracle_list.append(is_exact)
        under_alloc_list.append(is_under)
        over_budget_list.append(is_over)
        higher_inadm_list.append(is_higher_inadm)

        anchor_list.append(int(pred_k == anchor_k))
        downshift_list.append(int(pred_k < anchor_k))
        upshift_list.append(int(pred_k > anchor_k))

    n = max(1, len(f1_list))
    mean_f1 = float(np.mean(f1_list))

    f1_k8 = float(response_matrix_df["f1_k8"].mean())
    f1_oracle = float(np.mean([max(float(r[f"f1_k{k}"]) for k in budgets) for _, r in response_matrix_df.iterrows()]))
    denom = f1_oracle - f1_k8
    roh = float((mean_f1 - f1_k8) / denom) if abs(denom) > 1e-6 else 0.0

    return {
        "mean_f1": mean_f1,
        "mean_tokens": float(np.mean(token_list)),
        "mean_k": float(np.mean(k_list)),
        "mean_regret": float(np.mean(regret_list)),
        "admissible_rate": float(np.mean(admissible_list)),
        "exact_oracle_rate": float(np.mean(exact_oracle_list)),
        "under_allocation_rate": float(np.mean(under_alloc_list)),
        "over_budget_rate": float(np.mean(over_budget_list)),
        "higher_budget_inadmissible_rate": float(np.mean(higher_inadm_list)),
        "anchor_rate": float(np.mean(anchor_list)),
        "downshift_rate": float(np.mean(downshift_list)),
        "upshift_rate": float(np.mean(upshift_list)),
        "ROH_sem": roh,
        "n_queries": n,
    }


def select_best_fidelity_policy(
    df_grid: pd.DataFrame,
    f1_k4_reference: float,
    eta: float = 0.005,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Select best policy using Quality-Constrained Cost Minimization on INTERNAL-VAL.

    Rule:
    1. Filter: mean_f1 >= f1_k4_reference - eta
    2. Select lowest mean_tokens
    3. Tie-break with lower under_allocation_rate, then simpler model.
    """
    df_eval = df_grid.copy()
    f1_thresh = f1_k4_reference - eta
    df_eval["qualifies_eta"] = df_eval["mean_f1"] >= f1_thresh

    qualifying = df_eval[df_eval["qualifies_eta"]].copy()

    if len(qualifying) > 0:
        sort_cols = ["mean_tokens", "under_allocation_rate"]
        sort_asc = [True, True]
        if "tau_down" in qualifying.columns:
            sort_cols.extend(["tau_down", "tau_up", "delta_up"])
            sort_asc.extend([False, False, True])

        qualifying = qualifying.sort_values(by=sort_cols, ascending=sort_asc).reset_index(drop=True)
        winner = qualifying.iloc[0].to_dict()
        winner["selection_status"] = "QUALIFIED_LOWEST_TOKENS"
    else:
        logger.warning("No policy satisfied eta=%f constraint (ref F1=%.4f)! Falling back to highest F1.", eta, f1_k4_reference)
        fallback = df_eval.sort_values(by=["mean_f1", "mean_tokens"], ascending=[False, True]).reset_index(drop=True)
        winner = fallback.iloc[0].to_dict()
        winner["selection_status"] = "FALLBACK_HIGHEST_F1"

    winner["eta"] = eta
    winner["f1_k4_reference"] = f1_k4_reference
    return winner, df_eval
