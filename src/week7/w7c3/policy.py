"""Anchor-and-Deviate Decision Policy and Quality-Constrained Model Selection for Week 7-C3.

Implements:
- Asymmetric Anchor-and-Deviate Decision Rule:
    Default anchor = k4
    1. If P_down(q) >= tau_down: hat_k = 2
    2. Else if P_up(q) >= tau_up:
        U_delta = {k in {5,6,8} : hat_Delta_k >= hat_Delta_max - delta_up}
        hat_k = min U_delta
    3. Else: hat_k = 4  (Anchor retained)
    NO k8 fallback assumption.

- Quality-Constrained Cost Minimization Model Selection:
    1. Filter policies with OOF F1 >= F1_{k4, OOF} - eta
    2. Among qualifying, choose LOWEST mean context tokens
    3. Tie-breaker: lower under-allocation rate, then simpler thresholds.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def apply_anchor_deviate_decision_rule(
    p_down: float,
    p_up: float,
    pred_deltas: Dict[int, float],
    tau_down: float = 0.85,
    tau_up: float = 0.70,
    delta_up: float = 0.0,
    anchor_k: int = 4,
    upshift_budgets: Sequence[int] = (5, 6, 8),
) -> int:
    """Apply the asymmetric Anchor-and-Deviate decision rule.

    Args:
        p_down: Predicted probability that k=2 is safe (epsilon-admissible).
        p_up: Predicted probability that k > 4 provides quality gain > epsilon.
        pred_deltas: Predicted marginal gains Delta_k = F(k) - F(4) for k in upshift_budgets.
        tau_down: Confidence threshold for safe downshift.
        tau_up: Confidence threshold for selective upshift.
        delta_up: Near-tie margin for selecting among larger budgets.
        anchor_k: Default deployment anchor (default 4).
        upshift_budgets: Candidate larger budgets (default 5, 6, 8).

    Returns:
        Selected budget hat_k in {2, 4, 5, 6, 8}.
    """
    # 1. Safe downshift check
    if p_down >= tau_down:
        return 2

    # 2. Selective upshift check
    if p_up >= tau_up:
        sorted_up = sorted(upshift_budgets)
        max_delta = max(pred_deltas[k] for k in sorted_up)
        near_best = [k for k in sorted_up if pred_deltas[k] >= max_delta - delta_up]
        return min(near_best)

    # 3. Default anchor
    return anchor_k


def evaluate_downstream_anchor_policy(
    predicted_ks: Dict[str, int],
    response_matrix_df: pd.DataFrame,
    budgets: Sequence[int] = (2, 4, 5, 6, 8),
    anchor_k: int = 4,
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
    anchor_chosen_list: List[int] = []
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

        # Error categorization
        is_under = int(pred_k not in adm_ks and pred_k < min_oracle_k)
        is_over = int(pred_k in adm_ks and pred_k > min_oracle_k)
        is_higher_inadm = int(pred_k not in adm_ks and pred_k > min_oracle_k)

        # Policy decision tracking
        is_anchor = int(pred_k == anchor_k)
        is_down = int(pred_k < anchor_k)
        is_up = int(pred_k > anchor_k)

        f1_list.append(actual_f1)
        token_list.append(actual_tokens)
        k_list.append(pred_k)
        regret_list.append(f1_regret)
        admissible_list.append(is_adm)
        exact_oracle_list.append(is_exact)
        under_alloc_list.append(is_under)
        over_budget_list.append(is_over)
        higher_inadm_list.append(is_higher_inadm)
        anchor_chosen_list.append(is_anchor)
        downshift_list.append(is_down)
        upshift_list.append(is_up)

    n = max(1, len(f1_list))
    mean_f1 = float(np.mean(f1_list))

    # Reference static values for ROH
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
        "anchor_rate": float(np.mean(anchor_chosen_list)),
        "downshift_rate": float(np.mean(downshift_list)),
        "upshift_rate": float(np.mean(upshift_list)),
        "ROH_sem": roh,
        "n_queries": n,
    }


def select_best_anchor_policy(
    df_grid: pd.DataFrame,
    f1_k4_reference: float,
    eta: float = 0.005,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Select the best policy using Quality-Constrained Cost Minimization.

    Selection Rule:
    1. Identify qualifying policies: mean_f1 >= f1_k4_reference - eta
    2. Among qualifying policies, choose LOWEST mean context tokens.
    3. Tie-breaker: lower under_allocation_rate, then simpler thresholds.
    """
    df_eval = df_grid.copy()
    f1_thresh = f1_k4_reference - eta
    df_eval["qualifies_eta"] = df_eval["mean_f1"] >= f1_thresh

    qualifying = df_eval[df_eval["qualifies_eta"]].copy()

    if len(qualifying) > 0:
        # Sort by: lowest tokens, lowest under-allocation rate, highest tau_down (more conservative), highest tau_up, lowest delta_up
        qualifying = qualifying.sort_values(
            by=["mean_tokens", "under_allocation_rate", "tau_down", "tau_up", "delta_up"],
            ascending=[True, True, False, False, True],
        ).reset_index(drop=True)
        winner = qualifying.iloc[0].to_dict()
        winner["selection_status"] = "QUALIFIED_LOWEST_TOKENS"
    else:
        # Fallback if no policy satisfies eta: pick highest F1, then lowest tokens
        logger.warning("No policy satisfied eta=%f constraint (ref F1=%.4f)! Falling back to highest F1.", eta, f1_k4_reference)
        fallback = df_eval.sort_values(
            by=["mean_f1", "mean_tokens"],
            ascending=[False, True],
        ).reset_index(drop=True)
        winner = fallback.iloc[0].to_dict()
        winner["selection_status"] = "FALLBACK_HIGHEST_F1"

    winner["eta"] = eta
    winner["f1_k4_reference"] = f1_k4_reference
    return winner, df_eval
