"""Decision policy and threshold calibration for Week 7-C.

Implements:
- Candidate-budget sufficiency decision rule: hat_k = min {k in K : P_k >= tau}, fallback k=8
- Risk-aware decision rule combining P_k >= tau and r_hat_k <= r_max
- Strict INTERNAL-VAL threshold calibration and downstream policy selection
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def apply_sufficiency_decision_rule(
    pred_probs_by_k: Dict[int, float],
    tau: float = 0.70,
    budgets: Sequence[int] = (2, 4, 5, 6, 8),
    fallback_budget: int = 8,
) -> int:
    """Apply the primary W7-C decision rule to select representation budget.

    Decision Rule:
        hat_k = min { k in budgets : P(z(q,k)=1) >= tau }
        Fallback: fallback_budget (default 8).

    Args:
        pred_probs_by_k: Mapping of candidate budget k -> predicted sufficiency probability.
        tau: Decision threshold in [0.5, 0.9].
        budgets: Allowed candidate budgets sorted ascending.
        fallback_budget: Fallback if no candidate meets threshold.

    Returns:
        Selected integer budget hat_k.
    """
    sorted_budgets = sorted(budgets)
    for k in sorted_budgets:
        prob = pred_probs_by_k.get(k, 0.0)
        if prob >= tau:
            return int(k)
    return int(fallback_budget)


def apply_risk_aware_decision_rule(
    pred_probs_by_k: Dict[int, float],
    pred_regrets_by_k: Dict[int, float],
    tau: float = 0.70,
    r_max: float = 0.05,
    budgets: Sequence[int] = (2, 4, 5, 6, 8),
    fallback_budget: int = 8,
) -> int:
    """Apply risk-aware decision rule requiring both sufficiency and bounded regret.

    Decision Rule:
        hat_k = min { k in budgets : P_k >= tau AND r_hat_k <= r_max }
        Fallback: fallback_budget.
    """
    sorted_budgets = sorted(budgets)
    for k in sorted_budgets:
        p = pred_probs_by_k.get(k, 0.0)
        r = pred_regrets_by_k.get(k, 1.0)
        if p >= tau and r <= r_max:
            return int(k)
    return int(fallback_budget)


def evaluate_policy_on_response_surface(
    predicted_budgets: Dict[str, int],
    response_matrix_df: pd.DataFrame,
    budgets: Sequence[int] = (2, 4, 5, 6, 8),
    epsilon: float = 0.01,
) -> Dict[str, Any]:
    """Evaluate a downstream budget policy by looking up actual response-surface metrics.

    Args:
        predicted_budgets: Mapping question_id -> predicted budget hat_k.
        response_matrix_df: Wide response matrix with f1_k2, tokens_k2, etc.
        budgets: Allowed budgets.
        epsilon: Oracle tolerance.

    Returns:
        Dictionary of mean downstream F1, tokens, k, regret, under/over-allocation rates.
    """
    f1_list: List[float] = []
    token_list: List[float] = []
    k_list: List[int] = []
    regret_list: List[float] = []
    under_alloc_count = 0
    over_alloc_count = 0
    admissible_count = 0

    for _, row in response_matrix_df.iterrows():
        qid = str(row["question_id"]).strip()
        pred_k = int(predicted_budgets.get(qid, max(budgets)))
        if pred_k not in budgets:
            pred_k = max(budgets)

        actual_f1 = float(row[f"f1_k{pred_k}"])
        actual_tokens = float(row[f"tokens_k{pred_k}"])

        # Max F1 and oracle k
        all_f1s = {k: float(row[f"f1_k{k}"]) for k in budgets}
        max_f1 = max(all_f1s.values())
        admissible_ks = [k for k in sorted(budgets) if all_f1s[k] >= max_f1 - epsilon]
        oracle_k = admissible_ks[0] if admissible_ks else max(budgets)

        regret = max(0.0, max_f1 - actual_f1)

        f1_list.append(actual_f1)
        token_list.append(actual_tokens)
        k_list.append(pred_k)
        regret_list.append(regret)

        if pred_k < oracle_k:
            under_alloc_count += 1
        elif pred_k > oracle_k:
            over_alloc_count += 1
        else:
            admissible_count += 1

    n_q = max(1, len(f1_list))
    return {
        "mean_f1": float(np.mean(f1_list)),
        "std_f1": float(np.std(f1_list)),
        "mean_tokens": float(np.mean(token_list)),
        "mean_k": float(np.mean(k_list)),
        "mean_regret": float(np.mean(regret_list)),
        "under_alloc_rate": float(under_alloc_count / n_q),
        "over_alloc_rate": float(over_alloc_count / n_q),
        "admissible_rate": float(admissible_count / n_q),
        "per_query_f1s": f1_list,
        "per_query_tokens": token_list,
        "per_query_ks": k_list,
    }


def calibrate_tau_on_internal_val(
    val_candidate_df: pd.DataFrame,
    val_matrix_df: pd.DataFrame,
    pred_probs: np.ndarray,
    tau_grid: Sequence[float] = (0.50, 0.60, 0.70, 0.80, 0.90),
    budgets: Sequence[int] = (2, 4, 5, 6, 8),
    fallback_budget: int = 8,
) -> Tuple[float, pd.DataFrame]:
    """Calibrate decision threshold tau strictly on INTERNAL-VAL.

    Selection Rule:
        1. Maximize downstream F1 on INTERNAL-VAL.
        2. If within 0.003 F1, select the threshold yielding lower mean tokens.
        3. If still tied, prefer the more conservative (higher) tau.

    Returns:
        (best_tau, summary_dataframe)
    """
    df_pred = val_candidate_df[["question_id", "k"]].copy()
    df_pred["prob"] = pred_probs

    summary_rows: List[Dict[str, Any]] = []

    # Map qid -> dict of {k: prob}
    probs_by_qid: Dict[str, Dict[int, float]] = {}
    for _, r in df_pred.iterrows():
        qid = str(r["question_id"]).strip()
        k = int(r["k"])
        if qid not in probs_by_qid:
            probs_by_qid[qid] = {}
        probs_by_qid[qid][k] = float(r["prob"])

    for tau in tau_grid:
        pred_k_map = {
            qid: apply_sufficiency_decision_rule(k_map, tau=tau, budgets=budgets, fallback_budget=fallback_budget)
            for qid, k_map in probs_by_qid.items()
        }
        metrics = evaluate_policy_on_response_surface(pred_k_map, val_matrix_df, budgets=budgets)
        summary_rows.append({
            "tau": float(tau),
            "mean_f1": metrics["mean_f1"],
            "mean_tokens": metrics["mean_tokens"],
            "mean_k": metrics["mean_k"],
            "mean_regret": metrics["mean_regret"],
            "under_alloc_rate": metrics["under_alloc_rate"],
            "over_alloc_rate": metrics["over_alloc_rate"],
            "admissible_rate": metrics["admissible_rate"],
        })

    df_tau_summary = pd.DataFrame(summary_rows)

    # Apply tie-breaker selection rule
    best_f1 = df_tau_summary["mean_f1"].max()
    candidate_rows = df_tau_summary[df_tau_summary["mean_f1"] >= best_f1 - 0.003]
    # Pick lowest tokens among near-optimal F1
    best_row = candidate_rows.sort_values(by=["mean_tokens", "mean_f1"], ascending=[True, False]).iloc[0]
    best_tau = float(best_row["tau"])

    logger.info(
        "INTERNAL-VAL tau calibration complete: selected tau=%.2f (F1=%.4f, tokens=%.1f, mean_k=%.2f)",
        best_tau, best_row["mean_f1"], best_row["mean_tokens"], best_row["mean_k"]
    )
    return best_tau, df_tau_summary
