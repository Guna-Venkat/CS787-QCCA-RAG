"""Evaluation metrics and policy simulation for Week 5 Content-Aware Allocation.

Implements:
- End-to-end response routing through frozen fixed-k response surface
- Quality metrics: Policy F1, Mean Regret, Zero-Regret Rate
- Cost metrics: Mean selected k, Context Tokens, Token Reduction vs static k=8
- Recovered Oracle Headroom (ROH): (F1_method - F1_k8) / (F1_oracle - F1_k8)
- Target classification & regression metrics
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error

CONTEXT_TOKEN_COSTS = {
    2: 534.5,
    4: 946.1,
    5: 1157.0,
    6: 1359.1,
    8: 1758.5,
}

# Frozen DEV Reference Values from Weeks 2-3
STATIC_K8_F1_REF = 0.3950
ORACLE_F1_REF = 0.4895
STATIC_K8_TOKENS_REF = 1758.5


def evaluate_allocation_policy(
    df_supervision: pd.DataFrame,
    predicted_k_series: pd.Series | np.ndarray,
) -> Dict[str, float]:
    """Evaluate an allocated budget vector through the response surface."""
    preds = np.asarray(predicted_k_series)
    assert len(preds) == len(df_supervision), f"Size mismatch: {len(preds)} vs {len(df_supervision)}"

    policy_f1s = []
    regrets = []
    token_costs = []
    exact_matches_oracle = []

    for i, (_, row) in enumerate(df_supervision.iterrows()):
        k_val = int(preds[i])
        assert k_val in CONTEXT_TOKEN_COSTS, f"Invalid budget k={k_val}"

        f1_routed = float(row[f"f1_k{k_val}"])
        best_f1 = float(row["best_f1"])
        oracle_k = int(row["oracle_k_eps001"])

        policy_f1s.append(f1_routed)
        regrets.append(best_f1 - f1_routed)
        token_costs.append(CONTEXT_TOKEN_COSTS[k_val])
        exact_matches_oracle.append(1.0 if k_val == oracle_k else 0.0)

    mean_f1 = float(np.mean(policy_f1s))
    mean_k = float(np.mean(preds))
    mean_tokens = float(np.mean(token_costs))
    mean_regret = float(np.mean(regrets))
    zero_regret_pct = float(np.mean(np.array(regrets) <= 0.01 + 1e-9) * 100)
    token_reduction_pct = float((STATIC_K8_TOKENS_REF - mean_tokens) / STATIC_K8_TOKENS_REF * 100)

    # Recovered Oracle Headroom (ROH)
    headroom_denom = ORACLE_F1_REF - STATIC_K8_F1_REF
    roh = float((mean_f1 - STATIC_K8_F1_REF) / headroom_denom) if headroom_denom > 0 else 0.0

    return {
        "policy_mean_f1": round(mean_f1, 4),
        "mean_selected_k": round(mean_k, 2),
        "mean_context_tokens": round(mean_tokens, 1),
        "token_reduction_pct_vs_k8": round(token_reduction_pct, 2),
        "mean_regret": round(mean_regret, 4),
        "zero_regret_pct": round(zero_regret_pct, 2),
        "recovered_oracle_headroom": round(roh, 4),
        "oracle_budget_accuracy": round(float(np.mean(exact_matches_oracle)), 4),
    }


def evaluate_classification_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute secondary classification diagnostics."""
    return {
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "mae_k": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "mean_predicted_k": round(float(np.mean(y_pred)), 2),
    }
