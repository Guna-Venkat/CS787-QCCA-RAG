"""Decision policies, tie-breaking mechanisms, and model selection for Week 7-C2.

Implements:
- M1 Direct Regression policy: argmax hat_F with delta-near-tie cost minimization.
- M2 Ranking policy: argmax rank score with delta_rank-near-tie cost minimization.
- Strictly enforced INTERNAL-VAL model selection rule:
    1. Highest downstream F1
    2. If within 0.003 F1, lowest context tokens
    3. If tied, simpler model.
- NO fallback-to-k8 assumption.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def apply_m1_regression_decision_rule(
    pred_f1_by_k: Dict[int, float],
    tokens_by_k: Optional[Dict[int, float]] = None,
    delta: float = 0.01,
    budgets: Sequence[int] = (2, 4, 5, 6, 8),
) -> int:
    """Apply M1 Direct Regression decision rule with cost-aware near-tie selection.

    Formula:
        hat_F_max = max_k hat_F(q, k)
        Admissible set: A_delta = {k : hat_F(q, k) >= hat_F_max - delta}
        Choose cheapest budget in A_delta.
    """
    sorted_budgets = sorted(budgets)
    max_pred_f1 = max(pred_f1_by_k[k] for k in sorted_budgets)

    # Near-tie candidates
    candidate_ks = [k for k in sorted_budgets if pred_f1_by_k[k] >= max_pred_f1 - delta]

    if tokens_by_k:
        # Pick lowest token cost
        return min(candidate_ks, key=lambda k: tokens_by_k[k])
    else:
        # Smaller budget k corresponds monotonically to lower token cost
        return min(candidate_ks)


def apply_m2_ranking_decision_rule(
    pred_scores_by_k: Dict[int, float],
    tokens_by_k: Optional[Dict[int, float]] = None,
    score_tie_margin: float = 0.0,
    budgets: Sequence[int] = (2, 4, 5, 6, 8),
) -> int:
    """Apply M2 Query-Grouped Ranking decision rule.

    Formula:
        score_max = max_k s(q, k)
        If score_tie_margin > 0:
            Near-tie set: {k : s(q, k) >= score_max - score_tie_margin}
            Choose cheapest budget in set.
        Else:
            hat_k = argmax_k s(q, k)
    """
    sorted_budgets = sorted(budgets)
    max_score = max(pred_scores_by_k[k] for k in sorted_budgets)

    if score_tie_margin <= 1e-6:
        # Exact argmax
        best_ks = [k for k in sorted_budgets if abs(pred_scores_by_k[k] - max_score) <= 1e-8]
        return min(best_ks)  # Break exact numerical tie by smaller k

    candidate_ks = [k for k in sorted_budgets if pred_scores_by_k[k] >= max_score - score_tie_margin]
    if tokens_by_k:
        return min(candidate_ks, key=lambda k: tokens_by_k[k])
    else:
        return min(candidate_ks)


def evaluate_downstream_policy(
    predicted_budgets: Dict[str, int],
    response_matrix_df: pd.DataFrame,
    budgets: Sequence[int] = (2, 4, 5, 6, 8),
    epsilon: float = 0.01,
) -> Dict[str, Any]:
    """Evaluate downstream policy by looking up actual response-surface metrics."""
    f1_list: List[float] = []
    token_list: List[float] = []
    k_list: List[int] = []
    regret_list: List[float] = []
    admissible_list: List[int] = []
    exact_oracle_list: List[int] = []

    for _, row in response_matrix_df.iterrows():
        qid = str(row["question_id"]).strip()
        pred_k = int(predicted_budgets[qid])

        actual_f1 = float(row[f"f1_k{pred_k}"])
        actual_tokens = float(row[f"tokens_k{pred_k}"])

        f1_vals = {k: float(row[f"f1_k{k}"]) for k in budgets}
        max_f1 = max(f1_vals.values())
        adm_ks = [k for k in sorted(budgets) if f1_vals[k] >= max_f1 - epsilon]
        oracle_k = adm_ks[0] if adm_ks else max(budgets)

        regret = max(0.0, max_f1 - actual_f1)
        is_admissible = 1 if pred_k in adm_ks else 0
        is_exact = 1 if pred_k == oracle_k else 0

        f1_list.append(actual_f1)
        token_list.append(actual_tokens)
        k_list.append(pred_k)
        regret_list.append(regret)
        admissible_list.append(is_admissible)
        exact_oracle_list.append(is_exact)

    n_q = max(1, len(f1_list))
    return {
        "mean_f1": float(np.mean(f1_list)),
        "std_f1": float(np.std(f1_list)),
        "mean_tokens": float(np.mean(token_list)),
        "mean_k": float(np.mean(k_list)),
        "mean_regret": float(np.mean(regret_list)),
        "admissible_rate": float(np.mean(admissible_list)),
        "exact_oracle_rate": float(np.mean(exact_oracle_list)),
        "per_query_f1s": f1_list,
        "per_query_tokens": token_list,
        "per_query_ks": k_list,
    }


def select_best_model_internal_val(
    model_eval_records: Sequence[Dict[str, Any]],
    tolerance_f1: float = 0.003,
) -> Dict[str, Any]:
    """Select winning model using the strict quantitative selection rule:

    1. Maximize downstream F1 on INTERNAL-VAL.
    2. If within tolerance_f1 (0.003) of best F1, prefer lower context tokens.
    3. If still tied, prefer the simpler model (lower complexity rank).
    """
    if not model_eval_records:
        raise ValueError("No model evaluation records provided!")

    df_records = pd.DataFrame(model_eval_records)
    best_f1 = df_records["mean_f1"].max()

    # Filter to models within tolerance_f1
    near_optimal = df_records[df_records["mean_f1"] >= best_f1 - tolerance_f1].copy()

    # Sort primarily by lowest tokens, secondarily by highest F1, tertiarily by complexity_rank
    if "complexity_rank" not in near_optimal.columns:
        near_optimal["complexity_rank"] = 1

    sorted_candidates = near_optimal.sort_values(
        by=["mean_tokens", "mean_f1", "complexity_rank"],
        ascending=[True, False, True],
    )
    winning_record = sorted_candidates.iloc[0].to_dict()

    logger.info(
        "Quantitative Model Selection: Winner '%s' (F1=%.4f, tokens=%.1f, mean_k=%.2f) selected under %.3f tolerance.",
        winning_record.get("model_name", "unnamed"),
        winning_record["mean_f1"],
        winning_record["mean_tokens"],
        winning_record["mean_k"],
        tolerance_f1,
    )
    return winning_record
