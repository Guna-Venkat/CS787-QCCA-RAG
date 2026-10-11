"""Evaluation metrics, policy simulation, and paper-clustered bootstrap for Week 5.

Implements:
- End-to-end policy evaluation through frozen fixed-k response surface
- Six primary policy metrics:
    1. Mean Answer F1
    2. Mean Selected k
    3. Mean Context Tokens
    4. Mean Oracle Regret
    5. Context Reduction vs Static k=8
    6. Recovered Oracle Headroom (ROH): (F1_policy - 0.3950) / (0.4895 - 0.3950)
- Secondary Target 1 classification diagnostics: Macro F1, Balanced Accuracy, Per-Class P/R, Confusion Matrix
- Secondary Target 2 regression diagnostics: MAE per k, Mean MAE, Spearman correlation
- Paper-clustered bootstrap for statistical hypothesis testing (clustered by paper_id)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
)

from src.week5.data import K_CLASSES, CLASS_TO_K

# Frozen DEV Reference Values (Weeks 2-3)
STATIC_K8_F1_REF = 0.3950
ORACLE_F1_REF = 0.4895
STATIC_K8_TOKENS_REF = 1750.5  # Exact empirical mean for k=8 on DEV (231 queries)

# Empirical DEV token means per budget
DEV_TOKEN_COSTS_DEFAULT = {
    2: 526.5,
    4: 938.1,
    5: 1149.0,
    6: 1351.1,
    8: 1750.5,
}


def evaluate_allocation_policy(
    df_supervision: pd.DataFrame,
    predicted_k_series: Union[pd.Series, np.ndarray, List[int]],
    dev_tokens_map: Optional[Dict[Tuple[str, int], int]] = None,
) -> Dict[str, Any]:
    """Evaluate an allocated budget vector through the frozen response surface.

    Args:
        df_supervision: DataFrame with columns f1_k2..f1_k8, best_f1, oracle_k_eps001, question_id
        predicted_k_series: Predicted k in {2, 4, 5, 6, 8} for each row
        dev_tokens_map: Optional dictionary of (qid_str, k) -> context_tokens
    """
    preds = np.asarray(predicted_k_series)
    assert len(preds) == len(df_supervision), f"Size mismatch: {len(preds)} vs {len(df_supervision)}"

    policy_f1s: List[float] = []
    regrets: List[float] = []
    token_costs: List[float] = []
    exact_matches_oracle: List[float] = []

    for i, (_, row) in enumerate(df_supervision.iterrows()):
        k_val = int(preds[i])
        assert k_val in K_CLASSES, f"Invalid budget k={k_val}, must be in {K_CLASSES}"

        qid_str = str(row["question_id"])
        f1_routed = float(row[f"f1_k{k_val}"])
        best_f1 = float(row["best_f1"])
        oracle_k = int(row["oracle_k_eps001"])

        policy_f1s.append(f1_routed)
        regrets.append(best_f1 - f1_routed)

        if dev_tokens_map and (qid_str, k_val) in dev_tokens_map:
            token_costs.append(float(dev_tokens_map[(qid_str, k_val)]))
        else:
            token_costs.append(DEV_TOKEN_COSTS_DEFAULT.get(k_val, 1750.5))

        exact_matches_oracle.append(1.0 if k_val == oracle_k else 0.0)

    mean_f1 = float(np.mean(policy_f1s))
    mean_k = float(np.mean(preds))
    mean_tokens = float(np.mean(token_costs))
    mean_regret = float(np.mean(regrets))
    zero_regret_pct = float(np.mean(np.array(regrets) <= 0.01 + 1e-9) * 100.0)
    token_reduction_pct = float((STATIC_K8_TOKENS_REF - mean_tokens) / STATIC_K8_TOKENS_REF * 100.0)

    # Recovered Oracle Headroom (ROH)
    headroom_denom = ORACLE_F1_REF - STATIC_K8_F1_REF
    roh = float((mean_f1 - STATIC_K8_F1_REF) / headroom_denom) if headroom_denom > 0 else 0.0

    return {
        "mean_answer_f1": round(mean_f1, 4),
        "mean_selected_k": round(mean_k, 2),
        "mean_context_tokens": round(mean_tokens, 1),
        "token_reduction_pct_vs_k8": round(token_reduction_pct, 2),
        "mean_oracle_regret": round(mean_regret, 4),
        "zero_regret_pct": round(zero_regret_pct, 2),
        "recovered_oracle_headroom": round(roh, 4),
        "oracle_budget_accuracy": round(float(np.mean(exact_matches_oracle)), 4),
        "per_query_f1s": np.array(policy_f1s, dtype=np.float32),
        "per_query_tokens": np.array(token_costs, dtype=np.float32),
        "per_query_regrets": np.array(regrets, dtype=np.float32),
        "per_query_selected_k": preds,
    }


def evaluate_classification_diagnostics(
    y_true_classes: np.ndarray,
    y_pred_classes: np.ndarray,
) -> Dict[str, Any]:
    """Compute secondary classification diagnostics for T1."""
    macro_f1 = float(f1_score(y_true_classes, y_pred_classes, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true_classes, y_pred_classes))
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true_classes, y_pred_classes, labels=list(range(len(K_CLASSES))), zero_division=0
    )
    conf_mat = confusion_matrix(y_true_classes, y_pred_classes, labels=list(range(len(K_CLASSES))))

    per_class_metrics = {}
    for idx, k in enumerate(K_CLASSES):
        per_class_metrics[f"k{k}"] = {
            "precision": round(float(prec[idx]), 4),
            "recall": round(float(rec[idx]), 4),
            "f1": round(float(f1[idx]), 4),
            "support": int(support[idx]),
        }

    return {
        "macro_f1": round(macro_f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "per_class": per_class_metrics,
        "confusion_matrix": conf_mat.tolist(),
    }


def evaluate_regression_diagnostics(
    y_true_regrets: np.ndarray,
    y_pred_regrets: np.ndarray,
) -> Dict[str, Any]:
    """Compute secondary regression diagnostics for T2."""
    mae_per_k: Dict[str, float] = {}
    spearman_per_k: Dict[str, float] = {}

    for idx, k in enumerate(K_CLASSES):
        true_k = y_true_regrets[:, idx]
        pred_k = y_pred_regrets[:, idx]
        mae_per_k[f"mae_k{k}"] = round(float(mean_absolute_error(true_k, pred_k)), 4)
        corr, _ = spearmanr(true_k, pred_k)
        spearman_per_k[f"spearman_k{k}"] = round(float(corr), 4) if not np.isnan(corr) else 0.0

    overall_mae = float(mean_absolute_error(y_true_regrets, y_pred_regrets))

    return {
        "overall_mae": round(overall_mae, 4),
        "mae_per_k": mae_per_k,
        "spearman_per_k": spearman_per_k,
    }


def compute_paper_clustered_bootstrap(
    paper_ids: List[str],
    policy_f1s: np.ndarray,
    baseline_f1s: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Compute paper-clustered bootstrap confidence interval for Delta F1.

    Resamples at the paper level (clustering questions by paper_id) to account for
    intra-paper question correlations.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "paper_id": paper_ids,
        "policy_f1": policy_f1s,
        "baseline_f1": baseline_f1s,
    })
    unique_papers = df["paper_id"].unique()
    n_papers = len(unique_papers)

    # Pre-aggregate mean F1 differences per paper for ultra-fast sampling
    grouped = df.groupby("paper_id")[["policy_f1", "baseline_f1"]].agg(["sum", "count"])
    paper_policy_sums = grouped[("policy_f1", "sum")].values
    paper_baseline_sums = grouped[("baseline_f1", "sum")].values
    paper_counts = grouped[("policy_f1", "count")].values

    observed_delta = float(np.mean(policy_f1s) - np.mean(baseline_f1s))
    bootstrap_deltas = []

    for _ in range(n_bootstrap):
        sampled_indices = rng.choice(n_papers, size=n_papers, replace=True)
        tot_policy = np.sum(paper_policy_sums[sampled_indices])
        tot_baseline = np.sum(paper_baseline_sums[sampled_indices])
        tot_q = np.sum(paper_counts[sampled_indices])
        if tot_q > 0:
            delta_b = (tot_policy - tot_baseline) / tot_q
            bootstrap_deltas.append(delta_b)

    bootstrap_deltas = np.array(bootstrap_deltas)
    ci_lower = float(np.percentile(bootstrap_deltas, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(bootstrap_deltas, 100.0 * (1.0 - alpha / 2.0)))
    p_value = float(np.mean(bootstrap_deltas <= 0.0) if observed_delta > 0 else np.mean(bootstrap_deltas >= 0.0))

    return {
        "observed_delta_f1": round(observed_delta, 4),
        "ci_lower_95": round(ci_lower, 4),
        "ci_upper_95": round(ci_upper, 4),
        "p_value": round(p_value, 4),
        "n_bootstrap": n_bootstrap,
    }
