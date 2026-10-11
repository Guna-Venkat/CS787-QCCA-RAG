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
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
    roc_auc_score,
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


def evaluate_multilabel_diagnostics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    tau: float = 0.50,
) -> Dict[str, Any]:
    """Compute multi-label classification diagnostics for T3.

    Args:
        y_true: (N, 5) binary indicators (0 or 1)
        y_pred_probs: (N, 5) sigmoid probabilities
        tau: Decision threshold for binary metrics
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred_probs = np.asarray(y_pred_probs, dtype=np.float32)
    assert y_true.shape == y_pred_probs.shape, f"Shape mismatch: {y_true.shape} vs {y_pred_probs.shape}"

    per_budget_auroc: Dict[str, float] = {}
    per_budget_auprc: Dict[str, float] = {}
    per_budget_metrics: Dict[str, Dict[str, float]] = {}

    aurocs: List[float] = []
    auprcs: List[float] = []

    for idx, k in enumerate(K_CLASSES):
        yt = y_true[:, idx]
        yp = y_pred_probs[:, idx]

        # Check if both classes are present
        if len(np.unique(yt)) > 1:
            auc = float(roc_auc_score(yt, yp))
            ap = float(average_precision_score(yt, yp))
        else:
            auc = 0.5
            ap = float(np.mean(yt))

        aurocs.append(auc)
        auprcs.append(ap)
        per_budget_auroc[f"k{k}"] = round(auc, 4)
        per_budget_auprc[f"k{k}"] = round(ap, 4)

        # Binary metrics at tau
        pred_bin = (yp >= tau).astype(np.int64)
        prec, rec, f1, _ = precision_recall_fscore_support(yt, pred_bin, average="binary", zero_division=0)
        brier = float(np.mean((yp - yt) ** 2))

        per_budget_metrics[f"k{k}"] = {
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "brier": round(brier, 4),
            "positive_rate": round(float(np.mean(yt)), 4),
            "predicted_positive_rate": round(float(np.mean(pred_bin)), 4),
        }

    macro_auroc = float(np.mean(aurocs))
    macro_auprc = float(np.mean(auprcs))

    try:
        micro_auprc = float(average_precision_score(y_true, y_pred_probs, average="micro"))
    except Exception:
        micro_auprc = macro_auprc

    overall_brier = float(np.mean((y_pred_probs - y_true) ** 2))

    return {
        "macro_auroc": round(macro_auroc, 4),
        "macro_auprc": round(macro_auprc, 4),
        "micro_auprc": round(micro_auprc, 4),
        "overall_brier": round(overall_brier, 4),
        "per_budget_auroc": per_budget_auroc,
        "per_budget_auprc": per_budget_auprc,
        "per_budget_metrics": per_budget_metrics,
    }


def select_tau_internal_val(
    val_supervision_df: pd.DataFrame,
    val_pred_probs: np.ndarray,
    val_tokens_map: Optional[Dict[Tuple[str, int], int]] = None,
    tau_grid: Optional[List[float]] = None,
    quality_tolerance: float = 0.005,
) -> Dict[str, Any]:
    """Select inference probability threshold tau on INTERNAL-VAL using frozen decision rule:

    Grid: tau in {0.30, 0.40, 0.50, 0.60, 0.70}
    Rule:
    1. Prefer policies whose INTERNAL-VAL mean answer F1 is within 0.005 absolute of static k=8 INTERNAL-VAL F1.
    2. Among those, choose the policy with lowest mean context tokens.
    3. If none satisfy the 0.005 quality tolerance, select the highest-F1 policy and report constraint_met = False.
    """
    if tau_grid is None:
        tau_grid = [0.30, 0.40, 0.50, 0.60, 0.70]

    # Evaluate static k=8 on INTERNAL-VAL as anchor
    k8_preds = [8] * len(val_supervision_df)
    k8_eval = evaluate_allocation_policy(val_supervision_df, k8_preds, val_tokens_map)
    static_k8_f1 = k8_eval["mean_answer_f1"]

    grid_results = []

    for tau in tau_grid:
        k_preds = []
        fallbacks = []
        for row in val_pred_probs:
            valid_ks = [K_CLASSES[idx] for idx, p in enumerate(row) if p >= tau]
            if valid_ks:
                k_preds.append(min(valid_ks))
                fallbacks.append(0.0)
            else:
                k_preds.append(8)
                fallbacks.append(1.0)

        policy_eval = evaluate_allocation_policy(val_supervision_df, k_preds, val_tokens_map)
        fallback_rate = round(float(np.mean(fallbacks) * 100.0), 2)

        grid_results.append({
            "tau": round(float(tau), 2),
            "mean_answer_f1": policy_eval["mean_answer_f1"],
            "mean_context_tokens": policy_eval["mean_context_tokens"],
            "mean_selected_k": policy_eval["mean_selected_k"],
            "mean_oracle_regret": policy_eval["mean_oracle_regret"],
            "fallback_rate": fallback_rate,
            "zero_regret_pct": policy_eval["zero_regret_pct"],
            "token_reduction_pct_vs_k8": policy_eval["token_reduction_pct_vs_k8"],
            "policy_eval": policy_eval,
        })

    # Decision rule: within 0.005 of static k=8 F1
    admissible = [r for r in grid_results if r["mean_answer_f1"] >= static_k8_f1 - quality_tolerance]

    if admissible:
        # Lowest mean context tokens (tie break: higher F1)
        best_choice = sorted(admissible, key=lambda x: (x["mean_context_tokens"], -x["mean_answer_f1"]))[0]
        constraint_met = True
    else:
        # Highest F1
        best_choice = sorted(grid_results, key=lambda x: -x["mean_answer_f1"])[0]
        constraint_met = False

    return {
        "selected_tau": best_choice["tau"],
        "constraint_met": constraint_met,
        "static_k8_f1": static_k8_f1,
        "grid_results": grid_results,
        "selected_result": best_choice,
    }


def compute_paper_clustered_bootstrap(
    paper_ids: List[str],
    policy_f1s: np.ndarray,
    baseline_f1s: np.ndarray,
    policy_tokens: Optional[np.ndarray] = None,
    baseline_tokens: Optional[np.ndarray] = None,
    n_bootstrap: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Compute paper-clustered bootstrap confidence interval for Delta F1 and Delta Tokens.

    Resamples at the paper level (clustering questions by paper_id) to account for
    intra-paper question correlations.
    """
    rng = np.random.default_rng(seed)
    df_dict = {
        "paper_id": paper_ids,
        "policy_f1": policy_f1s,
        "baseline_f1": baseline_f1s,
    }
    has_tokens = policy_tokens is not None and baseline_tokens is not None
    if has_tokens:
        df_dict["policy_tokens"] = policy_tokens
        df_dict["baseline_tokens"] = baseline_tokens

    df = pd.DataFrame(df_dict)
    unique_papers = df["paper_id"].unique()
    n_papers = len(unique_papers)

    # Pre-aggregate sums per paper
    cols_to_agg = ["policy_f1", "baseline_f1"]
    if has_tokens:
        cols_to_agg.extend(["policy_tokens", "baseline_tokens"])

    grouped = df.groupby("paper_id")[cols_to_agg].agg(["sum", "count"])
    paper_policy_sums = grouped[("policy_f1", "sum")].values
    paper_baseline_sums = grouped[("baseline_f1", "sum")].values
    paper_counts = grouped[("policy_f1", "count")].values

    if has_tokens:
        paper_policy_tok_sums = grouped[("policy_tokens", "sum")].values
        paper_baseline_tok_sums = grouped[("baseline_tokens", "sum")].values

    observed_delta_f1 = float(np.mean(policy_f1s) - np.mean(baseline_f1s))
    bootstrap_deltas_f1 = []
    bootstrap_deltas_tok = []

    for _ in range(n_bootstrap):
        sampled_indices = rng.choice(n_papers, size=n_papers, replace=True)
        tot_q = np.sum(paper_counts[sampled_indices])
        if tot_q > 0:
            tot_p_f1 = np.sum(paper_policy_sums[sampled_indices])
            tot_b_f1 = np.sum(paper_baseline_sums[sampled_indices])
            bootstrap_deltas_f1.append((tot_p_f1 - tot_b_f1) / tot_q)

            if has_tokens:
                tot_p_tok = np.sum(paper_policy_tok_sums[sampled_indices])
                tot_b_tok = np.sum(paper_baseline_tok_sums[sampled_indices])
                bootstrap_deltas_tok.append((tot_p_tok - tot_b_tok) / tot_q)

    boot_f1 = np.array(bootstrap_deltas_f1)
    ci_lower_f1 = float(np.percentile(boot_f1, 100.0 * (alpha / 2.0)))
    ci_upper_f1 = float(np.percentile(boot_f1, 100.0 * (1.0 - alpha / 2.0)))
    p_val_f1 = float(np.mean(boot_f1 <= 0.0) if observed_delta_f1 > 0 else np.mean(boot_f1 >= 0.0))

    out: Dict[str, Any] = {
        "observed_delta_f1": round(observed_delta_f1, 4),
        "ci_lower_95": round(ci_lower_f1, 4),
        "ci_upper_95": round(ci_upper_f1, 4),
        "p_value": round(p_val_f1, 4),
        "n_bootstrap": n_bootstrap,
    }

    if has_tokens:
        boot_tok = np.array(bootstrap_deltas_tok)
        observed_delta_tok = float(np.mean(policy_tokens) - np.mean(baseline_tokens))
        ci_lower_tok = float(np.percentile(boot_tok, 100.0 * (alpha / 2.0)))
        ci_upper_tok = float(np.percentile(boot_tok, 100.0 * (1.0 - alpha / 2.0)))
        out["observed_delta_tokens"] = round(observed_delta_tok, 1)
        out["tokens_ci_lower_95"] = round(ci_lower_tok, 1)
        out["tokens_ci_upper_95"] = round(ci_upper_tok, 1)

    return out
