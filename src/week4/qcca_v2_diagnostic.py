"""QCCA-V2 Diagnostic and Policy-Refinement Library.

Provides comprehensive diagnostic tooling to answer:
"Why does the learned sequential policy fail to convert feature signal into a policy that clearly improves over static-k baselines, and what policy structure is justified by the data?"

Key Analyses Implemented:
1. Stage-Wise Error Diagnosis:
   - Evaluates out-of-fold performance for transitions 2->4, 4->6, 6->8 across models.
   - Computes PR-AUC, ROC-AUC, Balanced Accuracy, Precision, Recall, Specificity, Brier score, and ECE.
   - Quantifies the downstream policy consequence of each error type (false expansion vs false stopping).
2. Error Propagation & Policy Structure Variations:
   - Compares Policy A (2->4->6->8), Policy B (start k=4->6->8), Policy C (start k=4, stop k=6), Policy D (conservative/high-precision S1).
   - Determines whether Stage 1 errors bottleneck the stronger Stage 2 signal.
3. Threshold Optimization & In-Fold Tuning:
   - Sweeps decision thresholds over t in [0.10, 0.90].
   - Performs strictly nested in-fold threshold selection (leak-free).
4. Cost-Sensitive & Expected Utility Decision Rules:
   - Evaluates expected gain, expected regret, and utility: U = Delta F1 - lambda * Delta Tokens.
5. Direct Gain Regression vs Binary Classification:
   - Compares Ridge/DecisionTree regression vs Logistic/DecisionTree classification.
6. Feature Importance & Ablation by Stage:
   - Evaluates individual feature families and leave-one-family-out (LOFO) per stage.
7. Compact Interpretable Subsets:
   - Tests whether top 1-3 features per stage match or exceed 14-feature models.
8. Probability Calibration:
   - Computes reliability diagrams, Brier score, ECE, and evaluates in-fold calibration.
9. Oracle Gap Analysis & Granular Failure Profiling:
   - Maps Epsilon Oracle -> Oracle Sequential -> Best Static -> Learned Policies.
   - Profiles failures into stopped too early, expanded too far, wrong S1/S2/S3.
10. Multi-Objective Pareto Frontier & Bootstrap Hypothesis Testing:
    - Identifies non-dominated systems and computes paper-clustered bootstrap 95% CIs.
"""

import ast
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.week4.qcca_v2 import (
    CONTEXT_TOKEN_COSTS,
    STAGE_1_FEATURES,
    STAGE_2_FEATURES,
    STAGE_3_FEATURES,
    PROVISIONAL_14_FEATURES,
    normalize_qid,
    construct_transition_targets,
    load_qcca_v2_data,
    build_classifier,
    compute_pareto_frontier,
)

logger = logging.getLogger(__name__)

# Feature family partitioning
FEATURE_FAMILIES = {
    "query_complexity": [
        "query_conjunction_count",
        "query_is_what_which",
        "query_is_numerical",
    ],
    "bm25_retrieval": [
        "bm25_mean_score",
        "bm25_first_gap_ratio",
    ],
    "semantic_similarity": [
        "sem_sim_mean_top6",
        "sem_sim_top1",
    ],
    "lexical_coverage": [
        "lex_coverage_top1",
        "lex_coverage_top2",
        "lex_coverage_top6",
        "lex_coverage_gain_1_to_4",
    ],
    "passage_diversity": [
        "passage_sim_std",
        "passage_cluster_count",
    ],
    "evidence_structure": [
        "evidence_query_cluster_span",
        "evidence_lexical_overlap_mean",
    ],
}


# =============================================================================
# 1. STAGE-WISE ERROR DIAGNOSIS & CALIBRATION METRICS
# =============================================================================

def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 5,
) -> Tuple[float, pd.DataFrame]:
    """Compute Expected Calibration Error (ECE) and binned reliability data."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    records = []
    ece = 0.0
    n = len(y_true)
    
    for b in range(n_bins):
        mask = bin_indices == b
        count = int(np.sum(mask))
        if count > 0:
            avg_prob = float(np.mean(y_prob[mask]))
            avg_acc = float(np.mean(y_true[mask]))
            bin_weight = count / n
            ece += bin_weight * abs(avg_prob - avg_acc)
            records.append({
                "bin_idx": b,
                "bin_low": bins[b],
                "bin_high": bins[b + 1],
                "bin_center": (bins[b] + bins[b + 1]) / 2.0,
                "count": count,
                "mean_pred_prob": round(avg_prob, 4),
                "empirical_accuracy": round(avg_acc, 4),
                "calibration_gap": round(avg_prob - avg_acc, 4),
            })
        else:
            records.append({
                "bin_idx": b,
                "bin_low": bins[b],
                "bin_high": bins[b + 1],
                "bin_center": (bins[b] + bins[b + 1]) / 2.0,
                "count": 0,
                "mean_pred_prob": round((bins[b] + bins[b + 1]) / 2.0, 4),
                "empirical_accuracy": 0.0,
                "calibration_gap": 0.0,
            })
            
    return float(ece), pd.DataFrame(records)


def train_eval_stage_oof_detailed(
    df_merged: pd.DataFrame,
    stage_name: str,
    target_col: str,
    feature_cols: List[str],
    model_name: str = "logistic_regression_balanced",
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate a stage transition model out-of-fold with comprehensive metrics.
    
    Computes:
      - ROC-AUC, PR-AUC, Balanced Accuracy, Precision, Recall, Specificity, F1
      - Brier score and Expected Calibration Error (ECE)
      - Confusion matrix (TP, FP, TN, FN)
      - Paper-disjoint GroupKFold fold-level mean and std
    """
    gkf = GroupKFold(n_splits=n_splits)
    groups = df_merged["paper_id"].values
    
    n = len(df_merged)
    oof_preds = np.zeros(n, dtype=int)
    oof_probs = np.zeros(n, dtype=float)
    fold_ids = np.zeros(n, dtype=int)
    
    y_true = df_merged[target_col].values.astype(int)
    X_df = df_merged[feature_cols].copy()
    
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_df, y_true, groups=groups)):
        X_train, y_train = X_df.iloc[train_idx], y_true[train_idx]
        X_val, y_val = X_df.iloc[val_idx], y_true[val_idx]
        
        fold_ids[val_idx] = fold
        model = build_classifier(model_name, random_state=seed + fold)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_val)
        oof_preds[val_idx] = preds
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_val)[:, 1]
        elif hasattr(model, "decision_function"):
            raw_scores = model.decision_function(X_val)
            probs = 1.0 / (1.0 + np.exp(-raw_scores))
        else:
            probs = preds.astype(float)
            
        oof_probs[val_idx] = probs
        
        # In-fold metrics
        try:
            val_roc = roc_auc_score(y_val, probs) if len(np.unique(y_val)) > 1 else 0.5
        except Exception:
            val_roc = 0.5
        try:
            val_pr = average_precision_score(y_val, probs) if len(np.unique(y_val)) > 1 else np.mean(y_val)
        except Exception:
            val_pr = np.mean(y_val)
            
        val_bal = balanced_accuracy_score(y_val, preds)
        val_f1 = f1_score(y_val, preds, zero_division=0)
        
        fold_metrics.append({
            "fold": fold,
            "roc_auc": val_roc,
            "pr_auc": val_pr,
            "balanced_acc": val_bal,
            "f1": val_f1,
        })
        
    # Global OOF metrics
    try:
        global_roc = float(roc_auc_score(y_true, oof_probs))
    except Exception:
        global_roc = 0.5
        
    try:
        global_pr = float(average_precision_score(y_true, oof_probs))
    except Exception:
        global_pr = float(np.mean(y_true))
        
    global_bal = float(balanced_accuracy_score(y_true, oof_preds))
    global_prec = float(precision_score(y_true, oof_preds, zero_division=0))
    global_rec = float(recall_score(y_true, oof_preds, zero_division=0))
    global_f1 = float(f1_score(y_true, oof_preds, zero_division=0))
    
    # Specificity = TN / (TN + FP)
    tn = int(np.sum((y_true == 0) & (oof_preds == 0)))
    fp = int(np.sum((y_true == 0) & (oof_preds == 1)))
    fn = int(np.sum((y_true == 1) & (oof_preds == 0)))
    tp = int(np.sum((y_true == 1) & (oof_preds == 1)))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    brier = float(brier_score_loss(y_true, oof_probs))
    ece, df_calib_bins = compute_expected_calibration_error(y_true, oof_probs, n_bins=5)
    
    fold_df = pd.DataFrame(fold_metrics)
    
    return {
        "stage": stage_name,
        "model_name": model_name,
        "n_samples": n,
        "prevalence_pos": float(np.mean(y_true)),
        "prevalence_neg": float(1.0 - np.mean(y_true)),
        "oof_roc_auc": round(global_roc, 4),
        "oof_pr_auc": round(global_pr, 4),
        "oof_balanced_acc": round(global_bal, 4),
        "oof_precision": round(global_prec, 4),
        "oof_recall": round(global_rec, 4),
        "oof_specificity": round(specificity, 4),
        "oof_f1": round(global_f1, 4),
        "oof_brier_score": round(brier, 4),
        "oof_ece": round(ece, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "fold_roc_mean": round(float(fold_df["roc_auc"].mean()), 4),
        "fold_roc_std": round(float(fold_df["roc_auc"].std()), 4),
        "fold_bal_mean": round(float(fold_df["balanced_acc"].mean()), 4),
        "fold_bal_std": round(float(fold_df["balanced_acc"].std()), 4),
        "oof_preds": oof_preds,
        "oof_probs": oof_probs,
        "fold_ids": fold_ids,
        "calibration_bins": df_calib_bins,
    }


def compute_stage_error_consequences(
    df_merged: pd.DataFrame,
    stage_name: str,
    target_col: str,
    gain_col: str,
    y_pred: np.ndarray,
    token_delta: float,
) -> pd.DataFrame:
    """Quantify the downstream policy consequence of each error type.
    
    Error Categories:
      - True Positive (TP): Beneficial expansion correctly executed.
      - False Positive (FP): False expansion. Context wasted (+token_delta), potential distractor loss.
      - False Negative (FN): False stopping. Beneficial evidence missed, answer F1 lost, regret incurred.
      - True Negative (TN): True stopping. Context saved, distractors correctly avoided.
    """
    y_true = df_merged[target_col].values.astype(int)
    gains = df_merged[gain_col].values.astype(float)
    
    n = len(df_merged)
    cats = np.empty(n, dtype=object)
    
    for i in range(n):
        if y_true[i] == 1 and y_pred[i] == 1:
            cats[i] = "TP (Beneficial Expansion)"
        elif y_true[i] == 0 and y_pred[i] == 1:
            cats[i] = "FP (False Expansion / Wasted Context)"
        elif y_true[i] == 1 and y_pred[i] == 0:
            cats[i] = "FN (False Stopping / Missed Gain)"
        else:
            cats[i] = "TN (True Stopping / Distractor Avoided)"
            
    df_diag = pd.DataFrame({
        "category": cats,
        "gain": gains,
    })
    
    records = []
    for cat_name in [
        "TP (Beneficial Expansion)",
        "FP (False Expansion / Wasted Context)",
        "FN (False Stopping / Missed Gain)",
        "TN (True Stopping / Distractor Avoided)",
    ]:
        sub = df_diag[df_diag["category"] == cat_name]
        cnt = len(sub)
        pct = (cnt / n) * 100.0 if n > 0 else 0.0
        mean_g = float(sub["gain"].mean()) if cnt > 0 else 0.0
        total_g = float(sub["gain"].sum()) if cnt > 0 else 0.0
        
        # Consequence descriptions
        if "TP" in cat_name:
            impact = f"Captured +{total_g:.2f} cumulative F1 gain; consumed +{token_delta:.1f} tokens per query."
        elif "FP" in cat_name:
            impact = f"Wasted +{token_delta:.1f} tokens per query; net marginal F1 change: {mean_g:+.4f}."
        elif "FN" in cat_name:
            impact = f"Missed opportunity cost: lost {total_g:.2f} cumulative F1; saved {token_delta:.1f} tokens."
        else:
            impact = f"Saved {token_delta:.1f} tokens per query; avoided net negative/neutral evidence."
            
        records.append({
            "stage": stage_name,
            "category": cat_name,
            "query_count": cnt,
            "percentage": round(pct, 2),
            "mean_marginal_gain": round(mean_g, 4),
            "total_marginal_gain": round(total_g, 4),
            "token_cost_delta": round(token_delta if ("TP" in cat_name or "FP" in cat_name) else 0.0, 1),
            "policy_impact": impact,
        })
        
    return pd.DataFrame(records)


# =============================================================================
# 2. ERROR PROPAGATION & SEQUENTIAL POLICY STRUCTURE VARIATIONS
# =============================================================================

def simulate_policy_structure(
    df_merged: pd.DataFrame,
    policy_type: str,
    s1_preds: np.ndarray,
    s2_preds: np.ndarray,
    s3_preds: np.ndarray,
    policy_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Simulate variations in sequential policy architecture.
    
    Supported policy structures:
      - 'A_full_cascade': 2 -> 4 -> 6 -> 8 (standard cascade)
      - 'B_start_k4': Always start at k=4; then apply Stage 2 (4->6) and Stage 3 (6->8)
      - 'C_start_k4_stop_k6': Always start at k=4; apply Stage 2 (4->6); stop at k=6 (never expand to 8)
      - 'C_start_k4_expand_k8': Always start at k=4; apply Stage 2 (4->6); if 6, always expand to 8
      - 'D_conservative_s1': 2 -> 4 -> 6 -> 8, with high-precision S1
      - 'D_aggressive_s1': 2 -> 4 -> 6 -> 8, with high-recall S1
    """
    n = len(df_merged)
    allocations = np.zeros(n, dtype=int)
    actual_f1s = np.zeros(n, dtype=float)
    actual_tokens = np.zeros(n, dtype=float)
    
    for i in range(n):
        if policy_type in ["A_full_cascade", "D_conservative_s1", "D_aggressive_s1"]:
            curr_k = 2
            if s1_preds[i] == 1:
                curr_k = 4
                if s2_preds[i] == 1:
                    curr_k = 6
                    if s3_preds[i] == 1:
                        curr_k = 8
        elif policy_type == "B_start_k4":
            curr_k = 4
            if s2_preds[i] == 1:
                curr_k = 6
                if s3_preds[i] == 1:
                    curr_k = 8
        elif policy_type == "C_start_k4_stop_k6":
            curr_k = 4
            if s2_preds[i] == 1:
                curr_k = 6  # Fixed stop at 6
        elif policy_type == "C_start_k4_expand_k8":
            curr_k = 4
            if s2_preds[i] == 1:
                curr_k = 8  # Jump directly to 8
        else:
            raise ValueError(f"Unknown policy_type: {policy_type}")
            
        allocations[i] = curr_k
        actual_f1s[i] = float(df_merged.iloc[i][f"F1_k{curr_k}"])
        actual_tokens[i] = CONTEXT_TOKEN_COSTS[curr_k]
        
    df_res = pd.DataFrame({
        "question_id": df_merged["question_id"],
        "paper_id": df_merged["paper_id"],
        "selected_k": allocations,
        "policy_f1": actual_f1s,
        "context_tokens": actual_tokens,
    })
    
    if "oracle_f1" in df_merged.columns:
        df_res["oracle_f1"] = df_merged["oracle_f1"].values
        df_res["regret"] = np.maximum(0.0, df_res["oracle_f1"] - df_res["policy_f1"])
    else:
        df_res["regret"] = 0.0
        
    mean_f1 = float(np.mean(actual_f1s))
    mean_k = float(np.mean(allocations))
    mean_tok = float(np.mean(actual_tokens))
    context_red_k8 = float((1758.5 - mean_tok) / 1758.5 * 100.0)
    eff = float(mean_f1 / (mean_tok / 1000.0))
    mean_reg = float(np.mean(df_res["regret"]))
    med_reg = float(np.median(df_res["regret"]))
    zero_reg_pct = float(np.mean(df_res["regret"] == 0.0) * 100.0)
    
    k_counts = pd.Series(allocations).value_counts(normalize=True).to_dict()
    
    name = policy_name or policy_type
    summary = {
        "policy": name,
        "mean_f1": round(mean_f1, 4),
        "mean_k": round(mean_k, 2),
        "mean_context_tokens": round(mean_tok, 1),
        "context_reduction_pct_vs_k8": round(context_red_k8, 2),
        "f1_per_1k_tokens": round(eff, 4),
        "mean_regret": round(mean_reg, 4),
        "median_regret": round(med_reg, 4),
        "fraction_zero_regret": round(zero_reg_pct, 2),
        "pct_k2": round(k_counts.get(2, 0.0) * 100.0, 1),
        "pct_k4": round(k_counts.get(4, 0.0) * 100.0, 1),
        "pct_k6": round(k_counts.get(6, 0.0) * 100.0, 1),
        "pct_k8": round(k_counts.get(8, 0.0) * 100.0, 1),
    }
    
    return df_res, summary


# =============================================================================
# 3. THRESHOLD OPTIMIZATION & IN-FOLD TUNING
# =============================================================================

def sweep_policy_thresholds(
    df_merged: pd.DataFrame,
    s1_probs: np.ndarray,
    s2_probs: np.ndarray,
    s3_probs: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    policy_base: str = "A_full_cascade",
) -> pd.DataFrame:
    """Sweep decision thresholds out-of-fold and compute downstream policy metrics."""
    if thresholds is None:
        thresholds = np.linspace(0.10, 0.90, 17)
        
    records = []
    for t in thresholds:
        s1_p = (s1_probs >= t).astype(int)
        s2_p = (s2_probs >= t).astype(int)
        s3_p = (s3_probs >= t).astype(int)
        
        _, summary = simulate_policy_structure(
            df_merged,
            policy_type=policy_base,
            s1_preds=s1_p,
            s2_preds=s2_p,
            s3_preds=s3_p,
            policy_name=f"Threshold_t={t:.2f}",
        )
        summary["threshold"] = round(float(t), 2)
        records.append(summary)
        
    return pd.DataFrame(records)


def nested_infold_threshold_selection(
    df_merged: pd.DataFrame,
    s1_probs: np.ndarray,
    s2_probs: np.ndarray,
    s3_probs: np.ndarray,
    fold_ids: np.ndarray,
    candidate_thresholds: Optional[np.ndarray] = None,
    objective: str = "utility",
    lambda_param: float = 0.0001,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    """Perform leak-free nested in-fold threshold selection.
    
    For each fold k in 0..4:
      - Search candidate thresholds on training folds (excluding k) to maximize target objective.
      - Apply the selected threshold t_k* exclusively to validation fold k.
    """
    if candidate_thresholds is None:
        candidate_thresholds = np.linspace(0.20, 0.80, 13)
        
    n = len(df_merged)
    final_allocations = np.zeros(n, dtype=int)
    final_f1s = np.zeros(n, dtype=float)
    final_tokens = np.zeros(n, dtype=float)
    
    unique_folds = np.unique(fold_ids)
    selected_thresholds = []
    
    for fold in unique_folds:
        val_mask = fold_ids == fold
        train_mask = ~val_mask
        
        df_train = df_merged[train_mask]
        s1_train, s2_train, s3_train = s1_probs[train_mask], s2_probs[train_mask], s3_probs[train_mask]
        
        best_t = 0.50
        best_score = -1e9
        
        for t in candidate_thresholds:
            p1 = (s1_train >= t).astype(int)
            p2 = (s2_train >= t).astype(int)
            p3 = (s3_train >= t).astype(int)
            
            _, sm = simulate_policy_structure(df_train, "A_full_cascade", p1, p2, p3)
            f1_val = sm["mean_f1"]
            tok_val = sm["mean_context_tokens"]
            
            if objective == "f1":
                score = f1_val
            elif objective == "utility":
                score = f1_val - lambda_param * tok_val
            elif objective == "efficiency":
                score = f1_val / (tok_val / 1000.0)
            else:
                score = f1_val
                
            if score > best_score:
                best_score = score
                best_t = t
                
        selected_thresholds.append({
            "val_fold": fold,
            "selected_threshold": round(float(best_t), 2),
            "train_best_score": round(float(best_score), 4),
        })
        
        # Apply selected threshold to validation fold
        df_val = df_merged[val_mask]
        s1_val = (s1_probs[val_mask] >= best_t).astype(int)
        s2_val = (s2_probs[val_mask] >= best_t).astype(int)
        s3_val = (s3_probs[val_mask] >= best_t).astype(int)
        
        df_val_res, _ = simulate_policy_structure(df_val, "A_full_cascade", s1_val, s2_val, s3_val)
        final_allocations[val_mask] = df_val_res["selected_k"].values
        final_f1s[val_mask] = df_val_res["policy_f1"].values
        final_tokens[val_mask] = df_val_res["context_tokens"].values
        
    df_res = pd.DataFrame({
        "question_id": df_merged["question_id"],
        "paper_id": df_merged["paper_id"],
        "selected_k": final_allocations,
        "policy_f1": final_f1s,
        "context_tokens": final_tokens,
    })
    
    if "oracle_f1" in df_merged.columns:
        df_res["oracle_f1"] = df_merged["oracle_f1"].values
        df_res["regret"] = np.maximum(0.0, df_res["oracle_f1"] - df_res["policy_f1"])
    else:
        df_res["regret"] = 0.0
        
    mean_f1 = float(np.mean(final_f1s))
    mean_k = float(np.mean(final_allocations))
    mean_tok = float(np.mean(final_tokens))
    context_red_k8 = float((1758.5 - mean_tok) / 1758.5 * 100.0)
    eff = float(mean_f1 / (mean_tok / 1000.0))
    mean_reg = float(np.mean(df_res["regret"]))
    
    k_counts = pd.Series(final_allocations).value_counts(normalize=True).to_dict()
    
    summary = {
        "policy": f"Nested_Tuned_OOF_{objective}",
        "mean_f1": round(mean_f1, 4),
        "mean_k": round(mean_k, 2),
        "mean_context_tokens": round(mean_tok, 1),
        "context_reduction_pct_vs_k8": round(context_red_k8, 2),
        "f1_per_1k_tokens": round(eff, 4),
        "mean_regret": round(mean_reg, 4),
        "median_regret": round(float(np.median(df_res["regret"])), 4),
        "pct_k2": round(k_counts.get(2, 0.0) * 100.0, 1),
        "pct_k4": round(k_counts.get(4, 0.0) * 100.0, 1),
        "pct_k6": round(k_counts.get(6, 0.0) * 100.0, 1),
        "pct_k8": round(k_counts.get(8, 0.0) * 100.0, 1),
    }
    
    return df_res, summary, pd.DataFrame(selected_thresholds)


# =============================================================================
# 4. COST-SENSITIVE & EXPECTED UTILITY DECISION RULES
# =============================================================================

def evaluate_cost_sensitive_rules(
    df_merged: pd.DataFrame,
    s1_probs: np.ndarray,
    s2_probs: np.ndarray,
    s3_probs: np.ndarray,
    s1_reg_preds: np.ndarray,
    s2_reg_preds: np.ndarray,
    s3_reg_preds: np.ndarray,
    lambda_values: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Evaluate decision rules under expected utility: U = predicted_gain - lambda * delta_tokens."""
    if lambda_values is None:
        lambda_values = [0.0, 1e-5, 2.5e-5, 5e-5, 1e-4, 2e-4, 5e-4]
        
    records = []
    
    # Baseline delta tokens
    tok_s1 = CONTEXT_TOKEN_COSTS[4] - CONTEXT_TOKEN_COSTS[2]  # 411.6
    tok_s2 = CONTEXT_TOKEN_COSTS[6] - CONTEXT_TOKEN_COSTS[4]  # 413.0
    tok_s3 = CONTEXT_TOKEN_COSTS[8] - CONTEXT_TOKEN_COSTS[6]  # 399.4
    
    # 1. Expected Gain via probability * mean historical positive gain
    pos_g1 = df_merged[df_merged["Y_2_4"] == 1]["G_2_4"].mean()
    pos_g2 = df_merged[df_merged["Y_4_6"] == 1]["G_4_6"].mean()
    pos_g3 = df_merged[df_merged["Y_6_8"] == 1]["G_6_8"].mean()
    
    exp_g1 = s1_probs * pos_g1
    exp_g2 = s2_probs * pos_g2
    exp_g3 = s3_probs * pos_g3
    
    # Pure expected gain > delta (0.01)
    p1 = (exp_g1 > 0.01).astype(int)
    p2 = (exp_g2 > 0.01).astype(int)
    p3 = (exp_g3 > 0.01).astype(int)
    _, sm = simulate_policy_structure(df_merged, "A_full_cascade", p1, p2, p3, policy_name="Expected_Gain_Threshold_0.01")
    records.append(sm)
    
    # 2. Continuous Gain Regression (Ridge) with utility sweep
    for lam in lambda_values:
        u1 = s1_reg_preds - (lam * tok_s1)
        u2 = s2_reg_preds - (lam * tok_s2)
        u3 = s3_reg_preds - (lam * tok_s3)
        
        pr1 = (u1 > 0.0).astype(int)
        pr2 = (u2 > 0.0).astype(int)
        pr3 = (u3 > 0.0).astype(int)
        
        _, sm = simulate_policy_structure(
            df_merged,
            "A_full_cascade",
            pr1,
            pr2,
            pr3,
            policy_name=f"Utility_Ridge_lambda={lam:.1e}",
        )
        records.append(sm)
        
    return pd.DataFrame(records)


# =============================================================================
# 5. DIRECT GAIN REGRESSION VS CLASSIFICATION
# =============================================================================

def train_eval_stage_regressors_oof(
    df_merged: pd.DataFrame,
    stage_name: str,
    gain_col: str,
    feature_cols: List[str],
    delta: float = 0.01,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train Ridge regressor on continuous marginal gains out-of-fold."""
    gkf = GroupKFold(n_splits=n_splits)
    groups = df_merged["paper_id"].values
    
    n = len(df_merged)
    oof_preds = np.zeros(n, dtype=float)
    y_true = df_merged[gain_col].values.astype(float)
    X_df = df_merged[feature_cols].copy()
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_df, y_true, groups=groups)):
        X_train, y_train = X_df.iloc[train_idx], y_true[train_idx]
        X_val = X_df.iloc[val_idx]
        
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0, random_state=seed + fold))
        ])
        pipe.fit(X_train, y_train)
        oof_preds[val_idx] = pipe.predict(X_val)
        
    mae = float(mean_absolute_error(y_true, oof_preds))
    r2 = float(r2_score(y_true, oof_preds))
    spearman_rho, spearman_p = stats.spearmanr(y_true, oof_preds)
    
    # Derived binary classification at delta
    y_bin_true = (y_true > delta).astype(int)
    y_bin_pred = (oof_preds > delta).astype(int)
    derived_bal_acc = float(balanced_accuracy_score(y_bin_true, y_bin_pred))
    
    try:
        derived_roc_auc = float(roc_auc_score(y_bin_true, oof_preds))
    except Exception:
        derived_roc_auc = 0.5
        
    return {
        "stage": stage_name,
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "spearman_rho": round(float(spearman_rho), 4),
        "spearman_p": round(float(spearman_p), 4),
        "derived_bal_acc": round(derived_bal_acc, 4),
        "derived_roc_auc": round(derived_roc_auc, 4),
        "oof_continuous_preds": oof_preds,
    }


# =============================================================================
# 6. FEATURE IMPORTANCE & ABLATION BY STAGE
# =============================================================================

def run_stage_feature_family_ablations(
    df_merged: pd.DataFrame,
    stage_name: str,
    target_col: str,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Run family-level ablations independently for a single transition stage."""
    records = []
    
    # 1. Full Provisional Set
    res_full = train_eval_stage_oof_detailed(
        df_merged,
        stage_name,
        target_col,
        PROVISIONAL_14_FEATURES,
        model_name="logistic_regression_balanced",
        n_splits=n_splits,
        seed=seed,
    )
    records.append({
        "stage": stage_name,
        "ablation_type": "Full Provisional Set (14)",
        "features_used": len(PROVISIONAL_14_FEATURES),
        "roc_auc": res_full["oof_roc_auc"],
        "pr_auc": res_full["oof_pr_auc"],
        "balanced_acc": res_full["oof_balanced_acc"],
        "brier_score": res_full["oof_brier_score"],
    })
    
    # 2. Individual Families
    for fam_name, fam_feats in FEATURE_FAMILIES.items():
        res_fam = train_eval_stage_oof_detailed(
            df_merged,
            stage_name,
            target_col,
            fam_feats,
            model_name="logistic_regression_balanced",
            n_splits=n_splits,
            seed=seed,
        )
        records.append({
            "stage": stage_name,
            "ablation_type": f"Only {fam_name}",
            "features_used": len(fam_feats),
            "roc_auc": res_fam["oof_roc_auc"],
            "pr_auc": res_fam["oof_pr_auc"],
            "balanced_acc": res_fam["oof_balanced_acc"],
            "brier_score": res_fam["oof_brier_score"],
        })
        
    # 3. Leave-One-Family-Out (LOFO)
    for fam_name, fam_feats in FEATURE_FAMILIES.items():
        lofo_feats = [f for f in PROVISIONAL_14_FEATURES if f not in fam_feats]
        res_lofo = train_eval_stage_oof_detailed(
            df_merged,
            stage_name,
            target_col,
            lofo_feats,
            model_name="logistic_regression_balanced",
            n_splits=n_splits,
            seed=seed,
        )
        records.append({
            "stage": stage_name,
            "ablation_type": f"LOFO (Drop {fam_name})",
            "features_used": len(lofo_feats),
            "roc_auc": res_lofo["oof_roc_auc"],
            "pr_auc": res_lofo["oof_pr_auc"],
            "balanced_acc": res_lofo["oof_balanced_acc"],
            "brier_score": res_lofo["oof_brier_score"],
        })
        
    return pd.DataFrame(records)


# =============================================================================
# 7. COMPACT INTERPRETABLE FEATURE SUBSETS
# =============================================================================

def evaluate_compact_subsets(
    df_merged: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compare full 14-feature set against compact top-k feature subsets."""
    # Subset definitions:
    subsets = {
        "Full_14_Features": (
            PROVISIONAL_14_FEATURES,
            PROVISIONAL_14_FEATURES,
            PROVISIONAL_14_FEATURES,
        ),
        "Hypothesis_Stage_Specific": (
            STAGE_1_FEATURES,
            STAGE_2_FEATURES,
            STAGE_3_FEATURES,
        ),
        "Top_3_Per_Stage": (
            ["query_conjunction_count", "query_is_what_which", "lex_coverage_top1"],
            ["sem_sim_mean_top6", "bm25_mean_score", "lex_coverage_top2"],
            ["lex_coverage_top2", "passage_sim_std", "lex_coverage_top6"],
        ),
        "Top_2_Per_Stage": (
            ["query_conjunction_count", "lex_coverage_top1"],
            ["sem_sim_mean_top6", "lex_coverage_top2"],
            ["lex_coverage_top2", "passage_sim_std"],
        ),
        "Top_1_Per_Stage": (
            ["query_conjunction_count"],
            ["sem_sim_mean_top6"],
            ["lex_coverage_top2"],
        ),
    }
    
    records = []
    
    for sub_name, (f1_list, f2_list, f3_list) in subsets.items():
        s1_res = train_eval_stage_oof_detailed(
            df_merged, "Stage 1", "Y_2_4", f1_list, "logistic_regression_balanced", n_splits, seed
        )
        s2_res = train_eval_stage_oof_detailed(
            df_merged, "Stage 2", "Y_4_6", f2_list, "logistic_regression_balanced", n_splits, seed
        )
        s3_res = train_eval_stage_oof_detailed(
            df_merged, "Stage 3", "Y_6_8", f3_list, "logistic_regression_balanced", n_splits, seed
        )
        
        _, summary = simulate_policy_structure(
            df_merged,
            "A_full_cascade",
            s1_res["oof_preds"],
            s2_res["oof_preds"],
            s3_res["oof_preds"],
            policy_name=sub_name,
        )
        
        summary["s1_bal_acc"] = s1_res["oof_balanced_acc"]
        summary["s2_bal_acc"] = s2_res["oof_balanced_acc"]
        summary["s3_bal_acc"] = s3_res["oof_balanced_acc"]
        records.append(summary)
        
    return pd.DataFrame(records)


# =============================================================================
# 8. ORACLE GAP ANALYSIS & FAILURE PROFILING
# =============================================================================

def conduct_detailed_oracle_gap_analysis(
    df_merged: pd.DataFrame,
    policy_df: pd.DataFrame,
    s1_preds: np.ndarray,
    s2_preds: np.ndarray,
    s3_preds: np.ndarray,
    s1_probs: np.ndarray,
    s2_probs: np.ndarray,
    s3_probs: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform granular per-query failure audit relative to the Epsilon Oracle."""
    n = len(df_merged)
    
    oracle_k = df_merged["oracle_k"].values.astype(int)
    oracle_f1 = df_merged["oracle_f1"].values.astype(float)
    policy_k = policy_df["selected_k"].values.astype(int)
    policy_f1 = policy_df["policy_f1"].values.astype(float)
    regret = policy_df["regret"].values.astype(float)
    
    y_true_s1 = df_merged["Y_2_4"].values.astype(int)
    y_true_s2 = df_merged["Y_4_6"].values.astype(int)
    y_true_s3 = df_merged["Y_6_8"].values.astype(int)
    
    failure_types = []
    
    for i in range(n):
        pk = policy_k[i]
        ok = oracle_k[i]
        
        if pk == ok:
            ft = "Exact Match"
        elif pk < ok:
            # Under-expansion
            if pk == 2 and y_true_s1[i] == 1:
                ft = "Stage 1 Premature Stop (Under-Expansion)"
            elif pk == 4 and y_true_s2[i] == 1:
                ft = "Stage 2 Premature Stop (Under-Expansion)"
            elif pk == 6 and y_true_s3[i] == 1:
                ft = "Stage 3 Premature Stop (Under-Expansion)"
            else:
                ft = "Other Under-Expansion"
        else:
            # Over-expansion
            if pk == 4 and y_true_s1[i] == 0:
                ft = "Stage 1 Over-Expansion (Distractor/Waste)"
            elif pk == 6 and y_true_s2[i] == 0:
                ft = "Stage 2 Over-Expansion (Distractor/Waste)"
            elif pk == 8 and y_true_s3[i] == 0:
                ft = "Stage 3 Over-Expansion (Distractor/Waste)"
            else:
                ft = "Other Over-Expansion"
                
        failure_types.append(ft)
        
    df_audit = pd.DataFrame({
        "question_id": df_merged["question_id"],
        "paper_id": df_merged["paper_id"],
        "oracle_k": oracle_k,
        "policy_k": policy_k,
        "oracle_f1": oracle_f1,
        "policy_f1": policy_f1,
        "regret": regret,
        "failure_type": failure_types,
        "s1_decision": s1_preds,
        "s2_decision": s2_preds,
        "s3_decision": s3_preds,
        "s1_prob": np.round(s1_probs, 4),
        "s2_prob": np.round(s2_probs, 4),
        "s3_prob": np.round(s3_probs, 4),
        "s1_true": y_true_s1,
        "s2_true": y_true_s2,
        "s3_true": y_true_s3,
    })
    
    # Summary of failure categories
    records = []
    for ft, grp in df_audit.groupby("failure_type"):
        cnt = len(grp)
        pct = (cnt / n) * 100.0
        mean_reg = float(grp["regret"].mean())
        tot_reg = float(grp["regret"].sum())
        mean_f1_gap = float((grp["oracle_f1"] - grp["policy_f1"]).mean())
        
        records.append({
            "failure_type": ft,
            "query_count": cnt,
            "percentage": round(pct, 2),
            "mean_regret": round(mean_reg, 4),
            "total_regret": round(tot_reg, 4),
            "mean_f1_gap": round(mean_f1_gap, 4),
        })
        
    df_summary = pd.DataFrame(records).sort_values(by="total_regret", ascending=False)
    return df_audit, df_summary


# =============================================================================
# 9. PAIRED BOOTSTRAP HYPOTHESIS TESTING
# =============================================================================

def run_policy_paired_bootstrap(
    df_merged: pd.DataFrame,
    policy_dfs: Dict[str, pd.DataFrame],
    baseline_ks: List[int] = [4, 6, 8],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute paper-clustered paired bootstrap differences vs static baselines."""
    rng = np.random.RandomState(seed)
    papers = df_merged["paper_id"].unique()
    n_papers = len(papers)
    
    records = []
    
    for pol_name, p_df in policy_dfs.items():
        pol_f1 = p_df["policy_f1"].values
        pol_tok = p_df["context_tokens"].values
        
        for k_base in baseline_ks:
            base_f1 = df_merged[f"F1_k{k_base}"].values.astype(float)
            base_tok = np.full(len(df_merged), CONTEXT_TOKEN_COSTS[k_base])
            
            diff_f1 = pol_f1 - base_f1
            diff_tok = pol_tok - base_tok
            
            boot_f1_means = np.zeros(n_bootstrap, dtype=float)
            boot_tok_means = np.zeros(n_bootstrap, dtype=float)
            
            # Group by paper indices
            paper_to_idx = {p: np.where(df_merged["paper_id"].values == p)[0] for p in papers}
            
            for b in range(n_bootstrap):
                sampled_papers = rng.choice(papers, size=n_papers, replace=True)
                sample_idx = np.concatenate([paper_to_idx[p] for p in sampled_papers])
                boot_f1_means[b] = np.mean(diff_f1[sample_idx])
                boot_tok_means[b] = np.mean(diff_tok[sample_idx])
                
            ci_f1_low, ci_f1_high = np.percentile(boot_f1_means, [2.5, 97.5])
            ci_tok_low, ci_tok_high = np.percentile(boot_tok_means, [2.5, 97.5])
            
            excludes_zero_f1 = bool((ci_f1_low > 0.0) or (ci_f1_high < 0.0))
            
            records.append({
                "policy": pol_name,
                "baseline": f"Static k={k_base}",
                "mean_delta_f1": round(float(np.mean(diff_f1)), 4),
                "ci_95_f1_low": round(float(ci_f1_low), 4),
                "ci_95_f1_high": round(float(ci_f1_high), 4),
                "significant_f1_diff": excludes_zero_f1,
                "mean_delta_tokens": round(float(np.mean(diff_tok)), 1),
                "ci_95_tok_low": round(float(ci_tok_low), 1),
                "ci_95_tok_high": round(float(ci_tok_high), 1),
            })
            
    return pd.DataFrame(records)


# =============================================================================
# 10. PUBLICATION-GRADE PLOTTING SUITE
# =============================================================================

def plot_stage_roc_pr_curves(
    stage_results: Dict[str, Dict[str, Any]],
    df_merged: pd.DataFrame,
    save_path: str = "results/week4/qcca_v2_diagnostic/figures/stage_roc_pr_curves.png",
):
    """Plot ROC and PR curves for each stage."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = {"Stage 1 (2->4)": "#1f77b4", "Stage 2 (4->6)": "#2ca02c", "Stage 3 (6->8)": "#d62728"}
    target_cols = {"Stage 1 (2->4)": "Y_2_4", "Stage 2 (4->6)": "Y_4_6", "Stage 3 (6->8)": "Y_6_8"}
    
    for s_name, res in stage_results.items():
        y_true = df_merged[target_cols[s_name]].values.astype(int)
        y_prob = res["oof_probs"]
        
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        axes[0].plot(fpr, tpr, label=f"{s_name} (AUC = {res['oof_roc_auc']:.3f})", color=colors[s_name], lw=2)
        
        # PR
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        axes[1].plot(rec, prec, label=f"{s_name} (PR-AUC = {res['oof_pr_auc']:.3f})", color=colors[s_name], lw=2)
        
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance (AUC = 0.500)")
    axes[0].set_title("ROC Curves Across Sequential Stages (Paper-Disjoint OOF)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate", fontsize=11)
    axes[0].set_ylabel("True Positive Rate", fontsize=11)
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title("Precision-Recall Curves Across Sequential Stages", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Recall", fontsize=11)
    axes[1].set_ylabel("Precision", fontsize=11)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_calibration_curves(
    stage_results: Dict[str, Dict[str, Any]],
    save_path: str = "results/week4/qcca_v2_diagnostic/figures/calibration_curves.png",
):
    """Plot reliability diagrams across stages."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for idx, (s_name, res) in enumerate(stage_results.items()):
        ax = axes[idx]
        df_cal = res["calibration_bins"]
        
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect Calibration")
        ax.plot(
            df_cal["mean_pred_prob"],
            df_cal["empirical_accuracy"],
            marker="o",
            lw=2,
            color="#1f77b4" if idx == 0 else ("#2ca02c" if idx == 1 else "#d62728"),
            label=f"Brier = {res['oof_brier_score']:.3f}\nECE = {res['oof_ece']:.3f}",
        )
        ax.set_title(f"{s_name} Calibration", fontsize=12, fontweight="bold")
        ax.set_xlabel("Mean Predicted Probability", fontsize=10)
        ax.set_ylabel("Observed Frequency", fontsize=10)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_threshold_curves(
    df_thresh: pd.DataFrame,
    save_f1_path: str = "results/week4/qcca_v2_diagnostic/figures/threshold_vs_f1.png",
    save_cost_path: str = "results/week4/qcca_v2_diagnostic/figures/threshold_vs_token_cost.png",
):
    """Plot threshold vs F1 and threshold vs token cost."""
    # F1
    plt.figure(figsize=(9, 5))
    plt.plot(df_thresh["threshold"], df_thresh["mean_f1"], marker="o", color="#2b5c8f", lw=2, label="Policy Mean F1")
    plt.axhline(0.3950, color="red", linestyle="--", alpha=0.7, label="Static k=8 (0.3950)")
    plt.axhline(0.3596, color="orange", linestyle="--", alpha=0.7, label="Static k=4 (0.3596)")
    plt.axhline(0.3038, color="gray", linestyle="--", alpha=0.7, label="Static k=2 (0.3038)")
    plt.title("Threshold vs Downstream Answer F1 (Paper-Disjoint OOF)", fontsize=13, fontweight="bold")
    plt.xlabel("Probability Threshold (t)", fontsize=11)
    plt.ylabel("Mean Answer F1", fontsize=11)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(save_f1_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_f1_path, dpi=300)
    plt.close()
    
    # Cost
    plt.figure(figsize=(9, 5))
    plt.plot(df_thresh["threshold"], df_thresh["mean_context_tokens"], marker="s", color="#8c2d19", lw=2, label="Mean Context Tokens")
    plt.axhline(1758.5, color="red", linestyle="--", alpha=0.7, label="Static k=8 (1758.5)")
    plt.axhline(946.1, color="orange", linestyle="--", alpha=0.7, label="Static k=4 (946.1)")
    plt.axhline(534.5, color="gray", linestyle="--", alpha=0.7, label="Static k=2 (534.5)")
    plt.title("Threshold vs Context Token Cost", fontsize=13, fontweight="bold")
    plt.xlabel("Probability Threshold (t)", fontsize=11)
    plt.ylabel("Mean Context Tokens", fontsize=11)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(save_cost_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_cost_path, dpi=300)
    plt.close()


def plot_policy_pareto_frontier(
    df_pareto: pd.DataFrame,
    save_path: str = "results/week4/qcca_v2_diagnostic/figures/policy_pareto_frontier.png",
):
    """Plot multi-objective Pareto frontier showing policies, baselines, and oracles."""
    plt.figure(figsize=(11, 7))
    
    # Plot points
    for _, row in df_pareto.iterrows():
        sys_name = row["system"]
        x = row["mean_context_tokens"]
        y = row["mean_f1"]
        is_opt = row.get("is_pareto_optimal", False)
        
        if "Oracle" in sys_name:
            marker = "*"
            size = 200
            color = "#d62728"
        elif "Static" in sys_name:
            marker = "s"
            size = 120
            color = "#4c72b0"
        else:
            marker = "o"
            size = 140
            color = "#2ca02c"
            
        plt.scatter(x, y, s=size, marker=marker, color=color, alpha=0.85, edgecolors="black", zorder=4)
        
        offset_y = 0.005 if is_opt else -0.008
        plt.annotate(
            sys_name,
            (x, y),
            textcoords="offset points",
            xytext=(0, 8 if is_opt else -12),
            ha="center",
            fontsize=9,
            fontweight="bold" if is_opt else "normal",
        )
        
    plt.title("QCCA-V2 Diagnostic Multi-Objective Pareto Frontier", fontsize=13, fontweight="bold")
    plt.xlabel("Mean Context Tokens (Cost)", fontsize=11)
    plt.ylabel("Mean Answer F1 (Quality)", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_oracle_gap_and_failures(
    df_fail_summary: pd.DataFrame,
    save_path: str = "results/week4/qcca_v2_diagnostic/figures/oracle_gap_breakdown.png",
):
    """Plot failure category proportions and total regret contributions."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.barplot(data=df_fail_summary, y="failure_type", x="percentage", ax=axes[0], palette="Blues_r")
    axes[0].set_title("Failure Category Frequency (%)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Percentage of Development Queries (%)", fontsize=11)
    axes[0].set_ylabel("")
    axes[0].grid(True, alpha=0.3)
    
    sns.barplot(data=df_fail_summary, y="failure_type", x="total_regret", ax=axes[1], palette="Reds_r")
    axes[1].set_title("Cumulative Regret Contribution", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Total Regret (Sum of Oracle F1 Deficit)", fontsize=11)
    axes[1].set_ylabel("")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_error_propagation_comparison(
    df_policies: pd.DataFrame,
    save_path: str = "results/week4/qcca_v2_diagnostic/figures/error_propagation.png",
):
    """Plot comparison across policy architectures (A vs B vs C vs D)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    sns.barplot(data=df_policies, x="policy", y="mean_f1", ax=axes[0], palette="mako")
    axes[0].set_title("Mean Answer F1", fontsize=12, fontweight="bold")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=35, ha="right")
    axes[0].set_xlabel("")
    axes[0].grid(True, alpha=0.3)
    
    sns.barplot(data=df_policies, x="policy", y="mean_context_tokens", ax=axes[1], palette="rocket")
    axes[1].set_title("Mean Context Tokens", fontsize=12, fontweight="bold")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=35, ha="right")
    axes[1].set_xlabel("")
    axes[1].grid(True, alpha=0.3)
    
    sns.barplot(data=df_policies, x="policy", y="mean_regret", ax=axes[2], palette="crest")
    axes[2].set_title("Mean Regret vs Oracle", fontsize=12, fontweight="bold")
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=35, ha="right")
    axes[2].set_xlabel("")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_budget_distributions(
    df_policies: pd.DataFrame,
    save_path: str = "results/week4/qcca_v2_diagnostic/figures/budget_distributions.png",
):
    """Plot stacked budget allocations across candidate policies."""
    plt.figure(figsize=(12, 6))
    
    pols = df_policies["policy"].values
    k2 = df_policies["pct_k2"].values
    k4 = df_policies["pct_k4"].values
    k6 = df_policies["pct_k6"].values
    k8 = df_policies["pct_k8"].values
    
    x = np.arange(len(pols))
    width = 0.55
    
    plt.bar(x, k2, width, label="k=2", color="#aec7e8")
    plt.bar(x, k4, width, bottom=k2, label="k=4", color="#1f77b4")
    plt.bar(x, k6, width, bottom=k2 + k4, label="k=6", color="#ff7f0e")
    plt.bar(x, k8, width, bottom=k2 + k4 + k6, label="k=8", color="#d62728")
    
    plt.xticks(x, pols, rotation=35, ha="right", fontsize=10)
    plt.ylabel("Percentage of Queries Allocated (%)", fontsize=11)
    plt.title("Budget Distribution (% k=2, 4, 6, 8) Across Policies", fontsize=13, fontweight="bold")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
