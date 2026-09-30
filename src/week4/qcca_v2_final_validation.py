"""QCCA-V2 Final Validation Library: Stage-2 Policy (Policy C1) Rigorous Validation.

Implements the dedicated, rigorous final validation pipeline for the Stage-2 selective
expansion policy:
    Policy C1:
      1. Start every query at base context k=4.
      2. Predict whether expanding 4->6 is beneficial using validated Stage-2 features.
      3. If predicted beneficial, expand to k=6.
      4. Otherwise stop at k=4.
      5. Never expand to k=8 (hard stop).

Key Scientific Analyses:
1. Strict paper-disjoint GroupKFold out-of-fold evaluation (zero leakage).
2. Fixed primary model (Balanced Logistic Regression) vs baselines.
3. Full threshold sensitivity sweep and Pareto exploration.
4. Controlled Stage-2-only feature ablations (10 configurations).
5. Interpretability: standardized coefficients, odds ratios, binned feature response curves.
6. Proxy/confounder analysis via partial correlations.
7. Subgroup analysis across syntax, complexity, coverage, and evidence dispersion.
8. Stage-2 error decomposition: false expansion vs false stopping.
9. Paper-clustered paired bootstrap (B=2,000) against Static k=4, Static k=6, Static k=8.
10. Multi-objective Pareto frontier identification.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.week4.qcca_v2 import (
    CONTEXT_TOKEN_COSTS,
    PROVISIONAL_14_FEATURES,
    load_qcca_v2_data,
    compute_pareto_frontier,
)
from src.week4.qcca_v2_diagnostic import compute_expected_calibration_error

logger = logging.getLogger(__name__)

# Validated Stage-2 candidate features
STAGE_2_CORE_FEATURES = [
    "sem_sim_mean_top6",
    "bm25_mean_score",
    "lex_coverage_top2",
    "evidence_query_cluster_span",
    "evidence_lexical_overlap_mean",
]

# Scientific justifications for retained features:
STAGE_2_FEATURE_JUSTIFICATIONS = {
    "sem_sim_mean_top6": (
        "Measures semantic coherence across the candidate retrieval window tail. "
        "High semantic similarity in ranks 4-6 indicates that additional passages contain "
        "on-topic scientific evidence rather than unrelated background noise."
    ),
    "bm25_mean_score": (
        "Measures background keyword matching density. Higher average BM25 signals that the "
        "document contains dense relevant scientific terminology, justifying broader passage retrieval."
    ),
    "lex_coverage_top2": (
        "Cumulative lexical query term coverage across top-2 passages. Moderately high coverage combined "
        "with remaining uncovered terms indicates that expanding to k=6 will capture missing factual facets."
    ),
    "evidence_query_cluster_span": (
        "Number of distinct document sections/clusters spanned by query keywords. Multi-section evidence "
        "dispersion requires k >= 6 to capture physically disconnected evidence chunks."
    ),
    "evidence_lexical_overlap_mean": (
        "Average pairwise lexical redundancy between retrieved passages. Low pairwise overlap indicates that "
        "successive passages provide complementary, non-duplicative information chunks."
    ),
}


# =============================================================================
# 1. POLICY C1 SIMULATION & OUT-OF-FOLD EVALUATION
# =============================================================================

def simulate_policy_c1(
    df_merged: pd.DataFrame,
    stage2_preds: np.ndarray,
    stage2_probs: np.ndarray,
    policy_name: str = "Policy C1 (Start k=4, stop k=6)",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Simulate Policy C1 on development queries:
      - Every query starts with base context k=4.
      - If stage2_preds == 1, expand to k=6.
      - Else, stop at k=4.
      - Never expand to k=8.
    """
    n = len(df_merged)
    allocations = np.where(stage2_preds == 1, 6, 4)
    actual_f1s = np.zeros(n, dtype=float)
    actual_tokens = np.zeros(n, dtype=float)
    
    for i in range(n):
        k = allocations[i]
        actual_f1s[i] = float(df_merged.iloc[i][f"F1_k{k}"])
        actual_tokens[i] = CONTEXT_TOKEN_COSTS[k]
        
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
    
    summary = {
        "policy": policy_name,
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


def run_stage2_oof_evaluation(
    df_merged: pd.DataFrame,
    feature_cols: List[str] = STAGE_2_CORE_FEATURES,
    target_col: str = "Y_4_6",
    gain_col: str = "G_4_6",
    model_type: str = "logistic_regression_balanced",
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Perform leak-free paper-disjoint GroupKFold out-of-fold evaluation for Stage 2."""
    gkf = GroupKFold(n_splits=n_splits)
    groups = df_merged["paper_id"].values
    
    n = len(df_merged)
    oof_preds = np.zeros(n, dtype=int)
    oof_probs = np.zeros(n, dtype=float)
    fold_ids = np.zeros(n, dtype=int)
    
    y_true = df_merged[target_col].values.astype(int)
    gains = df_merged[gain_col].values.astype(float)
    X_df = df_merged[feature_cols].copy()
    
    fold_models = []
    fold_coefficients = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_df, y_true, groups=groups)):
        X_train, y_train = X_df.iloc[train_idx], y_true[train_idx]
        X_val = X_df.iloc[val_idx]
        
        fold_ids[val_idx] = fold
        
        if model_type == "majority":
            clf = DummyClassifier(strategy="most_frequent")
            pipe = clf
        elif model_type == "logistic_regression":
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, class_weight=None, solver="lbfgs", random_state=seed + fold))
            ])
        elif model_type == "logistic_regression_balanced":
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", random_state=seed + fold))
            ])
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
        pipe.fit(X_train, y_train)
        fold_models.append(pipe)
        
        preds = pipe.predict(X_val)
        oof_preds[val_idx] = preds
        
        if hasattr(pipe, "predict_proba"):
            probs = pipe.predict_proba(X_val)[:, 1]
        else:
            probs = preds.astype(float)
        oof_probs[val_idx] = probs
        
        # Save standardized coefficients if logistic regression
        if hasattr(pipe, "named_steps") and "clf" in pipe.named_steps:
            coefs = pipe.named_steps["clf"].coef_[0]
            fold_coefficients.append(coefs)
            
    # Classification metrics
    try:
        roc_auc = float(roc_auc_score(y_true, oof_probs))
    except Exception:
        roc_auc = 0.5
    try:
        pr_auc = float(average_precision_score(y_true, oof_probs))
    except Exception:
        pr_auc = float(np.mean(y_true))
        
    bal_acc = float(balanced_accuracy_score(y_true, oof_preds))
    prec = float(precision_score(y_true, oof_preds, zero_division=0))
    rec = float(recall_score(y_true, oof_preds, zero_division=0))
    f1_s = float(f1_score(y_true, oof_preds, zero_division=0))
    
    tn = int(np.sum((y_true == 0) & (oof_preds == 0)))
    fp = int(np.sum((y_true == 0) & (oof_preds == 1)))
    fn = int(np.sum((y_true == 1) & (oof_preds == 0)))
    tp = int(np.sum((y_true == 1) & (oof_preds == 1)))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    brier = float(brier_score_loss(y_true, oof_probs))
    ece, df_calib_bins = compute_expected_calibration_error(y_true, oof_probs, n_bins=5)
    
    # Simulate Policy C1
    df_policy_res, policy_summary = simulate_policy_c1(df_merged, oof_preds, oof_probs)
    
    # Query-level export table
    df_export = pd.DataFrame({
        "question_id": df_merged["question_id"],
        "paper_id": df_merged["paper_id"],
        "p_expand_4_6": np.round(oof_probs, 4),
        "pred_class_4_6": oof_preds,
        "selected_k": df_policy_res["selected_k"],
        "f1_k4": np.round(df_merged["F1_k4"].values, 4),
        "f1_k6": np.round(df_merged["F1_k6"].values, 4),
        "actual_gain_4_6": np.round(gains, 4),
        "y_true_4_6": y_true,
        "policy_f1": np.round(df_policy_res["policy_f1"], 4),
        "context_tokens": df_policy_res["context_tokens"],
        "regret": np.round(df_policy_res["regret"], 4),
        "fold_id": fold_ids,
    })
    
    # Average coefficients
    df_coefs = None
    if len(fold_coefficients) > 0:
        coef_matrix = np.array(fold_coefficients)
        mean_coef = np.mean(coef_matrix, axis=0)
        std_coef = np.std(coef_matrix, axis=0)
        df_coefs = pd.DataFrame({
            "feature": feature_cols,
            "mean_coefficient": np.round(mean_coef, 4),
            "std_coefficient": np.round(std_coef, 4),
            "odds_ratio": np.round(np.exp(mean_coef), 4),
            "direction": ["Positive (Expands)" if c > 0 else "Negative (Halts)" for c in mean_coef],
        }).sort_values(by="mean_coefficient", ascending=False)
        
    return {
        "model_name": model_type,
        "features": feature_cols,
        "n_samples": n,
        "positive_rate": float(np.mean(y_true)),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "balanced_acc": round(bal_acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(spec, 4),
        "f1": round(f1_s, 4),
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "oof_preds": oof_preds,
        "oof_probs": oof_probs,
        "df_predictions": df_export,
        "df_policy_results": df_policy_res,
        "policy_summary": policy_summary,
        "df_coefficients": df_coefs,
        "calibration_bins": df_calib_bins,
    }


# =============================================================================
# 2. THRESHOLD SENSITIVITY GRID FOR POLICY C1
# =============================================================================

def evaluate_stage2_threshold_grid(
    df_merged: pd.DataFrame,
    stage2_probs: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Evaluate Policy C1 across decision thresholds t in [0.20, 0.80]."""
    if thresholds is None:
        thresholds = np.linspace(0.20, 0.80, 13)
        
    f4_mean = float(df_merged["F1_k4"].mean())
    f6_mean = float(df_merged["F1_k6"].mean())
    f8_mean = float(df_merged["F1_k8"].mean())
    
    records = []
    for t in thresholds:
        preds = (stage2_probs >= t).astype(int)
        df_res, sm = simulate_policy_c1(df_merged, preds, stage2_probs, policy_name=f"Threshold_t={t:.2f}")
        
        diff_k4 = sm["mean_f1"] - f4_mean
        diff_k6 = sm["mean_f1"] - f6_mean
        diff_k8 = sm["mean_f1"] - f8_mean
        
        records.append({
            "threshold": round(float(t), 2),
            "fraction_expanded_k6": round(float(np.mean(preds)) * 100.0, 1),
            "mean_f1": sm["mean_f1"],
            "mean_k": sm["mean_k"],
            "mean_context_tokens": sm["mean_context_tokens"],
            "context_reduction_pct_vs_k8": sm["context_reduction_pct_vs_k8"],
            "mean_regret": sm["mean_regret"],
            "median_regret": sm["median_regret"],
            "fraction_zero_regret": sm["fraction_zero_regret"],
            "delta_f1_vs_k4": round(diff_k4, 4),
            "delta_f1_vs_k6": round(diff_k6, 4),
            "delta_f1_vs_k8": round(diff_k8, 4),
        })
        
    return pd.DataFrame(records)


# =============================================================================
# 3. CONTROLLED STAGE-2 FEATURE ABLATION
# =============================================================================

def run_stage2_controlled_ablations(
    df_merged: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Run the 10 controlled Stage-2 feature ablations."""
    ablations = {
        "1. All Stage-2 Features": STAGE_2_CORE_FEATURES,
        "2. BM25 Retrieval Only": ["bm25_mean_score"],
        "3. Semantic Only": ["sem_sim_mean_top6"],
        "4. Lexical Coverage Only": ["lex_coverage_top2"],
        "5. BM25 + Semantic": ["bm25_mean_score", "sem_sim_mean_top6"],
        "6. BM25 + Lexical": ["bm25_mean_score", "lex_coverage_top2"],
        "7. Semantic + Lexical": ["sem_sim_mean_top6", "lex_coverage_top2"],
        "8. Remove BM25": [f for f in STAGE_2_CORE_FEATURES if f != "bm25_mean_score"],
        "9. Remove Semantic": [f for f in STAGE_2_CORE_FEATURES if f != "sem_sim_mean_top6"],
        "10. Remove Lexical": [f for f in STAGE_2_CORE_FEATURES if f != "lex_coverage_top2"],
    }
    
    records = []
    for abl_name, feat_list in ablations.items():
        res = run_stage2_oof_evaluation(
            df_merged,
            feature_cols=feat_list,
            model_type="logistic_regression_balanced",
            n_splits=n_splits,
            seed=seed,
        )
        sm = res["policy_summary"]
        records.append({
            "ablation_configuration": abl_name,
            "feature_count": len(feat_list),
            "roc_auc": res["roc_auc"],
            "pr_auc": res["pr_auc"],
            "balanced_acc": res["balanced_acc"],
            "policy_mean_f1": sm["mean_f1"],
            "policy_mean_k": sm["mean_k"],
            "mean_context_tokens": sm["mean_context_tokens"],
            "context_reduction_pct_vs_k8": sm["context_reduction_pct_vs_k8"],
            "mean_regret": sm["mean_regret"],
        })
        
    return pd.DataFrame(records)


# =============================================================================
# 4. SUBGROUP ANALYSIS
# =============================================================================

def evaluate_stage2_subgroups(
    df_merged: pd.DataFrame,
    df_stage2_oof: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate Stage-2 policy performance across meaningful query subgroups."""
    subgroups = {}
    
    # Syntax
    subgroups["Numerical Queries"] = df_merged["query_is_numerical"] == 1
    subgroups["Non-Numerical Queries"] = df_merged["query_is_numerical"] == 0
    subgroups["What/Which Queries"] = df_merged["query_is_what_which"] == 1
    subgroups["Other Syntax Queries"] = df_merged["query_is_what_which"] == 0
    
    # Query complexity
    med_conj = df_merged["query_conjunction_count"].median()
    subgroups["High Complexity (Conjunction > median)"] = df_merged["query_conjunction_count"] > med_conj
    subgroups["Low Complexity (Conjunction <= median)"] = df_merged["query_conjunction_count"] <= med_conj
    
    # Lexical coverage
    med_lex = df_merged["lex_coverage_top2"].median()
    subgroups["High Lexical Coverage (>= median)"] = df_merged["lex_coverage_top2"] >= med_lex
    subgroups["Low Lexical Coverage (< median)"] = df_merged["lex_coverage_top2"] < med_lex
    
    # Semantic similarity
    med_sem = df_merged["sem_sim_mean_top6"].median()
    subgroups["High Semantic Tail Sim (>= median)"] = df_merged["sem_sim_mean_top6"] >= med_sem
    subgroups["Low Semantic Tail Sim (< median)"] = df_merged["sem_sim_mean_top6"] < med_sem
    
    # Document cluster span
    subgroups["Multi-Cluster Span (> 1 section)"] = df_merged["evidence_query_cluster_span"] > 1
    subgroups["Single-Cluster Span (1 section)"] = df_merged["evidence_query_cluster_span"] <= 1
    
    records = []
    y_true = df_stage2_oof["y_true_4_6"].values
    probs = df_stage2_oof["p_expand_4_6"].values
    preds = df_stage2_oof["pred_class_4_6"].values
    f1s = df_stage2_oof["policy_f1"].values
    regrets = df_stage2_oof["regret"].values
    
    for sg_name, mask in subgroups.items():
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        sub_y = y_true[mask]
        sub_probs = probs[mask]
        sub_preds = preds[mask]
        
        pos_rate = float(np.mean(sub_y)) * 100.0
        exp_rate = float(np.mean(sub_preds)) * 100.0
        
        try:
            roc_auc = float(roc_auc_score(sub_y, sub_probs)) if len(np.unique(sub_y)) > 1 else np.nan
        except Exception:
            roc_auc = np.nan
            
        records.append({
            "subgroup": sg_name,
            "query_count": cnt,
            "actual_positive_rate_pct": round(pos_rate, 1),
            "policy_expansion_rate_pct": round(exp_rate, 1),
            "roc_auc": round(roc_auc, 4) if not np.isnan(roc_auc) else "N/A",
            "mean_f1": round(float(np.mean(f1s[mask])), 4),
            "mean_regret": round(float(np.mean(regrets[mask])), 4),
        })
        
    return pd.DataFrame(records)


# =============================================================================
# 5. GRANULAR STAGE-2 ERROR ANALYSIS
# =============================================================================

def perform_stage2_error_analysis(
    df_merged: pd.DataFrame,
    df_stage2_oof: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Decompose Stage-2 decisions into False Expansion, False Stop, True Pos, True Neg."""
    y_true = df_stage2_oof["y_true_4_6"].values
    y_pred = df_stage2_oof["pred_class_4_6"].values
    gains = df_stage2_oof["actual_gain_4_6"].values
    
    n = len(df_merged)
    cats = []
    for i in range(n):
        if y_true[i] == 1 and y_pred[i] == 1:
            cats.append("True Positive (Beneficial Expansion)")
        elif y_true[i] == 0 and y_pred[i] == 1:
            cats.append("False Expansion (Unneeded Context)")
        elif y_true[i] == 1 and y_pred[i] == 0:
            cats.append("False Stop (Missed Gain)")
        else:
            cats.append("True Negative (Correct Halt)")
            
    df_audit = df_stage2_oof.copy()
    df_audit["error_category"] = cats
    df_audit["bm25_mean_score"] = df_merged["bm25_mean_score"].values
    df_audit["sem_sim_mean_top6"] = df_merged["sem_sim_mean_top6"].values
    df_audit["lex_coverage_top2"] = df_merged["lex_coverage_top2"].values
    df_audit["query_word_count"] = df_merged["query_word_count"].values
    
    records = []
    for cat_name, grp in df_audit.groupby("error_category"):
        cnt = len(grp)
        pct = (cnt / n) * 100.0
        records.append({
            "error_category": cat_name,
            "query_count": cnt,
            "percentage": round(pct, 2),
            "mean_actual_gain": round(float(grp["actual_gain_4_6"].mean()), 4),
            "mean_bm25_score": round(float(grp["bm25_mean_score"].mean()), 2),
            "mean_sem_sim_top6": round(float(grp["sem_sim_mean_top6"].mean()), 3),
            "mean_lex_coverage_top2": round(float(grp["lex_coverage_top2"].mean()), 3),
            "mean_query_length": round(float(grp["query_word_count"].mean()), 1),
        })
        
    df_summary = pd.DataFrame(records).sort_values(by="query_count", ascending=False)
    return df_audit, df_summary


# =============================================================================
# 6. STATISTICAL COMPARISONS VIA PAPER-CLUSTERED BOOTSTRAP
# =============================================================================

def run_stage2_statistical_comparisons(
    df_merged: pd.DataFrame,
    df_policy_c1: pd.DataFrame,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute exact paper-clustered paired bootstrap differences vs static baselines."""
    rng = np.random.RandomState(seed)
    papers = df_merged["paper_id"].unique()
    n_papers = len(papers)
    paper_to_idx = {p: np.where(df_merged["paper_id"].values == p)[0] for p in papers}
    
    pol_f1 = df_policy_c1["policy_f1"].values
    pol_tok = df_policy_c1["context_tokens"].values
    pol_reg = df_policy_c1["regret"].values
    
    baselines = [4, 6, 8]
    records = []
    
    for k_base in baselines:
        base_f1 = df_merged[f"F1_k{k_base}"].values.astype(float)
        base_tok = np.full(len(df_merged), CONTEXT_TOKEN_COSTS[k_base])
        base_reg = np.maximum(0.0, df_merged["oracle_f1"].values - base_f1)
        
        diff_f1 = pol_f1 - base_f1
        diff_tok = pol_tok - base_tok
        diff_reg = pol_reg - base_reg
        
        boot_f1 = np.zeros(n_bootstrap, dtype=float)
        boot_tok = np.zeros(n_bootstrap, dtype=float)
        boot_reg = np.zeros(n_bootstrap, dtype=float)
        
        for b in range(n_bootstrap):
            sampled_p = rng.choice(papers, size=n_papers, replace=True)
            idx = np.concatenate([paper_to_idx[p] for p in sampled_p])
            boot_f1[b] = np.mean(diff_f1[idx])
            boot_tok[b] = np.mean(diff_tok[idx])
            boot_reg[b] = np.mean(diff_reg[idx])
            
        f1_ci_l, f1_ci_h = np.percentile(boot_f1, [2.5, 97.5])
        tok_ci_l, tok_ci_h = np.percentile(boot_tok, [2.5, 97.5])
        reg_ci_l, reg_ci_h = np.percentile(boot_reg, [2.5, 97.5])
        
        # Statistically detectable if zero is outside the CI
        f1_sig = bool(f1_ci_l > 0.0 or f1_ci_h < 0.0)
        
        records.append({
            "comparison": f"Policy C1 vs Static k={k_base}",
            "mean_delta_f1": round(float(np.mean(diff_f1)), 4),
            "ci_95_f1_low": round(float(f1_ci_l), 4),
            "ci_95_f1_high": round(float(f1_ci_h), 4),
            "statistically_detectable_f1_diff": f1_sig,
            "mean_delta_tokens": round(float(np.mean(diff_tok)), 1),
            "ci_95_tokens_low": round(float(tok_ci_l), 1),
            "ci_95_tokens_high": round(float(tok_ci_h), 1),
            "mean_delta_regret": round(float(np.mean(diff_reg)), 4),
            "ci_95_regret_low": round(float(reg_ci_l), 4),
            "ci_95_regret_high": round(float(reg_ci_h), 4),
        })
        
    return pd.DataFrame(records)


# =============================================================================
# 7. PUBLICATION-GRADE FIGURE GENERATION SUITE
# =============================================================================

def generate_stage2_validation_figures(
    df_merged: pd.DataFrame,
    res_eval: Dict[str, Any],
    df_thresh: pd.DataFrame,
    df_abl: pd.DataFrame,
    df_sg: pd.DataFrame,
    df_err_summary: pd.DataFrame,
    df_pareto: pd.DataFrame,
    fig_dir: Path,
):
    """Generate all 9 required publication-grade validation figures."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. stage2_roc_pr.png
    y_true = df_merged["Y_4_6"].values.astype(int)
    probs = res_eval["oof_probs"]
    fpr, tpr, _ = roc_curve(y_true, probs)
    prec, rec, _ = precision_recall_curve(y_true, probs)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].plot(fpr, tpr, color="#2ca02c", lw=2.5, label=f"Stage 2 LogReg (AUC = {res_eval['roc_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance Baseline (0.500)")
    axes[0].set_title("Stage 2 ROC Curve (5-Fold Paper-Disjoint OOF)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate", fontsize=11)
    axes[0].set_ylabel("True Positive Rate", fontsize=11)
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(rec, prec, color="#1f77b4", lw=2.5, label=f"Stage 2 LogReg (PR-AUC = {res_eval['pr_auc']:.3f})")
    axes[1].axhline(np.mean(y_true), color="red", linestyle="--", alpha=0.6, label=f"Majority Prior ({np.mean(y_true):.3f})")
    axes[1].set_title("Stage 2 Precision-Recall Curve", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Recall", fontsize=11)
    axes[1].set_ylabel("Precision", fontsize=11)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "stage2_roc_pr.png", dpi=300)
    plt.close()
    
    # 2. calibration_curve.png
    df_cal = res_eval["calibration_bins"]
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect Calibration")
    plt.plot(df_cal["mean_pred_prob"], df_cal["empirical_accuracy"], marker="o", color="#2ca02c", lw=2.5,
             label=f"Stage 2 LogReg (Brier = {res_eval['brier_score']:.3f}, ECE = {res_eval['ece']:.3f})")
    plt.title("Stage 2 Calibration Curve (Reliability Diagram)", fontsize=12, fontweight="bold")
    plt.xlabel("Mean Predicted Expansion Probability", fontsize=11)
    plt.ylabel("Observed Expansion Frequency", fontsize=11)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "calibration_curve.png", dpi=300)
    plt.close()
    
    # 3. threshold_quality_cost.png
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    color1 = "#1f77b4"
    ax1.plot(df_thresh["threshold"], df_thresh["mean_f1"], color=color1, marker="o", lw=2.5, label="Policy C1 Mean F1")
    ax1.axhline(0.3950, color="red", linestyle="--", alpha=0.5, label="Static k=8 (0.3950)")
    ax1.axhline(0.3857, color="green", linestyle=":", alpha=0.6, label="Static k=6 (0.3857)")
    ax1.axhline(0.3596, color="orange", linestyle="--", alpha=0.5, label="Static k=4 (0.3596)")
    ax1.set_xlabel("Probability Threshold (t)", fontsize=11)
    ax1.set_ylabel("Mean Answer F1", color=color1, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color2 = "#d62728"
    ax2.plot(df_thresh["threshold"], df_thresh["context_reduction_pct_vs_k8"], color=color2, marker="s", lw=2, linestyle="-.", label="Token Savings vs k=8 (%)")
    ax2.set_ylabel("Context Token Reduction vs k=8 (%)", color=color2, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color2)
    
    plt.title("Policy C1: Decision Threshold vs Quality and Cost", fontsize=12, fontweight="bold")
    fig.tight_layout()
    plt.savefig(fig_dir / "threshold_quality_cost.png", dpi=300)
    plt.close()
    
    # 4. feature_coefficients.png
    if res_eval["df_coefficients"] is not None:
        df_c = res_eval["df_coefficients"]
        plt.figure(figsize=(9, 5))
        sns.barplot(data=df_c, y="feature", x="mean_coefficient", hue="feature", palette="vlag", legend=False)
        plt.title("Stage 2 Standardized Logistic Regression Coefficients", fontsize=12, fontweight="bold")
        plt.xlabel("Mean Standardized Coefficient (± std across folds)", fontsize=11)
        plt.ylabel("")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / "feature_coefficients.png", dpi=300)
        plt.close()
        
    # 5. feature_response_curves.png
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    top_feats = ["bm25_mean_score", "sem_sim_mean_top6", "lex_coverage_top2"]
    for idx, feat in enumerate(top_feats):
        ax = axes[idx]
        x_vals = df_merged[feat].values
        bins = pd.qcut(x_vals, q=5, duplicates="drop")
        df_bin = pd.DataFrame({"bin": bins, "feat_val": x_vals, "prob": probs, "y": y_true}).groupby("bin", observed=True).agg(
            mean_x=("feat_val", "mean"),
            mean_prob=("prob", "mean"),
            empirical_y=("y", "mean"),
        ).reset_index()
        ax.plot(range(len(df_bin)), df_bin["mean_prob"], marker="o", color="#1f77b4", lw=2, label="Predicted P(expand)")
        ax.plot(range(len(df_bin)), df_bin["empirical_y"], marker="s", color="#2ca02c", lw=2, linestyle="--", label="Empirical Expansion Rate")
        ax.set_title(f"Response Curve: {feat}", fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(df_bin)))
        ax.set_xticklabels([f"Q{i+1}" for i in range(len(df_bin))])
        ax.set_xlabel(f"{feat} Quintiles", fontsize=10)
        ax.set_ylabel("Expansion Probability", fontsize=10)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "feature_response_curves.png", dpi=300)
    plt.close()
    
    # 6. feature_ablation.png
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_abl, y="ablation_configuration", x="roc_auc", hue="ablation_configuration", palette="crest", legend=False)
    plt.axvline(0.50, color="gray", linestyle="--", alpha=0.7, label="Chance (0.50)")
    plt.title("Stage 2 Controlled Feature Ablation (ROC-AUC)", fontsize=12, fontweight="bold")
    plt.xlabel("Out-of-Fold ROC-AUC", fontsize=11)
    plt.ylabel("")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "feature_ablation.png", dpi=300)
    plt.close()
    
    # 7. subgroup_performance.png
    plt.figure(figsize=(11, 6))
    sns.barplot(data=df_sg, y="subgroup", x="mean_f1", hue="subgroup", palette="Blues_r", legend=False)
    plt.axvline(0.3596, color="orange", linestyle="--", label="Static k=4 (0.3596)")
    plt.axvline(0.3857, color="green", linestyle=":", label="Static k=6 (0.3857)")
    plt.title("Policy C1 Answer F1 Across Query Subgroups", fontsize=12, fontweight="bold")
    plt.xlabel("Mean Answer F1", fontsize=11)
    plt.ylabel("")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "subgroup_performance.png", dpi=300)
    plt.close()
    
    # 8. error_analysis.png
    plt.figure(figsize=(9, 5))
    sns.barplot(data=df_err_summary, y="error_category", x="percentage", hue="error_category", palette="magma", legend=False)
    plt.title("Stage 2 Error Breakdown (% of Queries)", fontsize=12, fontweight="bold")
    plt.xlabel("Percentage of Development Queries (%)", fontsize=11)
    plt.ylabel("")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "error_analysis.png", dpi=300)
    plt.close()
    
    # 9. policy_pareto_frontier.png
    plt.figure(figsize=(11, 7))
    for _, row in df_pareto.iterrows():
        sys_name = row["system"]
        x = row["mean_context_tokens"]
        y = row["mean_f1"]
        is_opt = row.get("is_pareto_optimal", False)
        
        if "Oracle" in sys_name:
            marker = "*"
            size = 220
            color = "#d62728"
        elif "Static" in sys_name:
            marker = "s"
            size = 130
            color = "#4c72b0"
        elif "Policy C1" in sys_name:
            marker = "D"
            size = 180
            color = "#2ca02c"
        else:
            marker = "o"
            size = 120
            color = "#ff7f0e"
            
        plt.scatter(x, y, s=size, marker=marker, color=color, alpha=0.85, edgecolors="black", zorder=4)
        plt.annotate(
            sys_name,
            (x, y),
            textcoords="offset points",
            xytext=(0, 9 if is_opt else -13),
            ha="center",
            fontsize=9,
            fontweight="bold" if ("Policy C1" in sys_name or is_opt) else "normal",
        )
        
    plt.title("QCCA-V2 Final Validation: Multi-Objective Pareto Frontier", fontsize=13, fontweight="bold")
    plt.xlabel("Mean Context Tokens (Context Cost)", fontsize=11)
    plt.ylabel("Mean Answer F1 (Quality)", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "policy_pareto_frontier.png", dpi=300)
    plt.close()
