"""Additional Week 4 Scientific Analyses:
1. 1,000-seed Random Baseline Validation (Uniform & Target-Frequency Matched)
2. Logistic Regression C-Sensitivity & Class Weighting Analysis
3. Utility-Aware Allocation Experiments
4. Updated Pareto and Tradeoff Analysis
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.week4.evaluation import (
    CONTEXT_TOKEN_COSTS,
    FrozenResponseSurfaceLookup,
    evaluate_allocator_predictions
)
from src.week4.features import load_retrieval_features
from src.week4.pareto import compute_pareto_frontier
from src.week4.targets import load_epsilon_targets, prepare_training_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

K_ALLOC = [2, 4, 5, 6, 8]
FEATURE_COLS = ["rho_1", "delta_12", "entropy", "entropy_normalized", "query_token_length", "n_high"]


# =============================================================================
# 1. RANDOM BASELINE VALIDATION (1,000 SEEDS)
# =============================================================================
def validate_random_baselines(
    df_data: pd.DataFrame,
    lookup: FrozenResponseSurfaceLookup,
    n_seeds: int = 1000
) -> Dict[str, Any]:
    """Validate random allocation baselines over n_seeds runs.
    
    Evaluates:
    1. Uniform Random Allocation: P(k) = 1/5 for k in {2, 4, 5, 6, 8}
    2. Target-Frequency-Matched Random Allocation: P(k) matches true epsilon*=0.01 target distribution
    """
    logger.info("Running random baseline validation across %d seeds...", n_seeds)
    n_queries = len(df_data)
    
    # Target distribution probabilities from frozen dev set
    target_counts = df_data["target_k"].value_counts().sort_index()
    target_probs = np.array([target_counts.get(k, 0) for k in K_ALLOC], dtype=float)
    target_probs /= target_probs.sum()
    
    uniform_metrics = []
    freq_metrics = []
    
    uniform_k_counts = {k: 0 for k in K_ALLOC}
    freq_k_counts = {k: 0 for k in K_ALLOC}
    
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        
        # 1. Uniform
        u_k = rng.choice(K_ALLOC, size=n_queries)
        for val in u_k:
            uniform_k_counts[val] += 1
            
        df_u = pd.DataFrame({
            "question_id": df_data["question_id"],
            "target_k": df_data["target_k"],
            "pred_k": u_k
        })
        m_u = evaluate_allocator_predictions(df_u, "pred_k", lookup, "target_k")
        uniform_metrics.append(m_u)
        
        # 2. Target-frequency matched
        f_k = rng.choice(K_ALLOC, size=n_queries, p=target_probs)
        for val in f_k:
            freq_k_counts[val] += 1
            
        df_f = pd.DataFrame({
            "question_id": df_data["question_id"],
            "target_k": df_data["target_k"],
            "pred_k": f_k
        })
        m_f = evaluate_allocator_predictions(df_f, "pred_k", lookup, "target_k")
        freq_metrics.append(m_f)
        
    def summarize_runs(run_list, k_counts, total_draws):
        f1_vals = [r["mean_f1"] for r in run_list]
        k_vals = [r["mean_k"] for r in run_list]
        tokens_vals = [r["mean_context_tokens"] for r in run_list]
        reduct_vals = [r["context_reduction_pct_vs_k8"] for r in run_list]
        regret_vals = [r["mean_epsilon_regret"] for r in run_list]
        acc_vals = [r["target_accuracy"] for r in run_list]
        
        return {
            "mean_f1": float(np.mean(f1_vals)),
            "std_f1": float(np.std(f1_vals)),
            "ci95_f1": [float(np.percentile(f1_vals, 2.5)), float(np.percentile(f1_vals, 97.5))],
            "mean_k": float(np.mean(k_vals)),
            "std_k": float(np.std(k_vals)),
            "mean_context_tokens": float(np.mean(tokens_vals)),
            "context_reduction_pct_vs_k8": float(np.mean(reduct_vals)),
            "mean_epsilon_regret": float(np.mean(regret_vals)),
            "target_accuracy": float(np.mean(acc_vals)),
            "allocation_distribution_pct": {k: round(count / total_draws * 100.0, 2) for k, count in k_counts.items()}
        }
        
    total_samples = n_seeds * n_queries
    res_uniform = summarize_runs(uniform_metrics, uniform_k_counts, total_samples)
    res_freq = summarize_runs(freq_metrics, freq_k_counts, total_samples)
    
    return {
        "n_seeds": n_seeds,
        "n_queries": n_queries,
        "uniform_random": res_uniform,
        "target_frequency_matched_random": res_freq
    }


# =============================================================================
# 2. LOGISTIC REGRESSION C-SENSITIVITY ANALYSIS
# =============================================================================
def evaluate_logreg_c_sensitivity(
    df_data: pd.DataFrame,
    lookup: FrozenResponseSurfaceLookup,
    c_candidates: List[float] = [0.01, 0.1, 1.0, 10.0],
    class_weights: List[Optional[str]] = [None, "balanced"],
    n_splits: int = 5,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Pre-specified sensitivity analysis of Logistic Regression across C and class_weight."""
    logger.info("Running Logistic Regression C-sensitivity across C=%s, weights=%s", c_candidates, class_weights)
    
    gkf = GroupKFold(n_splits=n_splits)
    groups = df_data["paper_id"].values
    
    results = []
    confusion_matrices = {}
    
    for cw in class_weights:
        cw_label = "balanced" if cw == "balanced" else "standard"
        for c in c_candidates:
            oof_preds = np.zeros(len(df_data), dtype=int)
            
            for fold, (train_idx, val_idx) in enumerate(gkf.split(df_data, groups=groups)):
                train_df = df_data.iloc[train_idx]
                val_df = df_data.iloc[val_idx]
                
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(
                        C=c,
                        class_weight=cw,
                        solver="lbfgs",
                        max_iter=1000,
                        random_state=random_state
                    ))
                ])
                pipe.fit(train_df[FEATURE_COLS], train_df["target_k"])
                oof_preds[val_idx] = pipe.predict(val_df[FEATURE_COLS])
                
            df_eval = pd.DataFrame({
                "question_id": df_data["question_id"],
                "target_k": df_data["target_k"],
                "pred_k": oof_preds
            })
            
            metrics = evaluate_allocator_predictions(df_eval, "pred_k", lookup, "target_k")
            
            # Confusion matrix
            cm = pd.crosstab(df_eval["target_k"], df_eval["pred_k"], normalize="index").to_dict()
            cm_key = f"{cw_label}_C_{c}"
            confusion_matrices[cm_key] = cm
            
            # Allocation distribution
            counts = df_eval["pred_k"].value_counts().sort_index()
            alloc_dist = {f"pct_k{k}": round(counts.get(k, 0) / len(df_eval) * 100.0, 1) for k in K_ALLOC}
            
            results.append({
                "class_weight": cw_label,
                "C": c,
                "mean_f1": metrics["mean_f1"],
                "mean_k": metrics["mean_k"],
                "mean_context_tokens": metrics["mean_context_tokens"],
                "context_reduction_pct_vs_k8": metrics["context_reduction_pct_vs_k8"],
                "mean_epsilon_regret": metrics["mean_epsilon_regret"],
                "target_accuracy": metrics["target_accuracy"],
                "exact_match_pct": metrics["error_distribution"]["exact_match_pct"],
                "over_compression_pct": metrics["error_distribution"]["over_compression_pct"],
                "under_compression_pct": metrics["error_distribution"]["under_compression_pct"],
                **alloc_dist
            })
            
    df_sens = pd.DataFrame(results)
    return df_sens, confusion_matrices


# =============================================================================
# 3. UTILITY-AWARE ALLOCATION EXPERIMENTS
# =============================================================================
def evaluate_utility_aware_allocator(
    df_data: pd.DataFrame,
    lookup: FrozenResponseSurfaceLookup,
    lambda_candidates: List[float] = [0.02, 0.05, 0.10],
    n_splits: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """Implement simple utility-aware allocator using only training-fold response surface.
    
    Utility formulation:
        U(q, k) = F1(q, k) - lambda * [cost(k) / cost(k=8)]
        
    Approach:
    For each candidate budget k in {2, 4, 5, 6, 8}, fit a Ridge regressor on the training fold
    to predict F1_hat(q, k) from pre-generation features.
    Then on the validation fold, allocate:
        k_hat(q) = argmax_{k in K_alloc} [ F1_hat(q, k) - lambda * (cost(k) / 1758.5) ]
    """
    logger.info("Running Utility-Aware Allocator across lambda=%s", lambda_candidates)
    gkf = GroupKFold(n_splits=n_splits)
    groups = df_data["paper_id"].values
    
    k8_cost = CONTEXT_TOKEN_COSTS[8]  # 1758.5
    cost_ratios = {k: CONTEXT_TOKEN_COSTS[k] / k8_cost for k in K_ALLOC}
    
    results = []
    
    for lam in lambda_candidates:
        oof_preds = np.zeros(len(df_data), dtype=int)
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(df_data, groups=groups)):
            train_df = df_data.iloc[train_idx]
            val_df = df_data.iloc[val_idx]
            
            # Fit regressor for each k on training fold
            k_models = {}
            for k in K_ALLOC:
                # Get true training fold F1 scores for this k from lookup
                train_f1_k = [lookup.get_f1(qid, k) for qid in train_df["question_id"]]
                
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("reg", Ridge(alpha=10.0, random_state=random_state))
                ])
                pipe.fit(train_df[FEATURE_COLS], train_f1_k)
                k_models[k] = pipe
                
            # Predict on validation fold
            val_pred_f1 = {k: k_models[k].predict(val_df[FEATURE_COLS]) for k in K_ALLOC}
            
            # Select action maximizing expected utility U = F1_hat - lambda * cost_ratio
            for i, idx in enumerate(val_idx):
                best_k = max(K_ALLOC, key=lambda k: val_pred_f1[k][i] - lam * cost_ratios[k])
                oof_preds[idx] = best_k
                
        df_eval = pd.DataFrame({
            "question_id": df_data["question_id"],
            "target_k": df_data["target_k"],
            "pred_k": oof_preds
        })
        
        metrics = evaluate_allocator_predictions(df_eval, "pred_k", lookup, "target_k")
        counts = df_eval["pred_k"].value_counts().sort_index()
        alloc_dist = {f"pct_k{k}": round(counts.get(k, 0) / len(df_eval) * 100.0, 1) for k in K_ALLOC}
        
        results.append({
            "allocator": f"Utility-Aware (λ={lam})",
            "lambda": lam,
            "mean_f1": metrics["mean_f1"],
            "mean_k": metrics["mean_k"],
            "mean_context_tokens": metrics["mean_context_tokens"],
            "context_reduction_pct_vs_k8": metrics["context_reduction_pct_vs_k8"],
            "mean_epsilon_regret": metrics["mean_epsilon_regret"],
            "target_accuracy": metrics["target_accuracy"],
            "oracle_headroom_recovered_pct": metrics["oracle_headroom_recovered_pct"],
            **alloc_dist
        })
        
    return pd.DataFrame(results)


# =============================================================================
# 4. MASTER RUNNER & REPORT GENERATOR
# =============================================================================
def run_all_additional_analyses():
    df_feat = load_retrieval_features("results/week2/processed/dev_retrieval_features.jsonl")
    df_targets = load_epsilon_targets("results/week3/processed/epsilon_oracle_all.csv", epsilon_star=0.01)
    df_data = prepare_training_dataset(df_feat, df_targets)
    lookup = FrozenResponseSurfaceLookup("results/week2/processed/dev_per_query_k_matrix.csv")
    
    # 1. Random validation
    random_res = validate_random_baselines(df_data, lookup, n_seeds=1000)
    
    # 2. LogReg sensitivity
    df_sens, cms = evaluate_logreg_c_sensitivity(df_data, lookup)
    
    # 3. Utility-aware allocator
    df_util = evaluate_utility_aware_allocator(df_data, lookup)
    
    # Save processed CSVs
    p_dir = Path("results/week4/processed")
    p_dir.mkdir(parents=True, exist_ok=True)
    
    df_sens.to_csv(p_dir / "logreg_c_sensitivity.csv", index=False)
    df_util.to_csv(p_dir / "utility_allocator_results.csv", index=False)
    
    with open(p_dir / "random_baseline_validation.json", "w") as f:
        json.dump(random_res, f, indent=2)
        
    logger.info("Saved all additional analyses to %s", p_dir)
    return random_res, df_sens, df_util


if __name__ == "__main__":
    run_all_additional_analyses()
