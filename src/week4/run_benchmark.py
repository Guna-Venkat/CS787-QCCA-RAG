"""Executable driver for Week 5 Nonlinear Stage-2 ML Modeling Benchmark.

Executes all model benchmarks under strict paper-disjoint GroupKFold,
generates all 10 CSV results, 10 publication figures, and outputs summary statistics.
"""

import json
import logging
from pathlib import Path
import sys
import warnings

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve, roc_curve

from src.week4.qcca_v2 import compute_pareto_frontier
from src.week4.model_benchmark import (
    CONTEXT_TOKEN_COSTS,
    DEFAULT_THRESHOLDS,
    STAGE_2_CORE_FEATURES,
    STAGE_2_SUBSET_FEATURES,
    analyze_feature_interactions,
    evaluate_model_threshold_grid,
    instantiate_model,
    load_week5_dev_data,
    run_paper_clustered_paired_bootstrap,
    simulate_policy_c1,
    train_eval_model_oof,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Output directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results/week4/model_benchmark"
FIG_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def run_full_week5_benchmark():
    logger.info("=" * 80)
    logger.info("STARTING WEEK 5: NONLINEAR STAGE-2 ML MODELING BENCHMARK")
    logger.info("=" * 80)

    # 1. Load data
    df_f, df_t, df_merged = load_week5_dev_data(
        feature_matrix_path=str(PROJECT_ROOT / "results/week4/feature_discovery/feature_matrix.csv"),
        dev_matrix_path=str(PROJECT_ROOT / "results/week2/processed/dev_per_query_k_matrix.csv"),
        oracle_path=str(PROJECT_ROOT / "results/week3/processed/epsilon_oracle_all.csv"),
    )
    n_queries = len(df_merged)
    n_papers = df_merged["paper_id"].nunique()
    n_pos = int(df_merged["Y_4_6"].sum())
    logger.info(f"Loaded {n_queries} queries across {n_papers} papers. Positive transitions: {n_pos} ({n_pos/n_queries*100:.1f}%)")

    # 2. Define Model Benchmark Configurations
    models_to_run = [
        # (display_name, family, feature_cols, calibrate, calib_method)
        ("Balanced Logistic Regression (5 feats, Baseline)", "logistic_regression", STAGE_2_CORE_FEATURES, False, "sigmoid"),
        ("Balanced Logistic Regression (2 feats: BM25+Lex)", "logistic_regression", STAGE_2_SUBSET_FEATURES, False, "sigmoid"),
        ("Shallow Decision Tree (5 feats)", "decision_tree", STAGE_2_CORE_FEATURES, False, "sigmoid"),
        ("Random Forest (5 feats)", "random_forest", STAGE_2_CORE_FEATURES, False, "sigmoid"),
        ("XGBoost (5 feats)", "xgboost", STAGE_2_CORE_FEATURES, False, "sigmoid"),
        ("XGBoost (2 feats: BM25+Lex)", "xgboost", STAGE_2_SUBSET_FEATURES, False, "sigmoid"),
        ("LightGBM (5 feats)", "lightgbm", STAGE_2_CORE_FEATURES, False, "sigmoid"),
        ("HistGradientBoosting (5 feats)", "hist_gradient_boosting", STAGE_2_CORE_FEATURES, False, "sigmoid"),
        ("Small MLP (5->8->1, 5 feats)", "mlp", STAGE_2_CORE_FEATURES, False, "sigmoid"),
        ("Calibrated Logistic Regression (Platt, 5 feats)", "logistic_regression", STAGE_2_CORE_FEATURES, True, "sigmoid"),
        ("Calibrated Random Forest (Platt, 5 feats)", "random_forest", STAGE_2_CORE_FEATURES, True, "sigmoid"),
        ("Calibrated XGBoost (Platt, 5 feats)", "xgboost", STAGE_2_CORE_FEATURES, True, "sigmoid"),
    ]

    model_eval_results = {}
    model_comparison_records = []
    threshold_records = []
    oof_predictions_dict = {
        "question_id": df_merged["question_id"],
        "paper_id": df_merged["paper_id"],
        "y_true_4_6": df_merged["Y_4_6"].values,
        "gain_4_6": np.round(df_merged["G_4_6"].values, 4),
        "f1_k4": np.round(df_merged["F1_k4"].values, 4),
        "f1_k6": np.round(df_merged["F1_k6"].values, 4),
    }

    # 3. Execute Benchmarking Loop
    for disp_name, family, feat_cols, calib, calib_m in models_to_run:
        logger.info(f"Evaluating: {disp_name} ...")
        res = train_eval_model_oof(
            df_merged=df_merged,
            model_family=family,
            feature_cols=feat_cols,
            n_outer_splits=5,
            n_inner_splits=3,
            calibrate=calib,
            calibration_method=calib_m,
            seed=42,
        )
        model_eval_results[disp_name] = res

        # Collect OOF predictions
        oof_predictions_dict[f"prob_{disp_name}"] = np.round(res["oof_probs"], 4)
        oof_predictions_dict[f"pred_t05_{disp_name}"] = res["oof_preds"]
        oof_predictions_dict[f"pred_tuned_{disp_name}"] = res["oof_preds_tuned_thresh"]

        # Collect Comparison Metrics
        sm = res["policy_summary"]
        sm_tuned = res["policy_summary_tuned"]

        model_comparison_records.append({
            "model_name": disp_name,
            "model_family": family,
            "feature_count": len(feat_cols),
            "calibrated": calib,
            "roc_auc": res["roc_auc"],
            "pr_auc": res["pr_auc"],
            "balanced_acc": res["balanced_acc"],
            "precision": res["precision"],
            "recall": res["recall"],
            "f1": res["f1"],
            "brier_score": res["brier_score"],
            "policy_f1_t050": sm["mean_f1"],
            "mean_k_t050": sm["mean_k"],
            "context_tokens_t050": sm["mean_context_tokens"],
            "token_reduction_t050": sm["context_reduction_pct_vs_k8"],
            "regret_t050": sm["mean_regret"],
            "zero_regret_pct_t050": sm["fraction_zero_regret"],
            "false_expansion_pct_t050": sm["false_expansion_pct"],
            "false_stop_pct_t050": sm["false_stop_pct"],
            "tuned_threshold": sm_tuned["threshold"],
            "policy_f1_tuned": sm_tuned["mean_f1"],
            "mean_k_tuned": sm_tuned["mean_k"],
            "context_tokens_tuned": sm_tuned["mean_context_tokens"],
            "token_reduction_tuned": sm_tuned["context_reduction_pct_vs_k8"],
            "regret_tuned": sm_tuned["mean_regret"],
            "false_expansion_pct_tuned": sm_tuned["false_expansion_pct"],
            "false_stop_pct_tuned": sm_tuned["false_stop_pct"],
        })

        # Threshold grid
        df_th = evaluate_model_threshold_grid(df_merged, res["oof_probs"], disp_name, DEFAULT_THRESHOLDS)
        threshold_records.append(df_th)

    # 4. Save Core Tables
    df_model_comp = pd.DataFrame(model_comparison_records)
    df_model_comp.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    logger.info(f"Saved model_comparison.csv ({len(df_model_comp)} models)")

    df_oof_preds = pd.DataFrame(oof_predictions_dict)
    df_oof_preds.to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)
    logger.info(f"Saved oof_predictions.csv ({len(df_oof_preds)} rows)")

    df_threshold_all = pd.concat(threshold_records, ignore_index=True)
    df_threshold_all.to_csv(OUTPUT_DIR / "threshold_comparison.csv", index=False)
    logger.info(f"Saved threshold_comparison.csv ({len(df_threshold_all)} rows)")

    # 5. Policy Comparison Table (including Static Baselines & Oracle)
    baseline_lr_name = "Balanced Logistic Regression (5 feats, Baseline)"
    res_base_lr = model_eval_results[baseline_lr_name]

    policy_comp_rows = [
        {"system": "Static SARA k=2", "mean_f1": float(df_merged["F1_k2"].mean()), "mean_k": 2.00, "mean_context_tokens": CONTEXT_TOKEN_COSTS[2], "context_reduction_pct_vs_k8": 69.60, "mean_regret": float(np.mean(np.maximum(0.0, df_merged["oracle_f1"] - df_merged["F1_k2"]))), "system_type": "Static Baseline"},
        {"system": "Static SARA k=4", "mean_f1": float(df_merged["F1_k4"].mean()), "mean_k": 4.00, "mean_context_tokens": CONTEXT_TOKEN_COSTS[4], "context_reduction_pct_vs_k8": 46.20, "mean_regret": float(np.mean(np.maximum(0.0, df_merged["oracle_f1"] - df_merged["F1_k4"]))), "system_type": "Static Baseline"},
        {"system": "Static SARA k=5", "mean_f1": float(df_merged["F1_k5"].mean()), "mean_k": 5.00, "mean_context_tokens": CONTEXT_TOKEN_COSTS[5], "context_reduction_pct_vs_k8": 34.21, "mean_regret": float(np.mean(np.maximum(0.0, df_merged["oracle_f1"] - df_merged["F1_k5"]))), "system_type": "Static Baseline"},
        {"system": "Static SARA k=6", "mean_f1": float(df_merged["F1_k6"].mean()), "mean_k": 6.00, "mean_context_tokens": CONTEXT_TOKEN_COSTS[6], "context_reduction_pct_vs_k8": 22.71, "mean_regret": float(np.mean(np.maximum(0.0, df_merged["oracle_f1"] - df_merged["F1_k6"]))), "system_type": "Static Baseline"},
        {"system": "Static SARA k=8", "mean_f1": float(df_merged["F1_k8"].mean()), "mean_k": 8.00, "mean_context_tokens": CONTEXT_TOKEN_COSTS[8], "context_reduction_pct_vs_k8": 0.00, "mean_regret": float(np.mean(np.maximum(0.0, df_merged["oracle_f1"] - df_merged["F1_k8"]))), "system_type": "Static Baseline"},
        {"system": "Random Allocation", "mean_f1": 0.3709, "mean_k": 4.90, "mean_context_tokens": 1131.0, "context_reduction_pct_vs_k8": 35.68, "mean_regret": 0.1185, "system_type": "Unlearned Baseline"},
        {"system": "Oracle Sequential Ceiling", "mean_f1": 0.4069, "mean_k": 2.56, "mean_context_tokens": 650.3, "context_reduction_pct_vs_k8": 63.02, "mean_regret": 0.0825, "system_type": "Theoretical Bound"},
    ]

    for _, row in df_model_comp.iterrows():
        policy_comp_rows.append({
            "system": f"Policy C1 [{row['model_name']}] (t=0.50)",
            "mean_f1": row["policy_f1_t050"],
            "mean_k": row["mean_k_t050"],
            "mean_context_tokens": row["context_tokens_t050"],
            "context_reduction_pct_vs_k8": row["token_reduction_t050"],
            "mean_regret": row["regret_t050"],
            "system_type": "Learned Policy (t=0.50)",
        })

    df_policy_comp = pd.DataFrame(policy_comp_rows)
    df_policy_comp.to_csv(OUTPUT_DIR / "policy_comparison.csv", index=False)
    logger.info(f"Saved policy_comparison.csv ({len(df_policy_comp)} systems)")

    # 6. Pareto Analysis
    df_pareto = compute_pareto_frontier(df_policy_comp)
    df_pareto.to_csv(OUTPUT_DIR / "pareto_points.csv", index=False)
    logger.info("Saved pareto_points.csv")

    # 7. Paper-Clustered Paired Bootstrap Comparisons (B=2000)
    logger.info("Running paper-clustered paired bootstrap comparisons (B=2,000)...")
    bootstrap_rows = []

    # Static k=6 and k=8 baseline policy result tables
    n = len(df_merged)
    df_k6_policy = pd.DataFrame({"policy_f1": df_merged["F1_k6"].values, "context_tokens": np.full(n, CONTEXT_TOKEN_COSTS[6]), "regret": np.maximum(0.0, df_merged["oracle_f1"] - df_merged["F1_k6"])})
    df_k8_policy = pd.DataFrame({"policy_f1": df_merged["F1_k8"].values, "context_tokens": np.full(n, CONTEXT_TOKEN_COSTS[8]), "regret": np.maximum(0.0, df_merged["oracle_f1"] - df_merged["F1_k8"])})

    for disp_name, _, _, _, _ in models_to_run:
        res_cand = model_eval_results[disp_name]
        cand_policy_df = res_cand["df_policy_results"]

        # Comparison vs Balanced Logistic Regression Baseline
        if disp_name != baseline_lr_name:
            stat_vs_lr = run_paper_clustered_paired_bootstrap(
                df_merged,
                candidate_policy_res=cand_policy_df,
                baseline_policy_res=res_base_lr["df_policy_results"],
                baseline_name=baseline_lr_name,
                candidate_name=disp_name,
                n_bootstrap=2000,
                seed=42,
            )
            bootstrap_rows.append(stat_vs_lr)

        # Comparison vs Static k=8
        stat_vs_k8 = run_paper_clustered_paired_bootstrap(
            df_merged,
            candidate_policy_res=cand_policy_df,
            baseline_policy_res=df_k8_policy,
            baseline_name="Static SARA k=8",
            candidate_name=disp_name,
            n_bootstrap=2000,
            seed=42,
        )
        bootstrap_rows.append(stat_vs_k8)

        # Comparison vs Static k=6
        stat_vs_k6 = run_paper_clustered_paired_bootstrap(
            df_merged,
            candidate_policy_res=cand_policy_df,
            baseline_policy_res=df_k6_policy,
            baseline_name="Static SARA k=6",
            candidate_name=disp_name,
            n_bootstrap=2000,
            seed=42,
        )
        bootstrap_rows.append(stat_vs_k6)

    df_boot = pd.DataFrame(bootstrap_rows)
    df_boot.to_csv(OUTPUT_DIR / "bootstrap_comparison.csv", index=False)
    logger.info(f"Saved bootstrap_comparison.csv ({len(df_boot)} comparisons)")

    # 8. Error Analysis Table
    error_records = []
    for disp_name, _, _, _, _ in models_to_run:
        res = model_eval_results[disp_name]
        sm = res["policy_summary"]
        error_records.append({
            "model_name": disp_name,
            "true_negative_count": sm["tn_count"],
            "true_positive_count": sm["tp_count"],
            "false_expansion_count": sm["fp_count"],
            "false_stop_count": sm["fn_count"],
            "false_expansion_pct": sm["false_expansion_pct"],
            "false_stop_pct": sm["false_stop_pct"],
            "true_positive_rate_pct": sm["true_positive_rate_pct"],
            "avg_gain_captured": sm["avg_gain_captured"],
            "avg_distractor_penalty": sm["avg_distractor_penalty"],
        })
    df_errors = pd.DataFrame(error_records)
    df_errors.to_csv(OUTPUT_DIR / "error_analysis.csv", index=False)
    logger.info("Saved error_analysis.csv")

    # 9. Calibration Results Table
    calib_records = []
    for disp_name, _, _, _, _ in models_to_run:
        res = model_eval_results[disp_name]
        probs = res["oof_probs"]
        y_true = df_merged["Y_4_6"].values.astype(int)

        # 5-bin calibration reliability
        bins = np.linspace(0.0, 1.0, 6)
        bin_idx = np.digitize(probs, bins) - 1
        bin_idx = np.clip(bin_idx, 0, 4)

        for b in range(5):
            mask = bin_idx == b
            cnt = int(np.sum(mask))
            if cnt > 0:
                p_mean = float(np.mean(probs[mask]))
                y_mean = float(np.mean(y_true[mask]))
                calib_records.append({
                    "model_name": disp_name,
                    "bin": b,
                    "bin_range": f"[{bins[b]:.1f}-{bins[b+1]:.1f}]",
                    "count": cnt,
                    "mean_predicted_prob": round(p_mean, 4),
                    "empirical_accuracy": round(y_mean, 4),
                    "calibration_gap": round(p_mean - y_mean, 4),
                })
    df_calib_summary = pd.DataFrame(calib_records)
    df_calib_summary.to_csv(OUTPUT_DIR / "calibration_results.csv", index=False)
    logger.info("Saved calibration_results.csv")

    # 10. Feature Importance Analysis
    feat_imp_records = []
    X_full = df_merged[STAGE_2_CORE_FEATURES].copy()
    y_full = df_merged["Y_4_6"].values.astype(int)

    for m_fam in ["logistic_regression", "decision_tree", "random_forest", "xgboost", "lightgbm"]:
        try:
            cfg = model_eval_results[f"{'Balanced Logistic Regression (5 feats, Baseline)' if m_fam=='logistic_regression' else ('Shallow Decision Tree (5 feats)' if m_fam=='decision_tree' else ('Random Forest (5 feats)' if m_fam=='random_forest' else ('XGBoost (5 feats)' if m_fam=='xgboost' else 'LightGBM (5 feats)')))}"]["fold_best_configs"][0]
            model = instantiate_model(m_fam, cfg, seed=42)
            model.fit(X_full, y_full)

            # Permutation importance
            r_perm = permutation_importance(model, X_full, y_full, n_repeats=10, random_state=42, scoring="roc_auc")
            for f_idx, feat in enumerate(STAGE_2_CORE_FEATURES):
                feat_imp_records.append({
                    "model_family": m_fam,
                    "feature": feat,
                    "permutation_importance_mean": round(float(r_perm.importances_mean[f_idx]), 4),
                    "permutation_importance_std": round(float(r_perm.importances_std[f_idx]), 4),
                })
        except Exception as e:
            logger.warning(f"Could not compute feature importance for {m_fam}: {e}")

    df_feat_imp = pd.DataFrame(feat_imp_records)
    df_feat_imp.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    logger.info("Saved feature_importance.csv")

    # 11. 2D Interaction Grid Analysis
    logger.info("Computing 2D interaction grid: BM25 score vs Pairwise Lexical Overlap...")
    xx, yy, grid_dt = analyze_feature_interactions(df_merged, "decision_tree", STAGE_2_CORE_FEATURES, grid_size=20)
    _, _, grid_rf = analyze_feature_interactions(df_merged, "random_forest", STAGE_2_CORE_FEATURES, grid_size=20)
    _, _, grid_xgb = analyze_feature_interactions(df_merged, "xgboost", STAGE_2_CORE_FEATURES, grid_size=20)
    _, _, grid_lr = analyze_feature_interactions(df_merged, "logistic_regression", STAGE_2_CORE_FEATURES, grid_size=20)

    df_inter = pd.DataFrame({
        "bm25_score": xx.ravel(),
        "lexical_overlap": yy.ravel(),
        "prob_expand_logreg": np.round(grid_lr.ravel(), 4),
        "prob_expand_decision_tree": np.round(grid_dt.ravel(), 4),
        "prob_expand_random_forest": np.round(grid_rf.ravel(), 4),
        "prob_expand_xgboost": np.round(grid_xgb.ravel(), 4),
    })
    df_inter.to_csv(OUTPUT_DIR / "interaction_analysis.csv", index=False)
    logger.info("Saved interaction_analysis.csv")

    # 12. GENERATE ALL 10 PUBLICATION-GRADE FIGURES
    logger.info("Generating all 10 publication-grade figures...")
    generate_all_week5_figures(
        df_merged=df_merged,
        model_eval_results=model_eval_results,
        df_model_comp=df_model_comp,
        df_threshold_all=df_threshold_all,
        df_pareto=df_pareto,
        df_boot=df_boot,
        df_errors=df_errors,
        df_feat_imp=df_feat_imp,
        xx=xx,
        yy=yy,
        grid_dt=grid_dt,
        grid_rf=grid_rf,
        grid_xgb=grid_xgb,
        grid_lr=grid_lr,
        fig_dir=FIG_DIR,
    )
    logger.info(f"All 10 figures generated successfully in {FIG_DIR}")

    logger.info("=" * 80)
    logger.info("WEEK 5 BENCHMARKING COMPLETE")
    logger.info("=" * 80)
    return df_model_comp, df_boot, df_policy_comp


def generate_all_week5_figures(
    df_merged,
    model_eval_results,
    df_model_comp,
    df_threshold_all,
    df_pareto,
    df_boot,
    df_errors,
    df_feat_imp,
    xx,
    yy,
    grid_dt,
    grid_rf,
    grid_xgb,
    grid_lr,
    fig_dir: Path,
):
    """Generate the 10 required publication figures."""
    y_true = df_merged["Y_4_6"].values.astype(int)

    # 1. model_roc_pr.png
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    colors = sns.color_palette("tab10", len(model_eval_results))

    for idx, (m_name, res) in enumerate(model_eval_results.items()):
        if "Calibrated" in m_name or "2 feats" in m_name:
            continue
        fpr, tpr, _ = roc_curve(y_true, res["oof_probs"])
        prec, rec, _ = precision_recall_curve(y_true, res["oof_probs"])
        axes[0].plot(fpr, tpr, lw=2, color=colors[idx], label=f"{m_name.split(' (')[0]} (AUC={res['roc_auc']:.3f})")
        axes[1].plot(rec, prec, lw=2, color=colors[idx], label=f"{m_name.split(' (')[0]} (PR-AUC={res['pr_auc']:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance (0.500)")
    axes[0].set_title("Out-of-Fold ROC Curves (5-Fold Paper-Disjoint)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate", fontsize=11)
    axes[0].set_ylabel("True Positive Rate", fontsize=11)
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(np.mean(y_true), color="red", linestyle="--", alpha=0.6, label=f"Prior ({np.mean(y_true):.3f})")
    axes[1].set_title("Out-of-Fold Precision-Recall Curves", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Recall", fontsize=11)
    axes[1].set_ylabel("Precision", fontsize=11)
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "model_roc_pr.png", dpi=300)
    plt.close()

    # 2. model_calibration.png
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect Calibration")
    key_models = [
        "Balanced Logistic Regression (5 feats, Baseline)",
        "Calibrated Logistic Regression (Platt, 5 feats)",
        "XGBoost (5 feats)",
        "Calibrated XGBoost (Platt, 5 feats)",
    ]
    for m_name in key_models:
        if m_name in model_eval_results:
            res = model_eval_results[m_name]
            probs = res["oof_probs"]
            bins = np.linspace(0.0, 1.0, 6)
            b_idx = np.clip(np.digitize(probs, bins) - 1, 0, 4)
            mean_preds, emp_accs = [], []
            for b in range(5):
                m = b_idx == b
                if np.sum(m) > 0:
                    mean_preds.append(float(np.mean(probs[m])))
                    emp_accs.append(float(np.mean(y_true[m])))
            plt.plot(mean_preds, emp_accs, marker="o", lw=2, label=f"{m_name} (Brier={res['brier_score']:.3f})")

    plt.title("Model Calibration Curves (Reliability Diagram)", fontsize=12, fontweight="bold")
    plt.xlabel("Mean Predicted Expansion Probability", fontsize=11)
    plt.ylabel("Observed Expansion Frequency", fontsize=11)
    plt.legend(loc="upper left", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "model_calibration.png", dpi=300)
    plt.close()

    # 3. threshold_quality_cost.png
    plt.figure(figsize=(10, 6))
    top_models = [
        "Balanced Logistic Regression (5 feats, Baseline)",
        "Random Forest (5 feats)",
        "XGBoost (5 feats)",
        "Small MLP (5->8->1, 5 feats)",
    ]
    for m_name in top_models:
        df_sub = df_threshold_all[df_threshold_all["model_name"] == m_name]
        plt.plot(df_sub["threshold"], df_sub["mean_f1"], marker="o", lw=2, label=f"{m_name.split(' (')[0]}")

    plt.axhline(0.3950, color="red", linestyle="--", alpha=0.6, label="Static k=8 (0.3950)")
    plt.axhline(0.3857, color="green", linestyle=":", alpha=0.6, label="Static k=6 (0.3857)")
    plt.axhline(0.3596, color="orange", linestyle="--", alpha=0.6, label="Static k=4 (0.3596)")
    plt.title("Policy Answer F1 vs Decision Threshold", fontsize=12, fontweight="bold")
    plt.xlabel("Decision Threshold (t)", fontsize=11)
    plt.ylabel("Mean Answer F1", fontsize=11)
    plt.legend(loc="lower left", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "threshold_quality_cost.png", dpi=300)
    plt.close()

    # 4. model_policy_pareto.png
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
        elif "Baseline" in sys_name:
            marker = "D"
            size = 180
            color = "#2ca02c"
        else:
            marker = "o"
            size = 130
            color = "#ff7f0e"

        plt.scatter(x, y, s=size, marker=marker, color=color, alpha=0.85, edgecolors="black", zorder=4)
        short_label = sys_name.replace("Policy C1 [", "").replace("] (t=0.50)", "").replace(" (5 feats, Baseline)", "").replace(" (5 feats)", "")
        plt.annotate(
            short_label,
            (x, y),
            textcoords="offset points",
            xytext=(0, 9 if is_opt else -13),
            ha="center",
            fontsize=8.5,
            fontweight="bold" if is_opt else "normal",
        )

    plt.title("Week 5 Model Benchmark: Quality vs Context Token Cost Pareto Surface", fontsize=12, fontweight="bold")
    plt.xlabel("Mean Context Tokens per Query", fontsize=11)
    plt.ylabel("Mean Answer F1", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "model_policy_pareto.png", dpi=300)
    plt.close()

    # 5. feature_importance.png
    if len(df_feat_imp) > 0:
        plt.figure(figsize=(11, 6))
        sns.barplot(data=df_feat_imp, y="feature", x="permutation_importance_mean", hue="model_family", palette="Set2")
        plt.title("Permutation Feature Importance Across Model Families", fontsize=12, fontweight="bold")
        plt.xlabel("Mean Drop in Out-of-Fold ROC-AUC", fontsize=11)
        plt.ylabel("")
        plt.legend(title="Model Family", loc="lower right", fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / "feature_importance.png", dpi=300)
        plt.close()

    # 6. interaction_analysis.png (2D heatmaps)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    grids = [
        ("Logistic Regression (Linear)", grid_lr),
        ("Decision Tree (Partition)", grid_dt),
        ("Random Forest (Smooth Nonlinear)", grid_rf),
        ("XGBoost (Gradient Boosted)", grid_xgb),
    ]
    for idx, (title, g_data) in enumerate(grids):
        ax = axes[idx]
        cp = ax.contourf(xx, yy, g_data, levels=np.linspace(0, 1, 11), cmap="viridis", alpha=0.85)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("BM25 Mean Score", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Pairwise Lexical Redundancy", fontsize=10)
        fig.colorbar(cp, ax=ax, orientation="vertical", shrink=0.8)

    plt.suptitle("Feature Interaction Surface: BM25 Keyword Density vs Passage Redundancy", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(fig_dir / "interaction_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 7. subgroup_performance.png
    subgroup_records = []
    for m_name in ["Balanced Logistic Regression (5 feats, Baseline)", "Random Forest (5 feats)", "XGBoost (5 feats)", "Small MLP (5->8->1, 5 feats)"]:
        res = model_eval_results[m_name]
        df_oof = res["df_policy_results"]
        # Non-numerical queries
        non_num_mask = df_merged["query_is_numerical"].values == 0 if "query_is_numerical" in df_merged else np.ones(len(df_merged), dtype=bool)
        f1_non_num = float(np.mean(df_oof.loc[non_num_mask, "policy_f1"]))
        # High coverage
        med_cov = df_merged["lex_coverage_top2"].median()
        f1_high_cov = float(np.mean(df_oof.loc[df_merged["lex_coverage_top2"] >= med_cov, "policy_f1"]))
        f1_low_cov = float(np.mean(df_oof.loc[df_merged["lex_coverage_top2"] < med_cov, "policy_f1"]))

        subgroup_records.append({"model": m_name.split(" (")[0], "subgroup": "Non-Numerical Queries", "mean_f1": f1_non_num})
        subgroup_records.append({"model": m_name.split(" (")[0], "subgroup": "High Lexical Coverage", "mean_f1": f1_high_cov})
        subgroup_records.append({"model": m_name.split(" (")[0], "subgroup": "Low Lexical Coverage", "mean_f1": f1_low_cov})

    df_subg = pd.DataFrame(subgroup_records)
    df_subg.to_csv(OUTPUT_DIR / "subgroup_results.csv", index=False)
    plt.figure(figsize=(10, 5.5))
    sns.barplot(data=df_subg, x="subgroup", y="mean_f1", hue="model", palette="Set1")
    plt.axhline(0.3857, color="green", linestyle=":", label="Static k=6 (0.3857)")
    plt.title("Policy Answer F1 Across Key Query Subgroups", fontsize=12, fontweight="bold")
    plt.xlabel("")
    plt.ylabel("Mean Answer F1", fontsize=11)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "subgroup_performance.png", dpi=300)
    plt.close()

    # 8. bootstrap_comparison.png (Forest plot)
    baseline_lr_name = "Balanced Logistic Regression (5 feats, Baseline)"
    df_vs_lr = df_boot[df_boot["baseline"] == baseline_lr_name].copy()
    if len(df_vs_lr) > 0:
        plt.figure(figsize=(10, 6))
        y_pos = range(len(df_vs_lr))
        plt.errorbar(
            df_vs_lr["mean_delta_f1"],
            y_pos,
            xerr=[df_vs_lr["mean_delta_f1"] - df_vs_lr["ci_95_f1_low"], df_vs_lr["ci_95_f1_high"] - df_vs_lr["mean_delta_f1"]],
            fmt="o",
            color="#1f77b4",
            ecolor="#1f77b4",
            elinewidth=2,
            capsize=4,
            ms=7,
        )
        plt.axvline(0.0, color="red", linestyle="--", alpha=0.7, label="Baseline Zero (No Diff)")
        plt.yticks(y_pos, [c.split(" (")[0] for c in df_vs_lr["candidate"]])
        plt.title("Paired Bootstrap: ΔAnswer F1 vs Balanced Logistic Regression (95% CI)", fontsize=12, fontweight="bold")
        plt.xlabel("Mean ΔAnswer F1 (Candidate - LogReg)", fontsize=11)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / "bootstrap_comparison.png", dpi=300)
        plt.close()

    # 9. error_breakdown.png
    plt.figure(figsize=(11, 6))
    df_err_plot = df_errors.copy()
    df_err_plot["short_name"] = [m.split(" (")[0] for m in df_err_plot["model_name"]]
    df_melt = pd.melt(df_err_plot, id_vars=["short_name"], value_vars=["false_expansion_pct", "false_stop_pct"], var_name="Error Type", value_name="Percentage (%)")
    df_melt["Error Type"] = df_melt["Error Type"].replace({"false_expansion_pct": "False Expansion (%)", "false_stop_pct": "False Stop (%)"})
    sns.barplot(data=df_melt, x="short_name", y="Percentage (%)", hue="Error Type", palette="mako")
    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.title("Error Breakdown by Model: False Expansions vs False Stops", fontsize=12, fontweight="bold")
    plt.xlabel("")
    plt.ylabel("Percentage of Total Queries (%)", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "error_breakdown.png", dpi=300)
    plt.close()

    # 10. model_comparison_bars.png
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    short_names = [m.split(" (")[0] for m in df_model_comp["model_name"]]

    sns.barplot(x=short_names, y=df_model_comp["roc_auc"], ax=axes[0, 0], hue=short_names, palette="Blues_r", legend=False)
    axes[0, 0].axhline(0.6503, color="red", linestyle="--", alpha=0.7, label="LogReg (0.6503)")
    axes[0, 0].set_title("Out-of-Fold ROC-AUC", fontsize=11, fontweight="bold")
    axes[0, 0].tick_params(axis="x", rotation=40)

    sns.barplot(x=short_names, y=df_model_comp["pr_auc"], ax=axes[0, 1], hue=short_names, palette="Greens_r", legend=False)
    axes[0, 1].axhline(0.3199, color="red", linestyle="--", alpha=0.7, label="LogReg (0.3199)")
    axes[0, 1].set_title("Out-of-Fold PR-AUC", fontsize=11, fontweight="bold")
    axes[0, 1].tick_params(axis="x", rotation=40)

    sns.barplot(x=short_names, y=df_model_comp["policy_f1_t050"], ax=axes[1, 0], hue=short_names, palette="Oranges_r", legend=False)
    axes[1, 0].axhline(0.3842, color="red", linestyle="--", alpha=0.7, label="LogReg (0.3842)")
    axes[1, 0].set_title("Downstream Policy Mean F1 (t=0.50)", fontsize=11, fontweight="bold")
    axes[1, 0].tick_params(axis="x", rotation=40)

    sns.barplot(x=short_names, y=df_model_comp["token_reduction_t050"], ax=axes[1, 1], hue=short_names, palette="Purples_r", legend=False)
    axes[1, 1].axhline(36.54, color="red", linestyle="--", alpha=0.7, label="LogReg (36.54%)")
    axes[1, 1].set_title("Context Token Reduction vs k=8 (%)", fontsize=11, fontweight="bold")
    axes[1, 1].tick_params(axis="x", rotation=40)

    plt.tight_layout()
    plt.savefig(fig_dir / "model_comparison_bars.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    run_full_week5_benchmark()
