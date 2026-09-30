"""Script to generate the unexecuted Stage-2 Final Validation notebook."""

import json
from pathlib import Path


def generate_final_validation_notebook(output_path: str = "notebooks/week4/week4_qcca_v2_final_validation.ipynb"):
    cells = []

    def add_md(text: str):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.split("\n")],
        })

    def add_code(text: str):
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [line + "\n" for line in text.split("\n")],
        })

    # Cell 1: Header
    add_md("""# Week 4: Final Rigorous Validation of Stage-2 Selective Evidence Allocation (Policy C1)

**Project:** Query-Conditioned Context Allocator (QCCA) for Efficient Retrieval-Augmented Generation  
**Parent Architecture:** SARA — Selective and Adaptive Retrieval-augmented Generation with Context Compression  
**Objective:** Rigorous final empirical validation of Policy C1:
> *"Can pre-generation retrieval/semantic/coverage features reliably identify when expanding evidence from $k=4$ to $k=6$ is useful, enabling near-static-$k=6/k=8$ answer quality at lower context cost?"*

### Strict Governance Controls:
1. **Zero Test Set Access:** Development split only (86 papers, 231 queries). Historical test split remains 100% frozen and untouched.
2. **Strict Paper-Disjoint GroupKFold:** All modeling, scaling, and threshold evaluations occur strictly out-of-fold grouped by `paper_id`.
3. **No Feature Leakage:** Only deployable pre-generation observables are used.
4. **No Model Shopping:** Evaluates a single fixed primary model (Balanced Logistic Regression) against standard baselines.""")

    # Cell 2: Imports & Environment
    add_code("""import os
import sys
from pathlib import Path

# Robust project root discovery (supports running from repo root or notebooks/week4)
_curr = Path.cwd().resolve()
candidates = [_curr, _curr.parent, _curr.parent.parent]
PROJECT_ROOT = next((p for p in candidates if (p / "src").is_dir() and (p / "results").is_dir()), _curr)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
print(f"Working directory set to Project Root: {PROJECT_ROOT}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.week4.qcca_v2 import (
    CONTEXT_TOKEN_COSTS,
    PROVISIONAL_14_FEATURES,
    load_qcca_v2_data,
    compute_pareto_frontier,
)
from src.week4.qcca_v2_final_validation import (
    STAGE_2_CORE_FEATURES,
    STAGE_2_FEATURE_JUSTIFICATIONS,
    simulate_policy_c1,
    run_stage2_oof_evaluation,
    evaluate_stage2_threshold_grid,
    run_stage2_controlled_ablations,
    evaluate_stage2_subgroups,
    perform_stage2_error_analysis,
    run_stage2_statistical_comparisons,
    generate_stage2_validation_figures,
)

OUTPUT_DIR = PROJECT_ROOT / "results/week4/qcca_v2_final_validation"
FIG_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("Environment initialized successfully. Output paths configured.")""")

    # Cell 3: Data Loading
    add_md("""## 1. Data Loading & Feature Set Specification
Loads the 231 QASPER development queries across 86 papers with frozen response surfaces ($k \\in \\{2, 4, 6, 8\\}$) and deployable pre-generation features.""")

    add_code("""df_f, df_t, df_merged = load_qcca_v2_data(
    feature_matrix_path="results/week4/feature_discovery/feature_matrix.csv",
    dev_matrix_path="results/week2/processed/dev_per_query_k_matrix.csv",
    oracle_path="results/week3/processed/epsilon_oracle_all.csv",
)
print(f"Loaded {len(df_merged)} queries across {df_merged['paper_id'].nunique()} papers.")
print(f"Stage 2 Target (Y_4_6) Positive Rate: {df_merged['Y_4_6'].mean()*100:.1f}% ({df_merged['Y_4_6'].sum()}/{len(df_merged)})")

print("\\nValidated Stage-2 Core Features and Scientific Justifications:")
for feat, just in STAGE_2_FEATURE_JUSTIFICATIONS.items():
    print(f"- {feat}: {just}\\n")""")

    # Cell 5: Section 2 Rebuild Strict OOF Evaluation
    add_md("""## 2. Rebuild Strict Out-of-Fold Stage-2 Evaluation
Produces genuine out-of-fold predictions for every query under 5-fold paper-disjoint GroupKFold cross-validation. No query receives a prediction from a model trained on its paper.""")

    add_code("""res_primary = run_stage2_oof_evaluation(
    df_merged,
    feature_cols=STAGE_2_CORE_FEATURES,
    target_col="Y_4_6",
    gain_col="G_4_6",
    model_type="logistic_regression_balanced",
    n_splits=5,
    seed=42,
)

df_stage2_oof = res_primary["df_predictions"]
df_stage2_oof.to_csv(OUTPUT_DIR / "stage2_oof_predictions.csv", index=False)

print("Stage 2 OOF Predictions successfully saved to results/week4/qcca_v2_final_validation/stage2_oof_predictions.csv")
display(df_stage2_oof.head(10))""")

    # Cell 7: Section 3 Model Comparison
    add_md("""## 3. Fixed Primary Model vs Baselines (No Model Shopping)
Evaluates the primary model (Balanced Logistic Regression) against standard baselines (Majority Prior and Unweighted Logistic Regression).""")

    add_code("""res_majority = run_stage2_oof_evaluation(
    df_merged, STAGE_2_CORE_FEATURES, "Y_4_6", "G_4_6", model_type="majority", n_splits=5, seed=42
)
res_unweighted = run_stage2_oof_evaluation(
    df_merged, STAGE_2_CORE_FEATURES, "Y_4_6", "G_4_6", model_type="logistic_regression", n_splits=5, seed=42
)

models_comparison = [
    {
        "model_name": "Majority Baseline (Always Stop k=4)",
        "roc_auc": res_majority["roc_auc"],
        "pr_auc": res_majority["pr_auc"],
        "balanced_acc": res_majority["balanced_acc"],
        "precision": res_majority["precision"],
        "recall": res_majority["recall"],
        "f1": res_majority["f1"],
        "brier_score": res_majority["brier_score"],
        "ece": res_majority["ece"],
        "policy_mean_f1": res_majority["policy_summary"]["mean_f1"],
        "mean_k": res_majority["policy_summary"]["mean_k"],
        "mean_tokens": res_majority["policy_summary"]["mean_context_tokens"],
        "token_reduction_vs_k8": res_majority["policy_summary"]["context_reduction_pct_vs_k8"],
    },
    {
        "model_name": "Logistic Regression (Unweighted, t=0.5)",
        "roc_auc": res_unweighted["roc_auc"],
        "pr_auc": res_unweighted["pr_auc"],
        "balanced_acc": res_unweighted["balanced_acc"],
        "precision": res_unweighted["precision"],
        "recall": res_unweighted["recall"],
        "f1": res_unweighted["f1"],
        "brier_score": res_unweighted["brier_score"],
        "ece": res_unweighted["ece"],
        "policy_mean_f1": res_unweighted["policy_summary"]["mean_f1"],
        "mean_k": res_unweighted["policy_summary"]["mean_k"],
        "mean_tokens": res_unweighted["policy_summary"]["mean_context_tokens"],
        "token_reduction_vs_k8": res_unweighted["policy_summary"]["context_reduction_pct_vs_k8"],
    },
    {
        "model_name": "Logistic Regression (Balanced, Primary)",
        "roc_auc": res_primary["roc_auc"],
        "pr_auc": res_primary["pr_auc"],
        "balanced_acc": res_primary["balanced_acc"],
        "precision": res_primary["precision"],
        "recall": res_primary["recall"],
        "f1": res_primary["f1"],
        "brier_score": res_primary["brier_score"],
        "ece": res_primary["ece"],
        "policy_mean_f1": res_primary["policy_summary"]["mean_f1"],
        "mean_k": res_primary["policy_summary"]["mean_k"],
        "mean_tokens": res_primary["policy_summary"]["mean_context_tokens"],
        "token_reduction_vs_k8": res_primary["policy_summary"]["context_reduction_pct_vs_k8"],
    },
]

df_calib_models = pd.DataFrame(models_comparison)
df_calib_models.to_csv(OUTPUT_DIR / "calibration_results.csv", index=False)
display(df_calib_models)""")

    # Cell 9: Section 4 Threshold Sensitivity
    add_md("""## 4. Decision Threshold Analysis ($t \\in [0.20, 0.80]$)
Evaluates Policy C1 across decision thresholds on downstream answer F1, context tokens, regret, and differences relative to Static $k \\in \\{4, 6, 8\\}$.""")

    add_code("""df_threshold_sens = evaluate_stage2_threshold_grid(
    df_merged,
    res_primary["oof_probs"],
    thresholds=np.linspace(0.20, 0.80, 13),
)
df_threshold_sens.to_csv(OUTPUT_DIR / "threshold_sensitivity.csv", index=False)
display(df_threshold_sens)""")

    # Cell 11: Section 5 Controlled Feature Ablation
    add_md("""## 5. Controlled Feature Ablation — Stage 2 Only
Evaluates 10 controlled configurations under identical GroupKFold splits to determine whether Stage-2 signal is multi-family or driven by a single feature family.""")

    add_code("""df_feature_abl = run_stage2_controlled_ablations(df_merged, seed=42)
df_feature_abl.to_csv(OUTPUT_DIR / "feature_ablation.csv", index=False)
display(df_feature_abl)""")

    # Cell 13: Section 6 Feature Importance & Interpretability
    add_md("""## 6. Feature Importance & Interpretability
Examines standardized coefficients across folds, odds ratios, and directions of association.""")

    add_code("""df_coefs = res_primary["df_coefficients"]
df_coefs.to_csv(OUTPUT_DIR / "feature_coefficients.csv", index=False)
display(df_coefs)""")

    # Cell 15: Section 7 Confounder & Proxy Analysis
    add_md("""## 7. Confounder & Proxy Variable Analysis
Tests whether Stage-2 feature signals survive controlling for obvious correlated confounders (query length, passage length, BM25 vs Semantic similarity).""")

    add_code("""# Compute correlations with confounders
confounders = ["query_word_count", "bm25_mean_score", "sem_sim_mean_top6", "lex_coverage_top2"]
df_corr_conf = df_merged[confounders + ["G_4_6"]].corr(method="spearman")
print("Spearman Correlation Matrix with Marginal Gain G_4_6 and Potential Confounders:")
display(df_corr_conf)

# Check partial correlation of sem_sim_mean_top6 with G_4_6 partialling out query_word_count and bm25_mean_score
from scipy import stats

def partial_corr_spearman(df, x_col, y_col, covar_cols):
    from sklearn.linear_model import LinearRegression
    X_cov = df[covar_cols].values
    x_res = df[x_col].values - LinearRegression().fit(X_cov, df[x_col].values).predict(X_cov)
    y_res = df[y_col].values - LinearRegression().fit(X_cov, df[y_col].values).predict(X_cov)
    return stats.spearmanr(x_res, y_res)

rho_sem_adj, p_sem_adj = partial_corr_spearman(df_merged, "sem_sim_mean_top6", "G_4_6", ["query_word_count", "bm25_mean_score"])
rho_bm25_adj, p_bm25_adj = partial_corr_spearman(df_merged, "bm25_mean_score", "G_4_6", ["query_word_count"])
rho_lex_adj, p_lex_adj = partial_corr_spearman(df_merged, "lex_coverage_top2", "G_4_6", ["query_word_count", "bm25_mean_score"])

print(f"Partial Spearman rho for sem_sim_mean_top6 controlling for query length and BM25: {rho_sem_adj:+.4f} (p={p_sem_adj:.4f})")
print(f"Partial Spearman rho for bm25_mean_score controlling for query length: {rho_bm25_adj:+.4f} (p={p_bm25_adj:.4f})")
print(f"Partial Spearman rho for lex_coverage_top2 controlling for query length and BM25: {rho_lex_adj:+.4f} (p={p_lex_adj:.4f})")""")

    # Cell 17: Section 8 Subgroup Analysis
    add_md("""## 8. Subgroup Analysis
Evaluates Stage-2 performance across query types: numerical, definitional, complexity levels, coverage levels, and evidence dispersion.""")

    add_code("""df_subgroups = evaluate_stage2_subgroups(df_merged, df_stage2_oof)
df_subgroups.to_csv(OUTPUT_DIR / "subgroup_analysis.csv", index=False)
display(df_subgroups)""")

    # Cell 19: Section 9 Error Analysis
    add_md("""## 9. Granular Stage-2 Error Analysis
Decomposes Stage-2 decisions into False Expansions ($k=6$ chosen when $k=4$ was better) vs False Stoppings ($k=4$ chosen when $k=6$ was better).""")

    add_code("""df_audit, df_error_summary = perform_stage2_error_analysis(df_merged, df_stage2_oof)
df_error_summary.to_csv(OUTPUT_DIR / "error_analysis.csv", index=False)
display(df_error_summary)""")

    # Cell 21: Section 10 Statistical Comparison via Bootstrap
    add_md("""## 10. Statistical Comparison with Static Baselines (Paper-Clustered Paired Bootstrap)
Conducts $B=2,000$ paper-clustered bootstrap replicates to evaluate differences in Answer F1, Context Tokens, and Regret vs Static $k \\in \\{4, 6, 8\\}$.""")

    add_code("""df_stat_comp = run_stage2_statistical_comparisons(
    df_merged,
    res_primary["df_policy_results"],
    n_bootstrap=2000,
    seed=42,
)
df_stat_comp.to_csv(OUTPUT_DIR / "statistical_comparisons.csv", index=False)
display(df_stat_comp)""")

    # Cell 23: Section 11 Pareto Analysis
    add_md("""## 11. Multi-Objective Pareto Frontier Analysis
Maps Policy C1 against static baselines ($k=2, 4, 5, 6, 8$), Random Allocation, QCCA-V1, and the Oracle Sequential ceiling on the context-vs-quality surface.""")

    add_code("""all_systems = [
    {"system": "Static k=2", "mean_f1": float(df_merged["F1_k2"].mean()), "mean_context_tokens": CONTEXT_TOKEN_COSTS[2], "context_reduction_pct_vs_k8": 69.60},
    {"system": "Static k=4", "mean_f1": float(df_merged["F1_k4"].mean()), "mean_context_tokens": CONTEXT_TOKEN_COSTS[4], "context_reduction_pct_vs_k8": 46.20},
    {"system": "Static k=5", "mean_f1": float(df_merged["F1_k5"].mean()) if "F1_k5" in df_merged else 0.3629, "mean_context_tokens": CONTEXT_TOKEN_COSTS[5], "context_reduction_pct_vs_k8": 34.21},
    {"system": "Static k=6", "mean_f1": float(df_merged["F1_k6"].mean()), "mean_context_tokens": CONTEXT_TOKEN_COSTS[6], "context_reduction_pct_vs_k8": 22.71},
    {"system": "Static k=8", "mean_f1": float(df_merged["F1_k8"].mean()), "mean_context_tokens": CONTEXT_TOKEN_COSTS[8], "context_reduction_pct_vs_k8": 0.00},
    {"system": "Random Allocation", "mean_f1": 0.3709, "mean_context_tokens": 1131.0, "context_reduction_pct_vs_k8": 35.68},
    {"system": "QCCA-V1 (Decision Tree)", "mean_f1": 0.3282, "mean_context_tokens": 671.0, "context_reduction_pct_vs_k8": 61.84},
    {"system": "QCCA-V1 (Rule)", "mean_f1": 0.3286, "mean_context_tokens": 854.4, "context_reduction_pct_vs_k8": 51.41},
    {"system": "QCCA-V2 Policy C1 (Final)", "mean_f1": res_primary["policy_summary"]["mean_f1"], "mean_context_tokens": res_primary["policy_summary"]["mean_context_tokens"], "context_reduction_pct_vs_k8": res_primary["policy_summary"]["context_reduction_pct_vs_k8"]},
    {"system": "Oracle Sequential Ceiling", "mean_f1": 0.4069, "mean_context_tokens": 650.3, "context_reduction_pct_vs_k8": 63.02},
    {"system": "Epsilon Oracle (epsilon=0.01)", "mean_f1": float(df_merged["oracle_f1"].mean()), "mean_context_tokens": 841.3, "context_reduction_pct_vs_k8": 52.16},
]

df_pareto = compute_pareto_frontier(pd.DataFrame(all_systems))
df_pareto.to_csv(OUTPUT_DIR / "policy_pareto_points.csv", index=False)
print("Pareto Status of Candidate Systems:")
display(df_pareto[["system", "mean_f1", "mean_context_tokens", "context_reduction_pct_vs_k8", "is_pareto_optimal"]])""")

    # Cell 25: Section 12 Figure Generation
    add_md("""## 12. Publication-Grade Figure Generation
Generates all 9 validation figures into `results/week4/qcca_v2_final_validation/figures/`.""")

    add_code("""print("Generating all 9 publication-grade validation figures...")
generate_stage2_validation_figures(
    df_merged=df_merged,
    res_eval=res_primary,
    df_thresh=df_threshold_sens,
    df_abl=df_feature_abl,
    df_sg=df_subgroups,
    df_err_summary=df_error_summary,
    df_pareto=df_pareto,
    fig_dir=FIG_DIR,
)
print(f"All figures generated successfully in {FIG_DIR}")""")

    # Cell 27: Summary
    add_md("""## 13. Final Validation Summary""")
    add_code("""print("=" * 80)
print("QCCA-V2 FINAL STAGE-2 VALIDATION COMPLETE")
print("=" * 80)
print(f"Artifact directory: {OUTPUT_DIR}")
print("Generated CSVs:")
for p in sorted(OUTPUT_DIR.glob("*.csv")):
    print(f"  - {p.name}")
print("\\nGenerated Figures:")
for p in sorted(FIG_DIR.glob("*.png")):
    print(f"  - {p.name}")
print("=" * 80)""")

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (sara_env)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.16",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=2)
    print(f"Generated UNEXECUTED final validation notebook at {output_path} with {len(cells)} cells.")


if __name__ == "__main__":
    generate_final_validation_notebook()
