"""Script to generate the unexecuted QCCA-V2 diagnostic notebook."""

import json
from pathlib import Path


def generate_diagnostic_notebook(output_path: str = "notebooks/week4/week4_qcca_v2_diagnostic.ipynb"):
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
    add_md("""# Week 4: QCCA-V2 Diagnostic and Policy-Refinement Study

**Project:** Query-Conditioned Context Allocator (QCCA) for Efficient Retrieval-Augmented Generation  
**Parent Architecture:** SARA — Selective and Adaptive Retrieval-augmented Generation with Context Compression  
**Objective:** Diagnose why the current learned sequential policy fails to convert discovered feature signal into a policy that clearly improves over static-k baselines, and determine what sequential policy structure is actually justified by the data.

### Strict Governance Controls:
1. **Zero Test Set Access:** Development split only (86 papers, 231 queries). The historical test set remains completely frozen.
2. **Paper-Disjoint GroupKFold:** All modeling, scaling, and threshold evaluations are strictly out-of-fold grouped by `paper_id`.
3. **No Feature Leakage:** Only pre-generation observable features are used.""")

    # Cell 2: Imports & Environment
    add_code("""import os
import sys
from pathlib import Path

# Robust project root discovery (supports running from repo root or notebooks/week4)
_curr = Path.cwd().resolve()
candidates = [_curr, _curr.parent, _curr.parent.parent]
PROJECT_ROOT = next((p for p in candidates if (p / 'src').is_dir() and (p / 'results').is_dir()), _curr)
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
    STAGE_1_FEATURES,
    STAGE_2_FEATURES,
    STAGE_3_FEATURES,
    PROVISIONAL_14_FEATURES,
    load_qcca_v2_data,
    compute_pareto_frontier,
)
from src.week4.qcca_v2_diagnostic import (
    FEATURE_FAMILIES,
    train_eval_stage_oof_detailed,
    compute_stage_error_consequences,
    simulate_policy_structure,
    sweep_policy_thresholds,
    nested_infold_threshold_selection,
    evaluate_cost_sensitive_rules,
    train_eval_stage_regressors_oof,
    run_stage_feature_family_ablations,
    evaluate_compact_subsets,
    conduct_detailed_oracle_gap_analysis,
    run_policy_paired_bootstrap,
    plot_stage_roc_pr_curves,
    plot_calibration_curves,
    plot_threshold_curves,
    plot_policy_pareto_frontier,
    plot_oracle_gap_and_failures,
    plot_error_propagation_comparison,
    plot_budget_distributions,
)

OUTPUT_DIR = PROJECT_ROOT / "results/week4/qcca_v2_diagnostic"
FIG_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("Environment initialized successfully. Output paths configured.")""")

    # Cell 3: Data Loading
    add_md("""## 1. Data Loading & Alignment
Loads the 231 QASPER development queries across 86 papers with frozen response surfaces ($k \\in \\{2, 4, 6, 8\\}$) and deployable pre-generation features.""")

    add_code("""df_f, df_t, df_merged = load_qcca_v2_data(
    feature_matrix_path="results/week4/feature_discovery/feature_matrix.csv",
    dev_matrix_path="results/week2/processed/dev_per_query_k_matrix.csv",
    oracle_path="results/week3/processed/epsilon_oracle_all.csv",
)
print(f"Loaded {len(df_merged)} queries across {df_merged['paper_id'].nunique()} papers.")
print(f"Oracle F1 Mean: {df_merged['oracle_f1'].mean():.4f}, Oracle k Mean: {df_merged['oracle_k'].mean():.2f}")
df_merged[['question_id', 'paper_id', 'F1_k2', 'F1_k4', 'F1_k6', 'F1_k8', 'oracle_k', 'oracle_f1']].head()""")

    # Cell 5: Analysis 1 Stage-wise OOF
    add_md("""## 2. Analysis 1 — Stage-Wise Error Diagnosis (2→4, 4→6, 6→8)
Evaluates out-of-fold performance for all three transitions under paper-disjoint GroupKFold cross-validation. Computes ROC-AUC, PR-AUC, Balanced Accuracy, Precision, Recall, Specificity, Brier score, and ECE.""")

    add_code("""# Evaluate each transition stage out-of-fold
res_s1 = train_eval_stage_oof_detailed(df_merged, "Stage 1 (2->4)", "Y_2_4", STAGE_1_FEATURES, "logistic_regression_balanced")
res_s2 = train_eval_stage_oof_detailed(df_merged, "Stage 2 (4->6)", "Y_4_6", STAGE_2_FEATURES, "logistic_regression_balanced")
res_s3 = train_eval_stage_oof_detailed(df_merged, "Stage 3 (6->8)", "Y_6_8", STAGE_3_FEATURES, "logistic_regression_balanced")

stage_results = {
    "Stage 1 (2->4)": res_s1,
    "Stage 2 (4->6)": res_s2,
    "Stage 3 (6->8)": res_s3,
}

df_stagewise_metrics = pd.DataFrame([
    {k: v for k, v in res.items() if not isinstance(v, (np.ndarray, pd.DataFrame))}
    for res in stage_results.values()
])
df_stagewise_metrics.to_csv(OUTPUT_DIR / "stagewise_oof_metrics.csv", index=False)
display(df_stagewise_metrics)""")

    # Cell 7: Analysis 1 Consequence Accounting
    add_md("""### Downstream Policy Consequences of Stage-Wise Errors
Quantifies the actual answer F1 loss, wasted token cost, and regret resulting from False Expansions vs False Stoppings.""")

    add_code("""cons_s1 = compute_stage_error_consequences(df_merged, "Stage 1 (2->4)", "Y_2_4", "G_2_4", res_s1["oof_preds"], token_delta=411.6)
cons_s2 = compute_stage_error_consequences(df_merged, "Stage 2 (4->6)", "Y_4_6", "G_4_6", res_s2["oof_preds"], token_delta=413.0)
cons_s3 = compute_stage_error_consequences(df_merged, "Stage 3 (6->8)", "Y_6_8", "G_6_8", res_s3["oof_preds"], token_delta=399.4)

df_confusion_consequences = pd.concat([cons_s1, cons_s2, cons_s3], ignore_index=True)
df_confusion_consequences.to_csv(OUTPUT_DIR / "stagewise_confusion_matrices.csv", index=False)
display(df_confusion_consequences)""")

    # Cell 9: Analysis 2 Error Propagation
    add_md("""## 3. Analysis 2 — Error Propagation & Policy Structure Variations
Explicitly tests whether Stage 1 errors bottleneck the stronger Stage 2 signal by comparing:
- **Policy A:** Standard cascade ($2 \\rightarrow 4 \\rightarrow 6 \\rightarrow 8$)
- **Policy B:** Start at $k=4 \\rightarrow 6 \\rightarrow 8$ (bypasses Stage 1 entirely)
- **Policy C1:** Start at $k=4 \\rightarrow 6$, stop at $k=6$
- **Policy C2:** Start at $k=4 \\rightarrow 6$, expand to $k=8$
- **Policy D1:** Conservative Stage 1 ($t_{S1} = 0.60$, high-precision)
- **Policy D2:** High-Recall Stage 1 ($t_{S1} = 0.30$, low barrier)""")

    add_code("""# Policy A (Standard cascade)
df_res_a, sm_a = simulate_policy_structure(
    df_merged, "A_full_cascade", res_s1["oof_preds"], res_s2["oof_preds"], res_s3["oof_preds"], policy_name="Policy A (2->4->6->8)"
)

# Policy B (Start k=4)
df_res_b, sm_b = simulate_policy_structure(
    df_merged, "B_start_k4", res_s1["oof_preds"], res_s2["oof_preds"], res_s3["oof_preds"], policy_name="Policy B (Start k=4->6->8)"
)

# Policy C1 (Start k=4, stop k=6)
df_res_c1, sm_c1 = simulate_policy_structure(
    df_merged, "C_start_k4_stop_k6", res_s1["oof_preds"], res_s2["oof_preds"], res_s3["oof_preds"], policy_name="Policy C1 (Start k=4, stop k=6)"
)

# Policy C2 (Start k=4, expand k=8)
df_res_c2, sm_c2 = simulate_policy_structure(
    df_merged, "C_start_k4_expand_k8", res_s1["oof_preds"], res_s2["oof_preds"], res_s3["oof_preds"], policy_name="Policy C2 (Start k=4, expand k=8)"
)

# Policy D1 (Conservative S1: t=0.60)
s1_p_d1 = (res_s1["oof_probs"] >= 0.60).astype(int)
df_res_d1, sm_d1 = simulate_policy_structure(
    df_merged, "D_conservative_s1", s1_p_d1, res_s2["oof_preds"], res_s3["oof_preds"], policy_name="Policy D1 (Conservative S1, t=0.60)"
)

# Policy D2 (High-Recall S1: t=0.30)
s1_p_d2 = (res_s1["oof_probs"] >= 0.30).astype(int)
df_res_d2, sm_d2 = simulate_policy_structure(
    df_merged, "D_aggressive_s1", s1_p_d2, res_s2["oof_preds"], res_s3["oof_preds"], policy_name="Policy D2 (High-Recall S1, t=0.30)"
)

df_error_prop = pd.DataFrame([sm_a, sm_b, sm_c1, sm_c2, sm_d1, sm_d2])
df_error_prop.to_csv(OUTPUT_DIR / "error_propagation.csv", index=False)
display(df_error_prop)""")

    # Cell 11: Analysis 3 Threshold Sweep
    add_md("""## 4. Analysis 3 — Decision Threshold Optimization & Nested In-Fold Tuning
Sweeps decision thresholds over $t \\in [0.10, 0.90]$ and performs strictly leak-free nested in-fold threshold selection.""")

    add_code("""# Threshold sweep across grid
df_thresh_sweep = sweep_policy_thresholds(
    df_merged, res_s1["oof_probs"], res_s2["oof_probs"], res_s3["oof_probs"], thresholds=np.linspace(0.10, 0.90, 17)
)
df_thresh_sweep.to_csv(OUTPUT_DIR / "threshold_sensitivity.csv", index=False)
display(df_thresh_sweep)

# Leak-free nested in-fold threshold tuning
df_nested_res, sm_nested, df_nested_sel = nested_infold_threshold_selection(
    df_merged, res_s1["oof_probs"], res_s2["oof_probs"], res_s3["oof_probs"], res_s1["fold_ids"], objective="utility", lambda_param=0.0001
)
print("Nested In-Fold Selected Thresholds per validation fold:")
display(df_nested_sel)
print("Nested Tuned Policy Performance:")
display(pd.DataFrame([sm_nested]))""")

    # Cell 13: Analysis 4 Cost-Sensitive Decision Rules
    add_md("""## 5. Analysis 4 — Cost-Sensitive & Expected Utility Decision Rules
Investigates whether decision rules based on expected utility ($U = \\Delta F_1 - \\lambda \\times \\Delta \\text{Tokens}$) outperform flat probability thresholding.""")

    add_code("""# Train Ridge regressors for continuous gain prediction
reg_s1 = train_eval_stage_regressors_oof(df_merged, "Stage 1", "G_2_4", STAGE_1_FEATURES)
reg_s2 = train_eval_stage_regressors_oof(df_merged, "Stage 2", "G_4_6", STAGE_2_FEATURES)
reg_s3 = train_eval_stage_regressors_oof(df_merged, "Stage 3", "G_6_8", STAGE_3_FEATURES)

df_cost_sensitive = evaluate_cost_sensitive_rules(
    df_merged,
    res_s1["oof_probs"],
    res_s2["oof_probs"],
    res_s3["oof_probs"],
    reg_s1["oof_continuous_preds"],
    reg_s2["oof_continuous_preds"],
    reg_s3["oof_continuous_preds"],
)
display(df_cost_sensitive)""")

    # Cell 15: Analysis 5 Direct Gain Regression vs Binary Classification
    add_md("""## 6. Analysis 5 — Direct Gain Regression vs Binary Classification
Compares Continuous Gain Regression (Ridge) against Binary Classification (Balanced Logistic Regression).""")

    add_code("""_, sm_ridge = simulate_policy_structure(
    df_merged,
    "A_full_cascade",
    (reg_s1["oof_continuous_preds"] > 0.01).astype(int),
    (reg_s2["oof_continuous_preds"] > 0.01).astype(int),
    (reg_s3["oof_continuous_preds"] > 0.01).astype(int),
    policy_name="Ridge Regression (delta=0.01)",
)

df_model_comp = pd.DataFrame([
    {
        "model_type": "Logistic Regression (Balanced)",
        "s1_metric": res_s1["oof_balanced_acc"],
        "s2_metric": res_s2["oof_balanced_acc"],
        "s3_metric": res_s3["oof_balanced_acc"],
        "policy_f1": sm_a["mean_f1"],
        "policy_tokens": sm_a["mean_context_tokens"],
        "policy_regret": sm_a["mean_regret"],
    },
    {
        "model_type": "Ridge Regression (Continuous Gain)",
        "s1_metric": reg_s1["spearman_rho"],
        "s2_metric": reg_s2["spearman_rho"],
        "s3_metric": reg_s3["spearman_rho"],
        "policy_f1": sm_ridge["mean_f1"],
        "policy_tokens": sm_ridge["mean_context_tokens"],
        "policy_regret": sm_ridge["mean_regret"],
    },
])
df_model_comp.to_csv(OUTPUT_DIR / "transition_model_comparison.csv", index=False)
display(df_model_comp)""")

    # Cell 17: Analysis 6 Feature Ablations
    add_md("""## 7. Analysis 6 — Feature Family Ablations by Stage
Evaluates individual feature families and Leave-One-Family-Out (LOFO) per transition stage.""")

    add_code("""abl_s1 = run_stage_feature_family_ablations(df_merged, "Stage 1 (2->4)", "Y_2_4")
abl_s2 = run_stage_feature_family_ablations(df_merged, "Stage 2 (4->6)", "Y_4_6")
abl_s3 = run_stage_feature_family_ablations(df_merged, "Stage 3 (6->8)", "Y_6_8")

df_stage_ablation = pd.concat([abl_s1, abl_s2, abl_s3], ignore_index=True)
df_stage_ablation.to_csv(OUTPUT_DIR / "feature_ablation_by_stage.csv", index=False)

df_fam_summary = df_stage_ablation[df_stage_ablation["ablation_type"].str.startswith("Only")].pivot(
    index="ablation_type", columns="stage", values="roc_auc"
)
df_fam_summary.to_csv(OUTPUT_DIR / "feature_family_ablation.csv")
display(df_fam_summary)""")

    # Cell 19: Analysis 7 Compact Subsets
    add_md("""## 8. Analysis 7 — Check Whether 14-Feature Set is Too Large
Compares compact, highly interpretable subsets (top 1-3 features per stage) against the full 14-feature model.""")

    add_code("""df_compact = evaluate_compact_subsets(df_merged)
display(df_compact)""")

    # Cell 21: Analysis 8 Probability Calibration
    add_md("""## 9. Analysis 8 — Probability Calibration & Reliability Analysis
Evaluates Brier scores and Expected Calibration Errors (ECE) across stages.""")

    add_code("""df_calib_summary = pd.DataFrame([
    {
        "stage": s_name,
        "brier_score": res["oof_brier_score"],
        "ece": res["oof_ece"],
        "prevalence": res["prevalence_pos"],
    }
    for s_name, res in stage_results.items()
])
df_calib_summary.to_csv(OUTPUT_DIR / "calibration_results.csv", index=False)
display(df_calib_summary)""")

    # Cell 23: Analysis 9 Oracle Gap Analysis
    add_md("""## 10. Analysis 9 — Granular Oracle Gap Analysis & Failure Profiling
Profiles per-query deviations from the oracle ceiling and categorizes failure modes.""")

    add_code("""df_audit, df_fail_summary = conduct_detailed_oracle_gap_analysis(
    df_merged,
    df_res_a,
    res_s1["oof_preds"],
    res_s2["oof_preds"],
    res_s3["oof_preds"],
    res_s1["oof_probs"],
    res_s2["oof_probs"],
    res_s3["oof_probs"],
)
df_audit.to_csv(OUTPUT_DIR / "oracle_gap_analysis.csv", index=False)
display(df_fail_summary)""")

    # Cell 25: Analysis 10 Policy Comparison & Pareto Frontier
    add_md("""## 11. Analysis 10 — Policy Comparison Suite & Multi-Objective Pareto Frontier
Compares candidate sequential policies against Static baselines ($k \\in \\{2, 4, 6, 8\\}$), Oracle Sequential, and Epsilon Oracle.""")

    add_code("""candidate_policies = [
    sm_a,
    sm_b,
    sm_c1,
    sm_c2,
    sm_d1,
    sm_d2,
    sm_ridge,
    sm_nested,
]

static_baselines = [
    {"policy": "Static k=2", "mean_f1": float(df_merged["F1_k2"].mean()), "mean_k": 2.0, "mean_context_tokens": CONTEXT_TOKEN_COSTS[2], "context_reduction_pct_vs_k8": 69.60, "mean_regret": float(np.mean(np.maximum(0, df_merged["oracle_f1"] - df_merged["F1_k2"]))), "pct_k2": 100.0, "pct_k4": 0.0, "pct_k6": 0.0, "pct_k8": 0.0},
    {"policy": "Static k=4", "mean_f1": float(df_merged["F1_k4"].mean()), "mean_k": 4.0, "mean_context_tokens": CONTEXT_TOKEN_COSTS[4], "context_reduction_pct_vs_k8": 46.20, "mean_regret": float(np.mean(np.maximum(0, df_merged["oracle_f1"] - df_merged["F1_k4"]))), "pct_k2": 0.0, "pct_k4": 100.0, "pct_k6": 0.0, "pct_k8": 0.0},
    {"policy": "Static k=6", "mean_f1": float(df_merged["F1_k6"].mean()), "mean_k": 6.0, "mean_context_tokens": CONTEXT_TOKEN_COSTS[6], "context_reduction_pct_vs_k8": 22.71, "mean_regret": float(np.mean(np.maximum(0, df_merged["oracle_f1"] - df_merged["F1_k6"]))), "pct_k2": 0.0, "pct_k4": 0.0, "pct_k6": 100.0, "pct_k8": 0.0},
    {"policy": "Static k=8", "mean_f1": float(df_merged["F1_k8"].mean()), "mean_k": 8.0, "mean_context_tokens": CONTEXT_TOKEN_COSTS[8], "context_reduction_pct_vs_k8": 0.00, "mean_regret": float(np.mean(np.maximum(0, df_merged["oracle_f1"] - df_merged["F1_k8"]))), "pct_k2": 0.0, "pct_k4": 0.0, "pct_k6": 0.0, "pct_k8": 100.0},
    {"policy": "Oracle Sequential (delta=0.01)", "mean_f1": 0.4069, "mean_k": 2.56, "mean_context_tokens": 650.3, "context_reduction_pct_vs_k8": 63.02, "mean_regret": 0.0825, "pct_k2": 76.2, "pct_k4": 19.9, "pct_k6": 3.5, "pct_k8": 0.4},
    {"policy": "Epsilon Oracle (epsilon=0.01)", "mean_f1": float(df_merged["oracle_f1"].mean()), "mean_k": float(df_merged["oracle_k"].mean()), "mean_context_tokens": 841.3, "context_reduction_pct_vs_k8": 52.16, "mean_regret": 0.0, "pct_k2": 57.1, "pct_k4": 17.3, "pct_k6": 8.2, "pct_k8": 10.0},
]

df_all_policies = pd.DataFrame(candidate_policies + static_baselines)
df_all_policies.to_csv(OUTPUT_DIR / "policy_comparison.csv", index=False)
display(df_all_policies)

df_pareto = compute_pareto_frontier(df_all_policies.rename(columns={"policy": "system"}))
df_pareto.to_csv(OUTPUT_DIR / "policy_pareto_points.csv", index=False)
print("Pareto Optimal Systems:")
display(df_pareto[df_pareto["is_pareto_optimal"]][["system", "mean_f1", "mean_context_tokens", "context_reduction_pct_vs_k8"]])""")

    # Cell 27: Paper-Clustered Bootstrap
    add_md("""## 12. Paper-Clustered Paired Bootstrap Hypothesis Testing ($B = 2,000$)
Tests whether candidate policies achieve statistically significant quality differences relative to Static $k \\in \\{4, 6, 8\\}$.""")

    add_code("""policy_dfs_dict = {
    "Policy A (2->4->6->8)": df_res_a,
    "Policy B (Start k=4)": df_res_b,
    "Policy C1 (Start k=4, stop k=6)": df_res_c1,
    "Policy D1 (Conservative S1)": df_res_d1,
}

df_boot = run_policy_paired_bootstrap(
    df_merged,
    policy_dfs_dict,
    baseline_ks=[4, 6, 8],
    n_bootstrap=2000,
    seed=42,
)
df_boot.to_csv(OUTPUT_DIR / "policy_bootstrap_comparison.csv", index=False)
display(df_boot)""")

    # Cell 29: Plotting Suite
    add_md("""## 13. Publication-Grade Figures Generation
Generates all diagnostic figures into `results/week4/qcca_v2_diagnostic/figures/`.""")

    add_code("""print("Generating publication-grade diagnostic figures...")
plot_stage_roc_pr_curves(stage_results, df_merged, str(FIG_DIR / "stage_roc_pr_curves.png"))
plot_calibration_curves(stage_results, str(FIG_DIR / "calibration_curves.png"))
plot_threshold_curves(df_thresh_sweep, str(FIG_DIR / "threshold_vs_f1.png"), str(FIG_DIR / "threshold_vs_token_cost.png"))
plot_policy_pareto_frontier(df_pareto, str(FIG_DIR / "policy_pareto_frontier.png"))
plot_oracle_gap_and_failures(df_fail_summary, str(FIG_DIR / "oracle_gap_breakdown.png"))
plot_error_propagation_comparison(pd.DataFrame([sm_a, sm_b, sm_c1, sm_c2, sm_d1, sm_d2]), str(FIG_DIR / "error_propagation.png"))
plot_budget_distributions(pd.DataFrame([sm_a, sm_b, sm_c1, sm_d1, sm_d2, sm_ridge]), str(FIG_DIR / "budget_distributions.png"))

print(f"All 9 figures saved successfully to {FIG_DIR}")""")

    # Cell 31: Summary
    add_md("""## 14. Diagnostic Study Completion Summary""")
    add_code("""print("=" * 80)
print("QCCA-V2 DIAGNOSTIC STUDY COMPLETE")
print("=" * 80)
print(f"Artifacts successfully written to: {OUTPUT_DIR}")
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
    print(f"Generated UNEXECUTED notebook at {output_path} with {len(cells)} cells.")


if __name__ == "__main__":
    generate_diagnostic_notebook()
