"""Generate the unexecuted Jupyter notebook for Week 5 Nonlinear Stage-2 ML Modeling Benchmark."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks/week5/week5_model_benchmark.ipynb"


def build_week5_notebook():
    cells = []

    def add_md(source: str):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": source.strip().splitlines(keepends=True),
        })

    def add_code(source: str):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.strip().splitlines(keepends=True),
        })

    # Cell 1: Title
    add_md("""# Week 5: Nonlinear Stage-2 ML Modeling Benchmark

**Project:** Query-Conditioned Context Allocator (QCCA) for Efficient Retrieval-Augmented Generation  
**Parent Architecture:** SARA — Selective and Adaptive Retrieval-augmented Generation with Context Compression  
**Experiment:** Week 5 Confirmatory Benchmark of Nonlinear Stage-2 Classifiers  
**Evaluation Population:** QASPER Development Split (86 papers, 231 questions)  
**Governance:** Strict Paper-Disjoint GroupKFold. Held-out test set remains 100% frozen & untouched.  

---

### Research Question:
> **"Can a small nonlinear model (Decision Tree, Random Forest, XGBoost, LightGBM, HistGradientBoosting, or a tiny MLP) learn the validated $4 \\rightarrow 6$ expansion decision better than Balanced Logistic Regression, particularly by curbing false expansions caused by high keyword density + redundant evidence, while preserving recall on high-gain expansions?"**""")

    # Cell 2: Imports & Environment
    add_md("""## 1. Environment Verification & Library Imports""")
    add_code("""import os
import sys
from pathlib import Path

# Robust project root discovery
_curr = Path.cwd().resolve()
candidates = [_curr, _curr.parent, _curr.parent.parent]
PROJECT_ROOT = next((p for p in candidates if (p / "src").is_dir() and (p / "results").is_dir()), _curr)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
import torch
import xgboost as xgb
import lightgbm as lgb

from src.week5.model_benchmark import (
    CONTEXT_TOKEN_COSTS,
    DEFAULT_THRESHOLDS,
    STAGE_2_CORE_FEATURES,
    STAGE_2_SUBSET_FEATURES,
    load_week5_dev_data,
    simulate_policy_c1,
    train_eval_model_oof,
    evaluate_model_threshold_grid,
    run_paper_clustered_paired_bootstrap,
    analyze_feature_interactions,
)
from src.week4.qcca_v2 import compute_pareto_frontier

print(f"Project root: {PROJECT_ROOT}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"XGBoost version: {xgb.__version__}")
print(f"LightGBM version: {lgb.__version__}")
print("Libraries imported successfully.")""")

    # Cell 3: Data Loading
    add_md("""## 2. Load Development Split & Stage-2 Transition Targets
Loads development features, per-query response surfaces, and constructs the marginal gain $G_{4 \\rightarrow 6} = F_1(k=6) - F_1(k=4)$ and target $Y_{4 \\rightarrow 6} = \\mathbb{I}[G_{4 \\rightarrow 6} > 0.01]$.""")
    add_code("""df_f, df_t, df_merged = load_week5_dev_data(
    feature_matrix_path=str(PROJECT_ROOT / "results/week4/feature_discovery/feature_matrix.csv"),
    dev_matrix_path=str(PROJECT_ROOT / "results/week2/processed/dev_per_query_k_matrix.csv"),
    oracle_path=str(PROJECT_ROOT / "results/week3/processed/epsilon_oracle_all.csv"),
)

n_queries = len(df_merged)
n_papers = df_merged["paper_id"].nunique()
n_pos = int(df_merged["Y_4_6"].sum())

print(f"Loaded {n_queries} queries across {n_papers} papers.")
print(f"Stage 2 positive transitions: {n_pos} ({n_pos / n_queries * 100:.2f}% base rate)")
print(f"Validated 5 Stage-2 core features: {STAGE_2_CORE_FEATURES}")
df_merged[["question_id", "paper_id", "F1_k4", "F1_k6", "G_4_6", "Y_4_6"] + STAGE_2_CORE_FEATURES].head()""")

    # Cell 4: Phase 1 Baseline Reproduction
    add_md("""## 3. Phase 1: Bit-for-Bit Reproduction of Balanced Logistic Regression Baseline
Must reproduce:
- ROC-AUC ~0.6503
- PR-AUC ~0.3199
- Balanced Accuracy ~0.6448
- Policy F1 ~0.3842
- Mean k ~4.82
- Context Token Reduction vs k=8 ~36.54%""")
    add_code("""res_base_lr = train_eval_model_oof(
    df_merged=df_merged,
    model_family="logistic_regression",
    feature_cols=STAGE_2_CORE_FEATURES,
    n_outer_splits=5,
    n_inner_splits=3,
    calibrate=False,
    seed=42,
)

print("=" * 60)
print("PHASE 1 BASELINE REPRODUCTION RESULTS")
print("=" * 60)
print(f"ROC-AUC:           {res_base_lr['roc_auc']:.4f} (Expected: ~0.6503)")
print(f"PR-AUC:            {res_base_lr['pr_auc']:.4f} (Expected: ~0.3199)")
print(f"Balanced Accuracy: {res_base_lr['balanced_acc']:.4f} (Expected: ~0.6448)")
print(f"Precision:         {res_base_lr['precision']:.4f}")
print(f"Recall:            {res_base_lr['recall']:.4f}")
print(f"Brier Score:       {res_base_lr['brier_score']:.4f}")
print("-" * 60)
sm = res_base_lr['policy_summary']
print(f"Policy Mean F1:    {sm['mean_f1']:.4f} (Expected: ~0.3842)")
print(f"Mean Budget k:     {sm['mean_k']:.2f} (Expected: ~4.82)")
print(f"Context Tokens:    {sm['mean_context_tokens']:.1f} (Expected: ~1115.9)")
print(f"Token Reduction:   {sm['context_reduction_pct_vs_k8']:.2f}% (Expected: ~36.54%)")
print(f"Mean Regret:       {sm['mean_regret']:.4f}")
print(f"Zero-Regret %:     {sm['fraction_zero_regret']:.1f}%")
print(f"False Expansion %: {sm['false_expansion_pct']:.2f}% (66 queries)")
print(f"False Stop %:      {sm['false_stop_pct']:.2f}% (16 queries)")
print("=" * 60)""")

    # Cell 5: Phase 2 Benchmarking Candidates
    add_md("""## 4. Phase 2: Controlled Nonlinear Model Benchmark
Evaluates candidate models across identical paper-disjoint GroupKFold splits:
1. Balanced Logistic Regression (5 feats, Baseline)
2. Balanced Logistic Regression (2 feats: BM25 + Lexical Coverage)
3. Shallow Decision Tree (depths 2-5, balanced)
4. Random Forest (shallow, balanced)
5. XGBoost (small-data regularized, 5 feats)
6. XGBoost (2 feats: BM25 + Lexical Coverage)
7. LightGBM (small-data regularized)
8. HistGradientBoosting (balanced)
9. Small MLP (PyTorch, class-weighted BCE, early stopping)
10. Calibrated variants (Platt/sigmoid)""")
    add_code("""from src.week5.run_benchmark import run_full_week5_benchmark

# Execute the full leak-free benchmark suite
df_model_comp, df_boot, df_policy_comp = run_full_week5_benchmark()
display(df_model_comp[["model_name", "roc_auc", "pr_auc", "balanced_acc", "policy_f1_t050", "mean_k_t050", "token_reduction_t050", "false_expansion_pct_t050", "false_stop_pct_t050"]])""")

    # Cell 6: Statistical Comparison via Paired Bootstrap
    add_md("""## 5. Phase 3: Paper-Clustered Paired Bootstrap ($B=2,000$)
Rigorous statistical comparison against Balanced Logistic Regression and Static $k=8$.""")
    add_code("""df_boot_lr = df_boot[df_boot["baseline"] == "Balanced Logistic Regression (5 feats, Baseline)"].copy()
print("Paired Clustered Bootstrap Comparisons vs Balanced Logistic Regression (B=2,000):")
display(df_boot_lr[["candidate", "mean_delta_f1", "ci_95_f1_low", "ci_95_f1_high", "statistically_detectable_f1", "mean_delta_tokens", "ci_95_tokens_low", "ci_95_tokens_high"]])""")

    # Cell 7: Error Analysis & Interaction
    add_md("""## 6. Phase 4: False Expansion & Feature Interaction Analysis
Inspects whether nonlinear models curb false expansions caused by BM25 $\\times$ Redundancy.""")
    add_code("""df_errors = pd.read_csv(PROJECT_ROOT / "results/week5/model_benchmark/error_analysis.csv")
display(df_errors)

df_feat_imp = pd.read_csv(PROJECT_ROOT / "results/week5/model_benchmark/feature_importance.csv")
print("Permutation Feature Importance Across Families:")
display(df_feat_imp)""")

    # Cell 8: Pareto Analysis
    add_md("""## 7. Multi-Objective Pareto Frontier Analysis
Maps candidate learned policies against static baselines ($k=2, 4, 5, 6, 8$) and the Oracle sequential ceiling.""")
    add_code("""df_pareto = pd.read_csv(PROJECT_ROOT / "results/week5/model_benchmark/pareto_points.csv")
display(df_pareto[["system", "mean_f1", "mean_context_tokens", "context_reduction_pct_vs_k8", "mean_regret", "is_pareto_optimal"]])""")

    # Cell 9: Artifact Summary
    add_md("""## 8. Verification of Generated Artifacts""")
    add_code("""output_dir = PROJECT_ROOT / "results/week5/model_benchmark"
fig_dir = output_dir / "figures"

print("Generated CSV Tables:")
for p in sorted(output_dir.glob("*.csv")):
    print(f"  - {p.name}")

print("\\nGenerated Publication Figures:")
for p in sorted(fig_dir.glob("*.png")):
    print(f"  - {p.name}")""")

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

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=2)

    print(f"Successfully generated UNEXECUTED notebook at {NOTEBOOK_PATH} with {len(cells)} cells.")


if __name__ == "__main__":
    build_week5_notebook()
