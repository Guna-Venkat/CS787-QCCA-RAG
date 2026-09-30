"""Generate the complete Week 4 Jupyter Notebook: week4_qcca_development.ipynb."""

import json
from pathlib import Path

def create_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "sara_env (Python 3.11.16)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.16"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    def add_md(source):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.split("\n")]
        })

    def add_code(source):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.split("\n")]
        })

    # Header
    add_md("""# Week 4: Learning Query-Conditioned Evidence Allocation (QCCA)
**Project:** CS787 Generative AI — Query-Conditioned Context Allocator  
**Parent Framework:** SARA — Selective and Adaptive Retrieval-augmented Generation with Context Compression  
**Benchmark:** QASPER Development Split (Paper-disjoint from training)  
**Evaluation Protocol:** 5-Fold GroupKFold Cross-Validation by Paper ID (`paper_id`)  
**Frozen Contract:** Action space $\\mathcal{K}_{\\text{alloc}} = \\{2, 4, 5, 6, 8\\}$, Target tolerance $\\epsilon^* = 0.01$

---
## Mode Configuration (Smoke Test vs Full Complete Evaluation)
Set `RUN_SMOKE_TEST = False` for the complete 231-question scientific run, or `True` for a rapid 3-question smoke test.""")

    add_code("""# Configuration Toggle
RUN_SMOKE_TEST = False   # Set to True for fast 3-question smoke test, False for complete 231-question run
SMOKE_TEST_LIMIT = 3

print(f"Execution Mode: {'SMOKE TEST (limit = ' + str(SMOKE_TEST_LIMIT) + ' questions)' if RUN_SMOKE_TEST else 'COMPLETE FULL EVALUATION (231 questions)'}")
""")

    add_md("""## 1. Environment & Frozen Artifact Verification
Verify that frozen Week-2 fixed-$k$ response matrices and Week-3 oracle targets are intact.""")

    add_code("""import sys, os, json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cwd = Path.cwd().resolve()
if (cwd / "results").exists():
    PROJECT_ROOT = cwd
elif (cwd.parent / "results").exists():
    PROJECT_ROOT = cwd.parent
elif (cwd.parent.parent / "results").exists():
    PROJECT_ROOT = cwd.parent.parent
else:
    PROJECT_ROOT = cwd

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Verify frozen Week 2 and Week 3 artifacts
w2_matrix_path = PROJECT_ROOT / "results/week2/processed/dev_per_query_k_matrix.csv"
w2_feat_path = PROJECT_ROOT / "results/week2/processed/dev_retrieval_features.jsonl"
w3_targets_path = PROJECT_ROOT / "results/week3/processed/epsilon_oracle_all.csv"

assert w2_matrix_path.exists(), f"Week 2 response matrix missing at {w2_matrix_path}!"
assert w2_feat_path.exists(), f"Week 2 features missing at {w2_feat_path}!"
assert w3_targets_path.exists(), f"Week 3 targets missing at {w3_targets_path}!"

print("✓ All frozen source artifacts verified at PROJECT_ROOT:", PROJECT_ROOT)
""")

    add_md("""## 2. Load Pre-generation Retrieval Features
Load the immutable pre-generation retrieval features from Week 2 and audit for forbidden data leakage.""")

    add_code("""from src.week4.features import load_retrieval_features, audit_for_leakage

df_features = load_retrieval_features(
    features_path=str(w2_feat_path),
    smoke_test=RUN_SMOKE_TEST,
    smoke_limit=SMOKE_TEST_LIMIT
)

print(f"Loaded features table shape: {df_features.shape}")
print(f"Unique questions: {df_features['question_id'].nunique()}, Unique papers: {df_features['paper_id'].nunique()}")
display(df_features.head())
""")

    add_md("""## 3. Load Frozen Week-3 Targets ($\\epsilon^* = 0.01$)
Filter targets to the frozen development operating point $\\epsilon^* = 0.01$ over deployable action space $\\mathcal{K}_{\\text{alloc}} = \\{2, 4, 5, 6, 8\\}$.""")

    add_code("""from src.week4.targets import load_epsilon_targets, prepare_training_dataset

df_targets = load_epsilon_targets(
    targets_path=str(w3_targets_path),
    epsilon_star=0.01,
    k_alloc=[2, 4, 5, 6, 8]
)

df_data = prepare_training_dataset(df_features, df_targets)
print(f"Prepared combined dataset: {len(df_data)} rows across {df_data['paper_id'].nunique()} papers")
display(df_data[['question_id', 'paper_id', 'target_k', 'selected_f1', 'reference_f1_alloc', 'f1_regret']].head())
""")

    add_md("""## 4. Target Distribution Analysis
Inspect class balance across $\\mathcal{K}_{\\text{alloc}} = \\{2, 4, 5, 6, 8\\}$.""")

    add_code("""target_counts = df_data['target_k'].value_counts().sort_index()
target_pcts = (target_counts / len(df_data) * 100).round(2)
dist_df = pd.DataFrame({"Count": target_counts, "Percentage (%)": target_pcts})
print("=== Frozen Target Distribution (epsilon* = 0.01) ===")
display(dist_df)

plt.figure(figsize=(7, 4))
sns.barplot(x=target_counts.index.astype(str), y=target_counts.values, palette="Blues_d")
plt.title("Target Evidence Allocation Distribution (ε* = 0.01)", fontweight="bold")
plt.xlabel("Budget k")
plt.ylabel("Question Count")
plt.show()
""")

    add_md("""## 5. GroupKFold Cross-Validation Setup
Define 5-fold cross-validation partitioned strictly by `paper_id` to guarantee zero cross-question paper leakage.""")

    add_code("""from src.week4.cross_validation import run_grouped_cross_validation, verify_oof_integrity

FEATURE_COLS = ["rho_1", "delta_12", "entropy", "entropy_normalized", "query_token_length", "n_high"]

df_oof, cv_meta = run_grouped_cross_validation(
    df_data=df_data,
    feature_cols=FEATURE_COLS,
    target_col="target_k",
    group_col="paper_id",
    n_splits=5,
    random_state=42,
    logreg_C=1.0,
    tree_depth=3
)

print(f"Completed {cv_meta['n_splits']}-Fold GroupKFold Cross-Validation.")
for fold_info in cv_meta['folds']:
    print(f"Fold {fold_info['fold']}: Train Papers={fold_info['n_train_papers']}, Val Papers={fold_info['n_val_papers']}, Overlap={fold_info['paper_overlap_count']}")

verify_oof_integrity(df_oof, expected_count=len(df_data), k_alloc=[2, 4, 5, 6, 8])
""")

    add_md("""## 6. Model Evaluation on Out-of-Fold Predictions
Evaluate all systems using the frozen Week-2 matrix lookup. Compare QCCA models against static baselines, random control, and oracles.""")

    add_code("""from src.week4.evaluation import FrozenResponseSurfaceLookup, evaluate_allocator_predictions

lookup = FrozenResponseSurfaceLookup(str(w2_matrix_path))

# Populate synthetic baseline columns
k_alloc = [2, 4, 5, 6, 8]
df_oof["pred_k_static_5"] = 5
df_oof["pred_k_static_8"] = 8
df_oof["pred_k_static_best"] = 8
np.random.seed(42)
df_oof["pred_k_random"] = np.random.choice(k_alloc, size=len(df_oof))
df_oof["pred_k_quality_oracle"] = [min(k_alloc, key=lambda k: (-lookup.get_f1(str(r["question_id"]), k), k)) for _, r in df_oof.iterrows()]

systems = [
    ("Static k=5", "static_5"),
    ("Static k=8 (Best Static)", "static_8"),
    ("Random Allocation", "random"),
    ("QCCA-Rule", "pred_k_rule"),
    ("QCCA-LogReg", "pred_k_logreg"),
    ("QCCA-Tree", "pred_k_tree"),
    ("Epsilon Oracle (ε=0.01)", "target_k"),
    ("Quality Oracle", "quality_oracle")
]

records = []
for label, col_key in systems:
    col_name = col_key if col_key.startswith("pred_k_") or col_key == "target_k" else f"pred_k_{col_key}"
    metrics = evaluate_allocator_predictions(df_oof, col_name, lookup, target_col="target_k")
    records.append({
        "System": label,
        "Mean F1": metrics["mean_f1"],
        "Mean k": metrics["mean_k"],
        "Context Tokens": metrics["mean_context_tokens"],
        "Context Reduct. (%)": metrics["context_reduction_pct_vs_k8"],
        "Mean Regret": metrics["mean_epsilon_regret"],
        "Target Accuracy": metrics["target_accuracy"],
        "Headroom Recovered (%)": metrics["oracle_headroom_recovered_pct"]
    })

df_model_comparison = pd.DataFrame(records)
print("=== Table 1: Model Development Performance ===")
display(df_model_comparison)
""")

    add_md("""## 7. Oracle Headroom Recovery Analysis
Quantify how much of the +0.0945 F1 oracle headroom over static $k=8$ is recovered by each allocation system.""")

    add_code("""static_f1 = df_model_comparison[df_model_comparison["System"].str.contains("Best Static")]["Mean F1"].iloc[0]
oracle_f1 = df_model_comparison[df_model_comparison["System"] == "Quality Oracle"]["Mean F1"].iloc[0]
total_headroom = oracle_f1 - static_f1

headroom_rows = []
for sys_name in ["QCCA-Rule", "QCCA-LogReg", "QCCA-Tree", "Random Allocation", "Epsilon Oracle (ε=0.01)"]:
    m_f1 = df_model_comparison[df_model_comparison["System"] == sys_name]["Mean F1"].iloc[0]
    gain = m_f1 - static_f1
    rec = (gain / total_headroom * 100.0) if total_headroom > 1e-6 else 0.0
    headroom_rows.append({
        "System": sys_name,
        "Mean F1": m_f1,
        "Static k=8 F1": static_f1,
        "Quality Oracle F1": oracle_f1,
        "Gain vs Static": round(gain, 4),
        "Oracle Headroom Recovered (%)": round(rec, 2)
    })

df_headroom = pd.DataFrame(headroom_rows)
print("=== Table 2: Headroom Recovery ===")
display(df_headroom)
""")

    add_md("""## 8. Feature Set Ablation & Leave-One-Feature-Out
Assess whether concentration-only features are sufficient vs full retrieval-side signals, and evaluate sensitivity when removing individual features.""")

    add_code("""from src.week4.feature_ablation import run_feature_set_ablation, run_leave_one_feature_out

feature_sets = {
    "concentration_only": ["rho_1", "delta_12", "entropy_normalized", "n_high"],
    "full": FEATURE_COLS
}

df_ablation = run_feature_set_ablation(df_data, feature_sets, lookup, n_splits=cv_meta['n_splits'])
print("=== Table 3: Feature Set Ablation ===")
display(df_ablation)

df_lofo = run_leave_one_feature_out(df_data, FEATURE_COLS, lookup, n_splits=cv_meta['n_splits'])
print("=== Table 4: Leave-One-Feature-Out Sensitivity ===")
display(df_lofo)
""")

    add_md("""## 9. Feature Importance
Extract standardized regression coefficients and tree feature importances.""")

    add_code("""from src.week4.qcca_models import create_logistic_regression_pipeline, extract_feature_importance

logreg_full = create_logistic_regression_pipeline(C=1.0, random_state=42)
logreg_full.fit(df_data[FEATURE_COLS], df_data["target_k"])

df_importance = extract_feature_importance(logreg_full, FEATURE_COLS)
print("=== Table 5: Feature Importance (Logistic Regression) ===")
display(df_importance)

plt.figure(figsize=(8, 4))
plt.barh(df_importance["feature"], df_importance["importance_value"], color="teal")
plt.title("Mean Absolute Standardized Coefficient Across Classes", fontweight="bold")
plt.xlabel("Importance (|Weight|)")
plt.gca().invert_yaxis()
plt.show()
""")

    add_md("""## 10. Pareto Frontier Analysis (Context Cost vs F1 Quality)
Identify Pareto non-dominated operating points in (Context Prompt Tokens, Mean Token F1) space.""")

    add_code("""from src.week4.pareto import compute_pareto_frontier

pareto_data = [
    {"system": "Static k=2", "mean_f1": 0.3038, "mean_context_tokens": 534.5},
    {"system": "Static k=4", "mean_f1": 0.3596, "mean_context_tokens": 946.1},
    {"system": "Static k=5", "mean_f1": 0.3629, "mean_context_tokens": 1157.0},
    {"system": "Static k=6", "mean_f1": 0.3857, "mean_context_tokens": 1359.1},
    {"system": "Static k=8", "mean_f1": 0.3950, "mean_context_tokens": 1758.5},
    {"system": "Random Allocation", "mean_f1": df_model_comparison[df_model_comparison["System"] == "Random Allocation"]["Mean F1"].iloc[0], "mean_context_tokens": df_model_comparison[df_model_comparison["System"] == "Random Allocation"]["Context Tokens"].iloc[0]},
    {"system": "QCCA-Rule", "mean_f1": df_model_comparison[df_model_comparison["System"] == "QCCA-Rule"]["Mean F1"].iloc[0], "mean_context_tokens": df_model_comparison[df_model_comparison["System"] == "QCCA-Rule"]["Context Tokens"].iloc[0]},
    {"system": "QCCA-LogReg", "mean_f1": df_model_comparison[df_model_comparison["System"] == "QCCA-LogReg"]["Mean F1"].iloc[0], "mean_context_tokens": df_model_comparison[df_model_comparison["System"] == "QCCA-LogReg"]["Context Tokens"].iloc[0]},
    {"system": "QCCA-Tree", "mean_f1": df_model_comparison[df_model_comparison["System"] == "QCCA-Tree"]["Mean F1"].iloc[0], "mean_context_tokens": df_model_comparison[df_model_comparison["System"] == "QCCA-Tree"]["Context Tokens"].iloc[0]},
    {"system": "Epsilon Oracle", "mean_f1": df_model_comparison[df_model_comparison["System"] == "Epsilon Oracle (ε=0.01)"]["Mean F1"].iloc[0], "mean_context_tokens": df_model_comparison[df_model_comparison["System"] == "Epsilon Oracle (ε=0.01)"]["Context Tokens"].iloc[0]},
    {"system": "Quality Oracle", "mean_f1": df_model_comparison[df_model_comparison["System"] == "Quality Oracle"]["Mean F1"].iloc[0], "mean_context_tokens": df_model_comparison[df_model_comparison["System"] == "Quality Oracle"]["Context Tokens"].iloc[0]}
]

df_pareto = compute_pareto_frontier(pd.DataFrame(pareto_data))
print("=== Pareto Frontier Analysis ===")
display(df_pareto)

plt.figure(figsize=(9, 6))
for _, r in df_pareto.iterrows():
    is_opt = r["is_pareto_optimal"]
    color = "crimson" if "QCCA" in r["system"] else ("navy" if "Oracle" in r["system"] else "gray")
    marker = "*" if is_opt else "o"
    size = 140 if is_opt else 80
    plt.scatter(r["mean_context_tokens"], r["mean_f1"], color=color, marker=marker, s=size, zorder=5)
    plt.annotate(r["system"], (r["mean_context_tokens"], r["mean_f1"] + 0.003), fontsize=9, ha="center")

opt_pts = df_pareto[df_pareto["is_pareto_optimal"]].sort_values("mean_context_tokens")
plt.plot(opt_pts["mean_context_tokens"], opt_pts["mean_f1"], "r--", alpha=0.6, label="Pareto Frontier")
plt.title("Quality-Efficiency Tradeoff: Context Tokens vs Token F1", fontweight="bold")
plt.xlabel("Mean Context Tokens (Lower is Better)")
plt.ylabel("Mean Token F1 (Higher is Better)")
plt.legend()
plt.show()
""")

    add_md("""## 11. Proposed Freeze Specification for Week 5
Review the proposed model specification for final test evaluation in Week 5.""")

    add_code("""freeze_spec_path = PROJECT_ROOT / "results/week4/processed/qcca_freeze_spec.json"
if freeze_spec_path.exists():
    with open(freeze_spec_path) as f:
        freeze_spec = json.load(f)
    print(json.dumps(freeze_spec, indent=2))
else:
    print("Freeze spec will be written upon running master pipeline.")
""")

    out_file = Path("notebooks/week4/week4_qcca_development.ipynb")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created notebook: {out_file}")

if __name__ == "__main__":
    create_notebook()
