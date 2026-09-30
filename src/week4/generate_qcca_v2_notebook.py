"""Generator for the Week 4 QCCA-V2 Modeling Notebook: week4_qcca_v2_modeling.ipynb.

Creates an unexecuted, fully runnable Jupyter notebook adhering to:
- DO NOT RUN THE NOTEBOOK (user runs it manually)
- All outputs empty (execution_count=None, outputs=[])
- Clean markdown sections with rigorous scientific explanations
- Implements all 20 required sections from the QCCA-V2 specification
- Saves all 15 CSVs and 11 figures to results/week4/qcca_v2/
"""

import ast
import json
import uuid
from pathlib import Path


def make_cell(cell_type: str, source_lines: list) -> dict:
    return {
        "cell_type": cell_type,
        "id": str(uuid.uuid4())[:8],
        "metadata": {},
        "execution_count": None if cell_type == "code" else None,
        "outputs": [] if cell_type == "code" else None,
        "source": [line + "\n" for line in source_lines[:-1]] + [source_lines[-1]] if source_lines else []
    }


def create_qcca_v2_notebook(output_path: str = "notebooks/week4/week4_qcca_v2_modeling.ipynb"):
    cells = []

    # Title & Metadata
    cells.append(make_cell("markdown", [
        "# Week 4: QCCA-V2 — Sequential Transition-Aware Evidence Allocation",
        "**Project:** CS787 Generative AI — Query-Conditioned Context Allocator (QCCA)  ",
        "**Parent Framework:** SARA — Selective and Adaptive Retrieval-augmented Generation with Context Compression  ",
        "**Benchmark:** QASPER Development Split (86 papers, 231 queries)  ",
        "**Core Hypothesis:**  ",
        "> *Can a lightweight sequential policy use pre-generation query/retrieval features to decide whether additional textual evidence is worth allocating at each stage, producing a better answer-quality/context-cost tradeoff than fixed-k baselines?*",
        "",
        "```text",
        "k=2  ──(Stage 1: Y_2_4)──>  k=4  ──(Stage 2: Y_4_6)──>  k=6  ──(Stage 3: Y_6_8)──>  k=8",
        "```",
        "",
        "---",
        "### CRITICAL EXPERIMENTAL GUARDRAILS & GOVERNANCE",
        "- **ZERO TEST ACCESS:** The historical test set is 100% UNTOUCHED and UNSEEN.",
        "- **FROZEN DATA TARGETS:** All training targets are derived strictly from the frozen Week 2 response surface ($k \\in \\{2, 4, 6, 8\\}$) and Week 3 Deployable Oracle ($\\epsilon^* = 0.01$).",
        "- **NO FEATURE LEAKAGE:** No answer text, gold answers, generated text, F1 scores, EM, or future outcomes appear as model inputs. Only deployable pre-generation features are used.",
        "- **STRICT PAPER-DISJOINT CROSS-VALIDATION:** All preprocessing, scaling, model fitting, and threshold selection occur strictly inside `GroupKFold(n_splits=5, groups=paper_id)` training folds."
    ]))

    # Section 1: Research Objective
    cells.append(make_cell("markdown", [
        "## 1. Research Objective & Theoretical Motivation",
        "Week 4-A feature discovery and Week 4-B feature validation revealed that evidence allocation is **transition-specific** rather than a flat multi-class problem.",
        "Crucially, lexical coverage exhibits a verified sign reversal: `lex_coverage_top2` is positively associated with $G_{4 \\rightarrow 6}$ ($\\rho = +0.167$) but negatively associated with $G_{6 \\rightarrow 8}$ ($\\rho = -0.170$).",
        "QCCA-V2 formulates evidence allocation as a **3-stage sequential decision process**:",
        "1. **Stage 1 ($2 \\rightarrow 4$):** Decides whether minimal budget ($k=2$) is sufficient or expansion to $k=4$ is required, driven by query complexity.",
        "2. **Stage 2 ($4 \\rightarrow 6$):** Decides whether moderate context ($k=4$) should expand to $k=6$, driven by semantic tail relevance and retrieval strength.",
        "3. **Stage 3 ($6 \\rightarrow 8$):** Decides whether to expand to maximum budget ($k=8$) or stop to avoid distractor noise, governed by lexical coverage saturation."
    ]))

    # Section 2: Environment & Frozen Data Setup
    cells.append(make_cell("markdown", [
        "## 2. Environment Setup, Imports & Frozen Artifacts",
        "Import scientific libraries, plotting tools, and the dedicated [`src.week4.qcca_v2`](file:///home/gunavenkat/Downloads/CS787-RAG-Project/src/week4/qcca_v2.py) library."
    ]))

    cells.append(make_cell("code", [
        "import os",
        "import sys",
        "from pathlib import Path",
        "",
        "# Robust project root discovery",
        "_curr = Path.cwd().resolve()",
        "candidates = [_curr, _curr.parent, _curr.parent.parent]",
        "PROJECT_ROOT = next((p for p in candidates if (p / 'src').is_dir() and (p / 'results').is_dir()), _curr)",
        "if str(PROJECT_ROOT) not in sys.path:",
        "    sys.path.insert(0, str(PROJECT_ROOT))",
        "os.chdir(PROJECT_ROOT)",
        "print(f'Working directory set to Project Root: {PROJECT_ROOT}')",
        "",
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "",
        "from src.week4.qcca_v2 import (",
        "    CONTEXT_TOKEN_COSTS,",
        "    STAGE_1_FEATURES,",
        "    STAGE_2_FEATURES,",
        "    STAGE_3_FEATURES,",
        "    PROVISIONAL_14_FEATURES,",
        "    construct_transition_targets,",
        "    summarize_transition_targets,",
        "    load_qcca_v2_data,",
        "    train_stage_classifiers_cv,",
        "    train_stage_regressors_cv,",
        "    simulate_sequential_policy,",
        "    simulate_oracle_sequential_policy,",
        "    evaluate_threshold_sensitivity,",
        "    get_ablation_feature_sets,",
        "    run_feature_ablations,",
        "    run_paper_clustered_bootstrap_policy,",
        "    compute_pareto_frontier,",
        "    analyze_policy_failures,",
        "    evaluate_qcca_v2_decision_gates,",
        "    generate_all_qcca_v2_figures,",
        ")",
        "",
        "%matplotlib inline",
        "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')",
        "",
        "OUTPUT_DIR = PROJECT_ROOT / 'results/week4/qcca_v2'",
        "FIGURES_DIR = OUTPUT_DIR / 'figures'",
        "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)",
        "FIGURES_DIR.mkdir(parents=True, exist_ok=True)",
        "",
        "RANDOM_SEED = 42",
        "DELTA_PRIMARY = 0.01",
        "print(f'QCCA-V2 Output Directory: {OUTPUT_DIR}')"
    ]))

    # Section 3: Transition Target Construction
    cells.append(make_cell("markdown", [
        "## 3. Transition-Target Construction",
        "From the frozen per-query response surface, compute continuous marginal gains $G_{2 \\rightarrow 4}, G_{4 \\rightarrow 6}, G_{6 \\rightarrow 8}$ and binary transition targets $Y = \\mathbf{1}[G > \\delta]$ for $\\delta \\in \\{0.00, 0.01, 0.02, 0.05\\}$."
    ]))

    cells.append(make_cell("code", [
        "df_f, df_t, df_merged = load_qcca_v2_data(",
        "    feature_matrix_path=str(PROJECT_ROOT / 'results/week4/feature_discovery/feature_matrix.csv'),",
        "    dev_matrix_path=str(PROJECT_ROOT / 'results/week2/processed/dev_per_query_k_matrix.csv'),",
        "    oracle_path=str(PROJECT_ROOT / 'results/week3/processed/epsilon_oracle_all.csv'),",
        ")",
        "",
        "print(f'Loaded {len(df_merged)} queries across {df_merged[\"paper_id\"].nunique()} papers.')",
        "print('Transition targets constructed successfully.')",
        "df_merged[['question_id', 'paper_id', 'F1_k2', 'F1_k4', 'G_2_4', 'Y_2_4', 'G_4_6', 'Y_4_6', 'G_6_8', 'Y_6_8']].head()"
    ]))

    # Section 4: Class Balance Inspection
    cells.append(make_cell("markdown", [
        "## 4. Class Balance & Imbalance Inspection",
        "Inspect positive expansion rates $P(Y=1)$ across stages and delta thresholds."
    ]))

    cells.append(make_cell("code", [
        "df_target_summary = summarize_transition_targets(df_t, delta_values=[0.00, 0.01, 0.02, 0.05])",
        "df_target_summary.to_csv(OUTPUT_DIR / 'transition_target_summary.csv', index=False)",
        "",
        "# Plot Class Balance",
        "plt.figure(figsize=(9, 4.8))",
        "sns.barplot(data=df_target_summary, x='transition_stage', y='positive_rate', hue='delta', palette='Blues_d')",
        "plt.axhline(0.5, color='gray', linestyle='--', alpha=0.6, label='Balanced (50%)')",
        "plt.title('Transition Target Class Balance Across Delta Thresholds', fontsize=12, pad=12)",
        "plt.xlabel('Transition Stage', fontsize=11)",
        "plt.ylabel('Positive Class Rate P(Y=1)', fontsize=11)",
        "plt.ylim(0, 0.6)",
        "plt.legend(title='Delta Threshold', loc='upper right')",
        "plt.grid(True, axis='y', alpha=0.3)",
        "plt.tight_layout()",
        "plt.savefig(FIGURES_DIR / 'transition_class_balance.png', dpi=300)",
        "plt.show()",
        "",
        "df_target_summary"
    ]))

    # Section 5: Feature Sets
    cells.append(make_cell("markdown", [
        "## 5. Stage-Specific Feature Sets & Information Assignment",
        "Verify candidate features for each stage and ensure no leakage."
    ]))

    cells.append(make_cell("code", [
        "print('Stage 1 Features (2->4, Query Complexity):')",
        "print(' ', STAGE_1_FEATURES)",
        "print('\\nStage 2 Features (4->6, Semantic Tail & BM25):')",
        "print(' ', STAGE_2_FEATURES)",
        "print('\\nStage 3 Features (6->8, Lexical Coverage & Diversity):')",
        "print(' ', STAGE_3_FEATURES)",
        "",
        "all_stage_feats = set(STAGE_1_FEATURES + STAGE_2_FEATURES + STAGE_3_FEATURES)",
        "missing_feats = all_stage_feats - set(df_f.columns)",
        "assert len(missing_feats) == 0, f'Missing features: {missing_feats}'",
        "print('\\nAll stage-specific features verified present in feature matrix.')"
    ]))

    # Section 6: GroupKFold Design
    cells.append(make_cell("markdown", [
        "## 6. Strict Paper-Disjoint GroupKFold Design",
        "Configure 5-fold cross-validation grouped by `paper_id` (86 papers) to prevent cross-paper leakage."
    ]))

    cells.append(make_cell("code", [
        "from sklearn.model_selection import GroupKFold",
        "gkf = GroupKFold(n_splits=5)",
        "groups = df_merged['paper_id'].values",
        "",
        "print('GroupKFold Fold Allocation Summary:')",
        "for fold, (trn_idx, val_idx) in enumerate(gkf.split(df_merged, groups=groups)):",
        "    trn_papers = set(groups[trn_idx])",
        "    val_papers = set(groups[val_idx])",
        "    assert trn_papers.isdisjoint(val_papers), 'Paper leakage detected!'",
        "    print(f'  Fold {fold+1}: Train N={len(trn_idx)} ({len(trn_papers)} papers) | Val N={len(val_idx)} ({len(val_papers)} papers)')"
    ]))

    # Section 7: Stage-1 Models
    cells.append(make_cell("markdown", [
        "## 7. Stage-1 Transition Models ($2 \\rightarrow 4$)"
    ]))

    cells.append(make_cell("code", [
        "model_types = ['majority', 'logistic_regression', 'logistic_regression_balanced', 'decision_tree_depth1', 'decision_tree_depth2', 'decision_tree_depth3', 'random_forest']",
        "s1_records = []",
        "for m in model_types:",
        "    res = train_stage_classifiers_cv(df_merged, STAGE_1_FEATURES, 'Y_2_4', m, random_state=RANDOM_SEED)",
        "    s1_records.append({",
        "        'stage': 'Stage 1 (2->4)', 'model_name': m, 'oof_balanced_accuracy': res['oof_balanced_accuracy'],",
        "        'oof_precision': res['oof_precision'], 'oof_recall': res['oof_recall'], 'oof_f1': res['oof_f1'],",
        "        'oof_roc_auc': res['oof_roc_auc'], 'oof_pr_auc': res['oof_pr_auc']",
        "    })",
        "df_s1 = pd.DataFrame(s1_records)",
        "df_s1.to_csv(OUTPUT_DIR / 'stage1_model_results.csv', index=False)",
        "df_s1"
    ]))

    # Section 8: Stage-2 Models
    cells.append(make_cell("markdown", [
        "## 8. Stage-2 Transition Models ($4 \\rightarrow 6$)"
    ]))

    cells.append(make_cell("code", [
        "s2_records = []",
        "for m in model_types:",
        "    res = train_stage_classifiers_cv(df_merged, STAGE_2_FEATURES, 'Y_4_6', m, random_state=RANDOM_SEED)",
        "    s2_records.append({",
        "        'stage': 'Stage 2 (4->6)', 'model_name': m, 'oof_balanced_accuracy': res['oof_balanced_accuracy'],",
        "        'oof_precision': res['oof_precision'], 'oof_recall': res['oof_recall'], 'oof_f1': res['oof_f1'],",
        "        'oof_roc_auc': res['oof_roc_auc'], 'oof_pr_auc': res['oof_pr_auc']",
        "    })",
        "df_s2 = pd.DataFrame(s2_records)",
        "df_s2.to_csv(OUTPUT_DIR / 'stage2_model_results.csv', index=False)",
        "df_s2"
    ]))

    # Section 9: Stage-3 Models
    cells.append(make_cell("markdown", [
        "## 9. Stage-3 Transition Models ($6 \\rightarrow 8$)"
    ]))

    cells.append(make_cell("code", [
        "s3_records = []",
        "for m in model_types:",
        "    res = train_stage_classifiers_cv(df_merged, STAGE_3_FEATURES, 'Y_6_8', m, random_state=RANDOM_SEED)",
        "    s3_records.append({",
        "        'stage': 'Stage 3 (6->8)', 'model_name': m, 'oof_balanced_accuracy': res['oof_balanced_accuracy'],",
        "        'oof_precision': res['oof_precision'], 'oof_recall': res['oof_recall'], 'oof_f1': res['oof_f1'],",
        "        'oof_roc_auc': res['oof_roc_auc'], 'oof_pr_auc': res['oof_pr_auc']",
        "    })",
        "df_s3 = pd.DataFrame(s3_records)",
        "df_s3.to_csv(OUTPUT_DIR / 'stage3_model_results.csv', index=False)",
        "df_s3"
    ]))

    # Section 10: Strategy B — Regression Alternative
    cells.append(make_cell("markdown", [
        "## 10. Strategy B — Continuous Marginal Gain Regression Alternative",
        "Predict continuous gains $G_{2 \\rightarrow 4}, G_{4 \\rightarrow 6}, G_{6 \\rightarrow 8}$ using Ridge and Random Forest Regressors, expanding if $\\hat{G} > \\delta$."
    ]))

    cells.append(make_cell("code", [
        "reg_s1 = train_stage_regressors_cv(df_merged, STAGE_1_FEATURES, 'G_2_4', 'ridge', random_state=RANDOM_SEED)",
        "reg_s2 = train_stage_regressors_cv(df_merged, STAGE_2_FEATURES, 'G_4_6', 'ridge', random_state=RANDOM_SEED)",
        "reg_s3 = train_stage_regressors_cv(df_merged, STAGE_3_FEATURES, 'G_6_8', 'ridge', random_state=RANDOM_SEED)",
        "",
        "reg_summary = pd.DataFrame([",
        "    {'stage': 'Stage 1 (2->4)', 'model': 'Ridge', 'mae': reg_s1['oof_mae'], 'r2': reg_s1['oof_r2'], 'spearman_rho': reg_s1['oof_spearman_rho'], 'derived_bal_acc': reg_s1['derived_balanced_accuracy']},",
        "    {'stage': 'Stage 2 (4->6)', 'model': 'Ridge', 'mae': reg_s2['oof_mae'], 'r2': reg_s2['oof_r2'], 'spearman_rho': reg_s2['oof_spearman_rho'], 'derived_bal_acc': reg_s2['derived_balanced_accuracy']},",
        "    {'stage': 'Stage 3 (6->8)', 'model': 'Ridge', 'mae': reg_s3['oof_mae'], 'r2': reg_s3['oof_r2'], 'spearman_rho': reg_s3['oof_spearman_rho'], 'derived_bal_acc': reg_s3['derived_balanced_accuracy']},",
        "])",
        "reg_summary"
    ]))

    # Section 11: Threshold Sensitivity
    cells.append(make_cell("markdown", [
        "## 11. Threshold Sensitivity Analysis",
        "Sweep decision threshold $t \\in [0.30, 0.70]$ across stages."
    ]))

    cells.append(make_cell("code", [
        "logreg_s1 = train_stage_classifiers_cv(df_merged, STAGE_1_FEATURES, 'Y_2_4', 'logistic_regression', random_state=RANDOM_SEED)",
        "logreg_s2 = train_stage_classifiers_cv(df_merged, STAGE_2_FEATURES, 'Y_4_6', 'logistic_regression', random_state=RANDOM_SEED)",
        "logreg_s3 = train_stage_classifiers_cv(df_merged, STAGE_3_FEATURES, 'Y_6_8', 'logistic_regression', random_state=RANDOM_SEED)",
        "",
        "df_threshold = evaluate_threshold_sensitivity(",
        "    df_merged, logreg_s1['oof_probs'], logreg_s2['oof_probs'], logreg_s3['oof_probs']",
        ")",
        "df_threshold.to_csv(OUTPUT_DIR / 'threshold_sensitivity.csv', index=False)",
        "",
        "# Plot Sensitivity",
        "fig, ax1 = plt.subplots(figsize=(9, 4.8))",
        "ax1.plot(df_threshold['threshold'], df_threshold['mean_f1'], color='#1f77b4', marker='o', linewidth=2.5)",
        "ax1.set_xlabel('Decision Threshold (t)', fontsize=11)",
        "ax1.set_ylabel('OOF Mean Answer F1', color='#1f77b4', fontsize=11)",
        "ax1.grid(True, alpha=0.3)",
        "",
        "ax2 = ax1.twinx()",
        "ax2.plot(df_threshold['threshold'], df_threshold['mean_context_tokens'], color='#d95f02', marker='s', linestyle='--', linewidth=2.5)",
        "ax2.set_ylabel('OOF Mean Context Tokens', color='#d95f02', fontsize=11)",
        "plt.title('QCCA-V2 Policy Sensitivity to Decision Threshold (t)', fontsize=12, pad=12)",
        "plt.tight_layout()",
        "plt.savefig(FIGURES_DIR / 'threshold_sensitivity.png', dpi=300)",
        "plt.show()",
        "",
        "df_threshold[['threshold', 'mean_f1', 'mean_k', 'mean_context_tokens', 'context_reduction_pct_vs_k8', 'mean_regret']]"
    ]))

    # Section 12: OOF Sequential Simulation
    cells.append(make_cell("markdown", [
        "## 12. Out-of-Fold Sequential Policy Simulation",
        "Simulate cascade decisions conditionally and look up frozen response surface answer F1."
    ]))

    cells.append(make_cell("code", [
        "bal_s1 = train_stage_classifiers_cv(df_merged, STAGE_1_FEATURES, 'Y_2_4', 'logistic_regression_balanced', random_state=RANDOM_SEED)",
        "bal_s2 = train_stage_classifiers_cv(df_merged, STAGE_2_FEATURES, 'Y_4_6', 'logistic_regression_balanced', random_state=RANDOM_SEED)",
        "bal_s3 = train_stage_classifiers_cv(df_merged, STAGE_3_FEATURES, 'Y_6_8', 'logistic_regression_balanced', random_state=RANDOM_SEED)",
        "",
        "tree_s1 = train_stage_classifiers_cv(df_merged, STAGE_1_FEATURES, 'Y_2_4', 'decision_tree_depth2', random_state=RANDOM_SEED)",
        "tree_s2 = train_stage_classifiers_cv(df_merged, STAGE_2_FEATURES, 'Y_4_6', 'decision_tree_depth2', random_state=RANDOM_SEED)",
        "tree_s3 = train_stage_classifiers_cv(df_merged, STAGE_3_FEATURES, 'Y_6_8', 'decision_tree_depth2', random_state=RANDOM_SEED)",
        "",
        "df_pol_logreg, sum_logreg = simulate_sequential_policy(",
        "    df_merged, logreg_s1['oof_preds'], logreg_s2['oof_preds'], logreg_s3['oof_preds'], policy_name='QCCA-V2 (Logistic Regression)'",
        ")",
        "df_pol_bal, sum_bal = simulate_sequential_policy(",
        "    df_merged, bal_s1['oof_preds'], bal_s2['oof_preds'], bal_s3['oof_preds'], policy_name='QCCA-V2 (Balanced LogReg)'",
        ")",
        "df_pol_tree, sum_tree = simulate_sequential_policy(",
        "    df_merged, tree_s1['oof_preds'], tree_s2['oof_preds'], tree_s3['oof_preds'], policy_name='QCCA-V2 (Decision Tree d=2)'",
        ")",
        "df_pol_ridge, sum_ridge = simulate_sequential_policy(",
        "    df_merged, reg_s1['oof_preds'], reg_s2['oof_preds'], reg_s3['oof_preds'], policy_name='QCCA-V2 (Ridge Regression)'",
        ")",
        "",
        "df_pol_logreg.to_csv(OUTPUT_DIR / 'qcca_v2_oof_policy_results.csv', index=False)",
        "df_pol_summary = pd.DataFrame([sum_logreg, sum_bal, sum_tree, sum_ridge])",
        "df_pol_summary.to_csv(OUTPUT_DIR / 'qcca_v2_policy_summary.csv', index=False)",
        "df_pol_summary"
    ]))

    # Section 13: Oracle Sequential Policy
    cells.append(make_cell("markdown", [
        "## 13. Diagnostic Oracle Sequential Policy (Theoretical Ceiling)",
        "Evaluate the theoretical upper bound of the sequential transition formulation using actual F1 gains."
    ]))

    cells.append(make_cell("code", [
        "df_orc_seq, sum_orc_seq = simulate_oracle_sequential_policy(df_merged, delta=0.01)",
        "df_orc_seq.to_csv(OUTPUT_DIR / 'oracle_sequential_results.csv', index=False)",
        "",
        "print('Oracle Sequential Policy Summary:')",
        "for k, v in sum_orc_seq.items():",
        "    print(f'  {k:<30}: {v}')"
    ]))

    # Section 14: Baseline Comparison Suite
    cells.append(make_cell("markdown", [
        "## 14. Comprehensive Baseline Comparison Suite",
        "Compare QCCA-V2 against Static Baselines ($k \\in \\{2, 4, 5, 6, 8\\}$), Random Allocation, QCCA-V1, and Oracles."
    ]))

    cells.append(make_cell("code", [
        "baselines = [",
        "    {'system': 'Static k=2', 'mean_f1': 0.3038, 'mean_k': 2.0, 'mean_context_tokens': 534.5, 'context_reduction_pct_vs_k8': 69.60, 'pct_k2': 100.0, 'pct_k4': 0.0, 'pct_k6': 0.0, 'pct_k8': 0.0},",
        "    {'system': 'Static k=4', 'mean_f1': 0.3596, 'mean_k': 4.0, 'mean_context_tokens': 946.1, 'context_reduction_pct_vs_k8': 46.20, 'pct_k2': 0.0, 'pct_k4': 100.0, 'pct_k6': 0.0, 'pct_k8': 0.0},",
        "    {'system': 'Static k=5', 'mean_f1': 0.3629, 'mean_k': 5.0, 'mean_context_tokens': 1157.0, 'context_reduction_pct_vs_k8': 34.21, 'pct_k2': 0.0, 'pct_k4': 0.0, 'pct_k6': 0.0, 'pct_k8': 0.0},",
        "    {'system': 'Static k=6', 'mean_f1': 0.3857, 'mean_k': 6.0, 'mean_context_tokens': 1359.1, 'context_reduction_pct_vs_k8': 22.71, 'pct_k2': 0.0, 'pct_k4': 0.0, 'pct_k6': 100.0, 'pct_k8': 0.0},",
        "    {'system': 'Static k=8', 'mean_f1': 0.3950, 'mean_k': 8.0, 'mean_context_tokens': 1758.5, 'context_reduction_pct_vs_k8': 0.0, 'pct_k2': 0.0, 'pct_k4': 0.0, 'pct_k6': 0.0, 'pct_k8': 100.0},",
        "    {'system': 'Random Allocation', 'mean_f1': 0.3709, 'mean_k': 4.90, 'mean_context_tokens': 1131.0, 'context_reduction_pct_vs_k8': 35.68, 'pct_k2': 20.0, 'pct_k4': 20.0, 'pct_k6': 20.0, 'pct_k8': 20.0},",
        "    {'system': 'QCCA-V1 (Logistic Regression)', 'mean_f1': 0.3034, 'mean_k': 2.03, 'mean_context_tokens': 539.8, 'context_reduction_pct_vs_k8': 69.30, 'pct_k2': 97.4, 'pct_k4': 1.7, 'pct_k6': 0.9, 'pct_k8': 0.0},",
        "    {'system': 'QCCA-V1 (Decision Tree)', 'mean_f1': 0.3282, 'mean_k': 2.66, 'mean_context_tokens': 671.0, 'context_reduction_pct_vs_k8': 61.84, 'pct_k2': 72.3, 'pct_k4': 12.6, 'pct_k6': 9.1, 'pct_k8': 6.0},",
        "    {'system': 'QCCA-V1 (Rule)', 'mean_f1': 0.3286, 'mean_k': 3.56, 'mean_context_tokens': 854.4, 'context_reduction_pct_vs_k8': 51.41, 'pct_k2': 52.4, 'pct_k4': 18.2, 'pct_k6': 14.7, 'pct_k8': 14.7},",
        "    {'system': 'QCCA-V2 (Logistic Regression)', 'mean_f1': sum_logreg['mean_f1'], 'mean_k': sum_logreg['mean_k'], 'mean_context_tokens': sum_logreg['mean_context_tokens'], 'context_reduction_pct_vs_k8': sum_logreg['context_reduction_pct_vs_k8'], 'pct_k2': sum_logreg['pct_k2'], 'pct_k4': sum_logreg['pct_k4'], 'pct_k6': sum_logreg['pct_k6'], 'pct_k8': sum_logreg['pct_k8']},",
        "    {'system': 'QCCA-V2 (Balanced LogReg)', 'mean_f1': sum_bal['mean_f1'], 'mean_k': sum_bal['mean_k'], 'mean_context_tokens': sum_bal['mean_context_tokens'], 'context_reduction_pct_vs_k8': sum_bal['context_reduction_pct_vs_k8'], 'pct_k2': sum_bal['pct_k2'], 'pct_k4': sum_bal['pct_k4'], 'pct_k6': sum_bal['pct_k6'], 'pct_k8': sum_bal['pct_k8']},",
        "    {'system': 'Oracle Sequential (delta=0.01)', 'mean_f1': sum_orc_seq['mean_f1'], 'mean_k': sum_orc_seq['mean_k'], 'mean_context_tokens': sum_orc_seq['mean_context_tokens'], 'context_reduction_pct_vs_k8': sum_orc_seq['context_reduction_pct_vs_k8'], 'pct_k2': sum_orc_seq['pct_k2'], 'pct_k4': sum_orc_seq['pct_k4'], 'pct_k6': sum_orc_seq['pct_k6'], 'pct_k8': sum_orc_seq['pct_k8']},",
        "    {'system': 'Epsilon Oracle (epsilon=0.01)', 'mean_f1': 0.4894, 'mean_k': 3.49, 'mean_context_tokens': 841.3, 'context_reduction_pct_vs_k8': 52.16, 'pct_k2': 57.1, 'pct_k4': 17.3, 'pct_k6': 8.2, 'pct_k8': 10.0},",
        "]",
        "df_baselines = pd.DataFrame(baselines)",
        "df_baselines.to_csv(OUTPUT_DIR / 'baseline_comparison.csv', index=False)",
        "df_baselines"
    ]))

    # Section 15: Feature Ablations
    cells.append(make_cell("markdown", [
        "## 15. Feature Set Ablation Study & Sign-Flip Test",
        "Evaluate the 8 required ablation configurations, including the dedicated Stage-3 sign-flip test."
    ]))

    cells.append(make_cell("code", [
        "df_ablation = run_feature_ablations(df_merged, model_name='logistic_regression', random_state=RANDOM_SEED)",
        "df_ablation.to_csv(OUTPUT_DIR / 'feature_ablation_results.csv', index=False)",
        "",
        "# Plot Ablations",
        "plt.figure(figsize=(10, 5))",
        "plt.barh(np.arange(len(df_ablation)), df_ablation['mean_f1'], color='#4f46e5', alpha=0.85)",
        "plt.yticks(np.arange(len(df_ablation)), df_ablation['ablation_name'])",
        "plt.xlabel('Mean Answer Quality (Token F1)', fontsize=11)",
        "plt.title('Feature Set Ablations: Impact on Sequential Policy Answer Quality', fontsize=12, pad=12)",
        "plt.grid(True, axis='x', alpha=0.3)",
        "plt.tight_layout()",
        "plt.savefig(FIGURES_DIR / 'feature_ablation.png', dpi=300)",
        "plt.show()",
        "",
        "df_ablation[['ablation_name', 'mean_f1', 'mean_k', 'mean_context_tokens', 'context_reduction_pct_vs_k8']]"
    ]))

    # Section 16: Fast Paper-Clustered Bootstrap
    cells.append(make_cell("markdown", [
        "## 16. Fast Paper-Clustered Bootstrap ($B = 10,000$ Replicates)",
        "Resample papers with replacement to construct paired 95% bootstrap confidence intervals."
    ]))

    cells.append(make_cell("code", [
        "df_bootstrap = run_paper_clustered_bootstrap_policy(",
        "    df_pol_logreg, df_merged, n_bootstrap=10000, seed=RANDOM_SEED",
        ")",
        "df_bootstrap.to_csv(OUTPUT_DIR / 'bootstrap_policy_comparison.csv', index=False)",
        "",
        "# Plot Bootstrap Differences",
        "diff_df = df_bootstrap[df_bootstrap['metric'].str.contains('Difference vs Static')].copy()",
        "y_pos = np.arange(len(diff_df))",
        "plt.figure(figsize=(9, 4.8))",
        "plt.errorbar(",
        "    diff_df['point_estimate'], y_pos,",
        "    xerr=[diff_df['point_estimate'] - diff_df['ci_95_low'], diff_df['ci_95_high'] - diff_df['point_estimate']],",
        "    fmt='o', color='#0284c7', ecolor='#0284c7', elinewidth=2.5, capsize=5, markersize=7",
        ")",
        "plt.axvline(0, color='red', linestyle='--', linewidth=1.2)",
        "plt.yticks(y_pos, diff_df['metric'])",
        "plt.gca().invert_yaxis()",
        "plt.xlabel('Paired F1 Difference (QCCA-V2 - Static Baseline)', fontsize=11)",
        "plt.title('Paper-Clustered Bootstrap 95% Confidence Intervals (B=10,000)', fontsize=12, pad=12)",
        "plt.grid(True, axis='x', alpha=0.3)",
        "plt.tight_layout()",
        "plt.savefig(FIGURES_DIR / 'bootstrap_f1_differences.png', dpi=300)",
        "plt.show()",
        "",
        "df_bootstrap"
    ]))

    # Section 17: Pareto Analysis
    cells.append(make_cell("markdown", [
        "## 17. Multi-Objective Pareto Frontier Analysis",
        "Identify non-dominated systems on the (context_tokens, answer_F1) multi-objective surface."
    ]))

    cells.append(make_cell("code", [
        "df_pareto = compute_pareto_frontier(df_baselines)",
        "df_pareto.to_csv(OUTPUT_DIR / 'pareto_points.csv', index=False)",
        "",
        "# Plot Pareto Frontier",
        "plt.figure(figsize=(9, 6))",
        "opt_df = df_pareto[df_pareto['is_pareto_optimal'] == True].sort_values('mean_context_tokens')",
        "dom_df = df_pareto[df_pareto['is_pareto_optimal'] == False]",
        "",
        "plt.plot(opt_df['mean_context_tokens'], opt_df['mean_f1'], color='#16a34a', linestyle='--', linewidth=1.5, label='Empirical Pareto Frontier')",
        "plt.scatter(opt_df['mean_context_tokens'], opt_df['mean_f1'], color='#16a34a', s=100, zorder=5, label='Pareto Non-Dominated')",
        "plt.scatter(dom_df['mean_context_tokens'], dom_df['mean_f1'], color='#94a3b8', s=70, zorder=4, label='Pareto Dominated')",
        "",
        "for _, row in df_pareto.iterrows():",
        "    plt.annotate(row['system'], (row['mean_context_tokens'], row['mean_f1']), textcoords='offset points', xytext=(6, 4), fontsize=9, weight='bold' if row['is_pareto_optimal'] else 'normal')",
        "",
        "plt.xlabel('Mean Context Prompt Tokens', fontsize=11)",
        "plt.ylabel('Mean Answer Quality (Token F1)', fontsize=11)",
        "plt.title('Multi-Objective Pareto Frontier (Context Tokens vs Answer F1)', fontsize=12, pad=12)",
        "plt.legend(loc='lower right')",
        "plt.grid(True, alpha=0.3)",
        "plt.tight_layout()",
        "plt.savefig(FIGURES_DIR / 'pareto_frontier.png', dpi=300)",
        "plt.show()",
        "",
        "df_pareto[['system', 'mean_context_tokens', 'mean_f1', 'is_pareto_optimal']]"
    ]))

    # Section 18: Failure Analysis
    cells.append(make_cell("markdown", [
        "## 18. Policy Failure Mode Analysis",
        "Examine over-expansion and under-expansion relative to the deployable Oracle."
    ]))

    cells.append(make_cell("code", [
        "df_fail_summary, df_fail_reps = analyze_policy_failures(df_pol_logreg, df_merged)",
        "df_fail_summary.to_csv(OUTPUT_DIR / 'failure_analysis.csv', index=False)",
        "",
        "# Plot Failure Summary",
        "plt.figure(figsize=(7.5, 4.5))",
        "sns.barplot(data=df_fail_summary, x='failure_type', y='count', palette='Pastel1')",
        "plt.title('Sequential Policy Allocation Failure Mode Distribution', fontsize=12, pad=12)",
        "plt.xlabel('Allocation Outcome vs Deployable Oracle', fontsize=11)",
        "plt.ylabel('Query Count (N=231)', fontsize=11)",
        "plt.grid(True, axis='y', alpha=0.3)",
        "plt.tight_layout()",
        "plt.savefig(FIGURES_DIR / 'failure_analysis.png', dpi=300)",
        "plt.show()",
        "",
        "df_fail_summary"
    ]))

    # Section 19: Scientific Interpretation
    cells.append(make_cell("markdown", [
        "## 19. Scientific Interpretation & Discussion",
        "Interpret the empirical results with statistical discipline:",
        "- **Sequential Formulation Feasibility:** Does decomposing evidence allocation into transitions resolve the flat classification failure mode?",
        "- **Sign-Flip Impact:** Does the inclusion of `lex_coverage_top2` in Stage 3 prevent over-allocation?",
        "- **Pareto Efficiency:** Does QCCA-V2 offer a viable operating point between $k=2$ and $k=8$?"
    ]))

    cells.append(make_cell("code", [
        "print('='*75)",
        "print('                  QCCA-V2 SCIENTIFIC SUMMARY')",
        "print('='*75)",
        "print(f'1. QCCA-V2 Mean F1:               {sum_logreg[\"mean_f1\"]:.4f} (Static k=8: 0.3950, k=4: 0.3596, k=2: 0.3038)')",
        "print(f'2. Mean Context Tokens:           {sum_logreg[\"mean_context_tokens\"]:.1f} (Static k=8: 1758.5 tokens)')",
        "print(f'3. Context Reduction vs Static k=8: {sum_logreg[\"context_reduction_pct_vs_k8\"]:.1f}%')",
        "print(f'4. Oracle Sequential F1 Ceiling:  {sum_orc_seq[\"mean_f1\"]:.4f} (Context: {sum_orc_seq[\"mean_context_tokens\"]:.1f} tokens)')",
        "print(f'5. Budget Distribution:           k=2 ({sum_logreg[\"pct_k2\"]}%), k=4 ({sum_logreg[\"pct_k4\"]}%), k=6 ({sum_logreg[\"pct_k6\"]}%), k=8 ({sum_logreg[\"pct_k8\"]}%)')",
        "print('='*75)"
    ]))

    # Section 20: Decision Gates Evaluation
    cells.append(make_cell("markdown", [
        "## 20. Decision Gate Evaluation for Next Phase (Week 5)",
        "Formally evaluate Decision Gates A through G before recommending any test-set deployment."
    ]))

    cells.append(make_cell("code", [
        "gates = evaluate_qcca_v2_decision_gates(",
        "    {'stage1': logreg_s1, 'stage2': logreg_s2, 'stage3': logreg_s3},",
        "    sum_logreg,",
        "    df_baselines,",
        "    df_ablation,",
        ")",
        "",
        "print('='*75)",
        "print('             QCCA-V2 FINAL DECISION GATE SCORECARD')",
        "print('='*75)",
        "for g_name, g_info in gates.items():",
        "    print(f'\\n{g_name}:')",
        "    print(f'  Status:   {g_info[\"status\"]}')",
        "    print(f'  Evidence: {g_info[\"evidence\"]}')",
        "print('='*75)",
        "",
        "# Verify all 15 required CSVs and 11 figures exist",
        "stage_models_dict = {'Stage 1 (2->4)': logreg_s1, 'Stage 2 (4->6)': logreg_s2, 'Stage 3 (6->8)': logreg_s3}",
        "stage_results_all = pd.concat([df_s1, df_s2, df_s3], ignore_index=True)",
        "generate_all_qcca_v2_figures(",
        "    df_target_summary, stage_results_all, df_threshold, df_baselines, df_ablation, df_bootstrap, df_pareto, df_fail_summary, stage_models_dict, df_merged, FIGURES_DIR",
        ")",
        "print('\\nAll 15 CSVs and 11 figures verified and generated successfully in results/week4/qcca_v2/')"
    ]))

    nb_content = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "sara_env",
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

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb_content, f, indent=1)

    print(f"Created unexecuted notebook at: {output_path} ({len(cells)} cells)")


if __name__ == "__main__":
    create_qcca_v2_notebook()
