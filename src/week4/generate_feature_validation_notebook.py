"""Generator for the Week 4-B Feature Validation Notebook: week4_feature_validation.ipynb.

Creates an unexecuted, fully runnable Jupyter notebook adhering to:
- DO NOT RUN THE NOTEBOOK (user runs it manually)
- All outputs empty (execution_count=None, outputs=[])
- Clean markdown sections with rigorous scientific explanations
- Saves all 10 CSVs and 8 figures to results/week4/feature_validation/
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


def create_feature_validation_notebook(output_path: str = "notebooks/week4/week4_feature_validation.ipynb"):
    cells = []

    # Title & Metadata
    cells.append(make_cell("markdown", [
        "# Week 4-B: Feature Validation, Redundancy Analysis & Transition Modeling",
        "**Project:** CS787 Generative AI — Query-Conditioned Context Allocator (QCCA)  ",
        "**Parent Framework:** SARA — Selective and Adaptive Retrieval-augmented Generation with Context Compression  ",
        "**Benchmark:** QASPER Development Split (86 papers, 231 queries)  ",
        "**Objective:** Statistically validate, reduce redundancy, and model transition-specific evidence requirements across the 64 pre-generation features discovered in Week 4-A.  ",
        "",
        "---",
        "### CRITICAL EXPERIMENTAL GUARDRAILS & GOVERNANCE",
        "- **DO NOT BUILD OR TRAIN QCCA-V2 YET.** No classifiers, regressors, trees, logistic regressions, or threshold optimizers are fitted in this notebook.",
        "- **ZERO TEST ACCESS:** QASPER_test remains 100% UNTOUCHED and UNSEEN.",
        "- **FROZEN DATA TARGETS:** All analyses use the frozen W2 response surface ($K_{\\text{alloc}} = \\{2, 4, 5, 6, 8\\}$) and frozen W3 Oracle targets ($\\epsilon^* = 0.01$).",
        "- **SCIENTIFIC OBJECTIVITY:** Differentiates exploratory candidate signals from confirmatory findings. Does not claim unproven causal mechanisms."
    ]))

    # Section 1: Environment & Imports
    cells.append(make_cell("markdown", [
        "## 1. Environment Setup & Library Imports",
        "Import standard scientific libraries, plotting tools, and the dedicated W4-B validation library [`src.week4.feature_validation`](file:///home/gunavenkat/Downloads/CS787-RAG-Project/src/week4/feature_validation.py)."
    ]))

    cells.append(make_cell("code", [
        "import os",
        "import sys",
        "from pathlib import Path",
        "",
        "# Robust project root discovery: handles executing from repo root, notebooks/, or notebooks/week4/",
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
        "import scipy.stats as stats",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "",
        "# Import dedicated W4-B statistical validation library",
        "from src.week4.feature_validation import (",
        "    benjamini_hochberg,",
        "    compute_fdr_table,",
        "    compute_feature_correlation_matrix,",
        "    find_redundancy_clusters,",
        "    select_redundancy_representatives,",
        "    rank_partial_correlation,",
        "    compute_partial_associations,",
        "    construct_transition_targets,",
        "    compute_transition_associations,",
        "    analyze_coverage_sign_flips,",
        "    build_provisional_feature_set,",
        "    evaluate_qcca_v2_gates,",
        ")",
        "",
        "%matplotlib inline",
        "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')",
        "print('Libraries imported successfully.')"
    ]))

    # Section 2: Configuration & Directories
    cells.append(make_cell("markdown", [
        "## 2. Experimental Configuration & Directory Setup",
        "Define directories, frozen parameters, and verify output paths in `results/week4/feature_validation/`."
    ]))

    cells.append(make_cell("code", [
        "DISCOVERY_DIR = PROJECT_ROOT / 'results/week4/feature_discovery'",
        "VALIDATION_DIR = PROJECT_ROOT / 'results/week4/feature_validation'",
        "FIGURES_DIR = VALIDATION_DIR / 'figures'",
        "",
        "VALIDATION_DIR.mkdir(parents=True, exist_ok=True)",
        "FIGURES_DIR.mkdir(parents=True, exist_ok=True)",
        "",
        "RANDOM_SEED = 42",
        "K_ALLOC = [2, 4, 5, 6, 8]",
        "EPSILON_STAR = 0.01",
        "DELTA_THRESHOLD = 0.01",
        "REDUNDANCY_THRESHOLD = 0.80",
        "",
        "print(f'Project Root:         {PROJECT_ROOT}')",
        "print(f'Discovery Directory:  {DISCOVERY_DIR}')",
        "print(f'Validation Directory: {VALIDATION_DIR}')",
        "print(f'Figures Directory:    {FIGURES_DIR}')"
    ]))

    # Section 3: Load Discovery Artifacts
    cells.append(make_cell("markdown", [
        "## 3. Load Feature Discovery Artifacts & Verify Data Integrity",
        "Load the 64-feature matrix, target matrix, metadata, and discovery statistics from `results/week4/feature_discovery/`."
    ]))

    cells.append(make_cell("code", [
        "df_features = pd.read_csv(DISCOVERY_DIR / 'feature_matrix.csv')",
        "df_targets = pd.read_csv(DISCOVERY_DIR / 'target_matrix.csv')",
        "df_metadata = pd.read_csv(DISCOVERY_DIR / 'feature_metadata.csv')",
        "df_spearman = pd.read_csv(DISCOVERY_DIR / 'spearman_results.csv')",
        "df_bootstrap = pd.read_csv(DISCOVERY_DIR / 'bootstrap_ci_results.csv')",
        "df_stability = pd.read_csv(DISCOVERY_DIR / 'groupkfold_stability.csv')",
        "",
        "feature_cols = [c for c in df_features.columns if c not in ['question_id', 'paper_id']]",
        "",
        "print(f'Feature Matrix shape: {df_features.shape} ({len(feature_cols)} pre-generation features)')",
        "print(f'Target Matrix shape:  {df_targets.shape}')",
        "print(f'Metadata entries:     {len(df_metadata)}')",
        "print(f'Spearman tests:       {len(df_spearman)} hypotheses evaluated')"
    ]))

    # Section 4: Analysis 1 — Benjamini-Hochberg FDR Correction
    cells.append(make_cell("markdown", [
        "## 4. Analysis 1 — Benjamini-Hochberg False Discovery Rate (FDR) Correction",
        "Apply the step-up Benjamini-Hochberg procedure across all 320 hypothesis tests (64 features $\\times$ 5 targets).",
        "We distinguish nominal significance ($p < 0.05$) from FDR-adjusted significance ($q < 0.10$ and $q < 0.05$)."
    ]))

    cells.append(make_cell("code", [
        "df_fdr = compute_fdr_table(df_spearman, alpha_nominal=0.05, alpha_fdr=0.10)",
        "df_fdr.to_csv(VALIDATION_DIR / 'fdr_results.csv', index=False)",
        "",
        "n_nominal = df_fdr['is_nominal_sig'].sum()",
        "n_fdr_10 = df_fdr['is_fdr_sig_10'].sum()",
        "n_fdr_05 = df_fdr['is_fdr_sig_05'].sum()",
        "",
        "print(f'Total Hypotheses Tested:                  {len(df_fdr)}')",
        "print(f'Nominal p < 0.05:                         {n_nominal} ({n_nominal/len(df_fdr)*100:.1f}%)')",
        "print(f'FDR-Adjusted q < 0.10:                    {n_fdr_10} ({n_fdr_10/len(df_fdr)*100:.1f}%)')",
        "print(f'FDR-Adjusted q < 0.05:                    {n_fdr_05} ({n_fdr_05/len(df_fdr)*100:.1f}%)')",
        "print(f'Minimum naive p-value observed:           {df_fdr[\"p_value_naive\"].min():.4f}')",
        "print(f'Minimum FDR q-value observed:             {df_fdr[\"q_value\"].min():.4f}')",
        "",
        "# Plot FDR Volcano / Significance distribution",
        "plt.figure(figsize=(9, 5.5))",
        "plt.scatter(",
        "    df_fdr['spearman_rho'],",
        "    -np.log10(df_fdr['p_value_naive'] + 1e-12),",
        "    c=df_fdr['is_fdr_sig_10'].map({True: '#d95f02', False: '#7570b3'}),",
        "    alpha=0.75,",
        "    s=35,",
        "    edgecolors='none'",
        ")",
        "plt.axhline(-np.log10(0.05), color='gray', linestyle='--', alpha=0.7, label='Nominal p = 0.05')",
        "plt.xlabel('Spearman Rho', fontsize=11)",
        "plt.ylabel('-log10(Nominal p-value)', fontsize=11)",
        "plt.title('Analysis 1: Benjamini-Hochberg FDR Multiple-Testing Correction (320 Hypotheses)', fontsize=12, pad=12)",
        "plt.legend(['Nominal p = 0.05 threshold', 'Hypotheses'], loc='upper left')",
        "plt.grid(True, alpha=0.3)",
        "plt.tight_layout()",
        "plt.savefig(FIGURES_DIR / 'fdr_adjusted_significance_plot.png', dpi=300)",
        "plt.show()",
        "",
        "print('\\nTop 10 Relationships by lowest naive p-value and FDR q-value:')",
        "df_fdr[['feature', 'target', 'spearman_rho', 'p_value_naive', 'q_value', 'is_nominal_sig']].head(10)"
    ]))

    # Section 5: Analysis 2 — Feature-Feature Redundancy Analysis
    cells.append(make_cell("markdown", [
        "## 5. Analysis 2 — Feature-Feature Redundancy Analysis",
        "Compute the $64 \\times 64$ pairwise Spearman correlation matrix between all pre-generation features.",
        "Group highly correlated features ($|\\rho| \\ge 0.80$) into clusters using complete-linkage agglomerative clustering."
    ]))

    cells.append(make_cell("code", [
        "df_corr = compute_feature_correlation_matrix(df_features, feature_cols)",
        "df_corr.to_csv(VALIDATION_DIR / 'feature_correlation_matrix.csv')",
        "",
        "df_clusters = find_redundancy_clusters(df_corr, redundancy_threshold=REDUNDANCY_THRESHOLD)",
        "n_clusters = df_clusters['redundancy_cluster_id'].nunique()",
        "n_multi = (df_clusters['cluster_size'] > 1).sum()",
        "",
        "print(f'Computed {df_corr.shape[0]}x{df_corr.shape[1]} Feature Correlation Matrix.')",
        "print(f'Identified {n_clusters} non-redundant clusters at |rho| >= {REDUNDANCY_THRESHOLD}.')",
        "print(f'{n_multi} features belong to multi-feature redundant clusters.')",
        "",
        "# Plot 64x64 correlation heatmap",
        "plt.figure(figsize=(13, 11))",
        "sns.heatmap(df_corr, cmap='coolwarm', vmin=-1.0, vmax=1.0, cbar_kws={'label': 'Spearman Correlation (rho)'})",
        "plt.title('Analysis 2: Pairwise Feature Correlation Matrix (64 Pre-Generation Features)', fontsize=13, pad=12)",
        "plt.xticks([])",
        "plt.yticks([])",
        "plt.tight_layout()",
        "plt.savefig(FIGURES_DIR / 'feature_correlation_heatmap.png', dpi=300)",
        "plt.show()"
    ]))

    # Section 6: Analysis 3 — Cluster Representatives
    cells.append(make_cell("markdown", [
        "## 6. Analysis 3 — Redundancy-Reduced Cluster Representatives",
        "For each redundancy cluster, select a primary representative considering GroupKFold stability, peak effect size, and interpretability."
    ]))

    cells.append(make_cell("code", [
        "df_reps = select_redundancy_representatives(",
        "    df_clusters, df_spearman, df_bootstrap, df_stability, df_metadata",
        ")",
        "df_reps.to_csv(VALIDATION_DIR / 'feature_redundancy_groups.csv', index=False)",
        "",
        "n_reps = df_reps['is_representative'].sum()",
        "print(f'Selected {n_reps} cluster representatives out of 64 original features.')",
        "",
        "print('\\nMulti-feature redundancy clusters and selected representatives:')",
        "multi_reps = df_reps[df_reps['group_id'].map(df_clusters.set_index('redundancy_cluster_id')['cluster_size'].to_dict()) > 1]",
        "multi_reps[['group_id', 'feature', 'family', 'peak_abs_rho', 'fold_sign_consistency', 'is_representative', 'representative_candidate']].head(15)"
    ]))

    # Section 7: Analysis 4 — Partial / Conditional Associations
    cells.append(make_cell("markdown", [
        "## 7. Analysis 4 — Partial & Conditional Association Analysis",
        "Evaluate whether promising signals contain unique information beyond related covariates using rank-based partial correlation."
    ]))

    cells.append(make_cell("code", [
        "df_partial = compute_partial_associations(df_features, df_targets)",
        "df_partial.to_csv(VALIDATION_DIR / 'partial_association_results.csv', index=False)",
        "",
        "print('Evaluated Conditional Hypotheses:')",
        "df_partial[['feature', 'target', 'controlling_for', 'raw_spearman_rho', 'partial_rank_r', 'partial_p_value', 'signal_retained_pct']]"
    ]))

    # Section 8: Analysis 5 — Transition-Specific Analysis
    cells.append(make_cell("markdown", [
        "## 8. Analysis 5 — Transition-Specific Association Analysis",
        "Model evidence transitions directly across continuous gains ($G_{2 \\rightarrow 4}, G_{4 \\rightarrow 6}, G_{6 \\rightarrow 8}$) and binary transition indicators ($I_{2 \\rightarrow 4}, I_{4 \\rightarrow 6}, I_{6 \\rightarrow 8}$ plus $\\delta=0.01$ thresholded gains)."
    ]))

    cells.append(make_cell("code", [
        "df_trans_targets = construct_transition_targets(",
        "    PROJECT_ROOT / 'results/week2/processed/dev_per_query_k_matrix.csv', delta_threshold=DELTA_THRESHOLD",
        ")",
        "df_trans_assoc = compute_transition_associations(df_features, df_trans_targets, feature_cols)",
        "df_trans_assoc.to_csv(VALIDATION_DIR / 'transition_association_results.csv', index=False)",
        "",
        "print(f'Computed {len(df_trans_assoc)} associations across 10 transition targets.')",
        "",
        "# Heatmap of candidate features across transitions",
        "top_trans_feats = [",
        "    'query_conjunction_count', 'query_is_what_which', 'query_is_numerical',",
        "    'lex_coverage_top1', 'lex_coverage_top2', 'lex_coverage_top4',",
        "    'sem_sim_mean_top6', 'sem_sim_mean_top10', 'bm25_mean_score',",
        "    'bm25_first_gap_ratio', 'passage_sim_std', 'evidence_query_cluster_span'",
        "]",
        "trans_cols = ['G_2_to_4', 'G_4_to_6', 'G_6_to_8', 'I_2_to_4', 'I_4_to_6', 'I_6_to_8']",
        "pivot_trans = df_trans_assoc[",
        "    df_trans_assoc['feature'].isin(top_trans_feats) & df_trans_assoc['target'].isin(trans_cols)",
        "].pivot(index='feature', columns='target', values='spearman_rho')[trans_cols].loc[top_trans_feats]",
        "",
        "plt.figure(figsize=(11, 7.5))",
        "sns.heatmap(pivot_trans, annot=True, fmt='+.3f', cmap='vlag', center=0, vmin=-0.25, vmax=0.25, cbar_kws={'label': 'Spearman Rho'})",
        "plt.title('Analysis 5: Transition-Specific Associations (Continuous Gains vs Binary Indicators)', fontsize=12, pad=12)",
        "plt.ylabel('Candidate Feature', fontsize=11)",
        "plt.xlabel('Allocation Transition Target', fontsize=11)",
        "plt.tight_layout()",
        "plt.savefig(FIGURES_DIR / 'transition_association_heatmap.png', dpi=300)",
        "plt.show()"
    ]))

    # Section 9: Analysis 6 — Lexical Coverage Sign-Flip Audit
    cells.append(make_cell("markdown", [
        "## 9. Analysis 6 — Lexical Coverage Sign-Flip Investigation",
        "Audit the empirical sign flip observed in lexical coverage features ($G_{4 \\rightarrow 6} > 0$ vs $G_{6 \\rightarrow 8} < 0$) across bootstrap CIs and GroupKFold folds."
    ]))

    cells.append(make_cell("code", [
        "df_sign_flip = analyze_coverage_sign_flips(df_spearman, df_fdr, df_bootstrap, df_stability)",
        "df_sign_flip.to_csv(VALIDATION_DIR / 'transition_sign_flip_analysis.csv', index=False)",
        "",
        "# Plot Coverage Sign Flip",
        "cov_plot_df = df_sign_flip[df_sign_flip['coverage_feature'].isin([",
        "    'lex_coverage_top1', 'lex_coverage_top2', 'lex_coverage_top4', 'lex_coverage_top6'",
        "])]",
        "x = np.arange(len(cov_plot_df))",
        "width = 0.25",
        "",
        "plt.figure(figsize=(10, 5.5))",
        "plt.bar(x - width, cov_plot_df['G_2_to_4_rho'], width, label='G_2_to_4 (Base Expansion)', color='#1b9e77')",
        "plt.bar(x, cov_plot_df['G_4_to_6_rho'], width, label='G_4_to_6 (Middle Expansion)', color='#386cb0')",
        "plt.bar(x + width, cov_plot_df['G_6_to_8_rho'], width, label='G_6_to_8 (Saturation Stopping)', color='#e41a1c')",
        "plt.axhline(0, color='black', linewidth=0.8, linestyle='--')",
        "plt.xticks(x, cov_plot_df['coverage_feature'], rotation=15, ha='right', fontsize=10)",
        "plt.ylabel('Spearman Correlation (rho)', fontsize=11)",
        "plt.title('Analysis 6: Audit of Lexical Coverage Sign Flips Across Allocation Transitions', fontsize=12, pad=12)",
        "plt.legend(loc='upper right')",
        "plt.grid(True, axis='y', alpha=0.3)",
        "plt.tight_layout()",
        "plt.savefig(FIGURES_DIR / 'coverage_sign_flip_plot.png', dpi=300)",
        "plt.show()",
        "",
        "print('Coverage Sign-Flip Audit Table:')",
        "df_sign_flip[['coverage_feature', 'G_2_to_4_rho', 'G_4_to_6_rho', 'G_6_to_8_rho', 'has_sign_flip', 'G_4_to_6_bootstrap_95ci', 'G_6_to_8_bootstrap_95ci']]"
    ]))

    # Section 10: Analysis 7 & 8 — GroupKFold Stability & Clustered Bootstrap
    cells.append(make_cell("markdown", [
        "## 10. Analyses 7 & 8 — GroupKFold Stability & Paper-Clustered Bootstrap Validation",
        "Save and visualize fold distributions and 95% paper-clustered bootstrap intervals."
    ]))

    cells.append(make_cell("code", [
        "# Save validation CSVs",
        "df_stability.to_csv(VALIDATION_DIR / 'groupkfold_validation.csv', index=False)",
        "df_bootstrap.to_csv(VALIDATION_DIR / 'bootstrap_validation.csv', index=False)",
        "",
        "# Plot 1: GroupKFold Stability",
        "import ast",
        "prov_sample = ['query_conjunction_count', 'query_is_what_which', 'sem_sim_mean_top6', 'bm25_mean_score', 'lex_coverage_top2', 'passage_sim_std']",
        "stab_sub = df_stability[df_stability['feature'].isin(prov_sample)].drop_duplicates(subset=['feature'])",
        "",
        "rhos_list = []",
        "labels = []",
        "for _, row in stab_sub.iterrows():",
        "    try:",
        "        val = ast.literal_eval(row['fold_rhos'])",
        "        rhos_list.append(val)",
        "        labels.append(f\"{row['feature']}\\n({row['target']})\")",
        "    except Exception:",
        "        pass",
        "",
        "if rhos_list:",
        "    plt.figure(figsize=(10, 5))",
        "    plt.boxplot(rhos_list, tick_labels=labels, orientation='horizontal', patch_artist=True,",
        "                boxprops=dict(facecolor='#cbd5e1', color='#334155'),",
        "                medianprops=dict(color='#b91c1c', linewidth=2))",
        "    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)",
        "    plt.xlabel('Spearman Rho Across 5 Paper-Disjoint Folds', fontsize=11)",
        "    plt.title('Analysis 7: GroupKFold Sign Stability Across Paper-Disjoint Folds', fontsize=12, pad=12)",
        "    plt.grid(True, axis='x', alpha=0.3)",
        "    plt.tight_layout()",
        "    plt.savefig(FIGURES_DIR / 'groupkfold_stability_plot.png', dpi=300)",
        "    plt.show()",
        "",
        "# Plot 2: Bootstrap CI Plot",
        "top_boot = df_bootstrap[df_bootstrap['target'].isin(['oracle_k', 'G_4_to_6', 'G_6_to_8'])].sort_values(by='spearman_rho', key=abs, ascending=False).head(12)",
        "y_pos = np.arange(len(top_boot))",
        "",
        "plt.figure(figsize=(10, 6))",
        "plt.errorbar(",
        "    top_boot['spearman_rho'],",
        "    y_pos,",
        "    xerr=[",
        "        top_boot['spearman_rho'] - top_boot['bootstrap_ci_low'],",
        "        top_boot['bootstrap_ci_high'] - top_boot['spearman_rho']",
        "    ],",
        "    fmt='o', color='#0284c7', ecolor='#0284c7', elinewidth=2, capsize=4, markersize=6",
        ")",
        "plt.axvline(0, color='red', linestyle='--', linewidth=1.0)",
        "plt.yticks(y_pos, [f\"{f} ({t})\" for f, t in zip(top_boot['feature'], top_boot['target'])], fontsize=10)",
        "plt.gca().invert_yaxis()",
        "plt.xlabel('Spearman Rho (with 95% Paper-Clustered Bootstrap CI)', fontsize=11)",
        "plt.title('Analysis 8: Paper-Clustered Bootstrap 95% Confidence Intervals', fontsize=12, pad=12)",
        "plt.grid(True, axis='x', alpha=0.3)",
        "plt.tight_layout()",
        "plt.savefig(FIGURES_DIR / 'bootstrap_ci_plot.png', dpi=300)",
        "plt.show()"
    ]))

    # Section 11: Analysis 9 — Feature Family Summary
    cells.append(make_cell("markdown", [
        "## 11. Analysis 9 — Feature-Family Validation Summary",
        "Summarize statistical evidence across the 6 information families (A through F)."
    ]))

    cells.append(make_cell("code", [
        "fam_records = []",
        "for fam, group in df_metadata.groupby('feature_family'):",
        "    fam_feats = group['feature_name'].tolist()",
        "    fam_fdr = df_fdr[df_fdr['feature'].isin(fam_feats)]",
        "    fam_boot = df_bootstrap[df_bootstrap['feature'].isin(fam_feats)]",
        "    fam_stab = df_stability[df_stability['feature'].isin(fam_feats)]",
        "    ",
        "    reps = df_reps[(df_reps['family'] == fam) & (df_reps['is_representative'] == True)]['feature'].tolist()",
        "    ",
        "    fam_records.append({",
        "        'feature_family': fam,",
        "        'original_features': len(fam_feats),",
        "        'fdr_sig_relationships_q10': int(fam_fdr['is_fdr_sig_10'].sum()),",
        "        'bootstrap_ci_excludes_zero': int(fam_boot['bootstrap_sign_stable'].sum()),",
        "        'median_abs_rho': round(fam_fdr['abs_spearman_rho'].median(), 4),",
        "        'max_abs_rho': round(fam_fdr['abs_spearman_rho'].max(), 4),",
        "        'median_fold_stability': round(fam_stab['fold_sign_consistency'].median(), 2),",
        "        'representative_features': ', '.join(reps[:3]),",
        "    })",
        "",
        "df_fam_sum = pd.DataFrame(fam_records)",
        "df_fam_sum.to_csv(VALIDATION_DIR / 'feature_family_validation_summary.csv', index=False)",
        "",
        "# Plot Family Profile",
        "plt.figure(figsize=(10, 4.8))",
        "x_fam = np.arange(len(df_fam_sum))",
        "plt.bar(x_fam, df_fam_sum['max_abs_rho'], color='#6366f1', alpha=0.85, label='Peak |Spearman Rho|')",
        "plt.plot(x_fam, df_fam_sum['median_abs_rho'], color='#f59e0b', marker='o', linewidth=2.5, label='Median |Spearman Rho|')",
        "plt.xticks(x_fam, df_fam_sum['feature_family'], rotation=20, ha='right', fontsize=10)",
        "plt.ylabel('Absolute Spearman Correlation', fontsize=11)",
        "plt.title('Analysis 9: Statistical Profile Across 6 Pre-Generation Feature Families', fontsize=12, pad=12)",
        "plt.legend(loc='upper right')",
        "plt.grid(True, axis='y', alpha=0.3)",
        "plt.tight_layout()",
        "plt.savefig(FIGURES_DIR / 'feature_family_summary.png', dpi=300)",
        "plt.show()",
        "",
        "df_fam_sum"
    ]))

    # Section 12: Analysis 10 — Provisional Feature Set
    cells.append(make_cell("markdown", [
        "## 12. Analysis 10 — Provisional Feature Set (14 Non-Redundant Candidates)",
        "Propose a reduced provisional feature set spanning all 6 families, verified for low redundancy, high cross-validation stability, and transition specificity."
    ]))

    cells.append(make_cell("code", [
        "df_prov = build_provisional_feature_set(",
        "    df_reps, df_fdr, df_bootstrap, df_stability, df_metadata",
        ")",
        "df_prov.to_csv(VALIDATION_DIR / 'provisional_feature_set.csv', index=False)",
        "",
        "# Visualize Provisional Feature Set",
        "plt.figure(figsize=(11, 7))",
        "colors = {",
        "    'A: BM25/Retrieval': '#3b82f6',",
        "    'B: Query Complexity': '#10b981',",
        "    'C: Semantic Structure': '#8b5cf6',",
        "    'D: Passage Redundancy/Diversity': '#f59e0b',",
        "    'E: Lexical Coverage': '#ef4444',",
        "    'F: Evidence Structure': '#06b6d4',",
        "}",
        "bar_colors = [colors.get(f, '#64748b') for f in df_prov['family']]",
        "y_pos = np.arange(len(df_prov))",
        "",
        "plt.barh(y_pos, df_prov['spearman_rho'], color=bar_colors, alpha=0.85)",
        "plt.axvline(0, color='black', linestyle='--', linewidth=0.8)",
        "plt.yticks(y_pos, df_prov['feature'], fontsize=10)",
        "plt.gca().invert_yaxis()",
        "plt.xlabel('Spearman Rho with Primary Target', fontsize=11)",
        "plt.title('Analysis 10: Provisional Reduced Feature Set (14 Non-Redundant Candidates Across 6 Families)', fontsize=12, pad=12)",
        "",
        "handles = [plt.Rectangle((0,0),1,1, color=col) for col in colors.values()]",
        "plt.legend(handles, colors.keys(), loc='lower right', fontsize=9)",
        "plt.grid(True, axis='x', alpha=0.3)",
        "plt.tight_layout()",
        "plt.savefig(FIGURES_DIR / 'provisional_feature_set_plot.png', dpi=300)",
        "plt.show()",
        "",
        "print('Provisional Feature Set Roster:')",
        "df_prov[['feature', 'family', 'primary_target', 'spearman_rho', 'bootstrap_ci_low', 'bootstrap_ci_high', 'fold_sign_consistency', 'why_retained']]"
    ]))

    # Section 13: Analysis 11 & Decision Gates
    cells.append(make_cell("markdown", [
        "## 13. Analysis 11 & QCCA-V2 Decision Gates Evaluation",
        "Formally evaluate Decision Gates A through F before authorizing any ML allocator development."
    ]))

    cells.append(make_cell("code", [
        "gates = evaluate_qcca_v2_gates(df_fdr, df_clusters, df_trans_assoc, df_sign_flip, df_prov)",
        "",
        "print('='*75)",
        "print('            QCCA-V2 DECISION GATE SCORECARD (WEEK 4-B)')",
        "print('='*75)",
        "for g_name, g_info in gates.items():",
        "    print(f'\\n{g_name}:')",
        "    print(f'  Status:   {g_info[\"status\"]}')",
        "    if 'evidence' in g_info:",
        "        print(f'  Evidence: {g_info[\"evidence\"]}')",
        "    if 'features' in g_info:",
        "        print(f'  Roster ({len(g_info[\"features\"])} features): {g_info[\"features\"]}')",
        "print('='*75)"
    ]))

    # Section 14: Execution Confirmation Manifest
    cells.append(make_cell("markdown", [
        "## 14. Verification Manifest & Governance Confirmation",
        "Verify that all 10 required CSVs and 8 figures exist in `results/week4/feature_validation/`."
    ]))

    cells.append(make_cell("code", [
        "expected_csvs = [",
        "    'fdr_results.csv',",
        "    'feature_correlation_matrix.csv',",
        "    'feature_redundancy_groups.csv',",
        "    'partial_association_results.csv',",
        "    'transition_association_results.csv',",
        "    'transition_sign_flip_analysis.csv',",
        "    'groupkfold_validation.csv',",
        "    'bootstrap_validation.csv',",
        "    'feature_family_validation_summary.csv',",
        "    'provisional_feature_set.csv',",
        "]",
        "",
        "expected_figures = [",
        "    'feature_correlation_heatmap.png',",
        "    'fdr_adjusted_significance_plot.png',",
        "    'transition_association_heatmap.png',",
        "    'coverage_sign_flip_plot.png',",
        "    'groupkfold_stability_plot.png',",
        "    'bootstrap_ci_plot.png',",
        "    'feature_family_summary.png',",
        "    'provisional_feature_set_plot.png',",
        "]",
        "",
        "csv_status = {f: (VALIDATION_DIR / f).exists() for f in expected_csvs}",
        "fig_status = {f: (FIGURES_DIR / f).exists() for f in expected_figures}",
        "",
        "print('CSV Output Manifest:')",
        "for f, status in csv_status.items():",
        "    print(f'  - {f:<40} {\"EXISTS\" if status else \"MISSING\"}')",
        "",
        "print('\\nFigure Output Manifest:')",
        "for f, status in fig_status.items():",
        "    print(f'  - {f:<40} {\"EXISTS\" if status else \"MISSING\"}')",
        "",
        "assert all(csv_status.values()), 'Missing CSV outputs!'",
        "assert all(fig_status.values()), 'Missing Figure outputs!'",
        "print('\\nALL W4-B VALIDATION ARTIFACTS VERIFIED SUCCESSFULLY.')"
    ]))

    nb_content = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
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
    create_feature_validation_notebook()
