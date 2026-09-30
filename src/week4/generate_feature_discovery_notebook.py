"""Generator for the Week 4 Feature Discovery Notebook: week4_feature_discovery.ipynb.

Creates a 20-section unexecuted notebook strictly adhering to:
- DO NOT RUN THE NOTEBOOK
- All outputs empty (execution_count=None, outputs=[])
- Clean markdown explanations and runnable code cells
- Saves all artifacts to results/week4/feature_discovery/
"""

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


def create_feature_discovery_notebook(output_path: str = "notebooks/week4/week4_feature_discovery.ipynb"):
    cells = []

    # Section 1
    cells.append(make_cell("markdown", [
        "# Week 4: Feature Discovery for Query-Conditioned Evidence Allocation (QCCA)",
        "**Project:** CS787 Generative AI — Query-Conditioned Context Allocator  ",
        "**Parent Framework:** SARA — Selective and Adaptive Retrieval-augmented Generation with Context Compression  ",
        "**Benchmark:** QASPER Development Split (86 papers, 231 queries)  ",
        "**Objective:** Answer the fundamental scientific question:  ",
        "> *What information available before generation correlates with how much textual evidence a query benefits from?*",
        "",
        "---",
        "### Investigation Framing",
        "The initial Week 4 experiment tested six simple BM25 score-concentration features (`rho_1`, `delta_12`, `entropy`, etc.) and revealed weak associations with downstream evidence budgets.",
        "Rather than forcing a complex predictive model, this study builds and evaluates a principled **64-feature library** spanning **6 distinct pre-generation information families**:",
        "- **Family A: BM25 / Retrieval-Score Structure (17 features)**",
        "- **Family B: Query Complexity (12 features)**",
        "- **Family C: Query ↔ Retrieved-Passage Semantic Structure (11 features)**",
        "- **Family D: Passage Redundancy / Diversity (10 features)**",
        "- **Family E: Query / Passage Lexical Coverage (8 features)**",
        "- **Family F: Evidence Structure / Multi-Aspect Signals (6 features)**",
        "",
        "**CRITICAL EXPERIMENTAL BOUNDARIES:**",
        "- **NO ML ALLOCATOR IS TRAINED OR FROZEN IN THIS NOTEBOOK.**",
        "- **QASPER_test remains 100% UNTOUCHED and UNSEEN.**",
        "- **All analysis is purely exploratory on the Development split.**"
    ]))

    # Section 2
    cells.append(make_cell("markdown", [
        "## 2. Frozen Experimental Setup & Configuration",
        "Define paths, seeds, allocation action space $\\mathcal{K}_{\\text{alloc}} = \\{2, 4, 5, 6, 8\\}$, tolerance $\\epsilon^* = 0.01$, and target directories."
    ]))

    cells.append(make_cell("code", [
        "import sys",
        "import os",
        "from pathlib import Path",
        "",
        "# Ensure project root is on sys.path",
        "PROJECT_ROOT = Path(\"..\").resolve().parent if Path(\"..\").resolve().name == \"notebooks\" else Path(\".\").resolve()",
        "if str(PROJECT_ROOT) not in sys.path:",
        "    sys.path.insert(0, str(PROJECT_ROOT))",
        "",
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "",
        "# Set global visual style",
        "plt.style.use(\"seaborn-v0_8-whitegrid\")",
        "plt.rcParams[\"font.family\"] = \"sans-serif\"",
        "plt.rcParams[\"font.sans-serif\"] = [\"DejaVu Sans\", \"Arial\"]",
        "plt.rcParams[\"axes.edgecolor\"] = \"#cccccc\"",
        "plt.rcParams[\"axes.linewidth\"] = 1.0",
        "",
        "# Experimental Constants",
        "RANDOM_SEED = 42",
        "N_BOOTSTRAP = 1000",
        "N_CV_SPLITS = 5",
        "EPSILON_STAR = 0.01",
        "K_ALLOC = [2, 4, 5, 6, 8]",
        "",
        "# Output Artifact Directory",
        "OUT_DIR = PROJECT_ROOT / \"results/week4/feature_discovery\"",
        "FIG_DIR = OUT_DIR / \"figures\"",
        "OUT_DIR.mkdir(parents=True, exist_ok=True)",
        "FIG_DIR.mkdir(parents=True, exist_ok=True)",
        "",
        "print(f\"Project Root: {PROJECT_ROOT}\")",
        "print(f\"Output Artifacts: {OUT_DIR}\")",
        "print(f\"Figures: {FIG_DIR}\")",
        "print(f\"Action space K_alloc: {K_ALLOC}, epsilon*: {EPSILON_STAR}\")"
    ]))

    # Section 3
    cells.append(make_cell("markdown", [
        "## 3. Data & Artifact Verification",
        "Verify presence of frozen Week-2 retrieval passages, response matrix, Week-3 epsilon targets, and cached SFR embeddings."
    ]))

    cells.append(make_cell("code", [
        "RAW_RETRIEVAL_PATH = PROJECT_ROOT / \"results/week2/raw/dev_retrieval.jsonl\"",
        "DEV_MATRIX_PATH = PROJECT_ROOT / \"results/week2/processed/dev_per_query_k_matrix.csv\"",
        "EPSILON_ORACLE_PATH = PROJECT_ROOT / \"results/week3/processed/epsilon_oracle_all.csv\"",
        "PASSAGE_EMB_PATH = PROJECT_ROOT / \"results/week2/raw/dev_sfr_embeddings.pt\"",
        "QUERY_EMB_PATH = PROJECT_ROOT / \"results/week4/feature_discovery/dev_query_sfr_embeddings.pt\"",
        "",
        "for p, name in [",
        "    (RAW_RETRIEVAL_PATH, \"Raw retrieval records\"),",
        "    (DEV_MATRIX_PATH, \"Dev per-query k-matrix\"),",
        "    (EPSILON_ORACLE_PATH, \"Week-3 epsilon oracle targets\"),",
        "    (PASSAGE_EMB_PATH, \"Cached passage SFR embeddings\"),",
        "    (QUERY_EMB_PATH, \"Cached query SFR embeddings\"),",
        "]:",
        "    assert p.exists(), f\"Missing required artifact: {name} at {p}\"",
        "    print(f\"[VERIFIED] {name} ({p.stat().st_size / 1024:.1f} KB)\")"
    ]))

    # Section 4
    cells.append(make_cell("markdown", [
        "## 4. Leakage Policy & Automated Audit",
        "Strict boundary check: verify that NO feature is derived from gold answers, gold evidence, generated answers, model logits, F1, EM, or oracle targets."
    ]))

    cells.append(make_cell("code", [
        "from src.week4.feature_discovery import audit_feature_leakage, get_feature_metadata",
        "",
        "df_metadata = get_feature_metadata()",
        "print(f\"Total features specified in metadata: {len(df_metadata)}\")",
        "print(\"Feature distribution by family:\")",
        "display(df_metadata[\"feature_family\"].value_counts().to_frame(\"Feature Count\"))"
    ]))

    # Section 5
    cells.append(make_cell("markdown", [
        "## 5. Marginal Answer-Quality Gain & Oracle Target Construction",
        "Construct primary targets from the frozen Week-2 response surface and Week-3 oracle:",
        "- `oracle_k`: $\\epsilon^* = 0.01$ target budget $k^* \\in \\{2, 4, 5, 6, 8\\}$",
        "- `G_2_to_4`: $F_1(q, 4) - F_1(q, 2)$",
        "- `G_4_to_6`: $F_1(q, 6) - F_1(q, 4)$",
        "- `G_6_to_8`: $F_1(q, 8) - F_1(q, 6)$",
        "- `G_2_to_8`: $F_1(q, 8) - F_1(q, 2)$",
        "- Auxiliary targets: `G_2_to_5`, `G_5_to_8`"
    ]))

    cells.append(make_cell("code", [
        "from src.week4.feature_discovery import construct_target_matrix",
        "",
        "df_targets = construct_target_matrix(",
        "    dev_matrix_path=str(DEV_MATRIX_PATH),",
        "    epsilon_oracle_path=str(EPSILON_ORACLE_PATH),",
        "    epsilon_star=EPSILON_STAR",
        ")",
        "df_targets[\"paper_id\"] = df_targets[\"paper_id\"].astype(str).str.strip()",
        "df_targets[\"question_id\"] = df_targets[\"question_id\"].astype(str).str.strip()",
        "",
        "print(f\"Constructed target matrix with {len(df_targets)} rows and {len(df_targets.columns)} columns.\")",
        "print(\"\\nTarget distributions and summaries:\")",
        "display(df_targets.describe().T[[\"count\", \"mean\", \"std\", \"min\", \"25%\", \"50%\", \"75%\", \"max\"]])",
        "",
        "print(\"\\nOracle k target counts:\")",
        "print(df_targets[\"oracle_k\"].value_counts().sort_index())"
    ]))

    # Section 6
    cells.append(make_cell("markdown", [
        "## 6. Pre-Generation Feature Extraction Across All 6 Families",
        "Extract 64 observable pre-generation features from raw query text, top-10 retrieval scores, passage texts, and 4096-dim SFR embeddings."
    ]))

    cells.append(make_cell("code", [
        "from src.week4.feature_discovery import load_all_dev_features",
        "",
        "df_features, df_meta = load_all_dev_features(",
        "    raw_retrieval_path=str(RAW_RETRIEVAL_PATH),",
        "    passage_embeddings_path=str(PASSAGE_EMB_PATH),",
        "    query_embeddings_path=str(QUERY_EMB_PATH),",
        ")",
        "df_features[\"paper_id\"] = df_features[\"paper_id\"].astype(str).str.strip()",
        "df_features[\"question_id\"] = df_features[\"question_id\"].astype(str).str.strip()",
        "",
        "feature_cols = [c for c in df_features.columns if c not in [\"question_id\", \"paper_id\"]]",
        "print(f\"Extracted {len(feature_cols)} features for {len(df_features)} questions.\")"
    ]))

    # Section 7
    cells.append(make_cell("markdown", [
        "## 7. Feature Matrix Validation & Leakage Verification",
        "Run automated audit on the extracted feature matrix."
    ]))

    cells.append(make_cell("code", [
        "is_clean, violations = audit_feature_leakage(df_features)",
        "if not is_clean:",
        "    raise ValueError(f\"LEAKAGE DETECTED: {violations}\")",
        "else:",
        "    print(\"LEAKAGE AUDIT PASSED: Zero post-generation or target tokens detected in feature matrix.\")",
        "",
        "# Check for NaNs or Infinities",
        "nan_counts = df_features[feature_cols].isna().sum()",
        "nans_present = nan_counts[nan_counts > 0]",
        "if len(nans_present) > 0:",
        "    print(f\"Imputing NaNs in {len(nans_present)} columns with median:\")",
        "    print(nans_present)",
        "    for c in nans_present.index:",
        "        df_features[c] = df_features[c].fillna(df_features[c].median())",
        "else:",
        "    print(\"Zero missing or NaN values in feature matrix.\")"
    ]))

    # Section 8
    cells.append(make_cell("markdown", [
        "## 8. Descriptive Statistics of the Feature Matrix",
        "Inspect mean, standard deviation, and range of candidate features."
    ]))

    cells.append(make_cell("code", [
        "df_stats = df_features[feature_cols].describe().T[[\"mean\", \"std\", \"min\", \"50%\", \"max\"]]",
        "print(f\"Computed descriptive statistics for all {len(df_stats)} features.\")",
        "display(df_stats.head(15))"
    ]))

    # Section 9
    cells.append(make_cell("markdown", [
        "## 9. Primary Association Analysis: Spearman Rank Correlation",
        "Compute non-parametric monotonic rank correlation $\\rho$ and naive $p$-values against `oracle_k`, `G_2_to_4`, `G_4_to_6`, `G_6_to_8`, and `G_2_to_8`."
    ]))

    cells.append(make_cell("code", [
        "from src.week4.feature_discovery import compute_spearman_associations",
        "",
        "TARGET_COLS = [\"oracle_k\", \"G_2_to_4\", \"G_4_to_6\", \"G_6_to_8\", \"G_2_to_8\"]",
        "df_spearman = compute_spearman_associations(df_features, df_targets, feature_cols, TARGET_COLS)",
        "",
        "print(f\"Computed {len(df_spearman)} feature-target Spearman associations.\")",
        "print(\"\\nTop 10 features by absolute Spearman rho with oracle_k:\")",
        "display(df_spearman[df_spearman[\"target\"] == \"oracle_k\"].sort_values(by=\"abs_spearman_rho\", ascending=False).head(10))",
        "",
        "print(\"\\nTop 10 features by absolute Spearman rho with G_2_to_8 (Overall Evidence Benefit):\")",
        "display(df_spearman[df_spearman[\"target\"] == \"G_2_to_8\"].sort_values(by=\"abs_spearman_rho\", ascending=False).head(10))"
    ]))

    # Section 10
    cells.append(make_cell("markdown", [
        "## 10. Secondary Linear Association: Pearson Correlation",
        "Compute Pearson correlation for continuous marginal gain targets ($G_{2 \\rightarrow 4}, G_{4 \\rightarrow 6}, G_{6 \\rightarrow 8}, G_{2 \\rightarrow 8}$)."
    ]))

    cells.append(make_cell("code", [
        "from src.week4.feature_discovery import compute_pearson_associations",
        "",
        "CONT_TARGETS = [\"G_2_to_4\", \"G_4_to_6\", \"G_6_to_8\", \"G_2_to_8\"]",
        "df_pearson = compute_pearson_associations(df_features, df_targets, feature_cols, CONT_TARGETS)",
        "",
        "print(f\"Computed {len(df_pearson)} Pearson correlations.\")",
        "print(\"\\nTop 10 features by Pearson correlation with G_2_to_8:\")",
        "display(df_pearson[df_pearson[\"target\"] == \"G_2_to_8\"].sort_values(by=\"abs_pearson_r\", ascending=False).head(10))"
    ]))

    # Section 11
    cells.append(make_cell("markdown", [
        "## 11. Mutual Information as a Non-Linear Screening Diagnostic",
        "Estimate mutual information to detect potential non-linear relationships that monotonic correlation might miss."
    ]))

    cells.append(make_cell("code", [
        "from src.week4.feature_discovery import compute_mutual_information",
        "",
        "df_mi = compute_mutual_information(df_features, df_targets, feature_cols, TARGET_COLS, random_state=RANDOM_SEED)",
        "",
        "print(f\"Computed Mutual Information scores for all feature-target pairs.\")",
        "print(\"\\nTop 10 features by Mutual Information with oracle_k:\")",
        "display(df_mi[df_mi[\"target\"] == \"oracle_k\"].sort_values(by=\"mutual_info\", ascending=False).head(10))"
    ]))

    # Section 12
    cells.append(make_cell("markdown", [
        "## 12. Non-Linear & Quantile Binning Diagnostics",
        "Evaluate whether top candidate features exhibit non-linear or threshold effects across quantile bins."
    ]))

    cells.append(make_cell("code", [
        "# Merge features and targets for quantile diagnostic",
        "from src.week4.feature_discovery import safe_merge_features_targets",
        "df_all = safe_merge_features_targets(df_features, df_targets)",
        "",
        "# Select top candidates from Spearman analysis",
        "top_features_g28 = df_spearman[df_spearman[\"target\"] == \"G_2_to_8\"].sort_values(by=\"abs_spearman_rho\", ascending=False)[\"feature\"].head(3).tolist()",
        "",
        "fig, axes = plt.subplots(1, len(top_features_g28), figsize=(15, 4))",
        "for idx, feat_name in enumerate(top_features_g28):",
        "    ax = axes[idx] if len(top_features_g28) > 1 else axes",
        "    try:",
        "        df_all[\"bin\"] = pd.qcut(df_all[feat_name], q=4, duplicates=\"drop\")",
        "        bin_means = df_all.groupby(\"bin\", observed=False)[\"G_2_to_8\"].agg([\"mean\", \"std\", \"count\"])",
        "        bin_means[\"sem\"] = bin_means[\"std\"] / np.sqrt(bin_means[\"count\"])",
        "        x_pts = range(len(bin_means))",
        "        ax.errorbar(x_pts, bin_means[\"mean\"], yerr=bin_means[\"sem\"], fmt=\"-o\", color=\"#1f77b4\", capsize=4, lw=2)",
        "        ax.set_xticks(x_pts)",
        "        ax.set_xticklabels([f\"Q{i+1}\" for i in x_pts])",
        "        ax.set_title(f\"{feat_name} vs G_2_to_8\", fontsize=11, fontweight=\"bold\")",
        "        ax.set_xlabel(\"Feature Quartile\")",
        "        ax.set_ylabel(\"Mean Marginal Gain (G_2_to_8)\")",
        "    except Exception as e:",
        "        ax.set_title(f\"{feat_name} (insufficient bins: {e})\")",
        "plt.tight_layout()",
        "plt.savefig(FIG_DIR / \"feature_quantile_bins.png\", dpi=300)",
        "plt.show()"
    ]))

    # Section 13
    cells.append(make_cell("markdown", [
        "## 13. Paper-Clustered Bootstrap Estimation (B = 1,000 Replicates)",
        "Estimate 95% empirical confidence intervals for Spearman $\\rho$ by resampling **paper IDs** with replacement to account for document-level clustering across the 86 papers."
    ]))

    cells.append(make_cell("code", [
        "from src.week4.feature_discovery import compute_paper_clustered_bootstrap",
        "",
        "print(f\"Running {N_BOOTSTRAP} paper-clustered bootstrap replicates...\")",
        "df_bootstrap = compute_paper_clustered_bootstrap(",
        "    df_features, df_targets, feature_cols, TARGET_COLS,",
        "    n_bootstrap=N_BOOTSTRAP, seed=RANDOM_SEED",
        ")",
        "",
        "print(\"Bootstrap CI computation complete.\")",
        "print(\"\\nTop features for G_2_to_8 with Paper-Clustered 95% CI:\")",
        "top_boot_g28 = df_bootstrap[df_bootstrap[\"target\"] == \"G_2_to_8\"].sort_values(by=\"spearman_rho\", key=abs, ascending=False).head(10)",
        "display(top_boot_g28[[\"feature\", \"target\", \"spearman_rho\", \"bootstrap_ci_low\", \"bootstrap_ci_high\", \"bootstrap_ci_width\", \"bootstrap_sign_stable\"]])"
    ]))

    # Section 14
    cells.append(make_cell("markdown", [
        "## 14. GroupKFold Cross-Validation Stability (5 Folds by Paper)",
        "Assess whether feature association directions and magnitudes remain stable across 5 paper-disjoint cross-validation folds."
    ]))

    cells.append(make_cell("code", [
        "from src.week4.feature_discovery import compute_groupkfold_stability",
        "",
        "df_stability = compute_groupkfold_stability(",
        "    df_features, df_targets, feature_cols, TARGET_COLS,",
        "    n_splits=N_CV_SPLITS, seed=RANDOM_SEED",
        ")",
        "",
        "print(f\"Computed GroupKFold stability metrics across {N_CV_SPLITS} folds.\")",
        "print(\"\\nMost stable features for G_2_to_8 across validation folds:\")",
        "top_stab_g28 = df_stability[df_stability[\"target\"] == \"G_2_to_8\"].sort_values(by=\"fold_sign_consistency\", ascending=False).head(10)",
        "display(top_stab_g28[[\"feature\", \"target\", \"fold_rho_median\", \"fold_rho_iqr\", \"fold_sign_consistency\", \"is_fold_stable\"]])"
    ]))

    # Section 15
    cells.append(make_cell("markdown", [
        "## 15. Feature-Family Aggregation & Comparison",
        "Compare information families to answer: *Which information family contains the strongest and most robust signals?*"
    ]))

    cells.append(make_cell("code", [
        "from src.week4.feature_discovery import summarize_feature_families",
        "",
        "df_family_summary = summarize_feature_families(df_spearman, df_stability, df_metadata)",
        "print(\"=== Information Family Discovery Summary ===\")",
        "display(df_family_summary)"
    ]))

    # Section 16
    cells.append(make_cell("markdown", [
        "## 16. Pre-Specified Candidate Feature Screening",
        "Apply transparent, pre-specified criteria to identify promising candidate features:",
        "1. Meaningful effect size: $|\\rho| \\ge 0.12$",
        "2. Cross-fold sign stability: $\\text{consistency} \\ge 0.80$ (at least 4 of 5 folds agree in direction)",
        "3. Clustered bootstrap 95% CI supports non-zero association"
    ]))

    cells.append(make_cell("code", [
        "from src.week4.feature_discovery import screen_candidate_features",
        "",
        "df_candidates = screen_candidate_features(",
        "    df_spearman, df_bootstrap, df_stability, df_mi, df_metadata,",
        "    rho_threshold=0.12, consistency_threshold=0.80",
        ")",
        "",
        "promising = df_candidates[df_candidates[\"candidate_flag\"] == True]",
        "print(f\"Screened {len(promising)} candidate promising feature-target relationships out of {len(df_candidates)} tested:\")",
        "display(promising)"
    ]))

    # Section 17
    cells.append(make_cell("markdown", [
        "## 17. Visualizations",
        "Generate publication-quality diagnostic plots:"
    ]))

    cells.append(make_cell("code", [
        "# Plot 1: Feature-Target Correlation Heatmap for Top Features",
        "top_features_overall = df_spearman.groupby(\"feature\")[\"abs_spearman_rho\"].max().sort_values(ascending=False).head(20).index.tolist()",
        "pivot_sp = df_spearman[df_spearman[\"feature\"].isin(top_features_overall)].pivot(index=\"feature\", columns=\"target\", values=\"spearman_rho\")",
        "",
        "plt.figure(figsize=(10, 8))",
        "sns.heatmap(pivot_sp, annot=True, fmt=\".2f\", cmap=\"coolwarm\", center=0, vmin=-0.30, vmax=0.30, cbar_kws={\"label\": \"Spearman rho\"})",
        "plt.title(\"Top 20 Features vs Targets (Spearman rho)\", fontsize=14, fontweight=\"bold\", pad=12)",
        "plt.ylabel(\"Pre-Generation Feature\", fontsize=11)",
        "plt.xlabel(\"Target (Evidence Benefit / Oracle Budget)\", fontsize=11)",
        "plt.tight_layout()",
        "plt.savefig(FIG_DIR / \"feature_target_correlation_heatmap.png\", dpi=300)",
        "plt.show()"
    ]))

    cells.append(make_cell("code", [
        "# Plot 2: Top Associations Barplot for G_2_to_8 and oracle_k",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 6))",
        "",
        "for idx, tgt in enumerate([\"oracle_k\", \"G_2_to_8\"]):",
        "    ax = axes[idx]",
        "    sub = df_spearman[df_spearman[\"target\"] == tgt].sort_values(by=\"abs_spearman_rho\", ascending=False).head(10)",
        "    colors = [\"#2ca02c\" if r > 0 else \"#d62728\" for r in sub[\"spearman_rho\"]]",
        "    ax.barh(sub[\"feature\"][::-1], sub[\"spearman_rho\"][::-1], color=colors[::-1], alpha=0.85)",
        "    ax.axvline(0, color=\"black\", lw=0.8, ls=\"--\")",
        "    ax.set_title(f\"Top 10 Features for {tgt}\", fontsize=12, fontweight=\"bold\")",
        "    ax.set_xlabel(\"Spearman rho\")",
        "    ax.set_xlim(-0.30, 0.30)",
        "",
        "plt.tight_layout()",
        "plt.savefig(FIG_DIR / \"top_features_barplot.png\", dpi=300)",
        "plt.show()"
    ]))

    cells.append(make_cell("code", [
        "# Plot 3: Paper-Clustered Bootstrap 95% Confidence Intervals",
        "top_boot = df_bootstrap[df_bootstrap[\"target\"] == \"G_2_to_8\"].sort_values(by=\"spearman_rho\", key=abs, ascending=False).head(12)",
        "",
        "plt.figure(figsize=(9, 6))",
        "y_pos = range(len(top_boot))",
        "plt.errorbar(",
        "    top_boot[\"spearman_rho\"][::-1],",
        "    y_pos,",
        "    xerr=[",
        "        (top_boot[\"spearman_rho\"] - top_boot[\"bootstrap_ci_low\"])[::-1],",
        "        (top_boot[\"bootstrap_ci_high\"] - top_boot[\"spearman_rho\"])[::-1],",
        "    ],",
        "    fmt=\"o\", color=\"#1f77b4\", ecolor=\"#ff7f0e\", elinewidth=2, capsize=4, markersize=7",
        ")",
        "plt.axvline(0, color=\"gray\", linestyle=\"--\", lw=1)",
        "plt.yticks(y_pos, top_boot[\"feature\"][::-1], fontsize=10)",
        "plt.xlabel(\"Spearman rho (with Paper-Clustered 95% CI)\", fontsize=11)",
        "plt.title(\"Paper-Clustered Bootstrap CIs for Evidence Gain (G_2_to_8)\", fontsize=12, fontweight=\"bold\")",
        "plt.tight_layout()",
        "plt.savefig(FIG_DIR / \"bootstrap_confidence_intervals.png\", dpi=300)",
        "plt.show()"
    ]))

    cells.append(make_cell("code", [
        "# Plot 4: Feature-Family Summary Comparison",
        "plt.figure(figsize=(8, 4.5))",
        "fam_sorted = df_family_summary.sort_values(by=\"median_abs_spearman_rho\", ascending=True)",
        "plt.barh(fam_sorted[\"feature_family\"], fam_sorted[\"median_abs_spearman_rho\"], color=\"#4575b4\", alpha=0.85)",
        "plt.xlabel(\"Median Absolute Spearman rho across Targets\", fontsize=11)",
        "plt.title(\"Information Family Predictive Potential\", fontsize=12, fontweight=\"bold\")",
        "plt.tight_layout()",
        "plt.savefig(FIG_DIR / \"feature_family_comparison.png\", dpi=300)",
        "plt.show()"
    ]))

    # Section 18
    cells.append(make_cell("markdown", [
        "## 18. Scientific Interpretation Template",
        "Scientific evaluation questions to be answered upon reviewing the generated outputs:",
        "1. **Does any feature family clearly outperform pure BM25 concentration?**",
        "2. **Which pre-generation features exhibit robust paper-clustered confidence intervals?**",
        "3. **Do semantic query-passage similarity or passage diversity features provide stronger signals than token length or score entropy?**",
        "4. **Is marginal gain ($G_{2 \\rightarrow 8}$) more predictable from pre-generation observable features than the discrete oracle label ($k^*$)?**"
    ]))

    # Section 19
    cells.append(make_cell("markdown", [
        "## 19. Export Clean Artifacts",
        "Save all processed feature tables, targets, and discovery statistics."
    ]))

    cells.append(make_cell("code", [
        "# Export Feature Matrix and Target Matrix (strictly separated)",
        "df_features.to_csv(OUT_DIR / \"feature_matrix.csv\", index=False)",
        "df_targets.to_csv(OUT_DIR / \"target_matrix.csv\", index=False)",
        "df_metadata.to_csv(OUT_DIR / \"feature_metadata.csv\", index=False)",
        "",
        "# Export Discovery Statistics",
        "df_spearman.to_csv(OUT_DIR / \"spearman_results.csv\", index=False)",
        "df_pearson.to_csv(OUT_DIR / \"pearson_results.csv\", index=False)",
        "df_mi.to_csv(OUT_DIR / \"mutual_information_results.csv\", index=False)",
        "df_stability.to_csv(OUT_DIR / \"groupkfold_stability.csv\", index=False)",
        "df_bootstrap.to_csv(OUT_DIR / \"bootstrap_ci_results.csv\", index=False)",
        "df_family_summary.to_csv(OUT_DIR / \"feature_family_summary.csv\", index=False)",
        "df_candidates.to_csv(OUT_DIR / \"candidate_features.csv\", index=False)",
        "",
        "print(f\"Successfully exported 10 discovery artifacts to {OUT_DIR}/\")"
    ]))

    # Section 20
    cells.append(make_cell("markdown", [
        "## 20. Summary Table & Validation Checklist",
        "Machine-readable candidate summary table and formal sign-off checklist."
    ]))

    cells.append(make_cell("code", [
        "print(\"=== Machine-Readable Discovery Summary Table ===\")",
        "display(df_candidates[[",
        "    \"feature\", \"family\", \"target\", \"spearman_rho\",",
        "    \"bootstrap_ci_low\", \"bootstrap_ci_high\", \"fold_sign_consistency\",",
        "    \"median_fold_rho\", \"mi_score\", \"candidate_flag\", \"reason\"",
        "]].head(20))",
        "",
        "print(\"\\n\" + \"=\"*70)",
        "print(\"FEATURE DISCOVERY COMPLETE — REVIEW RESULTS BEFORE BUILDING QCCA-V2\")",
        "print(\"=\"*70)",
        "print(\"Verification Checklist:\")",
        "print(\"  [X] Feature matrix contains 0 post-generation leakage.\")",
        "print(\"  [X] Target matrix strictly separated from feature matrix.\")",
        "print(\"  [X] Paper-clustered bootstrap (B=1000) evaluated.\")",
        "print(\"  [X] GroupKFold (5 folds by paper_id) stability evaluated.\")",
        "print(\"  [X] NO ML ALLOCATOR TRAINED OR FROZEN.\")",
        "print(\"  [X] QASPER_test remains 100% UNTOUCHED.\")"
    ]))

    notebook_dict = {
        "cells": cells,
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

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(notebook_dict, f, indent=1)

    print(f"Generated clean unexecuted notebook at: {out_p}")


if __name__ == "__main__":
    create_feature_discovery_notebook()
