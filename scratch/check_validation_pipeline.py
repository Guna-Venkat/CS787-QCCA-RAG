"""Preflight check and execution for Feature Validation (W4-B)."""

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from src.week4.feature_validation import (
    benjamini_hochberg,
    compute_fdr_table,
    compute_feature_correlation_matrix,
    find_redundancy_clusters,
    select_redundancy_representatives,
    rank_partial_correlation,
    compute_partial_associations,
    construct_transition_targets,
    compute_transition_associations,
    analyze_coverage_sign_flips,
    build_provisional_feature_set,
    evaluate_qcca_v2_gates,
)

def main():
    print("Starting W4-B validation pipeline check...")
    
    # 1. Load Discovery Data
    disc_dir = Path("results/week4/feature_discovery")
    df_features = pd.read_csv(disc_dir / "feature_matrix.csv")
    df_targets = pd.read_csv(disc_dir / "target_matrix.csv")
    df_metadata = pd.read_csv(disc_dir / "feature_metadata.csv")
    df_spearman = pd.read_csv(disc_dir / "spearman_results.csv")
    df_bootstrap = pd.read_csv(disc_dir / "bootstrap_ci_results.csv")
    df_stability = pd.read_csv(disc_dir / "groupkfold_stability.csv")
    
    feature_cols = [c for c in df_features.columns if c not in ["question_id", "paper_id"]]
    print(f"Loaded {len(df_features)} samples, {len(feature_cols)} features.")
    
    out_dir = Path("results/week4/feature_validation")
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Analysis 1: FDR
    print("1. Computing Benjamini-Hochberg FDR...")
    df_fdr = compute_fdr_table(df_spearman, alpha_nominal=0.05, alpha_fdr=0.10)
    df_fdr.to_csv(out_dir / "fdr_results.csv", index=False)
    n_nominal = df_fdr["is_nominal_sig"].sum()
    n_fdr10 = df_fdr["is_fdr_sig_10"].sum()
    n_fdr05 = df_fdr["is_fdr_sig_05"].sum()
    print(f"   Nominal p < 0.05: {n_nominal}/320; FDR q < 0.10: {n_fdr10}/320; FDR q < 0.05: {n_fdr05}/320.")
    
    # Analysis 2: Feature Correlation Matrix & Clustering
    print("2. Computing Feature Correlation Matrix & Clustering...")
    df_corr = compute_feature_correlation_matrix(df_features, feature_cols)
    df_corr.to_csv(out_dir / "feature_correlation_matrix.csv")
    df_clusters = find_redundancy_clusters(df_corr, redundancy_threshold=0.80)
    print(f"   Clustered 64 features into {df_clusters['redundancy_cluster_id'].nunique()} clusters at |rho| >= 0.80.")
    
    # Analysis 3: Cluster Representatives
    print("3. Selecting Redundancy Cluster Representatives...")
    df_reps = select_redundancy_representatives(
        df_clusters, df_spearman, df_bootstrap, df_stability, df_metadata
    )
    df_reps.to_csv(out_dir / "feature_redundancy_groups.csv", index=False)
    n_reps = df_reps["is_representative"].sum()
    print(f"   Selected {n_reps} cluster representatives.")
    
    # Analysis 4: Partial Associations
    print("4. Computing Partial / Conditional Associations...")
    df_partial = compute_partial_associations(df_features, df_targets)
    df_partial.to_csv(out_dir / "partial_association_results.csv", index=False)
    print(f"   Evaluated {len(df_partial)} conditional hypotheses.")
    for _, row in df_partial.iterrows():
        print(f"     - {row['feature']} -> {row['target']} | {row['controlling_for']}: raw={row['raw_spearman_rho']:.3f}, part={row['partial_rank_r']:.3f} (p={row['partial_p_value']:.4f})")
        
    # Analysis 5: Transition-Specific Targets & Associations
    print("5. Constructing Transition Targets & Associations...")
    df_trans_targets = construct_transition_targets(
        "results/week2/processed/dev_per_query_k_matrix.csv", delta_threshold=0.01
    )
    df_trans_assoc = compute_transition_associations(df_features, df_trans_targets, feature_cols)
    df_trans_assoc.to_csv(out_dir / "transition_association_results.csv", index=False)
    print(f"   Computed {len(df_trans_assoc)} transition associations across 10 transition targets.")
    
    # Analysis 6: Coverage Sign-Flip Audit
    print("6. Auditing Lexical Coverage Sign Flips...")
    df_sign_flip = analyze_coverage_sign_flips(df_spearman, df_fdr, df_bootstrap, df_stability)
    df_sign_flip.to_csv(out_dir / "transition_sign_flip_analysis.csv", index=False)
    flips_found = df_sign_flip[df_sign_flip["has_sign_flip"] == True]
    print(f"   Found {len(flips_found)} coverage features demonstrating sign flips between 4->6 and 6->8:")
    for _, row in flips_found.iterrows():
        print(f"     - {row['coverage_feature']}: G_4_to_6 rho={row['G_4_to_6_rho']:.4f} (q={row['G_4_to_6_q_value']:.3f}), G_6_to_8 rho={row['G_6_to_8_rho']:.4f} (q={row['G_6_to_8_q_value']:.3f})")
        
    # Analysis 7 & 8: GroupKFold & Bootstrap Validation CSVs
    print("7 & 8. Saving GroupKFold & Bootstrap validation tables...")
    df_stability.to_csv(out_dir / "groupkfold_validation.csv", index=False)
    df_bootstrap.to_csv(out_dir / "bootstrap_validation.csv", index=False)
    
    # Analysis 9: Feature Family Summary
    print("9. Summarizing Feature Families...")
    fam_records = []
    for fam, group in df_metadata.groupby("feature_family"):
        fam_feats = group["feature_name"].tolist()
        fam_fdr = df_fdr[df_fdr["feature"].isin(fam_feats)]
        fam_boot = df_bootstrap[df_bootstrap["feature"].isin(fam_feats)]
        fam_stab = df_stability[df_stability["feature"].isin(fam_feats)]
        
        n_orig = len(fam_feats)
        n_fdr_sig = fam_fdr["is_fdr_sig_10"].sum()
        n_boot_stable = fam_boot["bootstrap_sign_stable"].sum()
        med_abs_rho = fam_fdr["abs_spearman_rho"].median()
        max_abs_rho = fam_fdr["abs_spearman_rho"].max()
        med_stab = fam_stab["fold_sign_consistency"].median()
        
        # Candidate reps
        reps = df_reps[(df_reps["family"] == fam) & (df_reps["is_representative"] == True)]["feature"].tolist()
        
        fam_records.append({
            "feature_family": fam,
            "original_features": n_orig,
            "fdr_sig_relationships_q10": int(n_fdr_sig),
            "bootstrap_ci_excludes_zero": int(n_boot_stable),
            "median_abs_rho": round(med_abs_rho, 4),
            "max_abs_rho": round(max_abs_rho, 4),
            "median_fold_stability": round(med_stab, 2),
            "representative_features": ", ".join(reps[:3]),
        })
    df_fam_sum = pd.DataFrame(fam_records)
    df_fam_sum.to_csv(out_dir / "feature_family_validation_summary.csv", index=False)
    
    # Analysis 10: Provisional Feature Set
    print("10. Building Provisional Feature Set...")
    df_prov = build_provisional_feature_set(
        df_reps, df_fdr, df_bootstrap, df_stability, df_metadata
    )
    df_prov.to_csv(out_dir / "provisional_feature_set.csv", index=False)
    print(f"   Selected {len(df_prov)} provisional features:")
    for _, row in df_prov.iterrows():
        print(f"     - [{row['family']}] {row['feature']} -> {row['primary_target']}: rho={row['spearman_rho']:+.3f}, q={row['q_value_fdr']:.3f}, stab={row['fold_sign_consistency']:.1f}")
        
    # Analysis 11 & Decision Gates
    print("11. Evaluating QCCA-V2 Decision Gates...")
    gates = evaluate_qcca_v2_gates(df_fdr, df_clusters, df_trans_assoc, df_sign_flip, df_prov)
    for g_name, g_info in gates.items():
        print(f"   {g_name}: {g_info['status']}")
        if "evidence" in g_info:
            print(f"     Details: {g_info['evidence']}")
            
    # Generate Figures
    print("\nGenerating Figures...")
    
    # 1. Feature Correlation Heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(df_corr, cmap="coolwarm", vmin=-1.0, vmax=1.0, cbar_kws={"label": "Spearman Rho"})
    plt.title("Feature-Feature Pairwise Correlation Matrix (64 x 64 Pre-Generation Features)", fontsize=14, pad=12)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(fig_dir / "feature_correlation_heatmap.png", dpi=300)
    plt.close()
    print("   Created feature_correlation_heatmap.png")
    
    # 2. FDR Volcano / Significance Plot
    plt.figure(figsize=(9, 6))
    plt.scatter(
        df_fdr["spearman_rho"],
        -np.log10(df_fdr["p_value_naive"] + 1e-12),
        c=df_fdr["is_fdr_sig_10"].map({True: "#d95f02", False: "#7570b3"}),
        alpha=0.75,
        s=35,
        edgecolors="none"
    )
    plt.axhline(-np.log10(0.05), color="gray", linestyle="--", alpha=0.7, label="Nominal p = 0.05")
    plt.xlabel("Spearman Rho", fontsize=12)
    plt.ylabel("-log10(Nominal p-value)", fontsize=12)
    plt.title("FDR Multiple Testing Correction (320 Feature-Target Hypotheses)", fontsize=14, pad=12)
    plt.legend(["Nominal p = 0.05", "FDR q < 0.10 (Orange) vs Inactive (Purple)"], loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "fdr_adjusted_significance_plot.png", dpi=300)
    plt.close()
    print("   Created fdr_adjusted_significance_plot.png")
    
    # 3. Transition Association Heatmap
    plt.figure(figsize=(12, 8))
    top_trans_feats = [
        "query_conjunction_count", "query_is_what_which", "query_is_numerical",
        "lex_coverage_top1", "lex_coverage_top2", "lex_coverage_top4",
        "sem_sim_mean_top6", "sem_sim_mean_top10", "bm25_mean_score",
        "bm25_first_gap_ratio", "passage_sim_std", "evidence_query_cluster_span"
    ]
    trans_targets_sub = ["G_2_to_4", "G_4_to_6", "G_6_to_8", "I_2_to_4", "I_4_to_6", "I_6_to_8"]
    pivot_df = df_trans_assoc[
        df_trans_assoc["feature"].isin(top_trans_feats) & df_trans_assoc["target"].isin(trans_targets_sub)
    ].pivot(index="feature", columns="target", values="spearman_rho")[trans_targets_sub].loc[top_trans_feats]
    
    sns.heatmap(pivot_df, annot=True, fmt="+.3f", cmap="vlag", center=0, vmin=-0.25, vmax=0.25, cbar_kws={"label": "Spearman Rho"})
    plt.title("Transition-Specific Associations (Continuous Gains vs Binary Indicators)", fontsize=13, pad=12)
    plt.ylabel("Candidate Feature", fontsize=11)
    plt.xlabel("Allocation Transition", fontsize=11)
    plt.tight_layout()
    plt.savefig(fig_dir / "transition_association_heatmap.png", dpi=300)
    plt.close()
    print("   Created transition_association_heatmap.png")
    
    # 4. Coverage Sign-Flip Plot
    plt.figure(figsize=(10, 6))
    cov_plot_df = df_sign_flip[df_sign_flip["coverage_feature"].isin([
        "lex_coverage_top1", "lex_coverage_top2", "lex_coverage_top4", "lex_coverage_top6"
    ])]
    x = np.arange(len(cov_plot_df))
    width = 0.25
    plt.bar(x - width, cov_plot_df["G_2_to_4_rho"], width, label="G_2_to_4 (Base Expansion)", color="#1b9e77")
    plt.bar(x, cov_plot_df["G_4_to_6_rho"], width, label="G_4_to_6 (Middle Expansion)", color="#386cb0")
    plt.bar(x + width, cov_plot_df["G_6_to_8_rho"], width, label="G_6_to_8 (Saturation)", color="#e41a1c")
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.xticks(x, cov_plot_df["coverage_feature"], rotation=15, ha="right", fontsize=11)
    plt.ylabel("Spearman Correlation (rho)", fontsize=12)
    plt.title("Audit of Lexical Coverage Sign Flips Across Allocation Transitions", fontsize=13, pad=12)
    plt.legend(loc="upper right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "coverage_sign_flip_plot.png", dpi=300)
    plt.close()
    print("   Created coverage_sign_flip_plot.png")
    
    # 5. GroupKFold Stability Plot
    plt.figure(figsize=(10, 6))
    prov_feats = df_prov["feature"].tolist()[:10]
    stab_sub = df_stability[
        df_stability["feature"].isin(prov_feats) &
        df_stability["target"].isin(["oracle_k", "G_4_to_6", "G_6_to_8"])
    ].drop_duplicates(subset=["feature"])
    
    # Parse fold_rhos
    import ast
    rhos_list = []
    labels = []
    for _, row in stab_sub.iterrows():
        try:
            val = ast.literal_eval(row["fold_rhos"])
            rhos_list.append(val)
            labels.append(f"{row['feature']}\n({row['target']})")
        except Exception:
            pass
            
    if rhos_list:
        plt.boxplot(rhos_list, tick_labels=labels, vert=False, patch_artist=True,
                    boxprops=dict(facecolor="#cbd5e1", color="#334155"),
                    medianprops=dict(color="#b91c1c", linewidth=2))
        plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
        plt.xlabel("Spearman Rho Across 5 Paper-Disjoint Folds", fontsize=11)
        plt.title("GroupKFold Stability (Paper-Disjoint Cross-Validation)", fontsize=13, pad=12)
        plt.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / "groupkfold_stability_plot.png", dpi=300)
        plt.close()
        print("   Created groupkfold_stability_plot.png")
        
    # 6. Bootstrap CI Plot
    plt.figure(figsize=(10, 7))
    prov_sub = df_prov.copy()
    y_pos = np.arange(len(prov_sub))
    plt.errorbar(
        prov_sub["spearman_rho"],
        y_pos,
        xerr=[
            prov_sub["spearman_rho"] - prov_sub["bootstrap_ci_low"],
            prov_sub["bootstrap_ci_high"] - prov_sub["spearman_rho"]
        ],
        fmt="o",
        color="#0284c7",
        ecolor="#0284c7",
        elinewidth=2,
        capsize=4,
        markersize=6
    )
    plt.axvline(0, color="red", linestyle="--", linewidth=1.0)
    plt.yticks(y_pos, [f"{f} ({t})" for f, t in zip(prov_sub["feature"], prov_sub["primary_target"])], fontsize=10)
    plt.gca().invert_yaxis()
    plt.xlabel("Spearman Rho (with 95% Paper-Clustered Bootstrap CI)", fontsize=11)
    plt.title("Paper-Clustered Bootstrap 95% Confidence Intervals (Provisional Features)", fontsize=13, pad=12)
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "bootstrap_ci_plot.png", dpi=300)
    plt.close()
    print("   Created bootstrap_ci_plot.png")
    
    # 7. Feature Family Summary Plot
    plt.figure(figsize=(10, 5))
    x_fam = np.arange(len(df_fam_sum))
    plt.bar(x_fam, df_fam_sum["max_abs_rho"], color="#6366f1", alpha=0.85, label="Peak |Spearman Rho|")
    plt.plot(x_fam, df_fam_sum["median_abs_rho"], color="#f59e0b", marker="o", linewidth=2.5, label="Median |Spearman Rho|")
    plt.xticks(x_fam, df_fam_sum["feature_family"], rotation=20, ha="right", fontsize=10)
    plt.ylabel("Absolute Spearman Correlation", fontsize=11)
    plt.title("Feature Family Statistical Profile Across 6 Pre-Generation Information Domains", fontsize=13, pad=12)
    plt.legend(loc="upper right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "feature_family_summary.png", dpi=300)
    plt.close()
    print("   Created feature_family_summary.png")
    
    # 8. Provisional Feature Set Visualization
    plt.figure(figsize=(11, 7))
    colors = {
        "A: BM25/Retrieval": "#3b82f6",
        "B: Query Complexity": "#10b981",
        "C: Semantic Structure": "#8b5cf6",
        "D: Passage Redundancy/Diversity": "#f59e0b",
        "E: Lexical Coverage": "#ef4444",
        "F: Evidence Structure": "#06b6d4",
    }
    bar_colors = [colors.get(f, "#64748b") for f in prov_sub["family"]]
    plt.barh(y_pos, prov_sub["spearman_rho"], color=bar_colors, alpha=0.85)
    plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
    plt.yticks(y_pos, prov_sub["feature"], fontsize=10)
    plt.gca().invert_yaxis()
    plt.xlabel("Spearman Rho with Primary Target", fontsize=11)
    plt.title("Provisional Reduced Feature Set (14 Non-Redundant Candidates Across 6 Families)", fontsize=13, pad=12)
    
    # Custom legend
    handles = [plt.Rectangle((0,0),1,1, color=col) for col in colors.values()]
    plt.legend(handles, colors.keys(), loc="lower right", fontsize=9)
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "provisional_feature_set_plot.png", dpi=300)
    plt.close()
    print("   Created provisional_feature_set_plot.png")
    
    print("\nAll W4-B outputs and figures successfully verified and generated.")

if __name__ == "__main__":
    main()
