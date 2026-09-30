"""Publication-Quality Plot Generator for Week 3.

Generates 9 publication-grade figures in results/week3/plots/:
- plot1_best_k_distribution.png
- plot2_f1_range_distribution.png
- plot3_oracle_gain_distribution.png
- plot4_mean_f1_comparison.png
- plot5_epsilon_vs_mean_k.png
- plot6_epsilon_vs_mean_regret.png
- plot7_feature_vs_oracle_k.png
- plot8_human_score_by_k.png
- plot9_human_vs_token_f1.png
"""

import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Aesthetic styling
plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300
})
PALETTE = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d"]


def generate_all_plots(
    processed_dir: str = "results/week3/processed",
    human_dir: str = "results/week3/human_audit",
    plots_dir: str = "results/week3/plots"
) -> None:
    """Generate all 9 Week-3 plots and save to plots_dir."""
    p_dir = Path(processed_dir)
    h_dir = Path(human_dir)
    out_dir = Path(plots_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df_oracle = pd.read_csv(p_dir / "dev_quality_oracle.csv")
    df_dist = pd.read_csv(p_dir / "quality_oracle_distribution.csv")
    df_het = pd.read_csv(p_dir / "per_query_heterogeneity.csv")
    df_eps = pd.read_csv(p_dir / "epsilon_summary.csv")
    df_feat = pd.read_csv(p_dir / "feature_signal_analysis.csv")
    
    # -------------------------------------------------------------------------
    # Plot 1: Best-k Distribution
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        [str(k) for k in df_dist["k"]],
        df_dist["pct"],
        color="#2b5c8f",
        edgecolor="black",
        linewidth=0.8,
        width=0.6,
        alpha=0.85
    )
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 1.0,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9
        )
    ax.set_xlabel("Optimal Budget $k^*_{\\text{qual}} \\in \\mathcal{K}_{\\text{alloc}}$")
    ax.set_ylabel("Percentage of Development Questions (%)")
    ax.set_title("Optimal Evidence Budget Distribution (Quality Oracle)")
    ax.set_ylim(0, max(df_dist["pct"]) + 10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(out_dir / "plot1_best_k_distribution.png", dpi=300)
    plt.close(fig)
    logger.info("Saved Plot 1: plot1_best_k_distribution.png")

    # -------------------------------------------------------------------------
    # Plot 2: Per-Query F1 Range Distribution
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(
        df_het["f1_range"],
        bins=20,
        kde=True,
        color="#7570b3",
        edgecolor="black",
        alpha=0.65,
        ax=ax
    )
    mean_range = df_het["f1_range"].mean()
    median_range = df_het["f1_range"].median()
    ax.axvline(mean_range, color="#d95f02", linestyle="--", linewidth=1.5, label=f"Mean = {mean_range:.3f}")
    ax.axvline(median_range, color="#e7298a", linestyle=":", linewidth=1.5, label=f"Median = {median_range:.3f}")
    ax.set_xlabel("F1 Sensitivity Range across $\\mathcal{K}_{\\text{alloc}}$ $(\\max F1 - \\min F1)$")
    ax.set_ylabel("Number of Questions")
    ax.set_title("Per-Query Evidence Sensitivity (F1 Range Distribution)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(out_dir / "plot2_f1_range_distribution.png", dpi=300)
    plt.close(fig)
    logger.info("Saved Plot 2: plot2_f1_range_distribution.png")

    # -------------------------------------------------------------------------
    # Plot 3: Oracle Gain Distribution over Best Static k=8
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    gains = df_het["oracle_gain"]
    sns.histplot(
        gains,
        bins=25,
        kde=True,
        color="#2ca02c",
        edgecolor="black",
        alpha=0.65,
        ax=ax
    )
    mean_gain = gains.mean()
    ax.axvline(mean_gain, color="#d95f02", linestyle="--", linewidth=1.5, label=f"Mean Gain = +{mean_gain:.4f}")
    pct_gain = (gains > 0.0).mean() * 100.0
    ax.set_xlabel("Oracle F1 Gain over Static SARA $k=8$")
    ax.set_ylabel("Number of Questions")
    ax.set_title(f"Headroom Distribution ({pct_gain:.1f}% Queries Outperform $k=8$)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(out_dir / "plot3_oracle_gain_distribution.png", dpi=300)
    plt.close(fig)
    logger.info("Saved Plot 3: plot3_oracle_gain_distribution.png")

    # -------------------------------------------------------------------------
    # Plot 4: Mean F1 Comparison: Static vs Quality Oracle vs Epsilon Oracles
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = ["Static k=8", "Quality Oracle", "Eps=0.00", "Eps=0.01*", "Eps=0.02", "Eps=0.05"]
    f1_vals = [
        0.3950,
        df_oracle["best_f1_qual"].mean(),
        df_eps.loc[df_eps["epsilon"] == 0.00, "mean_f1"].values[0],
        df_eps.loc[df_eps["epsilon"] == 0.01, "mean_f1"].values[0],
        df_eps.loc[df_eps["epsilon"] == 0.02, "mean_f1"].values[0],
        df_eps.loc[df_eps["epsilon"] == 0.05, "mean_f1"].values[0]
    ]
    colors = ["#7f7f7f", "#1f77b4", "#aec7e8", "#2ca02c", "#98df8a", "#d62728"]
    bars = ax.bar(labels, f1_vals, color=colors, edgecolor="black", linewidth=0.8, width=0.55)
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.008,
            f"{h:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9
        )
    ax.set_ylabel("Mean Token F1 on Development Split")
    ax.set_title("Performance Comparison: Static Baseline vs Oracle Targets")
    ax.set_ylim(0, 0.58)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(out_dir / "plot4_mean_f1_comparison.png", dpi=300)
    plt.close(fig)
    logger.info("Saved Plot 4: plot4_mean_f1_comparison.png")

    # -------------------------------------------------------------------------
    # Plot 5: Mean Selected k vs Epsilon
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(
        df_eps["epsilon"],
        df_eps["mean_k"],
        marker="o",
        markersize=8,
        color="#2b5c8f",
        linewidth=2,
        label="Mean Selected $k^*_\\epsilon$"
    )
    for _, row in df_eps.iterrows():
        ax.text(
            row["epsilon"],
            row["mean_k"] + 0.02,
            f"k={row['mean_k']:.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )
    ax.axhline(8.0, color="#d95f02", linestyle=":", label="Static SARA Baseline (k=8)")
    ax.set_xlabel("Quality Tolerance Epsilon ($\\epsilon$)")
    ax.set_ylabel("Mean Selected Evidence Budget ($k$)")
    ax.set_title("Evidence Budget Efficiency vs Epsilon")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(out_dir / "plot5_epsilon_vs_mean_k.png", dpi=300)
    plt.close(fig)
    logger.info("Saved Plot 5: plot5_epsilon_vs_mean_k.png")

    # -------------------------------------------------------------------------
    # Plot 6: Mean Regret vs Epsilon
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(
        df_eps["epsilon"],
        df_eps["mean_regret"],
        marker="s",
        markersize=8,
        color="#d95f02",
        linewidth=2,
        label="Mean F1 Regret vs Oracle"
    )
    for _, row in df_eps.iterrows():
        ax.text(
            row["epsilon"],
            row["mean_regret"] + 0.00005,
            f"{row['mean_regret']:.4f}",
            ha="center",
            va="bottom",
            fontsize=9
        )
    ax.set_xlabel("Quality Tolerance Epsilon ($\\epsilon$)")
    ax.set_ylabel("Empirical Mean F1 Regret $(F1^*_{\\text{alloc}} - F1_{\\epsilon})$")
    ax.set_title("Empirical Quality Regret vs Epsilon Threshold")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(out_dir / "plot6_epsilon_vs_mean_regret.png", dpi=300)
    plt.close(fig)
    logger.info("Saved Plot 6: plot6_epsilon_vs_mean_regret.png")

    # -------------------------------------------------------------------------
    # Plot 7: Oracle k vs Retrieval Feature Relationship (Boxplot)
    # -------------------------------------------------------------------------
    # Load aligned dataframe
    features_jsonl = p_dir / "dev_retrieval_features.jsonl"
    feat_records = [json.loads(line) for line in open("results/week2/processed/dev_retrieval_features.jsonl")]
    df_feat_raw = pd.DataFrame(feat_records)
    df_feat_raw["question_id"] = df_feat_raw["question_id"].astype(str)
    df_oracle_sub = df_oracle[["question_id", "best_k_qual"]].copy()
    df_oracle_sub["question_id"] = df_oracle_sub["question_id"].astype(str)
    df_merged = df_feat_raw.merge(df_oracle_sub, on="question_id")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.boxplot(x="best_k_qual", y="delta_12", data=df_merged, ax=ax1, palette="Blues")
    ax1.set_xlabel("Optimal Budget $k^*_{\\text{qual}}$")
    ax1.set_ylabel("Retrieval Score Gap $\\Delta_{12}$")
    ax1.set_title("Score Gap $\\Delta_{12}$ across Optimal Budgets")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    
    sns.boxplot(x="best_k_qual", y="n_high", data=df_merged, ax=ax2, palette="Greens")
    ax2.set_xlabel("Optimal Budget $k^*_{\\text{qual}}$")
    ax2.set_ylabel("Number of High-Scoring Passages $N_{\\text{high}}$")
    ax2.set_title("High-Scoring Count $N_{\\text{high}}$ across Optimal Budgets")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    fig.savefig(out_dir / "plot7_feature_vs_oracle_k.png", dpi=300)
    plt.close(fig)
    logger.info("Saved Plot 7: plot7_feature_vs_oracle_k.png")

    # -------------------------------------------------------------------------
    # Plot 8 & 9: Human Audit Visualizations (Pending-Aware)
    # -------------------------------------------------------------------------
    scores_path = h_dir / "human_scores.csv"
    has_genuine_scores = False
    if scores_path.exists():
        df_scores = pd.read_csv(scores_path)
        if "score_0_1_2" in df_scores.columns and not df_scores["score_0_1_2"].isna().all():
            try:
                valid_cnt = len(df_scores["score_0_1_2"].dropna().astype(int))
                if valid_cnt >= 150:
                    has_genuine_scores = True
            except Exception:
                has_genuine_scores = False

    if has_genuine_scores:
        # Plot 8: Human score by condition
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        bars = ax.bar(
            [f"k={k}" for k in df_h_sum["k"]],
            df_h_sum["mean_human_score"],
            color=["#1f77b4", "#ff7f0e", "#2ca02c"],
            edgecolor="black",
            linewidth=0.8,
            width=0.5
        )
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.03, f"{h:.2f} / 2.0", ha="center", va="bottom", fontweight="bold", fontsize=9)
        ax.set_ylabel("Mean Human Correctness Score (0–2 Scale)")
        ax.set_title("Blinded Human Audit Score by Evidence Budget ($N=50$)")
        ax.set_ylim(0, 2.0)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig.savefig(out_dir / "plot8_human_score_by_k.png", dpi=300)
        plt.close(fig)
        logger.info("Saved Plot 8: plot8_human_score_by_k.png")

        # Plot 9: Human score vs Token F1
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.boxplot(x="score_0_1_2", y="token_f1", data=df_scores, ax=ax, palette="Purples", width=0.45)
        sns.stripplot(x="score_0_1_2", y="token_f1", data=df_scores, color="black", alpha=0.35, jitter=0.2, ax=ax)
        sp_r = df_scores["score_0_1_2"].corr(df_scores["token_f1"], method="spearman")
        ax.set_xlabel("Human Correctness Judgment (0=Incorrect, 1=Partial, 2=Correct)")
        ax.set_ylabel("Automatic Token F1")
        ax.set_title(f"Human Audit Scores vs Token F1 (Spearman $\\rho = {sp_r:.3f}$)")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig.savefig(out_dir / "plot9_human_vs_token_f1.png", dpi=300)
        plt.close(fig)
        logger.info("Saved Plot 9: plot9_human_vs_token_f1.png")
    else:
        # Plot 8: Stratified Audit Sample Design (Pending Human Scoring)
        df_sample = pd.read_csv(h_dir / "audit_sample.csv")
        fig, ax = plt.subplots(figsize=(7, 4.2))
        strata_counts = df_sample["best_k_qual"].value_counts().sort_index()
        bars = ax.bar(
            [f"k*={k}" for k in strata_counts.index],
            strata_counts.values,
            color="#2b5c8f",
            edgecolor="black",
            linewidth=0.8,
            width=0.55
        )
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.3, f"n={int(h)}", ha="center", va="bottom", fontweight="bold", fontsize=9)
        ax.set_xlabel("Stratum: Optimal Deployable Budget ($k^*_{\\text{qual}}$)")
        ax.set_ylabel("Number of Sampled Questions")
        ax.set_title("Blinded Human Audit Sample Design ($N=50$ Questions, $150$ Judgments)\n[Status: PENDING_MANUAL_EVALUATION]")
        ax.set_ylim(0, max(strata_counts.values) + 3)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig.savefig(out_dir / "plot8_human_audit_design.png", dpi=300)
        # Also save as plot8_human_score_by_k.png for backward compatibility with notebooks
        fig.savefig(out_dir / "plot8_human_score_by_k.png", dpi=300)
        plt.close(fig)
        logger.info("Saved Plot 8: plot8_human_audit_design.png (Audit Design / Pending)")

        # Plot 9: Token F1 Distribution of Audit Sample across k in {2, 5, 10}
        df_raw_records = [json.loads(line) for line in open("results/week2/raw/dev_fixed_k_matrix.jsonl")]
        df_raw_all = pd.DataFrame(df_raw_records)
        df_raw_all["question_id"] = df_raw_all["question_id"].astype(str)
        sample_q_ids = set(df_sample["question_id"].astype(str))
        df_audit_perf = df_raw_all[(df_raw_all["question_id"].isin(sample_q_ids)) & (df_raw_all["k"].isin([2, 5, 10]))]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.boxplot(x="k", y="token_f1", data=df_audit_perf, ax=ax, palette="Blues", width=0.45)
        sns.stripplot(x="k", y="token_f1", data=df_audit_perf, color="black", alpha=0.35, jitter=0.2, ax=ax)
        ax.set_xlabel("Audited Condition ($k$)")
        ax.set_ylabel("Automatic Token F1")
        ax.set_title("Audit Sample Performance Across Conditions ($k \\in \\{2, 5, 10\\}$)\n[Benchmark Baseline Awaiting Human Judgment]")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig.savefig(out_dir / "plot9_token_f1_by_audit_condition.png", dpi=300)
        # Also save as plot9_human_vs_token_f1.png for backward compatibility
        fig.savefig(out_dir / "plot9_human_vs_token_f1.png", dpi=300)
        plt.close(fig)
        logger.info("Saved Plot 9: plot9_token_f1_by_audit_condition.png (Benchmark Baseline)")

    logger.info("All publication plots successfully generated in %s", plots_dir)


if __name__ == "__main__":
    generate_all_plots()
