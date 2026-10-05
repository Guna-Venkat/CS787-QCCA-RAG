"""Publication-Grade Plotting Utilities for Week 0.

Generates high-resolution figures for:
  - Part A: Dataset distributions, evidence properties, and retrieval diagnostics
  - Part B: Hidden-state similarity, representation drift, and evidence separation
  - Part C: Attention mass, ablation importance, utilization concentration, and signal correlations
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Apply publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["figure.titlesize"] = 13


def plot_dataset_distributions(
    df_questions: pd.DataFrame,
    df_papers: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Generate 4-panel figure showing document length, question length, answer length, and categories."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)

    # 1. Document word length distribution
    ax1 = axes[0, 0]
    sns.histplot(df_papers["total_doc_words"], kde=True, ax=ax1, color="#2b5c8f", bins=20)
    ax1.set_title("Paper Length Distribution (Words)")
    ax1.set_xlabel("Word Count per Document")
    ax1.set_ylabel("Document Count")
    median_doc = df_papers["total_doc_words"].median()
    ax1.axvline(median_doc, color="#d95f02", linestyle="--", label=f"Median: {int(median_doc)}")
    ax1.legend()

    # 2. Question word length distribution
    ax2 = axes[0, 1]
    sns.histplot(df_questions["query_word_count"], kde=True, ax=ax2, color="#2ca02c", discrete=True)
    ax2.set_title("Question Length Distribution (Words)")
    ax2.set_xlabel("Word Count per Query")
    ax2.set_ylabel("Query Count")
    mean_q = df_questions["query_word_count"].mean()
    ax2.axvline(mean_q, color="#d95f02", linestyle="--", label=f"Mean: {mean_q:.1f}")
    ax2.legend()

    # 3. Answer word length distribution
    ax3 = axes[1, 0]
    sns.histplot(df_questions["answer_word_count"], kde=True, ax=ax3, color="#7570b3", bins=25)
    ax3.set_title("Reference Answer Length (Words)")
    ax3.set_xlabel("Word Count per Answer")
    ax3.set_ylabel("Query Count")
    median_ans = df_questions["answer_word_count"].median()
    ax3.axvline(median_ans, color="#d95f02", linestyle="--", label=f"Median: {int(median_ans)}")
    ax3.legend()

    # 4. Question Category Breakdown
    ax4 = axes[1, 1]
    cat_counts = df_questions["question_category"].value_counts()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    ax4.bar(cat_counts.index, cat_counts.values, color=colors[:len(cat_counts)], alpha=0.85, edgecolor="black")
    ax4.set_title("Question Category Distribution")
    ax4.set_xlabel("Category")
    ax4.set_ylabel("Count")
    for idx, v in enumerate(cat_counts.values):
        ax4.text(idx, v + 1, str(v), ha="center", fontweight="bold")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved dataset distributions figure to {output_path}")


def plot_evidence_characteristics(
    df_evidence: pd.DataFrame,
    df_questions: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Generate 4-panel figure for evidence properties, depth, coverage, and WH-types."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)

    # 1. Evidence paragraphs per question
    ax1 = axes[0, 0]
    sns.countplot(data=df_evidence, x="num_evidence_paragraphs", hue="num_evidence_paragraphs", ax=ax1, palette="Blues_d", legend=False)
    ax1.set_title("Annotated Evidence Paragraphs per Query")
    ax1.set_xlabel("Number of Gold Evidence Paragraphs")
    ax1.set_ylabel("Query Count")

    # 2. Evidence depth within document
    ax2 = axes[0, 1]
    valid_depths = df_evidence["mean_evidence_depth"].dropna()
    sns.histplot(valid_depths, kde=True, ax=ax2, color="#e7298a", bins=20)
    ax2.set_title("Evidence Location within Paper (Relative Depth)")
    ax2.set_xlabel("Relative Depth (0.0 = Introduction, 1.0 = Conclusion)")
    ax2.set_ylabel("Count")
    ax2.set_xlim(0, 1)

    # 3. Query-Evidence lexical coverage
    ax3 = axes[1, 0]
    sns.histplot(df_evidence["query_evidence_coverage"], kde=True, ax=ax3, color="#1b9e77", bins=20)
    ax3.set_title("Query Term Coverage in Gold Evidence")
    ax3.set_xlabel("Fraction of Query Content Words in Evidence")
    ax3.set_ylabel("Count")
    ax3.set_xlim(0, 1)

    # 4. WH-question classification
    ax4 = axes[1, 1]
    wh_counts = df_questions["wh_type"].value_counts()
    ax4.bar(wh_counts.index, wh_counts.values, color="#e6ab02", alpha=0.85, edgecolor="black")
    ax4.set_title("Query Interrogative Structure (WH-Type)")
    ax4.set_xlabel("Interrogative Prefix")
    ax4.set_ylabel("Count")
    ax4.tick_params(axis="x", rotation=30)
    for idx, v in enumerate(wh_counts.values):
        ax4.text(idx, v + 1, str(v), ha="center", fontsize=8)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved evidence characteristics figure to {output_path}")


def plot_retrieval_diagnostics(
    df_retrieval: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Generate 4-panel figure for BM25 retrieval diagnostics (recall@k, score decay, delta12, entropy)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)

    # 1. Recall@k curves (Any vs All)
    ax1 = axes[0, 0]
    k_vals = [1, 2, 4, 6, 8, 10]
    col_any = "recall_any_at_{k}" if f"recall_any_at_1" in df_retrieval.columns else "recall_at_{k}"
    recall_any = [df_retrieval[col_any.format(k=k)].mean() * 100 for k in k_vals]
    ax1.plot(k_vals, recall_any, marker="o", linewidth=2.5, color="#d95f02", label="Recall_any@k (>=1 passage)", markersize=7)

    if f"recall_all_at_1" in df_retrieval.columns:
        recall_all = [df_retrieval[f"recall_all_at_{k}"].mean() * 100 for k in k_vals]
        ax1.plot(k_vals, recall_all, marker="s", linewidth=2.5, color="#2b83ba", label="Recall_all@k (All passages)", markersize=7)

    ax1.set_title("BM25 Gold Evidence Recall@k (Any vs All)")
    ax1.set_xlabel("Retrieval Cutoff Budget (k)")
    ax1.set_ylabel("Gold Evidence Recall (%)")
    ax1.set_xticks(k_vals)
    ax1.set_ylim(0, 105)
    ax1.legend(loc="lower right", framealpha=0.9)

    # 2. First gold passage rank distribution
    ax2 = axes[0, 1]
    hits = df_retrieval[df_retrieval["first_gold_rank"] > 0]["first_gold_rank"]
    sns.countplot(x=hits, hue=hits, ax=ax2, palette="viridis", legend=False)
    ax2.set_title("Rank of First Retrieved Gold Passage")
    ax2.set_xlabel("BM25 Rank (1..10)")
    ax2.set_ylabel("Query Count")

    # 3. Top-1 vs Top-2 margin (Delta_12)
    ax3 = axes[1, 0]
    sns.histplot(df_retrieval["delta_12"], kde=True, ax=ax3, color="#386cb0", bins=20)
    ax3.set_title("Top-1 / Top-2 Margin Distribution (Δ₁₂)")
    ax3.set_xlabel("Normalized Margin: (s₁ - s₂) / s₁")
    ax3.set_ylabel("Count")
    ax3.set_xlim(0, 1)

    # 4. Normalized score entropy (H_norm)
    ax4 = axes[1, 1]
    sns.histplot(df_retrieval["entropy_norm"], kde=True, ax=ax4, color="#f0027f", bins=20)
    ax4.set_title("Normalized Score Entropy (H_norm)")
    ax4.set_xlabel("Entropy / log(n)")
    ax4.set_ylabel("Count")
    ax4.set_xlim(0, 1)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved retrieval diagnostics figure to {output_path}")


def plot_layer_similarity_evolution(
    df_long: pd.DataFrame,
    output_similarity_path: str | Path,
    output_evidence_path: str | Path,
    output_drift_path: str | Path,
) -> None:
    """Generate Part B figures: query-passage similarity, evidence vs non-evidence gap, and representation drift."""
    # 1. Query-Passage Similarity by Layer
    plt.figure(figsize=(8, 5), dpi=300)
    sns.lineplot(
        data=df_long,
        x="layer",
        y="cosine_sim_with_query",
        hue="passage_rank",
        palette="viridis",
        marker="o",
    )
    plt.title("Evolution of Query ↔ Passage Cosine Similarity across Layers")
    plt.xlabel("Mistral Transformer Layer (0=Embedding, 31=Final)")
    plt.ylabel("Cosine Similarity: <q^(l), D_i^(l)>")
    plt.legend(title="Passage Rank", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    Path(output_similarity_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_similarity_path, bbox_inches="tight")
    plt.close()

    # 2. Evidence vs Non-Evidence Similarity by Layer
    plt.figure(figsize=(8, 5), dpi=300)
    sns.lineplot(
        data=df_long,
        x="layer",
        y="cosine_sim_with_query",
        hue="is_gold_evidence",
        palette={True: "#2ca02c", False: "#d62728"},
        style="is_gold_evidence",
        markers=True,
        dashes=False,
    )
    plt.title("Query Similarity: Gold Evidence vs. Non-Evidence Passages")
    plt.xlabel("Mistral Transformer Layer")
    plt.ylabel("Mean Cosine Similarity with Query")
    plt.legend(["Non-Evidence", "Gold Evidence"], loc="upper left")
    plt.tight_layout()
    Path(output_evidence_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_evidence_path, bbox_inches="tight")
    plt.close()

    # 3. Representation Drift (Query & Passage)
    plt.figure(figsize=(8, 5), dpi=300)
    sns.lineplot(data=df_long, x="layer", y="query_drift", label="Query Representation Drift <q^(l), q^(0)>", color="#1f77b4", marker="s")
    sns.lineplot(data=df_long, x="layer", y="passage_drift", label="Passage Representation Drift <D^(l), D^(0)>", color="#ff7f0e", marker="o")
    plt.title("Representation Drift relative to Embedding Layer (Layer 0)")
    plt.xlabel("Transformer Layer")
    plt.ylabel("Cosine Similarity with Layer 0")
    plt.ylim(0, 1.05)
    plt.legend(loc="lower left")
    plt.tight_layout()
    Path(output_drift_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_drift_path, bbox_inches="tight")
    plt.close()


def plot_evidence_utilization_figures(
    df_attn: pd.DataFrame,
    df_abl: pd.DataFrame,
    df_summary: pd.DataFrame,
    figures_dir: str | Path,
) -> None:
    """Generate Part C figures: attention heatmaps, ablation vs attention, concentration, and N_eff distribution."""
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Passage utilization distribution by rank
    plt.figure(figsize=(8, 5), dpi=300)
    sns.boxplot(data=df_abl, x="passage_rank", y="ablation_importance", hue="passage_rank", palette="Blues_r", legend=False)
    plt.title("Intervention Importance by Retrieved Passage Rank")
    plt.xlabel("Retrieval Rank (BM25)")
    plt.ylabel("Leave-One-Out Likelihood Drop Δlog P")
    plt.tight_layout()
    plt.savefig(fig_dir / "passage_utilization_distribution.png", bbox_inches="tight")
    plt.close()

    # 2. Attention vs Ablation scatter plot
    final_layer = df_attn["layer"].max() if not df_attn.empty else 31
    df_final_attn = df_attn[df_attn["layer"] == final_layer]
    merged = df_abl.merge(df_final_attn, on=["question_id", "passage_rank"], how="inner")

    if not merged.empty:
        plt.figure(figsize=(8, 5), dpi=300)
        sns.scatterplot(
            data=merged,
            x="attention_share",
            y="ablation_importance",
            hue="is_gold_evidence",
            palette={True: "#2ca02c", False: "#d62728"},
            alpha=0.75,
            s=40,
        )
        plt.title(f"Attention Share (Layer {final_layer}) vs. Intervention Importance")
        plt.xlabel("Normalized Attention Share")
        plt.ylabel("Intervention Importance (Δlog P)")
        plt.legend(title="Gold Evidence", loc="upper left")
        plt.tight_layout()
        plt.savefig(fig_dir / "attention_vs_ablation.png", bbox_inches="tight")
        plt.close()

    # 3. Retrieval score vs utilization
    if "bm25_score" in df_abl.columns:
        plt.figure(figsize=(8, 5), dpi=300)
        sns.scatterplot(
            data=df_abl,
            x="bm25_score",
            y="ablation_importance",
            hue="is_gold_evidence",
            palette={True: "#2ca02c", False: "#7570b3"},
            alpha=0.75,
            s=40,
        )
        plt.title("BM25 Retrieval Score vs. Intervention Importance")
        plt.xlabel("BM25 Score")
        plt.ylabel("Intervention Importance (Δlog P)")
        plt.legend(title="Gold Evidence", loc="upper right")
        plt.tight_layout()
        plt.savefig(fig_dir / "retrieval_vs_utilization.png", bbox_inches="tight")
        plt.close()

    # 4. Effective passage count (N_eff) distribution
    if "abl_effective_n" in df_summary.columns:
        plt.figure(figsize=(8, 5), dpi=300)
        sns.histplot(df_summary["abl_effective_n"], kde=True, color="#4daf4a", bins=15)
        plt.title("Distribution of Effective Utilized Passages (N_eff) across Queries")
        plt.xlabel("Effective Passage Count N_eff = exp(H)")
        plt.ylabel("Query Count")
        plt.axvline(df_summary["abl_effective_n"].mean(), color="#e41a1c", linestyle="--", label=f"Mean N_eff: {df_summary['abl_effective_n'].mean():.2f}")
        plt.xlim(1, 10)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "effective_passage_count_distribution.png", bbox_inches="tight")
        plt.close()

    # 5. Layerwise attention concentration
    if not df_attn.empty:
        layer_conc = []
        for l, l_df in df_attn.groupby("layer"):
            # Compute mean top-1 share and entropy per layer
            top1_shares = []
            for _, q_df in l_df.groupby("question_id"):
                top1_shares.append(q_df["attention_share"].max())
            layer_conc.append({
                "layer": l,
                "mean_top1_share": np.mean(top1_shares),
            })
        df_lc = pd.DataFrame(layer_conc)

        plt.figure(figsize=(8, 5), dpi=300)
        sns.lineplot(data=df_lc, x="layer", y="mean_top1_share", marker="o", color="#984ea3", linewidth=2.5)
        plt.title("Passage Attention Concentration across Transformer Layers")
        plt.xlabel("Mistral Transformer Layer")
        plt.ylabel("Mean Top-1 Passage Attention Share")
        plt.tight_layout()
        plt.savefig(fig_dir / "layerwise_attention_concentration.png", bbox_inches="tight")
        plt.close()
