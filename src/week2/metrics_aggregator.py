"""Stage 3: Metrics Aggregation, Baseline Determination, & Oracle Heterogeneity Analysis.

Processes raw output matrix (dev_fixed_k_matrix.jsonl) and generates:
1. dev_fixed_k_summary.csv: Summary metrics table across all k values.
2. dev_per_query_k_matrix.csv: Per-question F1 across all k values.
3. best_static_baseline.json: Deterministically selected SARA k_dev* baseline vs. Vanilla RAG ceiling.
4. per_query_best_k.csv: Detailed per-question best-k and oracle gain metadata.
5. oracle_vs_static_summary.json: Quantified Oracle Adaptive SARA gain over static k=8.
6. best_k_distribution.csv: Distribution of optimal k* across development questions.
7. Publication plots in results/week2/plots/ including heatmap and per-query response curves.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_raw_matrix(matrix_file: Path) -> pd.DataFrame:
    if not matrix_file.exists():
        raise FileNotFoundError(f"Raw matrix file not found: {matrix_file}")
    
    rows = []
    with open(matrix_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def compute_summary_table(df: pd.DataFrame, output_csv: Optional[Path] = None) -> pd.DataFrame:
    """Aggregate metrics across each k value."""
    k_order = sorted(df["k"].unique())
    
    # Baseline reference for k=10 tokens
    k10_rows = df[df["k"] == 10]
    mean_k10_tokens = k10_rows["input_tokens"].mean() if not k10_rows.empty else 1.0

    summary_rows = []
    for k in k_order:
        k_df = df[df["k"] == k]
        mean_ctx = k_df["input_tokens"].mean()
        ctx_reduction = ((mean_k10_tokens - mean_ctx) / mean_k10_tokens) * 100.0 if mean_k10_tokens > 0 else 0.0
        
        row = {
            "k": int(k),
            "method": "rag" if k == 10 else "sara",
            "n_questions": int(len(k_df)),
            "mean_f1": round(float(k_df["token_f1"].mean()), 4),
            "std_f1": round(float(k_df["token_f1"].std()), 4),
            "mean_em": round(float(k_df["exact_match"].mean()), 4),
            "mean_rouge_l": round(float(k_df["rouge_l"].mean()), 4),
            "mean_context_tokens": round(float(mean_ctx), 1),
            "mean_generated_tokens": round(float(k_df["generated_tokens"].mean()), 1),
            "context_reduction_pct": round(float(ctx_reduction), 2),
            "mean_gen_latency_sec": round(float(k_df["generation_latency"].mean()), 3),
            "mean_synthesized_total_latency_sec": round(float(k_df["synthesized_total_latency"].mean()), 3),
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_csv, index=False)
        print(f"Saved summary table to: {output_csv}")
    return summary_df


def compute_per_query_matrix(df: pd.DataFrame, output_csv: Optional[Path] = None) -> pd.DataFrame:
    """Generate per-question F1 across all k values."""
    k_values = sorted(df["k"].unique())
    pivoted = df.pivot(index=["question_id", "paper_id"], columns="k", values="token_f1").reset_index()
    
    # Rename columns to F1_k{k}
    rename_dict = {k: f"F1_k{k}" for k in k_values}
    pivoted.rename(columns=rename_dict, inplace=True)
    f1_cols = [f"F1_k{k}" for k in k_values]

    best_k_list = []
    best_f1_list = []

    for _, row in pivoted.iterrows():
        vals = [row[col] for col in f1_cols]
        max_f1 = max(vals)
        tied_k = [k for k in k_values if abs(row[f"F1_k{k}"] - max_f1) < 1e-6]
        best_k = min(tied_k)
        best_k_list.append(best_k)
        best_f1_list.append(round(max_f1, 4))

    pivoted["best_k"] = best_k_list
    pivoted["best_f1"] = best_f1_list

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        pivoted.to_csv(output_csv, index=False)
        print(f"Saved per-query k-matrix to: {output_csv}")
    return pivoted


def determine_best_static_baseline(
    summary_df: pd.DataFrame,
    output_json: Optional[Path] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Identify best SARA baseline strictly among {0,2,4,5,6,8} vs. Vanilla RAG k=10 reference ceiling."""
    sara_df = summary_df[summary_df["method"] == "sara"]
    max_sara_f1 = sara_df["mean_f1"].max()
    sara_candidates = sara_df[sara_df["mean_f1"] == max_sara_f1]
    best_sara_row = sara_candidates.sort_values(by="k").iloc[0]

    rag_df = summary_df[summary_df["k"] == 10]
    rag_row = rag_df.iloc[0] if not rag_df.empty else best_sara_row

    overall_max_f1 = summary_df["mean_f1"].max()
    overall_candidates = summary_df[summary_df["mean_f1"] == overall_max_f1]
    best_overall_row = overall_candidates.sort_values(by="k").iloc[0]

    baseline_info = {
        "best_static_sara_k": int(best_sara_row["k"]),
        "sara_k8_mean_f1": float(best_sara_row["mean_f1"]),
        "sara_k8_mean_em": float(best_sara_row["mean_em"]),
        "sara_k8_mean_rouge_l": float(best_sara_row["mean_rouge_l"]),
        "sara_k8_mean_context_tokens": float(best_sara_row["mean_context_tokens"]),
        "sara_k8_context_reduction_pct": float(best_sara_row["context_reduction_pct"]),
        "best_overall_fixed_k": int(best_overall_row["k"]),
        "vanilla_rag_k10_f1": float(rag_row["mean_f1"]),
        "vanilla_rag_k10_em": float(rag_row["mean_em"]),
        "vanilla_rag_k10_rouge_l": float(rag_row["mean_rouge_l"]),
        "selection_rule": "argmax_{k in {0,2,4,5,6,8}} Mean_F1_dev(k) with smaller-k tie-breaking for SARA baseline",
        "evaluated_question_count": int(best_sara_row["n_questions"]),
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "Static SARA baseline (k=8) selected strictly from SARA candidate set {0,2,4,5,6,8}. Vanilla RAG k=10 tracked separately as reference ceiling.",
    }

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(baseline_info, f, indent=2)
        print(f"Saved best static baseline metadata to: {output_json}")

    return baseline_info


def analyze_oracle_and_per_query_heterogeneity(
    df: pd.DataFrame,
    processed_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    """Compute per-query best k, oracle adaptive F1, and distribution metrics."""
    processed_dir.mkdir(parents=True, exist_ok=True)

    pivoted = df.pivot(index=["question_id", "paper_id"], columns="k", values="token_f1").reset_index()
    sara_cols = [0, 2, 4, 5, 6, 8]
    overall_cols = [0, 2, 4, 5, 6, 8, 10]

    per_query_records = []
    for _, row in pivoted.iterrows():
        qid = str(row["question_id"])
        pid = str(row["paper_id"])

        sara_vals = row[sara_cols]
        max_sara_f1 = sara_vals.max()
        best_sara_k = int(sara_vals[sara_vals == max_sara_f1].index.min())

        overall_vals = row[overall_cols]
        max_overall_f1 = overall_vals.max()
        best_overall_k = int(overall_vals[overall_vals == max_overall_f1].index.min())

        k8_f1 = float(row[8])
        k10_f1 = float(row[10])
        k5_f1 = float(row[5])
        k0_f1 = float(row[0])

        per_query_records.append({
            "question_id": qid,
            "paper_id": pid,
            "best_sara_k": best_sara_k,
            "best_sara_f1": round(float(max_sara_f1), 4),
            "best_overall_k": best_overall_k,
            "best_overall_f1": round(float(max_overall_f1), 4),
            "sara_k8_f1": round(k8_f1, 4),
            "vanilla_rag_k10_f1": round(k10_f1, 4),
            "sara_k5_f1": round(k5_f1, 4),
            "sara_k0_f1": round(k0_f1, 4),
            "oracle_sara_gain_over_k8": round(float(max_sara_f1 - k8_f1), 4),
            "gain_k8_to_k5": round(float(k5_f1 - k8_f1), 4),
            "gain_k8_to_k4": round(float(row[4] - k8_f1), 4),
            "gain_k8_to_k2": round(float(row[2] - k8_f1), 4),
            "gain_k8_to_k0": round(float(k0_f1 - k8_f1), 4),
        })

    df_per_query = pd.DataFrame(per_query_records)
    per_query_csv = processed_dir / "per_query_best_k.csv"
    df_per_query.to_csv(per_query_csv, index=False)
    print(f"Saved detailed per-query best k to: {per_query_csv}")

    # Best-k distribution table
    sara_counts = df_per_query["best_sara_k"].value_counts().reindex(sara_cols, fill_value=0)
    overall_counts = df_per_query["best_overall_k"].value_counts().reindex(overall_cols, fill_value=0)
    n_total = len(df_per_query)

    dist_rows = []
    for k in overall_cols:
        s_cnt = int(sara_counts.get(k, 0))
        o_cnt = int(overall_counts.get(k, 0))
        dist_rows.append({
            "k": k,
            "best_sara_k_count": s_cnt,
            "best_sara_k_pct": round((s_cnt / n_total) * 100.0, 2),
            "best_overall_k_count": o_cnt,
            "best_overall_k_pct": round((o_cnt / n_total) * 100.0, 2),
        })
    df_dist = pd.DataFrame(dist_rows)
    dist_csv = processed_dir / "best_k_distribution.csv"
    df_dist.to_csv(dist_csv, index=False)
    print(f"Saved best-k distribution to: {dist_csv}")

    # Tolerance analysis: % of questions where F1(q, k) >= Oracle_SARA_F1(q) - tol
    tol_01 = {}
    tol_02 = {}
    for k in sara_cols:
        col_f1 = pivoted[k]
        oracle_f1 = df_per_query["best_sara_f1"]
        tol_01[f"k={k}"] = round(float((col_f1 >= (oracle_f1 - 0.01)).mean() * 100.0), 2)
        tol_02[f"k={k}"] = round(float((col_f1 >= (oracle_f1 - 0.02)).mean() * 100.0), 2)

    mean_k8 = float(df_per_query["sara_k8_f1"].mean())
    mean_oracle_sara = float(df_per_query["best_sara_f1"].mean())
    mean_k10 = float(df_per_query["vanilla_rag_k10_f1"].mean())
    mean_oracle_overall = float(df_per_query["best_overall_f1"].mean())

    oracle_summary = {
        "n_questions": n_total,
        "mean_pure_compression_k0_f1": round(float(df_per_query["sara_k0_f1"].mean()), 4),
        "mean_parent_sara_k5_f1": round(float(df_per_query["sara_k5_f1"].mean()), 4),
        "mean_static_sara_k8_f1": round(mean_k8, 4),
        "mean_vanilla_rag_k10_f1": round(mean_k10, 4),
        "mean_oracle_sara_f1": round(mean_oracle_sara, 4),
        "oracle_sara_gain_over_k8": round(mean_oracle_sara - mean_k8, 4),
        "oracle_sara_pct_improvement_over_k8": round(((mean_oracle_sara - mean_k8) / mean_k8) * 100.0, 2),
        "mean_oracle_overall_f1": round(mean_oracle_overall, 4),
        "oracle_overall_gain_over_k10": round(mean_oracle_overall - mean_k10, 4),
        "oracle_overall_pct_improvement_over_k10": round(((mean_oracle_overall - mean_k10) / mean_k10) * 100.0, 2),
        "pct_questions_where_k0_equal_or_better_than_k8": round(float((df_per_query["sara_k0_f1"] >= df_per_query["sara_k8_f1"]).mean() * 100.0), 2),
        "pct_questions_where_k5_equal_or_better_than_k8": round(float((df_per_query["sara_k5_f1"] >= df_per_query["sara_k8_f1"]).mean() * 100.0), 2),
        "questions_within_0.01_of_sara_oracle_pct": tol_01,
        "questions_within_0.02_of_sara_oracle_pct": tol_02,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    oracle_json = processed_dir / "oracle_vs_static_summary.json"
    with open(oracle_json, "w", encoding="utf-8") as f:
        json.dump(oracle_summary, f, indent=2)
    print(f"Saved oracle vs static summary to: {oracle_json}")

    return df_per_query, oracle_summary, df_dist


def generate_plots(
    summary_df: pd.DataFrame,
    per_query_df: pd.DataFrame,
    df_raw: pd.DataFrame,
    plots_dir: Path,
    processed_dir: Path,
) -> List[Path]:
    """Generate all required publication-grade plots including heatmap and per-query response curves."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved_plots = []
    
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    k_vals = summary_df["k"].tolist()

    # 1. Mean Token F1 vs k
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(summary_df["k"], summary_df["mean_f1"], marker="o", color="#102C57", linewidth=2.2, label="Mean F1")
    ax.fill_between(
        summary_df["k"],
        summary_df["mean_f1"] - summary_df["std_f1"] * 0.1,
        summary_df["mean_f1"] + summary_df["std_f1"] * 0.1,
        alpha=0.15,
        color="#102C57",
    )
    ax.set_title("QASPER Development Set: Token F1 vs. Context Allocation k", fontsize=12, fontweight="bold")
    ax.set_xlabel("Natural Language Evidence Budget (k)", fontsize=10)
    ax.set_ylabel("Token F1", fontsize=10)
    ax.set_xticks(k_vals)
    ax.grid(True, linestyle="--", alpha=0.6)
    p1 = plots_dir / "01_mean_token_f1_vs_k.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    saved_plots.append(p1)

    # 2. Mean EM vs k
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(summary_df["k"], summary_df["mean_em"], marker="s", color="#0E7490", linewidth=2.2)
    ax.set_title("QASPER Development Set: Exact Match (EM) vs. Allocation k", fontsize=12, fontweight="bold")
    ax.set_xlabel("Natural Language Evidence Budget (k)", fontsize=10)
    ax.set_ylabel("Exact Match", fontsize=10)
    ax.set_xticks(k_vals)
    ax.grid(True, linestyle="--", alpha=0.6)
    p2 = plots_dir / "02_mean_em_vs_k.png"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    saved_plots.append(p2)

    # 3. Mean ROUGE-L vs k
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(summary_df["k"], summary_df["mean_rouge_l"], marker="^", color="#166534", linewidth=2.2)
    ax.set_title("QASPER Development Set: ROUGE-L vs. Allocation k", fontsize=12, fontweight="bold")
    ax.set_xlabel("Natural Language Evidence Budget (k)", fontsize=10)
    ax.set_ylabel("ROUGE-L", fontsize=10)
    ax.set_xticks(k_vals)
    ax.grid(True, linestyle="--", alpha=0.6)
    p3 = plots_dir / "03_mean_rouge_l_vs_k.png"
    fig.savefig(p3, bbox_inches="tight")
    plt.close(fig)
    saved_plots.append(p3)

    # 4. Mean Context Tokens vs k
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.bar(summary_df["k"].astype(str), summary_df["mean_context_tokens"], color="#64748B", alpha=0.85, width=0.6)
    ax.set_title("QASPER Development Set: Mean Context Tokens vs. Allocation k", fontsize=12, fontweight="bold")
    ax.set_xlabel("Natural Language Evidence Budget (k)", fontsize=10)
    ax.set_ylabel("Mean Context Tokens", fontsize=10)
    for i, v in enumerate(summary_df["mean_context_tokens"]):
        ax.text(i, v + 20, f"{int(v)}", ha="center", fontsize=9, fontweight="bold")
    p4 = plots_dir / "04_mean_input_tokens_vs_k.png"
    fig.savefig(p4, bbox_inches="tight")
    plt.close(fig)
    saved_plots.append(p4)

    # 5. Token F1 vs. Context Tokens (Pareto Frontier 1)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.scatter(summary_df["mean_context_tokens"], summary_df["mean_f1"], s=100, color="#102C57", zorder=3)
    for _, row in summary_df.iterrows():
        ax.annotate(
            f"k={int(row['k'])}",
            (row["mean_context_tokens"], row["mean_f1"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
            fontweight="bold",
        )
    ax.plot(summary_df["mean_context_tokens"], summary_df["mean_f1"], linestyle="--", color="#94A3B8", alpha=0.7)
    ax.set_title("Pareto Curve: Quality (Token F1) vs. Context Tokens", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean Context Tokens", fontsize=10)
    ax.set_ylabel("Token F1", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    p5 = plots_dir / "05_token_f1_vs_context_tokens.png"
    fig.savefig(p5, bbox_inches="tight")
    plt.close(fig)
    saved_plots.append(p5)

    # 6. Token F1 vs. Generation Latency
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.scatter(summary_df["mean_gen_latency_sec"], summary_df["mean_f1"], s=100, color="#0E7490", zorder=3)
    for _, row in summary_df.iterrows():
        ax.annotate(
            f"k={int(row['k'])}",
            (row["mean_gen_latency_sec"], row["mean_f1"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
            fontweight="bold",
        )
    ax.set_title("Quality (Token F1) vs. Measured Generation Latency", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean Generation Latency (seconds)", fontsize=10)
    ax.set_ylabel("Token F1", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    p6 = plots_dir / "06_token_f1_vs_generation_latency.png"
    fig.savefig(p6, bbox_inches="tight")
    plt.close(fig)
    saved_plots.append(p6)

    # 7. Token F1 vs. Synthesized Total Latency (Pareto Frontier 2)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.scatter(summary_df["mean_synthesized_total_latency_sec"], summary_df["mean_f1"], s=100, color="#6D28D9", zorder=3)
    for _, row in summary_df.iterrows():
        ax.annotate(
            f"k={int(row['k'])}",
            (row["mean_synthesized_total_latency_sec"], row["mean_f1"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
            fontweight="bold",
        )
    ax.set_title("Pareto Curve: Token F1 vs. Synthesized Total Latency", fontsize=12, fontweight="bold")
    ax.set_xlabel("Synthesized Total Latency [L_comp + L_gen] (seconds)", fontsize=10)
    ax.set_ylabel("Token F1", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    p7 = plots_dir / "07_token_f1_vs_synthesized_total_latency.png"
    fig.savefig(p7, bbox_inches="tight")
    plt.close(fig)
    saved_plots.append(p7)

    # 8. Distribution of Per-Question Best SARA k
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    sara_k_cols = [0, 2, 4, 5, 6, 8]
    sara_counts = per_query_df["best_k"].value_counts().reindex(sara_k_cols, fill_value=0)
    bars = ax.bar([str(k) for k in sara_k_cols], sara_counts.values, color="#102C57", alpha=0.85, width=0.6)
    ax.set_title("Distribution of Per-Question Optimal Allocation k* (SARA Candidate Set)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Optimal SARA Budget k* (smaller-k tie breaking)", fontsize=10)
    ax.set_ylabel("Number of Questions", fontsize=10)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{int(h)} ({h/len(per_query_df)*100:.1f}%)", ha="center", fontsize=9, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    p8 = plots_dir / "08_per_question_best_k_distribution.png"
    fig.savefig(p8, bbox_inches="tight")
    plt.close(fig)
    saved_plots.append(p8)

    # 9. Per-Question Response Heatmap (k_response_heatmap.png)
    pivoted_raw = df_raw.pivot(index="question_id", columns="k", values="token_f1")
    # Sort questions by best_sara_k then mean F1
    pivoted_raw["mean_f1"] = pivoted_raw.mean(axis=1)
    pivoted_raw = pivoted_raw.sort_values(by=["mean_f1"], ascending=False).drop(columns=["mean_f1"])

    fig, ax = plt.subplots(figsize=(8, 10), dpi=300)
    sns.heatmap(pivoted_raw, cmap="YlGnBu", cbar_kws={'label': 'Token F1'}, ax=ax, yticklabels=False)
    ax.set_title("Per-Question F1 Response Surface across Allocation k (231 Questions)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Evidence Allocation k", fontsize=10)
    ax.set_ylabel("231 QASPER Development Questions (Sorted by Mean F1)", fontsize=10)
    p9 = processed_dir / "k_response_heatmap.png"
    if p9.parent == processed_dir:
        # Also copy to plots_dir
        fig.savefig(plots_dir / "k_response_heatmap.png", bbox_inches="tight")
        fig.savefig(p9, bbox_inches="tight")
    plt.close(fig)
    saved_plots.append(p9)

    # 10. Per-Question Response Curves (per_query_response_curves.png)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=300, sharey=True)
    axes = axes.flatten()
    profiles = [0, 2, 4, 5, 6, 8]
    piv_f1 = df_raw.pivot(index="question_id", columns="k", values="token_f1")

    for i, profile_k in enumerate(profiles):
        ax = axes[i]
        # Find sample questions whose best SARA k is profile_k
        q_ids = per_query_df[per_query_df["best_k"] == profile_k]["question_id"].tolist()[:5]
        for qid in q_ids:
            if qid in piv_f1.index:
                y_vals = [piv_f1.loc[qid, k] for k in k_vals]
                ax.plot(k_vals, y_vals, marker="o", label=f"QID {qid}", alpha=0.8)
        ax.set_title(f"Target Profile: Optimal k* = {profile_k}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Allocation k", fontsize=9)
        if i % 3 == 0:
            ax.set_ylabel("Token F1", fontsize=9)
        ax.set_xticks(k_vals)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Representative Per-Question F1 Response Profiles across k", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p10 = processed_dir / "per_query_response_curves.png"
    fig.savefig(plots_dir / "per_query_response_curves.png", bbox_inches="tight")
    fig.savefig(p10, bbox_inches="tight")
    plt.close(fig)
    saved_plots.append(p10)

    print(f"Generated {len(saved_plots)} publication-quality plots in: {plots_dir}")
    return saved_plots


def main():
    parser = argparse.ArgumentParser(description="Aggregate fixed-k metrics, audit baseline selection, and run oracle analysis.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs/week2/sweep_config.yaml"),
        help="Path to sweep_config.yaml",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths_cfg = cfg["paths"]
    matrix_file = REPO_ROOT / paths_cfg["dev_fixed_k_matrix_file"]
    summary_csv = REPO_ROOT / paths_cfg["dev_fixed_k_summary_file"]
    per_query_csv = REPO_ROOT / paths_cfg["dev_per_query_k_matrix_file"]
    baseline_json = REPO_ROOT / paths_cfg["best_static_baseline_file"]
    plots_dir = REPO_ROOT / paths_cfg["plots_dir"]
    processed_dir = summary_csv.parent
    seed = cfg.get("seed", 42)

    df = load_raw_matrix(matrix_file)
    print(f"Loaded {len(df)} records from: {matrix_file}")

    summary_df = compute_summary_table(df, output_csv=summary_csv)
    per_query_df = compute_per_query_matrix(df, output_csv=per_query_csv)
    best_baseline = determine_best_static_baseline(summary_df, output_json=baseline_json, seed=seed)
    
    # Run Oracle Heterogeneity Analysis
    df_pq, oracle_summary, df_dist = analyze_oracle_and_per_query_heterogeneity(df, processed_dir=processed_dir)
    
    generate_plots(summary_df, per_query_df, df, plots_dir=plots_dir, processed_dir=processed_dir)

    print("\n============================================================")
    print("FIXED-K DEVELOPMENT RESPONSE SURFACE SUMMARY")
    print("============================================================")
    print(summary_df.to_string(index=False))
    print("============================================================")
    print(f"Best Static SARA Baseline (k_dev*): k={best_baseline['best_static_sara_k']} (Mean F1: {best_baseline['sara_k8_mean_f1']})")
    print(f"Vanilla RAG Reference Ceiling    : k=10 (Mean F1: {best_baseline['vanilla_rag_k10_f1']})")
    print(f"Oracle Adaptive SARA Potential   : Mean F1: {oracle_summary['mean_oracle_sara_f1']} (+{oracle_summary['oracle_sara_gain_over_k8']} F1 / +{oracle_summary['oracle_sara_pct_improvement_over_k8']}%)")
    print("============================================================\n")


if __name__ == "__main__":
    main()
