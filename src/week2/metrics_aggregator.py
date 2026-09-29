"""Stage 3: Metrics Aggregation, Baseline Determination, & Visualization.

Processes raw output matrix (dev_fixed_k_matrix.jsonl) and generates:
1. dev_fixed_k_summary.csv: Summary metrics table across all k values.
2. dev_per_query_k_matrix.csv: Per-question F1 across k (input for Week 3 oracle).
3. best_static_baseline.json: Deterministically selected k_dev* baseline.
4. 8 publication-grade plots in results/week2/plots/.
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
    mean_k10_tokens = k10_rows["context_tokens"].mean() if not k10_rows.empty else 1.0

    summary_rows = []
    for k in k_order:
        k_df = df[df["k"] == k]
        mean_ctx = k_df["context_tokens"].mean()
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
    """Generate per-question F1 across all k values with deterministic tie-breaking for best_k."""
    k_values = sorted(df["k"].unique())
    pivoted = df.pivot(index=["question_id", "paper_id"], columns="k", values="token_f1").reset_index()
    
    # Rename columns to F1_k{k}
    rename_dict = {k: f"F1_k{k}" for k in k_values}
    pivoted.rename(columns=rename_dict, inplace=True)
    f1_cols = [f"F1_k{k}" for k in k_values]

    best_k_list = []
    best_f1_list = []

    for _, row in pivoted.iterrows():
        # Find maximum F1
        vals = [row[col] for col in f1_cols]
        max_f1 = max(vals)
        # Deterministic tie-breaking: pick smallest k with max F1
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
    """Identify k_dev* on development set with deterministic tie-breaking."""
    max_f1 = summary_df["mean_f1"].max()
    candidates = summary_df[summary_df["mean_f1"] == max_f1]
    best_row = candidates.sort_values(by="k").iloc[0]

    baseline_info = {
        "best_static_k": int(best_row["k"]),
        "mean_f1": float(best_row["mean_f1"]),
        "exact_match": float(best_row["mean_em"]),
        "rouge_l": float(best_row["mean_rouge_l"]),
        "mean_context_tokens": float(best_row["mean_context_tokens"]),
        "context_reduction_pct": float(best_row["context_reduction_pct"]),
        "selection_rule": "argmax_k Mean_F1_dev(k) with smaller-k tie-breaking",
        "evaluated_question_count": int(best_row["n_questions"]),
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "Development set static baseline. Test set untouched.",
    }

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(baseline_info, f, indent=2)
        print(f"Saved best static baseline metadata to: {output_json}")

    return baseline_info


def generate_plots(summary_df: pd.DataFrame, per_query_df: pd.DataFrame, plots_dir: Path) -> List[Path]:
    """Generate the 8 required publication-grade plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved_plots = []
    
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    k_vals = summary_df["k"].tolist()

    # 1. Mean Token F1 vs k
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(summary_df["k"], summary_df["mean_f1"], marker="o", color="#102C57", linewidth=2.2, label="Mean F1")
    ax.fill_between(
        summary_df["k"],
        summary_df["mean_f1"] - summary_df["std_f1"] * 0.1,  # standard error approximation
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

    # 8. Distribution of Per-Question Best k
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    best_k_counts = per_query_df["best_k"].value_counts().reindex(k_vals, fill_value=0)
    bars = ax.bar([str(k) for k in k_vals], best_k_counts.values, color="#1E293B", alpha=0.85, width=0.6)
    ax.set_title("Distribution of Per-Question Optimal Allocation k*", fontsize=12, fontweight="bold")
    ax.set_xlabel("Optimal Budget k* (smaller-k tie breaking)", fontsize=10)
    ax.set_ylabel("Number of Questions", fontsize=10)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{int(h)}", ha="center", fontsize=9, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    p8 = plots_dir / "08_per_question_best_k_distribution.png"
    fig.savefig(p8, bbox_inches="tight")
    plt.close(fig)
    saved_plots.append(p8)

    print(f"Generated {len(saved_plots)} publication-quality plots in: {plots_dir}")
    return saved_plots


def main():
    parser = argparse.ArgumentParser(description="Aggregate fixed-k metrics and generate plots.")
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
    seed = cfg.get("seed", 42)

    df = load_raw_matrix(matrix_file)
    print(f"Loaded {len(df)} records from: {matrix_file}")

    summary_df = compute_summary_table(df, output_csv=summary_csv)
    per_query_df = compute_per_query_matrix(df, output_csv=per_query_csv)
    best_baseline = determine_best_static_baseline(summary_df, output_json=baseline_json, seed=seed)
    generate_plots(summary_df, per_query_df, plots_dir=plots_dir)

    print("\n============================================================")
    print("FIXED-K DEVELOPMENT RESPONSE SURFACE SUMMARY")
    print("============================================================")
    print(summary_df.to_string(index=False))
    print("============================================================")
    print(f"Best Static SARA Baseline (k_dev*): k={best_baseline['best_static_k']} (Mean F1: {best_baseline['mean_f1']})")
    print("============================================================\n")


if __name__ == "__main__":
    main()
