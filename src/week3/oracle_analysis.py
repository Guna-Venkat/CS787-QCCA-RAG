"""Oracle Quality and Headroom Analysis for Week 3.

Implements the Primary Quality Oracle over K_alloc = {2, 4, 5, 6, 8} with
deterministic smallest-k tie breaking, oracle headroom calculations,
best-k distribution, tie analysis, and gain attribution. Also computes
diagnostic RAG ceiling and pure compression gaps over K_sweep.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_K_ALLOC = [2, 4, 5, 6, 8]
DEFAULT_K_SWEEP = [0, 2, 4, 5, 6, 8, 10]


def load_raw_dev_matrix(raw_path: str) -> pd.DataFrame:
    """Load the frozen raw fixed-k matrix JSONL."""
    records = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    # Ensure types
    df["k"] = df["k"].astype(int)
    df["token_f1"] = df["token_f1"].astype(float)
    df["question_id"] = df["question_id"].astype(str)
    df["paper_id"] = df["paper_id"].astype(str)
    return df


def load_best_static_baseline(json_path: str) -> Dict[str, Any]:
    """Load the frozen best static baseline metadata."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def compute_per_query_f1_pivot(
    df_raw: pd.DataFrame,
    k_values: Optional[List[int]] = None
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Pivot raw dataframe into (question_id x k) matrix with paper_id mapping."""
    if k_values is None:
        k_values = DEFAULT_K_SWEEP
    
    # Filter to requested k_values
    df_sub = df_raw[df_raw["k"].isin(k_values)].copy()
    
    # Create question -> paper mapping
    q_to_paper = df_raw.groupby("question_id")["paper_id"].first().to_dict()
    
    # Pivot table
    df_pivot = df_sub.pivot(index="question_id", columns="k", values="token_f1")
    # Sort columns
    df_pivot = df_pivot[[k for k in k_values if k in df_pivot.columns]]
    return df_pivot, q_to_paper


def compute_quality_oracle(
    df_pivot: pd.DataFrame,
    k_alloc: Optional[List[int]] = None,
    static_k: int = 8
) -> pd.DataFrame:
    """Compute Quality Oracle over K_alloc with smallest-k tie breaking.
    
    Returns DataFrame with:
      question_id, best_k_qual, best_f1_qual, static_k, static_f1, oracle_gain,
      and individual f1_k{k} columns.
    """
    if k_alloc is None:
        k_alloc = DEFAULT_K_ALLOC
    
    # Ensure sorted order for deterministic smallest-k tie breaking
    sorted_k = sorted(k_alloc)
    sub_df = df_pivot[sorted_k].copy()
    
    results = []
    for q_id, row in sub_df.iterrows():
        f1_vals = [row[k] for k in sorted_k]
        max_f1 = max(f1_vals)
        # Smallest k achieving max_f1
        best_k = sorted_k[f1_vals.index(max_f1)]
        
        static_f1 = row[static_k] if static_k in row else np.nan
        gain = max_f1 - static_f1 if not np.isnan(static_f1) else np.nan
        
        res = {
            "question_id": str(q_id),
            "best_k_qual": int(best_k),
            "best_f1_qual": float(max_f1),
            "best_static_k": int(static_k),
            "best_static_f1": float(static_f1),
            "oracle_gain": float(gain)
        }
        for k in sorted_k:
            res[f"f1_k{k}"] = float(row[k])
        results.append(res)
        
    return pd.DataFrame(results)


def compute_oracle_headroom(
    oracle_df: pd.DataFrame
) -> Dict[str, float]:
    """Compute aggregate oracle headroom metrics."""
    mean_oracle_f1 = float(oracle_df["best_f1_qual"].mean())
    mean_static_f1 = float(oracle_df["best_static_f1"].mean())
    abs_headroom = mean_oracle_f1 - mean_static_f1
    rel_headroom = (abs_headroom / mean_static_f1 * 100.0) if mean_static_f1 > 0 else 0.0
    ratio = (mean_oracle_f1 / mean_static_f1) if mean_static_f1 > 0 else 0.0
    
    return {
        "mean_oracle_f1": round(mean_oracle_f1, 4),
        "mean_best_static_f1": round(mean_static_f1, 4),
        "absolute_headroom": round(abs_headroom, 4),
        "relative_headroom_pct": round(rel_headroom, 2),
        "oracle_static_ratio": round(ratio, 4),
        "n_questions": int(len(oracle_df))
    }


def compute_best_k_distribution(
    oracle_df: pd.DataFrame,
    k_alloc: Optional[List[int]] = None
) -> pd.DataFrame:
    """Compute frequency and cumulative distribution of best_k_qual."""
    if k_alloc is None:
        k_alloc = DEFAULT_K_ALLOC
        
    n_total = len(oracle_df)
    counts = oracle_df["best_k_qual"].value_counts().to_dict()
    
    rows = []
    cum_count = 0
    for k in sorted(k_alloc):
        c = counts.get(k, 0)
        cum_count += c
        pct = (c / n_total) * 100.0
        cum_pct = (cum_count / n_total) * 100.0
        rows.append({
            "k": k,
            "count": c,
            "pct": round(pct, 2),
            "cumulative_count": cum_count,
            "cumulative_pct": round(cum_pct, 2)
        })
    return pd.DataFrame(rows)


def compute_tie_analysis(
    df_pivot: pd.DataFrame,
    k_alloc: Optional[List[int]] = None,
    tolerance: float = 1e-4,
    near_tie_threshold: float = 0.01
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Analyze ties at maximum F1 across K_alloc."""
    if k_alloc is None:
        k_alloc = DEFAULT_K_ALLOC
    sorted_k = sorted(k_alloc)
    
    records = []
    summary_counts = {
        "strict_unique_winner": 0,
        "two_way_tie": 0,
        "three_way_tie": 0,
        "four_way_tie": 0,
        "all_five_tied": 0,
        "all_zero_tied": 0,
        "near_ties_within_0_01": 0
    }
    
    for q_id, row in df_pivot[sorted_k].iterrows():
        vals = np.array([row[k] for k in sorted_k])
        max_val = np.max(vals)
        
        # Exact ties
        tied_indices = np.where(np.abs(vals - max_val) <= tolerance)[0]
        tied_k = [sorted_k[i] for i in tied_indices]
        n_tied = len(tied_k)
        
        # Near ties
        near_indices = np.where((max_val - vals) <= near_tie_threshold)[0]
        n_near = len(near_indices)
        
        is_all_zero = bool(max_val == 0.0 and n_tied == len(sorted_k))
        is_strict = (n_tied == 1)
        
        if is_all_zero:
            summary_counts["all_zero_tied"] += 1
        if is_strict:
            summary_counts["strict_unique_winner"] += 1
        elif n_tied == 2:
            summary_counts["two_way_tie"] += 1
        elif n_tied == 3:
            summary_counts["three_way_tie"] += 1
        elif n_tied == 4:
            summary_counts["four_way_tie"] += 1
        elif n_tied == 5:
            summary_counts["all_five_tied"] += 1
            
        if n_near > 1:
            summary_counts["near_ties_within_0_01"] += 1
            
        records.append({
            "question_id": str(q_id),
            "max_f1": float(round(max_val, 4)),
            "n_tied_k": int(n_tied),
            "tied_k_set": ",".join(str(k) for k in tied_k),
            "smallest_tied_k": int(tied_k[0]),
            "is_strict_winner": is_strict,
            "is_all_zero_tie": is_all_zero,
            "n_near_tied_k": int(n_near)
        })
        
    df_ties = pd.DataFrame(records)
    return df_ties, summary_counts


def compute_gain_attribution(
    oracle_df: pd.DataFrame,
    k_alloc: Optional[List[int]] = None
) -> pd.DataFrame:
    """Attribute oracle gain over static k to each oracle-selected k."""
    if k_alloc is None:
        k_alloc = DEFAULT_K_ALLOC
        
    total_gain = float(oracle_df["oracle_gain"].sum())
    rows = []
    for k in sorted(k_alloc):
        sub = oracle_df[oracle_df["best_k_qual"] == k]
        k_gain = float(sub["oracle_gain"].sum())
        n_q = len(sub)
        mean_k_gain = float(sub["oracle_gain"].mean()) if n_q > 0 else 0.0
        pct_of_total_gain = (k_gain / total_gain * 100.0) if total_gain > 0 else 0.0
        
        rows.append({
            "best_k_qual": k,
            "question_count": n_q,
            "total_gain": round(k_gain, 4),
            "mean_gain_per_question": round(mean_k_gain, 4),
            "pct_of_total_gain": round(pct_of_total_gain, 2)
        })
    return pd.DataFrame(rows)


def compute_diagnostics_sweep(
    df_pivot: pd.DataFrame,
    k_alloc: Optional[List[int]] = None
) -> Dict[str, Any]:
    """Compute diagnostic RAG ceiling gap (k=10) and Pure Compression gap (k=0)."""
    if k_alloc is None:
        k_alloc = DEFAULT_K_ALLOC
    sorted_k = sorted(k_alloc)
    
    max_alloc = df_pivot[sorted_k].max(axis=1)
    
    # RAG ceiling gap: F1(q, 10) - max_alloc(q)
    f1_k10 = df_pivot[10]
    rag_gap = f1_k10 - max_alloc
    k10_wins = (rag_gap > 1e-4)
    
    # Pure compression gap: F1(q, 0) - max_alloc(q)
    f1_k0 = df_pivot[0]
    comp_gap = f1_k0 - max_alloc
    k0_wins = (comp_gap > 1e-4)
    
    both_win = (k10_wins & k0_wins)
    
    return {
        "k10_strictly_exceeds_alloc_count": int(k10_wins.sum()),
        "k10_strictly_exceeds_alloc_pct": round(float(k10_wins.mean() * 100.0), 2),
        "mean_rag_ceiling_gap_overall": round(float(rag_gap.mean()), 4),
        "mean_rag_ceiling_gap_when_winning": round(float(rag_gap[k10_wins].mean()), 4) if k10_wins.sum() > 0 else 0.0,
        "max_rag_ceiling_gap": round(float(rag_gap.max()), 4),
        "k0_strictly_exceeds_alloc_count": int(k0_wins.sum()),
        "k0_strictly_exceeds_alloc_pct": round(float(k0_wins.mean() * 100.0), 2),
        "mean_comp_gap_overall": round(float(comp_gap.mean()), 4),
        "mean_comp_gap_when_winning": round(float(comp_gap[k0_wins].mean()), 4) if k0_wins.sum() > 0 else 0.0,
        "both_k0_k10_exceed_count": int(both_win.sum())
    }


def run_oracle_analysis(
    raw_matrix_path: str,
    best_static_path: str,
    output_dir: str
) -> Dict[str, Any]:
    """Run full oracle analysis pipeline and save processed artifacts."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading raw fixed-k matrix from %s", raw_matrix_path)
    df_raw = load_raw_dev_matrix(raw_matrix_path)
    
    logger.info("Loading best static baseline from %s", best_static_path)
    best_static_info = load_best_static_baseline(best_static_path)
    static_k = int(best_static_info.get("best_static_sara_k", 8))
    
    logger.info("Pivoting per-query matrix across K_sweep")
    df_pivot, q_to_paper = compute_per_query_f1_pivot(df_raw, DEFAULT_K_SWEEP)
    
    logger.info("Computing primary Quality Oracle over K_alloc = %s", DEFAULT_K_ALLOC)
    oracle_df = compute_quality_oracle(df_pivot, DEFAULT_K_ALLOC, static_k=static_k)
    oracle_df["paper_id"] = oracle_df["question_id"].map(q_to_paper)
    
    # Reorder columns
    cols_order = [
        "question_id", "paper_id", "best_k_qual", "best_f1_qual",
        "best_static_k", "best_static_f1", "oracle_gain",
        "f1_k2", "f1_k4", "f1_k5", "f1_k6", "f1_k8"
    ]
    oracle_df = oracle_df[cols_order]
    
    # Save dev_quality_oracle.csv
    oracle_csv = out_path / "dev_quality_oracle.csv"
    oracle_df.to_csv(oracle_csv, index=False)
    logger.info("Saved quality oracle records to %s", oracle_csv)
    
    # Compute Headroom
    headroom = compute_oracle_headroom(oracle_df)
    logger.info("Oracle Headroom: Mean Oracle=%.4f, Static(k=%d)=%.4f, Headroom=+%.4f (+%.2f%%)",
                headroom["mean_oracle_f1"], static_k, headroom["mean_best_static_f1"],
                headroom["absolute_headroom"], headroom["relative_headroom_pct"])
    
    # Compute Distribution
    dist_df = compute_best_k_distribution(oracle_df, DEFAULT_K_ALLOC)
    dist_csv = out_path / "quality_oracle_distribution.csv"
    dist_df.to_csv(dist_csv, index=False)
    logger.info("Saved quality oracle distribution to %s", dist_csv)
    
    # Compute Ties
    ties_df, tie_summary = compute_tie_analysis(df_pivot, DEFAULT_K_ALLOC)
    ties_csv = out_path / "tie_analysis.csv"
    ties_df.to_csv(ties_csv, index=False)
    logger.info("Saved tie analysis to %s", ties_csv)
    
    # Compute Gain Attribution
    gain_df = compute_gain_attribution(oracle_df, DEFAULT_K_ALLOC)
    gain_csv = out_path / "oracle_gain_attribution.csv"
    gain_df.to_csv(gain_csv, index=False)
    logger.info("Saved gain attribution to %s", gain_csv)
    
    # Compute Diagnostics
    diag_summary = compute_diagnostics_sweep(df_pivot, DEFAULT_K_ALLOC)
    logger.info("Diagnostic RAG ceiling gap: k=10 wins in %d queries (%.2f%%)",
                diag_summary["k10_strictly_exceeds_alloc_count"],
                diag_summary["k10_strictly_exceeds_alloc_pct"])
    
    summary = {
        "headroom": headroom,
        "best_k_distribution": dist_df.to_dict(orient="records"),
        "tie_summary": tie_summary,
        "gain_attribution": gain_df.to_dict(orient="records"),
        "diagnostic_sweep": diag_summary
    }
    
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Quality Oracle Analysis")
    parser.add_argument("--raw_matrix", default="results/week2/raw/dev_fixed_k_matrix.jsonl")
    parser.add_argument("--best_static", default="results/week2/processed/best_static_baseline.json")
    parser.add_argument("--output_dir", default="results/week3/processed")
    args = parser.parse_args()
    
    summary = run_oracle_analysis(args.raw_matrix, args.best_static, args.output_dir)
    print(json.dumps(summary, indent=2))
