"""Ranking disagreement analysis and future adaptive-gate features for Week 6-B.

Implements:
- Jaccard set overlap and differing passage counts between BM25 and Semantic routing.
- Rank correlation (Spearman rho) between lexical and semantic score orders.
- Disagreement stratification vs Delta F1 = F1(R1) - F1(R0).
- Score-confidence feature extraction for future adaptive-alpha gates (tables/w6b_future_gate_features.csv).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.week6.hybrid_routing import (
    compute_ranks_from_scores,
    normalize_scores_minmax,
)
from src.week6.routing import route_r0_bm25_prefix, route_r1_semantic_topk

logger = logging.getLogger(__name__)


def compute_jaccard_overlap(
    set_a: Sequence[int],
    set_b: Sequence[int],
) -> float:
    """Compute Jaccard similarity between two sets of passage indices:
        J(A, B) = |A ∩ B| / |A ∪ B|
    """
    s_a = set(set_a)
    s_b = set(set_b)
    if not s_a and not s_b:
        return 1.0
    union = s_a | s_b
    if not union:
        return 1.0
    intersection = s_a & s_b
    return float(len(intersection) / len(union))


def compute_differing_passages_count(
    set_a: Sequence[int],
    set_b: Sequence[int],
) -> int:
    """Number of passages in set_b not present in set_a:
        |B \\ A|
    """
    return len(set(set_b) - set(set_a))


def compute_rank_correlation(
    scores_or_ranks_a: Union[np.ndarray, Sequence[float]],
    scores_or_ranks_b: Union[np.ndarray, Sequence[float]],
) -> float:
    """Compute Spearman rank correlation between two score/rank vectors.

    Returns:
        corr: Float in [-1.0, 1.0], or 0.0 if variance is zero.
    """
    arr_a = np.asarray(scores_or_ranks_a, dtype=np.float32)
    arr_b = np.asarray(scores_or_ranks_b, dtype=np.float32)

    if np.all(arr_a == arr_a[0]) or np.all(arr_b == arr_b[0]):
        return 0.0

    corr, _ = spearmanr(arr_a, arr_b)
    if np.isnan(corr):
        return 0.0
    return float(corr)


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute numerically stable softmax probabilities."""
    e_x = np.exp((x - np.max(x)) / max(temperature, 1e-6))
    sum_e = np.sum(e_x)
    return e_x / (sum_e + 1e-12)


def extract_future_gate_features_per_query(
    question_id: str,
    paper_id: str,
    bm25_scores: Sequence[float],
    sem_scores: Sequence[float],
    budgets: Sequence[int] = (2, 4, 6),
) -> Dict[str, Any]:
    """Extract query-level confidence and cross-signal features for future adaptive alpha gates.

    Does NOT train any model. Saves features to tables/w6b_future_gate_features.csv.

    Features:
    - BM25: top1, top1-top2 margin, entropy, normalized entropy, std.
    - Semantic: top1, top1-top2 margin, entropy, normalized entropy, std.
    - Cross-signal: rank correlation, top1 agreement, mean normalized score difference, top-k Jaccard.
    """
    b_arr = np.asarray(bm25_scores, dtype=np.float32)
    s_arr = np.asarray(sem_scores, dtype=np.float32)
    n = len(b_arr)

    # 1. BM25 score features
    b_sorted = np.sort(b_arr)[::-1]
    bm25_top1 = float(b_sorted[0])
    bm25_margin = float(b_sorted[0] - b_sorted[1]) if n > 1 else 0.0
    b_probs = softmax(b_arr)
    bm25_entropy = float(-np.sum(b_probs * np.log(b_probs + 1e-12)))
    bm25_norm_entropy = float(bm25_entropy / np.log(max(n, 2)))
    bm25_std = float(np.std(b_arr))

    # 2. Semantic score features
    s_sorted = np.sort(s_arr)[::-1]
    sem_top1 = float(s_sorted[0])
    sem_margin = float(s_sorted[0] - s_sorted[1]) if n > 1 else 0.0
    s_probs = softmax(s_arr)
    sem_entropy = float(-np.sum(s_probs * np.log(s_probs + 1e-12)))
    sem_norm_entropy = float(sem_entropy / np.log(max(n, 2)))
    sem_std = float(np.std(s_arr))

    # 3. Cross-signal features
    b_norm = normalize_scores_minmax(b_arr)
    s_norm = normalize_scores_minmax(s_arr)
    mean_norm_diff = float(np.mean(np.abs(b_norm - s_norm)))

    b_ranks = compute_ranks_from_scores(b_arr, reverse=True)
    s_ranks = compute_ranks_from_scores(s_arr, reverse=True)
    rank_corr = compute_rank_correlation(b_ranks, s_ranks)

    top1_b_idx = int(np.argmin(b_ranks))
    top1_s_idx = int(np.argmin(s_ranks))
    top1_agreement = 1.0 if top1_b_idx == top1_s_idx else 0.0

    features: Dict[str, Any] = {
        "question_id": str(question_id),
        "paper_id": str(paper_id),
        "bm25_top1": bm25_top1,
        "bm25_margin": bm25_margin,
        "bm25_entropy": bm25_entropy,
        "bm25_norm_entropy": bm25_norm_entropy,
        "bm25_std": bm25_std,
        "sem_top1": sem_top1,
        "sem_margin": sem_margin,
        "sem_entropy": sem_entropy,
        "sem_norm_entropy": sem_norm_entropy,
        "sem_std": sem_std,
        "cross_rank_corr": rank_corr,
        "cross_top1_agreement": top1_agreement,
        "cross_mean_norm_diff": mean_norm_diff,
    }

    # Add budget-specific Jaccard overlap features
    for k in budgets:
        sel_r0 = route_r0_bm25_prefix(k=k, n_passages=n)
        sel_r1 = route_r1_semantic_topk(s_arr, k=k)
        j_k = compute_jaccard_overlap(sel_r0, sel_r1)
        exact_k = 1.0 if j_k == 1.0 else 0.0
        features[f"cross_jaccard_k{k}"] = j_k
        features[f"cross_exact_agreement_k{k}"] = exact_k

    return features


def build_disagreement_analysis_dataframe(
    question_ids: Sequence[str],
    paper_ids: Sequence[str],
    bm25_scores_list: Sequence[Sequence[float]],
    sem_scores_list: Sequence[Sequence[float]],
    r0_f1_by_k: Optional[Dict[int, Sequence[float]]] = None,
    r1_f1_by_k: Optional[Dict[int, Sequence[float]]] = None,
    budgets: Sequence[int] = (2, 4, 6),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build query-level and aggregated disagreement analysis tables.

    Returns:
        (df_queries, df_summary):
            df_queries: Per-query disagreement metrics and Delta F1.
            df_summary: Stratified Delta F1 by disagreement bins (High, Medium, Low).
    """
    rows = []
    n_queries = len(question_ids)

    for i in range(n_queries):
        qid = str(question_ids[i])
        pid = str(paper_ids[i])
        b_scores = bm25_scores_list[i]
        s_scores = sem_scores_list[i]

        b_ranks = compute_ranks_from_scores(b_scores, reverse=True)
        s_ranks = compute_ranks_from_scores(s_scores, reverse=True)
        corr = compute_rank_correlation(b_ranks, s_ranks)

        row = {
            "question_id": qid,
            "paper_id": pid,
            "rank_correlation": corr,
        }

        for k in budgets:
            sel_r0 = route_r0_bm25_prefix(k=k, n_passages=len(b_scores))
            sel_r1 = route_r1_semantic_topk(s_scores, k=k)
            jaccard = compute_jaccard_overlap(sel_r0, sel_r1)
            diff_count = compute_differing_passages_count(sel_r0, sel_r1)

            row[f"jaccard_k{k}"] = jaccard
            row[f"differing_passages_k{k}"] = diff_count
            row[f"exact_match_k{k}"] = 1 if jaccard == 1.0 else 0

            # Delta F1 if provided
            if r0_f1_by_k and r1_f1_by_k and k in r0_f1_by_k and k in r1_f1_by_k:
                f1_0 = float(r0_f1_by_k[k][i])
                f1_1 = float(r1_f1_by_k[k][i])
                row[f"f1_r0_k{k}"] = f1_0
                row[f"f1_r1_k{k}"] = f1_1
                row[f"delta_f1_k{k}"] = f1_1 - f1_0

        rows.append(row)

    df_queries = pd.DataFrame(rows)

    # Aggregated Summary by Bins
    summary_rows = []
    for k in budgets:
        j_col = f"jaccard_k{k}"
        delta_col = f"delta_f1_k{k}"

        mean_j = float(df_queries[j_col].mean())
        exact_rate = float(df_queries[f"exact_match_k{k}"].mean())

        has_f1 = delta_col in df_queries.columns

        # Define bins: High (J >= 0.67), Medium (0.33 <= J < 0.67), Low (J < 0.33)
        bins = [
            ("High Agreement (J >= 0.67)", df_queries[df_queries[j_col] >= 0.67]),
            ("Medium Agreement (0.33 <= J < 0.67)", df_queries[(df_queries[j_col] >= 0.33) & (df_queries[j_col] < 0.67)]),
            ("Low Agreement (J < 0.33)", df_queries[df_queries[j_col] < 0.33]),
        ]

        spearman_j_delta = np.nan
        if has_f1:
            spearman_j_delta = compute_rank_correlation(df_queries[j_col], df_queries[delta_col])

        for bin_name, bin_df in bins:
            bin_count = len(bin_df)
            bin_pct = bin_count / n_queries * 100.0 if n_queries > 0 else 0.0
            mean_delta = float(bin_df[delta_col].mean()) if (has_f1 and bin_count > 0) else np.nan

            summary_rows.append({
                "budget_k": k,
                "agreement_bin": bin_name,
                "n_queries": bin_count,
                "query_pct": bin_pct,
                "mean_delta_f1": mean_delta,
                "overall_mean_jaccard": mean_j,
                "overall_exact_rate": exact_rate,
                "spearman_corr_jaccard_delta_f1": spearman_j_delta,
            })

    df_summary = pd.DataFrame(summary_rows)
    return df_queries, df_summary
