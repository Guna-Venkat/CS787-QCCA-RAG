"""Feature Families B, C, D: Retrieval Geometry, QPP, and Cross-Retriever Interaction for Week 7-C.

Implements:
- Family B: Semantic Score Geometry (full pool + candidate budget k boundary)
- Family C: BM25 Classical Retrieval Quality & QPP metrics (NQC, WIG, CV, concentration)
- Family D: Cross-Retriever Interaction (Spearman, Kendall, Jaccard, displacement)
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


def _safe_entropy(scores: Sequence[float]) -> Tuple[float, float]:
    """Compute Shannon entropy and normalized entropy of softmax distribution."""
    arr = np.array(scores, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0
    # Softmax
    max_val = np.max(arr)
    exp_s = np.exp(arr - max_val)
    sum_exp = np.sum(exp_s)
    if sum_exp <= 0.0 or np.isnan(sum_exp):
        return 0.0, 0.0
    p = exp_s / sum_exp
    p = np.clip(p, 1e-12, 1.0)
    entropy = float(-np.sum(p * np.log(p)))
    norm_entropy = float(entropy / np.log(len(arr))) if len(arr) > 1 else 0.0
    return entropy, norm_entropy


def compute_semantic_pool_geometry(semantic_scores: Sequence[float]) -> Dict[str, float]:
    """Compute query-level Family B full ranking geometry over 10 candidates."""
    arr = np.array(sorted(semantic_scores, reverse=True), dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {}

    s1 = float(arr[0])
    s2 = float(arr[1]) if n > 1 else s1
    s4_mean = float(np.mean(arr[:4])) if n >= 4 else float(np.mean(arr))
    s6_mean = float(np.mean(arr[:6])) if n >= 6 else float(np.mean(arr))
    s10_mean = float(np.mean(arr))
    std_s = float(np.std(arr))
    min_s = float(np.min(arr))
    max_s = float(np.max(arr))

    entropy, norm_entropy = _safe_entropy(arr)

    # Ranked linear slope
    ranks = np.arange(n, dtype=np.float64)
    if std_s > 1e-12 and n > 1:
        slope = float(np.polyfit(ranks, arr, 1)[0])
    else:
        slope = 0.0

    # Cumulative softmax mass of top 4
    exp_s = np.exp(arr - max_s)
    p = exp_s / max(1e-12, np.sum(exp_s))
    cum_mass_top4 = float(np.sum(p[:4])) if n >= 4 else 1.0

    return {
        "sem_top1": s1,
        "sem_top2": s2,
        "sem_top4_mean": s4_mean,
        "sem_top6_mean": s6_mean,
        "sem_top10_mean": s10_mean,
        "sem_std": std_s,
        "sem_min": min_s,
        "sem_max": max_s,
        "sem_entropy": entropy,
        "sem_norm_entropy": norm_entropy,
        "sem_top1_top2_margin": s1 - s2,
        "sem_top2_top4_margin": s2 - (float(arr[3]) if n >= 4 else s2),
        "sem_ranked_slope": slope,
        "sem_cum_mass_top4": cum_mass_top4,
    }


def compute_semantic_budget_geometry(
    sorted_semantic_scores: Sequence[float],
    k: int,
) -> Dict[str, float]:
    """Compute candidate-budget k specific Family B boundary features."""
    arr = np.array(sorted_semantic_scores, dtype=np.float64)
    n = len(arr)
    k_clamp = min(k, n)

    selected = arr[:k_clamp]
    tail = arr[k_clamp:]

    sel_mean = float(np.mean(selected)) if len(selected) > 0 else 0.0
    sel_min = float(np.min(selected)) if len(selected) > 0 else 0.0
    sel_std = float(np.std(selected)) if len(selected) > 0 else 0.0

    tail_mean = float(np.mean(tail)) if len(tail) > 0 else sel_min
    tail_max = float(np.max(tail)) if len(tail) > 0 else sel_min

    boundary_score = float(arr[k_clamp - 1]) if k_clamp > 0 else 0.0
    next_score = float(arr[k_clamp]) if k_clamp < n else boundary_score
    boundary_margin = boundary_score - next_score

    return {
        "sem_k_selected_mean": sel_mean,
        "sem_k_selected_min": sel_min,
        "sem_k_selected_std": sel_std,
        "sem_k_tail_mean": tail_mean,
        "sem_k_tail_max": tail_max,
        "sem_k_selected_vs_tail_gap": sel_mean - tail_mean,
        "sem_k_boundary_score": boundary_score,
        "sem_k_next_score": next_score,
        "sem_k_boundary_margin": boundary_margin,
    }


def compute_bm25_qpp_features(
    bm25_scores: Sequence[float],
    semantic_ranks: Sequence[int],
    k: int,
) -> Dict[str, float]:
    """Compute Family C BM25 pool geometry, QPP metrics, and candidate k features."""
    b_arr = np.array(bm25_scores, dtype=np.float64)
    n = len(b_arr)
    if n == 0:
        return {}

    b_sorted = np.sort(b_arr)[::-1]
    b_top1 = float(b_sorted[0])
    b_top2 = float(b_sorted[1]) if n > 1 else b_top1
    b_mean = float(np.mean(b_arr))
    b_std = float(np.std(b_arr))
    b_min = float(np.min(b_arr))
    b_max = float(np.max(b_arr))

    entropy, norm_entropy = _safe_entropy(b_sorted)

    # QPP metrics
    # NQC: Normalized Query Commitment (standard deviation of BM25 scores / mean)
    nqc = float(b_std / (b_mean + 1e-6))
    # WIG proxy: Weighted Information Gain (score separation of top from mean)
    wig = float((b_top1 - b_mean) / (b_std + 1e-6))
    # Coefficient of variation
    cv = float(b_std / (b_mean + 1e-6))
    # Score concentration: top 2 share of total BM25 mass
    concentration = float((b_top1 + b_top2) / (np.sum(b_arr) + 1e-6))

    # Candidate k BM25 features (evaluated on semantic top-k passages)
    selected_indices = semantic_ranks[:k]
    selected_b = [b_arr[idx] for idx in selected_indices if idx < n]
    bm25_k_cum = float(np.sum(selected_b)) if selected_b else 0.0
    bm25_k_mean = float(np.mean(selected_b)) if selected_b else 0.0

    boundary_idx = semantic_ranks[min(k - 1, n - 1)]
    next_idx = semantic_ranks[min(k, n - 1)]
    bm25_k_boundary = float(b_arr[boundary_idx]) if boundary_idx < n else 0.0
    bm25_k_next = float(b_arr[next_idx]) if next_idx < n else bm25_k_boundary
    bm25_k_margin = bm25_k_boundary - bm25_k_next

    return {
        "bm25_top1": b_top1,
        "bm25_top2": b_top2,
        "bm25_mean": b_mean,
        "bm25_std": b_std,
        "bm25_min": b_min,
        "bm25_max": b_max,
        "bm25_entropy": entropy,
        "bm25_norm_entropy": norm_entropy,
        "bm25_top1_top2_margin": b_top1 - b_top2,
        "bm25_nqc": nqc,
        "bm25_wig": wig,
        "bm25_coeff_variation": cv,
        "bm25_concentration": concentration,
        "bm25_k_cum_score": bm25_k_cum,
        "bm25_k_mean": bm25_k_mean,
        "bm25_k_boundary_score": bm25_k_boundary,
        "bm25_k_boundary_margin": bm25_k_margin,
    }


def compute_cross_retriever_features(
    bm25_scores: Sequence[float],
    semantic_scores: Sequence[float],
    semantic_ranks: Sequence[int],
    k: int,
) -> Dict[str, float]:
    """Compute Family D Cross-Retriever Interaction features."""
    b_arr = np.array(bm25_scores, dtype=np.float64)
    s_arr = np.array(semantic_scores, dtype=np.float64)
    n = len(b_arr)
    if n == 0 or len(s_arr) == 0:
        return {}

    # BM25 rank order is 0..n-1
    bm25_ranks_arr = np.arange(n)
    # Ranks of each passage under semantic scoring
    sem_rank_of_passage = np.empty(n, dtype=int)
    for r, p_idx in enumerate(semantic_ranks):
        if p_idx < n:
            sem_rank_of_passage[p_idx] = r

    # Rank correlations
    try:
        rho, _ = spearmanr(bm25_ranks_arr, sem_rank_of_passage)
        rho_val = float(rho) if not np.isnan(rho) else 0.0
    except Exception:
        rho_val = 0.0

    try:
        tau, _ = kendalltau(bm25_ranks_arr, sem_rank_of_passage)
        tau_val = float(tau) if not np.isnan(tau) else 0.0
    except Exception:
        tau_val = 0.0

    # Top-1 agreement
    top1_agree = 1.0 if (semantic_ranks[0] == 0) else 0.0

    # Jaccard set overlap at fixed budgets 2, 4, 6
    def _jaccard(k_val: int) -> float:
        set_bm25 = set(range(min(k_val, n)))
        set_sem = set(semantic_ranks[:k_val])
        return float(len(set_bm25.intersection(set_sem)) / max(1, len(set_bm25.union(set_sem))))

    jaccard_2 = _jaccard(2)
    jaccard_4 = _jaccard(4)
    jaccard_6 = _jaccard(6)

    # Rank displacements
    displacements = np.abs(bm25_ranks_arr - sem_rank_of_passage)
    mean_disp = float(np.mean(displacements))
    max_disp = float(np.max(displacements))

    # Normalized score disagreement
    b_norm = (b_arr - np.min(b_arr)) / (np.max(b_arr) - np.min(b_arr) + 1e-9)
    s_norm = (s_arr - np.min(s_arr)) / (np.max(s_arr) - np.min(s_arr) + 1e-9)
    mean_score_disagree = float(np.mean(np.abs(b_norm - s_norm)))

    # Candidate k specific cross-retriever features
    set_bm25_k = set(range(min(k, n)))
    selected_k_indices = semantic_ranks[:k]
    set_sem_k = set(selected_k_indices)
    cand_jaccard = float(len(set_bm25_k.intersection(set_sem_k)) / max(1, len(set_bm25_k.union(set_sem_k))))

    # Passages in semantic top-k whose original BM25 rank was >= k (i.e. outside BM25 top-k)
    passages_below_bm25_k = [idx for idx in selected_k_indices if idx >= k]
    count_below_k = len(passages_below_bm25_k)
    max_bm25_rank = int(max(selected_k_indices)) if selected_k_indices else 0
    frac_bm25_tail = float(count_below_k / max(1, k))

    return {
        "cross_spearman_rho": rho_val,
        "cross_kendall_tau": tau_val,
        "cross_top1_agreement": top1_agree,
        "cross_top2_jaccard": jaccard_2,
        "cross_top4_jaccard": jaccard_4,
        "cross_top6_jaccard": jaccard_6,
        "cross_mean_abs_displacement": mean_disp,
        "cross_max_displacement": max_disp,
        "cross_mean_norm_score_disagreement": mean_score_disagree,
        "cross_k_jaccard": cand_jaccard,
        "cross_k_semantic_below_bm25_k": float(count_below_k),
        "cross_k_max_bm25_rank": float(max_bm25_rank),
        "cross_k_frac_bm25_tail": frac_bm25_tail,
    }
