"""Feature Families E & F: Set Redundancy, Diversity, and Tail Threat for Week 7-C.

Implements:
- Family E: Selected passage set diversity, centroid dispersion, participation ratio, redundancy ratio
- Family F: Tail Threat & Marginal Evidence (novelty, tail relevance mass, boundary drop, threat count)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np


def _unit_normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize vectors along last dimension."""
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.clip(norm, 1e-9, None)


def compute_set_diversity_features(
    passage_embeddings: np.ndarray,
    semantic_ranks: Sequence[int],
    k: int,
) -> Dict[str, float]:
    """Compute Family E set diversity, redundancy, and centroid dispersion for candidate k.

    Args:
        passage_embeddings: Shape (10, D) float array of passage embeddings.
        semantic_ranks: Length 10 sequence of passage indices sorted by semantic score descending.
        k: Candidate budget in {2, 4, 5, 6, 8}.

    Returns:
        Dictionary of Family E features.
    """
    n_total = len(passage_embeddings)
    k_clamp = min(k, n_total)
    sel_indices = [idx for idx in semantic_ranks[:k_clamp] if idx < n_total]
    tail_indices = [idx for idx in semantic_ranks[k_clamp:] if idx < n_total]

    if len(sel_indices) < 2:
        return {
            "div_k_mean_pairwise_sim": 1.0,
            "div_k_max_pairwise_sim": 1.0,
            "div_k_min_pairwise_sim": 1.0,
            "div_k_std_pairwise_sim": 0.0,
            "div_k_centroid_dispersion": 0.0,
            "div_k_effective_clusters": 1.0,
            "div_k_redundancy_ratio": 1.0,
            "div_k_tail_mean_pairwise_sim": 0.0,
            "div_k_tail_to_selected_dist": 0.0,
        }

    E_norm = _unit_normalize(passage_embeddings)
    E_sel = E_norm[sel_indices]  # Shape (k, D)

    # Pairwise cosine similarity matrix C
    C_sel = np.matmul(E_sel, E_sel.T)  # Shape (k, k)
    # Extract upper triangle off-diagonal elements
    triu_indices = np.triu_indices(len(sel_indices), k=1)
    pair_sims = C_sel[triu_indices]

    mean_pair = float(np.mean(pair_sims)) if len(pair_sims) > 0 else 1.0
    max_pair = float(np.max(pair_sims)) if len(pair_sims) > 0 else 1.0
    min_pair = float(np.min(pair_sims)) if len(pair_sims) > 0 else 1.0
    std_pair = float(np.std(pair_sims)) if len(pair_sims) > 0 else 0.0

    # Centroid dispersion: distance of selected items to their normalized centroid
    centroid = np.mean(E_sel, axis=0, keepdims=True)
    centroid_norm = _unit_normalize(centroid)
    dists_to_centroid = 1.0 - np.sum(E_sel * centroid_norm, axis=1)
    centroid_dispersion = float(np.mean(dists_to_centroid))

    # Effective number of clusters via participation ratio of Gram matrix: Tr(C)^2 / ||C||_F^2
    tr_c = np.trace(C_sel)
    frob_sq = np.sum(C_sel ** 2)
    effective_clusters = float((tr_c ** 2) / max(1e-9, frob_sq))

    # Redundancy ratio: fraction of pairs with cosine similarity > 0.80
    redundancy_ratio = float(np.mean(pair_sims > 0.80)) if len(pair_sims) > 0 else 0.0

    # Tail set properties
    if len(tail_indices) >= 2:
        E_tail = E_norm[tail_indices]
        C_tail = np.matmul(E_tail, E_tail.T)
        tail_pairs = C_tail[np.triu_indices(len(tail_indices), k=1)]
        tail_mean_pair = float(np.mean(tail_pairs))
    else:
        tail_mean_pair = 0.0

    if len(tail_indices) > 0:
        E_tail = E_norm[tail_indices]
        tail_to_centroid = float(np.mean(1.0 - np.sum(E_tail * centroid_norm, axis=1)))
    else:
        tail_to_centroid = 0.0

    return {
        "div_k_mean_pairwise_sim": mean_pair,
        "div_k_max_pairwise_sim": max_pair,
        "div_k_min_pairwise_sim": min_pair,
        "div_k_std_pairwise_sim": std_pair,
        "div_k_centroid_dispersion": centroid_dispersion,
        "div_k_effective_clusters": effective_clusters,
        "div_k_redundancy_ratio": redundancy_ratio,
        "div_k_tail_mean_pairwise_sim": tail_mean_pair,
        "div_k_tail_to_selected_dist": tail_to_centroid,
    }


def compute_tail_threat_features(
    semantic_scores: Sequence[float],
    passage_embeddings: np.ndarray,
    semantic_ranks: Sequence[int],
    k: int,
) -> Dict[str, float]:
    """Compute Family F Tail Threat & Marginal Evidence features for candidate budget k.

    Formulation:
        Selected: S_k = {D_1^sem, ..., D_k^sem}
        Tail:     T_k = {D_{k+1}^sem, ..., D_10^sem}
        For each tail passage D_t:
            novelty(D_t) = sim(q, D_t) - max_{D_s in S_k} cos(D_t, D_s)

    Args:
        semantic_scores: Sequence of 10 float semantic relevance scores.
        passage_embeddings: Shape (10, D) float array of passage embeddings.
        semantic_ranks: Length 10 sequence of passage indices sorted by semantic score descending.
        k: Candidate budget in {2, 4, 5, 6, 8}.

    Returns:
        Dictionary of Family F features.
    """
    s_arr = np.array(semantic_scores, dtype=np.float64)
    n_total = len(s_arr)
    k_clamp = min(k, n_total)

    sel_indices = [idx for idx in semantic_ranks[:k_clamp] if idx < n_total]
    tail_indices = [idx for idx in semantic_ranks[k_clamp:] if idx < n_total]

    total_sem_mass = float(np.sum(s_arr)) if n_total > 0 else 1.0
    median_sem = float(np.median(s_arr)) if n_total > 0 else 0.0

    boundary_score = float(s_arr[sel_indices[-1]]) if sel_indices else 0.0

    if not tail_indices:
        return {
            "tail_k_max_semantic_relevance": 0.0,
            "tail_k_top2_mean_relevance": 0.0,
            "tail_k_relevance_mass": 0.0,
            "tail_k_boundary_drop": 0.0,
            "tail_k_max_novelty": 0.0,
            "tail_k_mean_pos_novelty": 0.0,
            "tail_k_top2_novelty": 0.0,
            "tail_k_high_threat_count": 0.0,
        }

    tail_scores = [float(s_arr[idx]) for idx in tail_indices]
    tail_scores_sorted = sorted(tail_scores, reverse=True)

    max_tail_rel = tail_scores_sorted[0]
    top2_tail_mean = float(np.mean(tail_scores_sorted[:2])) if len(tail_scores_sorted) >= 2 else max_tail_rel
    tail_mass = float(np.sum(tail_scores) / max(1e-9, total_sem_mass))
    boundary_drop = boundary_score - max_tail_rel

    # Novelty computation
    E_norm = _unit_normalize(passage_embeddings)
    E_sel = E_norm[sel_indices]   # Shape (k, D)
    E_tail = E_norm[tail_indices] # Shape (tail_len, D)

    # Cosine similarities between tail and selected: shape (tail_len, k)
    sim_tail_sel = np.matmul(E_tail, E_sel.T)
    max_sim_to_sel = np.max(sim_tail_sel, axis=1)  # Shape (tail_len,)

    # Novelty: semantic score of tail minus max similarity to selected set
    novelties = np.array(tail_scores, dtype=np.float64) - max_sim_to_sel
    max_novelty = float(np.max(novelties))
    mean_pos_novelty = float(np.mean(np.maximum(0.0, novelties)))
    sorted_novelties = np.sort(novelties)[::-1]
    top2_novelty = float(np.mean(sorted_novelties[:2])) if len(sorted_novelties) >= 2 else max_novelty

    # High threat count: tail passages with semantic relevance >= median AND positive novelty
    threat_mask = (np.array(tail_scores) >= median_sem) & (novelties > 0.0)
    high_threat_count = float(np.sum(threat_mask))

    return {
        "tail_k_max_semantic_relevance": max_tail_rel,
        "tail_k_top2_mean_relevance": top2_tail_mean,
        "tail_k_relevance_mass": tail_mass,
        "tail_k_boundary_drop": boundary_drop,
        "tail_k_max_novelty": max_novelty,
        "tail_k_mean_pos_novelty": mean_pos_novelty,
        "tail_k_top2_novelty": top2_novelty,
        "tail_k_high_threat_count": high_threat_count,
    }
