"""Hybrid Lexical-Semantic Passage Fidelity Routing for Week 6-B.

Implements:
- Per-query min-max and z-score score normalization.
- R3: Fixed score fusion:
    h_i(alpha) = alpha * b_tilde_i + (1 - alpha) * s_tilde_i
- R4: Reciprocal Rank Fusion (RRF):
    RRF_i = 1 / (c + r_bm25(i)) + 1 / (c + r_sem(i))
- R5: Rank-Sum fusion:
    rank_sum_i = -r_bm25(i) - r_sem(i)
- Stable tie-breaking preserving retrieval prefix when alpha=1.0.
- Routing invariant validation.
- INTERNAL-VAL alpha selection rule.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch

from src.week6.routing import validate_routing_assignment

logger = logging.getLogger(__name__)


def normalize_scores_minmax(
    scores: Union[np.ndarray, torch.Tensor, Sequence[float]],
    eps: float = 1e-8,
) -> np.ndarray:
    """Per-query min-max score normalization into [0.0, 1.0].

    Formula:
        tilde_s_i = (s_i - min_j s_j) / (max_j s_j - min_j s_j + eps)

    Safe handling for constant-score vectors:
        If max_j s_j - min_j s_j < eps, returns all 1.0.
    """
    if isinstance(scores, torch.Tensor):
        s_arr = scores.detach().cpu().numpy().astype(np.float32)
    else:
        s_arr = np.asarray(scores, dtype=np.float32)

    min_val = float(np.min(s_arr))
    max_val = float(np.max(s_arr))
    spread = max_val - min_val

    if spread < eps:
        # Constant score vector: assign uniform 1.0
        return np.ones_like(s_arr, dtype=np.float32)

    normalized = (s_arr - min_val) / (spread + eps)
    # Clip to guarantee numerical safety in [0.0, 1.0]
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def normalize_scores_zscore(
    scores: Union[np.ndarray, torch.Tensor, Sequence[float]],
    eps: float = 1e-8,
) -> np.ndarray:
    """Per-query z-score score normalization (diagnostic).

    Formula:
        tilde_s_i = (s_i - mean(s)) / (std(s) + eps)

    Safe handling for constant-score vectors:
        If std(s) < eps, returns all 0.0.
    """
    if isinstance(scores, torch.Tensor):
        s_arr = scores.detach().cpu().numpy().astype(np.float32)
    else:
        s_arr = np.asarray(scores, dtype=np.float32)

    mean_val = float(np.mean(s_arr))
    std_val = float(np.std(s_arr))

    if std_val < eps:
        return np.zeros_like(s_arr, dtype=np.float32)

    return ((s_arr - mean_val) / (std_val + eps)).astype(np.float32)


def compute_ranks_from_scores(
    scores: Union[np.ndarray, torch.Tensor, Sequence[float]],
    reverse: bool = True,
) -> np.ndarray:
    """Compute 1-indexed rank positions (1..N) from scores with stable tie-breaking.

    Args:
        scores: 1D array of scores.
        reverse: If True, higher score gets rank 1 (descending). If False, lower gets rank 1.

    Returns:
        ranks: 1-indexed integer array where ranks[i] is the rank of passage i.
    """
    if isinstance(scores, torch.Tensor):
        s_arr = scores.detach().cpu().numpy()
    else:
        s_arr = np.asarray(scores)

    n = len(s_arr)
    # Stable sort by score; on ties, preserve original index order
    order = sorted(range(n), key=lambda idx: -float(s_arr[idx]) if reverse else float(s_arr[idx]))
    ranks = np.empty(n, dtype=np.int32)
    for rank_pos, idx in enumerate(order):
        ranks[idx] = rank_pos + 1
    return ranks


def fuse_bm25_semantic_scores(
    bm25_scores: Union[np.ndarray, Sequence[float]],
    sem_scores: Union[np.ndarray, Sequence[float]],
    alpha: float,
    norm_method: str = "min_max",
    eps: float = 1e-8,
) -> np.ndarray:
    """R3 Score Fusion:
        h_i(alpha) = alpha * tilde_b_i + (1 - alpha) * tilde_s_i

    Args:
        bm25_scores: Raw BM25 scores for the 10 passages.
        sem_scores: Semantic similarity scores for the 10 passages.
        alpha: Interpolation weight in [0.0, 1.0].
        norm_method: "min_max" (default) or "z_score".
        eps: Small constant for numerical stability.

    Returns:
        fused_scores: Array of shape (len(bm25_scores),) containing fused scores.
    """
    assert 0.0 <= alpha <= 1.0, f"Alpha must be in [0.0, 1.0], got {alpha}"

    if norm_method == "min_max":
        b_norm = normalize_scores_minmax(bm25_scores, eps=eps)
        s_norm = normalize_scores_minmax(sem_scores, eps=eps)
    elif norm_method == "z_score":
        b_norm = normalize_scores_zscore(bm25_scores, eps=eps)
        s_norm = normalize_scores_zscore(sem_scores, eps=eps)
    else:
        raise ValueError(f"Unknown normalization method: {norm_method}")

    fused = alpha * b_norm + (1.0 - alpha) * s_norm
    return fused.astype(np.float32)


def route_r3_fixed_fusion(
    bm25_scores: Union[np.ndarray, Sequence[float]],
    sem_scores: Union[np.ndarray, Sequence[float]],
    alpha: float,
    k: int,
    norm_method: str = "min_max",
    eps: float = 1e-8,
) -> List[int]:
    """R3: Select top-k passages by fixed score fusion h_i(alpha).

    Presentation order preserves original BM25 retrieval rank order!
    Uses stable descending sort to break score ties by original BM25 index.

    Guarantees:
    - alpha = 1.0 reproduces R0 BM25 prefix if BM25 scores are descending.
    - alpha = 0.0 reproduces R1 semantic top-k.
    - Exactly k passages are selected.
    """
    n_passages = len(bm25_scores)
    assert 0 <= k <= n_passages, f"Budget k={k} must be in [0, {n_passages}]"
    if k == 0:
        return []
    if k == n_passages:
        return list(range(n_passages))

    fused = fuse_bm25_semantic_scores(
        bm25_scores=bm25_scores,
        sem_scores=sem_scores,
        alpha=alpha,
        norm_method=norm_method,
        eps=eps,
    )

    # Stable descending sort: higher score first; ties preserve smaller index
    topk_desc = sorted(range(n_passages), key=lambda idx: -float(fused[idx]))[:k]
    # Presentation order preserves original BM25 rank order
    selected_indices = sorted(int(idx) for idx in topk_desc)

    validate_routing_assignment(selected_indices, k=k, total_passages=n_passages)
    return selected_indices


def compute_rrf_scores(
    bm25_ranks: Union[np.ndarray, Sequence[int]],
    sem_ranks: Union[np.ndarray, Sequence[int]],
    c: float = 60.0,
) -> np.ndarray:
    """R4: Compute Reciprocal Rank Fusion (RRF) scores:
        RRF_i = 1 / (c + r_bm25(i)) + 1 / (c + r_sem(i))

    Args:
        bm25_ranks: 1-indexed rank positions for BM25 (e.g. 1..10).
        sem_ranks: 1-indexed rank positions for semantic ranking (1..10).
        c: Fixed RRF constant (standard default: 60.0).

    Returns:
        rrf_scores: Array of shape (len(bm25_ranks),).
    """
    r_b = np.asarray(bm25_ranks, dtype=np.float32)
    r_s = np.asarray(sem_ranks, dtype=np.float32)
    assert np.all(r_b >= 1), "BM25 ranks must be 1-indexed (>= 1)"
    assert np.all(r_s >= 1), "Semantic ranks must be 1-indexed (>= 1)"

    rrf = (1.0 / (c + r_b)) + (1.0 / (c + r_s))
    return rrf.astype(np.float32)


def route_r4_rrf(
    bm25_ranks: Union[np.ndarray, Sequence[int]],
    sem_ranks: Union[np.ndarray, Sequence[int]],
    k: int,
    c: float = 60.0,
) -> List[int]:
    """R4: Select top-k passages by Reciprocal Rank Fusion (RRF).

    Presentation order preserves original BM25 retrieval rank order!
    """
    n_passages = len(bm25_ranks)
    assert 0 <= k <= n_passages, f"Budget k={k} must be in [0, {n_passages}]"
    if k == 0:
        return []
    if k == n_passages:
        return list(range(n_passages))

    rrf = compute_rrf_scores(bm25_ranks, sem_ranks, c=c)

    # Stable descending sort: higher RRF first; ties preserve smaller index
    topk_desc = sorted(range(n_passages), key=lambda idx: -float(rrf[idx]))[:k]
    selected_indices = sorted(int(idx) for idx in topk_desc)

    validate_routing_assignment(selected_indices, k=k, total_passages=n_passages)
    return selected_indices


def compute_rank_sum_scores(
    bm25_ranks: Union[np.ndarray, Sequence[int]],
    sem_ranks: Union[np.ndarray, Sequence[int]],
) -> np.ndarray:
    """R5: Compute Rank-Sum fusion scores:
        score_i = -r_bm25(i) - r_sem(i)
    """
    r_b = np.asarray(bm25_ranks, dtype=np.float32)
    r_s = np.asarray(sem_ranks, dtype=np.float32)
    return (-r_b - r_s).astype(np.float32)


def route_r5_rank_sum(
    bm25_ranks: Union[np.ndarray, Sequence[int]],
    sem_ranks: Union[np.ndarray, Sequence[int]],
    k: int,
) -> List[int]:
    """R5: Select top-k passages by Rank-Sum score.

    Presentation order preserves original BM25 retrieval rank order!
    """
    n_passages = len(bm25_ranks)
    assert 0 <= k <= n_passages, f"Budget k={k} must be in [0, {n_passages}]"
    if k == 0:
        return []
    if k == n_passages:
        return list(range(n_passages))

    scores = compute_rank_sum_scores(bm25_ranks, sem_ranks)
    topk_desc = sorted(range(n_passages), key=lambda idx: -float(scores[idx]))[:k]
    selected_indices = sorted(int(idx) for idx in topk_desc)

    validate_routing_assignment(selected_indices, k=k, total_passages=n_passages)
    return selected_indices


def select_best_alpha_from_internal_val(
    df_val_results: pd.DataFrame,
    budget: int,
    f1_tolerance: float = 0.002,
) -> float:
    """Select optimal alpha for a fixed budget k using INTERNAL-VAL downstream metrics ONLY.

    Frozen Selection Rule:
    1. Highest INTERNAL-VAL answer F1.
    2. If F1 differences <= f1_tolerance (0.002), choose candidate with lower mean context tokens.
    3. If still tied, choose alpha closer to simpler endpoint (0.0 or 1.0) as deterministic tie-break.

    Args:
        df_val_results: DataFrame containing columns:
            ['budget' or 'k', 'alpha', 'answer_f1_mean', 'context_tokens_mean']
        budget: High-fidelity budget k (e.g. 2, 4, 6).
        f1_tolerance: Maximum F1 difference to trigger context-token tie breaking (default: 0.002).

    Returns:
        best_alpha: Chosen alpha in [0.0, 1.0].
    """
    k_col = "budget" if "budget" in df_val_results.columns else "k"
    df_k = df_val_results[df_val_results[k_col] == budget].copy()

    if df_k.empty:
        raise ValueError(f"No results found for budget k={budget} in df_val_results")

    max_f1 = float(df_k["answer_f1_mean"].max())

    # Filter to candidates within f1_tolerance of max_f1
    candidates = df_k[df_k["answer_f1_mean"] >= max_f1 - f1_tolerance].copy()

    if len(candidates) == 1:
        return float(candidates["alpha"].iloc[0])

    # Tie-break 1: Lowest context tokens
    min_tokens = float(candidates["context_tokens_mean"].min())
    token_candidates = candidates[candidates["context_tokens_mean"] <= min_tokens + 1e-4].copy()

    if len(token_candidates) == 1:
        return float(token_candidates["alpha"].iloc[0])

    # Tie-break 2: Distance to simpler endpoint (0.0 or 1.0)
    def dist_to_endpoint(a: float) -> float:
        return min(abs(float(a) - 0.0), abs(float(a) - 1.0))

    token_candidates["endpoint_dist"] = token_candidates["alpha"].apply(dist_to_endpoint)
    token_candidates = token_candidates.sort_values(by=["endpoint_dist", "alpha"])

    best_alpha = float(token_candidates["alpha"].iloc[0])
    return best_alpha
