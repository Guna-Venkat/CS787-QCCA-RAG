"""Routing policies, invariants, and change analytics for Week 7-A.

Implements:
- BM25-prefix routing under dynamic budget k(q).
- Semantic top-k routing under dynamic budget k(q) (preserving original BM25 presentation order).
- Routing change metrics: Jaccard overlap, replaced passages count, mean/max BM25 rank.
- Invariant assertion checks.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from src.week6.routing import (
    route_r0_bm25_prefix,
    route_r1_semantic_topk,
    validate_routing_assignment,
)
from src.week6.disagreement import compute_jaccard_overlap

logger = logging.getLogger(__name__)


def construct_w7a_routing_assignments(
    predicted_k_list: Sequence[int],
    semantic_scores_arr: np.ndarray,
    n_passages: int = 10,
) -> Tuple[List[List[int]], List[List[int]]]:
    """Construct paired routing assignments (BM25 vs Semantic) under identical predicted budgets.

    Args:
        predicted_k_list: Length N sequence of budgets k(q) in {2, 4, 5, 6, 8}.
        semantic_scores_arr: Shape (N, n_passages) array of query-passage relevance scores.
        n_passages: Total retrieved passages per query (10).

    Returns:
        (bm25_assignments, sem_assignments):
            bm25_assignments: list of length-k 0-indexed indices [0, 1, ..., k-1].
            sem_assignments: list of length-k 0-indexed indices chosen by semantic score,
                             sorted in ascending order to preserve original BM25 presentation.
    """
    n_queries = len(predicted_k_list)
    assert semantic_scores_arr.shape[0] == n_queries, "Mismatch between queries count and semantic scores"

    bm25_assignments: List[List[int]] = []
    sem_assignments: List[List[int]] = []

    for i in range(n_queries):
        k = int(predicted_k_list[i])
        s_scores = semantic_scores_arr[i]

        # 1. BM25 Prefix Assignment
        sel_bm25 = route_r0_bm25_prefix(k=k, n_passages=n_passages)
        validate_routing_assignment(sel_bm25, k=k, total_passages=n_passages)

        # 2. Semantic Top-k Assignment (preserves original BM25 presentation order)
        sel_sem = route_r1_semantic_topk(s_scores, k=k)
        validate_routing_assignment(sel_sem, k=k, total_passages=n_passages)

        # Invariant: exactly the same k chosen
        assert len(sel_bm25) == len(sel_sem) == k

        bm25_assignments.append(sel_bm25)
        sem_assignments.append(sel_sem)

    return bm25_assignments, sem_assignments


def compute_w7a_routing_change_records(
    question_ids: Sequence[str],
    paper_ids: Sequence[str],
    predicted_k_list: Sequence[int],
    bm25_assignments: Sequence[Sequence[int]],
    sem_assignments: Sequence[Sequence[int]],
) -> pd.DataFrame:
    """Compute per-query routing change statistics between BM25 and Semantic routing."""
    rows = []
    n_queries = len(question_ids)

    for i in range(n_queries):
        qid = str(question_ids[i])
        pid = str(paper_ids[i])
        k = int(predicted_k_list[i])
        s_bm25 = list(bm25_assignments[i])
        s_sem = list(sem_assignments[i])

        jaccard = compute_jaccard_overlap(s_bm25, s_sem)
        overlap_count = len(set(s_bm25) & set(s_sem))
        n_changed = k - overlap_count

        # 1-indexed original BM25 ranks of passages selected by semantic routing
        sem_ranks_1based = [idx + 1 for idx in s_sem]
        mean_bm25_rank = float(np.mean(sem_ranks_1based)) if sem_ranks_1based else 0.0
        max_bm25_rank = int(np.max(sem_ranks_1based)) if sem_ranks_1based else 0
        has_below_k = any(r > k for r in sem_ranks_1based)

        rows.append({
            "question_id": qid,
            "paper_id": pid,
            "predicted_k": k,
            "jaccard_overlap": jaccard,
            "n_changed_passages": n_changed,
            "pct_changed_passages": (n_changed / k) * 100.0 if k > 0 else 0.0,
            "mean_selected_bm25_rank": mean_bm25_rank,
            "max_selected_bm25_rank": max_bm25_rank,
            "has_passage_below_predicted_k": has_below_k,
            "selected_bm25_indices": str([x + 1 for x in s_bm25]),
            "selected_semantic_indices": str([x + 1 for x in s_sem]),
        })

    return pd.DataFrame(rows)
