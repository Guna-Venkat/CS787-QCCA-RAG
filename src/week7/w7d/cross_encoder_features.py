"""Feature extraction from joint query-passage cross-encoder scores for D0.

Computes:
- Top-k means: ce_top1, ce_top2_mean, ce_top4_mean, ce_top6_mean, ce_top10_mean
- Tail & Boundary metrics: ce_boundary_score_k2, ce_boundary_gap_k2, ce_boundary_score_k4,
    ce_boundary_gap_k4, ce_tail_max_k4, ce_tail_mean_k4, ce_top2_vs_tail_gap, ce_top4_vs_tail_gap
- Specific rank probes: ce_rank5_score, ce_rank6_score, ce_rank8_score
- Rank alignment & displacement: ce_spearman_rho_semantic, ce_spearman_rho_bm25,
    ce_mean_abs_rank_displacement, ce_top4_overlap_semantic
- Merges with existing W7-C feature bank.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.week7.w7d.data import assert_zero_feature_leakage

logger = logging.getLogger(__name__)


def extract_query_cross_encoder_features(
    scores_for_query: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    """Extract summary cross-encoder features for a single query from its passage score records.

    Args:
        scores_for_query: List of passage dicts for a query, sorted by semantic_rank 1..10.
            Each dict has: semantic_rank, bm25_rank, cross_encoder_score, passage_length.
    """
    sorted_passages = sorted(scores_for_query, key=lambda p: int(p.get("semantic_rank", 1)))
    scores = [float(p.get("cross_encoder_score", 0.0)) for p in sorted_passages]
    n = len(scores)

    if n == 0:
        return {f"feat_ce_{k}": 0.0 for k in [
            "top1", "top2_mean", "top4_mean", "top6_mean", "top10_mean",
            "selected_min_k4", "selected_var_k4", "tail_max_k4", "tail_mean_k4",
            "boundary_score_k2", "boundary_gap_k2", "boundary_score_k4", "boundary_gap_k4",
            "top2_vs_tail_gap", "top4_vs_tail_gap", "rank5_score", "rank6_score", "rank8_score",
            "spearman_rho_semantic", "spearman_rho_bm25", "mean_abs_rank_displacement",
            "top4_overlap_semantic"
        ]}

    c_arr = np.array(scores, dtype=float)

    # Top-k means
    ce_top1 = float(np.max(c_arr))
    ce_top2_mean = float(np.mean(c_arr[:min(2, n)]))
    ce_top4_mean = float(np.mean(c_arr[:min(4, n)]))
    ce_top6_mean = float(np.mean(c_arr[:min(6, n)]))
    ce_top10_mean = float(np.mean(c_arr))

    # K4 selected vs tail
    sel_k4 = c_arr[:min(4, n)]
    tail_k4 = c_arr[min(4, n):] if n > 4 else np.array([c_arr[-1]])

    ce_selected_min_k4 = float(np.min(sel_k4))
    ce_selected_var_k4 = float(np.var(sel_k4))
    ce_tail_max_k4 = float(np.max(tail_k4))
    ce_tail_mean_k4 = float(np.mean(tail_k4))
    ce_top4_vs_tail_gap = ce_top4_mean - ce_tail_mean_k4

    # K2 selected vs tail
    tail_k2 = c_arr[min(2, n):] if n > 2 else np.array([c_arr[-1]])
    ce_top2_vs_tail_gap = ce_top2_mean - float(np.mean(tail_k2))

    # Boundaries
    b_score_k2 = float(c_arr[1]) if n >= 2 else float(c_arr[0])
    next_score_k2 = float(c_arr[2]) if n >= 3 else b_score_k2
    b_gap_k2 = b_score_k2 - next_score_k2

    b_score_k4 = float(c_arr[3]) if n >= 4 else float(c_arr[-1])
    next_score_k4 = float(c_arr[4]) if n >= 5 else b_score_k4
    b_gap_k4 = b_score_k4 - next_score_k4

    # Specific rank values
    rank5_score = float(c_arr[4]) if n >= 5 else 0.0
    rank6_score = float(c_arr[5]) if n >= 6 else 0.0
    rank8_score = float(c_arr[7]) if n >= 8 else 0.0

    # Cross-encoder ranking vs semantic ranking
    ce_ranks = np.argsort(-c_arr) + 1  # 1-indexed rank under cross-encoder
    sem_ranks = np.arange(1, n + 1)
    bm25_ranks = np.array([int(p.get("bm25_rank", i + 1)) for i, p in enumerate(sorted_passages)], dtype=float)

    # Correlations
    rho_sem = float(spearmanr(sem_ranks, ce_ranks).statistic) if n > 2 else 0.0
    if np.isnan(rho_sem):
        rho_sem = 0.0

    rho_bm25 = float(spearmanr(bm25_ranks, ce_ranks).statistic) if n > 2 else 0.0
    if np.isnan(rho_bm25):
        rho_bm25 = 0.0

    rank_disp = float(np.mean(np.abs(sem_ranks - ce_ranks)))

    # Top-4 overlap
    top4_ce_set = set(np.argsort(-c_arr)[:4])
    top4_sem_set = set(range(min(4, n)))
    top4_overlap = len(top4_ce_set.intersection(top4_sem_set)) / 4.0

    return {
        "feat_ce_top1": ce_top1,
        "feat_ce_top2_mean": ce_top2_mean,
        "feat_ce_top4_mean": ce_top4_mean,
        "feat_ce_top6_mean": ce_top6_mean,
        "feat_ce_top10_mean": ce_top10_mean,
        "feat_ce_selected_min_k4": ce_selected_min_k4,
        "feat_ce_selected_var_k4": ce_selected_var_k4,
        "feat_ce_tail_max_k4": ce_tail_max_k4,
        "feat_ce_tail_mean_k4": ce_tail_mean_k4,
        "feat_ce_boundary_score_k2": b_score_k2,
        "feat_ce_boundary_gap_k2": b_gap_k2,
        "feat_ce_boundary_score_k4": b_score_k4,
        "feat_ce_boundary_gap_k4": b_gap_k4,
        "feat_ce_top2_vs_tail_gap": ce_top2_vs_tail_gap,
        "feat_ce_top4_vs_tail_gap": ce_top4_vs_tail_gap,
        "feat_ce_rank5_score": rank5_score,
        "feat_ce_rank6_score": rank6_score,
        "feat_ce_rank8_score": rank8_score,
        "feat_ce_spearman_rho_semantic": rho_sem,
        "feat_ce_spearman_rho_bm25": rho_bm25,
        "feat_ce_mean_abs_rank_displacement": rank_disp,
        "feat_ce_top4_overlap_semantic": top4_overlap,
    }


def build_d0_feature_dataframe(
    df_scores_cache: pd.DataFrame,
    df_base_features: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build complete query-level D0 feature DataFrame from cached cross-encoder scores.

    If df_base_features (from W7-C) is provided, joins on question_id.
    """
    df_scores_cache["question_id"] = df_scores_cache["question_id"].astype(str).str.strip()

    records: List[Dict[str, Any]] = []
    for qid, group in df_scores_cache.groupby("question_id"):
        passages = group.to_dict(orient="records")
        feats = extract_query_cross_encoder_features(passages)
        feats["question_id"] = qid
        feats["paper_id"] = group["paper_id"].iloc[0] if "paper_id" in group else ""
        records.append(feats)

    df_ce_feats = pd.DataFrame(records)

    feat_cols = [c for c in df_ce_feats.columns if c.startswith("feat_")]
    assert_zero_feature_leakage(feat_cols)

    if df_base_features is not None:
        df_base = df_base_features.copy()
        df_base["question_id"] = df_base["question_id"].astype(str).str.strip()

        # If df_base is candidate rows (k=2,4,5,6,8), extract k=4 anchor rows
        if "k" in df_base.columns:
            df_base = df_base[df_base["k"] == 4].copy()

        # Avoid column name collisions
        existing_cols = set(df_ce_feats.columns)
        join_cols = ["question_id"] + [c for c in df_base.columns if c not in existing_cols and c.startswith("feat_") or c in ["paper_id"]]
        df_merged = pd.merge(df_ce_feats, df_base[join_cols], on="question_id", how="inner")
        logger.info("Merged D0 features: %d rows, %d total feature columns.", len(df_merged), len(df_merged.columns))
        return df_merged

    logger.info("Built D0 features: %d rows, %d feature columns.", len(df_ce_feats), len(feat_cols))
    return df_ce_feats
