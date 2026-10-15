"""Data loading, dataset building, and feature preparation for Week 7-C3 Anchor-and-Deviate.

Provides:
- Strict TEST isolation guard.
- Manifest loading for paper-disjoint splits.
- Feature extraction for Downshift (k=2 vs k=4) and Upshift (k=4 tail / k>4).
- Strict zero-leakage assertions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ALLOWED_BUDGETS: List[int] = [2, 4, 5, 6, 8]
ANCHOR_K: int = 4


def assert_test_disabled(split_name: str) -> None:
    """Ensure TEST split is strictly inaccessible."""
    if "test" in str(split_name).lower():
        raise RuntimeError("TEST split access is strictly prohibited in Week 7-C3! TEST must remain untouched.")


FORBIDDEN_FEATURE_SUBSTRINGS: Set[str] = {
    "f1",
    "oracle",
    "regret",
    "answer",
    "gold",
    "target",
    "ground_truth",
}


def assert_zero_feature_leakage(feature_cols: Sequence[str]) -> None:
    """Ensure no downstream target columns leak into feature vectors."""
    for col in feature_cols:
        col_lower = str(col).lower()
        for f in FORBIDDEN_FEATURE_SUBSTRINGS:
            if f in col_lower:
                raise ValueError(f"LEAKAGE DETECTED: Column '{col}' contains forbidden target keyword '{f}'!")


def load_w7c3_splits(
    train_fit_manifest_path: Union[str, Path] = "results/week5/content_allocator/w5b/splits/train_fit_manifest.csv",
    internal_val_manifest_path: Union[str, Path] = "results/week5/content_allocator/w5b/splits/internal_val_manifest.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load canonical paper-disjoint TRAIN-FIT and INTERNAL-VAL split manifests."""
    df_fit = pd.read_csv(train_fit_manifest_path)
    df_val = pd.read_csv(internal_val_manifest_path)

    df_fit["question_id"] = df_fit["question_id"].astype(str).str.strip()
    df_val["question_id"] = df_val["question_id"].astype(str).str.strip()
    df_fit["paper_id"] = df_fit["paper_id"].astype(str).str.strip()
    df_val["paper_id"] = df_val["paper_id"].astype(str).str.strip()

    fit_qids = set(df_fit["question_id"])
    val_qids = set(df_val["question_id"])
    overlap_qids = fit_qids.intersection(val_qids)
    if overlap_qids:
        raise ValueError(f"DATA LEAKAGE: {len(overlap_qids)} questions exist in both TRAIN-FIT and INTERNAL-VAL!")

    fit_papers = set(df_fit["paper_id"])
    val_papers = set(df_val["paper_id"])
    overlap_papers = fit_papers.intersection(val_papers)
    if overlap_papers:
        raise ValueError(f"DATA LEAKAGE: {len(overlap_papers)} papers exist in both TRAIN-FIT and INTERNAL-VAL!")

    logger.info(
        "Validated paper-disjoint split: TRAIN-FIT=%d questions (%d papers), INTERNAL-VAL=%d questions (%d papers).",
        len(df_fit), len(fit_papers), len(df_val), len(val_papers),
    )
    return df_fit, df_val


def build_w7c3_query_dataset(
    candidate_features_df: pd.DataFrame,
    response_matrix_df: pd.DataFrame,
    budgets: Sequence[int] = (2, 4, 5, 6, 8),
    anchor_k: int = 4,
    epsilon: float = 0.01,
    split_name: str = "train",
) -> pd.DataFrame:
    """Build a unified per-query dataset combining anchor/candidate features and supervisory targets.

    For each unique question_id:
    - Anchors at k=4.
    - Extracts downshift features: query features + k=2 candidate features + diffs (k4 - k2).
    - Extracts upshift features: query features + k=4 candidate features (tail features outside top 4).
    - Computes ground-truth targets from response matrix:
        - target_y_down: 1[F(q, 2) >= F*(q) - epsilon]
        - target_y_up: 1[max_{k in {5,6,8}} F(q, k) - F(q, 4) > epsilon]
        - target_delta_5: F(q, 5) - F(q, 4)
        - target_delta_6: F(q, 6) - F(q, 4)
        - target_delta_8: F(q, 8) - F(q, 4)
        - target_y_4to6: 1[F(q, 6) - F(q, 4) > epsilon]
        - oracle_k: minimum-cost epsilon-admissible budget
        - f1_k2, f1_k4, f1_k5, f1_k6, f1_k8
        - tokens_k2, tokens_k4, tokens_k5, tokens_k6, tokens_k8
    """
    assert_test_disabled(split_name)

    df_cand = candidate_features_df.copy()
    df_cand["question_id"] = df_cand["question_id"].astype(str).str.strip()
    df_cand["paper_id"] = df_cand["paper_id"].astype(str).str.strip()
    df_cand["k"] = df_cand["k"].astype(int)

    df_mat = response_matrix_df.copy()
    df_mat["question_id"] = df_mat["question_id"].astype(str).str.strip()
    df_mat["paper_id"] = df_mat["paper_id"].astype(str).str.strip()

    valid_qids = set(df_mat["question_id"])
    df_cand = df_cand[df_cand["question_id"].isin(valid_qids)].copy()

    # Split candidates by budget k=2 and k=4
    df_k2 = df_cand[df_cand["k"] == 2].copy().set_index("question_id")
    df_k4 = df_cand[df_cand["k"] == anchor_k].copy().set_index("question_id")

    # Metadata columns to exclude from numeric feature sets
    meta_cols = {"split", "question_id", "paper_id", "k"}

    # Base candidate feature columns
    base_feat_cols = [c for c in df_cand.columns if c not in meta_cols]
    assert_zero_feature_leakage(base_feat_cols)

    # Build per-query rows
    records: List[Dict[str, Any]] = []

    for _, row_mat in df_mat.iterrows():
        qid = str(row_mat["question_id"]).strip()
        if qid not in df_k2.index or qid not in df_k4.index:
            continue

        paper_id = str(row_mat["paper_id"]).strip()

        # Actual response surface values
        f1_vals = {k: float(row_mat[f"f1_k{k}"]) for k in budgets}
        tok_vals = {k: float(row_mat[f"tokens_k{k}"]) for k in budgets}

        max_f1 = max(f1_vals.values())
        admissible_ks = [k for k in sorted(budgets) if f1_vals[k] >= max_f1 - epsilon]
        oracle_k = min(admissible_ks) if admissible_ks else max(budgets)

        # Target A: Safe Downshift
        y_down = 1 if f1_vals[2] >= max_f1 - epsilon else 0

        # Target B: Selective Upshift
        f_up_star = max(f1_vals[5], f1_vals[6], f1_vals[8])
        gain_up = f_up_star - f1_vals[anchor_k]
        y_up = 1 if gain_up > epsilon else 0

        # Target C: Marginal Gains relative to k=4
        delta_5 = f1_vals[5] - f1_vals[anchor_k]
        delta_6 = f1_vals[6] - f1_vals[anchor_k]
        delta_8 = f1_vals[8] - f1_vals[anchor_k]

        # Optional Target: 4->6 Gain
        y_4to6 = 1 if (f1_vals[6] - f1_vals[anchor_k]) > epsilon else 0

        # Record dict
        rec: Dict[str, Any] = {
            "question_id": qid,
            "paper_id": paper_id,
            "target_y_down": y_down,
            "target_y_up": y_up,
            "target_delta_5": delta_5,
            "target_delta_6": delta_6,
            "target_delta_8": delta_8,
            "target_y_4to6": y_4to6,
            "target_oracle_k": oracle_k,
            "target_max_f1": max_f1,
            "target_gain_up": gain_up,
        }

        # Store response surface metrics for lookup
        for k in budgets:
            rec[f"f1_k{k}"] = f1_vals[k]
            rec[f"tokens_k{k}"] = tok_vals[k]

        # Features from k=4 (Anchor representation)
        row_k4 = df_k4.loc[qid]
        for col in base_feat_cols:
            rec[f"feat_k4_{col}"] = row_k4[col]

        # Features from k=2 (Downshift candidate representation)
        row_k2 = df_k2.loc[qid]
        for col in base_feat_cols:
            rec[f"feat_k2_{col}"] = row_k2[col]

        # Explicit differential features between k4 and k2
        if "cost_k_tokens" in row_k4 and "cost_k_tokens" in row_k2:
            rec["feat_diff_tokens_k4_minus_k2"] = float(row_k4["cost_k_tokens"]) - float(row_k2["cost_k_tokens"])
        if "cov_k_mean" in row_k4 and "cov_k_mean" in row_k2:
            rec["feat_diff_cov_k4_minus_k2"] = float(row_k4["cov_k_mean"]) - float(row_k2["cov_k_mean"])
        if "sem_k_selected_mean" in row_k4 and "sem_k_selected_mean" in row_k2:
            rec["feat_diff_sem_k4_minus_k2"] = float(row_k4["sem_k_selected_mean"]) - float(row_k2["sem_k_selected_mean"])

        records.append(rec)

    df_out = pd.DataFrame(records)
    logger.info("Built unified W7-C3 dataset for %s: %d queries across %d papers.", split_name, len(df_out), df_out["paper_id"].nunique())
    return df_out


def get_feature_subsets(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """Partition unified feature columns into dedicated subsets for Downshift, Upshift, and Delta models.

    Returns:
        (downshift_features, upshift_features, delta_features)
    """
    all_feat_cols = [c for c in df.columns if c.startswith("feat_")]
    assert_zero_feature_leakage(all_feat_cols)

    # Downshift features: emphasize top2 features (feat_k2_*), query geometry, coverage differences
    downshift_features = [
        c for c in all_feat_cols
        if c.startswith("feat_k2_")
        or "diff_" in c
        or "feat_k4_q_" in c
        or "feat_k4_sem_top" in c
        or "feat_k4_bm25_top" in c
        or "feat_k4_cross_" in c
    ]

    # Upshift features: emphasize top4 anchor tail, boundary 4->5, 4->6, tail novelty, query complexity
    upshift_features = [
        c for c in all_feat_cols
        if c.startswith("feat_k4_")
    ]

    # Delta features: reuse upshift representation (information about tail passages ranks 5..10)
    delta_features = list(upshift_features)

    return downshift_features, upshift_features, delta_features
