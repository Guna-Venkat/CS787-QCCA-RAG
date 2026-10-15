"""Data loading, split management, and ranking group dataset construction for Week 7-C2.

Provides:
- Strict TEST isolation guard.
- Manifest loading for paper-disjoint TRAIN-FIT and INTERNAL-VAL.
- Query-grouped candidate dataset construction with 5 candidate rows per query.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ALLOWED_BUDGETS: List[int] = [2, 4, 5, 6, 8]


def assert_test_disabled(split_name: str) -> None:
    """Ensure TEST split is strictly inaccessible."""
    if "test" in str(split_name).lower():
        raise RuntimeError("TEST split access is strictly prohibited in Week 7-C2! TEST must remain untouched.")


def load_w7c2_splits(
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


def build_w7c2_ranking_dataset(
    candidate_features_df: pd.DataFrame,
    response_matrix_df: pd.DataFrame,
    budgets: Sequence[int] = (2, 4, 5, 6, 8),
    epsilon_pref: float = 0.01,
    split_name: str = "train",
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Build a query-grouped candidate dataset for ranking models (M2) and regression (M1).

    Ensures:
    - Exactly len(budgets) rows per query, sorted in ascending order of budget k.
    - Each query forms an unbroken block (group size = len(budgets)).
    - Attaches downstream target F1, tokens, and lexicographic relevance grades.

    Returns:
        (df_ranking_candidates, group_sizes_array)
    """
    assert_test_disabled(split_name)

    df_cand = candidate_features_df.copy()
    df_cand["question_id"] = df_cand["question_id"].astype(str).str.strip()
    df_cand["paper_id"] = df_cand["paper_id"].astype(str).str.strip()
    df_cand["k"] = df_cand["k"].astype(int)

    df_mat = response_matrix_df.copy()
    df_mat["question_id"] = df_mat["question_id"].astype(str).str.strip()
    df_mat["paper_id"] = df_mat["paper_id"].astype(str).str.strip()

    # Precompute per-query targets and relevance grades
    from src.week7.w7c2.targets import compute_lexicographic_relevance_grades

    q_targets: Dict[str, Dict[str, Any]] = {}
    for _, row in df_mat.iterrows():
        qid = str(row["question_id"]).strip()
        f1_vals = {k: float(row[f"f1_k{k}"]) for k in budgets}
        tok_vals = {k: float(row[f"tokens_k{k}"]) for k in budgets}
        grades = compute_lexicographic_relevance_grades(f1_vals, tok_vals, epsilon_pref=epsilon_pref)

        max_f1 = max(f1_vals.values())
        admissible_ks = [k for k in sorted(budgets) if f1_vals[k] >= max_f1 - epsilon_pref]
        oracle_k = admissible_ks[0] if admissible_ks else max(budgets)

        q_targets[qid] = {
            "f1_by_k": f1_vals,
            "tokens_by_k": tok_vals,
            "grades_by_k": grades,
            "max_f1": max_f1,
            "oracle_k": oracle_k,
        }

    # Filter candidate features to queries present in response matrix
    valid_qids = set(df_mat["question_id"])
    df_cand_filtered = df_cand[df_cand["question_id"].isin(valid_qids)].copy()

    # Sort strictly by question_id and k to form contiguous 5-row groups
    df_cand_filtered = df_cand_filtered.sort_values(by=["question_id", "k"]).reset_index(drop=True)

    # Attach targets
    target_f1s: List[float] = []
    target_tokens: List[float] = []
    target_grades: List[int] = []
    target_regrets: List[float] = []
    oracle_ks: List[int] = []

    for _, row in df_cand_filtered.iterrows():
        qid = row["question_id"]
        k = row["k"]
        q_info = q_targets[qid]

        f1_k = q_info["f1_by_k"][k]
        tok_k = q_info["tokens_by_k"][k]
        gr_k = q_info["grades_by_k"][k]
        reg_k = max(0.0, q_info["max_f1"] - f1_k)

        target_f1s.append(f1_k)
        target_tokens.append(tok_k)
        target_grades.append(gr_k)
        target_regrets.append(reg_k)
        oracle_ks.append(q_info["oracle_k"])

    df_cand_filtered["target_f1"] = target_f1s
    df_cand_filtered["target_tokens"] = target_tokens
    df_cand_filtered["target_relevance_grade"] = target_grades
    df_cand_filtered["target_regret"] = target_regrets
    df_cand_filtered["target_oracle_k"] = oracle_ks

    # Compute group sizes
    counts = df_cand_filtered.groupby("question_id", sort=False).size().values
    if not all(c == len(budgets) for c in counts):
        raise ValueError(f"Not all queries have exactly {len(budgets)} candidate rows! Found counts: {set(counts)}")

    logger.info("Built ranking dataset for %s: %d rows across %d queries.", split_name, len(df_cand_filtered), len(counts))
    return df_cand_filtered, counts
