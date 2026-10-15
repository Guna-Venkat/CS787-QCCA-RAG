"""Semantic-routing-aware oracle derivation and BM25-vs-semantic oracle shift analytics for Week 7-B.

Implements:
- Semantic Quality Oracle: F1*(q) = max_{k in K} F1_sem(q, k).
- Semantic Epsilon Oracle: k*_eps(q) = min {k in K : F1_sem(q, k) >= F1*(q) - eps} with eps=0.01.
- Transition matrix between BM25-based oracle targets and semantic-based oracle targets.
- Delta k* and Delta Tokens* shift statistics (% lower, % unchanged, % higher, mean, median).
- Recovered Oracle Headroom (ROH) computation with declared static semantic reference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BUDGETS: List[int] = [2, 4, 5, 6, 8]


def compute_semantic_oracle_dataframe(
    df_matrix: pd.DataFrame,
    epsilon: float = 0.01,
    budgets: Sequence[int] = (2, 4, 5, 6, 8),
) -> pd.DataFrame:
    """Derive semantic quality oracle and epsilon-oracle targets for each query in the response matrix.

    Args:
        df_matrix: Wide response surface matrix (must contain f1_k{k} and tokens_k{k} columns).
        epsilon: Allowed quality tolerance below the peak (default 0.01).
        budgets: Budget set K.

    Returns:
        DataFrame with columns:
        question_id, paper_id, best_f1, quality_oracle_k, oracle_k_eps001,
        oracle_f1, oracle_tokens, regret
    """
    assert epsilon == 0.01, f"Invariant violation: epsilon must be fixed at 0.01, got {epsilon}"

    rows = []
    for _, row in df_matrix.iterrows():
        qid = str(row["question_id"]).strip()
        pid = str(row.get("paper_id", "")).strip()

        f1_by_k = {k: float(row[f"f1_k{k}"]) for k in budgets}
        tokens_by_k = {k: int(row[f"tokens_k{k}"]) for k in budgets}

        # Quality oracle: maximum F1 across candidate budgets
        best_f1 = max(f1_by_k.values())
        # Tie-breaker: smallest k
        qual_k = min([k for k in budgets if np.isclose(f1_by_k[k], best_f1, atol=1e-6)])

        # Epsilon oracle: smallest k achieving within epsilon of best_f1
        threshold = best_f1 - epsilon
        eps_admissible_k = [k for k in budgets if f1_by_k[k] >= threshold - 1e-6]
        eps_k = min(eps_admissible_k)

        oracle_f1 = f1_by_k[eps_k]
        oracle_tokens = tokens_by_k[eps_k]
        regret = max(0.0, best_f1 - oracle_f1)

        row_dict: Dict[str, Any] = {
            "question_id": qid,
            "paper_id": pid,
            "best_f1": best_f1,
            "quality_oracle_k": qual_k,
            "oracle_k_eps001": eps_k,
            "oracle_f1": oracle_f1,
            "oracle_tokens": oracle_tokens,
            "regret": regret,
        }

        # Also store per-budget f1 and tokens
        for k in budgets:
            row_dict[f"f1_k{k}"] = f1_by_k[k]
            row_dict[f"tokens_k{k}"] = tokens_by_k[k]
            row_dict[f"eps_good_k{k}"] = int(f1_by_k[k] >= threshold - 1e-6)

        rows.append(row_dict)

    return pd.DataFrame(rows)


def summarize_semantic_oracle(
    df_oracle: pd.DataFrame,
    split_name: str = "DEV",
    static_k8_ref_tokens: Optional[float] = None,
) -> Dict[str, Any]:
    """Generate comprehensive summary statistics for the semantic oracle targets."""
    mean_best_f1 = float(df_oracle["best_f1"].mean())
    mean_eps_f1 = float(df_oracle["oracle_f1"].mean())
    mean_eps_k = float(df_oracle["oracle_k_eps001"].mean())
    mean_tokens = float(df_oracle["oracle_tokens"].mean())
    mean_regret = float(df_oracle["regret"].mean())

    k_counts = df_oracle["oracle_k_eps001"].value_counts().sort_index().to_dict()
    total_q = len(df_oracle)
    k_distribution = {int(k): round(count / total_q * 100.0, 2) for k, count in k_counts.items()}

    ref_tok = static_k8_ref_tokens if static_k8_ref_tokens is not None else float(df_oracle.get("tokens_k8", pd.Series([1750.5])).mean())
    token_reduction_pct = float((ref_tok - mean_tokens) / ref_tok * 100.0) if ref_tok > 0 else 0.0

    return {
        "split": split_name,
        "n_queries": total_q,
        "quality_oracle_mean_f1": mean_best_f1,
        "epsilon_oracle_mean_f1": mean_eps_f1,
        "mean_oracle_k": mean_eps_k,
        "mean_context_tokens": mean_tokens,
        "static_k8_tokens_ref": ref_tok,
        "token_reduction_pct_vs_k8": token_reduction_pct,
        "mean_regret": mean_regret,
        "k_distribution_pct": k_distribution,
    }


def compute_bm25_vs_semantic_oracle_shift(
    df_bm25_supervision: pd.DataFrame,
    df_sem_oracle: pd.DataFrame,
    budgets: Sequence[int] = (2, 4, 5, 6, 8),
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    """Compare BM25-based oracle targets k*_eps,bm25 against semantic-based targets k*_eps,sem.

    Args:
        df_bm25_supervision: DataFrame with question_id and oracle_k_eps001 (from Week 3 / Week 5).
        df_sem_oracle: DataFrame with question_id and oracle_k_eps001 (from Week 7-B semantic oracle).
        budgets: Budget list.

    Returns:
        (df_transition_matrix, shift_summary, df_paired_queries)
    """
    df_b = df_bm25_supervision.copy()
    df_s = df_sem_oracle.copy()

    df_b["question_id"] = df_b["question_id"].astype(str).str.strip()
    df_s["question_id"] = df_s["question_id"].astype(str).str.strip()

    merged = pd.merge(
        df_b[["question_id", "paper_id", "oracle_k_eps001"]].rename(columns={"oracle_k_eps001": "k_bm25_oracle"}),
        df_s[["question_id", "oracle_k_eps001", "oracle_tokens"]].rename(columns={"oracle_k_eps001": "k_sem_oracle", "oracle_tokens": "tokens_sem_oracle"}),
        on="question_id",
    )

    if "tokens_bm25_oracle" not in merged.columns:
        # If tokens for bm25 oracle are available in df_b
        if "oracle_tokens" in df_b.columns:
            merged = pd.merge(
                merged,
                df_b[["question_id", "oracle_tokens"]].rename(columns={"oracle_tokens": "tokens_bm25_oracle"}),
                on="question_id",
            )
        else:
            merged["tokens_bm25_oracle"] = merged["tokens_sem_oracle"]

    # Compute shifts
    merged["delta_k"] = merged["k_sem_oracle"] - merged["k_bm25_oracle"]
    merged["delta_tokens"] = merged["tokens_sem_oracle"] - merged["tokens_bm25_oracle"]

    n = len(merged)
    pct_lower = float(np.mean(merged["delta_k"] < 0) * 100.0)
    pct_unchanged = float(np.mean(merged["delta_k"] == 0) * 100.0)
    pct_higher = float(np.mean(merged["delta_k"] > 0) * 100.0)
    mean_delta_k = float(merged["delta_k"].mean())
    median_delta_k = float(merged["delta_k"].median())
    mean_delta_tokens = float(merged["delta_tokens"].mean())

    # Build 5x5 Transition Matrix
    matrix_counts = pd.crosstab(
        index=merged["k_bm25_oracle"],
        columns=merged["k_sem_oracle"],
    ).reindex(index=budgets, columns=budgets, fill_value=0)

    summary = {
        "n_queries": n,
        "pct_lower_under_semantic": pct_lower,
        "pct_unchanged": pct_unchanged,
        "pct_higher_under_semantic": pct_higher,
        "mean_delta_k": mean_delta_k,
        "median_delta_k": median_delta_k,
        "mean_delta_tokens": mean_delta_tokens,
    }

    return matrix_counts, summary, merged


def compute_recovered_oracle_headroom(
    policy_f1: float,
    reference_f1: float,
    oracle_f1: float,
) -> float:
    """Compute Recovered Oracle Headroom (ROH): (F1_policy - F1_ref) / (F1_oracle - F1_ref)."""
    denom = oracle_f1 - reference_f1
    if abs(denom) < 1e-8:
        return 0.0
    return float((policy_f1 - reference_f1) / denom)
