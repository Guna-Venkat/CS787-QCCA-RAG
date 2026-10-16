"""Comprehensive Evaluation, Safety Metrics, and Stratification for Week 8 DAFiR.

Calculates:
1. Core QA metrics: Token F1, Exact Match, ROUGE-L.
2. Safety Metrics:
   - HELP precision: Fraction of promoted passages with Delta_i > 0.01.
   - HURT rate: Fraction of promoted passages with Delta_i < -0.01.
   - Oracle recall: Fraction where chosen candidate matches Oracle-1 chosen candidate.
   - Near-oracle recall: Fraction where chosen candidate Delta_i is within 0.01 of Oracle-1 max Delta.
   - Unnecessary promotion rate: Promoted when max_i(Delta_i) <= 0.01.
3. Promotion-Rank Distribution: Frequency of promotion across ranks 5..10 vs Oracle-1.
4. Stratified Query-Type Analysis: Extractive vs Abstractive vs Yes/No, query length, confidence.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_safety_and_controller_metrics(
    controller_decisions: Sequence[Dict[str, Any]],
    oracle_records_by_qid: Dict[str, Dict[str, Any]],
    epsilon_help: float = 0.01,
    epsilon_hurt: float = -0.01,
) -> Dict[str, float]:
    """Compute controller safety metrics comparing decisions against oracle promotion ground truth."""
    total_queries = len(controller_decisions)
    if total_queries == 0:
        return {}

    promoted_count = 0
    help_count = 0
    hurt_count = 0
    oracle_match_count = 0
    near_oracle_count = 0
    unnecessary_prom_count = 0

    for dec in controller_decisions:
        qid = str(dec["question_id"]).strip()
        action = dec.get("action", "NO_PROMOTION")
        promoted_idx = dec.get("chosen_candidate_idx")

        oracle_info = oracle_records_by_qid.get(qid, {})
        oracle_best_idx = oracle_info.get("best_candidate_idx")
        oracle_max_delta = oracle_info.get("max_delta", 0.0)
        candidate_deltas = oracle_info.get("candidate_deltas", {})

        if action == "PROMOTE" and promoted_idx is not None:
            promoted_count += 1
            actual_delta = candidate_deltas.get(promoted_idx, 0.0)

            if actual_delta > epsilon_help:
                help_count += 1
            elif actual_delta < epsilon_hurt:
                hurt_count += 1

            if promoted_idx == oracle_best_idx:
                oracle_match_count += 1

            if actual_delta >= (oracle_max_delta - epsilon_help):
                near_oracle_count += 1

            if oracle_max_delta <= epsilon_help:
                unnecessary_prom_count += 1

    help_precision = help_count / max(promoted_count, 1)
    hurt_rate = hurt_count / max(promoted_count, 1)
    oracle_recall = oracle_match_count / max(total_queries, 1)
    near_oracle_recall = near_oracle_count / max(total_queries, 1)
    unnecessary_rate = unnecessary_prom_count / max(promoted_count, 1)
    promotion_rate = promoted_count / max(total_queries, 1)

    return {
        "total_queries": float(total_queries),
        "promoted_queries": float(promoted_count),
        "promotion_rate": float(promotion_rate),
        "help_precision": float(help_precision),
        "hurt_rate": float(hurt_rate),
        "oracle_recall": float(oracle_recall),
        "near_oracle_recall": float(near_oracle_recall),
        "unnecessary_promotion_rate": float(unnecessary_rate),
    }


def analyze_promotion_rank_distribution(
    controller_promotions: Sequence[int],
    oracle_best_ranks: Sequence[int],
    candidate_ranks: Sequence[int] = (5, 6, 7, 8, 9, 10),
) -> pd.DataFrame:
    """Compare rank selection distribution of controller vs oracle."""
    c_counts = {r: 0 for r in candidate_ranks}
    o_counts = {r: 0 for r in candidate_ranks}

    for r in controller_promotions:
        if r in c_counts:
            c_counts[r] += 1

    for r in oracle_best_ranks:
        if r in o_counts:
            o_counts[r] += 1

    total_c = max(len(controller_promotions), 1)
    total_o = max(len(oracle_best_ranks), 1)

    rows = []
    for r in candidate_ranks:
        rows.append({
            "semantic_rank": r,
            "controller_count": c_counts[r],
            "controller_pct": c_counts[r] / total_c,
            "oracle_count": o_counts[r],
            "oracle_pct": o_counts[r] / total_o,
        })

    return pd.DataFrame(rows)


def stratify_by_query_type(
    results_df: pd.DataFrame,
    answer_types: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Breakdown evaluation metrics by QASPER answer type (extractive, abstractive, yes/no)."""
    df = results_df.copy()
    if answer_types is not None:
        df["answer_type"] = df["question_id"].astype(str).map(answer_types).fillna("unknown")
    elif "answer_type" not in df.columns:
        df["answer_type"] = "all"

    grouped = df.groupby("answer_type").agg(
        num_queries=("token_f1", "count"),
        mean_token_f1=("token_f1", "mean"),
        mean_exact_match=("exact_match", "mean"),
        mean_final_tokens=("final_context_tokens", "mean"),
        mean_cum_tokens=("cumulative_prompt_tokens", "mean"),
    ).reset_index()

    return grouped
