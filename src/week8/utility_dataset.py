"""W8-C: Counterfactual Passage Utility Dataset Construction.

Constructs training and evaluation datasets mapping:
    (q, y_0, D_i) -> Delta_i, HELP, HURT, 3-way label

Heads:
- Head 1: P(HELP) = 1[Delta_i > +0.01]
- Head 2: P(HURT) = 1[Delta_i < -0.01]
- Head 3: continuous Delta_i (regression target)
- Head 4: 3-way label (0: NEUTRAL, 1: HELP, 2: HURT)

Strict Leakage Invariant:
- Gold answers and F1 metrics are targets only; NEVER present in model input features.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_utility_dataset(
    df_promotions: pd.DataFrame,
    df_drafts: pd.DataFrame,
    df_uncertainty: Optional[pd.DataFrame] = None,
    df_self_consistency: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Merge promotion results, draft answers, uncertainty, and agreement features."""
    # Ensure string question_id for robust merging
    df_prom = df_promotions.copy()
    df_prom["question_id"] = df_prom["question_id"].astype(str)

    df_dr = df_drafts.copy()
    df_dr["question_id"] = df_dr["question_id"].astype(str)
    # Select non-leaking draft fields
    draft_cols = [
        "question_id",
        "draft_answer",
        "context_tokens",
    ]
    df_dr_sub = df_dr[[c for c in draft_cols if c in df_dr.columns]]

    # Merge promotion records with draft
    merged = pd.merge(df_prom, df_dr_sub, on="question_id", how="inner", suffixes=("", "_draft"))

    # Merge uncertainty features if available
    if df_uncertainty is not None:
        df_unc = df_uncertainty.copy()
        df_unc["question_id"] = df_unc["question_id"].astype(str)
        merged = pd.merge(merged, df_unc, on="question_id", how="left")

    # Merge self-consistency features if available
    if df_self_consistency is not None:
        df_sc = df_self_consistency.copy()
        df_sc["question_id"] = df_sc["question_id"].astype(str)
        merged = pd.merge(merged, df_sc, on="question_id", how="left")

    # Construct the 5 multitask targets
    merged["target_continuous_delta"] = merged["delta_f1"].astype(float)
    merged["target_help_binary"] = (merged["delta_f1"] > 0.01).astype(int)
    merged["target_hurt_binary"] = (merged["delta_f1"] < -0.01).astype(int)
    ans_chg = merged["answer_changed"].astype(int) if "answer_changed" in merged.columns else (merged["delta_f1"].abs() > 1e-4).astype(int)
    merged["target_beneficial_change"] = ((ans_chg == 1) & (merged["delta_f1"] > 0.01)).astype(int)
    if "delta_gold_logprob" in merged.columns:
        merged["target_dense_gold_ll"] = merged["delta_gold_logprob"].astype(float)
    elif "delta_ll" in merged.columns:
        merged["target_dense_gold_ll"] = merged["delta_ll"].astype(float)
    else:
        merged["target_dense_gold_ll"] = merged["delta_f1"].astype(float)

    # 3-way label: 0=NEUTRAL, 1=HELP, 2=HURT
    def map_3way(val: float) -> int:
        if val > 0.01:
            return 1
        elif val < -0.01:
            return 2
        else:
            return 0

    merged["target_3way_label"] = merged["delta_f1"].apply(map_3way)

    logger.info("Constructed utility dataset with %d rows and 5 multitask targets.", len(merged))
    return merged


def get_non_leaking_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return strictly verified non-leaking numeric feature column names."""
    forbidden = {
        "gold_answers",
        "gold_answers_reformatted",
        "base_f1",
        "promoted_f1",
        "delta_f1",
        "target_continuous_delta",
        "target_help_binary",
        "target_hurt_binary",
        "target_beneficial_change",
        "target_3way_label",
        "target_dense_gold_ll",
        "delta_gold_logprob",
        "base_gold_logprob",
        "promoted_gold_logprob",
        "utility_label",
        "evidence_utilization_type",
        "promoted_prediction",
        "base_answer_complete",
        "question_id",
        "paper_id",
        "question",
        "candidate_text",
        "draft_answer",
        "sampled_answers",
        "majority_answer",
    }
    numeric_cols = [
        c for c in df.columns
        if c not in forbidden and np.issubdtype(df[c].dtype, np.number)
    ]
    return sorted(numeric_cols)


# Aliases
assemble_multitask_utility_dataset = build_utility_dataset


def extract_utility_targets(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract standard non-leaking utility targets from a promotion record."""
    delta = float(record.get("delta_f1", 0.0))
    is_help = 1.0 if delta > 0.01 else 0.0
    is_hurt = 1.0 if delta < -0.01 else 0.0
    label_3way = 1 if delta > 0.01 else (2 if delta < -0.01 else 0)
    delta_ll = float(record.get("delta_gold_logprob", record.get("delta_ll", 0.0)))
    return {
        "delta_f1": delta,
        "is_help": is_help,
        "is_hurt": is_hurt,
        "label_3way": label_3way,
        "delta_gold_logprob": delta_ll,
    }

