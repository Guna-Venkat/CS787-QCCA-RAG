"""W8-C: Counterfactual Passage Utility Dataset Construction and Leakage Auditing.

Constructs training and evaluation datasets mapping:
    (q, y_0, D_i) -> Delta_i, Regret, HELP, HURT, 3-way label, Masked LL

Heads and Targets:
- Head 1: P(HELP) = 1[Delta_i > +0.01]
- Head 2: P(HURT) = 1[Delta_i < -0.01]
- Head 3: continuous Delta_i (regression target)
- Head 4: oracle-relative regret = Delta_q^* - Delta_i
- Head 5: 3-way label (0: NEUTRAL, 1: HELP, 2: HURT)
- Head 6: masked continuous gold log-likelihood change (Delta LL) with mask_ll

Strict Leakage Invariant:
- Gold answers, F1 metrics, oracle metrics, and target labels are targets only;
  NEVER present in model input features. Hard assertions strictly enforced.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

logger = logging.getLogger(__name__)

# Absolute forbidden column names (targets, leakages, or text identifiers)
FORBIDDEN_EXACT_FEATURE_NAMES: Set[str] = {
    "gold_answers",
    "gold_answers_reformatted",
    "base_f1",
    "promoted_f1",
    "delta_f1",
    "delta_f1_g0",
    "delta_f1_g1",
    "target_continuous_delta",
    "target_help_binary",
    "target_hurt_binary",
    "target_beneficial_change",
    "target_3way_label",
    "target_dense_gold_ll",
    "target_mask_ll",
    "target_regret",
    "target_is_oracle_best",
    "target_is_near_oracle",
    "oracle_best_idx",
    "oracle_max_delta",
    "oracle_best_rank",
    "oracle_action",
    "delta_gold_logprob",
    "base_gold_logprob",
    "promoted_gold_logprob",
    "utility_label",
    "evidence_utilization_type",
    "promoted_prediction",
    "base_prediction",
    "base_answer_complete",
    "is_gold_evidence",
    "gold_evidence_overlap",
    "answer_changed",
    "exact_match",
    "rouge_l",
    "sampled_answers",
    "majority_answer",
    "question_id",
    "paper_id",
    "question",
    "candidate_text",
    "draft_answer",
}

FORBIDDEN_SUBSTRINGS: Sequence[str] = (
    "target",
    "gold",
    "f1",
    "oracle",
    "utility_label",
    "promoted_pred",
    "is_gold",
    "evidence_utilization",
)


def assert_no_feature_leakage(feature_cols: Sequence[str]) -> None:
    """Hard assertion enforcing strict leakage prevention on feature columns."""
    for col in feature_cols:
        col_lower = col.lower().strip()
        if col in FORBIDDEN_EXACT_FEATURE_NAMES:
            raise AssertionError(
                f"CRITICAL LEAKAGE DETECTED: Column '{col}' is explicitly forbidden as an input feature!"
            )
        for sub in FORBIDDEN_SUBSTRINGS:
            if sub in col_lower:
                raise AssertionError(
                    f"CRITICAL LEAKAGE DETECTED: Column '{col}' contains forbidden substring '{sub}'!"
                )


def build_utility_dataset(
    df_promotions: pd.DataFrame,
    df_drafts: Optional[pd.DataFrame] = None,
    df_uncertainty: Optional[pd.DataFrame] = None,
    df_self_consistency: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Merge promotion results, draft answers, uncertainty, and agreement features with complete targets."""
    df_prom = df_promotions.copy()
    df_prom["question_id"] = df_prom["question_id"].astype(str)

    if df_drafts is not None and len(df_drafts) > 0:
        df_dr = df_drafts.copy()
        df_dr["question_id"] = df_dr["question_id"].astype(str)
        draft_cols = ["question_id", "draft_answer", "context_tokens"]
        df_dr_sub = df_dr[[c for c in draft_cols if c in df_dr.columns]]
        merged = pd.merge(df_prom, df_dr_sub, on="question_id", how="inner", suffixes=("", "_draft"))
    else:
        merged = df_prom
        if "draft_answer" not in merged.columns:
            merged["draft_answer"] = merged.get("base_prediction", "")

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

    # 1. Standard G0 counterfactual Delta F1
    merged["delta_f1"] = merged["delta_f1"].astype(float)
    merged["delta_f1_g0"] = merged["delta_f1"]
    merged["target_continuous_delta"] = merged["delta_f1"]

    # 2. Binary HELP and HURT
    merged["target_help_binary"] = (merged["delta_f1"] > 0.01).astype(int)
    merged["target_hurt_binary"] = (merged["delta_f1"] < -0.01).astype(int)

    # 3. Beneficial change
    ans_chg = (
        merged["answer_changed"].astype(int)
        if "answer_changed" in merged.columns
        else (merged["delta_f1"].abs() > 1e-4).astype(int)
    )
    merged["target_beneficial_change"] = ((ans_chg == 1) & (merged["delta_f1"] > 0.01)).astype(int)

    # 4. Dense gold log-likelihood with explicit mask
    if "delta_gold_logprob" in merged.columns and merged["delta_gold_logprob"].notna().any():
        merged["target_dense_gold_ll"] = merged["delta_gold_logprob"].fillna(0.0).astype(float)
        merged["target_mask_ll"] = merged["delta_gold_logprob"].notna().astype(float)
    elif "delta_ll" in merged.columns and merged["delta_ll"].notna().any():
        merged["target_dense_gold_ll"] = merged["delta_ll"].fillna(0.0).astype(float)
        merged["target_mask_ll"] = merged["delta_ll"].notna().astype(float)
    else:
        merged["target_dense_gold_ll"] = 0.0
        merged["target_mask_ll"] = 0.0

    # 5. 3-way label: 0=NEUTRAL, 1=HELP, 2=HURT
    def map_3way(val: float) -> int:
        if val > 0.01:
            return 1
        elif val < -0.01:
            return 2
        else:
            return 0

    merged["target_3way_label"] = merged["delta_f1"].apply(map_3way)

    # 6. Oracle-relative regret target: Regret_qi = Delta_q^* - Delta_qi
    # Compute Delta_q^* across all candidates for each question
    max_delta_per_q = merged.groupby("question_id")["delta_f1"].transform("max")
    merged["oracle_max_delta"] = max_delta_per_q
    merged["target_regret"] = (max_delta_per_q - merged["delta_f1"]).clip(lower=0.0)
    merged["target_is_oracle_best"] = (merged["delta_f1"] >= (max_delta_per_q - 1e-6)).astype(int)
    merged["target_is_near_oracle"] = (merged["target_regret"] <= 0.01).astype(int)

    logger.info(
        "Constructed utility dataset with %d rows across %d queries with full targets and regret.",
        len(merged),
        merged["question_id"].nunique(),
    )
    return merged


def get_non_leaking_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return strictly verified non-leaking numeric feature column names.

    Enforces hard assertion against leakage before returning.
    """
    numeric_cols = [
        c
        for c in df.columns
        if c not in FORBIDDEN_EXACT_FEATURE_NAMES
        and not any(sub in c.lower() for sub in FORBIDDEN_SUBSTRINGS)
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    # Enforce hard assertion check
    assert_no_feature_leakage(numeric_cols)
    return sorted(numeric_cols)


def group_utility_dataset_by_query(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Group candidate rows by question_id preserving rank order."""
    grouped = {}
    for qid, group in df.groupby("question_id"):
        # Sort by candidate_passage_index or semantic_rank if present
        sort_col = "candidate_passage_index" if "candidate_passage_index" in group.columns else "semantic_rank"
        if sort_col in group.columns:
            group = group.sort_values(sort_col)
        grouped[str(qid)] = group.reset_index(drop=True)
    return grouped


def get_paper_grouped_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Generate paper-grouped cross-validation splits (GroupKFold by paper_id).

    Ensures no paper is shared between train and validation folds.
    """
    if "paper_id" not in df.columns:
        raise ValueError("DataFrame must contain 'paper_id' column for paper-grouped cross validation.")

    gkf = GroupKFold(n_splits=n_splits)
    groups = df["paper_id"].values
    return gkf.split(df, groups=groups)


# Aliases
assemble_multitask_utility_dataset = build_utility_dataset


def extract_utility_targets(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract standard non-leaking utility targets from a promotion record."""
    delta = float(record.get("delta_f1", 0.0))
    is_help = 1.0 if delta > 0.01 else 0.0
    is_hurt = 1.0 if delta < -0.01 else 0.0
    label_3way = 1 if delta > 0.01 else (2 if delta < -0.01 else 0)
    delta_ll = float(record.get("delta_gold_logprob", record.get("delta_ll", 0.0)))
    has_ll = 1.0 if ("delta_gold_logprob" in record or "delta_ll" in record) else 0.0
    return {
        "delta_f1": delta,
        "is_help": is_help,
        "is_hurt": is_hurt,
        "label_3way": label_3way,
        "delta_gold_logprob": delta_ll,
        "mask_ll": has_ll,
    }
