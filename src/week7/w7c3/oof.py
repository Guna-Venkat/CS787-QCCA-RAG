"""Out-of-Fold (OOF) GroupKFold training and policy calibration for Week 7-C3 Anchor-and-Deviate.

Implements:
- 5-fold GroupKFold grouped strictly by paper_id across all TRAIN queries.
- Leakage-free out-of-fold probability and delta regression estimation.
- Comprehensive grid evaluation of policy configurations (tau_down, tau_up, delta_up).
- Quality-constrained model selection on OOF predictions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.week7.w7c3.delta_models import MarginalGainDeltaRegressors
from src.week7.w7c3.downshift import SafeDownshiftClassifier
from src.week7.w7c3.policy import apply_anchor_deviate_decision_rule, evaluate_downstream_anchor_policy
from src.week7.w7c3.upshift import SelectiveUpshiftClassifier

logger = logging.getLogger(__name__)


def run_group_kfold_oof(
    df_train_query: pd.DataFrame,
    downshift_features: Sequence[str],
    upshift_features: Sequence[str],
    delta_features: Sequence[str],
    n_splits: int = 5,
    group_col: str = "paper_id",
    model_params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Run 5-fold GroupKFold cross-validation on TRAIN queries to generate OOF predictions.

    Ensures:
    - Queries from the same paper remain strictly in the same fold.
    - Every query in df_train_query receives out-of-fold predictions exactly once.
    - Zero data leakage into training folds.

    Returns:
        DataFrame with columns:
        question_id, paper_id, fold, p_down, p_up, pred_delta_5, pred_delta_6, pred_delta_8
    """
    params = model_params or {}
    down_params = params.get("downshift_classifier", {})
    up_params = params.get("upshift_classifier", {})
    delta_params = params.get("delta_regressors", {})

    gkf = GroupKFold(n_splits=n_splits)
    groups = df_train_query[group_col].values

    oof_records: List[Dict[str, Any]] = []

    logger.info("Starting %d-fold GroupKFold OOF on %d queries across %d papers...", n_splits, len(df_train_query), df_train_query[group_col].nunique())

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(df_train_query, groups=groups)):
        df_train_fold = df_train_query.iloc[train_idx].copy()
        df_val_fold = df_train_query.iloc[val_idx].copy()

        # Verify group separation
        train_papers = set(df_train_fold[group_col])
        val_papers = set(df_val_fold[group_col])
        assert train_papers.isdisjoint(val_papers), f"LEAKAGE in fold {fold_idx}: overlapping papers!"

        # 1. Train Downshift Classifier
        down_clf = SafeDownshiftClassifier(**down_params)
        down_clf.fit(df_train_fold[list(downshift_features)], df_train_fold["target_y_down"])
        p_down_val = down_clf.predict_proba(df_val_fold[list(downshift_features)])

        # 2. Train Upshift Classifier
        up_clf = SelectiveUpshiftClassifier(**up_params)
        up_clf.fit(df_train_fold[list(upshift_features)], df_train_fold["target_y_up"])
        p_up_val = up_clf.predict_proba(df_val_fold[list(upshift_features)])

        # 3. Train Delta Regressors
        delta_regs = MarginalGainDeltaRegressors(**delta_params)
        delta_regs.fit(df_train_fold[list(delta_features)], df_train_fold)
        pred_deltas_val = delta_regs.predict_deltas(df_val_fold[list(delta_features)])

        for i, (_, row) in enumerate(df_val_fold.iterrows()):
            qid = str(row["question_id"]).strip()
            pid = str(row[group_col]).strip()
            oof_records.append({
                "question_id": qid,
                "paper_id": pid,
                "fold": fold_idx,
                "p_down": float(p_down_val[i]),
                "p_up": float(p_up_val[i]),
                "pred_delta_5": float(pred_deltas_val[5][i]),
                "pred_delta_6": float(pred_deltas_val[6][i]),
                "pred_delta_8": float(pred_deltas_val[8][i]),
            })

    df_oof = pd.DataFrame(oof_records)
    if len(df_oof) != len(df_train_query):
        raise ValueError(f"OOF row count mismatch! Expected {len(df_train_query)}, got {len(df_oof)}")

    logger.info("Completed GroupKFold OOF generation for %d queries.", len(df_oof))
    return df_oof


def evaluate_oof_policy_grid(
    df_oof: pd.DataFrame,
    df_response_matrix: pd.DataFrame,
    tau_down_grid: Sequence[float] = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95),
    tau_up_grid: Sequence[float] = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85),
    delta_up_grid: Sequence[float] = (0.0, 0.005, 0.01, 0.02),
    anchor_k: int = 4,
    budgets: Sequence[int] = (2, 4, 5, 6, 8),
    epsilon: float = 0.01,
) -> pd.DataFrame:
    """Evaluate grid of policy configurations over OOF predictions."""
    results: List[Dict[str, Any]] = []

    # Map qid to OOF row for fast lookup
    oof_by_qid = {str(r["question_id"]).strip(): r for _, r in df_oof.iterrows()}

    for tau_down in tau_down_grid:
        for tau_up in tau_up_grid:
            for delta_up in delta_up_grid:
                # Apply policy to each query
                pred_ks: Dict[str, int] = {}
                for qid, oof_row in oof_by_qid.items():
                    p_down = float(oof_row["p_down"])
                    p_up = float(oof_row["p_up"])
                    deltas = {
                        5: float(oof_row["pred_delta_5"]),
                        6: float(oof_row["pred_delta_6"]),
                        8: float(oof_row["pred_delta_8"]),
                    }
                    k_choice = apply_anchor_deviate_decision_rule(
                        p_down=p_down,
                        p_up=p_up,
                        pred_deltas=deltas,
                        tau_down=tau_down,
                        tau_up=tau_up,
                        delta_up=delta_up,
                        anchor_k=anchor_k,
                    )
                    pred_ks[qid] = k_choice

                eval_metrics = evaluate_downstream_anchor_policy(
                    predicted_ks=pred_ks,
                    response_matrix_df=df_response_matrix,
                    budgets=budgets,
                    anchor_k=anchor_k,
                    epsilon=epsilon,
                )

                eval_metrics["tau_down"] = tau_down
                eval_metrics["tau_up"] = tau_up
                eval_metrics["delta_up"] = delta_up
                results.append(eval_metrics)

    df_grid = pd.DataFrame(results)
    logger.info("Evaluated %d policy configurations on OOF predictions.", len(df_grid))
    return df_grid
