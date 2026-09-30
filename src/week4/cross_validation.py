"""Grouped Cross-Validation (by paper_id) and out-of-fold prediction runner for Week 4."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.week4.evaluation import FrozenResponseSurfaceLookup
from src.week4.qcca_models import (
    create_decision_tree_pipeline,
    create_logistic_regression_pipeline
)
from src.week4.qcca_rule import QCCARuleAllocator

logger = logging.getLogger(__name__)


def run_grouped_cross_validation(
    df_data: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target_k",
    group_col: str = "paper_id",
    n_splits: int = 5,
    random_state: int = 42,
    logreg_C: float = 1.0,
    tree_depth: int = 3
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run strictly nested GroupKFold cross-validation by paper_id."""
    unique_groups = df_data[group_col].nunique()
    effective_splits = min(n_splits, unique_groups)
    if effective_splits < n_splits:
        logger.warning(
            "Unique groups (%d) < requested n_splits (%d). Clamping n_splits to %d.",
            unique_groups, n_splits, effective_splits
        )
        
    gkf = GroupKFold(n_splits=effective_splits)
    groups = df_data[group_col].values
    
    oof_records = []
    trained_models = {
        "rule": [],
        "logreg": [],
        "tree": []
    }
    
    # Store fold verification records
    fold_verifications = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df_data, groups=groups)):
        train_df = df_data.iloc[train_idx].copy()
        val_df = df_data.iloc[val_idx].copy()
        
        # Rigorous check: train papers and val papers must be mutually disjoint
        train_papers = set(train_df[group_col].unique())
        val_papers = set(val_df[group_col].unique())
        paper_overlap = train_papers.intersection(val_papers)
        if paper_overlap:
            raise ValueError(f"Fold {fold} integrity violation: Papers {paper_overlap} in both train and validation!")
            
        fold_verifications.append({
            "fold": fold,
            "n_train_questions": len(train_df),
            "n_val_questions": len(val_df),
            "n_train_papers": len(train_papers),
            "n_val_papers": len(val_papers),
            "paper_overlap_count": len(paper_overlap)
        })
        
        X_train = train_df[feature_cols]
        y_train = train_df[target_col].values
        
        X_val = val_df[feature_cols]
        y_val = val_df[target_col].values
        
        # 1. Train QCCA-Rule
        rule_model = QCCARuleAllocator()
        rule_model.fit(X_train, y_train)
        pred_rule = rule_model.predict(X_val)
        trained_models["rule"].append(rule_model)
        
        # 2. Train QCCA-Learned (Logistic Regression inside Pipeline)
        logreg_model = create_logistic_regression_pipeline(C=logreg_C, random_state=random_state)
        logreg_model.fit(X_train, y_train)
        pred_logreg = logreg_model.predict(X_val)
        trained_models["logreg"].append(logreg_model)
        
        # 3. Train Secondary Decision Tree
        tree_model = create_decision_tree_pipeline(max_depth=tree_depth, random_state=random_state)
        tree_model.fit(X_train, y_train)
        pred_tree = tree_model.predict(X_val)
        trained_models["tree"].append(tree_model)
        
        for i, idx in enumerate(val_idx):
            row = df_data.iloc[idx]
            oof_records.append({
                "question_id": row["question_id"],
                "paper_id": row["paper_id"],
                "fold": fold,
                "target_k": int(row[target_col]),
                "pred_k_rule": int(pred_rule[i]),
                "pred_k_logreg": int(pred_logreg[i]),
                "pred_k_tree": int(pred_tree[i])
            })
            
    df_oof = pd.DataFrame(oof_records)
    
    # Sort by original question_id ordering
    df_oof = df_oof.sort_values(by="question_id").reset_index(drop=True)
    
    cv_metadata = {
        "n_splits": effective_splits,
        "group_column": group_col,
        "folds": fold_verifications,
        "models": trained_models
    }
    return df_oof, cv_metadata


def verify_oof_integrity(df_oof: pd.DataFrame, expected_count: int, k_alloc: List[int]) -> None:
    """Run strict integrity audits on out-of-fold predictions table."""
    # 1. Total row count
    if len(df_oof) != expected_count:
        raise ValueError(f"OOF row count mismatch: expected {expected_count}, got {len(df_oof)}")
        
    # 2. Each question appears exactly once
    if df_oof["question_id"].duplicated().any():
        dup_qids = df_oof[df_oof["question_id"].duplicated()]["question_id"].tolist()
        raise ValueError(f"Duplicate questions in OOF table: {dup_qids}")
        
    # 3. Metadata columns present
    for col in ["question_id", "paper_id", "fold", "target_k"]:
        if col not in df_oof.columns:
            raise ValueError(f"Missing required OOF column: {col}")
            
    # 4. No missing values
    if df_oof.isna().any().any():
        raise ValueError(f"NaN values found in OOF table:\n{df_oof.isna().sum()}")
        
    # 5. Predictions in K_alloc
    for pred_col in ["pred_k_rule", "pred_k_logreg", "pred_k_tree"]:
        if pred_col in df_oof.columns:
            invalid = set(df_oof[pred_col]) - set(k_alloc)
            if invalid:
                raise ValueError(f"Invalid allocation actions in {pred_col}: {invalid}")
                
    logger.info("OOF Integrity Verification Passed: %d predictions verified.", len(df_oof))
