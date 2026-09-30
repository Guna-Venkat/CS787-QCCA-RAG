"""Model definitions and pipelines for Week 4 QCCA."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.week4.qcca_rule import QCCARuleAllocator

logger = logging.getLogger(__name__)


def create_logistic_regression_pipeline(
    C: float = 1.0,
    class_weight: Optional[str] = None,
    random_state: int = 42
) -> Pipeline:
    """Create a strictly nested Pipeline with StandardScaler and Logistic Regression."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=C,
            solver="lbfgs",
            class_weight=class_weight,
            max_iter=1000,
            random_state=random_state
        ))
    ])


def create_decision_tree_pipeline(
    max_depth: int = 3,
    min_samples_leaf: int = 5,
    random_state: int = 42
) -> DecisionTreeClassifier:
    """Create a shallow interpretable DecisionTreeClassifier."""
    return DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )


def extract_feature_importance(
    model: Any,
    feature_names: List[str]
) -> pd.DataFrame:
    """Extract interpretable feature importance or coefficient summaries."""
    records = []
    
    if isinstance(model, Pipeline) and isinstance(model.named_steps.get("classifier"), LogisticRegression):
        clf = model.named_steps["classifier"]
        # shape: (n_classes, n_features)
        coefs = clf.coef_
        classes = clf.classes_
        
        # Mean absolute coefficient across classes
        mean_abs_coef = np.mean(np.abs(coefs), axis=0)
        for idx, feat in enumerate(feature_names):
            class_breakdown = {f"coef_class_{c}": float(coefs[c_idx, idx]) for c_idx, c in enumerate(classes)}
            rec = {
                "feature": feat,
                "model": "LogisticRegression",
                "importance_metric": "mean_abs_coefficient",
                "importance_value": float(mean_abs_coef[idx]),
                **class_breakdown
            }
            records.append(rec)
            
    elif isinstance(model, DecisionTreeClassifier):
        importances = model.feature_importances_
        for idx, feat in enumerate(feature_names):
            records.append({
                "feature": feat,
                "model": "DecisionTree",
                "importance_metric": "gini_importance",
                "importance_value": float(importances[idx])
            })
            
    elif isinstance(model, QCCARuleAllocator):
        for col in ["rho_1", "delta_12", "entropy_normalized", "n_high"]:
            if col in feature_names:
                records.append({
                    "feature": col,
                    "model": "QCCA-Rule",
                    "importance_metric": "rule_weight",
                    "importance_value": 1.0 if col in ["rho_1", "delta_12"] else -1.0
                })
                
    df_imp = pd.DataFrame(records)
    if not df_imp.empty and "importance_value" in df_imp.columns:
        df_imp = df_imp.sort_values(by="importance_value", ascending=False).reset_index(drop=True)
    return df_imp
