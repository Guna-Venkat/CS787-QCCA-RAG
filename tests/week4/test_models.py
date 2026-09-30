"""Unit tests for QCCA models: rule-based, logistic regression, and decision tree."""

import numpy as np
import pandas as pd
import pytest

from src.week4.qcca_models import (
    create_decision_tree_pipeline,
    create_logistic_regression_pipeline,
    extract_feature_importance
)
from src.week4.qcca_rule import QCCARuleAllocator


def test_rule_allocator_predictions_in_k_alloc():
    """Verify rule allocator outputs strictly in K_alloc."""
    X = pd.DataFrame({
        "rho_1": [0.9, 0.4, 0.1, 0.5, 0.3],
        "delta_12": [0.5, 0.2, 0.05, 0.1, 0.02],
        "entropy_normalized": [0.2, 0.5, 0.9, 0.4, 0.8],
        "n_high": [1, 2, 8, 3, 5]
    })
    y = np.array([2, 4, 8, 5, 6])
    
    rule = QCCARuleAllocator()
    rule.fit(X, y)
    preds = rule.predict(X)
    assert set(preds).issubset({2, 4, 5, 6, 8})


def test_logistic_regression_predictions_in_k_alloc():
    """Verify logistic regression outputs strictly in K_alloc and feature importance is extracted."""
    X = pd.DataFrame({
        "rho_1": np.random.uniform(0.1, 0.9, 20),
        "delta_12": np.random.uniform(0.01, 0.5, 20),
        "entropy_normalized": np.random.uniform(0.1, 0.9, 20),
        "n_high": np.random.randint(1, 10, 20)
    })
    y = np.random.choice([2, 4, 5, 6, 8], size=20)

    clf = create_logistic_regression_pipeline(C=1.0)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert set(preds).issubset({2, 4, 5, 6, 8})

    df_imp = extract_feature_importance(clf, list(X.columns))
    assert len(df_imp) == 4
    assert "importance_value" in df_imp.columns
