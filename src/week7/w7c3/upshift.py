"""Selective Upshift Classifier (U1) for Week 7-C3 Anchor-and-Deviate.

Predicts P_up(q) = P(y_up=1 | x_q), where y_up=1 indicates that at least one
larger budget k in {5, 6, 8} provides meaningful downstream improvement (> epsilon)
over the default anchor k=4.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

logger = logging.getLogger(__name__)


class SelectiveUpshiftClassifier:
    """Estimates probability that expanding beyond k=4 improves answer quality materially."""

    def __init__(
        self,
        model_family: str = "lgbm",
        max_depth: int = 4,
        num_leaves: int = 15,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.model_family = model_family
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.extra_kwargs = kwargs

        self.model: Optional[Any] = None
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> "SelectiveUpshiftClassifier":
        """Fit the upshift classifier."""
        self.feature_names_ = list(X.columns)
        y_arr = np.asarray(y, dtype=int)

        if self.model_family == "lgbm":
            self.model = lgb.LGBMClassifier(
                max_depth=self.max_depth,
                num_leaves=self.num_leaves,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                min_child_samples=self.min_child_samples,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state,
                verbose=-1,
            )
        else:
            self.model = HistGradientBoostingClassifier(
                max_depth=self.max_depth,
                max_iter=self.n_estimators,
                learning_rate=self.learning_rate,
                min_samples_leaf=self.min_child_samples,
                random_state=self.random_state,
            )

        self.model.fit(X, y_arr)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of selective upshift P_up(q)."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted!")
        probs = self.model.predict_proba(X)
        if probs.shape[1] == 2:
            return probs[:, 1]
        elif probs.shape[1] == 1:
            return np.ones(len(X), dtype=float) if self.model.classes_[0] == 1 else np.zeros(len(X), dtype=float)
        return probs[:, 1]

    def get_feature_importances(self) -> pd.Series:
        """Return feature importances."""
        if self.model is None or not hasattr(self.model, "feature_importances_"):
            return pd.Series(dtype=float)
        return pd.Series(self.model.feature_importances_, index=self.feature_names_).sort_values(ascending=False)
