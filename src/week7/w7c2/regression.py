"""Model Family M1: Direct Response-Surface Regressor for Week 7-C2.

Predicts downstream answer quality hat_F(q, k) directly for each candidate budget k in {2, 4, 5, 6, 8}.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM not installed. Falling back to HistGradientBoostingRegressor.")

from sklearn.ensemble import HistGradientBoostingRegressor


class DirectF1Regressor:
    """Primary M1 model predicting continuous downstream F1 quality hat_F(q, k)."""

    def __init__(
        self,
        max_depth: int = 5,
        num_leaves: int = 15,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.params = {
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
        }
        self.random_state = random_state
        self.model = None
        self.feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "DirectF1Regressor":
        """Fit regressor on candidate-budget features X and downstream F1 scores y."""
        self.feature_names = list(X.columns)

        if HAS_LIGHTGBM:
            self.model = lgb.LGBMRegressor(
                max_depth=self.params["max_depth"],
                num_leaves=self.params["num_leaves"],
                n_estimators=self.params["n_estimators"],
                learning_rate=self.params["learning_rate"],
                min_child_samples=self.params["min_child_samples"],
                subsample=self.params["subsample"],
                colsample_bytree=self.params["colsample_bytree"],
                random_state=self.random_state,
                verbose=-1,
            )
            self.model.fit(X, y)
        else:
            self.model = HistGradientBoostingRegressor(
                max_depth=self.params["max_depth"],
                max_iter=self.params["n_estimators"],
                learning_rate=self.params["learning_rate"],
                min_samples_leaf=self.params["min_child_samples"],
                random_state=self.random_state,
            )
            self.model.fit(X, y)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict downstream answer F1 hat_F(q, k)."""
        if self.model is None:
            raise RuntimeError("DirectF1Regressor is not fitted.")
        return np.clip(self.model.predict(X), 0.0, 1.0)

    def get_feature_importance(self) -> pd.DataFrame:
        if self.model is None or not self.feature_names:
            return pd.DataFrame()
        if HAS_LIGHTGBM and hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        else:
            importances = np.ones(len(self.feature_names))
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances,
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)
