"""Model Family 1: Gradient-Boosted Decision Trees for Week 7-C.

Implements:
- M1-A: Sufficiency Classifier: x(q, k) -> P(z(q,k) = 1)
- M1-B: Regret Regressor (Auxiliary): x(q, k) -> r_hat(q, k) (trained on log(1 + regret))
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Check LightGBM availability
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM not installed. Falling back to sklearn HistGradientBoosting.")

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


class SufficiencyGBDTClassifier:
    """Primary GBDT model predicting sufficiency probability P(z(q,k)=1)."""

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

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "SufficiencyGBDTClassifier":
        """Fit binary classifier on candidate features X and sufficiency labels y."""
        self.feature_names = list(X.columns)
        if HAS_LIGHTGBM:
            self.model = lgb.LGBMClassifier(
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
            self.model = HistGradientBoostingClassifier(
                max_depth=self.params["max_depth"],
                max_iter=self.params["n_estimators"],
                learning_rate=self.params["learning_rate"],
                min_samples_leaf=self.params["min_child_samples"],
                random_state=self.random_state,
            )
            self.model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of sufficiency P(z=1). Returns 1D float array."""
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        probs = self.model.predict_proba(X)
        return probs[:, 1]

    def get_feature_importance(self) -> pd.DataFrame:
        """Extract feature importance ranking."""
        if self.model is None or not self.feature_names:
            return pd.DataFrame()
        if HAS_LIGHTGBM and hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        else:
            importances = np.ones(len(self.feature_names))
        df_imp = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances,
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)
        return df_imp


class RegretGBDTRegressor:
    """Auxiliary GBDT model predicting continuous regret r_hat(q, k)."""

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

    def fit(self, X: pd.DataFrame, y_regret: np.ndarray) -> "RegretGBDTRegressor":
        """Fit regressor on log1p(regret) to stabilize zero-inflation."""
        self.feature_names = list(X.columns)
        y_log = np.log1p(np.maximum(0.0, y_regret))

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
            self.model.fit(X, y_log)
        else:
            self.model = HistGradientBoostingRegressor(
                max_depth=self.params["max_depth"],
                max_iter=self.params["n_estimators"],
                learning_rate=self.params["learning_rate"],
                min_samples_leaf=self.params["min_child_samples"],
                random_state=self.random_state,
            )
            self.model.fit(X, y_log)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict continuous regret (transformed back via expm1)."""
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        pred_log = self.model.predict(X)
        pred_regret = np.expm1(np.maximum(0.0, pred_log))
        return np.maximum(0.0, pred_regret)
