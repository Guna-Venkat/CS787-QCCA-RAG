"""Model Family M2: Query-Grouped Learning-to-Rank for Week 7-C2 (PRIMARY MODEL).

Trains a query-grouped pairwise/listwise ranker over the 5 candidate representation budgets
using lexicographic quality-cost relevance grades.
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

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class QueryGroupedBudgetRanker:
    """Primary M2 ranking model learning relative utility over candidate budgets."""

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

    def fit(
        self,
        X: pd.DataFrame,
        y_grades: np.ndarray,
        group_sizes: Sequence[int],
    ) -> "QueryGroupedBudgetRanker":
        """Fit ranking model on query-grouped candidate features and relevance grades.

        Args:
            X: Candidate feature DataFrame sorted contiguously by query.
            y_grades: Array of integer relevance grades (higher is better, e.g. 0 to 4).
            group_sizes: Sequence of integer group sizes (each query has 5 rows).
        """
        self.feature_names = list(X.columns)
        total_rows = len(X)
        if sum(group_sizes) != total_rows:
            raise ValueError(f"Sum of group_sizes ({sum(group_sizes)}) does not match X rows ({total_rows})!")

        if HAS_LIGHTGBM:
            self.model = lgb.LGBMRanker(
                objective="lambdarank",
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
            self.model.fit(X, y_grades, group=list(group_sizes))

        elif HAS_XGBOOST:
            self.model = xgb.XGBRanker(
                objective="rank:pairwise",
                max_depth=self.params["max_depth"],
                n_estimators=self.params["n_estimators"],
                learning_rate=self.params["learning_rate"],
                random_state=self.random_state,
            )
            self.model.fit(X, y_grades, group=list(group_sizes))

        else:
            raise RuntimeError("Neither LightGBM nor XGBoost is installed for ranking!")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict ranking scores s(q, k) for each candidate budget row.

        Higher score implies higher predicted utility.
        """
        if self.model is None:
            raise RuntimeError("QueryGroupedBudgetRanker is not fitted.")
        return self.model.predict(X)

    def get_feature_importance(self) -> pd.DataFrame:
        if self.model is None or not self.feature_names:
            return pd.DataFrame()
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        else:
            importances = np.ones(len(self.feature_names))
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances,
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)
