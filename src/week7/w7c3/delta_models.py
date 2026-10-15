"""Marginal Gain Delta Regressors (U2) for Week 7-C3 Anchor-and-Deviate.

Predicts marginal answer quality gains relative to the default anchor k=4:
    Delta_5(q) = F(q, 5) - F(q, 4)
    Delta_6(q) = F(q, 6) - F(q, 4)
    Delta_8(q) = F(q, 8) - F(q, 4)

Provides:
- Multi-target delta regression models.
- Near-best candidate set construction with cost-aware selection:
    U_delta = {k in {5, 6, 8} : hat_Delta_k >= hat_Delta_max - delta_up}
    hat_k = min U_delta  (chooses cheapest budget within delta_up of max gain).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

logger = logging.getLogger(__name__)

DEFAULT_UPSHIFT_BUDGETS: List[int] = [5, 6, 8]


class MarginalGainDeltaRegressors:
    """Trains and predicts marginal gain deltas relative to anchor k=4."""

    def __init__(
        self,
        budgets: Sequence[int] = (5, 6, 8),
        targets: Optional[Sequence[int]] = None,
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
        self.budgets = list(targets if targets is not None else budgets)
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

        self.models: Dict[int, Any] = {}
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, targets_df: pd.DataFrame) -> "MarginalGainDeltaRegressors":
        """Fit separate regressors for each upshift target Delta_k."""
        self.feature_names_ = list(X.columns)

        for k in self.budgets:
            col_name = f"target_delta_{k}"
            if col_name not in targets_df.columns:
                raise ValueError(f"Target column '{col_name}' missing from targets DataFrame!")

            y_arr = targets_df[col_name].values.astype(float)

            if self.model_family == "lgbm":
                reg = lgb.LGBMRegressor(
                    max_depth=self.max_depth,
                    num_leaves=self.num_leaves,
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    min_child_samples=self.min_child_samples,
                    subsample=self.subsample,
                    colsample_bytree=self.colsample_bytree,
                    random_state=self.random_state + k,
                    verbose=-1,
                )
            else:
                reg = HistGradientBoostingRegressor(
                    max_depth=self.max_depth,
                    max_iter=self.n_estimators,
                    learning_rate=self.learning_rate,
                    min_samples_leaf=self.min_child_samples,
                    random_state=self.random_state + k,
                )

            reg.fit(X, y_arr)
            self.models[k] = reg

        return self

    def predict_deltas(self, X: pd.DataFrame) -> Dict[int, np.ndarray]:
        """Predict marginal gains for each candidate budget in {5, 6, 8}."""
        if not self.models:
            raise RuntimeError("Models have not been fitted!")

        preds: Dict[int, np.ndarray] = {}
        for k in self.budgets:
            preds[k] = self.models[k].predict(X)
        return preds

    def select_larger_budget(
        self,
        predicted_deltas: Dict[int, float],
        delta_up: float = 0.0,
    ) -> int:
        """Select larger budget using cost-aware near-tie rule.

        Formula:
            hat_Delta_max = max(hat_Delta_5, hat_Delta_6, hat_Delta_8)
            U_delta = {k in {5, 6, 8} : hat_Delta_k >= hat_Delta_max - delta_up}
            hat_k = min U_delta  (since 5 < 6 < 8 corresponds to increasing cost).
        """
        sorted_budgets = sorted(self.budgets)
        max_delta = max(predicted_deltas[k] for k in sorted_budgets)

        near_best = [k for k in sorted_budgets if predicted_deltas[k] >= max_delta - delta_up]
        return min(near_best)
