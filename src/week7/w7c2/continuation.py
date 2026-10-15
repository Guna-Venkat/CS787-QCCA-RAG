"""Optional Model Family M3: Stagewise Continuation Policy for Week 7-C2.

Evaluates progressive budget expansion without assuming monotonic quality:
At stage k, asks: Is there meaningful utility available from any later budget?
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

logger = logging.getLogger(__name__)


class StagewiseContinuationPolicy:
    """Stage-specific continuation classifiers for stages k in {2, 4, 5, 6}."""

    def __init__(
        self,
        stages: Sequence[int] = (2, 4, 5, 6),
        max_depth: int = 4,
        n_estimators: int = 50,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ):
        self.stages = list(stages)
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.classifiers: Dict[int, HistGradientBoostingClassifier] = {}

    def fit_stage(self, stage_k: int, X_stage: pd.DataFrame, y_continue: np.ndarray) -> None:
        """Fit a binary continuation classifier for a specific stage k."""
        clf = HistGradientBoostingClassifier(
            max_depth=self.max_depth,
            max_iter=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )
        clf.fit(X_stage, y_continue)
        self.classifiers[stage_k] = clf

    def predict_continuation_prob(self, stage_k: int, X_stage: pd.DataFrame) -> np.ndarray:
        """Predict continuation probability P(y_continue=1) at stage k."""
        if stage_k not in self.classifiers:
            return np.full(len(X_stage), 0.0)
        return self.classifiers[stage_k].predict_proba(X_stage)[:, 1]

    def evaluate_stagewise_query(
        self,
        query_stage_features: Dict[int, pd.DataFrame],
        tau_continue: float = 0.50,
    ) -> int:
        """Evaluate stagewise continuation for a single query from stage k=2 onward.

        Returns selected budget hat_k.
        """
        for k in self.stages:
            if k not in query_stage_features or k not in self.classifiers:
                return k
            X_k = query_stage_features[k]
            p_cont = self.predict_continuation_prob(k, X_k)[0]
            if p_cont < tau_continue:
                return k  # STOP at current budget
        # Reached terminal stage
        return 8
