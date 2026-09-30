"""QCCA-Rule: Interpretable Rule-Based Context Allocator."""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

logger = logging.getLogger(__name__)


class QCCARuleAllocator(BaseEstimator, ClassifierMixin):
    """Interpretable, deterministic rule-based allocator.
    
    Principle:
    - High retrieval concentration (high rho_1, high delta_12, low entropy, low N_high)
      indicates that leading evidence is sufficient -> allocate smaller k (e.g. k=2, 4).
    - Diffuse retrieval distribution (low rho_1, low delta_12, high entropy, high N_high)
      indicates evidence is dispersed across multiple passages -> allocate larger k (e.g. k=6, 8).
      
    Thresholds on the composite concentration index are fitted strictly on training data
    based on training fold quantile cutoffs matching the action space K_alloc = {2, 4, 5, 6, 8}.
    """
    
    def __init__(
        self,
        k_alloc: Optional[List[int]] = None
    ):
        self.k_alloc = k_alloc if k_alloc is not None else [2, 4, 5, 6, 8]
        self.thresholds_: List[float] = []
        self.feature_means_: Dict[str, float] = {}
        self.feature_stds_: Dict[str, float] = {}
        self.classes_ = np.array(sorted(self.k_alloc))
        
    def _compute_concentration_index(self, X: pd.DataFrame) -> np.ndarray:
        """Compute standardized composite concentration index."""
        # Concentration index: + rho_1 + delta_12 - entropy_normalized - n_high
        idx = np.zeros(len(X))
        
        components = [
            ("rho_1", 1.0),
            ("delta_12", 1.0),
            ("entropy_normalized", -1.0),
            ("n_high", -1.0)
        ]
        
        for col, sign in components:
            if col in X.columns:
                mean = self.feature_means_.get(col, 0.0)
                std = self.feature_stds_.get(col, 1.0)
                if std < 1e-6:
                    std = 1.0
                z = (X[col].values - mean) / std
                idx += sign * z
                
        return idx
        
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """Fit rule thresholds strictly on training fold."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        y = np.asarray(y)
        
        # Calculate feature means and stds on training data
        for col in ["rho_1", "delta_12", "entropy_normalized", "n_high"]:
            if col in X.columns:
                self.feature_means_[col] = float(np.mean(X[col]))
                self.feature_stds_[col] = float(np.std(X[col]))
                
        # Compute concentration scores on training data
        conc_scores = self._compute_concentration_index(X)
        
        # Derive class distribution in training data for k_alloc = [2, 4, 5, 6, 8]
        # High concentration score -> small k; low concentration score -> large k
        # We compute quantiles for cutoffs:
        p2 = np.mean(y == 2)
        p4 = np.mean(y == 4)
        p5 = np.mean(y == 5)
        p6 = np.mean(y == 6)
        
        # Cumulative cutoffs from highest concentration (k=2) to lowest (k=8)
        # k=2: top p2 -> percentile 100 * (1 - p2)
        # k=4: next p4 -> percentile 100 * (1 - p2 - p4)
        # k=5: next p5 -> percentile 100 * (1 - p2 - p4 - p5)
        # k=6: next p6 -> percentile 100 * (1 - p2 - p4 - p5 - p6)
        # k=8: bottom remainder
        q2 = max(0.0, 1.0 - p2)
        q4 = max(0.0, q2 - p4)
        q5 = max(0.0, q4 - p5)
        q6 = max(0.0, q5 - p6)
        
        t2 = float(np.quantile(conc_scores, q2))
        t4 = float(np.quantile(conc_scores, q4))
        t5 = float(np.quantile(conc_scores, q5))
        t6 = float(np.quantile(conc_scores, q6))
        
        self.thresholds_ = [t2, t4, t5, t6]
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict evidence allocation k in {2, 4, 5, 6, 8}."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        conc_scores = self._compute_concentration_index(X)
        t2, t4, t5, t6 = self.thresholds_
        
        preds = np.zeros(len(conc_scores), dtype=int)
        for i, score in enumerate(conc_scores):
            if score >= t2:
                preds[i] = 2
            elif score >= t4:
                preds[i] = 4
            elif score >= t5:
                preds[i] = 5
            elif score >= t6:
                preds[i] = 6
            else:
                preds[i] = 8
                
        return preds
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """One-hot style pseudo probabilities for interface consistency."""
        preds = self.predict(X)
        probas = np.zeros((len(preds), len(self.classes_)))
        for i, p in enumerate(preds):
            idx = np.where(self.classes_ == p)[0][0]
            probas[i, idx] = 1.0
        return probas
