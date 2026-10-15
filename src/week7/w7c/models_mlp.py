"""Model Family 2: Small Multitask MLP for Week 7-C.

Implements a shared-trunk neural architecture with:
- Head 1: Binary sufficiency classification logit (BCE loss)
- Head 2: Continuous regret regression (Huber loss on log1p(regret))
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class MultitaskMLPNetwork(nn.Module):
    """PyTorch network with shared trunk and dual output heads."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim

        self.trunk = nn.Sequential(*layers)
        # Head 1: Sufficiency classification logit
        self.cls_head = nn.Linear(curr_dim, 1)
        # Head 2: Regret regression (predicting log1p(regret))
        self.reg_head = nn.Linear(curr_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(x)
        cls_logit = self.cls_head(features).squeeze(-1)
        reg_pred = self.reg_head(features).squeeze(-1)
        return cls_logit, reg_pred


class MultitaskMLPWrapper:
    """Sklearn-style wrapper for training and evaluating MultitaskMLPNetwork."""

    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 30,
        batch_size: int = 64,
        lambda_cls: float = 1.0,
        lambda_reg: float = 0.5,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.lambda_cls = lambda_cls
        self.lambda_reg = lambda_reg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state

        self.network: Optional[MultitaskMLPNetwork] = None
        self.feature_names: List[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y_cls: np.ndarray,
        y_regret: np.ndarray,
    ) -> "MultitaskMLPWrapper":
        """Fit multitask MLP using BCE + Huber loss."""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.feature_names = list(X.columns)
        input_dim = len(self.feature_names)

        X_t = torch.tensor(X.values, dtype=torch.float32)
        y_cls_t = torch.tensor(y_cls, dtype=torch.float32)
        y_reg_t = torch.tensor(np.log1p(np.maximum(0.0, y_regret)), dtype=torch.float32)

        dataset = TensorDataset(X_t, y_cls_t, y_reg_t)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.network = MultitaskMLPNetwork(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)

        bce_loss_fn = nn.BCEWithLogitsLoss()
        huber_loss_fn = nn.HuberLoss(delta=0.1)
        optimizer = optim.AdamW(self.network.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.network.train()
        for epoch in range(self.epochs):
            for batch_x, batch_cls, batch_reg in dataloader:
                batch_x = batch_x.to(self.device)
                batch_cls = batch_cls.to(self.device)
                batch_reg = batch_reg.to(self.device)

                optimizer.zero_grad()
                pred_cls_logit, pred_reg = self.network(batch_x)

                loss_cls = bce_loss_fn(pred_cls_logit, batch_cls)
                loss_reg = huber_loss_fn(pred_reg, batch_reg)
                total_loss = self.lambda_cls * loss_cls + self.lambda_reg * loss_reg

                total_loss.backward()
                optimizer.step()

        self.network.eval()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict sufficiency probabilities P(z=1)."""
        if self.network is None:
            raise RuntimeError("MLP is not fitted.")
        self.network.eval()
        X_t = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            cls_logits, _ = self.network(X_t)
            probs = torch.sigmoid(cls_logits).cpu().numpy()
        return probs

    def predict_regret(self, X: pd.DataFrame) -> np.ndarray:
        """Predict continuous regret r_hat."""
        if self.network is None:
            raise RuntimeError("MLP is not fitted.")
        self.network.eval()
        X_t = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            _, reg_preds = self.network(X_t)
            pred_log = reg_preds.cpu().numpy()
            pred_reg = np.expm1(np.maximum(0.0, pred_log))
        return np.maximum(0.0, pred_reg)
