"""W8-C & W8-D: Passage Utility Models (U0, U1, U2, U4).

Models:
- U0: Tabular baseline (GBDT + Logistic) on engineered features.
- U1: Pretrained Cross-Encoder on [q, D_i] with multitask heads.
- U2: Draft-Conditioned Cross-Encoder on [q, y_0, D_i] with multitask heads (PRIMARY).
- U4: Full Hybrid Model fusing U2 representation + all numeric features (uncertainty, attribution, MIG).

Heads:
- Head 1: P(HELP) binary BCE
- Head 2: P(HURT) binary BCE
- Head 3: continuous Delta_F1 regression MSE
- Head 4: 3-way classification CrossEntropy (0: NEUTRAL, 1: HELP, 2: HURT)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Multitask Head Module
# ---------------------------------------------------------------------------
class MultitaskUtilityHeads(nn.Module):
    """Multitask heads: HELP (binary), HURT (binary), Delta (regression), 3-way (ce)."""

    def __init__(
        self,
        in_features: Optional[int] = None,
        hidden_dim: int = 64,
        input_dim: Optional[int] = None,
    ):
        super().__init__()
        in_dim = in_features if in_features is not None else (input_dim if input_dim is not None else 64)
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.head_help = nn.Linear(hidden_dim, 1)        # Binary logit
        self.head_hurt = nn.Linear(hidden_dim, 1)        # Binary logit
        self.head_delta = nn.Linear(hidden_dim, 1)       # Continuous delta F1
        self.head_delta_ll = nn.Linear(hidden_dim, 1)    # Dense continuous gold log-likelihood change (Delta LL)
        self.head_3way = nn.Linear(hidden_dim, 3)        # 3 classes: 0=NEU, 1=HELP, 2=HURT

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.shared(x)
        logit_help = self.head_help(feat).squeeze(-1)
        logit_hurt = self.head_hurt(feat).squeeze(-1)
        pred_delta = self.head_delta(feat).squeeze(-1)
        pred_delta_ll = self.head_delta_ll(feat).squeeze(-1)
        logits_3way = self.head_3way(feat)

        return {
            "logit_help": logit_help,
            "logit_hurt": logit_hurt,
            "pred_delta": pred_delta,
            "pred_delta_f1": pred_delta,
            "pred_delta_ll": pred_delta_ll,
            "pred_delta_gold_logprob": pred_delta_ll,
            "logits_3way": logits_3way,
            "label_logits": logits_3way,
            "prob_help": torch.sigmoid(logit_help),
            "prob_hurt": torch.sigmoid(logit_hurt),
        }


# ---------------------------------------------------------------------------
# U0: Tabular Baseline
# ---------------------------------------------------------------------------
class UtilityModelU0:
    """Tabular GBDT and Logistic baseline on engineered features."""

    def __init__(self):
        self.clf_help = HistGradientBoostingClassifier(random_state=42)
        self.clf_hurt = HistGradientBoostingClassifier(random_state=42)
        self.reg_delta = HistGradientBoostingRegressor(random_state=42)
        self.reg_delta_ll = HistGradientBoostingRegressor(random_state=42)
        self.clf_3way = HistGradientBoostingClassifier(random_state=42)

    def fit(
        self,
        X: np.ndarray,
        y_help: np.ndarray,
        y_hurt: np.ndarray,
        y_delta: np.ndarray,
        y_3way: np.ndarray,
        y_delta_ll: Optional[np.ndarray] = None,
    ):
        self.clf_help.fit(X, y_help)
        self.clf_hurt.fit(X, y_hurt)
        self.reg_delta.fit(X, y_delta)
        self.clf_3way.fit(X, y_3way)
        if y_delta_ll is not None:
            self.reg_delta_ll.fit(X, y_delta_ll)

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        p_help = self.clf_help.predict_proba(X)[:, 1]
        p_hurt = self.clf_hurt.predict_proba(X)[:, 1]
        p_delta = self.reg_delta.predict(X)
        p_3way = self.clf_3way.predict_proba(X)
        res = {
            "prob_help": p_help,
            "prob_hurt": p_hurt,
            "pred_delta": p_delta,
            "pred_delta_f1": p_delta,
            "probs_3way": p_3way,
        }
        if hasattr(self.reg_delta_ll, "is_fitted_") or hasattr(self.reg_delta_ll, "_is_fitted"):
            res["pred_delta_ll"] = self.reg_delta_ll.predict(X)
            res["pred_delta_gold_logprob"] = res["pred_delta_ll"]
        return res


# ---------------------------------------------------------------------------
# U1 & U2: Cross-Encoder Architectures
# ---------------------------------------------------------------------------
class CrossEncoderUtilityModel(nn.Module):
    """Pretrained Cross-Encoder with 4 multitask utility prediction heads."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        freeze_backbone: bool = False,
    ):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_name)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        hidden_size = self.backbone.config.hidden_size
        self.heads = MultitaskUtilityHeads(hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.backbone(**kwargs)
        cls_rep = outputs.last_hidden_state[:, 0, :]
        preds = self.heads(cls_rep)
        preds["embedding"] = cls_rep
        return preds


def format_u1_text(query: str, candidate: str) -> str:
    """Format input for U1 (Pre-generation query/passage cross-encoder)."""
    return f"[QUESTION] {query} [CANDIDATE] {candidate}"


def format_u2_text(query: str, draft: str, candidate: str) -> str:
    """Format input for U2 (Draft-conditioned cross-encoder)."""
    return f"[QUESTION] {query} [DRAFT ANSWER] {draft} [CANDIDATE EVIDENCE] {candidate}"


# ---------------------------------------------------------------------------
# U4: Full Hybrid Model (U2 Embedding + All Numeric Features)
# ---------------------------------------------------------------------------
class FullHybridUtilityModelU4(nn.Module):
    """U4: Fuses U2 CrossEncoder representation + numeric features via MLP to 4 heads."""

    def __init__(
        self,
        cross_encoder_dim: Optional[int] = None,
        numeric_dim: Optional[int] = None,
        hidden_dim: int = 128,
        ce_dim: Optional[int] = None,
        num_numeric_features: Optional[int] = None,
    ):
        super().__init__()
        c_dim = cross_encoder_dim if cross_encoder_dim is not None else (ce_dim if ce_dim is not None else 384)
        n_dim = numeric_dim if numeric_dim is not None else (num_numeric_features if num_numeric_features is not None else 20)
        total_in = c_dim + n_dim

        self.fusion = nn.Sequential(
            nn.Linear(total_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.heads = MultitaskUtilityHeads(hidden_dim)

    def forward(
        self,
        ce_embedding: torch.Tensor,
        numeric_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        fused = torch.cat([ce_embedding, numeric_features], dim=-1)
        feat = self.fusion(fused)
        return self.heads(feat)


# Aliases
TabularUtilityModel_U0 = UtilityModelU0
PassageCrossEncoder_U1 = CrossEncoderUtilityModel
DraftConditionedCrossEncoder_U2 = CrossEncoderUtilityModel
FullHybridPassageUtilityModel_U4 = FullHybridUtilityModelU4
