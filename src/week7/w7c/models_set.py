"""Optional Model Family 3: Retrieval-Set Encoder for Week 7-C.

Processes 10 retrieved passages jointly through a shallow set-level encoder
combined with query and budget embeddings.
Disabled by default (ENABLE_SET_ENCODER: false).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PassageSetTransformer(nn.Module):
    """Shallow transformer / DeepSets encoder over 10 retrieved passage vectors."""

    def __init__(
        self,
        passage_feat_dim: int = 6,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.passage_proj = nn.Linear(passage_feat_dim, d_model)
        self.budget_embed = nn.Embedding(11, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output heads
        self.cls_head = nn.Linear(d_model, 1)
        self.reg_head = nn.Linear(d_model, 1)

    def forward(
        self,
        passage_feats: torch.Tensor,  # Shape (B, 10, passage_feat_dim)
        budget_k: torch.Tensor,       # Shape (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = passage_feats.size(0)
        p_emb = self.passage_proj(passage_feats)  # Shape (B, 10, d_model)
        k_emb = self.budget_embed(budget_k).unsqueeze(1)  # Shape (B, 1, d_model)

        # Concatenate budget token as special prefix token
        seq = torch.cat([k_emb, p_emb], dim=1)  # Shape (B, 11, d_model)
        out = self.transformer(seq)

        # Pool over prefix budget token
        pooled = out[:, 0, :]
        cls_logit = self.cls_head(pooled).squeeze(-1)
        reg_pred = self.reg_head(pooled).squeeze(-1)
        return cls_logit, reg_pred


class RetrievalSetEncoderWrapper:
    """Wrapper exposing disabled / fallback interface for optional set encoder."""

    def __init__(self, enabled: bool = False, device: Optional[str] = None):
        self.enabled = enabled
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def fit(self, *args, **kwargs) -> "RetrievalSetEncoderWrapper":
        if not self.enabled:
            logger.info("RetrievalSetEncoder is disabled. Skipping fit.")
            return self
        # Implementation when enabled
        return self

    def predict_proba(self, n_samples: int) -> np.ndarray:
        return np.full(n_samples, 0.5)
