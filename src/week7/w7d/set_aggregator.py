"""Set-level Transformer aggregator and attention pooling module for D1.

Aggregates 10 joint query-passage representations:
    tilde_H in R^{B x 10 x d_in} -> Transformer Encoder -> H' in R^{B x 10 x d_model}
    r_q = AttentionPool(H') in R^{B x d_model} (retrieval-state vector)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SetLevelAggregator(nn.Module):
    """Aggregates a set of 10 passage representations into a single query retrieval-state vector."""

    def __init__(
        self,
        d_in: int = 384 + 5,      # Cross-encoder hidden dim + 5 metadata features
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model

        # Linear projection into model dimension
        self.input_proj = nn.Linear(d_in, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Multi-head attention pooling query
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=dropout)

    def forward(
        self,
        passage_reps: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            passage_reps: Tensor of shape (batch_size, 10, d_in)
            mask: Optional boolean mask (batch_size, 10) where True indicates padding

        Returns:
            (r_q, H_prime):
                r_q: (batch_size, d_model) retrieval-state vector
                H_prime: (batch_size, 10, d_model) contextualized passage representations
        """
        B, N, _ = passage_reps.shape

        # Project and normalize
        h = self.layer_norm(self.input_proj(passage_reps))  # (B, N, d_model)

        # Contextualize across passages
        h_prime = self.transformer_encoder(h, src_key_padding_mask=mask)  # (B, N, d_model)

        # Attention pool to query vector
        q = self.pool_query.expand(B, -1, -1)  # (B, 1, d_model)
        pooled, _ = self.pool_attn(query=q, key=h_prime, value=h_prime, key_padding_mask=mask)
        r_q = pooled.squeeze(1)  # (B, d_model)

        return r_q, h_prime
