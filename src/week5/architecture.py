"""Modular PyTorch semantic encoder architecture for Week 5 Content-Aware Allocation.

Implements Candidate 1:
- Shared Passage Cross-Encoder representation h_i for i in 1..10
- Learnable passage-rank positional embeddings r_i for ranks 1..10
- Lightweight 2-layer Transformer Aggregator over the 10 passage representations
- Pre-generation tabular feature projection and fusion
- Modular heads:
    T1: Direct 5-class allocation classifier (k in {2, 4, 5, 6, 8})
    T2: Multi-output regret predictor [regret_k2, ..., regret_k8]
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

K_CLASSES = [2, 4, 5, 6, 8]
NUM_CLASSES = len(K_CLASSES)


class PassageTransformerAggregator(nn.Module):
    """Aggregates 10 passage-level representations via shallow self-attention."""

    def __init__(
        self,
        d_model: int = 384,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_passages: int = 10,
        feature_dim: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_passages = max_passages
        self.feature_dim = feature_dim

        # Learnable rank embeddings for ranks 1..10
        self.rank_embeddings = nn.Embedding(max_passages, d_model)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Attention pooling query vector
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model))

        # Optional tabular feature projection
        if feature_dim > 0:
            self.feat_proj = nn.Sequential(
                nn.Linear(feature_dim, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.out_dim = d_model + (d_model // 2)
        else:
            self.feat_proj = None
            self.out_dim = d_model

    def forward(
        self,
        passage_reps: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        passage_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:

        passage_reps: (batch_size, num_passages, d_model)
        features: (batch_size, feature_dim) or None
        passage_mask: (batch_size, num_passages) boolean mask where True = keep,
        False = pad
        Returns:
            aggregated_context: (batch_size, out_dim)
        """
        B, N, D = passage_reps.shape
        ranks = torch.arange(N, device=passage_reps.device).unsqueeze(0).expand(B, N)
        h = passage_reps + self.rank_embeddings(ranks)

        # Key padding mask: True indicates padding position to ignore in Transformer
        key_padding_mask = ~passage_mask if passage_mask is not None else None
        h_trans = self.transformer(h, src_key_padding_mask=key_padding_mask)

        # Attention pooling
        # query: (1, 1, D) -> (B, 1, D)
        q = self.pool_query.expand(B, -1, -1)
        # scores: (B, 1, N)
        attn_scores = torch.bmm(q, h_trans.transpose(1, 2)) / (D ** 0.5)
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(key_padding_mask.unsqueeze(1), -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)
        # pooled: (B, D)
        pooled = torch.bmm(attn_weights, h_trans).squeeze(1)

        # Tabular feature fusion if present
        if self.feat_proj is not None and features is not None:
            feat_h = self.feat_proj(features)
            out = torch.cat([pooled, feat_h], dim=-1)
        else:
            out = pooled

        return out


class ContentAwareAllocatorModel(nn.Module):
    """End-to-end model supporting both Direct k Classification and Regret Prediction."""

    def __init__(
        self,
        d_model: int = 384,
        feature_dim: int = 0,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.aggregator = PassageTransformerAggregator(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_passages=10,
            feature_dim=feature_dim,
        )

        in_dim = self.aggregator.out_dim

        # Target 1: Direct k Classification Head (5 logits)
        self.classifier_head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, NUM_CLASSES),
        )

        # Target 2: Regret Prediction Head (5 continuous regret predictions)
        self.regret_head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, NUM_CLASSES),
            nn.Softplus(),  # Regrets are non-negative
        )

    def forward(
        self,
        passage_reps: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        passage_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns:

        logits_k: (batch_size, 5) class logits for k in {2, 4, 5, 6, 8}
        pred_regret: (batch_size, 5) predicted regrets for k in {2, 4, 5, 6, 8}
        """
        ctx_emb = self.aggregator(passage_reps, features=features, passage_mask=passage_mask)
        logits_k = self.classifier_head(ctx_emb)
        pred_regret = self.regret_head(ctx_emb)
        return logits_k, pred_regret
