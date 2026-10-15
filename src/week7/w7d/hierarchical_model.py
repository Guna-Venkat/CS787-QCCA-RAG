"""Hierarchical Cross-Encoder Router (D1) for Week 7-D.

Architecture:
1. CrossEncoder: (q, D_i) -> h_i in R^d (for i=1..10)
2. Metadata concatenation: tilde_h_i = [h_i ; ranks ; scores ; length]
3. SetLevelAggregator: tilde_H in R^{B x 10 x d_in} -> Transformer Encoder -> r_q in R^{B x d_model}
4. MultiTaskUtilityHeads: r_q -> y_down, y_up, deltas, admissibility_vector, aux_k.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from src.week7.w7d.multitask_heads import MultiTaskUtilityHeads
from src.week7.w7d.set_aggregator import SetLevelAggregator

logger = logging.getLogger(__name__)


class HierarchicalCrossEncoderRouter(nn.Module):
    """Hierarchical Cross-Encoder model predicting representation fidelity requirements."""

    def __init__(
        self,
        encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        freeze_encoder: bool = True,
        unfreeze_last_n_layers: int = 0,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        meta_dim: int = 5,
        enable_aux_k: bool = True,
        enable_passage_utility: bool = False,
    ) -> None:
        super().__init__()
        self.encoder_name = encoder_name
        self.freeze_encoder = freeze_encoder
        self.unfreeze_last_n_layers = unfreeze_last_n_layers
        self.meta_dim = meta_dim

        # Load HuggingFace backbone
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.d_encoder = self.encoder.config.hidden_size  # Typically 384 for MiniLM

        # Apply freezing strategy
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

            if unfreeze_last_n_layers > 0 and hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
                # Unfreeze last n layers (D1-B)
                for layer in self.encoder.encoder.layer[-unfreeze_last_n_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                logger.info("Unfroze last %d layers of cross-encoder.", unfreeze_last_n_layers)

        # Set-level aggregator
        d_in = self.d_encoder + meta_dim
        self.aggregator = SetLevelAggregator(
            d_in=d_in,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Multi-task output heads
        self.heads = MultiTaskUtilityHeads(
            d_in=d_model,
            hidden_dim=d_model // 2,
            dropout=dropout,
            enable_aux_k=enable_aux_k,
            enable_passage_utility=enable_passage_utility,
        )

    def extract_passage_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a batch of (query, passage) pairs into CLS token representations.

        Args:
            input_ids: (B * 10, seq_len)
            attention_mask: (B * 10, seq_len)

        Returns:
            (B * 10, d_encoder)
        """
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**kwargs)
        # Use [CLS] representation (first token)
        cls_rep = outputs.last_hidden_state[:, 0, :]
        return cls_rep

    def forward_from_cached_embeddings(
        self,
        passage_reps: torch.Tensor,
        metadata: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass directly from precomputed passage representations.

        Args:
            passage_reps: (B, 10, d_encoder)
            metadata: (B, 10, meta_dim)

        Returns:
            Dict of predicted outputs from multi-task heads.
        """
        # Concat metadata
        tilde_h = torch.cat([passage_reps, metadata], dim=-1)  # (B, 10, d_encoder + meta_dim)

        # Set-level aggregation
        r_q, h_prime = self.aggregator(tilde_h)  # r_q: (B, d_model), h_prime: (B, 10, d_model)

        # Multi-task heads
        predictions = self.heads(r_q, h_prime)
        predictions["r_q"] = r_q
        predictions["h_passages"] = h_prime
        return predictions

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        metadata: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """End-to-end forward pass from raw token IDs.

        Args:
            input_ids: (B, 10, seq_len)
            attention_mask: (B, 10, seq_len)
            metadata: (B, 10, meta_dim)

        Returns:
            Dict of predicted outputs.
        """
        B, N, L = input_ids.shape

        # Flatten to (B * 10, L)
        flat_input_ids = input_ids.view(B * N, L)
        flat_mask = attention_mask.view(B * N, L)
        flat_token_types = token_type_ids.view(B * N, L) if token_type_ids is not None else None

        # Encode passages
        h_flat = self.extract_passage_embeddings(flat_input_ids, flat_mask, flat_token_types)  # (B * 10, d_enc)
        h_passages = h_flat.view(B, N, self.d_encoder)  # (B, 10, d_enc)

        return self.forward_from_cached_embeddings(h_passages, metadata)
