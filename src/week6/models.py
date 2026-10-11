"""Neural architectures and scorers for Week 6 Passage Routing.

Implements:
- R1: Pretrained Semantic Scorer (frozen sentence-transformers/all-MiniLM-L6-v2 bi-encoder cosine similarity).
- R2: Learned Passage Importance Router (frozen MiniLM encoder + weak rank embedding + lightweight MLP).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pool token embeddings with attention mask."""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


class PretrainedSemanticScorer(nn.Module):
    """R1: Zero-shot pretrained semantic scorer using frozen sentence-transformers MiniLM.

    Computes query-conditioned semantic relevance scores for retrieved passages.
    By default uses bi-encoder cosine similarity cos(embed(query), embed(passage_i)).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Freeze encoder completely
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        if device:
            self.to(device)

    @torch.no_grad()
    def score_query_passages(
        self,
        query: str,
        passages: list[str],
        device: Optional[str] = None,
    ) -> torch.Tensor:
        """Compute cosine similarities between query and candidate passages.

        Args:
            query: Query string.
            passages: List of 10 passage strings.

        Returns:
            scores: Tensor of shape (len(passages),) containing cosine similarity scores.
        """
        dev = device or next(self.parameters()).device
        texts = [query] + passages
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(dev)

        outputs = self.encoder(**encoded)
        embeddings = mean_pooling(outputs.last_hidden_state, encoded["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)

        q_embed = embeddings[0:1]  # (1, D)
        p_embeds = embeddings[1:]  # (N, D)

        scores = torch.mm(p_embeds, q_embed.t()).squeeze(-1)  # (N,)
        return scores


class LearnedPassageRouter(nn.Module):
    """R2: Lightweight Learned Passage Importance Router.

    Architecture:
    - Frozen transformer encoder (sentence-transformers/all-MiniLM-L6-v2, 384 hidden dim).
    - Query-passage pair pooling.
    - Optional small rank embedding (rank 0..9 -> d_rank dim) with small initialization.
    - Lightweight MLP classification head: Linear -> GELU -> Dropout -> Linear -> scalar logit.
    - Outputs unnormalized logits of shape (B, 10).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_rank_feature: bool = True,
        rank_embedding_dim: int = 16,
    ):
        super().__init__()
        self.model_name = model_name
        self.use_rank_feature = use_rank_feature
        self.rank_embedding_dim = rank_embedding_dim

        # Frozen base encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        encoder_dim = self.encoder.config.hidden_size  # 384

        if use_rank_feature:
            self.rank_embed = nn.Embedding(10, rank_embedding_dim)
            # Small initialization so rank prior does not drown out semantic signals
            nn.init.normal_(self.rank_embed.weight, mean=0.0, std=0.02)
            head_in_dim = encoder_dim + rank_embedding_dim
        else:
            self.rank_embed = None
            head_in_dim = encoder_dim

        # Small MLP scoring head
        self.mlp = nn.Sequential(
            nn.Linear(head_in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rank_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass computing passage relevance logits.

        Args:
            input_ids: Tensor of shape (B, 10, S)
            attention_mask: Tensor of shape (B, 10, S)
            rank_indices: Optional tensor of shape (B, 10) with integers 0..9

        Returns:
            logits: Tensor of shape (B, 10)
        """
        B, N, S = input_ids.shape
        flat_input_ids = input_ids.reshape(B * N, S)
        flat_attention_mask = attention_mask.reshape(B * N, S)

        # Frozen encoder forward pass
        with torch.no_grad():
            outputs = self.encoder(
                input_ids=flat_input_ids,
                attention_mask=flat_attention_mask,
            )
            pooled = mean_pooling(outputs.last_hidden_state, flat_attention_mask)  # (B*N, 384)

        if self.use_rank_feature:
            if rank_indices is None:
                rank_indices = torch.arange(N, device=input_ids.device).unsqueeze(0).expand(B, N)
            flat_ranks = rank_indices.reshape(B * N)
            rank_repr = self.rank_embed(flat_ranks)  # (B*N, d_rank)
            features = torch.cat([pooled, rank_repr], dim=-1)  # (B*N, 384 + d_rank)
        else:
            features = pooled

        logits_flat = self.mlp(features)  # (B*N, 1)
        logits = logits_flat.reshape(B, N)  # (B, 10)
        return logits

    def get_parameter_counts(self) -> Dict[str, int]:
        """Return parameter count breakdown."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": frozen_params,
        }
