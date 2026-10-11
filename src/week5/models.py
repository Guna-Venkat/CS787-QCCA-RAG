"""Model architectures for Week 5 Content-Aware Adaptive Representation Allocation.

Supports:
- Four Information Conditions:
    I1: QUERY_ONLY (q)
    I2: QUERY_PLUS_FEATURES (q + x_W4)
    I3: QUERY_PLUS_PASSAGES (q + D1...D10)
    I4: QUERY_PLUS_PASSAGES_PLUS_FEATURES (q + D1...D10 + x_W4)
- Two Target Formulations:
    T1: 5-class direct budget classification (k in {2, 4, 5, 6, 8})
    T2: Multi-output regret regression [regret_k2, ..., regret_k8] >= 0 via Softplus
- Modest pretrained base encoder (sentence-transformers/all-MiniLM-L6-v2, 22.7M params)
- Configurable encoder_train_mode: 'frozen' (default) or 'top_layers'
- Lightweight 2-layer Transformer Aggregator over top-10 passage representations with rank embeddings
- Feature projection and fusion MLP for W4 tabular features
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, PreTrainedModel

from src.week5.data import K_CLASSES, NUM_CLASSES, CLASS_TO_K


def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Perform attention-weighted mean pooling over token embeddings."""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class PassageTransformerAggregator(nn.Module):
    """Aggregates 10 passage-level representations via learnable rank embeddings and shallow self-attention."""

    def __init__(
        self,
        d_model: int = 384,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_passages: int = 10,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_passages = max_passages

        # Learnable rank embeddings for ranks 1..10
        self.rank_embeddings = nn.Embedding(max_passages, d_model)

        # Shallow Transformer encoder
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

    def forward(
        self,
        passage_reps: torch.Tensor,
        passage_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
            passage_reps: (batch_size, num_passages, d_model)
            passage_mask: (batch_size, num_passages) boolean mask (True = valid, False = pad)
        Returns:
            pooled_semantic_state: (batch_size, d_model)
        """
        B, N, D = passage_reps.shape
        ranks = torch.arange(N, device=passage_reps.device).unsqueeze(0).expand(B, N)
        h = passage_reps + self.rank_embeddings(ranks)

        key_padding_mask = ~passage_mask if passage_mask is not None else None
        h_trans = self.transformer(h, src_key_padding_mask=key_padding_mask)

        # Attention pooling
        q = self.pool_query.expand(B, -1, -1)  # (B, 1, D)
        attn_scores = torch.bmm(q, h_trans.transpose(1, 2)) / (D ** 0.5)  # (B, 1, N)
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(key_padding_mask.unsqueeze(1), -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)
        pooled = torch.bmm(attn_weights, h_trans).squeeze(1)  # (B, D)
        return pooled


class TabularFeatureProjector(nn.Module):
    """Projects tabular W4 features into a compact representation for fusion."""

    def __init__(self, in_features: int = 6, out_dim: int = 192, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContentAwareAllocator(nn.Module):
    """Unified Content-Aware Allocator supporting conditions I1, I2, I3, I4 and targets T1, T2."""

    def __init__(
        self,
        base_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        condition: str = "I3",  # I1, I2, I3, I4
        target_mode: str = "T1",  # T1 (classification), T2 (regret regression), or Both
        encoder_train_mode: str = "frozen",  # 'frozen' or 'top_layers'
        d_model: int = 384,
        feature_dim: int = 6,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert condition in ["I1", "I2", "I3", "I4"], f"Unknown condition: {condition}"
        assert target_mode in ["T1", "T2", "Both"], f"Unknown target_mode: {target_mode}"
        self.condition = condition
        self.target_mode = target_mode
        self.d_model = d_model
        self.encoder_train_mode = encoder_train_mode

        # 1. Base Pretrained Encoder
        self.encoder: PreTrainedModel = AutoModel.from_pretrained(base_encoder_name)
        self._configure_encoder_training(encoder_train_mode)

        # 2. Passage Aggregator (for I3, I4)
        if self.condition in ["I3", "I4"]:
            self.aggregator = PassageTransformerAggregator(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                max_passages=10,
            )
        else:
            self.aggregator = None

        # 3. Tabular Feature Projector (for I2, I4)
        if self.condition in ["I2", "I4"]:
            self.feat_dim = d_model // 2
            self.feat_proj = TabularFeatureProjector(
                in_features=feature_dim,
                out_dim=self.feat_dim,
                dropout=dropout,
            )
            self.hidden_in_dim = d_model + self.feat_dim
        else:
            self.feat_proj = None
            self.hidden_in_dim = d_model

        # 4. Heads
        # T1: 5-class allocation classification
        if target_mode in ["T1", "Both"]:
            self.classifier_head = nn.Sequential(
                nn.Linear(self.hidden_in_dim, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, NUM_CLASSES),
            )
        else:
            self.classifier_head = None

        # T2: Multi-output regret prediction (non-negative via Softplus)
        if target_mode in ["T2", "Both"]:
            self.regret_head = nn.Sequential(
                nn.Linear(self.hidden_in_dim, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, NUM_CLASSES),
                nn.Softplus(),
            )
        else:
            self.regret_head = None

    def _configure_encoder_training(self, mode: str) -> None:
        """Set freeze / top-layer policy on base encoder."""
        if mode == "frozen":
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
        elif mode == "top_layers":
            # Keep all frozen except top layer
            for param in self.encoder.parameters():
                param.requires_grad = False
            # Unfreeze top transformer layer if available
            if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
                for param in self.encoder.encoder.layer[-1].parameters():
                    param.requires_grad = True
        else:
            raise ValueError(f"Unknown encoder_train_mode: {mode}")

    def train(self, mode: bool = True) -> ContentAwareAllocator:
        super().train(mode)
        # If encoder is frozen, keep in eval mode to disable dropout
        if self.encoder_train_mode == "frozen":
            self.encoder.eval()
        return self

    def extract_semantic_features(
        self,
        batch: Dict[str, Any],
    ) -> torch.Tensor:
        """Extract pooled semantic state according to condition I1..I4."""
        device = next(self.parameters()).device

        if self.condition in ["I1", "I2"]:
            # Query-only condition
            input_ids = batch["query_input_ids"].to(device)
            attention_mask = batch["query_attention_mask"].to(device)

            if self.encoder_train_mode == "frozen":
                with torch.no_grad():
                    out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            else:
                out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

            pooled = mean_pooling(out.last_hidden_state, attention_mask)

        elif self.condition in ["I3", "I4"]:
            # Query + Top-10 Passages condition
            input_ids = batch["passage_input_ids"].to(device)  # (B, 10, L)
            attention_mask = batch["passage_attention_mask"].to(device)  # (B, 10, L)
            passage_mask = batch.get("passage_mask")
            if passage_mask is not None:
                passage_mask = passage_mask.to(device)

            B, N, L = input_ids.shape
            input_ids_flat = input_ids.view(B * N, L)
            attention_mask_flat = attention_mask.view(B * N, L)

            if self.encoder_train_mode == "frozen":
                with torch.no_grad():
                    out = self.encoder(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
            else:
                out = self.encoder(input_ids=input_ids_flat, attention_mask=attention_mask_flat)

            reps_flat = mean_pooling(out.last_hidden_state, attention_mask_flat)  # (B*10, D)
            reps = reps_flat.view(B, N, self.d_model)  # (B, 10, D)

            pooled = self.aggregator(reps, passage_mask=passage_mask)  # (B, D)

        # Tabular Feature Fusion (for I2, I4)
        if self.condition in ["I2", "I4"]:
            feats = batch["features"].to(device)
            feat_h = self.feat_proj(feats)
            fused = torch.cat([pooled, feat_h], dim=-1)
            return fused
        else:
            return pooled

    def forward(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """Forward pass returning logits for T1 and/or predicted regrets for T2."""
        state = self.extract_semantic_features(batch)
        out: Dict[str, torch.Tensor] = {}

        if self.classifier_head is not None:
            out["logits"] = self.classifier_head(state)
        if self.regret_head is not None:
            out["pred_regret"] = self.regret_head(state)

        return out

    def predict_k(
        self,
        batch: Dict[str, Any],
        epsilon: float = 0.01,
    ) -> np.ndarray:
        """Inference policy returning selected k in {2, 4, 5, 6, 8} for each example."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(batch)

            if self.target_mode == "T1":
                logits = outputs["logits"]  # (B, 5)
                class_preds = torch.argmax(logits, dim=-1).cpu().numpy()
                k_preds = np.array([CLASS_TO_K[c] for c in class_preds])
                return k_preds

            elif self.target_mode == "T2":
                pred_regret = outputs["pred_regret"].cpu().numpy()  # (B, 5)
                k_preds = []
                for row in pred_regret:
                    # Select smallest k with predicted regret <= epsilon
                    valid_ks = [K_CLASSES[idx] for idx, reg in enumerate(row) if reg <= epsilon + 1e-6]
                    if valid_ks:
                        k_preds.append(min(valid_ks))
                    else:
                        # Fallback to k=8
                        k_preds.append(8)
                return np.array(k_preds)

            else:
                # Default to T1 if both exist
                class_preds = torch.argmax(outputs["logits"], dim=-1).cpu().numpy()
                return np.array([CLASS_TO_K[c] for c in class_preds])

    def get_parameter_counts(self) -> Dict[str, int]:
        """Return total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        return {
            "total_params": total,
            "trainable_params": trainable,
            "frozen_encoder_params": encoder_params if self.encoder_train_mode == "frozen" else 0,
            "trainable_head_and_aggregator_params": trainable,
        }
