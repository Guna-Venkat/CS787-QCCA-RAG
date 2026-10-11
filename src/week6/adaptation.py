"""Module for Week 6 Semantic Capacity and Encoder Adaptation Study.

Implements:
- Local architecture audit of sentence-transformers/all-MiniLM-L6-v2
- Precise layer freezing/unfreezing for Capacity Ladder:
    C0: Frozen MiniLM (0 trainable encoder layers, W5-B reference)
    C1: Last 1 Transformer block trainable (layer 5 unfrozen, embeddings & 0..4 frozen)
    C2: Last 2 Transformer blocks trainable (layers 4, 5 unfrozen, embeddings & 0..3 frozen)
- Strict layer assertions preventing accidental full-model unfreezing
- Parameter group construction with differential learning rates (encoder_lr vs head_lr)
- AdaptedContentAwareAllocator model class retaining W5-B corrected aggregator
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel

from src.week5.models import (
    ContentAwareAllocator,
    PassageTransformerAggregator,
    mean_pooling,
)

logger = logging.getLogger(__name__)

MINILM_NUM_LAYERS: int = 6
MINILM_LAYER_PARAMS: int = 1774464
MINILM_EMBEDDINGS_PARAMS: int = 11918592
MINILM_TOTAL_PARAMS: int = 22713216


def audit_minilm_architecture(encoder: PreTrainedModel) -> Dict[str, Any]:
    """Inspect and report the exact architecture of sentence-transformers/all-MiniLM-L6-v2.

    Returns:
        Dictionary containing layer counts, parameter paths, and sub-module dimensions.
    """
    assert hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"), (
        "Encoder does not follow standard BERT architecture with encoder.layer"
    )
    num_layers = len(encoder.encoder.layer)
    assert num_layers == MINILM_NUM_LAYERS, (
        f"Expected {MINILM_NUM_LAYERS} Transformer layers, found {num_layers}"
    )

    has_pooler = hasattr(encoder, "pooler") and encoder.pooler is not None

    param_counts: Dict[str, int] = {}
    total_encoder_params = 0
    for name, param in encoder.named_parameters():
        param_counts[name] = param.numel()
        total_encoder_params += param.numel()

    layer_5_modules = [
        name for name in param_counts if name.startswith("encoder.layer.5.")
    ]
    layer_4_modules = [
        name for name in param_counts if name.startswith("encoder.layer.4.")
    ]

    return {
        "num_transformer_layers": num_layers,
        "has_pooler": has_pooler,
        "total_encoder_params": total_encoder_params,
        "layer_5_param_count": sum(param_counts[m] for m in layer_5_modules),
        "layer_4_param_count": sum(param_counts[m] for m in layer_4_modules),
        "embeddings_param_count": sum(
            p for m, p in param_counts.items() if m.startswith("embeddings.")
        ),
        "layer_5_module_names": layer_5_modules,
        "layer_4_module_names": layer_4_modules,
    }


def configure_encoder_adaptation(
    encoder: PreTrainedModel,
    condition: str,
) -> Dict[str, Any]:
    """Configure parameter freezing/unfreezing for capacity conditions C0, C1, C2.

    Guarantees:
    - Embeddings are ALWAYS frozen.
    - Pooler (if present) is ALWAYS frozen.
    - C0: All encoder parameters frozen (0 trainable).
    - C1: Exactly layer 5 is unfrozen (1,774,464 trainable). Layers 0..4 frozen.
    - C2: Exactly layers 4 and 5 are unfrozen (3,548,928 trainable). Layers 0..3 frozen.

    Args:
        encoder: PreTrainedModel instance (BertModel)
        condition: "C0", "C1", or "C2"

    Returns:
        Summary dict of parameter counts.
    """
    assert condition in ["C0", "C1", "C2"], f"Invalid capacity condition: {condition}"

    # Step 1: Freeze entire encoder first
    for param in encoder.parameters():
        param.requires_grad = False

    # Step 2: Unfreeze selected layers according to condition
    if condition == "C0":
        pass  # Everything remains frozen

    elif condition == "C1":
        # Unfreeze final Transformer block only (layer 5)
        for param in encoder.encoder.layer[5].parameters():
            param.requires_grad = True

    elif condition == "C2":
        # Unfreeze final 2 Transformer blocks (layers 4 and 5)
        for param in encoder.encoder.layer[4].parameters():
            param.requires_grad = True
        for param in encoder.encoder.layer[5].parameters():
            param.requires_grad = True

    # Step 3: Strict Auditing Assertions
    # 3.1 Embeddings must remain frozen
    for name, p in encoder.embeddings.named_parameters():
        assert not p.requires_grad, f"Embedding parameter {name} was accidentally unfrozen!"

    # 3.2 Pooler must remain frozen
    if hasattr(encoder, "pooler") and encoder.pooler is not None:
        for name, p in encoder.pooler.named_parameters():
            assert not p.requires_grad, f"Pooler parameter {name} was accidentally unfrozen!"

    # 3.3 Check layer-by-layer trainability
    for i in range(MINILM_NUM_LAYERS):
        layer_params = list(encoder.encoder.layer[i].parameters())
        if condition == "C0":
            assert all(not p.requires_grad for p in layer_params), f"Layer {i} unexpectedly trainable in C0!"
        elif condition == "C1":
            if i == 5:
                assert all(p.requires_grad for p in layer_params), "Layer 5 parameters must all be trainable in C1!"
            else:
                assert all(not p.requires_grad for p in layer_params), f"Layer {i} unexpectedly trainable in C1!"
        elif condition == "C2":
            if i in [4, 5]:
                assert all(p.requires_grad for p in layer_params), f"Layer {i} parameters must all be trainable in C2!"
            else:
                assert all(not p.requires_grad for p in layer_params), f"Layer {i} unexpectedly trainable in C2!"

    trainable_encoder = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    frozen_encoder = sum(p.numel() for p in encoder.parameters() if not p.requires_grad)
    total_encoder = sum(p.numel() for p in encoder.parameters())

    if condition == "C0":
        assert trainable_encoder == 0, f"Expected 0 trainable encoder params in C0, got {trainable_encoder}"
    elif condition == "C1":
        assert trainable_encoder == MINILM_LAYER_PARAMS, (
            f"Expected {MINILM_LAYER_PARAMS} trainable encoder params in C1, got {trainable_encoder}"
        )
    elif condition == "C2":
        assert trainable_encoder == 2 * MINILM_LAYER_PARAMS, (
            f"Expected {2 * MINILM_LAYER_PARAMS} trainable encoder params in C2, got {trainable_encoder}"
        )

    return {
        "condition": condition,
        "total_encoder_params": total_encoder,
        "trainable_encoder_params": trainable_encoder,
        "frozen_encoder_params": frozen_encoder,
        "trainable_encoder_pct": round(trainable_encoder / total_encoder * 100.0, 2),
    }


def build_optimizer_param_groups(
    model: nn.Module,
    encoder_lr: float = 2.0e-5,
    head_lr: float = 2.0e-4,
    weight_decay: float = 0.01,
) -> List[Dict[str, Any]]:
    """Build differential learning rate parameter groups for AdamW.

    Separates:
    - Trainable encoder parameters (encoder_lr)
    - Trainable aggregator and head parameters (head_lr)

    Guarantees:
    - No parameter appears in multiple groups
    - Frozen parameters are excluded completely
    - Every trainable parameter belongs to exactly one group
    """
    encoder_trainable_params: List[nn.Parameter] = []
    head_aggregator_trainable_params: List[nn.Parameter] = []

    encoder_param_ids: Set[int] = set()
    if hasattr(model, "encoder"):
        encoder_param_ids = {id(p) for p in model.encoder.parameters()}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in encoder_param_ids:
            encoder_trainable_params.append(param)
        else:
            head_aggregator_trainable_params.append(param)

    # Sanity checks
    all_trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assigned_ids = {id(p) for p in encoder_trainable_params}.union(
        {id(p) for p in head_aggregator_trainable_params}
    )
    assert all_trainable_ids == assigned_ids, "Mismatch in optimizer parameter group assignment!"
    assert len(set(id(p) for p in encoder_trainable_params).intersection(
        set(id(p) for p in head_aggregator_trainable_params)
    )) == 0, "Parameter present in both encoder and head groups!"

    param_groups: List[Dict[str, Any]] = []
    if encoder_trainable_params:
        param_groups.append({
            "params": encoder_trainable_params,
            "lr": encoder_lr,
            "weight_decay": weight_decay,
            "name": "encoder_adapted",
        })

    param_groups.append({
        "params": head_aggregator_trainable_params,
        "lr": head_lr,
        "weight_decay": weight_decay,
        "name": "head_and_aggregator",
    })

    return param_groups


class AdaptedContentAwareAllocator(ContentAwareAllocator):
    """Week 6 Semantic Allocator supporting Capacity Ladder C0, C1, C2.

    Extends ContentAwareAllocator to support:
    - Specific capacity conditions C0, C1, C2
    - Direct backpropagation into unfrozen encoder layers for query-passage pairs
    - Retaining the corrected W5-B rank embedding and LayerNorm aggregator
    - Target formulation T3 (Multi-Label Epsilon Admissibility)
    """

    def __init__(
        self,
        capacity_condition: str = "C1",  # "C0", "C1", "C2"
        base_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        d_model: int = 384,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        assert capacity_condition in ["C0", "C1", "C2"], f"Invalid condition {capacity_condition}"
        self.capacity_condition = capacity_condition

        # Initialize base ContentAwareAllocator with I3 and T3
        # Use 'top_layers' in super().__init__ if C1/C2 so it doesn't default-freeze
        encoder_mode = "top_layers" if capacity_condition in ["C1", "C2"] else "frozen"
        super().__init__(
            base_encoder_name=base_encoder_name,
            condition="I3",
            target_mode="T3",
            encoder_train_mode=encoder_mode,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Apply exact W6 adaptation freezing/unfreezing policy
        self.adaptation_stats = configure_encoder_adaptation(self.encoder, capacity_condition)

    def train(self, mode: bool = True) -> AdaptedContentAwareAllocator:
        super().train(mode)
        # In C0, keep encoder in eval mode to disable dropout
        if self.capacity_condition == "C0":
            self.encoder.eval()
        return self

    def extract_semantic_features(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Extract pooled semantic state from top-10 passages.

        If condition is C1 or C2, gradients flow through the encoder layers.
        If condition is C0, encoder executes under torch.no_grad().
        """
        device = next(self.parameters()).device
        passage_mask = batch.get("passage_mask")
        if passage_mask is not None:
            passage_mask = passage_mask.to(device)

        # If C0 and passage_embeddings already cached, allow using it
        if self.capacity_condition == "C0" and "passage_embeddings" in batch:
            reps = batch["passage_embeddings"].to(device)
        else:
            # Must forward through encoder
            input_ids = batch["passage_input_ids"].to(device)  # (B, 10, L)
            attention_mask = batch["passage_attention_mask"].to(device)  # (B, 10, L)

            B, N, L = input_ids.shape
            input_ids_flat = input_ids.view(B * N, L)
            attention_mask_flat = attention_mask.view(B * N, L)

            if self.capacity_condition == "C0":
                with torch.no_grad():
                    out = self.encoder(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
            else:
                out = self.encoder(input_ids=input_ids_flat, attention_mask=attention_mask_flat)

            reps_flat = mean_pooling(out.last_hidden_state, attention_mask_flat)  # (B*10, D)
            reps = reps_flat.view(B, N, self.d_model)  # (B, 10, D)

        pooled = self.aggregator(reps, passage_mask=passage_mask)  # (B, D)
        return pooled

    def get_parameter_breakdown(self) -> Dict[str, Any]:
        """Compute exact parameter breakdown across encoder, aggregator, and head."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        enc_total = sum(p.numel() for p in self.encoder.parameters())
        enc_trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        agg_params = sum(p.numel() for p in self.aggregator.parameters())
        head_params = sum(p.numel() for p in self.admissibility_head.parameters())

        return {
            "capacity_condition": self.capacity_condition,
            "total_params": total,
            "trainable_params": trainable,
            "trainable_pct": round(trainable / total * 100.0, 2),
            "encoder_total_params": enc_total,
            "encoder_trainable_params": enc_trainable,
            "encoder_frozen_params": enc_total - enc_trainable,
            "aggregator_params": agg_params,
            "admissibility_head_params": head_params,
            "head_and_aggregator_trainable_params": agg_params + head_params,
        }
