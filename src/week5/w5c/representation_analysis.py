"""Representation shift and geometry diagnostics for Week 6.

Evaluates whether task-specific adaptation reshapes semantic representations:
- Cosine similarity between pretrained (frozen) vs adapted passage representations
- Mean representation shift norm ||h_adapted - h_pretrained||_2
- Within-query passage representation variance
- Across-query representation variance
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.week5.models import mean_pooling
from src.week5.w5c.adaptation import AdaptedContentAwareAllocator

logger = logging.getLogger(__name__)


def compute_representation_shift(
    pretrained_model: AdaptedContentAwareAllocator,
    adapted_model: AdaptedContentAwareAllocator,
    val_loader: DataLoader,
    max_batches: int = 5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """Compute representation shift metrics between pretrained and adapted models on an INTERNAL-VAL subset.

    Args:
        pretrained_model: C0 or unadapted model instance
        adapted_model: C1 or C2 adapted model instance
        val_loader: DataLoader yielding INTERNAL-VAL batches
        max_batches: Maximum number of batches to evaluate for efficiency
        device: Device to run computation on

    Returns:
        Dictionary containing cosine similarity, shift norm, and within/across variance metrics.
    """
    pretrained_model = pretrained_model.to(device).eval()
    adapted_model = adapted_model.to(device).eval()

    all_cos_sims: List[float] = []
    all_shift_norms: List[float] = []
    all_within_vars_pre: List[float] = []
    all_within_vars_adapt: List[float] = []
    query_means_pre: List[np.ndarray] = []
    query_means_adapt: List[np.ndarray] = []

    batch_count = 0
    with torch.no_grad():
        for batch in val_loader:
            if batch_count >= max_batches:
                break

            input_ids = batch["passage_input_ids"].to(device)  # (B, 10, L)
            attention_mask = batch["passage_attention_mask"].to(device)  # (B, 10, L)

            B, N, L = input_ids.shape
            input_ids_flat = input_ids.view(B * N, L)
            attention_mask_flat = attention_mask.view(B * N, L)

            # Pretrained representations
            out_pre = pretrained_model.encoder(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
            h_pre_flat = mean_pooling(out_pre.last_hidden_state, attention_mask_flat)  # (B*10, D)
            h_pre = h_pre_flat.view(B, N, -1)  # (B, 10, D)

            # Adapted representations
            out_adapt = adapted_model.encoder(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
            h_adapt_flat = mean_pooling(out_adapt.last_hidden_state, attention_mask_flat)  # (B*10, D)
            h_adapt = h_adapt_flat.view(B, N, -1)  # (B, 10, D)

            # 1. Cosine similarity
            cos_sim = F.cosine_similarity(h_pre_flat, h_adapt_flat, dim=-1)  # (B*10,)
            all_cos_sims.extend(cos_sim.cpu().numpy().tolist())

            # 2. Shift norm
            shift_norm = torch.norm(h_adapt_flat - h_pre_flat, p=2, dim=-1)  # (B*10,)
            all_shift_norms.extend(shift_norm.cpu().numpy().tolist())

            # 3. Within-query variance (across the 10 passages of each query)
            # Var across dimension 1, summed across embedding dimension D
            w_var_pre = torch.var(h_pre, dim=1, unbiased=False).sum(dim=-1)  # (B,)
            w_var_adapt = torch.var(h_adapt, dim=1, unbiased=False).sum(dim=-1)  # (B,)
            all_within_vars_pre.extend(w_var_pre.cpu().numpy().tolist())
            all_within_vars_adapt.extend(w_var_adapt.cpu().numpy().tolist())

            # 4. For across-query variance: collect query mean passage representation
            q_mean_pre = h_pre.mean(dim=1).cpu().numpy()  # (B, D)
            q_mean_adapt = h_adapt.mean(dim=1).cpu().numpy()  # (B, D)
            query_means_pre.append(q_mean_pre)
            query_means_adapt.append(q_mean_adapt)

            batch_count += 1

    # Across-query variance
    all_q_pre = np.concatenate(query_means_pre, axis=0)  # (N_queries, D)
    all_q_adapt = np.concatenate(query_means_adapt, axis=0)  # (N_queries, D)
    across_var_pre = float(np.var(all_q_pre, axis=0).sum())
    across_var_adapt = float(np.var(all_q_adapt, axis=0).sum())

    results = {
        "mean_cosine_similarity": round(float(np.mean(all_cos_sims)), 4),
        "std_cosine_similarity": round(float(np.std(all_cos_sims)), 4),
        "min_cosine_similarity": round(float(np.min(all_cos_sims)), 4),
        "mean_representation_shift_norm": round(float(np.mean(all_shift_norms)), 4),
        "std_representation_shift_norm": round(float(np.std(all_shift_norms)), 4),
        "within_query_passage_var_pretrained": round(float(np.mean(all_within_vars_pre)), 4),
        "within_query_passage_var_adapted": round(float(np.mean(all_within_vars_adapt)), 4),
        "across_query_var_pretrained": round(across_var_pre, 4),
        "across_query_var_adapted": round(across_var_adapt, 4),
        "evaluated_pairs_count": len(all_cos_sims),
    }

    return results
