"""Optional Feature Family I: Frozen Cross-Encoder Scores for Week 7-C.

Provides a modular, cached wrapper for an optional frozen query-document cross-encoder.
Disabled by default (ENABLE_CROSS_ENCODER_FEATURES: false) to maintain fast, deterministic execution.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)


class OptionalFrozenCrossEncoderScorer:
    """Lightweight optional query-document cross-encoder scorer.

    Uses a frozen pretrained cross-encoder (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2)
    strictly in inference mode.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        enabled: bool = False,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.enabled = enabled
        self.device = device
        self.model = None

        if self.enabled:
            try:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(model_name, device=device)
                logger.info("Loaded optional frozen cross-encoder: %s", model_name)
            except Exception as e:
                logger.warning("Could not load cross-encoder %s: %s. Disabling.", model_name, e)
                self.enabled = False

    def score_query_passages(
        self,
        query: str,
        passage_texts: Sequence[str],
    ) -> List[float]:
        """Compute cross-encoder scores for 10 passages."""
        if not self.enabled or self.model is None:
            return [0.0] * len(passage_texts)

        pairs = [[query, text] for text in passage_texts]
        scores = self.model.predict(pairs)
        return [float(s) for s in scores]


def compute_cross_encoder_budget_features(
    cross_encoder_scores: Sequence[float],
    semantic_ranks: Sequence[int],
    k: int,
    enabled: bool = False,
) -> Dict[str, float]:
    """Compute Family I candidate-budget k features from cross-encoder scores.

    If disabled, returns default zeroes with standard schema.
    """
    if not enabled or not cross_encoder_scores or len(cross_encoder_scores) == 0:
        return {
            "ce_top1": 0.0,
            "ce_k_mean": 0.0,
            "ce_k_min": 0.0,
            "ce_k_boundary_score": 0.0,
            "ce_k_boundary_gap": 0.0,
            "ce_k_tail_max": 0.0,
            "ce_k_selected_vs_tail_gap": 0.0,
        }

    c_arr = np.array(cross_encoder_scores, dtype=np.float64)
    n = len(c_arr)
    k_clamp = min(k, n)

    sel_indices = [idx for idx in semantic_ranks[:k_clamp] if idx < n]
    tail_indices = [idx for idx in semantic_ranks[k_clamp:] if idx < n]

    sel_scores = [c_arr[idx] for idx in sel_indices] if sel_indices else [0.0]
    tail_scores = [c_arr[idx] for idx in tail_indices] if tail_indices else [0.0]

    ce_top1 = float(np.max(c_arr))
    ce_k_mean = float(np.mean(sel_scores))
    ce_k_min = float(np.min(sel_scores))
    boundary_score = float(sel_scores[-1]) if sel_scores else 0.0
    tail_max = float(np.max(tail_scores)) if tail_scores else boundary_score
    boundary_gap = boundary_score - tail_max
    selected_vs_tail_gap = ce_k_mean - float(np.mean(tail_scores)) if tail_scores else 0.0

    return {
        "ce_top1": ce_top1,
        "ce_k_mean": ce_k_mean,
        "ce_k_min": ce_k_min,
        "ce_k_boundary_score": boundary_score,
        "ce_k_boundary_gap": boundary_gap,
        "ce_k_tail_max": tail_max,
        "ce_k_selected_vs_tail_gap": selected_vs_tail_gap,
    }
