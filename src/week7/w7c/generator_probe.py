"""Optional Model Family 4: Generator-Aware Sufficiency Probe for Week 7-C.

Implements prefill-only hidden state extraction infrastructure from SARA/Mistral.
Disabled by default (ENABLE_GENERATOR_PROBE: false) to prevent expensive prefill sweeps.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class GeneratorSufficiencyProbe:
    """Lightweight probe trained on generator prefill hidden states."""

    def __init__(
        self,
        enabled: bool = False,
        candidate_layers: Sequence[int] = (16, 24, 31),
        device: Optional[str] = None,
    ):
        self.enabled = enabled
        self.candidate_layers = list(candidate_layers)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.probes: Dict[int, LogisticRegression] = {}

    def extract_prefill_hidden_states(
        self,
        model: Any,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        query_token_indices: Sequence[int],
    ) -> Dict[int, np.ndarray]:
        """Run prefill forward pass (NO autoregressive generation) with output_hidden_states=True.

        Extracts mean query-token hidden representation at candidate layers.
        """
        if not self.enabled or model is None:
            return {}

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                output_hidden_states=True,
            )

        all_hidden = outputs.hidden_states  # Tuple of (B, T, D)
        layer_reps: Dict[int, np.ndarray] = {}

        q_idx_tensor = torch.tensor(query_token_indices, dtype=torch.long, device=self.device)

        for l_idx in self.candidate_layers:
            if l_idx < len(all_hidden):
                h = all_hidden[l_idx]  # Shape (B, T, D)
                q_h = h[:, q_idx_tensor, :]  # Shape (B, |q|, D)
                mean_q_h = torch.mean(q_h, dim=1).cpu().numpy()  # Shape (B, D)
                layer_reps[l_idx] = mean_q_h

        return layer_reps

    def fit_probe(self, X: np.ndarray, y: np.ndarray, layer_idx: int = 24) -> None:
        """Fit lightweight logistic regression probe on hidden states."""
        if not self.enabled:
            return
        clf = LogisticRegression(max_iter=500, random_state=42)
        clf.fit(X, y)
        self.probes[layer_idx] = clf

    def predict_proba(self, X: np.ndarray, layer_idx: int = 24) -> np.ndarray:
        """Predict sufficiency probabilities."""
        if not self.enabled or layer_idx not in self.probes:
            return np.full(len(X), 0.5)
        return self.probes[layer_idx].predict_proba(X)[:, 1]
