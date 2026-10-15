"""Multi-task prediction heads and multi-task loss formulation for Week 7-D.

Provides:
- MultiTaskUtilityHeads module:
    - Head A: Safe downshift (y_down) logit
    - Head B: Selective upshift (y_up) logit
    - Head C: Marginal gains (Delta_5, Delta_6, Delta_8)
    - Head D: Epsilon admissibility vector ([z_2, z_4, z_5, z_6, z_8]) logits
    - Head E (optional): Auxiliary oracle k class logits
    - Passage Utility Head (optional for W7-E): u_i per passage
- Multi-task loss computation (BCE, SmoothL1, CrossEntropy).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MultiTaskUtilityHeads(nn.Module):
    """Multi-task prediction heads operating on retrieval-state vector r_q."""

    def __init__(
        self,
        d_in: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        enable_aux_k: bool = True,
        enable_passage_utility: bool = False,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.enable_aux_k = enable_aux_k
        self.enable_passage_utility = enable_passage_utility

        # Shared projection trunk
        self.trunk = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Head A: Downshift logit (y_down)
        self.head_down = nn.Linear(hidden_dim, 1)

        # Head B: Upshift logit (y_up)
        self.head_up = nn.Linear(hidden_dim, 1)

        # Head C: Marginal gains (Delta_5, Delta_6, Delta_8)
        self.head_delta = nn.Linear(hidden_dim, 3)

        # Head D: Admissibility vector logits (z_2, z_4, z_5, z_6, z_8)
        self.head_admissibility = nn.Linear(hidden_dim, 5)

        # Head E: Auxiliary oracle k class logits (5 classes)
        if self.enable_aux_k:
            self.head_aux_k = nn.Linear(hidden_dim, 5)

        # Optional Passage Utility head (for W7-E tier routing)
        if self.enable_passage_utility:
            self.head_passage_utility = nn.Sequential(
                nn.Linear(d_in, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )

    def forward(
        self,
        r_q: torch.Tensor,
        h_passages: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            r_q: (batch_size, d_in) retrieval-state vector
            h_passages: Optional (batch_size, 10, d_in) contextualized passage representations

        Returns:
            Dict of predicted tensors:
                - logit_down: (B, 1)
                - logit_up: (B, 1)
                - pred_deltas: (B, 3)
                - logit_admissibility: (B, 5)
                - logit_aux_k: (B, 5) if enabled
                - pred_passage_utility: (B, 10) if enabled
        """
        feats = self.trunk(r_q)

        outputs: Dict[str, torch.Tensor] = {
            "logit_down": self.head_down(feats),
            "logit_up": self.head_up(feats),
            "pred_deltas": self.head_delta(feats),
            "logit_admissibility": self.head_admissibility(feats),
        }

        if self.enable_aux_k:
            outputs["logit_aux_k"] = self.head_aux_k(feats)

        if self.enable_passage_utility and h_passages is not None:
            # (B, 10, 1) -> (B, 10)
            u = self.head_passage_utility(h_passages).squeeze(-1)
            outputs["pred_passage_utility"] = u

        return outputs

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute multi-task loss.

        Loss = lambda_d L_down + lambda_u L_up + lambda_delta L_delta + lambda_z L_adm (+ lambda_k L_k)
        """
        weights = loss_weights or {
            "lambda_down": 1.0,
            "lambda_up": 1.0,
            "lambda_delta": 1.0,
            "lambda_admissibility": 1.0,
            "lambda_aux_k": 0.5,
        }

        # Loss A: Downshift BCE
        y_down_true = targets["target_y_down"].view(-1, 1).float()
        loss_down = F.binary_cross_entropy_with_logits(predictions["logit_down"], y_down_true)

        # Loss B: Upshift BCE
        y_up_true = targets["target_y_up"].view(-1, 1).float()
        loss_up = F.binary_cross_entropy_with_logits(predictions["logit_up"], y_up_true)

        # Loss C: Delta SmoothL1 / Huber
        deltas_true = targets["target_deltas"].float()  # (B, 3)
        loss_delta = F.smooth_l1_loss(predictions["pred_deltas"], deltas_true)

        # Loss D: Multi-label Admissibility BCE
        z_true = targets["target_admissibility"].float()  # (B, 5)
        loss_adm = F.binary_cross_entropy_with_logits(predictions["logit_admissibility"], z_true)

        total_loss = (
            weights.get("lambda_down", 1.0) * loss_down
            + weights.get("lambda_up", 1.0) * loss_up
            + weights.get("lambda_delta", 1.0) * loss_delta
            + weights.get("lambda_admissibility", 1.0) * loss_adm
        )

        loss_breakdown: Dict[str, float] = {
            "loss_down": float(loss_down.item()),
            "loss_up": float(loss_up.item()),
            "loss_delta": float(loss_delta.item()),
            "loss_admissibility": float(loss_adm.item()),
        }

        # Optional Loss E: Auxiliary k Cross-Entropy
        if self.enable_aux_k and "target_aux_k" in targets and "logit_aux_k" in predictions:
            k_true = targets["target_aux_k"].long().view(-1)
            loss_aux_k = F.cross_entropy(predictions["logit_aux_k"], k_true)
            total_loss = total_loss + weights.get("lambda_aux_k", 0.5) * loss_aux_k
            loss_breakdown["loss_aux_k"] = float(loss_aux_k.item())

        loss_breakdown["total_loss"] = float(total_loss.item())
        return total_loss, loss_breakdown
