"""W8-F: Query-Conditioned Latent Adapter (QCLA).

Architectures:
- QCLA-0: Identity pass-through (original SFR vector e_D)
- QCLA-1: Query-conditioned bottleneck residual adapter:
    x = [e_D, e_q, e_D * e_q, |e_D - e_q|] in R^(4 * 4096)
    g = sigmoid(MLP_g(x))
    z_D = LayerNorm(e_D + g * MLP_f(x)) in R^4096
- QCLA-2: Query + Draft-conditioned residual adapter:
    x = [e_D, e_q, e_y, e_D * e_q, e_D * e_y] in R^(5 * 4096)
    z_D = LayerNorm(e_D + g * MLP_f(x)) in R^4096
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BaseLatentAdapter(nn.Module):
    """Abstract base class for QCLA latent adapters."""

    def forward(
        self,
        e_d: torch.Tensor,
        e_q: Optional[torch.Tensor] = None,
        e_y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class QCLA0_Baseline(BaseLatentAdapter):
    """QCLA-0: Identity pass-through (original SFR vector e_D)."""

    def forward(
        self,
        e_d: torch.Tensor,
        e_q: Optional[torch.Tensor] = None,
        e_y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return e_d


class QCLA1_ResidualAdapter(BaseLatentAdapter):
    """QCLA-1: Query-conditioned bottleneck residual adapter."""

    def __init__(
        self,
        dim: int = 4096,
        hidden_dim: int = 2048,
    ):
        super().__init__()
        self.dim = dim
        in_dim = dim * 4
        # Feature transformation branch
        self.mlp_f = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        # Gating branch
        self.gate_mlp = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Linear(512, dim),
            nn.Sigmoid(),
        )
        self.layer_norm = nn.LayerNorm(dim)

        # Initialize gate to small values for near-identity initial behavior
        nn.init.constant_(self.gate_mlp[-2].bias, -2.0)

    def forward(
        self,
        e_d: torch.Tensor,
        e_q: Optional[torch.Tensor] = None,
        e_y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        orig_dim = e_d.dim()
        if e_d.dim() == 1:
            e_d = e_d.unsqueeze(0)

        if e_q is None:
            e_q = torch.zeros_like(e_d)
        elif e_q.dim() == 1:
            e_q = e_q.unsqueeze(0)

        if e_q.shape[0] != e_d.shape[0]:
            e_q = e_q.expand(e_d.shape[0], -1)

        mult = e_d * e_q
        diff = torch.abs(e_d - e_q)
        x = torch.cat([e_d, e_q, mult, diff], dim=-1)

        f_x = self.mlp_f(x)
        g_x = self.gate_mlp(x)

        z = e_d + g_x * f_x
        out = self.layer_norm(z)

        return out.squeeze(0) if orig_dim == 1 else out


class QCLA2_DraftConditionedAdapter(BaseLatentAdapter):
    """QCLA-2: Query + Draft conditioned bottleneck residual adapter."""

    def __init__(
        self,
        dim: int = 4096,
        hidden_dim: int = 2048,
    ):
        super().__init__()
        self.dim = dim
        in_dim = dim * 5
        self.mlp_f = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Linear(512, dim),
            nn.Sigmoid(),
        )
        self.layer_norm = nn.LayerNorm(dim)
        nn.init.constant_(self.gate_mlp[-2].bias, -2.0)

    def forward(
        self,
        e_d: torch.Tensor,
        e_q: Optional[torch.Tensor] = None,
        e_y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        orig_dim = e_d.dim()
        if e_d.dim() == 1:
            e_d = e_d.unsqueeze(0)

        if e_q is None:
            e_q = torch.zeros_like(e_d)
        elif e_q.dim() == 1:
            e_q = e_q.unsqueeze(0)

        if e_y is None:
            e_y = e_q
        elif e_y.dim() == 1:
            e_y = e_y.unsqueeze(0)

        if e_q.shape[0] != e_d.shape[0]:
            e_q = e_q.expand(e_d.shape[0], -1)
        if e_y.shape[0] != e_d.shape[0]:
            e_y = e_y.expand(e_d.shape[0], -1)

        mult_q = e_d * e_q
        mult_y = e_d * e_y
        x = torch.cat([e_d, e_q, e_y, mult_q, mult_y], dim=-1)

        f_x = self.mlp_f(x)
        g_x = self.gate_mlp(x)

        z = e_d + g_x * f_x
        out = self.layer_norm(z)

        return out.squeeze(0) if orig_dim == 1 else out


# Aliases for backward compatibility
QueryConditionedLatentAdapter = QCLA1_ResidualAdapter
QueryDraftLatentAdapter = QCLA2_DraftConditionedAdapter


def build_qcla_adapter(
    variant: str = "qcla_1",
    dim: int = 4096,
    hidden_dim: int = 2048,
) -> BaseLatentAdapter:
    """Build QCLA adapter by name."""
    v = variant.lower().strip()
    if v in ("qcla_0", "qcla0", "baseline"):
        return QCLA0_Baseline()
    elif v in ("qcla_1", "qcla1"):
        return QCLA1_ResidualAdapter(dim=dim, hidden_dim=hidden_dim)
    elif v in ("qcla_2", "qcla2"):
        return QCLA2_DraftConditionedAdapter(dim=dim, hidden_dim=hidden_dim)
    else:
        raise ValueError(f"Unknown QCLA adapter variant: {variant}")


def apply_qcla_to_compressed_embeds(
    adapter: Optional[BaseLatentAdapter],
    passage_embeds: torch.Tensor,
    query_embed: torch.Tensor,
    draft_embed: Optional[torch.Tensor] = None,
    variant: str = "qcla_1",
) -> torch.Tensor:
    """Transform compressed embeddings with the selected QCLA adapter."""
    if variant in ("qcla_0", "baseline") or adapter is None:
        return passage_embeds

    return adapter(passage_embeds, e_q=query_embed, e_y=draft_embed)
