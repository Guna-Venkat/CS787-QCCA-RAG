"""Target computations and mathematical definitions for Week 7-C3 Anchor-and-Deviate.

Defines:
- Oracle epsilon-admissible set: A_epsilon(q) = {k : F(q, k) >= F*(q) - epsilon}.
- Safe-downshift target: y_down(q) = 1[F(q, 2) >= F*(q) - epsilon].
- Selective-upshift target: y_up(q) = 1[max_{k in {5,6,8}} F(q, k) - F(q, 4) > epsilon].
- Marginal gain delta targets: Delta_k(q) = F(q, k) - F(q, 4) for k in {5, 6, 8}.
- Optional 4->6 gain target: y_4to6(q) = 1[F(q, 6) - F(q, 4) > epsilon].
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_oracle_admissible_set(
    f1_by_k: Dict[int, float],
    epsilon: float = 0.01,
) -> Set[int]:
    """Compute the epsilon-admissible candidate budget set: A_eps = {k : F(q, k) >= F*(q) - eps}."""
    max_f1 = max(f1_by_k.values())
    return {k for k, val in f1_by_k.items() if val >= max_f1 - epsilon}


def compute_minimum_cost_oracle_k(
    f1_by_k: Dict[int, float],
    tokens_by_k: Optional[Dict[int, float]] = None,
    epsilon: float = 0.01,
    budgets: Sequence[int] = (2, 4, 5, 6, 8),
) -> int:
    """Compute the minimum-cost epsilon-oracle budget k_eps*."""
    adm = compute_oracle_admissible_set(f1_by_k, epsilon=epsilon)
    if not adm:
        return max(budgets)

    if tokens_by_k:
        return min(adm, key=lambda k: (tokens_by_k[k], k))
    else:
        return min(adm)


def compute_safe_downshift_target(
    f1_by_k: Dict[int, float],
    epsilon: float = 0.01,
) -> int:
    """Compute safe-downshift label y_down(q).

    y_down(q) = 1 if F(q, 2) >= max_k F(q, k) - epsilon else 0.
    Meaning: k=2 is epsilon-admissible and can safely replace default k=4.
    """
    max_f1 = max(f1_by_k.values())
    return 1 if f1_by_k[2] >= max_f1 - epsilon else 0


def compute_selective_upshift_target(
    f1_by_k: Dict[int, float],
    epsilon: float = 0.01,
    anchor_k: int = 4,
    upshift_budgets: Sequence[int] = (5, 6, 8),
) -> Tuple[int, float]:
    """Compute selective-upshift label y_up(q) and gain_up(q).

    F_up*(q) = max_{k in {5,6,8}} F(q, k)
    Gain_up(q) = F_up*(q) - F(q, 4)
    y_up(q) = 1 if Gain_up(q) > epsilon else 0.
    Meaning: At least one larger budget provides meaningful improvement over k=4.
    """
    f_up_star = max(f1_by_k[k] for k in upshift_budgets)
    gain_up = f_up_star - f1_by_k[anchor_k]
    y_up = 1 if gain_up > epsilon else 0
    return y_up, gain_up


def compute_delta_regression_targets(
    f1_by_k: Dict[int, float],
    anchor_k: int = 4,
    upshift_budgets: Sequence[int] = (5, 6, 8),
) -> Dict[int, float]:
    """Compute marginal gain deltas relative to anchor k=4: Delta_k = F(q, k) - F(q, 4)."""
    f1_anchor = f1_by_k[anchor_k]
    return {k: f1_by_k[k] - f1_anchor for k in upshift_budgets}


def compute_4to6_gain_target(
    f1_by_k: Dict[int, float],
    epsilon: float = 0.01,
    anchor_k: int = 4,
) -> Tuple[int, float]:
    """Compute 4->6 marginal gain target for W7-D handoff."""
    gain_46 = f1_by_k[6] - f1_by_k[anchor_k]
    y_46 = 1 if gain_46 > epsilon else 0
    return y_46, gain_46
