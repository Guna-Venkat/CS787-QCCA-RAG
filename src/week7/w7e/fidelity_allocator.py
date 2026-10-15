"""Multi-fidelity allocation policies and invariant verification for W7-E.

Implements:
- Strict representation invariants:
    - Every passage gets exactly ONE representation (FULL, MEDIUM, or HEAVY).
    - Union of all indices is exactly {0..9}.
    - No overlaps between fidelity sets.
- Fixed policies:
    - E0: 4F + 0M + 6H (Static semantic k4 baseline)
    - E1: 4F + 2M + 4H (PRIMARY multi-fidelity policy)
    - E2: 6F + 0M + 4H (Static semantic k6 baseline)
    - E3: 2F + 2M + 6H (Compact multi-fidelity)
    - E4: 4F + 4M + 2H (Extended medium coverage)
- Cross-Encoder informed policies:
    - E1-X: 4F + 2M + 4H (Ranks 1-4 FULL; among ranks 5-10, top 2 by cross-encoder become MEDIUM, 4 become HEAVY)
    - E_CE_allocation: Top 4 CE -> FULL, next 2 CE -> MEDIUM, remaining 4 -> HEAVY
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def validate_multifidelity_assignment(
    full_indices: Sequence[int],
    medium_indices: Sequence[int],
    heavy_indices: Sequence[int],
    total_passages: int = 10,
) -> None:
    """Strictly assert Week 7-E multi-fidelity invariants:
    1. Every passage index in [0, total_passages - 1].
    2. No duplicates within any category.
    3. Mutually exclusive categories (no overlap between full, medium, heavy).
    4. Exact partition of all total_passages (full + medium + heavy == total_passages).
    5. Union is exactly set(range(total_passages)).
    """
    f_set = set(full_indices)
    m_set = set(medium_indices)
    h_set = set(heavy_indices)

    if len(f_set) != len(full_indices):
        raise ValueError(f"Invariant violation: duplicate full indices: {full_indices}")
    if len(m_set) != len(medium_indices):
        raise ValueError(f"Invariant violation: duplicate medium indices: {medium_indices}")
    if len(h_set) != len(heavy_indices):
        raise ValueError(f"Invariant violation: duplicate heavy indices: {heavy_indices}")

    # Check overlaps
    fm = f_set.intersection(m_set)
    if fm:
        raise ValueError(f"Invariant violation: overlap between FULL and MEDIUM: {fm}")
    fh = f_set.intersection(h_set)
    if fh:
        raise ValueError(f"Invariant violation: overlap between FULL and HEAVY: {fh}")
    mh = m_set.intersection(h_set)
    if mh:
        raise ValueError(f"Invariant violation: overlap between MEDIUM and HEAVY: {mh}")

    total_assigned = len(f_set) + len(m_set) + len(h_set)
    if total_assigned != total_passages:
        raise ValueError(f"Invariant violation: missing or extra passages, expected {total_passages} total passages, got {total_assigned}")

    union_set = f_set | m_set | h_set
    expected_set = set(range(total_passages))
    if union_set != expected_set:
        raise ValueError(f"Invariant violation: partition does not span 0..{total_passages - 1}: missing {expected_set - union_set}")


def allocate_e0_k4(n_passages: int = 10) -> Tuple[List[int], List[int], List[int]]:
    """E0 Baseline: 4F + 0M + 6H (Static semantic k4)."""
    full = list(range(4))
    medium = []
    heavy = list(range(4, n_passages))
    validate_multifidelity_assignment(full, medium, heavy, total_passages=n_passages)
    return full, medium, heavy


def allocate_e1_primary(n_passages: int = 10) -> Tuple[List[int], List[int], List[int]]:
    """E1 Primary Policy: 4F + 2M + 4H (Semantic ranks 1-4 FULL, 5-6 MEDIUM, 7-10 HEAVY)."""
    full = list(range(4))
    medium = list(range(4, 6))
    heavy = list(range(6, n_passages))
    validate_multifidelity_assignment(full, medium, heavy, total_passages=n_passages)
    return full, medium, heavy


def allocate_e2_k6(n_passages: int = 10) -> Tuple[List[int], List[int], List[int]]:
    """E2 Baseline: 6F + 0M + 4H (Static semantic k6)."""
    full = list(range(6))
    medium = []
    heavy = list(range(6, n_passages))
    validate_multifidelity_assignment(full, medium, heavy, total_passages=n_passages)
    return full, medium, heavy


def allocate_e3_compact(n_passages: int = 10) -> Tuple[List[int], List[int], List[int]]:
    """E3 Secondary Policy: 2F + 2M + 6H (Ranks 1-2 FULL, 3-4 MEDIUM, 5-10 HEAVY)."""
    full = list(range(2))
    medium = list(range(2, 4))
    heavy = list(range(4, n_passages))
    validate_multifidelity_assignment(full, medium, heavy, total_passages=n_passages)
    return full, medium, heavy


def allocate_e4_extended(n_passages: int = 10) -> Tuple[List[int], List[int], List[int]]:
    """E4 Secondary Policy: 4F + 4M + 2H (Ranks 1-4 FULL, 5-8 MEDIUM, 9-10 HEAVY)."""
    full = list(range(4))
    medium = list(range(4, 8))
    heavy = list(range(8, n_passages))
    validate_multifidelity_assignment(full, medium, heavy, total_passages=n_passages)
    return full, medium, heavy


def allocate_e1x_cross_encoder(
    ce_scores: Sequence[float],
    n_full: int = 4,
    n_medium: int = 2,
    n_passages: int = 10,
) -> Tuple[List[int], List[int], List[int]]:
    """E1-X Policy: 4F + 2M + 4H with Cross-Encoder informed marginal selection.

    Maintains exact counts:
    - Ranks 1..n_full remain FULL (0..3).
    - Among candidate ranks 4..n_passages-1, selects the n_medium passages with highest
      cross-encoder relevance scores as MEDIUM.
    - The remaining candidate passages become HEAVY.
    """
    assert len(ce_scores) >= n_passages, f"Expected at least {n_passages} cross-encoder scores, got {len(ce_scores)}"

    full = list(range(n_full))
    candidate_indices = list(range(n_full, n_passages))

    # Rank candidates by descending CE score
    candidate_scores = [(idx, float(ce_scores[idx])) for idx in candidate_indices]
    candidate_scores.sort(key=lambda x: x[1], reverse=True)

    # Top n_medium become MEDIUM
    medium = sorted([idx for idx, _ in candidate_scores[:n_medium]])
    # Remaining candidates become HEAVY
    heavy = sorted([idx for idx, _ in candidate_scores[n_medium:]])

    validate_multifidelity_assignment(full, medium, heavy, total_passages=n_passages)
    return full, medium, heavy


def allocate_ce_full_ranking(
    ce_scores: Sequence[float],
    n_full: int = 4,
    n_medium: int = 2,
    n_passages: int = 10,
) -> Tuple[List[int], List[int], List[int]]:
    """Optional CE Full Allocation:
    Ranks all 10 passages by cross-encoder score:
    - Top n_full -> FULL
    - Next n_medium -> MEDIUM
    - Remaining -> HEAVY
    """
    assert len(ce_scores) >= n_passages
    indexed_scores = [(idx, float(ce_scores[idx])) for idx in range(n_passages)]
    indexed_scores.sort(key=lambda x: x[1], reverse=True)

    full = sorted([idx for idx, _ in indexed_scores[:n_full]])
    medium = sorted([idx for idx, _ in indexed_scores[n_full : n_full + n_medium]])
    heavy = sorted([idx for idx, _ in indexed_scores[n_full + n_medium :]])

    validate_multifidelity_assignment(full, medium, heavy, total_passages=n_passages)
    return full, medium, heavy
