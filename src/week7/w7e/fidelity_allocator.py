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


def assert_multifidelity_row_invariants(
    row: Dict[str, Any],
    n_full_expected: int = 4,
    n_medium_expected: int = 2,
    n_heavy_expected: int = 4,
    hybrid_medium: bool = False,
) -> None:
    """Strictly assert representation invariants on a generated record:
    1. full_ids has exactly n_full_expected unique IDs.
    2. medium_ids has exactly n_medium_expected unique IDs.
    3. heavy_ids has exactly n_heavy_expected unique IDs.
    4. union(full, medium, heavy) == {0..9} with no duplicates.
    5. In Hybrid mode: n_compress == n_heavy + n_medium.
    6. In Extract-only mode: n_compress == n_heavy.
    """
    full = row["full_indices"]
    medium = row["medium_indices"]
    heavy = row["heavy_indices"]
    compressed = row.get("compressed_indices", [])

    assert len(full) == n_full_expected, f"Expected {n_full_expected} FULL, got {len(full)}"
    assert len(medium) == n_medium_expected, f"Expected {n_medium_expected} MEDIUM, got {len(medium)}"
    assert len(heavy) == n_heavy_expected, f"Expected {n_heavy_expected} HEAVY, got {len(heavy)}"

    validate_multifidelity_assignment(full, medium, heavy, total_passages=10)

    if hybrid_medium:
        assert len(compressed) == n_heavy_expected + n_medium_expected, (
            f"Hybrid MEDIUM invariant failed: expected {n_heavy_expected + n_medium_expected} compressed, got {len(compressed)}"
        )
        assert set(compressed) == (set(heavy) | set(medium)), "Compressed set must be union of heavy and medium in Hybrid mode"
    else:
        assert len(compressed) == n_heavy_expected, (
            f"Extract-only invariant failed: expected {n_heavy_expected} compressed, got {len(compressed)}"
        )
        assert set(compressed) == set(heavy), "Compressed set must equal heavy set in Extract-only mode"


def get_semantic_rank_order(
    semantic_scores: Optional[Sequence[float]] = None,
    semantic_ranking: Optional[Sequence[int]] = None,
    n_passages: int = 10,
) -> List[int]:
    """Return 0-indexed passage indices sorted by descending semantic relevance.

    If neither semantic_scores nor semantic_ranking is provided, falls back to
    BM25 order (0..n_passages-1).
    """
    if semantic_ranking is not None:
        assert len(semantic_ranking) == n_passages, f"Expected {n_passages} indices, got {len(semantic_ranking)}"
        assert set(semantic_ranking) == set(range(n_passages)), "semantic_ranking must be a permutation of 0..n_passages-1"
        return list(semantic_ranking)
    if semantic_scores is not None:
        if hasattr(semantic_scores, "detach"):
            s_arr = semantic_scores.detach().cpu().numpy()[:n_passages]
        else:
            s_arr = np.asarray(semantic_scores[:n_passages], dtype=np.float32)
        assert len(s_arr) >= n_passages, f"Expected at least {n_passages} scores, got {len(s_arr)}"
        return [int(idx) for idx in np.argsort(-s_arr)]
    return list(range(n_passages))


def allocate_e0_k4(
    n_passages: int = 10,
    semantic_scores: Optional[Sequence[float]] = None,
    semantic_ranking: Optional[Sequence[int]] = None,
) -> Tuple[List[int], List[int], List[int]]:
    """E0 Baseline: 4F + 0M + 6H (Static semantic k4)."""
    order = get_semantic_rank_order(semantic_scores, semantic_ranking, n_passages)
    full = sorted(order[:4])
    medium = []
    heavy = sorted(order[4:])
    validate_multifidelity_assignment(full, medium, heavy, total_passages=n_passages)
    return full, medium, heavy


def allocate_e1_primary(
    n_passages: int = 10,
    semantic_scores: Optional[Sequence[float]] = None,
    semantic_ranking: Optional[Sequence[int]] = None,
) -> Tuple[List[int], List[int], List[int]]:
    """E1 Primary Policy: 4F + 2M + 4H (Semantic ranks 1-4 FULL, 5-6 MEDIUM, 7-10 HEAVY)."""
    order = get_semantic_rank_order(semantic_scores, semantic_ranking, n_passages)
    full = sorted(order[:4])
    medium = sorted(order[4:6])
    heavy = sorted(order[6:])
    validate_multifidelity_assignment(full, medium, heavy, total_passages=n_passages)
    return full, medium, heavy


def allocate_e2_k6(
    n_passages: int = 10,
    semantic_scores: Optional[Sequence[float]] = None,
    semantic_ranking: Optional[Sequence[int]] = None,
) -> Tuple[List[int], List[int], List[int]]:
    """E2 Baseline: 6F + 0M + 4H (Static semantic k6)."""
    order = get_semantic_rank_order(semantic_scores, semantic_ranking, n_passages)
    full = sorted(order[:6])
    medium = []
    heavy = sorted(order[6:])
    validate_multifidelity_assignment(full, medium, heavy, total_passages=n_passages)
    return full, medium, heavy


def allocate_e3_compact(
    n_passages: int = 10,
    semantic_scores: Optional[Sequence[float]] = None,
    semantic_ranking: Optional[Sequence[int]] = None,
) -> Tuple[List[int], List[int], List[int]]:
    """E3 Secondary Policy: 2F + 2M + 6H (Semantic ranks 1-2 FULL, 3-4 MEDIUM, 5-10 HEAVY)."""
    order = get_semantic_rank_order(semantic_scores, semantic_ranking, n_passages)
    full = sorted(order[:2])
    medium = sorted(order[2:4])
    heavy = sorted(order[4:])
    validate_multifidelity_assignment(full, medium, heavy, total_passages=n_passages)
    return full, medium, heavy


def allocate_e4_extended(
    n_passages: int = 10,
    semantic_scores: Optional[Sequence[float]] = None,
    semantic_ranking: Optional[Sequence[int]] = None,
) -> Tuple[List[int], List[int], List[int]]:
    """E4 Secondary Policy: 4F + 4M + 2H (Semantic ranks 1-4 FULL, 5-8 MEDIUM, 9-10 HEAVY)."""
    order = get_semantic_rank_order(semantic_scores, semantic_ranking, n_passages)
    full = sorted(order[:4])
    medium = sorted(order[4:8])
    heavy = sorted(order[8:])
    validate_multifidelity_assignment(full, medium, heavy, total_passages=n_passages)
    return full, medium, heavy


def allocate_e1x_cross_encoder(
    ce_scores: Sequence[float],
    semantic_scores: Optional[Sequence[float]] = None,
    semantic_ranking: Optional[Sequence[int]] = None,
    n_full: int = 4,
    n_medium: int = 2,
    n_passages: int = 10,
) -> Tuple[List[int], List[int], List[int]]:
    """E1-X Policy: 4F + 2M + 4H with Cross-Encoder informed marginal selection.

    Maintains exact counts:
    - Semantic ranks 1..n_full remain FULL.
    - Among candidate semantic ranks n_full..n_passages-1, selects the n_medium passages
      with highest cross-encoder relevance scores as MEDIUM.
    - The remaining candidate passages become HEAVY.
    """
    assert len(ce_scores) >= n_passages, f"Expected at least {n_passages} cross-encoder scores, got {len(ce_scores)}"

    order = get_semantic_rank_order(semantic_scores, semantic_ranking, n_passages)
    full = sorted(order[:n_full])
    candidate_indices = order[n_full:]

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
