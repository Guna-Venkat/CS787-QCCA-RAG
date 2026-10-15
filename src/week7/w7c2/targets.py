"""Target definitions and lexicographic preference ranking construction for Week 7-C2.

Implements:
- Lexicographic Quality-Cost Total Ordering:
    Quality first (difference > epsilon_pref), Cost breaks near-ties.
- Graded relevance integer mapping for learning-to-rank: grades in {0, 1, 2, 3, 4}.
- M1 Direct regression targets (F1 and regret).
- M3 Stagewise continuation targets.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compare_budgets_lexicographic(
    k_a: int,
    k_b: int,
    f1_a: float,
    f1_b: float,
    tokens_a: float,
    tokens_b: float,
    epsilon_pref: float = 0.01,
) -> int:
    """Compare two candidate budgets using lexicographic quality-cost preference.

    Returns:
        -1 if k_a is preferred over k_b (k_a > k_b in utility)
        +1 if k_b is preferred over k_a
         0 if identical
    """
    if k_a == k_b:
        return 0

    diff_f1 = f1_a - f1_b
    if diff_f1 > epsilon_pref:
        # k_a has meaningfully higher quality
        return -1
    elif diff_f1 < -epsilon_pref:
        # k_b has meaningfully higher quality
        return 1
    else:
        # Near-tie in quality (|diff| <= epsilon_pref): prefer cheaper budget
        if tokens_a < tokens_b:
            return -1
        elif tokens_a > tokens_b:
            return 1
        else:
            # Secondary tie-breaker: smaller k index
            return -1 if k_a < k_b else 1


def compute_lexicographic_relevance_grades(
    f1_by_k: Dict[int, float],
    tokens_by_k: Dict[int, float],
    epsilon_pref: float = 0.01,
) -> Dict[int, int]:
    """Compute graded relevance scores in {0, 1, 2, 3, 4} for candidate budgets.

    The 5 candidate budgets are sorted under the lexicographic preference ordering.
    Best action receives grade 4; worst action receives grade 0.

    Args:
        f1_by_k: Dictionary mapping budget k -> downstream F1.
        tokens_by_k: Dictionary mapping budget k -> context tokens.
        epsilon_pref: Quality equivalence margin (default 0.01).

    Returns:
        Dictionary mapping budget k -> integer relevance grade in [0, 4].
    """
    budgets = list(f1_by_k.keys())

    def cmp_func(k_a: int, k_b: int) -> int:
        return compare_budgets_lexicographic(
            k_a=k_a,
            k_b=k_b,
            f1_a=f1_by_k[k_a],
            f1_b=f1_by_k[k_b],
            tokens_a=tokens_by_k[k_a],
            tokens_b=tokens_by_k[k_b],
            epsilon_pref=epsilon_pref,
        )

    # Sort ascending in order of preference (best to worst)
    sorted_budgets = sorted(budgets, key=functools.cmp_to_key(cmp_func))

    # Assign integer grades: best gets len-1 (e.g. 4), worst gets 0
    n = len(sorted_budgets)
    grades_by_k: Dict[int, int] = {}
    for rank_idx, k_val in enumerate(sorted_budgets):
        grades_by_k[k_val] = (n - 1) - rank_idx

    return grades_by_k


def compute_continuation_target(
    f1_by_k: Dict[int, float],
    current_stage_k: int,
    epsilon: float = 0.01,
) -> int:
    """Compute binary continuation target for stage current_stage_k.

    Target = 1 if any future budget j > current_stage_k provides a meaningful quality improvement
    exceeding epsilon: max_{j > k} F(q, j) - F(q, k) > epsilon.
    Otherwise 0 (STOP).
    """
    current_f1 = f1_by_k[current_stage_k]
    future_f1s = [f1_by_k[j] for j in sorted(f1_by_k.keys()) if j > current_stage_k]
    if not future_f1s:
        return 0  # Terminal stage (e.g. k=8)
    future_best = max(future_f1s)
    return 1 if (future_best - current_f1) > epsilon else 0
