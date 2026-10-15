"""Feature Families G & H: Query-Aspect Coverage and Passage Length/Cost for Week 7-C.

Implements:
- Family G: Deterministic query aspect coverage over selected passages
- Family H: Context length, passage tokens, projected token costs, and incremental cost
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if",
    "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or",
    "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should",
    "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
}


def _extract_query_aspects(question: str) -> List[str]:
    """Extract content terms / noun aspect tokens from question text."""
    words = re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", question.lower())
    aspects = [w for w in words if w not in STOPWORDS and len(w) >= 3]
    if not aspects:
        aspects = words if words else ["query"]
    return aspects


def _passage_term_support(aspect: str, passage_text: str) -> float:
    """Measure support of an aspect term in passage text."""
    p_lower = passage_text.lower()
    if not p_lower:
        return 0.0
    if aspect in p_lower:
        return 1.0
    # Substring / partial match
    if any(aspect in token or token in aspect for token in p_lower.split() if len(token) >= 4):
        return 0.5
    return 0.0


def compute_query_coverage_features(
    question: str,
    passages: Sequence[Dict[str, Any]],
    semantic_ranks: Sequence[int],
    k: int,
    budgets: Sequence[int] = (2, 4, 5, 6, 8),
) -> Dict[str, float]:
    """Compute Family G Query-Aspect Coverage features for candidate budget k.

    Formulation:
        For each query aspect a_j:
            coverage(a_j, k) = max_{i in S_k} sim(a_j, D_i)

    Args:
        question: Query string.
        passages: Sequence of 10 passage dicts with 'text'.
        semantic_ranks: Length 10 list of passage indices sorted by semantic score descending.
        k: Candidate budget.
        budgets: Allowed candidate budgets.

    Returns:
        Dictionary of Family G features.
    """
    aspects = _extract_query_aspects(question)
    n_aspects = len(aspects)
    n_passages = len(passages)
    k_clamp = min(k, n_passages)

    sel_indices = [idx for idx in semantic_ranks[:k_clamp] if idx < n_passages]
    sel_texts = [str(passages[idx].get("text", "")) for idx in sel_indices]

    coverages: List[float] = []
    for asp in aspects:
        max_supp = max((_passage_term_support(asp, text) for text in sel_texts), default=0.0)
        coverages.append(max_supp)

    cov_arr = np.array(coverages, dtype=np.float64)
    cov_mean = float(np.mean(cov_arr))
    cov_min = float(np.min(cov_arr))
    cov_p25 = float(np.percentile(cov_arr, 25))
    uncovered_frac = float(np.mean(cov_arr < 0.5))

    # Marginal coverage gain to next candidate budget
    sorted_budgets = sorted(budgets)
    if k in sorted_budgets and sorted_budgets.index(k) < len(sorted_budgets) - 1:
        next_k = sorted_budgets[sorted_budgets.index(k) + 1]
        next_sel_indices = [idx for idx in semantic_ranks[:min(next_k, n_passages)] if idx < n_passages]
        next_sel_texts = [str(passages[idx].get("text", "")) for idx in next_sel_indices]
        next_coverages = [max((_passage_term_support(asp, text) for text in next_sel_texts), default=0.0) for asp in aspects]
        cov_gain_next = float(np.mean(next_coverages) - cov_mean)
    else:
        cov_gain_next = 0.0

    return {
        "cov_k_mean": cov_mean,
        "cov_k_min": cov_min,
        "cov_k_p25": cov_p25,
        "cov_k_uncovered_frac": uncovered_frac,
        "cov_k_marginal_gain_next": cov_gain_next,
    }


def compute_cost_features(
    passages: Sequence[Dict[str, Any]],
    semantic_ranks: Sequence[int],
    k: int,
    budgets: Sequence[int] = (2, 4, 5, 6, 8),
    prompt_overhead_tokens: int = 120,
) -> Dict[str, float]:
    """Compute Family H Length and Cost features for candidate budget k.

    Args:
        passages: Sequence of 10 passage dicts with 'text'.
        semantic_ranks: Length 10 list of passage indices sorted by semantic score descending.
        k: Candidate budget.
        budgets: Allowed candidate budgets.
        prompt_overhead_tokens: Estimated fixed prompt overhead.

    Returns:
        Dictionary of Family H features.
    """
    n_passages = len(passages)
    k_clamp = min(k, n_passages)

    # Word-based token proxy: 1 word ~ 1.3 tokens
    lengths = [len(str(p.get("text", "")).split()) * 1.3 for p in passages]

    sel_indices = [idx for idx in semantic_ranks[:k_clamp] if idx < n_passages]
    tail_indices = [idx for idx in semantic_ranks[k_clamp:] if idx < n_passages]

    sel_lengths = [lengths[idx] for idx in sel_indices] if sel_indices else [0.0]
    tail_lengths = [lengths[idx] for idx in tail_indices] if tail_indices else [0.0]

    cost_tokens = float(np.sum(sel_lengths))
    cost_mean = float(np.mean(sel_lengths))
    cost_max = float(np.max(sel_lengths))
    cost_tail = float(np.sum(tail_lengths))
    projected_tokens = cost_tokens + float(prompt_overhead_tokens)

    # Incremental token cost to next budget
    sorted_budgets = sorted(budgets)
    if k in sorted_budgets and sorted_budgets.index(k) < len(sorted_budgets) - 1:
        next_k = sorted_budgets[sorted_budgets.index(k) + 1]
        next_indices = [idx for idx in semantic_ranks[k_clamp:min(next_k, n_passages)] if idx < n_passages]
        incremental_tokens = float(np.sum([lengths[idx] for idx in next_indices])) if next_indices else 0.0
    else:
        incremental_tokens = 0.0

    return {
        "cost_k_tokens": cost_tokens,
        "cost_k_mean_tokens": cost_mean,
        "cost_k_max_tokens": cost_max,
        "cost_k_tail_tokens": cost_tail,
        "cost_k_full_projected_tokens": projected_tokens,
        "cost_k_incremental_cost_next": incremental_tokens,
        "cost_k": float(k),
        "cost_k_ratio": float(k / 10.0),
    }
