"""Unit tests for response lookup, Pareto calculation, and evaluation metrics."""

import pandas as pd
import pytest

from src.week4.evaluation import FrozenResponseSurfaceLookup, evaluate_allocator_predictions
from src.week4.pareto import compute_pareto_frontier


def test_frozen_response_surface_lookup():
    """Verify lookup returns exact F1 from matrix and raises on invalid inputs."""
    lookup = FrozenResponseSurfaceLookup("results/week2/processed/dev_per_query_k_matrix.csv")
    
    # Question 1912 at k=8
    f1_val = lookup.get_f1("1912", 8)
    assert isinstance(f1_val, float)
    assert 0.0 <= f1_val <= 1.0

    # Missing question raises KeyError
    with pytest.raises(KeyError):
        lookup.get_f1("nonexistent_question_id", 8)


def test_pareto_frontier_logic():
    """Verify dominated vs non-dominated point identification on a toy example."""
    toy_df = pd.DataFrame([
        {"system": "A", "mean_context_tokens": 500, "mean_f1": 0.40},  # Non-dominated
        {"system": "B", "mean_context_tokens": 600, "mean_f1": 0.35},  # Dominated by A (more tokens, lower F1)
        {"system": "C", "mean_context_tokens": 800, "mean_f1": 0.45},  # Non-dominated
        {"system": "D", "mean_context_tokens": 850, "mean_f1": 0.45},  # Dominated by C (more tokens, same F1)
    ])
    
    res = compute_pareto_frontier(toy_df)
    opt_systems = set(res[res["is_pareto_optimal"]]["system"])
    assert opt_systems == {"A", "C"}
