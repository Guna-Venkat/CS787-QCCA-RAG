"""Tests for Epsilon Oracle target computation and feasibility."""

import pytest
from src.week3.epsilon_analysis import (
    compute_epsilon_targets_for_query,
    compute_diagnostic_sweep_epsilon_target
)


def test_epsilon_oracle_minimum_feasible_k():
    # Specification test case:
    # reference = 0.80, epsilon = 0.05
    # k2=.60, k4=.72, k5=.76, k6=.77, k8=.80 -> expected k*=5 because .76 >= .80 - .05 = .75
    f1_dict = {
        2: 0.60,
        4: 0.72,
        5: 0.76,
        6: 0.77,
        8: 0.80
    }
    k_alloc = [2, 4, 5, 6, 8]
    sel_k, sel_f1, regret = compute_epsilon_targets_for_query(f1_dict, epsilon=0.05, k_alloc=k_alloc)
    assert sel_k == 5
    assert pytest.approx(sel_f1, 1e-4) == 0.76
    assert pytest.approx(regret, 1e-4) == 0.04


def test_epsilon_oracle_zero_epsilon():
    # At epsilon=0.00, it selects the smallest k achieving exact maximum
    f1_dict = {
        2: 0.50,
        4: 0.80,
        5: 0.80,
        6: 0.70,
        8: 0.80
    }
    sel_k, sel_f1, regret = compute_epsilon_targets_for_query(f1_dict, epsilon=0.00, k_alloc=[2, 4, 5, 6, 8])
    assert sel_k == 4
    assert pytest.approx(sel_f1, 1e-4) == 0.80
    assert pytest.approx(regret, 1e-4) == 0.0


def test_diagnostic_sweep_infeasibility_detection():
    # Diagnostic test: When k=10 outperforms all K_alloc beyond epsilon
    f1_dict = {
        0: 0.10,
        2: 0.20,
        4: 0.30,
        5: 0.30,
        6: 0.30,
        8: 0.30,
        10: 0.80  # k=10 achieves 0.80, max(K_alloc)=0.30
    }
    # With eps=0.05, threshold is 0.80 - 0.05 = 0.75 > 0.30 -> Infeasible!
    sel_k, ref_sweep, regret, is_feasible = compute_diagnostic_sweep_epsilon_target(
        f1_dict, epsilon=0.05, k_alloc=[2, 4, 5, 6, 8], k_sweep=[0, 2, 4, 5, 6, 8, 10]
    )
    assert not is_feasible
    assert sel_k is None
    assert pytest.approx(ref_sweep, 1e-4) == 0.80
