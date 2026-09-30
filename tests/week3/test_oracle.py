"""Tests for Quality Oracle and Headroom computation."""

import pytest
import pandas as pd
import numpy as np
from src.week3.oracle_analysis import compute_quality_oracle, compute_oracle_headroom


def test_quality_oracle_argmax_and_tie_breaking():
    # Test case from spec: k2=.5, k4=.7, k5=.7, k6=.6, k8=.4 -> k*=4
    df_pivot = pd.DataFrame([{
        2: 0.5,
        4: 0.7,
        5: 0.7,
        6: 0.6,
        8: 0.4
    }], index=["q1"])
    
    oracle_df = compute_quality_oracle(df_pivot, k_alloc=[2, 4, 5, 6, 8], static_k=8)
    assert len(oracle_df) == 1
    assert oracle_df.iloc[0]["best_k_qual"] == 4
    assert pytest.approx(oracle_df.iloc[0]["best_f1_qual"], 1e-4) == 0.7
    assert pytest.approx(oracle_df.iloc[0]["best_static_f1"], 1e-4) == 0.4
    assert pytest.approx(oracle_df.iloc[0]["oracle_gain"], 1e-4) == 0.3


def test_quality_oracle_all_zero_tie():
    # When all F1 are 0.0, smallest k (2) must be selected
    df_pivot = pd.DataFrame([{
        2: 0.0,
        4: 0.0,
        5: 0.0,
        6: 0.0,
        8: 0.0
    }], index=["q2"])
    
    oracle_df = compute_quality_oracle(df_pivot, k_alloc=[2, 4, 5, 6, 8], static_k=8)
    assert oracle_df.iloc[0]["best_k_qual"] == 2
    assert oracle_df.iloc[0]["best_f1_qual"] == 0.0
    assert oracle_df.iloc[0]["oracle_gain"] == 0.0


def test_oracle_headroom_math():
    oracle_df = pd.DataFrame([
        {"best_f1_qual": 0.8, "best_static_f1": 0.6},
        {"best_f1_qual": 0.4, "best_static_f1": 0.4}
    ])
    headroom = compute_oracle_headroom(oracle_df)
    assert pytest.approx(headroom["mean_oracle_f1"], 1e-4) == 0.6
    assert pytest.approx(headroom["mean_best_static_f1"], 1e-4) == 0.5
    assert pytest.approx(headroom["absolute_headroom"], 1e-4) == 0.1
    assert pytest.approx(headroom["relative_headroom_pct"], 1e-2) == 20.0
