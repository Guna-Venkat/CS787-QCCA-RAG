"""Unit tests for retrieval feature extractor."""

import pytest
import numpy as np
from src.week2.feature_extractor import compute_retrieval_features


def test_normal_positive_scores():
    scores = [10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.5, 0.4, 0.2, 0.1]
    query = "What datasets are used to evaluate the model?"
    feats = compute_retrieval_features(scores, query=query)
    
    assert 0.0 < feats["rho_1"] < 1.0
    assert 0.0 < feats["delta_12"] < 1.0
    assert 0.0 < feats["entropy_normalized"] <= 1.0
    assert feats["query_token_length"] == 8
    # 8.0 >= 0.8 * 10.0 -> N_high should be 2 (10.0 and 8.0)
    assert feats["n_high"] == 2


def test_equal_scores():
    scores = [5.0] * 10
    feats = compute_retrieval_features(scores)
    
    # 1 / 10 = 0.1
    assert pytest.approx(feats["rho_1"], rel=1e-3) == 0.1
    # Margin should be 0.0
    assert pytest.approx(feats["delta_12"], abs=1e-5) == 0.0
    # Entropy should be maximum (1.0 normalized)
    assert pytest.approx(feats["entropy_normalized"], rel=1e-3) == 1.0
    # All scores equal top-1
    assert feats["n_high"] == 10


def test_all_zero_scores():
    scores = [0.0] * 10
    feats = compute_retrieval_features(scores)
    
    assert feats["rho_1"] == 0.1
    assert feats["delta_12"] == 0.0
    assert feats["entropy_normalized"] == 1.0
    assert feats["n_high"] == 10


def test_negative_scores():
    # Negative scores should be clamped to 0.0 via max(s, 0)
    scores = [-2.0, -5.0, -10.0]
    feats = compute_retrieval_features(scores)
    
    assert feats["rho_1"] == pytest.approx(1.0 / 3.0, rel=1e-3)
    assert feats["delta_12"] == 0.0
    assert feats["entropy_normalized"] == 1.0


def test_one_dominant_score():
    scores = [100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    feats = compute_retrieval_features(scores)
    
    # Concentration should be approximately 1.0
    assert pytest.approx(feats["rho_1"], rel=1e-3) == 1.0
    # Margin should be approximately 1.0
    assert pytest.approx(feats["delta_12"], rel=1e-3) == 1.0
    # Entropy should be 0.0 (minimum uncertainty)
    assert pytest.approx(feats["entropy_normalized"], abs=1e-4) == 0.0
    # Only 1 high-score passage
    assert feats["n_high"] == 1


def test_duplicated_scores():
    scores = [10.0, 10.0, 5.0, 5.0, 1.0]
    feats = compute_retrieval_features(scores)
    
    # Top 2 are equal -> delta_12 is 0.0
    assert feats["delta_12"] == 0.0
    assert feats["n_high"] == 2


def test_fewer_than_10_passages():
    scores = [8.0, 4.0]
    feats = compute_retrieval_features(scores)
    
    assert feats["rho_1"] == pytest.approx(8.0 / 12.0, rel=1e-3)
    assert feats["delta_12"] == pytest.approx((8.0 - 4.0) / 8.0, rel=1e-3)
    assert 0.0 < feats["entropy_normalized"] <= 1.0
    assert feats["n_high"] == 1


def test_single_passage():
    scores = [7.5]
    feats = compute_retrieval_features(scores)
    
    assert feats["rho_1"] == pytest.approx(1.0, rel=1e-3)
    assert feats["delta_12"] == 1.0
    assert feats["entropy_normalized"] == 0.0
    assert feats["n_high"] == 1


def test_empty_scores():
    scores = []
    feats = compute_retrieval_features(scores, query="test query")
    
    assert feats["rho_1"] == 0.0
    assert feats["delta_12"] == 0.0
    assert feats["entropy_normalized"] == 0.0
    assert feats["query_token_length"] == 2
    assert feats["n_high"] == 0


def test_numerical_stability():
    # Extremely large and small scores
    scores = [1e9, 1e-9, 1e-12, 0.0]
    feats = compute_retrieval_features(scores)
    
    assert np.isfinite(feats["rho_1"])
    assert np.isfinite(feats["delta_12"])
    assert np.isfinite(feats["entropy_normalized"])
    assert 0.0 <= feats["entropy_normalized"] <= 1.0
