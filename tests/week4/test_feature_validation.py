"""Unit tests for Week 4 Feature Validation library (W4-B)."""

import pytest
import numpy as np
import pandas as pd

from src.week4.feature_validation import (
    benjamini_hochberg,
    compute_fdr_table,
    compute_feature_correlation_matrix,
    find_redundancy_clusters,
    select_redundancy_representatives,
    rank_partial_correlation,
    compute_partial_associations,
    construct_transition_targets,
    compute_transition_associations,
    analyze_coverage_sign_flips,
    build_provisional_feature_set,
    evaluate_qcca_v2_gates,
)


def test_benjamini_hochberg():
    """Test Benjamini-Hochberg step-up procedure with known values and edge cases."""
    # Empty
    assert len(benjamini_hochberg(np.array([]))) == 0
    
    # Single value
    single = benjamini_hochberg(np.array([0.03]))
    assert np.isclose(single[0], 0.03)
    
    # 4 known p-values
    # p = [0.01, 0.04, 0.03, 0.20], n=4
    # sorted:
    # rank 1: p=0.01 -> 0.01 * 4 / 1 = 0.04
    # rank 2: p=0.03 -> 0.03 * 4 / 2 = 0.06
    # rank 3: p=0.04 -> 0.04 * 4 / 3 = 0.05333...
    # rank 4: p=0.20 -> 0.20 * 4 / 4 = 0.20
    # Enforcing backwards monotonicity:
    # rank 4: 0.20
    # rank 3: min(0.05333, 0.20) = 0.05333
    # rank 2: min(0.06, 0.05333) = 0.05333
    # rank 1: min(0.04, 0.05333) = 0.04
    p = np.array([0.01, 0.04, 0.03, 0.20])
    q = benjamini_hochberg(p)
    assert np.isclose(q[0], 0.04)
    assert np.isclose(q[1], 0.05333333)
    assert np.isclose(q[2], 0.05333333)
    assert np.isclose(q[3], 0.20)
    
    # Ensure monotonicity holds
    p_rand = np.random.uniform(0, 1, 100)
    q_rand = benjamini_hochberg(p_rand)
    order = np.argsort(p_rand)
    assert np.all(np.diff(q_rand[order]) >= -1e-12)
    assert np.all(q_rand >= 0.0)
    assert np.all(q_rand <= 1.0)


def test_compute_fdr_table():
    """Verify FDR table formatting and column additions."""
    df_sp = pd.DataFrame({
        "feature": ["f1", "f2", "f3", "f4"],
        "target": ["t1", "t1", "t1", "t1"],
        "spearman_rho": [0.3, 0.25, 0.15, 0.05],
        "p_value_naive": [0.001, 0.01, 0.04, 0.35],
    })
    df_fdr = compute_fdr_table(df_sp, alpha_nominal=0.05, alpha_fdr=0.10)
    assert "q_value" in df_fdr.columns
    assert "is_nominal_sig" in df_fdr.columns
    assert "is_fdr_sig_05" in df_fdr.columns
    assert "is_fdr_sig_10" in df_fdr.columns
    assert df_fdr["is_nominal_sig"].sum() == 3
    assert df_fdr["is_fdr_sig_10"].sum() >= 2


def test_correlation_matrix_and_clustering():
    """Verify feature correlation matrix and agglomerative clustering for redundancy."""
    np.random.seed(42)
    n = 100
    z = np.random.randn(n)
    f1 = z + np.random.randn(n) * 0.05  # Highly correlated with f2
    f2 = z + np.random.randn(n) * 0.05
    f3 = np.random.randn(n)             # Independent
    
    df_f = pd.DataFrame({"f1": f1, "f2": f2, "f3": f3})
    feature_cols = ["f1", "f2", "f3"]
    
    df_corr = compute_feature_correlation_matrix(df_f, feature_cols)
    assert df_corr.shape == (3, 3)
    assert np.isclose(df_corr.loc["f1", "f1"], 1.0)
    assert df_corr.loc["f1", "f2"] > 0.85
    assert abs(df_corr.loc["f1", "f3"]) < 0.40
    
    # Redundancy clustering
    df_clust = find_redundancy_clusters(df_corr, redundancy_threshold=0.80)
    assert len(df_clust) == 3
    # f1 and f2 should be in the same cluster
    c_f1 = df_clust.loc[df_clust["feature"] == "f1", "redundancy_cluster_id"].iloc[0]
    c_f2 = df_clust.loc[df_clust["feature"] == "f2", "redundancy_cluster_id"].iloc[0]
    c_f3 = df_clust.loc[df_clust["feature"] == "f3", "redundancy_cluster_id"].iloc[0]
    assert c_f1 == c_f2
    assert c_f1 != c_f3


def test_rank_partial_correlation():
    """Test rank-based partial correlation controlling for confounding variable."""
    np.random.seed(42)
    n = 200
    # Confounder z drives both x and y
    z = np.random.randn(n)
    x = z + np.random.randn(n) * 0.4
    y = z + np.random.randn(n) * 0.4
    
    # Raw correlation between x and y should be high
    raw_r = np.corrcoef(x, y)[0, 1]
    assert raw_r > 0.70
    
    # Controlling for z, partial correlation should collapse toward 0
    part_r, p_val = rank_partial_correlation(x, y, z)
    assert abs(part_r) < 0.20


def test_construct_transition_targets(tmp_path):
    """Test construction of continuous gains and binary transition targets."""
    # Create synthetic dev matrix
    dev_csv = tmp_path / "mock_dev_matrix.csv"
    df_mock = pd.DataFrame({
        "question_id": ["q1", "q2", "q3"],
        "paper_id": ["p1", "p1", "p2"],
        "F1_k2": [0.20, 0.40, 0.50],
        "F1_k4": [0.35, 0.405, 0.45], # q1 gain=0.15, q2 gain=0.005 (not >0.01), q3 gain=-0.05
        "F1_k6": [0.40, 0.45, 0.50],  # q1 gain=0.05, q2 gain=0.045, q3 gain=0.05
        "F1_k8": [0.42, 0.40, 0.52],  # q1 gain=0.02, q2 gain=-0.05, q3 gain=0.02
        "best_k": [8, 6, 8],
    })
    df_mock.to_csv(dev_csv, index=False)
    
    df_trans = construct_transition_targets(str(dev_csv), delta_threshold=0.01)
    
    assert len(df_trans) == 3
    # Check continuous gains
    assert np.isclose(df_trans.loc[0, "G_2_to_4"], 0.15)
    assert np.isclose(df_trans.loc[0, "G_4_to_6"], 0.05)
    assert np.isclose(df_trans.loc[0, "G_6_to_8"], 0.02)
    assert np.isclose(df_trans.loc[0, "G_2_to_8"], 0.22)
    
    # Check binary any gain
    assert df_trans.loc[0, "I_2_to_4"] == 1
    assert df_trans.loc[1, "I_2_to_4"] == 1  # 0.405 > 0.40
    assert df_trans.loc[2, "I_2_to_4"] == 0  # 0.45 < 0.50
    
    # Check binary > 0.01 gain
    assert df_trans.loc[0, "I_2_to_4_01"] == 1
    assert df_trans.loc[1, "I_2_to_4_01"] == 0  # 0.005 is not > 0.01
    assert df_trans.loc[2, "I_2_to_4_01"] == 0


def test_analyze_coverage_sign_flips():
    """Test dedicated coverage sign-flip detection logic."""
    df_sp = pd.DataFrame([
        {"feature": "lex_coverage_top2", "target": "G_2_to_4", "spearman_rho": 0.05},
        {"feature": "lex_coverage_top2", "target": "G_4_to_6", "spearman_rho": 0.1673},
        {"feature": "lex_coverage_top2", "target": "G_6_to_8", "spearman_rho": -0.1701},
        {"feature": "lex_coverage_top1", "target": "G_2_to_4", "spearman_rho": 0.02},
        {"feature": "lex_coverage_top1", "target": "G_4_to_6", "spearman_rho": 0.12},
        {"feature": "lex_coverage_top1", "target": "G_6_to_8", "spearman_rho": -0.14},
    ])
    df_fdr = pd.DataFrame([
        {"feature": "lex_coverage_top2", "target": "G_4_to_6", "q_value": 0.08},
        {"feature": "lex_coverage_top2", "target": "G_6_to_8", "q_value": 0.08},
        {"feature": "lex_coverage_top1", "target": "G_4_to_6", "q_value": 0.15},
        {"feature": "lex_coverage_top1", "target": "G_6_to_8", "q_value": 0.12},
    ])
    df_boot = pd.DataFrame([
        {"feature": "lex_coverage_top2", "target": "G_4_to_6", "bootstrap_ci_low": 0.03, "bootstrap_ci_high": 0.30},
        {"feature": "lex_coverage_top2", "target": "G_6_to_8", "bootstrap_ci_low": -0.30, "bootstrap_ci_high": -0.04},
        {"feature": "lex_coverage_top1", "target": "G_4_to_6", "bootstrap_ci_low": -0.01, "bootstrap_ci_high": 0.25},
        {"feature": "lex_coverage_top1", "target": "G_6_to_8", "bootstrap_ci_low": -0.28, "bootstrap_ci_high": -0.01},
    ])
    df_stab = pd.DataFrame([
        {"feature": "lex_coverage_top2", "target": "G_4_to_6", "fold_sign_consistency": 0.8, "fold_rhos": [0.1, 0.2, 0.15, 0.18, 0.05]},
        {"feature": "lex_coverage_top2", "target": "G_6_to_8", "fold_sign_consistency": 0.8, "fold_rhos": [-0.1, -0.2, -0.15, -0.18, 0.02]},
        {"feature": "lex_coverage_top1", "target": "G_4_to_6", "fold_sign_consistency": 0.6, "fold_rhos": []},
        {"feature": "lex_coverage_top1", "target": "G_6_to_8", "fold_sign_consistency": 0.8, "fold_rhos": []},
    ])
    
    df_flips = analyze_coverage_sign_flips(df_sp, df_fdr, df_boot, df_stab)
    row_top2 = df_flips[df_flips["coverage_feature"] == "lex_coverage_top2"].iloc[0]
    assert row_top2["has_sign_flip"] == True
    assert row_top2["G_4_to_6_rho"] > 0
    assert row_top2["G_6_to_8_rho"] < 0


def test_evaluate_qcca_v2_gates():
    """Verify that gate evaluation structure returns valid keys and values."""
    df_fdr = pd.DataFrame({
        "is_fdr_sig_10": [False] * 20,
        "is_nominal_sig": [True] * 18 + [False] * 2,
    })
    df_clusters = pd.DataFrame({"redundancy_cluster_id": list(range(12))})
    df_trans = pd.DataFrame()
    df_sign_flip = pd.DataFrame({"has_sign_flip": [True, True, True, False]})
    df_prov = pd.DataFrame({
        "feature": [f"f{i}" for i in range(10)],
        "fold_sign_consistency": [1.0] * 8 + [0.6] * 2,
        "bootstrap_excludes_zero": [True] * 4 + [False] * 6,
    })
    
    gates = evaluate_qcca_v2_gates(df_fdr, df_clusters, df_trans, df_sign_flip, df_prov)
    assert "Gate A (Reproducible Signals)" in gates
    assert "Gate B (Non-Redundant Signals)" in gates
    assert "Gate C (Transition Specificity)" in gates
    assert "Gate D (Paper-Grouped Stability)" in gates
    assert "Gate E (Carried Features)" in gates
    assert "Gate F (Sequential Policy Justification)" in gates
    
    assert gates["Gate A (Reproducible Signals)"]["status"] in ["PASSED", "SUPPORTED (EXPLORATORY)"]
    assert gates["Gate B (Non-Redundant Signals)"]["status"] == "PASSED"
    assert gates["Gate C (Transition Specificity)"]["status"] == "PASSED"
    assert gates["Gate D (Paper-Grouped Stability)"]["status"] == "PASSED"
    assert gates["Gate F (Sequential Policy Justification)"]["status"] == "SUPPORTED FOR TESTING"

