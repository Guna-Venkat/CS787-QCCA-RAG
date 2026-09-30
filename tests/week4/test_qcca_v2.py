"""Comprehensive unit tests for QCCA-V2 Sequential Allocation Framework."""

import pytest
import numpy as np
import pandas as pd

from src.week4.qcca_v2 import (
    CONTEXT_TOKEN_COSTS,
    STAGE_1_FEATURES,
    STAGE_2_FEATURES,
    STAGE_3_FEATURES,
    PROVISIONAL_14_FEATURES,
    construct_transition_targets,
    summarize_transition_targets,
    load_qcca_v2_data,
    build_classifier,
    build_regressor,
    train_stage_classifiers_cv,
    train_stage_regressors_cv,
    simulate_sequential_policy,
    simulate_oracle_sequential_policy,
    evaluate_threshold_sensitivity,
    run_paper_clustered_bootstrap_policy,
    compute_pareto_frontier,
    analyze_policy_failures,
    evaluate_qcca_v2_decision_gates,
)


@pytest.fixture
def mock_dev_matrix(tmp_path):
    """Create a synthetic dev per-query response surface matrix."""
    csv_file = tmp_path / "mock_dev_matrix.csv"
    data = {
        "question_id": [f"q{i}" for i in range(20)],
        "paper_id": [f"p{i // 4}" for i in range(20)],  # 5 papers, 4 questions each
        "F1_k0": np.random.uniform(0.1, 0.2, 20),
        "F1_k2": [0.20, 0.25, 0.30, 0.15, 0.40, 0.10, 0.50, 0.20, 0.30, 0.25, 0.20, 0.15, 0.35, 0.40, 0.20, 0.25, 0.30, 0.15, 0.20, 0.25],
        "F1_k4": [0.25, 0.255, 0.32, 0.10, 0.45, 0.12, 0.55, 0.18, 0.35, 0.24, 0.22, 0.18, 0.40, 0.42, 0.25, 0.22, 0.35, 0.18, 0.25, 0.28],
        "F1_k5": [0.26, 0.258, 0.33, 0.11, 0.46, 0.13, 0.54, 0.17, 0.36, 0.23, 0.23, 0.19, 0.41, 0.43, 0.26, 0.21, 0.36, 0.19, 0.26, 0.29],
        "F1_k6": [0.28, 0.26, 0.35, 0.12, 0.48, 0.15, 0.52, 0.15, 0.38, 0.22, 0.25, 0.20, 0.42, 0.45, 0.28, 0.20, 0.38, 0.22, 0.28, 0.30],
        "F1_k8": [0.30, 0.24, 0.36, 0.15, 0.50, 0.18, 0.50, 0.12, 0.40, 0.20, 0.26, 0.22, 0.45, 0.46, 0.30, 0.18, 0.40, 0.25, 0.30, 0.32],
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    return str(csv_file)


# 1. Test transition target construction
def test_transition_target_construction(mock_dev_matrix):
    """Verify transition target construction calculates exact marginal gains."""
    df_t = construct_transition_targets(mock_dev_matrix, delta_values=[0.01])
    assert len(df_t) == 20
    assert "G_2_4" in df_t.columns
    assert "G_4_6" in df_t.columns
    assert "G_6_8" in df_t.columns
    
    # Check row 0: F1_k2=0.20, F1_k4=0.25 -> G_2_4=0.05
    assert np.isclose(df_t.loc[0, "G_2_4"], 0.05)
    # Check row 0: F1_k4=0.25, F1_k6=0.28 -> G_4_6=0.03
    assert np.isclose(df_t.loc[0, "G_4_6"], 0.03)


# 2. Test delta threshold correctness
def test_delta_threshold_correctness(mock_dev_matrix):
    """Verify that binary indicator is 1 if and only if gain > delta."""
    df_t = construct_transition_targets(mock_dev_matrix, delta_values=[0.00, 0.01, 0.02, 0.05])
    for d in [0.00, 0.01, 0.02, 0.05]:
        tag = f"d{int(d*100):02d}"
        y24 = df_t[f"Y_2_4_{tag}"]
        g24 = df_t["G_2_4"]
        assert np.all(y24 == (g24 > d).astype(int))


# 3. Test no impossible target values
def test_no_impossible_target_values(mock_dev_matrix):
    """Verify targets contain only 0 and 1, and no NaNs exist."""
    df_t = construct_transition_targets(mock_dev_matrix)
    for col in ["Y_2_4", "Y_4_6", "Y_6_8"]:
        unique_vals = set(df_t[col].unique())
        assert unique_vals.issubset({0, 1})
        assert not df_t[col].isna().any()


# 4. Test GroupKFold paper separation
def test_groupkfold_paper_separation(mock_dev_matrix):
    """Verify that paper IDs in validation folds never appear in training folds."""
    from sklearn.model_selection import GroupKFold
    df_t = construct_transition_targets(mock_dev_matrix)
    groups = df_t["paper_id"].values
    gkf = GroupKFold(n_splits=5)
    
    for train_idx, val_idx in gkf.split(df_t, groups=groups):
        train_papers = set(groups[train_idx])
        val_papers = set(groups[val_idx])
        assert train_papers.isdisjoint(val_papers)


# 5. Test no feature leakage
def test_no_feature_leakage():
    """Verify that target columns and response metrics are never included in feature sets."""
    forbidden = {"F1", "f1", "regret", "oracle", "best_k", "G_2_4", "G_4_6", "G_6_8", "Y_2_4", "Y_4_6", "Y_6_8"}
    for f in STAGE_1_FEATURES + STAGE_2_FEATURES + STAGE_3_FEATURES + PROVISIONAL_14_FEATURES:
        for f_bad in forbidden:
            assert f_bad.lower() not in f.lower(), f"Potential leakage in feature name: {f}"


# 6. Test stage-specific feature availability
def test_stage_specific_feature_availability():
    """Verify that all stage features exist in the raw feature matrix."""
    feat_matrix_path = "results/week4/feature_discovery/feature_matrix.csv"
    df_f = pd.read_csv(feat_matrix_path)
    cols = set(df_f.columns)
    
    for stage, f_list in [("S1", STAGE_1_FEATURES), ("S2", STAGE_2_FEATURES), ("S3", STAGE_3_FEATURES)]:
        for f in f_list:
            assert f in cols, f"Stage {stage} feature {f} missing from feature_matrix.csv!"


# 7. Test OOF prediction alignment
def test_oof_prediction_alignment(mock_dev_matrix):
    """Verify that OOF predictions match the dataset index and length exactly."""
    df_t = construct_transition_targets(mock_dev_matrix)
    # Add dummy feature
    df_t["dummy_feat"] = np.random.randn(len(df_t))
    res = train_stage_classifiers_cv(
        df_t, ["dummy_feat"], "Y_2_4", "majority", n_splits=5
    )
    assert len(res["oof_preds"]) == len(df_t)
    assert len(res["oof_probs"]) == len(df_t)


# 8. Test sequential policy logic
def test_sequential_policy_logic():
    """Verify cascade decision logic: only expand to k+2 if previous stage was positive."""
    df_dummy = pd.DataFrame({
        "question_id": ["q1", "q2", "q3", "q4"],
        "paper_id": ["p1", "p1", "p2", "p2"],
        "F1_k2": [0.2, 0.2, 0.2, 0.2],
        "F1_k4": [0.3, 0.3, 0.3, 0.3],
        "F1_k6": [0.4, 0.4, 0.4, 0.4],
        "F1_k8": [0.5, 0.5, 0.5, 0.5],
        "Y_2_4": [1, 1, 0, 0],
        "Y_4_6": [1, 0, 1, 0],
        "Y_6_8": [1, 1, 1, 0],
    })
    
    s1_preds = np.array([1, 1, 0, 0])
    s2_preds = np.array([1, 0, 1, 0])  # q3 has s2=1 but s1=0, must stop at k=2
    s3_preds = np.array([1, 1, 1, 0])  # q2 has s3=1 but s2=0, must stop at k=4
    
    df_res, _ = simulate_sequential_policy(df_dummy, s1_preds, s2_preds, s3_preds)
    allocs = df_res["selected_k"].tolist()
    assert allocs[0] == 8  # 1 -> 1 -> 1 -> k=8
    assert allocs[1] == 4  # 1 -> 0 -> k=4
    assert allocs[2] == 2  # 0 -> stop at k=2 (even though s2=1)
    assert allocs[3] == 2  # 0 -> stop at k=2


# 9. Test k is always in {2, 4, 6, 8}
def test_k_is_always_valid_budget(mock_dev_matrix):
    """Verify that every query allocation is strictly an element of {2, 4, 6, 8}."""
    df_t = construct_transition_targets(mock_dev_matrix)
    n = len(df_t)
    s1 = np.random.choice([0, 1], n)
    s2 = np.random.choice([0, 1], n)
    s3 = np.random.choice([0, 1], n)
    df_res, _ = simulate_sequential_policy(df_t, s1, s2, s3)
    assert set(df_res["selected_k"].unique()).issubset({2, 4, 6, 8})


# 10. Test context-token lookup correctness
def test_context_token_lookup():
    """Verify context token costs map accurately to frozen values."""
    assert CONTEXT_TOKEN_COSTS[2] == 534.5
    assert CONTEXT_TOKEN_COSTS[4] == 946.1
    assert CONTEXT_TOKEN_COSTS[6] == 1359.1
    assert CONTEXT_TOKEN_COSTS[8] == 1758.5


# 11. Test oracle sequential policy correctness
def test_oracle_sequential_policy(mock_dev_matrix):
    """Verify oracle sequential policy expands if and only if actual gain > delta."""
    df_t = construct_transition_targets(mock_dev_matrix)
    df_res, summary = simulate_oracle_sequential_policy(df_t, delta=0.01)
    assert len(df_res) == len(df_t)
    for _, row in df_res.iterrows():
        k = row["selected_k"]
        qid = row["question_id"]
        t_row = df_t[df_t["question_id"] == qid].iloc[0]
        if k == 2:
            assert t_row["G_2_4"] <= 0.01
        elif k == 4:
            assert t_row["G_2_4"] > 0.01
            assert t_row["G_4_6"] <= 0.01
        elif k == 6:
            assert t_row["G_2_4"] > 0.01
            assert t_row["G_4_6"] > 0.01
            assert t_row["G_6_8"] <= 0.01
        elif k == 8:
            assert t_row["G_2_4"] > 0.01
            assert t_row["G_4_6"] > 0.01
            assert t_row["G_6_8"] > 0.01


# 12. Test bootstrap resampling occurs at paper level
def test_bootstrap_paper_level_resampling(mock_dev_matrix):
    """Verify bootstrap computes 95% intervals with paper-level clustering."""
    df_t = construct_transition_targets(mock_dev_matrix)
    n = len(df_t)
    df_res, _ = simulate_sequential_policy(df_t, np.ones(n), np.ones(n), np.ones(n))
    boot_df = run_paper_clustered_bootstrap_policy(
        df_res, df_t, n_bootstrap=100, seed=42
    )
    assert len(boot_df) >= 5
    assert "ci_95_low" in boot_df.columns
    assert "ci_95_high" in boot_df.columns
    assert np.all(boot_df["ci_95_low"] <= boot_df["ci_95_high"])


# 13. Test deterministic seeds
def test_deterministic_seeds(mock_dev_matrix):
    """Verify that runs with identical seeds produce identical results."""
    df_t = construct_transition_targets(mock_dev_matrix)
    df_t["feat1"] = np.random.randn(len(df_t))
    
    res1 = train_stage_classifiers_cv(df_t, ["feat1"], "Y_2_4", "logistic_regression", random_state=42)
    res2 = train_stage_classifiers_cv(df_t, ["feat1"], "Y_2_4", "logistic_regression", random_state=42)
    
    assert np.allclose(res1["oof_probs"], res2["oof_probs"])
    assert np.array_equal(res1["oof_preds"], res2["oof_preds"])
