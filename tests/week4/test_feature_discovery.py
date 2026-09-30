"""Unit tests for Week 4 Feature Discovery library."""

import pytest
import numpy as np
import pandas as pd
import torch

from src.week4.feature_discovery import (
    extract_query_features,
    get_feature_metadata,
    audit_feature_leakage,
    construct_target_matrix,
    load_all_dev_features,
    compute_spearman_associations,
    compute_pearson_associations,
    compute_mutual_information,
    compute_paper_clustered_bootstrap,
    compute_groupkfold_stability,
    screen_candidate_features,
    summarize_feature_families,
)


def test_feature_metadata_completeness():
    """Verify that feature metadata specifies exactly 64 unique pre-generation features."""
    df_meta = get_feature_metadata()
    assert len(df_meta) == 64
    assert df_meta["feature_name"].nunique() == 64
    assert set(df_meta["available_pre_generation"]) == {True}
    assert set(df_meta["leakage_status"]) == {"VERIFIED_PRE_GEN"}
    
    expected_families = {
        "A: BM25/Retrieval",
        "B: Query Complexity",
        "C: Semantic Structure",
        "D: Passage Redundancy/Diversity",
        "E: Lexical Coverage",
        "F: Evidence Structure",
    }
    assert set(df_meta["feature_family"].unique()) == expected_families


def test_extract_query_features_synthetic():
    """Test feature extraction on a synthetic query and passage set."""
    record = {
        "question_id": "test_q1",
        "paper_id": "test_p1",
        "question": "What is the difference between model A and model B, and how many parameters are used?",
        "passages": [
            {"passage_id": "p1", "score": 10.5, "text": "Model A has 10 million parameters and uses transformer layers."},
            {"passage_id": "p2", "score": 8.0, "text": "Model B differs by having 20 million parameters."},
            {"passage_id": "p3", "score": 5.2, "text": "Comparison of accuracy between models."},
        ]
    }
    q_emb = torch.randn(4096)
    p_embs = torch.randn(10, 4096)
    
    feat = extract_query_features(record, q_emb, p_embs)
    assert feat["question_id"] == "test_q1"
    assert feat["paper_id"] == "test_p1"
    
    # Check feature count
    feature_keys = [k for k in feat.keys() if k not in ["question_id", "paper_id"]]
    assert len(feature_keys) == 64
    
    # Check query complexity detection
    assert feat["query_is_comparison"] == 1.0
    assert feat["query_is_how_why"] == 1.0
    assert feat["query_is_numerical"] == 1.0
    assert feat["query_comma_count"] == 1


def test_audit_feature_leakage():
    """Verify leakage audit passes clean data and detects injected post-generation leakage."""
    clean_df = pd.DataFrame([{
        "question_id": "1",
        "paper_id": "10",
        "rho_1": 0.5,
        "query_word_count": 8,
        "sem_sim_top1": 0.75,
        "passage_sim_mean": 0.60,
    }])
    is_clean, violations = audit_feature_leakage(clean_df)
    assert is_clean is True
    assert len(violations) == 0
    
    # Injected leakage columns
    leaky_df = clean_df.copy()
    leaky_df["generated_answer_f1"] = 0.45
    leaky_df["oracle_target_budget"] = 4
    leaky_df["gold_token_count"] = 12
    
    is_clean_leaky, violations_leaky = audit_feature_leakage(leaky_df)
    assert is_clean_leaky is False
    assert len(violations_leaky) >= 3


def test_construct_target_matrix():
    """Verify target matrix constructs the 5 primary and 2 auxiliary targets."""
    df_targets = construct_target_matrix()
    assert len(df_targets) == 231
    expected_cols = [
        "question_id", "paper_id", "oracle_k",
        "G_2_to_4", "G_4_to_6", "G_6_to_8", "G_2_to_8",
        "G_2_to_5", "G_5_to_8"
    ]
    for col in expected_cols:
        assert col in df_targets.columns
    assert set(df_targets["oracle_k"].unique()).issubset({2, 4, 5, 6, 8})


def test_load_all_dev_features_and_leakage():
    """Verify end-to-end extraction across all 231 dev questions with 0 leakage violations."""
    df_features, df_meta = load_all_dev_features()
    assert len(df_features) == 231
    assert len(df_features.columns) == 66  # question_id + paper_id + 64 features
    
    is_clean, violations = audit_feature_leakage(df_features)
    assert is_clean is True, f"Found leakage violations: {violations}"


def test_statistical_association_routines():
    """Verify statistical correlation, bootstrap, stability, and screening routines."""
    # Synthetic small dataset
    np.random.seed(42)
    n = 30
    df_feat = pd.DataFrame({
        "question_id": [f"q{i}" for i in range(n)],
        "paper_id": [f"p{i // 3}" for i in range(n)],  # 10 papers
        "feat_signal": np.linspace(0, 1, n) + np.random.normal(0, 0.1, n),
        "feat_noise": np.random.normal(0, 1, n),
    })
    df_targ = pd.DataFrame({
        "question_id": [f"q{i}" for i in range(n)],
        "paper_id": [f"p{i // 3}" for i in range(n)],
        "oracle_k": np.random.choice([2, 4, 6, 8], size=n),
        "G_2_to_4": df_feat["feat_signal"] * 0.5 + np.random.normal(0, 0.1, n),
    })
    
    f_cols = ["feat_signal", "feat_noise"]
    t_cols = ["oracle_k", "G_2_to_4"]
    
    # 1. Spearman
    df_sp = compute_spearman_associations(df_feat, df_targ, f_cols, t_cols)
    assert len(df_sp) == 4
    signal_g = df_sp[(df_sp["feature"] == "feat_signal") & (df_sp["target"] == "G_2_to_4")]["spearman_rho"].iloc[0]
    assert signal_g > 0.5
    
    # 2. Pearson
    df_pe = compute_pearson_associations(df_feat, df_targ, f_cols, ["G_2_to_4"])
    assert len(df_pe) == 2
    
    # 3. Mutual Information
    df_mi = compute_mutual_information(df_feat, df_targ, f_cols, t_cols)
    assert len(df_mi) == 4
    
    # 4. Bootstrap
    df_boot = compute_paper_clustered_bootstrap(df_feat, df_targ, f_cols, t_cols, n_bootstrap=100)
    assert len(df_boot) == 4
    assert "bootstrap_ci_low" in df_boot.columns
    assert "bootstrap_ci_high" in df_boot.columns
    
    # 5. Stability
    df_stab = compute_groupkfold_stability(df_feat, df_targ, f_cols, t_cols, n_splits=3)
    assert len(df_stab) == 4
    assert "fold_sign_consistency" in df_stab.columns
