"""Unit tests for QCCA-V2 Diagnostic and Policy Refinement Library."""

import numpy as np
import pandas as pd
import pytest

from src.week4.qcca_v2_diagnostic import (
    compute_expected_calibration_error,
    train_eval_stage_oof_detailed,
    compute_stage_error_consequences,
    simulate_policy_structure,
    sweep_policy_thresholds,
    nested_infold_threshold_selection,
    evaluate_cost_sensitive_rules,
    conduct_detailed_oracle_gap_analysis,
    run_policy_paired_bootstrap,
    FEATURE_FAMILIES,
)


@pytest.fixture
def mock_diagnostic_data():
    """Create synthetic development dataset with paper clusters and response surface."""
    np.random.seed(42)
    n = 60
    papers = np.repeat([f"paper_{i}" for i in range(12)], 5)
    
    # Synthetic features
    f_data = {
        "question_id": [f"q_{i}" for i in range(n)],
        "paper_id": papers,
        "query_conjunction_count": np.random.poisson(1.0, n),
        "query_is_what_which": np.random.binomial(1, 0.4, n),
        "query_is_numerical": np.random.binomial(1, 0.2, n),
        "lex_coverage_top1": np.random.uniform(0.2, 0.8, n),
        "lex_coverage_top2": np.random.uniform(0.3, 0.9, n),
        "lex_coverage_top6": np.random.uniform(0.4, 1.0, n),
        "lex_coverage_gain_1_to_4": np.random.uniform(0.0, 0.4, n),
        "sem_sim_mean_top6": np.random.uniform(0.4, 0.9, n),
        "sem_sim_top1": np.random.uniform(0.5, 0.95, n),
        "bm25_mean_score": np.random.uniform(5.0, 25.0, n),
        "bm25_first_gap_ratio": np.random.uniform(0.0, 0.6, n),
        "passage_sim_std": np.random.uniform(0.02, 0.15, n),
        "passage_cluster_count": np.random.randint(1, 5, n),
        "evidence_query_cluster_span": np.random.randint(1, 4, n),
        "evidence_lexical_overlap_mean": np.random.uniform(0.1, 0.5, n),
    }
    df = pd.DataFrame(f_data)
    
    # Response surfaces (F1)
    df["F1_k2"] = np.random.uniform(0.1, 0.5, n)
    df["F1_k4"] = df["F1_k2"] + np.random.normal(0.05, 0.1, n)
    df["F1_k6"] = df["F1_k4"] + np.random.normal(0.03, 0.08, n)
    df["F1_k8"] = df["F1_k6"] + np.random.normal(0.01, 0.06, n)
    
    # Clip F1 to [0, 1]
    for k in [2, 4, 6, 8]:
        df[f"F1_k{k}"] = np.clip(df[f"F1_k{k}"], 0.0, 1.0)
        
    df["G_2_4"] = df["F1_k4"] - df["F1_k2"]
    df["G_4_6"] = df["F1_k6"] - df["F1_k4"]
    df["G_6_8"] = df["F1_k8"] - df["F1_k6"]
    
    df["Y_2_4"] = (df["G_2_4"] > 0.01).astype(int)
    df["Y_4_6"] = (df["G_4_6"] > 0.01).astype(int)
    df["Y_6_8"] = (df["G_6_8"] > 0.01).astype(int)
    
    # Mock oracle
    df["oracle_k"] = np.random.choice([2, 4, 6, 8], size=n, p=[0.5, 0.2, 0.15, 0.15])
    df["oracle_f1"] = [df.iloc[i][f"F1_k{df.iloc[i]['oracle_k']}"] for i in range(n)]
    df["oracle_regret"] = 0.0
    
    return df


def test_compute_expected_calibration_error():
    """Test ECE and reliability table calculation."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.6, 0.4, 0.85, 0.15])
    
    ece, df_bins = compute_expected_calibration_error(y_true, y_prob, n_bins=5)
    assert isinstance(ece, float)
    assert 0.0 <= ece <= 1.0
    assert len(df_bins) == 5
    assert "mean_pred_prob" in df_bins.columns
    assert "empirical_accuracy" in df_bins.columns


def test_train_eval_stage_oof_detailed(mock_diagnostic_data):
    """Test detailed stage out-of-fold cross-validation."""
    feats = ["query_conjunction_count", "lex_coverage_top1", "bm25_first_gap_ratio"]
    res = train_eval_stage_oof_detailed(
        mock_diagnostic_data,
        stage_name="Stage 1 (2->4)",
        target_col="Y_2_4",
        feature_cols=feats,
        model_name="logistic_regression_balanced",
        n_splits=3,
        seed=42,
    )
    
    assert res["stage"] == "Stage 1 (2->4)"
    assert len(res["oof_preds"]) == len(mock_diagnostic_data)
    assert len(res["oof_probs"]) == len(mock_diagnostic_data)
    assert 0.0 <= res["oof_roc_auc"] <= 1.0
    assert 0.0 <= res["oof_balanced_acc"] <= 1.0
    assert 0.0 <= res["oof_brier_score"] <= 1.0
    assert res["tp"] + res["fp"] + res["tn"] + res["fn"] == len(mock_diagnostic_data)


def test_compute_stage_error_consequences(mock_diagnostic_data):
    """Test accounting of downstream policy consequences per error type."""
    y_pred = np.random.binomial(1, 0.4, len(mock_diagnostic_data))
    df_cons = compute_stage_error_consequences(
        mock_diagnostic_data,
        stage_name="Stage 1 (2->4)",
        target_col="Y_2_4",
        gain_col="G_2_4",
        y_pred=y_pred,
        token_delta=411.6,
    )
    
    assert len(df_cons) == 4
    assert np.isclose(df_cons["percentage"].sum(), 100.0)
    assert "TP (Beneficial Expansion)" in df_cons["category"].values
    assert "FP (False Expansion / Wasted Context)" in df_cons["category"].values
    assert "FN (False Stopping / Missed Gain)" in df_cons["category"].values
    assert "TN (True Stopping / Distractor Avoided)" in df_cons["category"].values


def test_simulate_policy_structure(mock_diagnostic_data):
    """Test simulation across policy architecture variants."""
    n = len(mock_diagnostic_data)
    s1 = np.ones(n, dtype=int)
    s2 = np.ones(n, dtype=int)
    s3 = np.zeros(n, dtype=int)
    
    # Policy A: 2 -> 4 -> 6 -> 8
    df_a, sm_a = simulate_policy_structure(mock_diagnostic_data, "A_full_cascade", s1, s2, s3)
    assert sm_a["mean_k"] == 6.0
    assert sm_a["pct_k6"] == 100.0
    
    # Policy B: Always start at k=4
    df_b, sm_b = simulate_policy_structure(mock_diagnostic_data, "B_start_k4", s1, s2, s3)
    assert (df_b["selected_k"] >= 4).all()
    assert sm_b["pct_k2"] == 0.0
    
    # Policy C: Always start at k=4, stop at k=6
    df_c, sm_c = simulate_policy_structure(mock_diagnostic_data, "C_start_k4_stop_k6", s1, s2, s3)
    assert (df_c["selected_k"] == 6).all()
    assert sm_c["pct_k8"] == 0.0


def test_sweep_policy_thresholds(mock_diagnostic_data):
    """Test policy threshold sweep."""
    n = len(mock_diagnostic_data)
    p1 = np.random.uniform(0.1, 0.9, n)
    p2 = np.random.uniform(0.1, 0.9, n)
    p3 = np.random.uniform(0.1, 0.9, n)
    
    df_thresh = sweep_policy_thresholds(mock_diagnostic_data, p1, p2, p3, thresholds=[0.3, 0.5, 0.7])
    assert len(df_thresh) == 3
    assert "mean_f1" in df_thresh.columns
    assert "mean_context_tokens" in df_thresh.columns


def test_nested_infold_threshold_selection(mock_diagnostic_data):
    """Test leak-free nested in-fold threshold selection."""
    n = len(mock_diagnostic_data)
    p1 = np.random.uniform(0.1, 0.9, n)
    p2 = np.random.uniform(0.1, 0.9, n)
    p3 = np.random.uniform(0.1, 0.9, n)
    fold_ids = np.repeat([0, 1, 2], n // 3)
    
    df_res, sm, df_sel = nested_infold_threshold_selection(
        mock_diagnostic_data, p1, p2, p3, fold_ids, candidate_thresholds=[0.3, 0.5, 0.7]
    )
    assert len(df_res) == n
    assert "mean_f1" in sm
    assert len(df_sel) == 3


def test_conduct_detailed_oracle_gap_analysis(mock_diagnostic_data):
    """Test granular failure audit relative to the oracle."""
    n = len(mock_diagnostic_data)
    policy_df = pd.DataFrame({
        "selected_k": np.repeat([2, 4, 6, 8], n // 4),
        "policy_f1": np.random.uniform(0.2, 0.8, n),
        "regret": np.random.uniform(0.0, 0.3, n),
    })
    
    s1 = np.ones(n, dtype=int)
    s2 = np.zeros(n, dtype=int)
    s3 = np.zeros(n, dtype=int)
    p = np.full(n, 0.5)
    
    df_audit, df_summary = conduct_detailed_oracle_gap_analysis(
        mock_diagnostic_data, policy_df, s1, s2, s3, p, p, p
    )
    assert len(df_audit) == n
    assert "failure_type" in df_audit.columns
    assert "total_regret" in df_summary.columns
    assert len(df_summary) > 0


def test_run_policy_paired_bootstrap(mock_diagnostic_data):
    """Test paired paper-clustered bootstrap comparison."""
    n = len(mock_diagnostic_data)
    policy_df = pd.DataFrame({
        "policy_f1": mock_diagnostic_data["F1_k4"].values,
        "context_tokens": np.full(n, 946.1),
    })
    
    df_boot = run_policy_paired_bootstrap(
        mock_diagnostic_data,
        policy_dfs={"TestPolicy": policy_df},
        baseline_ks=[4, 6],
        n_bootstrap=50,
        seed=42,
    )
    assert len(df_boot) == 2
    assert "mean_delta_f1" in df_boot.columns
    assert "ci_95_f1_low" in df_boot.columns
    assert "ci_95_f1_high" in df_boot.columns
