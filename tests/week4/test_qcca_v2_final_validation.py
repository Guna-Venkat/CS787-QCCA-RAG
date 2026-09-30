"""Unit tests for QCCA-V2 Final Validation (Stage-2 Policy C1)."""

import numpy as np
import pandas as pd
import pytest

from src.week4.qcca_v2_final_validation import (
    STAGE_2_CORE_FEATURES,
    STAGE_2_FEATURE_JUSTIFICATIONS,
    simulate_policy_c1,
    run_stage2_oof_evaluation,
    evaluate_stage2_threshold_grid,
    run_stage2_controlled_ablations,
    evaluate_stage2_subgroups,
    perform_stage2_error_analysis,
    run_stage2_statistical_comparisons,
    generate_stage2_validation_figures,
)


@pytest.fixture
def mock_validation_data():
    """Create synthetic development dataset with paper clusters and response surface."""
    np.random.seed(42)
    n = 60
    papers = np.repeat([f"paper_{i}" for i in range(12)], 5)
    
    # Synthetic Stage-2 features
    f_data = {
        "question_id": [f"q_{i}" for i in range(n)],
        "paper_id": papers,
        "sem_sim_mean_top6": np.random.uniform(0.4, 0.9, n),
        "bm25_mean_score": np.random.uniform(5.0, 25.0, n),
        "lex_coverage_top2": np.random.uniform(0.3, 0.9, n),
        "evidence_query_cluster_span": np.random.randint(1, 4, n),
        "evidence_lexical_overlap_mean": np.random.uniform(0.1, 0.5, n),
        "query_is_numerical": np.random.binomial(1, 0.2, n),
        "query_is_what_which": np.random.binomial(1, 0.4, n),
        "query_conjunction_count": np.random.poisson(1.0, n),
        "query_word_count": np.random.randint(5, 20, n),
    }
    df = pd.DataFrame(f_data)
    
    # Response surfaces (F1)
    df["F1_k2"] = np.random.uniform(0.1, 0.5, n)
    df["F1_k4"] = df["F1_k2"] + np.random.normal(0.05, 0.1, n)
    df["F1_k6"] = df["F1_k4"] + np.random.normal(0.03, 0.08, n)
    df["F1_k8"] = df["F1_k6"] + np.random.normal(0.01, 0.06, n)
    
    for k in [2, 4, 6, 8]:
        df[f"F1_k{k}"] = np.clip(df[f"F1_k{k}"], 0.0, 1.0)
        
    df["G_4_6"] = df["F1_k6"] - df["F1_k4"]
    df["Y_4_6"] = (df["G_4_6"] > 0.01).astype(int)
    
    df["oracle_k"] = np.random.choice([2, 4, 6, 8], size=n, p=[0.5, 0.2, 0.15, 0.15])
    df["oracle_f1"] = [df.iloc[i][f"F1_k{df.iloc[i]['oracle_k']}"] for i in range(n)]
    
    return df


def test_stage2_feature_justifications():
    """Verify all core Stage-2 features have explicit scientific justifications."""
    for feat in STAGE_2_CORE_FEATURES:
        assert feat in STAGE_2_FEATURE_JUSTIFICATIONS
        assert len(STAGE_2_FEATURE_JUSTIFICATIONS[feat]) > 20


def test_simulate_policy_c1(mock_validation_data):
    """Verify Policy C1 logic: allocates strictly k in {4, 6}, never k=2 or k=8."""
    n = len(mock_validation_data)
    preds = np.random.binomial(1, 0.4, n)
    probs = np.random.uniform(0.1, 0.9, n)
    
    df_res, sm = simulate_policy_c1(mock_validation_data, preds, probs)
    assert len(df_res) == n
    assert set(df_res["selected_k"].unique()).issubset({4, 6})
    assert sm["pct_k2"] == 0.0
    assert sm["pct_k8"] == 0.0
    assert sm["pct_k4"] + sm["pct_k6"] == 100.0


def test_run_stage2_oof_evaluation(mock_validation_data):
    """Verify strict paper-disjoint GroupKFold out-of-fold Stage-2 evaluation."""
    res = run_stage2_oof_evaluation(
        mock_validation_data,
        feature_cols=STAGE_2_CORE_FEATURES,
        target_col="Y_4_6",
        gain_col="G_4_6",
        model_type="logistic_regression_balanced",
        n_splits=3,
        seed=42,
    )
    
    assert res["model_name"] == "logistic_regression_balanced"
    assert 0.0 <= res["roc_auc"] <= 1.0
    assert 0.0 <= res["balanced_acc"] <= 1.0
    assert len(res["df_predictions"]) == len(mock_validation_data)
    assert "p_expand_4_6" in res["df_predictions"].columns
    assert "pred_class_4_6" in res["df_predictions"].columns
    assert "policy_f1" in res["df_predictions"].columns
    assert res["df_coefficients"] is not None
    assert len(res["df_coefficients"]) == len(STAGE_2_CORE_FEATURES)


def test_evaluate_stage2_threshold_grid(mock_validation_data):
    """Verify threshold sensitivity grid evaluation."""
    n = len(mock_validation_data)
    probs = np.random.uniform(0.1, 0.9, n)
    df_thresh = evaluate_stage2_threshold_grid(mock_validation_data, probs, thresholds=[0.3, 0.5, 0.7])
    
    assert len(df_thresh) == 3
    assert "fraction_expanded_k6" in df_thresh.columns
    assert "delta_f1_vs_k4" in df_thresh.columns
    assert "delta_f1_vs_k6" in df_thresh.columns
    assert "delta_f1_vs_k8" in df_thresh.columns


def test_run_stage2_controlled_ablations(mock_validation_data):
    """Verify the 10 controlled Stage-2 feature ablations."""
    df_abl = run_stage2_controlled_ablations(mock_validation_data, seed=42)
    assert len(df_abl) == 10
    assert "ablation_configuration" in df_abl.columns
    assert "roc_auc" in df_abl.columns
    assert "policy_mean_f1" in df_abl.columns


def test_evaluate_stage2_subgroups(mock_validation_data):
    """Verify subgroup analysis."""
    n = len(mock_validation_data)
    df_oof = pd.DataFrame({
        "y_true_4_6": mock_validation_data["Y_4_6"].values,
        "p_expand_4_6": np.random.uniform(0.1, 0.9, n),
        "pred_class_4_6": np.random.binomial(1, 0.4, n),
        "policy_f1": mock_validation_data["F1_k4"].values,
        "regret": np.zeros(n),
    })
    
    df_sg = evaluate_stage2_subgroups(mock_validation_data, df_oof)
    assert len(df_sg) > 0
    assert "subgroup" in df_sg.columns
    assert "actual_positive_rate_pct" in df_sg.columns
    assert "policy_expansion_rate_pct" in df_sg.columns


def test_perform_stage2_error_analysis(mock_validation_data):
    """Verify error analysis decomposition."""
    n = len(mock_validation_data)
    df_oof = pd.DataFrame({
        "question_id": mock_validation_data["question_id"],
        "paper_id": mock_validation_data["paper_id"],
        "y_true_4_6": mock_validation_data["Y_4_6"].values,
        "actual_gain_4_6": mock_validation_data["G_4_6"].values,
        "pred_class_4_6": np.random.binomial(1, 0.4, n),
        "p_expand_4_6": np.random.uniform(0.1, 0.9, n),
        "selected_k": np.random.choice([4, 6], n),
        "policy_f1": mock_validation_data["F1_k4"].values,
        "context_tokens": np.full(n, 946.1),
        "regret": np.zeros(n),
    })
    
    df_audit, df_summary = perform_stage2_error_analysis(mock_validation_data, df_oof)
    assert len(df_audit) == n
    assert "error_category" in df_audit.columns
    assert len(df_summary) <= 4


def test_run_stage2_statistical_comparisons(mock_validation_data):
    """Verify paper-clustered paired bootstrap comparisons."""
    n = len(mock_validation_data)
    df_c1 = pd.DataFrame({
        "policy_f1": mock_validation_data["F1_k4"].values,
        "context_tokens": np.full(n, 1115.9),
        "regret": np.zeros(n),
    })
    
    df_stat = run_stage2_statistical_comparisons(mock_validation_data, df_c1, n_bootstrap=50, seed=42)
    assert len(df_stat) == 3
    assert "mean_delta_f1" in df_stat.columns
    assert "ci_95_f1_low" in df_stat.columns
    assert "ci_95_f1_high" in df_stat.columns


def test_generate_stage2_validation_figures(tmp_path, mock_validation_data):
    """Verify that all 9 validation figures generate properly."""
    res_eval = run_stage2_oof_evaluation(
        mock_validation_data,
        feature_cols=STAGE_2_CORE_FEATURES,
        target_col="Y_4_6",
        gain_col="G_4_6",
        model_type="logistic_regression_balanced",
        n_splits=3,
        seed=42,
    )
    df_thresh = evaluate_stage2_threshold_grid(mock_validation_data, res_eval["oof_probs"], thresholds=[0.3, 0.5])
    df_abl = run_stage2_controlled_ablations(mock_validation_data, n_splits=3, seed=42)
    df_sg = evaluate_stage2_subgroups(mock_validation_data, res_eval["df_predictions"])
    _, df_err_summary = perform_stage2_error_analysis(mock_validation_data, res_eval["df_predictions"])
    
    df_pareto = pd.DataFrame([
        {"system": "Static k=4", "mean_context_tokens": 946.1, "mean_f1": 0.3596, "is_pareto_optimal": False},
        {"system": "Policy C1", "mean_context_tokens": 1115.9, "mean_f1": 0.3842, "is_pareto_optimal": True},
        {"system": "Oracle Sequential", "mean_context_tokens": 650.3, "mean_f1": 0.4069, "is_pareto_optimal": True},
    ])
    
    generate_stage2_validation_figures(
        df_merged=mock_validation_data,
        res_eval=res_eval,
        df_thresh=df_thresh,
        df_abl=df_abl,
        df_sg=df_sg,
        df_err_summary=df_err_summary,
        df_pareto=df_pareto,
        fig_dir=tmp_path,
    )
    
    expected_figures = [
        "stage2_roc_pr.png",
        "calibration_curve.png",
        "threshold_quality_cost.png",
        "feature_coefficients.png",
        "feature_response_curves.png",
        "feature_ablation.png",
        "subgroup_performance.png",
        "error_analysis.png",
        "policy_pareto_frontier.png",
    ]
    for fig_name in expected_figures:
        assert (tmp_path / fig_name).is_file(), f"Missing figure {fig_name}"

