"""Unit tests for Week 5 Nonlinear Stage-2 ML Modeling Benchmark."""

import numpy as np
import pandas as pd
import pytest

from src.week4.model_benchmark import (
    CONTEXT_TOKEN_COSTS,
    STAGE_2_CORE_FEATURES,
    STAGE_2_SUBSET_FEATURES,
    SmallPyTorchMLP,
    analyze_feature_interactions,
    evaluate_model_threshold_grid,
    instantiate_model,
    run_paper_clustered_paired_bootstrap,
    simulate_policy_c1,
    train_eval_model_oof,
)


@pytest.fixture
def mock_dev_data():
    """Create synthetic development dataset with paper clusters and response surface."""
    np.random.seed(42)
    n = 60
    papers = np.repeat([f"paper_{i}" for i in range(12)], 5)

    df = pd.DataFrame({
        "question_id": [f"q_{i}" for i in range(n)],
        "paper_id": papers,
        "bm25_mean_score": np.random.uniform(5.0, 25.0, n),
        "lex_coverage_top2": np.random.uniform(0.3, 0.9, n),
        "sem_sim_mean_top6": np.random.uniform(0.4, 0.9, n),
        "evidence_query_cluster_span": np.random.randint(1, 4, n),
        "evidence_lexical_overlap_mean": np.random.uniform(0.1, 0.5, n),
        "F1_k2": np.random.uniform(0.1, 0.5, n),
        "F1_k4": np.random.uniform(0.2, 0.6, n),
        "F1_k5": np.random.uniform(0.2, 0.6, n),
        "F1_k6": np.random.uniform(0.2, 0.7, n),
        "F1_k8": np.random.uniform(0.2, 0.7, n),
        "oracle_f1": np.random.uniform(0.5, 0.8, n),
    })

    df["G_4_6"] = df["F1_k6"] - df["F1_k4"]
    df["Y_4_6"] = 0
    # Evenly distribute positive labels across papers (1 per paper, 20% positive rate)
    df.loc[df.index % 5 == 0, "Y_4_6"] = 1

    return df


def test_simulate_policy_c1(mock_dev_data):
    """Verify Policy C1 simulation logic and metrics."""
    n = len(mock_dev_data)
    probs = np.linspace(0.1, 0.9, n)
    df_res, summary = simulate_policy_c1(mock_dev_data, probs, threshold=0.50)

    assert len(df_res) == n
    assert "selected_k" in df_res.columns
    assert set(df_res["selected_k"].unique()).issubset({4, 6})
    assert summary["threshold"] == 0.50
    assert 0.0 <= summary["mean_f1"] <= 1.0
    assert 4.0 <= summary["mean_k"] <= 6.0
    assert summary["tp_count"] + summary["tn_count"] + summary["fp_count"] + summary["fn_count"] == n


def test_instantiate_models():
    """Verify that candidate model pipelines instantiate properly."""
    m_lr = instantiate_model("logistic_regression", {"C": 1.0, "class_weight": "balanced"})
    assert m_lr is not None

    m_dt = instantiate_model("decision_tree", {"max_depth": 3, "min_samples_leaf": 5, "class_weight": "balanced"})
    assert m_dt is not None

    m_rf = instantiate_model("random_forest", {"n_estimators": 10, "max_depth": 2, "class_weight": "balanced"})
    assert m_rf is not None

    m_hgb = instantiate_model("hist_gradient_boosting", {"max_iter": 10, "max_depth": 2, "class_weight": "balanced"})
    assert m_hgb is not None

    m_mlp = instantiate_model("mlp", {"hidden_sizes": (8,), "epochs": 5})
    assert m_mlp is not None


def test_train_eval_model_oof_logreg(mock_dev_data):
    """Verify leak-free paper-disjoint GroupKFold OOF evaluation for Logistic Regression."""
    res = train_eval_model_oof(
        mock_dev_data,
        model_family="logistic_regression",
        feature_cols=STAGE_2_CORE_FEATURES,
        n_outer_splits=3,
        n_inner_splits=2,
        seed=42,
    )
    assert res["model_family"] == "logistic_regression"
    assert 0.0 <= res["roc_auc"] <= 1.0
    assert 0.0 <= res["balanced_acc"] <= 1.0
    assert len(res["oof_probs"]) == len(mock_dev_data)
    assert "policy_summary" in res
    assert "policy_summary_tuned" in res


def test_train_eval_model_oof_decision_tree(mock_dev_data):
    """Verify leak-free decision tree evaluation with inner-CV tuning."""
    res = train_eval_model_oof(
        mock_dev_data,
        model_family="decision_tree",
        feature_cols=STAGE_2_CORE_FEATURES,
        n_outer_splits=3,
        n_inner_splits=2,
        seed=42,
    )
    assert res["model_family"] == "decision_tree"
    assert len(res["fold_best_configs"]) == 3
    assert "max_depth" in res["fold_best_configs"][0]


def test_train_eval_model_oof_random_forest(mock_dev_data):
    """Verify leak-free random forest evaluation."""
    res = train_eval_model_oof(
        mock_dev_data,
        model_family="random_forest",
        feature_cols=STAGE_2_CORE_FEATURES,
        n_outer_splits=3,
        n_inner_splits=2,
        seed=42,
    )
    assert res["model_family"] == "random_forest"
    assert len(res["oof_probs"]) == len(mock_dev_data)


def test_train_eval_model_oof_hist_gradient_boosting(mock_dev_data):
    """Verify HistGradientBoosting evaluation."""
    res = train_eval_model_oof(
        mock_dev_data,
        model_family="hist_gradient_boosting",
        feature_cols=STAGE_2_CORE_FEATURES,
        n_outer_splits=3,
        n_inner_splits=2,
        seed=42,
    )
    assert res["model_family"] == "hist_gradient_boosting"


def test_train_eval_model_oof_lightgbm(mock_dev_data):
    """Verify LightGBM evaluation."""
    res = train_eval_model_oof(
        mock_dev_data,
        model_family="lightgbm",
        feature_cols=STAGE_2_CORE_FEATURES,
        n_outer_splits=3,
        n_inner_splits=2,
        seed=42,
    )
    assert res["model_family"] == "lightgbm"
    assert len(res["oof_probs"]) == len(mock_dev_data)


def test_train_eval_model_oof_xgboost(mock_dev_data):
    """Verify XGBoost evaluation."""
    res = train_eval_model_oof(
        mock_dev_data,
        model_family="xgboost",
        feature_cols=STAGE_2_CORE_FEATURES,
        n_outer_splits=3,
        n_inner_splits=2,
        seed=42,
    )
    assert res["model_family"] == "xgboost"
    assert len(res["oof_probs"]) == len(mock_dev_data)


def test_small_pytorch_mlp(mock_dev_data):
    """Verify PyTorch Small MLP module."""
    X = mock_dev_data[STAGE_2_CORE_FEATURES].values
    y = mock_dev_data["Y_4_6"].values

    mlp = SmallPyTorchMLP(hidden_sizes=(8,), epochs=10, batch_size=16, random_state=42)
    mlp.fit(X, y)
    probs = mlp.predict_proba(X)

    assert probs.shape == (len(X), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    preds = mlp.predict(X)
    assert len(preds) == len(X)


def test_evaluate_model_threshold_grid(mock_dev_data):
    """Verify threshold sensitivity grid evaluation."""
    n = len(mock_dev_data)
    probs = np.random.uniform(0.1, 0.9, n)
    df_grid = evaluate_model_threshold_grid(mock_dev_data, probs, "TestModel", thresholds=[0.3, 0.5, 0.7])

    assert len(df_grid) == 3
    assert "false_expansion_pct" in df_grid.columns
    assert "false_stop_pct" in df_grid.columns


def test_run_paper_clustered_paired_bootstrap(mock_dev_data):
    """Verify paper-clustered paired bootstrap comparisons."""
    n = len(mock_dev_data)
    cand_res = pd.DataFrame({
        "policy_f1": mock_dev_data["F1_k6"].values,
        "context_tokens": np.full(n, 1359.1),
        "regret": np.zeros(n),
    })
    base_res = pd.DataFrame({
        "policy_f1": mock_dev_data["F1_k4"].values,
        "context_tokens": np.full(n, 946.1),
        "regret": np.zeros(n),
    })

    res = run_paper_clustered_paired_bootstrap(
        mock_dev_data,
        cand_res,
        base_res,
        n_bootstrap=50,
        seed=42,
    )
    assert "mean_delta_f1" in res
    assert "ci_95_f1_low" in res
    assert "ci_95_f1_high" in res
    assert "mean_delta_tokens" in res


def test_analyze_feature_interactions(mock_dev_data):
    """Verify 2D interaction grid generation."""
    xx, yy, grid_probs = analyze_feature_interactions(
        mock_dev_data,
        model_family="decision_tree",
        grid_size=5,
        seed=42,
    )
    assert xx.shape == (5, 5)
    assert yy.shape == (5, 5)
    assert grid_probs.shape == (5, 5)
    assert 0.0 <= np.min(grid_probs) <= np.max(grid_probs) <= 1.0
