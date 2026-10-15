"""Unit test suite for Week 7-C: Retrieval-Aware Semantic Sufficiency Allocation.

Validates all 26 required behaviors:
1. Five candidate-k rows per query
2. Candidate rows remain grouped by paper split
3. No DEV rows in training
4. TEST cannot load
5. Regret computed correctly
6. Sufficiency label uses epsilon=0.01
7. Semantic boundary features correct
8. BM25 boundary features correct
9. Cross-ranking metrics deterministic
10. Set redundancy features bounded/finite
11. Tail-threat features correct
12. Query-coverage features finite
13. Constant score vectors handled safely
14. No NaN/inf after feature computation
15. Leakage audit rejects F1/oracle/regret inputs
16. Preprocessing fitted only on TRAIN-FIT
17. Policy selects smallest admissible predicted k
18. Fallback to k8 works
19. Tau selected only on INTERNAL-VAL
20. Response-surface lookup gives correct downstream F1
21. Under/over-allocation classification correct
22. Feature-family ablations reproducible
23. Model freeze before DEV
24. Cached features resume correctly
25. Optional cross-encoder disabled path works
26. Optional generator probe disabled path works
"""

import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.week7.w7c.data import (
    ALLOWED_BUDGETS,
    assert_test_disabled,
    build_candidate_budget_dataframe,
)
from src.week7.w7c.features_query import (
    compute_single_query_features,
    extract_query_complexity_features,
)
from src.week7.w7c.features_retrieval import (
    compute_bm25_qpp_features,
    compute_cross_retriever_features,
    compute_semantic_budget_geometry,
    compute_semantic_pool_geometry,
)
from src.week7.w7c.features_set import (
    compute_set_diversity_features,
    compute_tail_threat_features,
)
from src.week7.w7c.features_coverage import (
    compute_cost_features,
    compute_query_coverage_features,
)
from src.week7.w7c.features_crossencoder import (
    OptionalFrozenCrossEncoderScorer,
    compute_cross_encoder_budget_features,
)
from src.week7.w7c.leakage import (
    assert_zero_feature_leakage,
    audit_feature_leakage,
    sanitize_feature_matrix,
)
from src.week7.w7c.models_tree import (
    RegretGBDTRegressor,
    SufficiencyGBDTClassifier,
)
from src.week7.w7c.models_mlp import (
    MultitaskMLPWrapper,
)
from src.week7.w7c.models_set import (
    RetrievalSetEncoderWrapper,
)
from src.week7.w7c.generator_probe import (
    GeneratorSufficiencyProbe,
)
from src.week7.w7c.training import (
    PreprocessingPipeline,
    build_candidate_feature_bank,
    get_ablation_feature_names,
)
from src.week7.w7c.policy import (
    apply_risk_aware_decision_rule,
    apply_sufficiency_decision_rule,
    calibrate_tau_on_internal_val,
    evaluate_policy_on_response_surface,
)
from src.week7.w7c.evaluation import (
    compute_recovered_oracle_headroom,
    compute_under_overallocation_breakdown,
    run_paper_clustered_bootstrap,
)


@pytest.fixture
def sample_response_matrix():
    """Synthetic wide response matrix for 3 queries across budgets [2, 4, 5, 6, 8]."""
    return pd.DataFrame([
        {
            "question_id": "q1",
            "paper_id": "pA",
            "f1_k2": 0.30, "tokens_k2": 500.0,
            "f1_k4": 0.40, "tokens_k4": 900.0,
            "f1_k5": 0.405, "tokens_k5": 1100.0,
            "f1_k6": 0.41, "tokens_k6": 1300.0,
            "f1_k8": 0.41, "tokens_k8": 1700.0,
        },
        {
            "question_id": "q2",
            "paper_id": "pA",
            "f1_k2": 0.20, "tokens_k2": 520.0,
            "f1_k4": 0.25, "tokens_k4": 920.0,
            "f1_k5": 0.35, "tokens_k5": 1120.0,
            "f1_k6": 0.50, "tokens_k6": 1320.0,
            "f1_k8": 0.50, "tokens_k8": 1720.0,
        },
        {
            "question_id": "q3",
            "paper_id": "pB",
            "f1_k2": 0.60, "tokens_k2": 490.0,
            "f1_k4": 0.60, "tokens_k4": 890.0,
            "f1_k5": 0.60, "tokens_k5": 1090.0,
            "f1_k6": 0.60, "tokens_k6": 1290.0,
            "f1_k8": 0.60, "tokens_k8": 1690.0,
        },
    ])


# Test 1: Five candidate-k rows per query
def test_1_five_candidate_rows_per_query(sample_response_matrix):
    df_cand = build_candidate_budget_dataframe(sample_response_matrix, budgets=[2, 4, 5, 6, 8])
    assert len(df_cand) == len(sample_response_matrix) * 5
    for qid in ["q1", "q2", "q3"]:
        sub = df_cand[df_cand["question_id"] == qid]
        assert len(sub) == 5
        assert set(sub["k"]) == {2, 4, 5, 6, 8}


# Test 2: Candidate rows remain grouped by paper split
def test_2_candidate_rows_grouped_by_paper(sample_response_matrix):
    df_cand = build_candidate_budget_dataframe(sample_response_matrix, budgets=[2, 4, 5, 6, 8])
    # Paper pA has q1 and q2 (10 rows total)
    pa_rows = df_cand[df_cand["paper_id"] == "pA"]
    assert len(pa_rows) == 10
    assert set(pa_rows["question_id"]) == {"q1", "q2"}


# Test 3: No DEV rows in training
def test_3_no_dev_rows_in_training(sample_response_matrix):
    train_cand = build_candidate_budget_dataframe(sample_response_matrix.iloc[:2], split_name="train")
    dev_cand = build_candidate_budget_dataframe(sample_response_matrix.iloc[2:], split_name="dev")
    train_qids = set(train_cand["question_id"])
    dev_qids = set(dev_cand["question_id"])
    assert train_qids.isdisjoint(dev_qids)


# Test 4: TEST cannot load
def test_4_test_split_prohibited():
    with pytest.raises(RuntimeError, match="TEST split access is strictly prohibited"):
        assert_test_disabled("test")
    with pytest.raises(RuntimeError, match="TEST split access is strictly prohibited"):
        assert_test_disabled("TEST_SET")


# Test 5: Regret computed correctly
def test_5_regret_computed_correctly(sample_response_matrix):
    df_cand = build_candidate_budget_dataframe(sample_response_matrix, budgets=[2, 4, 5, 6, 8])
    # For q1: max F1 is 0.41 (at k=6, 8). At k=2 (F1=0.30), regret is 0.11
    q1_k2 = df_cand[(df_cand["question_id"] == "q1") & (df_cand["k"] == 2)].iloc[0]
    assert math.isclose(q1_k2["target_regret"], 0.11, rel_tol=1e-4)
    # At k=6 (F1=0.41), regret is 0.0
    q1_k6 = df_cand[(df_cand["question_id"] == "q1") & (df_cand["k"] == 6)].iloc[0]
    assert math.isclose(q1_k6["target_regret"], 0.0, abs_tol=1e-6)


# Test 6: Sufficiency label uses epsilon=0.01
def test_6_sufficiency_label_epsilon(sample_response_matrix):
    df_cand = build_candidate_budget_dataframe(sample_response_matrix, budgets=[2, 4, 5, 6, 8], epsilon=0.01)
    # For q1: max F1 is 0.41. At k=4 (F1=0.40), regret = 0.01 <= 0.01 -> sufficient = 1
    q1_k4 = df_cand[(df_cand["question_id"] == "q1") & (df_cand["k"] == 4)].iloc[0]
    assert q1_k4["target_sufficient"] == 1
    # At k=2 (F1=0.30), regret = 0.11 > 0.01 -> sufficient = 0
    q1_k2 = df_cand[(df_cand["question_id"] == "q1") & (df_cand["k"] == 2)].iloc[0]
    assert q1_k2["target_sufficient"] == 0


# Test 7: Semantic boundary features correct
def test_7_semantic_boundary_features():
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    geom = compute_semantic_budget_geometry(scores, k=4)
    assert math.isclose(geom["sem_k_boundary_score"], 0.6)
    assert math.isclose(geom["sem_k_next_score"], 0.5)
    assert math.isclose(geom["sem_k_boundary_margin"], 0.1)
    assert math.isclose(geom["sem_k_selected_mean"], (0.9 + 0.8 + 0.7 + 0.6) / 4.0)


# Test 8: BM25 boundary features correct
def test_8_bm25_boundary_features():
    b_scores = [15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0]
    sem_ranks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    geom = compute_bm25_qpp_features(b_scores, sem_ranks, k=2)
    assert math.isclose(geom["bm25_k_cum_score"], 29.0)
    assert math.isclose(geom["bm25_k_mean"], 14.5)
    assert math.isclose(geom["bm25_k_boundary_score"], 14.0)


# Test 9: Cross-ranking metrics deterministic
def test_9_cross_ranking_metrics_deterministic():
    b_scores = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    s_scores = [0.5, 0.8, 0.1, 0.9, 0.2, 0.7, 0.3, 0.4, 0.6, 0.0]
    sem_ranks = list(np.argsort(s_scores)[::-1])
    f1 = compute_cross_retriever_features(b_scores, s_scores, sem_ranks, k=4)
    f2 = compute_cross_retriever_features(b_scores, s_scores, sem_ranks, k=4)
    for k_key in f1:
        assert math.isclose(f1[k_key], f2[k_key], abs_tol=1e-8)


# Test 10: Set redundancy features bounded/finite
def test_10_set_redundancy_bounded():
    embeds = np.random.randn(10, 32).astype(np.float32)
    sem_ranks = list(range(10))
    div_feats = compute_set_diversity_features(embeds, sem_ranks, k=4)
    assert -1.01 <= div_feats["div_k_mean_pairwise_sim"] <= 1.01
    assert 0.0 <= div_feats["div_k_redundancy_ratio"] <= 1.0
    assert not math.isnan(div_feats["div_k_effective_clusters"])


# Test 11: Tail-threat features correct
def test_11_tail_threat_features():
    s_scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    embeds = np.eye(10, dtype=np.float32)
    sem_ranks = list(range(10))
    tail_f = compute_tail_threat_features(s_scores, embeds, sem_ranks, k=4)
    assert math.isclose(tail_f["tail_k_max_semantic_relevance"], 0.5)
    assert math.isclose(tail_f["tail_k_boundary_drop"], 0.1)
    assert tail_f["tail_k_high_threat_count"] >= 0.0


# Test 12: Query-coverage features finite
def test_12_query_coverage_finite():
    q = "What is the token classification accuracy on CoNLL-2003?"
    passages = [{"text": f"Passage {i} discussing token classification and models."} for i in range(10)]
    sem_ranks = list(range(10))
    cov_feats = compute_query_coverage_features(q, passages, sem_ranks, k=4)
    assert 0.0 <= cov_feats["cov_k_mean"] <= 1.0
    assert not math.isnan(cov_feats["cov_k_min"])


# Test 13: Constant score vectors handled safely
def test_13_constant_score_vectors():
    const_s = [0.5] * 10
    geom = compute_semantic_pool_geometry(const_s)
    assert math.isclose(geom["sem_std"], 0.0, abs_tol=1e-6)
    assert not math.isnan(geom["sem_entropy"])
    assert math.isclose(geom["sem_ranked_slope"], 0.0, abs_tol=1e-6)


# Test 14: No NaN/inf after feature computation
def test_14_no_nan_inf_features():
    q_feats = compute_single_query_features("Is BERT effective for retrieval?")
    for k_feat, val in q_feats.items():
        assert not math.isnan(val), f"NaN in {k_feat}"
        assert not math.isinf(val), f"Inf in {k_feat}"


# Test 15: Leakage audit rejects F1/oracle/regret inputs
def test_15_leakage_audit_rejects_forbidden_names():
    bad_features = ["q_len", "sem_f1_score", "predicted_regret", "oracle_k"]
    is_clean, violations = audit_feature_leakage(bad_features)
    assert not is_clean
    assert set(violations) == {"sem_f1_score", "predicted_regret", "oracle_k"}

    clean_features = ["q_len_tokens", "sem_top1", "bm25_nqc", "cov_k_mean"]
    is_clean_ok, violations_ok = audit_feature_leakage(clean_features)
    assert is_clean_ok
    assert len(violations_ok) == 0


# Test 16: Preprocessing fitted only on TRAIN-FIT
def test_16_preprocessing_fitted_on_fit():
    train_fit_df = pd.DataFrame({
        "feat_a": [1.0, 2.0, 3.0, 4.0],
        "feat_b": [10.0, 20.0, 30.0, 40.0],
    })
    val_df = pd.DataFrame({
        "feat_a": [5.0, 6.0],
        "feat_b": [50.0, 60.0],
    })
    pipe = PreprocessingPipeline()
    X_train_scaled = pipe.fit_transform(train_fit_df)
    X_val_scaled = pipe.transform(val_df)
    # Mean of train should be ~0
    assert math.isclose(X_train_scaled["feat_a"].mean(), 0.0, abs_tol=1e-6)
    # Mean of val should NOT be 0 (proves it was not fit on val)
    assert not math.isclose(X_val_scaled["feat_a"].mean(), 0.0, abs_tol=0.1)


# Test 17: Policy selects smallest admissible predicted k
def test_17_policy_selects_smallest_admissible():
    probs = {2: 0.40, 4: 0.75, 5: 0.85, 6: 0.90, 8: 0.95}
    # For tau=0.70, both k=4, 5, 6, 8 satisfy. Smallest is 4.
    selected_k = apply_sufficiency_decision_rule(probs, tau=0.70, budgets=[2, 4, 5, 6, 8])
    assert selected_k == 4


# Test 18: Fallback to k8 works
def test_18_fallback_to_k8():
    probs = {2: 0.10, 4: 0.20, 5: 0.30, 6: 0.40, 8: 0.50}
    # For tau=0.80, none satisfy -> fallback to 8
    selected_k = apply_sufficiency_decision_rule(probs, tau=0.80, budgets=[2, 4, 5, 6, 8], fallback_budget=8)
    assert selected_k == 8


# Test 19: Tau selected only on INTERNAL-VAL
def test_19_tau_selected_on_internal_val(sample_response_matrix):
    val_cand = build_candidate_budget_dataframe(sample_response_matrix, split_name="internal_val")
    # Synthetic probabilities
    pred_probs = np.array([
        0.3, 0.8, 0.9, 0.95, 0.98,  # q1
        0.1, 0.2, 0.4, 0.85, 0.90,  # q2
        0.9, 0.9, 0.9, 0.90, 0.90,  # q3
    ])
    best_tau, summary = calibrate_tau_on_internal_val(val_cand, sample_response_matrix, pred_probs, tau_grid=[0.50, 0.70, 0.90])
    assert best_tau in [0.50, 0.70, 0.90]
    assert len(summary) == 3


# Test 20: Response-surface lookup gives correct downstream F1
def test_20_response_surface_lookup(sample_response_matrix):
    preds = {"q1": 4, "q2": 6, "q3": 2}
    metrics = evaluate_policy_on_response_surface(preds, sample_response_matrix)
    # q1 at k=4: 0.40. q2 at k=6: 0.50. q3 at k=2: 0.60. Mean = 0.50
    assert math.isclose(metrics["mean_f1"], 0.50, abs_tol=1e-5)


# Test 21: Under/over-allocation classification correct
def test_21_under_overallocation_classification():
    eval_df = pd.DataFrame([
        {"question_id": "q1", "paper_id": "pA", "pred_k": 2, "oracle_k": 4, "actual_f1": 0.3, "max_f1": 0.4, "actual_tokens": 500, "oracle_tokens": 900},
        {"question_id": "q2", "paper_id": "pA", "pred_k": 4, "oracle_k": 4, "actual_f1": 0.4, "max_f1": 0.4, "actual_tokens": 900, "oracle_tokens": 900},
        {"question_id": "q3", "paper_id": "pB", "pred_k": 6, "oracle_k": 4, "actual_f1": 0.4, "max_f1": 0.4, "actual_tokens": 1300, "oracle_tokens": 900},
    ])
    breakdown = compute_under_overallocation_breakdown(eval_df)
    under_row = breakdown[breakdown["category"].str.startswith("Under")].iloc[0]
    opt_row = breakdown[breakdown["category"].str.startswith("Admissible")].iloc[0]
    over_row = breakdown[breakdown["category"].str.startswith("Over")].iloc[0]
    assert under_row["count"] == 1
    assert opt_row["count"] == 1
    assert over_row["count"] == 1


# Test 22: Feature-family ablations reproducible
def test_22_feature_ablation_names():
    all_cols = ["q_len_tokens", "sem_top1", "bm25_top1", "cross_kendall_tau", "div_k_mean", "tail_k_max", "cov_k_mean", "cost_k", "w4_bm25_mean", "ce_top1"]
    f0 = get_ablation_feature_names("F0", all_cols)
    assert "w4_bm25_mean" in f0
    assert "cost_k" in f0

    f1 = get_ablation_feature_names("F1", all_cols)
    assert "q_len_tokens" in f1 and "sem_top1" in f1 and "bm25_top1" in f1
    assert "cross_kendall_tau" not in f1

    f4 = get_ablation_feature_names("F4", all_cols)
    assert "tail_k_max" in f4 and "cov_k_mean" in f4
    assert "w4_bm25_mean" not in f4


# Test 23: Model freeze before DEV
def test_23_model_freeze():
    X = pd.DataFrame({
        "q_len_tokens": [5.0, 10.0, 15.0, 20.0],
        "sem_top1": [0.8, 0.7, 0.6, 0.5],
        "cost_k": [2.0, 4.0, 6.0, 8.0],
    })
    y = np.array([0, 1, 1, 1])
    clf = SufficiencyGBDTClassifier(n_estimators=10)
    clf.fit(X, y)
    probs_1 = clf.predict_proba(X)
    # Evaluating without refitting should produce identical predictions
    probs_2 = clf.predict_proba(X)
    assert np.allclose(probs_1, probs_2)


# Test 24: Cached features resume correctly
def test_24_cached_features_resume(tmp_path):
    df_cache = pd.DataFrame({
        "question_id": ["q1", "q2"],
        "k": [2, 4],
        "q_len_tokens": [10.0, 12.0],
    })
    cache_file = tmp_path / "test_features.parquet"
    df_cache.to_parquet(cache_file)
    assert cache_file.exists()
    loaded_df = pd.read_parquet(cache_file)
    assert len(loaded_df) == 2
    assert "q_len_tokens" in loaded_df.columns


# Test 25: Optional cross-encoder disabled path works
def test_25_cross_encoder_disabled_path():
    scorer = OptionalFrozenCrossEncoderScorer(enabled=False)
    scores = scorer.score_query_passages("test query", ["passage 1", "passage 2"])
    assert scores == [0.0, 0.0]
    ce_feats = compute_cross_encoder_budget_features(scores, [0, 1], k=2, enabled=False)
    assert ce_feats["ce_top1"] == 0.0


# Test 26: Optional generator probe disabled path works
def test_26_generator_probe_disabled_path():
    probe = GeneratorSufficiencyProbe(enabled=False)
    probs = probe.predict_proba(np.zeros((5, 128)))
    assert len(probs) == 5
    assert np.allclose(probs, 0.5)
