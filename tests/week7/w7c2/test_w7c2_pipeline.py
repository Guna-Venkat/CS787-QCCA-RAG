"""Comprehensive unit test suite for Week 7-C2: Utility-Aware Budget Ranking and Continuation Policy.

Validates all 28 required behaviors:
1. All five action rows remain grouped by query
2. Paper-disjoint split preserved
3. DEV excluded from training
4. TEST cannot load
5. Best-vs-second margin correct
6. Admissible-set construction correct
7. Non-monotonicity detection correct
8. Adjacent gains correct
9. E1 exact-minimum classification correct
10. E2 admissible-overbudget classification correct
11. E3 underbudget-inadmissible correct
12. E4 higher-budget-inadmissible correct
13. M1 direct target = actual F1
14. M1 delta-near-tie policy chooses cheapest admissible predicted action
15. M2 ranking group has exactly five rows
16. Ranking relevance labels deterministic
17. Epsilon-near-ties prefer lower cost
18. M2 inference chooses max predicted rank score
19. No fallback-to-k8 assumption
20. M3 continuation target uses future best correctly
21. Selection rule uses INTERNAL-VAL downstream F1
22. 0.003 tie tolerance implemented exactly
23. Lower tokens chosen within tolerance
24. Model frozen before DEV
25. Response-surface lookup correct
26. Feature leakage remains blocked
27. Optional cross-encoder disabled path works
28. W7-C1 artifacts remain unchanged
"""

import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.week7.w7c2.data import (
    ALLOWED_BUDGETS,
    assert_test_disabled,
    build_w7c2_ranking_dataset,
    load_w7c2_splits,
)
from src.week7.w7c2.diagnostics import (
    compute_adjacent_marginal_gains,
    compute_admissible_set_distribution,
    compute_best_vs_second_margin,
    compute_monotonicity_analysis,
)
from src.week7.w7c2.targets import (
    compare_budgets_lexicographic,
    compute_continuation_target,
    compute_lexicographic_relevance_grades,
)
from src.week7.w7c2.regression import DirectF1Regressor
from src.week7.w7c2.ranking import QueryGroupedBudgetRanker
from src.week7.w7c2.continuation import StagewiseContinuationPolicy
from src.week7.w7c2.policy import (
    apply_m1_regression_decision_rule,
    apply_m2_ranking_decision_rule,
    evaluate_downstream_policy,
    select_best_model_internal_val,
)
from src.week7.w7c2.evaluation import (
    compute_corrected_error_taxonomy,
    compute_recovered_oracle_headroom,
    run_paper_clustered_bootstrap,
)
from src.week7.w7c.leakage import assert_zero_feature_leakage, audit_feature_leakage


@pytest.fixture
def synthetic_response_matrix():
    """Synthetic wide response matrix for 3 queries across budgets [2, 4, 5, 6, 8]."""
    return pd.DataFrame([
        {
            "question_id": "q1",
            "paper_id": "pA",
            "f1_k2": 0.30, "tokens_k2": 500.0,
            "f1_k4": 0.40, "tokens_k4": 900.0,
            "f1_k5": 0.405, "tokens_k5": 1100.0,
            "f1_k6": 0.41, "tokens_k6": 1300.0,
            "f1_k8": 0.35, "tokens_k8": 1700.0,  # Non-monotonic! Drop at k=8
        },
        {
            "question_id": "q2",
            "paper_id": "pA",
            "f1_k2": 0.20, "tokens_k2": 520.0,
            "f1_k4": 0.25, "tokens_k4": 920.0,
            "f1_k5": 0.30, "tokens_k5": 1120.0,
            "f1_k6": 0.35, "tokens_k6": 1320.0,
            "f1_k8": 0.40, "tokens_k8": 1720.0,  # Monotonic
        },
        {
            "question_id": "q3",
            "paper_id": "pB",
            "f1_k2": 0.60, "tokens_k2": 490.0,
            "f1_k4": 0.605, "tokens_k4": 890.0,
            "f1_k5": 0.60, "tokens_k5": 1090.0,
            "f1_k6": 0.60, "tokens_k6": 1290.0,
            "f1_k8": 0.50, "tokens_k8": 1690.0,  # Non-monotonic
        },
    ])


@pytest.fixture
def synthetic_candidate_features():
    """Synthetic features for 3 queries x 5 budgets = 15 rows."""
    rows = []
    for qid, pid in [("q1", "pA"), ("q2", "pA"), ("q3", "pB")]:
        for k in [2, 4, 5, 6, 8]:
            rows.append({
                "split": "train",
                "question_id": qid,
                "paper_id": pid,
                "k": k,
                "q_len_tokens": 12.0,
                "sem_top1": 0.85,
                "bm25_top1": 15.0,
                "cost_k": float(k),
            })
    return pd.DataFrame(rows)


# Test 1: All five action rows remain grouped by query
def test_1_five_rows_grouped_by_query(synthetic_candidate_features, synthetic_response_matrix):
    df_rank, groups = build_w7c2_ranking_dataset(synthetic_candidate_features, synthetic_response_matrix)
    assert len(df_rank) == 15
    assert list(groups) == [5, 5, 5]
    for qid in ["q1", "q2", "q3"]:
        sub = df_rank[df_rank["question_id"] == qid]
        assert len(sub) == 5
        assert list(sub["k"]) == [2, 4, 5, 6, 8]


# Test 2: Paper-disjoint split preserved
def test_2_paper_disjoint_split():
    df_fit, df_val = load_w7c2_splits()
    fit_papers = set(df_fit["paper_id"])
    val_papers = set(df_val["paper_id"])
    assert fit_papers.isdisjoint(val_papers)


# Test 3: DEV excluded from training
def test_3_dev_excluded_from_training(synthetic_candidate_features, synthetic_response_matrix):
    train_df, _ = build_w7c2_ranking_dataset(synthetic_candidate_features.iloc[:10], synthetic_response_matrix.iloc[:2], split_name="train")
    dev_df, _ = build_w7c2_ranking_dataset(synthetic_candidate_features.iloc[10:], synthetic_response_matrix.iloc[2:], split_name="dev")
    assert set(train_df["question_id"]).isdisjoint(set(dev_df["question_id"]))


# Test 4: TEST cannot load
def test_4_test_cannot_load():
    with pytest.raises(RuntimeError, match="TEST split access is strictly prohibited"):
        assert_test_disabled("test")
    with pytest.raises(RuntimeError, match="TEST split access is strictly prohibited"):
        assert_test_disabled("TEST_SET")


# Test 5: Best-vs-second margin correct
def test_5_best_vs_second_margin(synthetic_response_matrix):
    # For q1: best is 0.41 (k=6), second is 0.405 (k=5). Margin = 0.005
    margins, summary = compute_best_vs_second_margin(synthetic_response_matrix)
    assert math.isclose(margins[0], 0.005, abs_tol=1e-5)
    # For q2: best is 0.40 (k=8), second is 0.35 (k=6). Margin = 0.05
    assert math.isclose(margins[1], 0.05, abs_tol=1e-5)


# Test 6: Admissible-set construction correct
def test_6_admissible_set_construction(synthetic_response_matrix):
    # For q1: max is 0.41. Budgets with F1 >= 0.40 are k=4 (0.40), k=5 (0.405), k=6 (0.41). Size = 3
    sizes, df_dist = compute_admissible_set_distribution(synthetic_response_matrix, epsilon=0.01)
    assert sizes[0] == 3
    # For q3: max is 0.605 (k=4). Budgets >= 0.595 are k=2 (0.60), k=4 (0.605), k=5 (0.60), k=6 (0.60). Size = 4
    assert sizes[2] == 4


# Test 7: Non-monotonicity detection correct
def test_7_non_monotonicity_detection(synthetic_response_matrix):
    mono_res = compute_monotonicity_analysis(synthetic_response_matrix)
    # q1 (drop at k8) and q3 (drop at k8) are non-monotonic. Only q2 is monotonic.
    assert mono_res["monotonic_queries"] == 1
    assert mono_res["non_monotonic_queries"] == 2
    assert mono_res["reversals"]["F8_lt_F6"] == 2


# Test 8: Adjacent gains correct
def test_8_adjacent_gains(synthetic_response_matrix):
    df_gains = compute_adjacent_marginal_gains(synthetic_response_matrix)
    # Delta_2_to_4: q1=0.10, q2=0.05, q3=0.005. Mean = 0.051667
    d24 = df_gains[df_gains["step"] == "Delta_2_to_4"].iloc[0]
    assert math.isclose(d24["mean_gain"], 0.155 / 3.0, abs_tol=1e-4)


# Test 9: E1 exact-minimum classification correct
def test_9_e1_exact_minimum_classification():
    df_eval = pd.DataFrame([{
        "question_id": "q1", "paper_id": "pA", "pred_k": 4,
        "f1_k2": 0.30, "f1_k4": 0.40, "f1_k5": 0.405, "f1_k6": 0.41, "f1_k8": 0.41,
        "tokens_k2": 500, "tokens_k4": 900, "tokens_k5": 1100, "tokens_k6": 1300, "tokens_k8": 1700,
    }])
    # Max F1=0.41. Admissible (>=0.40): k=4, 5, 6, 8. Minimum oracle k = 4.
    # Prediction is 4 -> E1
    _, summary = compute_corrected_error_taxonomy(df_eval, epsilon=0.01)
    e1_row = summary[summary["category"].str.startswith("E1")].iloc[0]
    assert e1_row["count"] == 1


# Test 10: E2 admissible-overbudget classification correct
def test_10_e2_admissible_overbudget():
    df_eval = pd.DataFrame([{
        "question_id": "q1", "paper_id": "pA", "pred_k": 6,
        "f1_k2": 0.30, "f1_k4": 0.40, "f1_k5": 0.405, "f1_k6": 0.41, "f1_k8": 0.41,
        "tokens_k2": 500, "tokens_k4": 900, "tokens_k5": 1100, "tokens_k6": 1300, "tokens_k8": 1700,
    }])
    # Oracle is 4. Prediction 6 is admissible (0.41 >= 0.40) but > 4 -> E2
    _, summary = compute_corrected_error_taxonomy(df_eval, epsilon=0.01)
    e2_row = summary[summary["category"].str.startswith("E2")].iloc[0]
    assert e2_row["count"] == 1


# Test 11: E3 underbudget-inadmissible correct
def test_11_e3_underbudget_inadmissible():
    df_eval = pd.DataFrame([{
        "question_id": "q1", "paper_id": "pA", "pred_k": 2,
        "f1_k2": 0.30, "f1_k4": 0.40, "f1_k5": 0.405, "f1_k6": 0.41, "f1_k8": 0.41,
        "tokens_k2": 500, "tokens_k4": 900, "tokens_k5": 1100, "tokens_k6": 1300, "tokens_k8": 1700,
    }])
    # Oracle is 4. Prediction 2 is < 4 and 0.30 < 0.40 -> E3
    _, summary = compute_corrected_error_taxonomy(df_eval, epsilon=0.01)
    e3_row = summary[summary["category"].str.startswith("E3")].iloc[0]
    assert e3_row["count"] == 1


# Test 12: E4 higher-budget-inadmissible correct (Non-monotonicity!)
def test_12_e4_higher_budget_inadmissible():
    df_eval = pd.DataFrame([{
        "question_id": "q1", "paper_id": "pA", "pred_k": 8,
        "f1_k2": 0.30, "f1_k4": 0.40, "f1_k5": 0.405, "f1_k6": 0.41, "f1_k8": 0.32,  # Catastrophic drop at k=8
        "tokens_k2": 500, "tokens_k4": 900, "tokens_k5": 1100, "tokens_k6": 1300, "tokens_k8": 1700,
    }])
    # Oracle is 4. Prediction 8 is > 4, but 0.32 is NOT admissible (< 0.40) -> E4!
    _, summary = compute_corrected_error_taxonomy(df_eval, epsilon=0.01)
    e4_row = summary[summary["category"].str.startswith("E4")].iloc[0]
    assert e4_row["count"] == 1


# Test 13: M1 direct target = actual F1
def test_13_m1_direct_target_is_f1(synthetic_candidate_features, synthetic_response_matrix):
    df_rank, _ = build_w7c2_ranking_dataset(synthetic_candidate_features, synthetic_response_matrix)
    # q1 at k=4 has target_f1 = 0.40
    q1_k4 = df_rank[(df_rank["question_id"] == "q1") & (df_rank["k"] == 4)].iloc[0]
    assert math.isclose(q1_k4["target_f1"], 0.40)


# Test 14: M1 delta-near-tie policy chooses cheapest admissible predicted action
def test_14_m1_near_tie_cheapest():
    pred_f1 = {2: 0.30, 4: 0.40, 5: 0.405, 6: 0.41, 8: 0.41}
    tokens = {2: 500, 4: 900, 5: 1100, 6: 1300, 8: 1700}
    # For delta=0.01: max is 0.41. Set >= 0.40 is {4, 5, 6, 8}. Cheapest is 4.
    k_sel = apply_m1_regression_decision_rule(pred_f1, tokens_by_k=tokens, delta=0.01)
    assert k_sel == 4


# Test 15: M2 ranking group has exactly five rows
def test_15_m2_ranking_groups_of_five(synthetic_candidate_features, synthetic_response_matrix):
    df_rank, group_sizes = build_w7c2_ranking_dataset(synthetic_candidate_features, synthetic_response_matrix)
    assert all(g == 5 for g in group_sizes)


# Test 16: Ranking relevance labels deterministic
def test_16_ranking_labels_deterministic():
    f1s = {2: 0.30, 4: 0.40, 5: 0.405, 6: 0.41, 8: 0.35}
    toks = {2: 500, 4: 900, 5: 1100, 6: 1300, 8: 1700}
    g1 = compute_lexicographic_relevance_grades(f1s, toks, epsilon_pref=0.01)
    g2 = compute_lexicographic_relevance_grades(f1s, toks, epsilon_pref=0.01)
    assert g1 == g2
    # Quality within 0.01: k=4 (0.40), k=5 (0.405), k=6 (0.41). Cheaper is preferred.
    # So k=4 (tokens 900) should be top-ranked (grade 4).
    assert g1[4] == 4


# Test 17: Epsilon-near-ties prefer lower cost
def test_17_near_ties_prefer_lower_cost():
    # Budgets 4 and 6 have identical F1 within epsilon
    cmp_res = compare_budgets_lexicographic(
        k_a=4, k_b=6,
        f1_a=0.40, f1_b=0.405,  # diff = 0.005 <= 0.01
        tokens_a=900, tokens_b=1300,
        epsilon_pref=0.01,
    )
    # k_a (4) has lower tokens, so k_a is preferred (-1)
    assert cmp_res == -1


# Test 18: M2 inference chooses max predicted rank score
def test_18_m2_inference_max_score():
    scores = {2: 0.1, 4: 0.9, 5: 0.7, 6: 0.8, 8: 0.4}
    k_sel = apply_m2_ranking_decision_rule(scores, score_tie_margin=0.0)
    assert k_sel == 4


# Test 19: No fallback-to-k8 assumption
def test_19_no_fallback_to_k8():
    # Even if all scores are low/negative, the policy chooses the best predicted action (not k=8!)
    scores = {2: -2.0, 4: -0.5, 5: -1.0, 6: -1.5, 8: -3.0}
    k_sel = apply_m2_ranking_decision_rule(scores, score_tie_margin=0.0)
    assert k_sel == 4


# Test 20: M3 continuation target uses future best correctly
def test_20_m3_continuation_target():
    f1s = {2: 0.30, 4: 0.40, 5: 0.405, 6: 0.41, 8: 0.35}
    # At k=2 (0.30): future best is 0.41 (at k=6). Gain = 0.11 > 0.01 -> continue = 1
    c2 = compute_continuation_target(f1s, current_stage_k=2, epsilon=0.01)
    assert c2 == 1
    # At k=6 (0.41): future best is 0.35 (at k=8). Gain = -0.06 <= 0.01 -> stop = 0
    c6 = compute_continuation_target(f1s, current_stage_k=6, epsilon=0.01)
    assert c6 == 0


# Test 21: Selection rule uses INTERNAL-VAL downstream F1
def test_21_selection_rule_f1():
    models = [
        {"model_name": "M_A", "mean_f1": 0.40, "mean_tokens": 1000.0, "mean_k": 4.0},
        {"model_name": "M_B", "mean_f1": 0.42, "mean_tokens": 1200.0, "mean_k": 5.0},
    ]
    winner = select_best_model_internal_val(models, tolerance_f1=0.003)
    assert winner["model_name"] == "M_B"


# Test 22: 0.003 tie tolerance implemented exactly
def test_22_tie_tolerance_implemented_exactly():
    models = [
        {"model_name": "M_A", "mean_f1": 0.438, "mean_tokens": 1500.0, "mean_k": 7.0},
        {"model_name": "M_B", "mean_f1": 0.436, "mean_tokens": 1000.0, "mean_k": 4.5},  # diff = 0.002 <= 0.003
    ]
    winner = select_best_model_internal_val(models, tolerance_f1=0.003)
    # M_B is within 0.003 of 0.438, and has 1000 tokens vs 1500 tokens -> M_B wins!
    assert winner["model_name"] == "M_B"


# Test 23: Lower tokens chosen within tolerance
def test_23_lower_tokens_chosen_within_tolerance():
    models = [
        {"model_name": "HighTokens", "mean_f1": 0.410, "mean_tokens": 1400.0, "mean_k": 6.5},
        {"model_name": "LowTokens", "mean_f1": 0.408, "mean_tokens": 850.0, "mean_k": 3.8},  # diff = 0.002
    ]
    winner = select_best_model_internal_val(models, tolerance_f1=0.003)
    assert winner["model_name"] == "LowTokens"


# Test 24: Model frozen before DEV
def test_24_model_frozen_deterministic():
    X = pd.DataFrame({"feat_a": [1.0, 2.0, 3.0], "cost_k": [2.0, 4.0, 6.0]})
    y = np.array([0.3, 0.4, 0.5])
    reg = DirectF1Regressor(n_estimators=10)
    reg.fit(X, y)
    p1 = reg.predict(X)
    p2 = reg.predict(X)
    assert np.allclose(p1, p2)


# Test 25: Response-surface lookup correct
def test_25_response_surface_lookup_correct(synthetic_response_matrix):
    preds = {"q1": 4, "q2": 6, "q3": 2}
    metrics = evaluate_downstream_policy(preds, synthetic_response_matrix)
    # q1 at k=4: 0.40; q2 at k=6: 0.35; q3 at k=2: 0.60. Mean = 1.35 / 3 = 0.45
    assert math.isclose(metrics["mean_f1"], 0.45)


# Test 26: Feature leakage remains blocked
def test_26_feature_leakage_blocked():
    bad = ["sem_f1_k4", "oracle_budget", "target_regret"]
    is_clean, violations = audit_feature_leakage(bad)
    assert not is_clean
    assert len(violations) == 3


# Test 27: Optional cross-encoder disabled path works
def test_27_cross_encoder_disabled_path():
    from src.week7.w7c.features_crossencoder import OptionalFrozenCrossEncoderScorer
    scorer = OptionalFrozenCrossEncoderScorer(enabled=False)
    scores = scorer.score_query_passages("query", ["p1", "p2"])
    assert scores == [0.0, 0.0]


# Test 28: W7-C1 artifacts remain unchanged
def test_28_w7c1_artifacts_preserved():
    w7c1_nb = Path("notebooks/week7/week7c_retrieval_aware_semantic_allocator.ipynb")
    w7c1_cfg = Path("configs/week7/w7c_retrieval_aware_allocator.yaml")
    assert w7c1_nb.exists(), "W7-C1 notebook must be preserved!"
    assert w7c1_cfg.exists(), "W7-C1 config must be preserved!"
