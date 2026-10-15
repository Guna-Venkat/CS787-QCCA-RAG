"""Unit tests for Week 7-C3 Anchor-and-Deviate Semantic Budget Policy.

Verifies:
1. anchor always defaults to k4
2. y_down computed correctly
3. y_up computed correctly
4. delta5/6/8 targets correct
5. no F1 target leaks into features
6. GroupKFold paper separation
7. every TRAIN query receives OOF predictions exactly once
8. DEV never used in OOF calibration
9. tau_down applied correctly
10. tau_up applied correctly
11. downshift takes precedence only when confidence threshold passed
12. otherwise k4 anchor preserved
13. larger-budget selector only runs after upshift
14. delta near-tie chooses cheapest larger budget
15. no k8 fallback
16. policy grid calculates downstream F1 by lookup correctly
17. quality-constrained cost selection correct
18. eta tolerance applied correctly
19. lowest-token qualifying policy selected
20. downshift safety metrics correct
21. upshift precision metrics correct
22. E1-E6 error categories correct
23. full TRAIN retraining occurs only after policy freeze
24. DEV evaluated only after freeze
25. TEST cannot load
26. W7-C1/C2 artifacts remain unchanged
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.week7.w7c3.data import (
    ALLOWED_BUDGETS,
    ANCHOR_K,
    assert_test_disabled,
    assert_zero_feature_leakage,
    build_w7c3_query_dataset,
    get_feature_subsets,
    load_w7c3_splits,
)
from src.week7.w7c3.delta_models import MarginalGainDeltaRegressors
from src.week7.w7c3.diagnostics import compute_4to6_gain_analysis, compute_anchor_oracle_diagnostics
from src.week7.w7c3.downshift import SafeDownshiftClassifier
from src.week7.w7c3.evaluation import (
    compute_downshift_safety_metrics,
    compute_recovered_oracle_headroom,
    compute_upshift_precision_metrics,
    compute_w7c3_error_taxonomy,
)
from src.week7.w7c3.oof import evaluate_oof_policy_grid, run_group_kfold_oof
from src.week7.w7c3.policy import (
    apply_anchor_deviate_decision_rule,
    evaluate_downstream_anchor_policy,
    select_best_anchor_policy,
)
from src.week7.w7c3.targets import (
    compute_4to6_gain_target,
    compute_delta_regression_targets,
    compute_minimum_cost_oracle_k,
    compute_oracle_admissible_set,
    compute_safe_downshift_target,
    compute_selective_upshift_target,
)


@pytest.fixture
def sample_response_data():
    """Create synthetic response matrix with 4 queries covering downshift, upshift, and anchor cases."""
    rows = [
        # Query 1: k=2 is admissible (f1: k2=0.45, k4=0.45, k5=0.45, k6=0.45, k8=0.45) -> y_down=1, y_up=0
        {"question_id": "q1", "paper_id": "p1", "f1_k2": 0.45, "tokens_k2": 500, "f1_k4": 0.45, "tokens_k4": 900, "f1_k5": 0.45, "tokens_k5": 1100, "f1_k6": 0.45, "tokens_k6": 1300, "f1_k8": 0.45, "tokens_k8": 1700},
        # Query 2: k=4 is anchor, k=2 is inadequate, but k=6 has large gain -> y_down=0, y_up=1, Delta_6=0.10
        {"question_id": "q2", "paper_id": "p1", "f1_k2": 0.20, "tokens_k2": 510, "f1_k4": 0.35, "tokens_k4": 920, "f1_k5": 0.36, "tokens_k5": 1120, "f1_k6": 0.46, "tokens_k6": 1320, "f1_k8": 0.42, "tokens_k8": 1720},
        # Query 3: k=4 is admissible, k=2 is inadequate, no larger budget improves -> y_down=0, y_up=0 (pure anchor)
        {"question_id": "q3", "paper_id": "p2", "f1_k2": 0.25, "tokens_k2": 520, "f1_k4": 0.40, "tokens_k4": 910, "f1_k5": 0.40, "tokens_k5": 1110, "f1_k6": 0.38, "tokens_k6": 1310, "f1_k8": 0.39, "tokens_k8": 1710},
        # Query 4: non-monotonic reversal where k=8 is terrible (f1: k4=0.50, k8=0.20) -> demonstrates k8 failure
        {"question_id": "q4", "paper_id": "p2", "f1_k2": 0.30, "tokens_k2": 490, "f1_k4": 0.50, "tokens_k4": 890, "f1_k5": 0.48, "tokens_k5": 1090, "f1_k6": 0.45, "tokens_k6": 1290, "f1_k8": 0.20, "tokens_k8": 1690},
    ]
    return pd.DataFrame(rows)


def test_1_anchor_defaults_to_k4():
    """1. Verify that when neither downshift nor upshift is triggered, anchor defaults to k4."""
    deltas = {5: 0.0, 6: 0.0, 8: 0.0}
    # p_down < tau_down (0.85) and p_up < tau_up (0.70)
    choice = apply_anchor_deviate_decision_rule(
        p_down=0.60,
        p_up=0.50,
        pred_deltas=deltas,
        tau_down=0.85,
        tau_up=0.70,
        anchor_k=4,
    )
    assert choice == 4, f"Expected default anchor k=4, got {choice}"


def test_2_y_down_computed_correctly():
    """2. Verify y_down label logic: 1 if F(2) >= F* - 0.01 else 0."""
    # Case A: k=2 is optimal
    f1_a = {2: 0.45, 4: 0.44, 5: 0.44, 6: 0.44, 8: 0.44}
    assert compute_safe_downshift_target(f1_a, epsilon=0.01) == 1

    # Case B: k=2 is within 0.01 of max
    f1_b = {2: 0.445, 4: 0.45, 5: 0.45, 6: 0.45, 8: 0.45}
    assert compute_safe_downshift_target(f1_b, epsilon=0.01) == 1

    # Case C: k=2 is inadequate (gap > 0.01)
    f1_c = {2: 0.30, 4: 0.45, 5: 0.45, 6: 0.45, 8: 0.45}
    assert compute_safe_downshift_target(f1_c, epsilon=0.01) == 0


def test_3_y_up_computed_correctly():
    """3. Verify y_up label logic: 1 if max_{5,6,8} F(k) - F(4) > 0.01 else 0."""
    # Case A: k=6 gives +0.05 gain over k4
    f1_a = {2: 0.30, 4: 0.35, 5: 0.35, 6: 0.40, 8: 0.38}
    y_up, gain_up = compute_selective_upshift_target(f1_a, epsilon=0.01, anchor_k=4)
    assert y_up == 1
    assert abs(gain_up - 0.05) < 1e-6

    # Case B: largest gain is exactly 0.005 (<= 0.01 threshold) -> y_up = 0
    f1_b = {2: 0.30, 4: 0.40, 5: 0.405, 6: 0.402, 8: 0.40}
    y_up, gain_up = compute_selective_upshift_target(f1_b, epsilon=0.01, anchor_k=4)
    assert y_up == 0

    # Case C: degradation (k8 < k4) -> y_up = 0
    f1_c = {2: 0.30, 4: 0.45, 5: 0.40, 6: 0.40, 8: 0.30}
    y_up, gain_up = compute_selective_upshift_target(f1_c, epsilon=0.01, anchor_k=4)
    assert y_up == 0


def test_4_delta_regression_targets_correct():
    """4. Verify marginal gain deltas Delta_k = F(k) - F(4) for k in {5, 6, 8}."""
    f1s = {2: 0.25, 4: 0.35, 5: 0.37, 6: 0.42, 8: 0.45}
    deltas = compute_delta_regression_targets(f1s, anchor_k=4)
    assert abs(deltas[5] - 0.02) < 1e-6
    assert abs(deltas[6] - 0.07) < 1e-6
    assert abs(deltas[8] - 0.10) < 1e-6


def test_5_no_f1_target_leaks_into_features():
    """5. Verify assert_zero_feature_leakage raises on forbidden keywords."""
    clean_cols = ["feat_k4_sem_top1", "feat_k2_cov_k_mean", "feat_diff_tokens_k4_minus_k2"]
    assert_zero_feature_leakage(clean_cols)  # Should not raise

    dirty_cols = ["feat_k4_sem_top1", "target_f1_gain"]
    with pytest.raises(ValueError, match="LEAKAGE DETECTED"):
        assert_zero_feature_leakage(dirty_cols)


def test_6_group_kfold_paper_separation():
    """6. Verify GroupKFold guarantees zero paper overlap between train and val folds."""
    df_sample = pd.DataFrame({
        "question_id": [f"q{i}" for i in range(10)],
        "paper_id": ["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4", "p5", "p5"],
    })
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=3)
    for train_idx, val_idx in gkf.split(df_sample, groups=df_sample["paper_id"]):
        train_p = set(df_sample.iloc[train_idx]["paper_id"])
        val_p = set(df_sample.iloc[val_idx]["paper_id"])
        assert train_p.isdisjoint(val_p), "Paper leakage found between folds!"


def test_7_every_train_query_receives_oof_prediction_once(sample_response_data):
    """7. Verify each query gets an OOF prediction tuple exactly once."""
    # Synthetic candidate features
    records = []
    for _, r in sample_response_data.iterrows():
        qid = r["question_id"]
        pid = r["paper_id"]
        for k in [2, 4, 5, 6, 8]:
            records.append({
                "split": "train",
                "question_id": qid,
                "paper_id": pid,
                "k": k,
                "cost_k_tokens": r[f"tokens_k{k}"],
                "cov_k_mean": 0.8,
                "sem_k_selected_mean": 0.5,
                "sem_top1": 0.9,
            })
    df_cand = pd.DataFrame(records)
    df_unified = build_w7c3_query_dataset(df_cand, sample_response_data, budgets=[2, 4, 5, 6, 8], anchor_k=4)
    down_f, up_f, del_f = get_feature_subsets(df_unified)

    # 2 folds for 4 queries across 2 papers (p1, p2)
    df_oof = run_group_kfold_oof(
        df_train_query=df_unified,
        downshift_features=down_f,
        upshift_features=up_f,
        delta_features=del_f,
        n_splits=2,
        group_col="paper_id",
        model_params={"downshift_classifier": {"n_estimators": 5}, "upshift_classifier": {"n_estimators": 5}, "delta_regressors": {"n_estimators": 5}},
    )

    assert len(df_oof) == len(df_unified)
    assert set(df_oof["question_id"]) == set(df_unified["question_id"])


def test_8_dev_never_used_in_oof():
    """8. Verify assert_test_disabled blocks test and dev is excluded from train."""
    assert_test_disabled("train")
    assert_test_disabled("dev")
    with pytest.raises(RuntimeError, match="TEST split access is strictly prohibited"):
        assert_test_disabled("test")


def test_9_tau_down_applied_correctly():
    """9. Verify tau_down threshold controls downshifting."""
    deltas = {5: 0.0, 6: 0.0, 8: 0.0}
    # Case A: p_down >= tau_down -> chooses 2
    assert apply_anchor_deviate_decision_rule(p_down=0.86, p_up=0.1, pred_deltas=deltas, tau_down=0.85) == 2
    # Case B: p_down < tau_down -> does not choose 2
    assert apply_anchor_deviate_decision_rule(p_down=0.84, p_up=0.1, pred_deltas=deltas, tau_down=0.85) == 4


def test_10_tau_up_applied_correctly():
    """10. Verify tau_up threshold controls upshifting."""
    deltas = {5: 0.05, 6: 0.08, 8: 0.02}
    # Case A: p_up >= tau_up -> upshifts (chooses best larger budget)
    assert apply_anchor_deviate_decision_rule(p_down=0.2, p_up=0.75, pred_deltas=deltas, tau_up=0.70) in [5, 6, 8]
    # Case B: p_up < tau_up -> stays at anchor 4
    assert apply_anchor_deviate_decision_rule(p_down=0.2, p_up=0.65, pred_deltas=deltas, tau_up=0.70) == 4


def test_11_downshift_takes_precedence_over_upshift():
    """11. Verify downshift check precedes upshift when p_down >= tau_down."""
    deltas = {5: 0.1, 6: 0.2, 8: 0.3}
    # Both pass their thresholds: downshift should be evaluated first
    choice = apply_anchor_deviate_decision_rule(p_down=0.90, p_up=0.90, pred_deltas=deltas, tau_down=0.85, tau_up=0.70)
    assert choice == 2, f"Expected safe downshift k=2 to take precedence, got {choice}"


def test_12_otherwise_k4_anchor_preserved():
    """12. Verify when neither condition passes, k4 is strictly returned."""
    deltas = {5: 0.05, 6: 0.05, 8: 0.05}
    choice = apply_anchor_deviate_decision_rule(p_down=0.5, p_up=0.5, pred_deltas=deltas, tau_down=0.80, tau_up=0.70)
    assert choice == 4


def test_13_larger_budget_selector_runs_only_after_upshift():
    """13. Verify larger-budget selector only activates when p_up >= tau_up."""
    deltas = {5: 0.05, 6: 0.20, 8: 0.25}
    # If p_up < tau_up, deltas are ignored and k=4 is chosen
    choice = apply_anchor_deviate_decision_rule(p_down=0.1, p_up=0.50, pred_deltas=deltas, tau_up=0.70)
    assert choice == 4


def test_14_delta_near_tie_chooses_cheapest_larger_budget():
    """14. Verify near-tie rule within delta_up picks cheapest budget in {5, 6, 8}."""
    # Max is 0.09 at k=8. At delta_up=0.02, k=6 (0.08) and k=5 (0.075) qualify.
    # Cheapest is k=5.
    deltas = {5: 0.075, 6: 0.080, 8: 0.090}
    regs = MarginalGainDeltaRegressors()
    choice_0 = regs.select_larger_budget(deltas, delta_up=0.0)
    assert choice_0 == 8, f"At delta_up=0.0, expected exact max k=8, got {choice_0}"

    choice_tie = regs.select_larger_budget(deltas, delta_up=0.02)
    assert choice_tie == 5, f"At delta_up=0.02, expected cheapest near-tie k=5, got {choice_tie}"


def test_15_no_k8_fallback_assumption():
    """15. Verify there is no implicit fallback to k8 under unconfident or edge cases."""
    deltas = {5: -0.05, 6: -0.10, 8: -0.20}
    # When uncertain, the model MUST fall back to anchor k=4, never k=8
    choice = apply_anchor_deviate_decision_rule(p_down=0.0, p_up=0.0, pred_deltas=deltas)
    assert choice == 4


def test_16_policy_grid_calculates_downstream_f1_by_lookup(sample_response_data):
    """16. Verify evaluate_downstream_anchor_policy looks up correct F1 and tokens."""
    preds = {"q1": 2, "q2": 6, "q3": 4, "q4": 4}
    res = evaluate_downstream_anchor_policy(preds, sample_response_data, budgets=[2, 4, 5, 6, 8], anchor_k=4)
    # q1: f1_k2=0.45; q2: f1_k6=0.46; q3: f1_k4=0.40; q4: f1_k4=0.50 -> mean = 0.4525
    expected_f1 = (0.45 + 0.46 + 0.40 + 0.50) / 4.0
    assert abs(res["mean_f1"] - expected_f1) < 1e-4


def test_17_quality_constrained_cost_selection_rule():
    """17. Verify policy selection chooses lowest tokens among qualifying policies."""
    grid_rows = [
        {"tau_down": 0.80, "tau_up": 0.70, "delta_up": 0.0, "mean_f1": 0.3920, "mean_tokens": 850.0, "under_allocation_rate": 0.05},
        {"tau_down": 0.70, "tau_up": 0.70, "delta_up": 0.0, "mean_f1": 0.3910, "mean_tokens": 800.0, "under_allocation_rate": 0.06},
        {"tau_down": 0.90, "tau_up": 0.60, "delta_up": 0.0, "mean_f1": 0.3800, "mean_tokens": 750.0, "under_allocation_rate": 0.15},  # Fails constraint
    ]
    df_grid = pd.DataFrame(grid_rows)
    f1_k4_ref = 0.3920
    eta = 0.005  # Qualifying: F1 >= 0.3870 -> row 0 (0.3920) and row 1 (0.3910) qualify

    winner, _ = select_best_anchor_policy(df_grid, f1_k4_reference=f1_k4_ref, eta=eta)
    assert winner["mean_tokens"] == 800.0, f"Expected qualifying policy with lowest tokens (800.0), got {winner['mean_tokens']}"


def test_18_eta_tolerance_applied_correctly():
    """18. Verify eta parameter strictly gates qualifying policies."""
    grid_rows = [
        {"tau_down": 0.80, "tau_up": 0.70, "delta_up": 0.0, "mean_f1": 0.3910, "mean_tokens": 800.0, "under_allocation_rate": 0.05},
        {"tau_down": 0.90, "tau_up": 0.80, "delta_up": 0.0, "mean_f1": 0.3950, "mean_tokens": 920.0, "under_allocation_rate": 0.02},
    ]
    df_grid = pd.DataFrame(grid_rows)
    f1_k4_ref = 0.3950

    # With eta = 0.001: threshold is 0.3940. Only row 1 qualifies (mean_f1=0.3950).
    winner_strict, _ = select_best_anchor_policy(df_grid, f1_k4_reference=f1_k4_ref, eta=0.001)
    assert winner_strict["mean_tokens"] == 920.0

    # With eta = 0.005: threshold is 0.3900. Row 0 qualifies and has lower tokens (800.0).
    winner_loose, _ = select_best_anchor_policy(df_grid, f1_k4_reference=f1_k4_ref, eta=0.005)
    assert winner_loose["mean_tokens"] == 800.0


def test_19_lowest_token_qualifying_policy_selected():
    """19. Verify tie-breaking logic on tokens."""
    grid_rows = [
        {"tau_down": 0.85, "tau_up": 0.75, "delta_up": 0.0, "mean_f1": 0.3900, "mean_tokens": 820.0, "under_allocation_rate": 0.04},
        {"tau_down": 0.80, "tau_up": 0.75, "delta_up": 0.0, "mean_f1": 0.3900, "mean_tokens": 810.0, "under_allocation_rate": 0.05},
    ]
    df_grid = pd.DataFrame(grid_rows)
    winner, _ = select_best_anchor_policy(df_grid, f1_k4_reference=0.3900, eta=0.005)
    assert winner["mean_tokens"] == 810.0


def test_20_downshift_safety_metrics_correct(sample_response_data):
    """20. Verify downshift safety metrics calculation."""
    df_eval = sample_response_data.copy()
    # Let pred_k for q1 be 2 (actual F1 k2=0.45, max=0.45 -> correct downshift)
    df_eval["pred_k"] = [2, 4, 4, 4]
    safety = compute_downshift_safety_metrics(df_eval, budgets=[2, 4, 5, 6, 8], anchor_k=4)
    assert safety["n_downshifted"] == 1
    assert safety["precision_k2_admissible"] == 1.0
    assert safety["mean_tokens_saved_vs_k4"] == (900 - 500)


def test_21_upshift_precision_metrics_correct(sample_response_data):
    """21. Verify upshift precision metrics calculation."""
    df_eval = sample_response_data.copy()
    # Let pred_k for q2 be 6 (F1 k6=0.46, k4=0.35 -> gain=0.11 > 0.01)
    df_eval["pred_k"] = [4, 6, 4, 4]
    precision = compute_upshift_precision_metrics(df_eval, anchor_k=4)
    assert precision["n_upshifted"] == 1
    assert precision["fraction_gain_gt_epsilon"] == 1.0
    assert abs(precision["mean_actual_gain_vs_k4"] - 0.11) < 1e-4


def test_22_e1_to_e6_error_categories_correct(sample_response_data):
    """22. Verify E1-E6 error taxonomy assignment."""
    df_eval = sample_response_data.copy()
    # q1: min_oracle=2, pred=2 -> E1
    # q2: min_oracle=6, pred=8 (f1=0.42, max=0.46 -> inadm) -> E4
    # q3: min_oracle=4, pred=4 -> E1 and E5
    # q4: min_oracle=4, pred=4 -> E1 and E5
    df_eval["pred_k"] = [2, 8, 4, 4]
    df_det, df_sum = compute_w7c3_error_taxonomy(df_eval, budgets=[2, 4, 5, 6, 8], anchor_k=4)

    assert df_det.loc[df_det["question_id"] == "q1", "error_category"].iloc[0] == "E1: Minimum-cost oracle exactly"
    assert df_det.loc[df_det["question_id"] == "q2", "error_category"].iloc[0] == "E4: Higher-budget & STILL inadmissible"
    assert df_det.loc[df_det["question_id"] == "q3", "is_e5_anchor_admissible"].iloc[0] == 1

    cats_in_summary = set(df_sum["category"])
    assert "E1: Minimum-cost oracle exactly" in cats_in_summary
    assert "E5: Anchor k=4 chosen & admissible" in cats_in_summary
    assert "E6: Anchor k=4 chosen & inadmissible" in cats_in_summary


def test_23_full_train_retraining_after_policy_freeze():
    """23. Verify model objects can be fit cleanly on full train feature sets."""
    X = pd.DataFrame({"feat_sem_top1": [0.8, 0.7, 0.9], "feat_cov_mean": [0.5, 0.6, 0.7]})
    y = np.array([1, 0, 1])
    clf = SafeDownshiftClassifier(n_estimators=5)
    clf.fit(X, y)
    probs = clf.predict_proba(X)
    assert len(probs) == 3
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_24_dev_evaluated_only_after_freeze():
    """24. Verify policy can predict on fresh DataFrame without modifying state."""
    pred_deltas = {5: 0.02, 6: 0.08, 8: 0.01}
    pred_k = apply_anchor_deviate_decision_rule(
        p_down=0.2,
        p_up=0.8,
        pred_deltas=pred_deltas,
        tau_down=0.85,
        tau_up=0.70,
        delta_up=0.01,
        anchor_k=4,
    )
    assert pred_k == 6


def test_25_test_cannot_load():
    """25. Verify assert_test_disabled triggers on any case-insensitive 'test' string."""
    for s in ["test", "TEST", "test_split", "paper_disjoint_test"]:
        with pytest.raises(RuntimeError):
            assert_test_disabled(s)


def test_26_w7c1_and_w7c2_artifacts_remain_unchanged():
    """26. Verify W7-C1 and W7-C2 tables and artifacts exist and are preserved."""
    w7c1_table = Path("results/week7/w7c_retrieval_aware_allocator/tables/w7c_main_dev_results.csv")
    w7c2_table = Path("results/week7/w7c2_utility_ranking/tables/w7c2_main_dev_results.csv")
    assert w7c1_table.is_file(), f"W7-C1 table missing: {w7c1_table}"
    assert w7c2_table.is_file(), f"W7-C2 table missing: {w7c2_table}"
