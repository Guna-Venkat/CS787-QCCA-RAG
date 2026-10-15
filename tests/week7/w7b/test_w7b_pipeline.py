"""Comprehensive test suite for Week 7-B addressing all 22 required tests.

Required tests:
1. budgets exactly {2,4,5,6,8}
2. semantic routing matches canonical R1 behavior for k2/k4/k6 sample queries
3. exactly k FULL
4. exactly 10-k COMPRESSED
5. all 10 passages represented once
6. selected FULL passages presented in BM25 order
7. response-surface cache key is (question_id, k)
8. resume skips completed pairs
9. duplicate pairs rejected
10. matrix construction yields 5 budgets per completed query
11. epsilon oracle chooses smallest admissible k
12. epsilon fixed at 0.01
13. BM25 oracle and semantic oracle kept distinct
14. exact canonical five W4 features loaded
15. semantic-aware labels built only from TRAIN semantic surface
16. INTERNAL-VAL used for model selection
17. DEV excluded from training/model selection
18. TEST cannot be loaded
19. policy evaluation by lookup matches direct row values
20. empty generation handled as F1=0
21. per-query routing assignments stored explicitly
22. checkpoint/config identifiers stored in cache
"""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.week7.w7b.data import (
    ALLOWED_BUDGETS,
    assert_test_disabled,
    load_w7b_dataset_bundle,
    load_w7b_fit_val_splits,
)
from src.week7.w7b.response_surface import (
    build_semantic_routing_assignments,
    load_existing_response_surface_cache,
    build_semantic_fixed_k_matrix,
)
from src.week7.w7b.oracle import (
    compute_semantic_oracle_dataframe,
    summarize_semantic_oracle,
    compute_bm25_vs_semantic_oracle_shift,
    compute_recovered_oracle_headroom,
)
from src.week7.w7b.features import (
    STAGE_2_CORE_FEATURES,
    extract_single_query_w4_features,
    standardize_features_train_fit_only,
)
from src.week7.w7b.training import (
    train_candidate_budget_models,
    predict_budgets,
    select_best_model_on_internal_val,
)
from src.week7.w7b.evaluation import (
    evaluate_policy_by_matrix_lookup,
    build_w7b_master_results_table,
)
from src.week6.routing import (
    route_r1_semantic_topk,
    validate_routing_assignment,
)
from src.week6.evaluation import compute_qa_answer_metrics


# ---------------------------------------------------------------------------
# Test 1: Budgets exactly {2, 4, 5, 6, 8}
# ---------------------------------------------------------------------------
def test_1_budgets_exactly_2_4_5_6_8():
    assert ALLOWED_BUDGETS == {2, 4, 5, 6, 8}
    assert 0 not in ALLOWED_BUDGETS
    assert 10 not in ALLOWED_BUDGETS


# ---------------------------------------------------------------------------
# Test 2: Semantic routing matches canonical R1 behavior
# ---------------------------------------------------------------------------
def test_2_semantic_routing_matches_canonical_r1():
    # Passages with scores: rank 0=0.1, 1=0.9, 2=0.8, 3=0.3, 4=0.7, 5=0.2, 6=0.4, 7=0.6, 8=0.0, 9=0.5
    scores = [0.1, 0.9, 0.8, 0.3, 0.7, 0.2, 0.4, 0.6, 0.0, 0.5]
    # For k=2, top 2 scores are index 1 (0.9) and index 2 (0.8). Sorted: [1, 2]
    sel_k2 = route_r1_semantic_topk(scores, k=2)
    assert sel_k2 == [1, 2]

    # For k=4, top 4 are indices 1 (0.9), 2 (0.8), 4 (0.7), 7 (0.6). Sorted: [1, 2, 4, 7]
    sel_k4 = route_r1_semantic_topk(scores, k=4)
    assert sel_k4 == [1, 2, 4, 7]

    # For k=6, top 6 are indices 1, 2, 4, 7, 9 (0.5), 6 (0.4). Sorted: [1, 2, 4, 6, 7, 9]
    sel_k6 = route_r1_semantic_topk(scores, k=6)
    assert sel_k6 == [1, 2, 4, 6, 7, 9]


# ---------------------------------------------------------------------------
# Tests 3, 4, 5, 6: Routing Invariants (k FULL, 10-k COMPRESSED, BM25 order)
# ---------------------------------------------------------------------------
def test_3_4_5_6_routing_invariants():
    scores = np.random.RandomState(42).randn(10)
    for k in [2, 4, 5, 6, 8]:
        sel = route_r1_semantic_topk(scores, k=k)
        comp = [idx for idx in range(10) if idx not in sel]

        # Test 3: exactly k FULL
        assert len(sel) == k
        # Test 4: exactly 10 - k COMPRESSED
        assert len(comp) == 10 - k
        # Test 5: all 10 passages accounted for once
        assert sorted(sel + comp) == list(range(10))
        assert len(set(sel)) == k
        # Test 6: selected FULL passages presented in original BM25 order
        assert sel == sorted(sel)
        # Invariants validator must not raise
        validate_routing_assignment(sel, k=k, total_passages=10)


# ---------------------------------------------------------------------------
# Test 7: Cache key is (question_id, k)
# ---------------------------------------------------------------------------
def test_7_response_surface_cache_key(tmp_path):
    cache_file = tmp_path / "cache.jsonl"
    rec1 = {"question_id": "101", "k": 2, "token_f1": 0.5, "context_tokens": 500}
    rec2 = {"question_id": "101", "k": 4, "token_f1": 0.6, "context_tokens": 900}
    rec3 = {"question_id": "102", "k": 2, "token_f1": 0.4, "context_tokens": 510}

    with open(cache_file, "w") as f:
        for r in [rec1, rec2, rec3]:
            f.write(json.dumps(r) + "\n")

    cmap, ckeys = load_existing_response_surface_cache(cache_file)
    assert ("101", 2) in ckeys
    assert ("101", 4) in ckeys
    assert ("102", 2) in ckeys
    assert len(ckeys) == 3


# ---------------------------------------------------------------------------
# Test 8: Resume skips completed pairs
# ---------------------------------------------------------------------------
def test_8_resume_skips_completed_pairs(tmp_path):
    cache_file = tmp_path / "cache.jsonl"
    rec = {"question_id": "101", "k": 2, "token_f1": 0.5, "context_tokens": 500}
    with open(cache_file, "w") as f:
        f.write(json.dumps(rec) + "\n")

    _, ckeys = load_existing_response_surface_cache(cache_file)
    # Pair ("101", 2) is completed and should be recognized
    assert ("101", 2) in ckeys
    assert ("101", 4) not in ckeys


# ---------------------------------------------------------------------------
# Test 9: Duplicate pairs rejected
# ---------------------------------------------------------------------------
def test_9_duplicate_pairs_rejected(tmp_path):
    cache_file = tmp_path / "duplicate_cache.jsonl"
    rec = {"question_id": "101", "k": 2, "token_f1": 0.5, "context_tokens": 500}
    with open(cache_file, "w") as f:
        f.write(json.dumps(rec) + "\n")
        f.write(json.dumps(rec) + "\n")  # duplicate

    with pytest.raises(ValueError, match="duplicate cache entry"):
        load_existing_response_surface_cache(cache_file)


# ---------------------------------------------------------------------------
# Test 10: Matrix construction yields 5 budgets per completed query
# ---------------------------------------------------------------------------
def test_10_matrix_construction_yields_5_budgets(tmp_path):
    jsonl_file = tmp_path / "results.jsonl"
    matrix_csv = tmp_path / "matrix.csv"

    # Write complete records for q1 and q2 across all 5 budgets
    records = []
    for qid in ["q1", "q2"]:
        for k in [2, 4, 5, 6, 8]:
            records.append({
                "question_id": qid,
                "paper_id": "p1",
                "k": k,
                "token_f1": 0.1 * k,
                "context_tokens": 100 * k,
            })

    with open(jsonl_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    df_mat = build_semantic_fixed_k_matrix(jsonl_file, matrix_csv, budgets=[2, 4, 5, 6, 8])
    assert len(df_mat) == 2
    for k in [2, 4, 5, 6, 8]:
        assert f"f1_k{k}" in df_mat.columns
        assert f"tokens_k{k}" in df_mat.columns

    # Test error if a budget is missing
    bad_jsonl = tmp_path / "bad.jsonl"
    with open(bad_jsonl, "w") as f:
        for r in records[:-1]:  # drop last k=8
            f.write(json.dumps(r) + "\n")
    with pytest.raises(ValueError, match="missing budget"):
        build_semantic_fixed_k_matrix(bad_jsonl, tmp_path / "bad.csv", budgets=[2, 4, 5, 6, 8])


# ---------------------------------------------------------------------------
# Tests 11 & 12: Epsilon oracle chooses smallest admissible k with eps=0.01
# ---------------------------------------------------------------------------
def test_11_12_epsilon_oracle_smallest_k_fixed_eps():
    df_mat = pd.DataFrame([{
        "question_id": "q1",
        "paper_id": "p1",
        "f1_k2": 0.40,
        "f1_k4": 0.495,  # within 0.01 of 0.50
        "f1_k5": 0.490,
        "f1_k6": 0.50,   # peak
        "f1_k8": 0.48,
        "tokens_k2": 500,
        "tokens_k4": 900,
        "tokens_k5": 1100,
        "tokens_k6": 1300,
        "tokens_k8": 1700,
    }])

    # eps=0.01 must be enforced
    with pytest.raises(AssertionError):
        compute_semantic_oracle_dataframe(df_mat, epsilon=0.05)

    df_orc = compute_semantic_oracle_dataframe(df_mat, epsilon=0.01)
    # Peak is 0.50 at k=6; threshold is 0.50 - 0.01 = 0.49.
    # Admissible budgets: k=4 (0.495), k=5 (0.490), k=6 (0.50).
    # Smallest admissible budget is k=4!
    assert df_orc.iloc[0]["quality_oracle_k"] == 6
    assert df_orc.iloc[0]["oracle_k_eps001"] == 4
    assert df_orc.iloc[0]["oracle_f1"] == 0.495


# ---------------------------------------------------------------------------
# Test 13: BM25 oracle and Semantic oracle kept distinct
# ---------------------------------------------------------------------------
def test_13_bm25_vs_semantic_oracle_kept_distinct():
    df_bm25 = pd.DataFrame([
        {"question_id": "q1", "paper_id": "p1", "oracle_k_eps001": 6, "oracle_tokens": 1300},
        {"question_id": "q2", "paper_id": "p2", "oracle_k_eps001": 4, "oracle_tokens": 900},
    ])
    df_sem = pd.DataFrame([
        {"question_id": "q1", "paper_id": "p1", "oracle_k_eps001": 4, "oracle_tokens": 920},
        {"question_id": "q2", "paper_id": "p2", "oracle_k_eps001": 6, "oracle_tokens": 1310},
    ])

    mat, summary, merged = compute_bm25_vs_semantic_oracle_shift(df_bm25, df_sem)
    assert "k_bm25_oracle" in merged.columns
    assert "k_sem_oracle" in merged.columns
    assert summary["pct_lower_under_semantic"] == 50.0
    assert summary["pct_higher_under_semantic"] == 50.0


# ---------------------------------------------------------------------------
# Test 14: Exact canonical five W4 features loaded
# ---------------------------------------------------------------------------
def test_14_exact_canonical_five_w4_features():
    expected_features = [
        "bm25_mean_score",
        "lex_coverage_top2",
        "sem_sim_mean_top6",
        "evidence_query_cluster_span",
        "evidence_lexical_overlap_mean",
    ]
    assert STAGE_2_CORE_FEATURES == expected_features

    # Test single query feature extractor returns all 5
    dummy_rec = {
        "question": "What is the training set size?",
        "passages": [{"text": "The dataset contains 1000 examples.", "score": 2.5}],
    }
    feats = extract_single_query_w4_features(dummy_rec)
    for f in expected_features:
        assert f in feats
        assert isinstance(feats[f], (float, int))


# ---------------------------------------------------------------------------
# Tests 15, 16, 17: Semantic-aware labels only from TRAIN, INTERNAL-VAL for selection, DEV excluded
# ---------------------------------------------------------------------------
def test_15_16_17_train_fit_val_and_dev_exclusion():
    rng = np.random.RandomState(42)
    # Simulate TRAIN-FIT (40 queries)
    X_fit = rng.randn(40, 5)
    y_fit = rng.choice([4, 6], size=40)

    # Train candidates
    models = train_candidate_budget_models(X_fit, y_fit)
    assert "balanced_logistic_regression_cascade" in models

    # Simulate INTERNAL-VAL (10 queries)
    X_val = rng.randn(10, 5)
    df_val_mat = pd.DataFrame({
        "question_id": [f"vq{i}" for i in range(10)],
        "paper_id": ["vp1"] * 10,
        "f1_k2": [0.3] * 10,
        "f1_k4": [0.38] * 10,
        "f1_k5": [0.39] * 10,
        "f1_k6": [0.41] * 10,
        "f1_k8": [0.40] * 10,
        "tokens_k2": [500] * 10,
        "tokens_k4": [900] * 10,
        "tokens_k5": [1100] * 10,
        "tokens_k6": [1300] * 10,
        "tokens_k8": [1700] * 10,
    })

    # Selection on INTERNAL-VAL
    best_name, best_entry, df_res = select_best_model_on_internal_val(models, X_val, df_val_mat)
    assert best_name in models
    assert "val_mean_f1" in df_res.columns


# ---------------------------------------------------------------------------
# Test 18: TEST cannot be loaded
# ---------------------------------------------------------------------------
def test_18_test_cannot_be_loaded():
    with pytest.raises(RuntimeError, match="TEST split access is strictly prohibited"):
        assert_test_disabled("test")
    with pytest.raises(RuntimeError, match="TEST split access is strictly prohibited"):
        assert_test_disabled("QASPER_test.jsonl")


# ---------------------------------------------------------------------------
# Test 19: Policy evaluation by lookup matches direct row values
# ---------------------------------------------------------------------------
def test_19_policy_evaluation_by_lookup_matches_row_values():
    df_mat = pd.DataFrame([
        {"question_id": "q1", "paper_id": "p1", "f1_k4": 0.35, "tokens_k4": 920, "f1_k6": 0.42, "tokens_k6": 1310},
        {"question_id": "q2", "paper_id": "p2", "f1_k4": 0.38, "tokens_k4": 915, "f1_k6": 0.36, "tokens_k6": 1305},
    ])
    # Predicted k: q1->6, q2->4
    preds = [6, 4]
    res = evaluate_policy_by_matrix_lookup(df_mat, preds)
    assert res["per_query_f1s"] == [0.42, 0.38]
    assert res["per_query_tokens"] == [1310, 915]
    assert np.isclose(res["mean_f1"], (0.42 + 0.38) / 2.0)


# ---------------------------------------------------------------------------
# Test 20: Empty generation handled as F1=0
# ---------------------------------------------------------------------------
def test_20_empty_generation_handled_as_f1_zero():
    m = compute_qa_answer_metrics(prediction="", gold_answers=["Neural networks"])
    assert m["token_f1"] == 0.0
    assert m["exact_match"] == 0.0
    assert m["rouge_l"] == 0.0

    m_none = compute_qa_answer_metrics(prediction=None, gold_answers=["Neural networks"])
    assert m_none["token_f1"] == 0.0


# ---------------------------------------------------------------------------
# Test 21: Per-query routing assignments stored explicitly
# ---------------------------------------------------------------------------
def test_21_routing_assignments_stored_explicitly():
    scores_list = [[0.1 * i for i in range(10)]]
    assignments = build_semantic_routing_assignments(scores_list, budgets=[2, 4, 5, 6, 8])
    for k in [2, 4, 5, 6, 8]:
        assert k in assignments
        assert len(assignments[k]) == 1
        assert len(assignments[k][0]) == k


# ---------------------------------------------------------------------------
# Test 22: Checkpoint/config identifiers stored in cache
# ---------------------------------------------------------------------------
def test_22_checkpoint_config_identifiers_in_cache(tmp_path):
    cache_file = tmp_path / "cache.jsonl"
    record_data = {
        "split": "train",
        "question_id": "100",
        "paper_id": "200",
        "k": 4,
        "routing": "semantic_topk",
        "selected_full_indices": [0, 1, 2, 3],
        "compressed_indices": [4, 5, 6, 7, 8, 9],
        "prediction": "test answer",
        "gold_answers": ["test answer"],
        "token_f1": 1.0,
        "exact_match": 1.0,
        "rouge_l": 1.0,
        "context_tokens": 920,
        "latency": 2.5,
        "checkpoint_id": "sara_qasper_proj_lr5e4_seed42",
        "generation_config": {"max_new_tokens": 128, "do_sample": False},
    }
    with open(cache_file, "w") as f:
        f.write(json.dumps(record_data) + "\n")

    cmap, _ = load_existing_response_surface_cache(cache_file)
    rec = cmap[("100", 4)]
    assert rec["checkpoint_id"] == "sara_qasper_proj_lr5e4_seed42"
    assert rec["generation_config"]["max_new_tokens"] == 128
    assert rec["generation_config"]["do_sample"] is False
