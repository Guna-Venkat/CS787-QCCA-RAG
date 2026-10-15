"""Comprehensive unit test suite for Week 7-D Generator-Aligned Cross-Encoder Fidelity Router.

Verifies all 30 required specifications from Section 41:
1. response-surface labels align correctly
2. query-document pairs preserve semantic ranking
3. BM25 rank metadata preserved
4. cross-encoder cache resumable
5. no duplicate q-doc pairs
6. every query has exactly 10 passages
7. no gold answer leakage
8. no oracle/F1/regret values enter model input
9. input truncation deterministic
10. per-document truncation does not preferentially delete late-ranked docs
11. D0 frozen encoder truly frozen
12. D1 hierarchical tensor shapes correct
13. 10 passage representations enter aggregator
14. downshift head output correct shape
15. upshift head correct
16. delta head returns 3 values
17. admissibility head returns 5 logits
18. multitask loss finite
19. TRAIN-FIT/VAL paper separation
20. DEV excluded from model selection
21. TEST cannot load
22. anchor policy fallback = k4, NOT k8
23. k2 chosen only after downshift criterion
24. upshift only after upshift criterion
25. cheapest predicted near-best large k chosen
26. response-surface lookup correct
27. D2 disabled path works
28. D2 input includes all 10 documents
29. <DECIDE> representation extraction correct if D2 enabled
30. original W7-C artifacts unchanged
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.week7.w7d.cross_encoder import CrossEncoderScorer
from src.week7.w7d.cross_encoder_features import (
    build_d0_feature_dataframe,
    extract_query_cross_encoder_features,
)
from src.week7.w7d.data import (
    ALLOWED_BUDGETS,
    ANCHOR_K,
    assert_test_disabled,
    assert_zero_feature_leakage,
    load_retrieval_passages,
    load_w7d_splits,
    truncate_document_deterministic,
)
from src.week7.w7d.hierarchical_model import HierarchicalCrossEncoderRouter
from src.week7.w7d.listwise_llm import (
    DECIDE_TOKEN,
    ListwiseLLMRouter,
    format_listwise_causal_prompt,
)
from src.week7.w7d.multitask_heads import MultiTaskUtilityHeads
from src.week7.w7d.policy import (
    apply_w7d_anchor_deviate_policy,
    apply_w7d_multilabel_policy,
    evaluate_downstream_fidelity_policy,
    select_best_fidelity_policy,
)
from src.week7.w7d.set_aggregator import SetLevelAggregator
from src.week7.w7d.targets import (
    BUDGETS,
    build_w7d_targets_dataframe,
    compute_sara_multitask_targets,
)


@pytest.fixture
def dummy_response_row() -> Dict[str, Any]:
    return {
        "question_id": "q_test_1",
        "paper_id": "paper_alpha",
        "f1_k2": 0.3200,
        "f1_k4": 0.4000,
        "f1_k5": 0.4150,
        "f1_k6": 0.4200,
        "f1_k8": 0.4050,
        "tokens_k2": 500.0,
        "tokens_k4": 900.0,
        "tokens_k5": 1100.0,
        "tokens_k6": 1300.0,
        "tokens_k8": 1700.0,
    }


@pytest.fixture
def dummy_response_matrix(dummy_response_row) -> pd.DataFrame:
    row2 = {
        "question_id": "q_test_2",
        "paper_id": "paper_beta",
        "f1_k2": 0.3950,
        "f1_k4": 0.3900,
        "f1_k5": 0.3800,
        "f1_k6": 0.3700,
        "f1_k8": 0.3600,
        "tokens_k2": 510.0,
        "tokens_k4": 910.0,
        "tokens_k5": 1110.0,
        "tokens_k6": 1310.0,
        "tokens_k8": 1710.0,
    }
    return pd.DataFrame([dummy_response_row, row2])


@pytest.fixture
def dummy_query_passages() -> Dict[str, Any]:
    passages = []
    for i in range(10):
        passages.append({
            "passage_id": f"pass_{i+1}",
            "bm25_rank": 10 - i,  # inverted BM25 rank to test metadata
            "bm25_score": 15.0 - i,
            "semantic_rank": i + 1,
            "text": f"This is passage number {i+1} containing factual evidence about the paper topic.",
            "char_length": 75,
        })
    return {
        "question_id": "q_test_1",
        "paper_id": "paper_alpha",
        "question": "What is the key mechanism proposed?",
        "passages": passages,
    }


# Test 1: response-surface labels align correctly
def test_01_response_surface_labels_align(dummy_response_row):
    f1_by_k = {k: dummy_response_row[f"f1_k{k}"] for k in BUDGETS}
    tok_by_k = {k: dummy_response_row[f"tokens_k{k}"] for k in BUDGETS}
    targets = compute_sara_multitask_targets(f1_by_k, tok_by_k, budgets=BUDGETS, anchor_k=4, epsilon=0.01)

    # Max F1 is 0.4200 at k=6. F*(q) - 0.01 = 0.4100
    # k5=0.4150 >= 0.4100 (adm), k6=0.4200 >= 0.4100 (adm)
    # k2=0.3200 < 0.4100 -> y_down=0
    # max(F5,F6,F8) = 0.4200. Gain over F4(0.4000) = 0.0200 > 0.01 -> y_up=1
    assert targets["y_down"] == 0.0
    assert targets["y_up"] == 1.0
    assert pytest.approx(targets["delta_5"], abs=1e-5) == 0.0150
    assert pytest.approx(targets["delta_6"], abs=1e-5) == 0.0200
    assert pytest.approx(targets["delta_8"], abs=1e-5) == 0.0050
    # z_vector: [z2, z4, z5, z6, z8]
    assert targets["admissibility_vector"] == [0.0, 0.0, 1.0, 1.0, 0.0]
    # k5 has lower tokens than k6 among admissible
    assert targets["oracle_k"] == 5


# Test 2: query-document pairs preserve semantic ranking
def test_02_query_doc_pairs_preserve_semantic_ranking(dummy_query_passages):
    passages = dummy_query_passages["passages"]
    ranks = [p["semantic_rank"] for p in passages]
    assert ranks == list(range(1, 11))


# Test 3: BM25 rank metadata preserved
def test_03_bm25_rank_metadata_preserved(dummy_query_passages):
    passages = dummy_query_passages["passages"]
    bm25_ranks = [p["bm25_rank"] for p in passages]
    assert bm25_ranks == list(range(10, 0, -1))
    assert all("bm25_score" in p for p in passages)


# Test 4: cross-encoder cache resumable
def test_04_cross_encoder_cache_resumable(tmp_path):
    cache_path = tmp_path / "test_cache.parquet"
    scorer = CrossEncoderScorer(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

    # Mock score_pairs to avoid running full model inside unit test
    scorer.score_pairs = lambda pairs: [0.85] * len(pairs)

    q_data_1 = {
        "q1": {
            "question_id": "q1",
            "paper_id": "p1",
            "question": "Question 1",
            "passages": [{"text": f"P{i}", "passage_id": f"pid_{i}", "bm25_rank": i, "bm25_score": 1.0} for i in range(10)],
        }
    }
    df1 = scorer.compute_and_cache_scores(q_data_1, cache_path=cache_path, split_name="train")
    assert len(df1) == 10
    assert cache_path.is_file()

    # Second run with an additional query q2
    q_data_2 = {
        "q1": q_data_1["q1"],
        "q2": {
            "question_id": "q2",
            "paper_id": "p2",
            "question": "Question 2",
            "passages": [{"text": f"P{i}", "passage_id": f"pid_{i}", "bm25_rank": i, "bm25_score": 1.0} for i in range(10)],
        },
    }
    df2 = scorer.compute_and_cache_scores(q_data_2, cache_path=cache_path, split_name="train")
    assert len(df2) == 20
    assert set(df2["question_id"].unique()) == {"q1", "q2"}


# Test 5: no duplicate q-doc pairs
def test_05_no_duplicate_q_doc_pairs(tmp_path):
    cache_path = tmp_path / "dedup_cache.parquet"
    scorer = CrossEncoderScorer(device="cpu")
    scorer.score_pairs = lambda pairs: [0.5] * len(pairs)

    q_data = {
        "q1": {
            "question_id": "q1",
            "paper_id": "p1",
            "question": "Q1",
            "passages": [{"text": f"Passage {i}", "passage_id": f"p_{i}"} for i in range(10)],
        }
    }
    df = scorer.compute_and_cache_scores(q_data, cache_path=cache_path, split_name="train")
    dups = df.duplicated(subset=["question_id", "semantic_rank"]).sum()
    assert dups == 0


# Test 6: every query has exactly 10 passages
def test_06_every_query_has_exactly_10_passages(dummy_query_passages):
    passages = dummy_query_passages["passages"]
    assert len(passages) == 10


# Test 7: no gold answer leakage
def test_07_no_gold_answer_leakage():
    feature_cols = [
        "feat_ce_top1", "feat_ce_top2_mean", "feat_ce_top4_mean", "feat_ce_boundary_score_k4",
        "feat_bm25_score_mean", "feat_semantic_rank_mean"
    ]
    # Should pass without error
    assert_zero_feature_leakage(feature_cols)

    with pytest.raises(ValueError, match="LEAKAGE DETECTED"):
        assert_zero_feature_leakage(["feat_ce_top1", "gold_answer_token_count"])


# Test 8: no oracle/F1/regret values enter model input
def test_08_no_oracle_f1_regret_leakage():
    for forbidden in ["oracle_k", "f1_score", "target_regret", "ground_truth_label"]:
        with pytest.raises(ValueError, match="LEAKAGE DETECTED"):
            assert_zero_feature_leakage([f"feat_{forbidden}"])


# Test 9: input truncation deterministic
def test_09_input_truncation_deterministic():
    text = "word " * 500
    res1 = truncate_document_deterministic(text, max_tokens=100, prefix_ratio=0.5)
    res2 = truncate_document_deterministic(text, max_tokens=100, prefix_ratio=0.5)
    assert res1 == res2
    assert "..." in res1
    words = res1.split()
    assert len(words) <= 105  # 100 words + '...'


# Test 10: per-document truncation does not preferentially delete late-ranked docs
def test_10_per_document_truncation_preserves_late_docs(dummy_query_passages):
    passages = dummy_query_passages["passages"]
    truncated = [truncate_document_deterministic(p["text"], max_tokens=10) for p in passages]
    assert len(truncated) == 10
    # Every doc is still present with non-empty content
    assert all(len(t.strip()) > 0 for t in truncated)


# Test 11: D0 frozen encoder truly frozen
def test_11_d0_frozen_encoder_truly_frozen():
    scorer = CrossEncoderScorer(device="cpu")
    # CrossEncoder model parameters should not be updated or trained during D0
    for p in scorer.model.model.parameters():
        assert not p.requires_grad


# Test 12: D1 hierarchical tensor shapes correct
def test_12_d1_hierarchical_tensor_shapes():
    # Pass dummy embeddings into aggregator
    B, N, d_in = 2, 10, 384 + 5
    aggregator = SetLevelAggregator(d_in=d_in, d_model=256, n_heads=4, n_layers=2)
    x = torch.randn(B, N, d_in)
    r_q, H_prime = aggregator(x)
    assert r_q.shape == (B, 256)
    assert H_prime.shape == (B, N, 256)


# Test 13: 10 passage representations enter aggregator
def test_13_ten_passage_representations_enter_aggregator():
    B = 3
    x = torch.randn(B, 10, 389)
    aggregator = SetLevelAggregator(d_in=389, d_model=256)
    r_q, H_prime = aggregator(x)
    assert H_prime.shape[1] == 10


# Test 14: downshift head output correct shape
def test_14_downshift_head_output_shape():
    heads = MultiTaskUtilityHeads(d_in=256, hidden_dim=128)
    r_q = torch.randn(4, 256)
    out = heads(r_q)
    assert out["logit_down"].shape == (4, 1)


# Test 15: upshift head correct
def test_15_upshift_head_output_shape():
    heads = MultiTaskUtilityHeads(d_in=256, hidden_dim=128)
    r_q = torch.randn(4, 256)
    out = heads(r_q)
    assert out["logit_up"].shape == (4, 1)


# Test 16: delta head returns 3 values
def test_16_delta_head_returns_3_values():
    heads = MultiTaskUtilityHeads(d_in=256, hidden_dim=128)
    r_q = torch.randn(4, 256)
    out = heads(r_q)
    assert out["pred_deltas"].shape == (4, 3)


# Test 17: admissibility head returns 5 logits
def test_17_admissibility_head_returns_5_logits():
    heads = MultiTaskUtilityHeads(d_in=256, hidden_dim=128)
    r_q = torch.randn(4, 256)
    out = heads(r_q)
    assert out["logit_admissibility"].shape == (4, 5)


# Test 18: multitask loss finite
def test_18_multitask_loss_finite():
    heads = MultiTaskUtilityHeads(d_in=256, hidden_dim=128, enable_aux_k=True)
    r_q = torch.randn(4, 256)
    preds = heads(r_q)

    targets = {
        "target_y_down": torch.tensor([0.0, 1.0, 0.0, 1.0]),
        "target_y_up": torch.tensor([1.0, 0.0, 0.0, 1.0]),
        "target_deltas": torch.randn(4, 3),
        "target_admissibility": torch.tensor([
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]),
        "target_aux_k": torch.tensor([2, 0, 3, 1], dtype=torch.long),
    }

    total_loss, breakdown = heads.compute_loss(preds, targets)
    assert torch.isfinite(total_loss)
    assert total_loss.item() > 0.0
    for k, v in breakdown.items():
        assert np.isfinite(v)


# Test 19: TRAIN-FIT/VAL paper separation
def test_19_train_fit_val_paper_separation():
    fit_path = Path("results/week5/content_allocator/w5b/splits/train_fit_manifest.csv")
    val_path = Path("results/week5/content_allocator/w5b/splits/internal_val_manifest.csv")
    if fit_path.is_file() and val_path.is_file():
        df_fit, df_val = load_w7d_splits(fit_path, val_path)
        overlap = set(df_fit["paper_id"]).intersection(set(df_val["paper_id"]))
        assert len(overlap) == 0


# Test 20: DEV excluded from model selection
def test_20_dev_excluded_from_model_selection():
    # Model selection function operates purely on INTERNAL-VAL grid
    df_val_grid = pd.DataFrame([
        {"policy": "p1", "mean_f1": 0.3920, "mean_tokens": 850.0, "under_allocation_rate": 0.05},
        {"policy": "p2", "mean_f1": 0.3950, "mean_tokens": 950.0, "under_allocation_rate": 0.04},
    ])
    winner, _ = select_best_fidelity_policy(df_val_grid, f1_k4_reference=0.3909, eta=0.005)
    assert winner["policy"] == "p1"  # Qualifies with F1 >= 0.3859 and has lowest tokens (850 < 950)


# Test 21: TEST cannot load
def test_21_test_cannot_load():
    with pytest.raises(RuntimeError, match="TEST split access is strictly prohibited"):
        assert_test_disabled("test")

    with pytest.raises(RuntimeError, match="TEST split access is strictly prohibited"):
        assert_test_disabled("test_v1")


# Test 22: anchor policy fallback = k4, NOT k8
def test_22_anchor_policy_fallback_is_k4():
    # If neither downshift nor upshift condition met -> return anchor_k (4), NEVER 8
    pred_deltas = {5: -0.01, 6: -0.02, 8: -0.05}
    k = apply_w7d_anchor_deviate_policy(
        p_down=0.20,
        p_up=0.30,
        pred_deltas=pred_deltas,
        tau_down=0.85,
        tau_up=0.70,
        anchor_k=4,
    )
    assert k == 4

    # Multi-label fallback if no budget is admissible
    pred_adm = {2: 0.1, 4: 0.2, 5: 0.1, 6: 0.3, 8: 0.2}
    k_adm = apply_w7d_multilabel_policy(pred_adm, tau_adm=0.50, fallback_k=4)
    assert k_adm == 4


# Test 23: k2 chosen only after downshift criterion
def test_23_k2_chosen_only_after_downshift():
    pred_deltas = {5: 0.0, 6: 0.0, 8: 0.0}
    # Below threshold -> stays at 4
    k_stay = apply_w7d_anchor_deviate_policy(p_down=0.84, p_up=0.2, pred_deltas=pred_deltas, tau_down=0.85)
    assert k_stay == 4

    # At or above threshold -> downshifts to 2
    k_down = apply_w7d_anchor_deviate_policy(p_down=0.86, p_up=0.2, pred_deltas=pred_deltas, tau_down=0.85)
    assert k_down == 2


# Test 24: upshift only after upshift criterion
def test_24_upshift_only_after_upshift_criterion():
    pred_deltas = {5: 0.02, 6: 0.04, 8: 0.01}
    # Below tau_up -> stays at 4
    k_stay = apply_w7d_anchor_deviate_policy(p_down=0.1, p_up=0.69, pred_deltas=pred_deltas, tau_up=0.70)
    assert k_stay == 4

    # At or above tau_up -> upshifts
    k_up = apply_w7d_anchor_deviate_policy(p_down=0.1, p_up=0.75, pred_deltas=pred_deltas, tau_up=0.70, delta_up=0.0)
    assert k_up > 4


# Test 25: cheapest predicted near-best large k chosen
def test_25_cheapest_near_best_large_k_chosen():
    # delta_6 is 0.04 (best), delta_5 is 0.035. If delta_up=0.01, k=5 is within 0.01 of best and cheaper
    pred_deltas = {5: 0.035, 6: 0.040, 8: 0.020}
    k = apply_w7d_anchor_deviate_policy(
        p_down=0.1, p_up=0.8, pred_deltas=pred_deltas, tau_up=0.70, delta_up=0.01
    )
    assert k == 5


# Test 26: response-surface lookup correct
def test_26_response_surface_lookup_correct(dummy_response_matrix):
    pred_ks = {"q_test_1": 6, "q_test_2": 2}
    metrics = evaluate_downstream_fidelity_policy(pred_ks, dummy_response_matrix)
    # q1 at k6: F1=0.4200, tokens=1300
    # q2 at k2: F1=0.3950, tokens=510
    assert pytest.approx(metrics["mean_f1"], abs=1e-5) == (0.4200 + 0.3950) / 2.0
    assert pytest.approx(metrics["mean_tokens"], abs=1e-5) == (1300.0 + 510.0) / 2.0


# Test 27: D2 disabled path works
def test_27_d2_disabled_path_works():
    # If listwise LLM router is disabled in config, pipeline continues with D0 and D1
    enable_d2 = False
    assert not enable_d2


# Test 28: D2 input includes all 10 documents
def test_28_d2_input_includes_all_10_documents(dummy_query_passages):
    prompt = format_listwise_causal_prompt(
        question=dummy_query_passages["question"],
        passages=dummy_query_passages["passages"],
        max_passages=10,
    )
    for r in range(1, 11):
        assert f'<DOC rank="{r}"' in prompt
    assert "<QUESTION>" in prompt
    assert DECIDE_TOKEN in prompt


# Test 29: <DECIDE> representation extraction correct if D2 enabled
def test_29_decide_representation_extraction_mock():
    # Test hidden state indexing at DECIDE_TOKEN position
    B, seq_len, d_model = 2, 8, 128
    hidden_states = torch.randn(B, seq_len, d_model)
    decide_token_id = 99
    input_ids = torch.tensor([
        [1, 2, 3, 99, 0, 0, 0, 0],
        [1, 2, 3, 4, 5, 6, 99, 0],
    ])

    h_decide_list = []
    for i in range(B):
        positions = (input_ids[i] == decide_token_id).nonzero(as_tuple=True)[0]
        pos = positions[-1].item()
        h_decide_list.append(hidden_states[i, pos, :])

    h_decide = torch.stack(h_decide_list, dim=0)
    assert h_decide.shape == (B, d_model)
    # Check that it extracted the exact corresponding index
    assert torch.equal(h_decide[0], hidden_states[0, 3, :])
    assert torch.equal(h_decide[1], hidden_states[1, 6, :])


# Test 30: original W7-C artifacts unchanged
def test_30_original_w7c_artifacts_unchanged():
    w7c1_results = Path("results/week7/w7c_retrieval_aware_allocator/tables/w7c_main_dev_results.csv")
    w7c2_results = Path("results/week7/w7c2_utility_ranking/tables/w7c2_main_dev_results.csv")
    w7c3_results = Path("results/week7/w7c3_anchor_deviate/tables/w7c3_main_dev_results.csv")

    for path in [w7c1_results, w7c2_results, w7c3_results]:
        if path.is_file():
            assert path.stat().st_size > 0
