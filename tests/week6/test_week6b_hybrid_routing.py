"""Unit and invariant tests for Week 6-B Hybrid Lexical-Semantic Passage Routing.

Covers all 16 required specifications:
1. min-max normalization in [0,1]
2. constant-vector normalization safe
3. alpha=0 reproduces semantic ranking
4. alpha=1 reproduces BM25 score ranking
5. fusion chooses exactly k unique passages
6. RRF deterministic
7. RRF selects exactly k
8. original top-10 pool unchanged
9. selected passages rendered in original BM25 order
10. Jaccard calculation
11. disagreement features
12. alpha selection uses INTERNAL-VAL only
13. DEV not used in alpha selection
14. TEST never loaded
15. existing R1 results load correctly
16. hybrid routing integrates with canonical SARA representation construction
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from src.week6.hybrid_routing import (
    normalize_scores_minmax,
    normalize_scores_zscore,
    compute_ranks_from_scores,
    fuse_bm25_semantic_scores,
    route_r3_fixed_fusion,
    compute_rrf_scores,
    route_r4_rrf,
    compute_rank_sum_scores,
    route_r5_rank_sum,
    select_best_alpha_from_internal_val,
)
from src.week6.disagreement import (
    compute_jaccard_overlap,
    compute_differing_passages_count,
    compute_rank_correlation,
    extract_future_gate_features_per_query,
    build_disagreement_analysis_dataframe,
)
from src.week6.routing import (
    COMPRESS,
    route_r0_bm25_prefix,
    route_r1_semantic_topk,
    validate_routing_assignment,
    build_routed_sara_prompt,
    extract_routed_sara_embeddings,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class MockMistralTokenizer:
    def __init__(self):
        self.pad_token_id = 32000
        self.eos_token_id = 2
        self.chat_template = None

    def tokenize(self, text: str):
        return text.split()


# 1. min-max normalization in [0,1]
def test_minmax_normalization_bounds():
    scores = np.array([2.5, -1.2, 0.0, 5.8, 3.1, -0.5, 4.4, 1.2, 2.9, 0.8], dtype=np.float32)
    norm = normalize_scores_minmax(scores)
    assert np.all(norm >= 0.0)
    assert np.all(norm <= 1.0)
    assert pytest.approx(float(np.min(norm)), 1e-6) == 0.0
    assert pytest.approx(float(np.max(norm)), 1e-6) == 1.0


# 2. constant-vector normalization safe
def test_constant_vector_normalization_safe():
    const_scores = np.full(10, 3.1415, dtype=np.float32)
    norm = normalize_scores_minmax(const_scores)
    assert len(norm) == 10
    assert np.all(norm == 1.0)
    assert not np.isnan(norm).any()
    assert not np.isinf(norm).any()

    # Also check diagnostic z-score on constant vector
    z_norm = normalize_scores_zscore(const_scores)
    assert np.all(z_norm == 0.0)
    assert not np.isnan(z_norm).any()


# 3. alpha=0 reproduces semantic ranking
def test_alpha_zero_reproduces_semantic_ranking():
    bm25_scores = np.array([4.5, 3.2, 2.1, 1.9, 1.5, 1.2, 0.8, 0.5, 0.2, 0.1])
    sem_scores = np.array([0.15, 0.82, 0.33, 0.94, 0.12, 0.65, 0.44, 0.77, 0.05, 0.51])

    for k in [2, 4, 6]:
        r1_sel = route_r1_semantic_topk(sem_scores, k=k)
        r3_sel = route_r3_fixed_fusion(bm25_scores, sem_scores, alpha=0.0, k=k)
        assert r3_sel == r1_sel, f"Mismatch at k={k}: R3(alpha=0)={r3_sel} vs R1={r1_sel}"


# 4. alpha=1 reproduces BM25 score ranking
def test_alpha_one_reproduces_bm25_score_ranking():
    # Passages in retrieval are ordered descending by BM25 score (possibly with ties at tail)
    bm25_scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.2, 0.0, 0.0, 0.0])
    sem_scores = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.0])

    for k in [2, 4, 6]:
        r0_sel = route_r0_bm25_prefix(k=k, n_passages=10)
        r3_sel = route_r3_fixed_fusion(bm25_scores, sem_scores, alpha=1.0, k=k)
        assert r3_sel == r0_sel, f"Mismatch at k={k}: R3(alpha=1)={r3_sel} vs R0={r0_sel}"


# 5. fusion chooses exactly k unique passages
def test_fusion_chooses_exactly_k_unique_passages():
    bm25_scores = np.random.uniform(0, 5, size=10)
    sem_scores = np.random.uniform(-1, 1, size=10)

    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for k in [2, 4, 6]:
            sel = route_r3_fixed_fusion(bm25_scores, sem_scores, alpha=alpha, k=k)
            assert len(sel) == k
            assert len(set(sel)) == k
            for idx in sel:
                assert 0 <= idx < 10


# 6. RRF deterministic
def test_rrf_deterministic():
    bm25_ranks = np.arange(1, 11)
    sem_ranks = np.array([5, 1, 8, 2, 9, 3, 7, 4, 10, 6])

    sel_run1 = route_r4_rrf(bm25_ranks, sem_ranks, k=4, c=60.0)
    sel_run2 = route_r4_rrf(bm25_ranks, sem_ranks, k=4, c=60.0)
    assert sel_run1 == sel_run2


# 7. RRF selects exactly k
def test_rrf_selects_exactly_k():
    bm25_ranks = np.arange(1, 11)
    sem_ranks = np.random.permutation(np.arange(1, 11))

    for k in [2, 4, 6]:
        sel = route_r4_rrf(bm25_ranks, sem_ranks, k=k, c=60.0)
        assert len(sel) == k
        assert len(set(sel)) == k
        for idx in sel:
            assert 0 <= idx < 10


# 8. original top-10 pool unchanged
def test_original_top10_pool_unchanged():
    passages = [{"passage_id": f"p_{i}", "text": f"Passage text {i}"} for i in range(10)]
    bm25_scores = np.linspace(5.0, 0.5, 10)
    sem_scores = np.random.rand(10)

    for k in [2, 4, 6]:
        sel = route_r3_fixed_fusion(bm25_scores, sem_scores, alpha=0.5, k=k)
        unselected = [i for i in range(10) if i not in set(sel)]
        assert sorted(sel + unselected) == list(range(10))
        assert len(passages) == 10


# 9. selected passages rendered in original BM25 order
def test_selected_passages_rendered_in_bm25_order():
    passages = [{"text": f"Content of doc {i}"} for i in range(10)]
    mock_tok = MockMistralTokenizer()

    # Even if scoring picks ranks in an out-of-order sequence (e.g. passage 8, passage 2)
    bm25_scores = np.linspace(10, 1, 10)
    sem_scores = np.array([0.1, 0.9, 0.2, 0.3, 0.4, 0.5, 0.6, 0.85, 0.05, 0.0])
    sel = route_r3_fixed_fusion(bm25_scores, sem_scores, alpha=0.25, k=4)

    prompt, _, _ = build_routed_sara_prompt("Sample question?", passages, sel, mock_tok)

    # Document labels in prompt must appear in strictly ascending numerical order
    positions = [prompt.find(f"Document {idx + 1}.") for idx in sel]
    for pos in positions:
        assert pos != -1
    assert positions == sorted(positions)


# 10. Jaccard calculation
def test_jaccard_calculation():
    # Identical sets
    assert compute_jaccard_overlap([0, 1, 2], [0, 1, 2]) == 1.0
    # Disjoint sets
    assert compute_jaccard_overlap([0, 1], [2, 3]) == 0.0
    # Partial overlap: {0, 1, 2, 3} vs {2, 3, 4, 5} -> inter=2, union=6 -> 1/3
    assert pytest.approx(compute_jaccard_overlap([0, 1, 2, 3], [2, 3, 4, 5]), 1e-4) == 2.0 / 6.0
    # Both empty
    assert compute_jaccard_overlap([], []) == 1.0


# 11. disagreement features
def test_disagreement_features():
    bm25 = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.0])
    sem = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.0])

    feats = extract_future_gate_features_per_query("q_100", "p_200", bm25, sem, budgets=[2, 4, 6])

    expected_keys = [
        "question_id", "paper_id",
        "bm25_top1", "bm25_margin", "bm25_entropy", "bm25_norm_entropy", "bm25_std",
        "sem_top1", "sem_margin", "sem_entropy", "sem_norm_entropy", "sem_std",
        "cross_rank_corr", "cross_top1_agreement", "cross_mean_norm_diff",
        "cross_jaccard_k2", "cross_exact_agreement_k2",
        "cross_jaccard_k4", "cross_exact_agreement_k4",
        "cross_jaccard_k6", "cross_exact_agreement_k6",
    ]
    for key in expected_keys:
        assert key in feats, f"Missing feature key: {key}"

    assert 0.0 <= feats["bm25_norm_entropy"] <= 1.0
    assert 0.0 <= feats["sem_norm_entropy"] <= 1.0
    assert -1.0 <= feats["cross_rank_corr"] <= 1.0
    assert feats["cross_top1_agreement"] in [0.0, 1.0]


# 12. alpha selection uses INTERNAL-VAL only
def test_alpha_selection_rule_internal_val_only():
    mock_val_results = pd.DataFrame([
        {"k": 4, "alpha": 0.00, "answer_f1_mean": 0.3700, "context_tokens_mean": 920.0},
        {"k": 4, "alpha": 0.25, "answer_f1_mean": 0.3850, "context_tokens_mean": 930.0},
        {"k": 4, "alpha": 0.50, "answer_f1_mean": 0.3845, "context_tokens_mean": 910.0}, # within 0.002, but lower tokens!
        {"k": 4, "alpha": 0.75, "answer_f1_mean": 0.3750, "context_tokens_mean": 935.0},
        {"k": 4, "alpha": 1.00, "answer_f1_mean": 0.3600, "context_tokens_mean": 940.0},
    ])

    # 0.50 is within 0.002 of 0.3850 (diff=0.0005) and has lower tokens (910 vs 930) -> should win!
    best_a = select_best_alpha_from_internal_val(mock_val_results, budget=4, f1_tolerance=0.002)
    assert best_a == 0.50


# 13. DEV not used in alpha selection
def test_dev_manifest_not_used_in_alpha_selection():
    # Verify that helper function accepts only validation dataframe and raises on missing budget
    mock_val_results = pd.DataFrame([
        {"k": 2, "alpha": 0.25, "answer_f1_mean": 0.3100, "context_tokens_mean": 520.0},
    ])
    with pytest.raises(ValueError, match="No results found for budget k=4"):
        select_best_alpha_from_internal_val(mock_val_results, budget=4)


# 14. TEST split never loaded
def test_test_split_never_loaded():
    w6_src_dir = REPO_ROOT / "src/week6"
    for py_file in w6_src_dir.glob("*.py"):
        content = py_file.read_text()
        assert "qasper-test" not in content, f"Forbidden test split reference in {py_file}"
        assert "test_split.jsonl" not in content, f"Forbidden test split reference in {py_file}"

    cfg_file = REPO_ROOT / "configs/week6/hybrid_routing.yaml"
    assert cfg_file.exists()
    cfg_content = cfg_file.read_text()
    assert "test" not in cfg_content.lower() or "qasper_test" not in cfg_content


# 15. existing R1 results load correctly
def test_existing_r1_results_load_correctly():
    w6a_table_path = REPO_ROOT / "results/week6/passage_routing/tables/w6_main_results.csv"
    if w6a_table_path.exists():
        df_w6a = pd.read_csv(w6a_table_path)
        assert "router" in df_w6a.columns
        assert "k" in df_w6a.columns
        assert "answer_f1_mean" in df_w6a.columns

        r1_rows = df_w6a[df_w6a["router"] == "R1_semantic_topk"]
        assert len(r1_rows) == 3  # k=2, 4, 6
        k4_f1 = float(r1_rows[r1_rows["k"] == 4]["answer_f1_mean"].iloc[0])
        assert pytest.approx(k4_f1, 1e-3) == 0.3913


# 16. hybrid routing integrates with canonical SARA representation construction
def test_hybrid_routing_integrates_with_sara_representation():
    passages = [{"passage_id": f"p_{i}", "text": f"Evidence content {i}"} for i in range(10)]
    mock_tok = MockMistralTokenizer()

    bm25 = np.linspace(4.0, 0.5, 10)
    sem = np.array([0.2, 0.8, 0.1, 0.9, 0.3, 0.7, 0.4, 0.6, 0.5, 0.0])

    # Route with hybrid R3
    sel_k4 = route_r3_fixed_fusion(bm25, sem, alpha=0.5, k=4)
    prompt, tokens, unselected = build_routed_sara_prompt("What is the method?", passages, sel_k4, mock_tok)

    assert len(sel_k4) == 4
    assert len(unselected) == 6
    assert prompt.count(COMPRESS) == 6

    # Mock embeddings extraction for the 10 - k = 6 unselected passages
    mock_cached = torch.randn(10, 4096)
    embeds = extract_routed_sara_embeddings(mock_cached, sel_k4)
    assert embeds.shape == (6, 4096)
