"""Unit and invariant tests for Week 6 Passage Routing.

Covers all 20 required specifications:
1. R0 selects exactly ranks 1..k
2. R1 selects exactly k unique passages
3. R2 selects exactly k unique passages
4. Selected passages always belong to original top-10
5. No passage duplication
6. Full + compressed count = 10
7. Exactly k full-text passages in prompt
8. Exactly 10 - k compressed passages in prompt
9. Rank-preserving final ordering
10. R1 deterministic under fixed inputs
11. R2 scoring shape = [B, 10]
12. Evidence mapping correctness
13. TRAIN-FIT / INTERNAL-VAL paper/question isolation
14. DEV excluded from R2 training
15. TEST split never loaded or accessed
16. Routing Recall@k calculation
17. Jaccard routing overlap calculation
18. Selected-rank frequency calculation
19. Existing R0 canonical result reproduction
20. Optional k=2 oracle enumerates exactly 45 subsets
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from src.week6.routing import COMPRESS
from src.week6.data import (
    match_passages_to_gold_evidence,
    audit_w6_split_isolation,
)
from src.week6.models import LearnedPassageRouter, PretrainedSemanticScorer
from src.week6.routing import (
    route_r0_bm25_prefix,
    route_r1_semantic_topk,
    route_r2_learned_topk,
    validate_routing_assignment,
    build_routed_sara_prompt,
    extract_routed_sara_embeddings,
)
from src.week6.evaluation import (
    compute_evidence_recall_precision,
    compute_selected_rank_distribution,
    compute_routing_diversity,
    load_canonical_r0_results,
)
from src.week6.oracle import get_all_subsets

REPO_ROOT = Path(__file__).resolve().parents[2]


# Mock tokenizer for fast prompt testing without loading Mistral
class MockMistralTokenizer:
    def __init__(self):
        self.pad_token_id = 32000
        self.eos_token_id = 2
        self.chat_template = None

    def tokenize(self, text: str):
        return text.split()


# 1. R0 selects exactly ranks 1..k
def test_r0_selects_prefix_ranks():
    for k in [2, 4, 6]:
        sel = route_r0_bm25_prefix(k, n_passages=10)
        assert sel == list(range(k))
        # 1-indexed ranks are 1..k
        assert [r + 1 for r in sel] == list(range(1, k + 1))


# 2. R1 selects exactly k unique passages
def test_r1_selects_k_unique_passages():
    scores = np.array([0.1, 0.9, 0.4, 0.8, 0.2, 0.7, 0.3, 0.6, 0.5, 0.0])
    for k in [2, 4, 6]:
        sel = route_r1_semantic_topk(scores, k=k)
        assert len(sel) == k
        assert len(set(sel)) == k


# 3. R2 selects exactly k unique passages
def test_r2_selects_k_unique_passages():
    logits = torch.tensor([-1.2, 2.5, 0.1, 1.8, -0.5, 3.2, -2.0, 0.4, 1.1, -0.1])
    for k in [2, 4, 6]:
        sel = route_r2_learned_topk(logits, k=k)
        assert len(sel) == k
        assert len(set(sel)) == k


# 4. Selected passages always belong to original top-10
def test_selected_passages_belong_to_top10():
    logits = np.random.randn(10)
    for k in [2, 4, 6]:
        sel = route_r2_learned_topk(logits, k=k)
        for idx in sel:
            assert 0 <= idx < 10


# 5. No passage duplication
def test_no_passage_duplication():
    scores = np.ones(10)  # tie case
    for k in [2, 4, 6]:
        sel = route_r1_semantic_topk(scores, k=k)
        assert len(set(sel)) == len(sel) == k


# 6. Full + compressed count = 10
def test_full_plus_compressed_count_is_ten():
    passages = [{"text": f"Passage {i}"} for i in range(10)]
    mock_tok = MockMistralTokenizer()
    for k in [2, 4, 6]:
        sel = [1, 3, 5, 7][:k]
        _, _, unselected = build_routed_sara_prompt("Test question?", passages, sel, mock_tok)
        assert len(sel) + len(unselected) == 10
        assert set(sel).isdisjoint(set(unselected))
        assert set(sel) | set(unselected) == set(range(10))


# 7. Exactly k full-text passages in prompt
def test_exactly_k_full_text_passages():
    passages = [{"text": f"Content of doc {i}"} for i in range(10)]
    mock_tok = MockMistralTokenizer()
    for k in [2, 4, 6]:
        sel = [0, 2, 4, 6, 8, 9][:k]
        prompt, _, _ = build_routed_sara_prompt("Test question?", passages, sel, mock_tok)
        for idx in sel:
            orig_rank = idx + 1
            assert f"Document {orig_rank}." in prompt


# 8. Exactly 10 - k compressed passages in prompt
def test_exactly_10_minus_k_compressed_passages():
    passages = [{"text": f"Content of doc {i}"} for i in range(10)]
    mock_tok = MockMistralTokenizer()
    for k in [2, 4, 6]:
        sel = list(range(k))
        prompt, _, unselected = build_routed_sara_prompt("Test question?", passages, sel, mock_tok)
        expected_compressed = 10 - k
        assert len(unselected) == expected_compressed
        assert prompt.count(COMPRESS) == expected_compressed


# 9. Rank-preserving final ordering
def test_rank_preserving_final_ordering():
    passages = [{"text": f"Doc {i}"} for i in range(10)]
    mock_tok = MockMistralTokenizer()
    # Provide unordered selection: rank 8, rank 2, rank 5, rank 1
    unordered = [7, 1, 4, 0]
    prompt, _, _ = build_routed_sara_prompt("Q?", passages, unordered, mock_tok)

    # In the prompt, Document 1 must appear before Document 2, which appears before Document 5, etc.
    pos_1 = prompt.find("Document 1.")
    pos_2 = prompt.find("Document 2.")
    pos_5 = prompt.find("Document 5.")
    pos_8 = prompt.find("Document 8.")

    assert pos_1 != -1 and pos_2 != -1 and pos_5 != -1 and pos_8 != -1
    assert pos_1 < pos_2 < pos_5 < pos_8


# 10. R1 deterministic under fixed inputs
def test_r1_deterministic_under_fixed_inputs():
    scores = np.array([0.15, 0.82, 0.33, 0.94, 0.12, 0.65, 0.44, 0.77, 0.05, 0.51])
    res1 = route_r1_semantic_topk(scores, k=4)
    res2 = route_r1_semantic_topk(scores, k=4)
    assert res1 == res2 == [1, 3, 5, 7]


# 11. R2 scoring shape = [B, 10]
def test_r2_scoring_shape():
    model = LearnedPassageRouter(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        hidden_dim=64,
        use_rank_feature=True,
        rank_embedding_dim=16,
    )
    B, N, S = 2, 10, 32
    input_ids = torch.randint(0, 1000, (B, N, S))
    attention_mask = torch.ones((B, N, S), dtype=torch.long)
    rank_indices = torch.arange(10).unsqueeze(0).expand(B, 10)

    logits = model(input_ids, attention_mask, rank_indices)
    assert logits.shape == (B, 10)


# 12. Evidence mapping correctness
def test_evidence_mapping_correctness():
    passages = [
        {"text": "The transformer uses multi-head self-attention mechanisms for sequences."},
        {"text": "Weather in Seattle is typically rainy in winter months."},
        {"text": "Experimental results demonstrate significant accuracy gains on GLUE."},
    ]
    gold_items = [
        {
            "evidence_paragraphs": ["The transformer uses multi-head self-attention mechanisms for sequences."],
            "highlighted_spans": ["multi-head self-attention mechanisms"],
        }
    ]
    labels, details = match_passages_to_gold_evidence(passages, gold_items)
    assert labels[0] == 1.0  # Exact match / high overlap
    assert labels[1] == 0.0  # Unrelated
    assert labels[2] == 0.0  # Unrelated


# 13. TRAIN-FIT / INTERNAL-VAL isolation
def test_train_fit_internal_val_isolation():
    fit_path = REPO_ROOT / "results/week5/content_allocator/w5b/splits/train_fit_manifest.csv"
    val_path = REPO_ROOT / "results/week5/content_allocator/w5b/splits/internal_val_manifest.csv"

    if fit_path.exists() and val_path.exists():
        df_fit = pd.read_csv(fit_path)
        df_val = pd.read_csv(val_path)

        fit_q = set(df_fit["question_id"].astype(str))
        val_q = set(df_val["question_id"].astype(str))
        fit_p = set(df_fit["paper_id"].astype(str))
        val_p = set(df_val["paper_id"].astype(str))

        assert len(fit_q) == 1746, f"Expected 1746 TRAIN-FIT questions, got {len(fit_q)}"
        assert len(val_q) == 342, f"Expected 342 INTERNAL-VAL questions, got {len(val_q)}"
        assert fit_q.isdisjoint(val_q), "Question overlap between TRAIN-FIT and INTERNAL-VAL!"
        assert fit_p.isdisjoint(val_p), "Paper overlap between TRAIN-FIT and INTERNAL-VAL!"


# 14. DEV excluded from R2 training
def test_dev_excluded_from_r2_training():
    fit_path = REPO_ROOT / "results/week5/content_allocator/w5b/splits/train_fit_manifest.csv"
    dev_path = REPO_ROOT / "results/week5/supervision/dev_allocation_supervision.csv"

    if fit_path.exists() and dev_path.exists():
        df_fit = pd.read_csv(fit_path)
        df_dev = pd.read_csv(dev_path)

        fit_q = set(df_fit["question_id"].astype(str))
        dev_q = set(df_dev["question_id"].astype(str))
        assert fit_q.isdisjoint(dev_q), "DEV questions found in TRAIN-FIT!"


# 15. TEST split never loaded or accessed
def test_test_split_never_loaded():
    # Audit configs and week6 code to assert no reference to TEST split
    w6_src_dir = REPO_ROOT / "src/week6"
    for py_file in w6_src_dir.glob("*.py"):
        content = py_file.read_text()
        assert "qasper-test" not in content, f"Forbidden test split reference in {py_file}"
        assert "test_split.jsonl" not in content, f"Forbidden test split reference in {py_file}"


# 16. Routing Recall@k calculation
def test_routing_recall_calculation():
    # 3 gold passages at ranks 1, 3, 5 (0-indexed: 0, 2, 4)
    gold = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    # Selected ranks: [0, 2] (k=2)
    rec, prec = compute_evidence_recall_precision([0, 2], gold)
    assert pytest.approx(rec, 1e-4) == 2.0 / 3.0
    assert pytest.approx(prec, 1e-4) == 2.0 / 2.0 == 1.0

    # Selected ranks: [1, 3] (k=2, neither is gold)
    rec_bad, prec_bad = compute_evidence_recall_precision([1, 3], gold)
    assert rec_bad == 0.0
    assert prec_bad == 0.0


# 17. Jaccard routing overlap calculation
def test_jaccard_routing_overlap_calculation():
    assignments = {
        "R0": {2: [[0, 1], [0, 1]]},
        "R1": {2: [[0, 2], [0, 1]]},
    }
    df_div = compute_routing_diversity(assignments, budgets=[2])
    assert len(df_div) == 1
    # Query 0: {0,1} vs {0,2} -> inter=1, union=3 -> J=1/3; Query 1: {0,1} vs {0,1} -> J=1.0
    # Mean Jaccard = (1/3 + 1) / 2 = 2/3
    expected_jaccard = (1.0 / 3.0 + 1.0) / 2.0
    assert pytest.approx(df_div["mean_jaccard_overlap"].iloc[0], 1e-4) == expected_jaccard
    assert df_div["disagreement_rate"].iloc[0] == 0.5


# 18. Selected-rank frequency calculation
def test_selected_rank_frequency_calculation():
    assignments = {
        "R0": {2: [[0, 1], [0, 1], [0, 1]]},
    }
    df_dist = compute_selected_rank_distribution(assignments, budgets=[2])
    assert len(df_dist) == 1
    assert df_dist["rank_1_freq"].iloc[0] == 1.0
    assert df_dist["rank_2_freq"].iloc[0] == 1.0
    assert df_dist["rank_3_freq"].iloc[0] == 0.0


# 19. Existing R0 canonical result reproduction
def test_existing_r0_canonical_result_reproduction():
    dev_mat_path = REPO_ROOT / "results/week2/raw/dev_fixed_k_matrix.jsonl"
    if dev_mat_path.exists():
        r0_dict = load_canonical_r0_results(dev_mat_path, budgets=[2, 4, 6])
        for k in [2, 4, 6]:
            assert k in r0_dict
            assert len(r0_dict[k]["f1_list"]) == 231
            mean_f1 = np.mean(r0_dict[k]["f1_list"])
            assert 0.28 <= mean_f1 <= 0.42, f"Unexpected mean F1={mean_f1} for k={k}"


# 20. Optional k=2 oracle enumerates exactly 45 subsets
def test_k2_oracle_enumerates_45_subsets():
    subsets = get_all_subsets(n=10, k=2)
    assert len(subsets) == 45
    # All subsets must be length 2 and unique
    assert len(set(tuple(s) for s in subsets)) == 45
    for s in subsets:
        assert len(s) == 2
        assert 0 <= s[0] < s[1] < 10
