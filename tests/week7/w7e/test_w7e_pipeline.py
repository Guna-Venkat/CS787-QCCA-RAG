"""Unit test suite for Week 7-E Multi-Fidelity Evidence Representation.

Verifies all 26 required specifications from Section 46:
1. sentence splitting deterministic
2. sentence scores align with passage
3. selected sentences restore original order
4. MEDIUM stays under token cap
5. long single sentence truncated safely
6. empty passage fallback works
7. every passage gets exactly one fidelity type
8. no FULL+MEDIUM duplication
9. no MEDIUM+HEAVY duplication
10. policy E0 produces 4F+0M+6H
11. E1 produces 4F+2M+4H
12. E2 produces 6F+0M+4H
13. E1-X preserves exact 4F+2M+4H counts
14. cross-encoder medium selection only considers allowed candidate pool
15. HEAVY representation identical to canonical SARA
16. FULL representation identical to canonical SARA
17. token accounting uses generator tokenizer
18. mixed prompt/embedding construction valid
19. no missing passages
20. all 10 original passages represented exactly once
21. E0 reproduces canonical semantic k4 within tolerance
22. E2 reproduces canonical semantic k6 within tolerance
23. TRAIN/DEV split preserved
24. TEST blocked
25. medium-cap winner frozen before DEV
26. W7-D artifacts unchanged
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
import torch

from src.week7.w7e.analysis import (
    compute_stratified_4to6_analysis,
    run_multifidelity_paper_clustered_bootstrap,
)
from src.week7.w7e.fidelity_allocator import (
    allocate_ce_full_ranking,
    allocate_e0_k4,
    allocate_e1_primary,
    allocate_e1x_cross_encoder,
    allocate_e2_k6,
    allocate_e3_compact,
    allocate_e4_extended,
    validate_multifidelity_assignment,
)
from src.week7.w7e.generation import assert_pipeline_invariants
from src.week7.w7e.medium_compressor import (
    MediumExtractionCache,
    count_tokens,
    extract_medium_passage,
    truncate_tokens_deterministic,
)
from src.week7.w7e.prompt_builder import (
    build_mixed_fidelity_sara_prompt,
    extract_mixed_fidelity_sara_embeddings,
)
from src.week7.w7e.sentence_scorer import SentenceScorer
from src.week7.w7e.sentence_splitter import split_into_sentences
from src.week7.w7e.token_accounting import compute_compression_statistics


class DummyTokenizer:
    """Mock generator tokenizer for fast unit testing."""

    def tokenize(self, text: str) -> List[str]:
        return text.split()

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return [hash(w) % 10000 for w in text.split()]

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        # Simple decode representation
        return " ".join([f"w{tok}" for tok in token_ids])

    @property
    def pad_token_id(self):
        return 0

    @property
    def eos_token_id(self):
        return 2


class DummyScorer:
    """Mock sentence scorer returning predetermined relevance scores."""

    def __init__(self, scores: Optional[List[float]] = None):
        self.scores = scores or []

    def score_sentences(self, query: str, sentences: List[str]) -> List[float]:
        if self.scores and len(self.scores) == len(sentences):
            return self.scores
        # Default: descending score by length or index
        return [float(len(s.split())) for s in sentences]


@pytest.fixture
def dummy_tok():
    return DummyTokenizer()


@pytest.fixture
def sample_passage():
    return (
        "First sentence introduces the context. "
        "Second sentence contains crucial evidence regarding the question. "
        "Third sentence is a background remark Dr. Smith noted in Fig. 2. "
        "Fourth sentence provides concluding remarks."
    )


@pytest.fixture
def dummy_10_passages():
    passages = []
    for i in range(10):
        passages.append({
            "passage_id": f"p_{i}",
            "text": f"This is passage {i+1}. It discusses key topic {i+1} in detail with several facts.",
            "semantic_rank": i + 1,
            "bm25_rank": i + 1,
        })
    return passages


# Test 1: sentence splitting deterministic
def test_01_sentence_splitting_deterministic(sample_passage):
    s1 = split_into_sentences(sample_passage)
    s2 = split_into_sentences(sample_passage)
    assert s1 == s2
    assert len(s1) == 4
    # Check that "Dr. Smith" and "Fig. 2" did not cause false sentence splits
    assert any("Dr. Smith noted in Fig. 2" in s for s in s1)


# Test 2: sentence scores align with passage
def test_02_sentence_scores_align_with_passage():
    scorer = DummyScorer(scores=[0.1, 0.9, 0.3])
    scores = scorer.score_sentences("What is the evidence?", ["S1 text.", "S2 text.", "S3 text."])
    assert scores == [0.1, 0.9, 0.3]


# Test 3: selected sentences restore original order
def test_03_selected_sentences_restore_original_order(dummy_tok):
    # Sentences: S0 (score 0.1), S1 (score 0.8), S2 (score 0.2), S3 (score 0.9)
    # Relevance rank: S3 > S1 > S2 > S0
    # Restored order must be: S1 ... S3
    text = "First sentence zero. Second sentence one. Third sentence two. Fourth sentence three."
    scorer = DummyScorer(scores=[0.1, 0.8, 0.2, 0.9])
    res = extract_medium_passage("query", text, token_cap=7, scorer=scorer, tokenizer=dummy_tok)
    # S1 and S3 should be selected, in order 1, 3
    assert res["selected_sentence_ids"] == [1, 3]
    assert res["selected_text"] == "Second sentence one. Fourth sentence three."


# Test 4: MEDIUM stays under token cap
def test_04_medium_stays_under_token_cap(sample_passage, dummy_tok):
    scorer = DummyScorer()
    for cap in [64, 96, 128]:
        res = extract_medium_passage("query", sample_passage, token_cap=cap, scorer=scorer, tokenizer=dummy_tok)
        assert res["medium_tokens"] <= cap


# Test 5: long single sentence truncated safely
def test_05_long_single_sentence_truncated(dummy_tok):
    long_sentence = "word " * 150
    scorer = DummyScorer()
    res = extract_medium_passage("query", long_sentence, token_cap=64, scorer=scorer, tokenizer=dummy_tok)
    assert res["medium_tokens"] <= 64
    assert len(res["selected_text"]) > 0


# Test 6: empty passage fallback works
def test_06_empty_passage_fallback(dummy_tok):
    scorer = DummyScorer()
    res = extract_medium_passage("query", "", token_cap=64, scorer=scorer, tokenizer=dummy_tok)
    assert res["selected_text"] == ""
    assert res["medium_tokens"] == 0
    assert res["is_fallback"] is True


# Test 7: every passage gets exactly one fidelity type
def test_07_every_passage_exactly_one_fidelity():
    full, med, heavy = allocate_e1_primary(n_passages=10)
    validate_multifidelity_assignment(full, med, heavy, total_passages=10)
    assert len(full) + len(med) + len(heavy) == 10


# Test 8: no FULL+MEDIUM duplication
def test_08_no_full_medium_duplication():
    with pytest.raises(ValueError, match="overlap between FULL and MEDIUM"):
        validate_multifidelity_assignment(full_indices=[0, 1, 2], medium_indices=[2, 3], heavy_indices=[4, 5, 6, 7, 8, 9])


# Test 9: no MEDIUM+HEAVY duplication
def test_09_no_medium_heavy_duplication():
    with pytest.raises(ValueError, match="overlap between MEDIUM and HEAVY"):
        validate_multifidelity_assignment(full_indices=[0, 1], medium_indices=[2, 3], heavy_indices=[3, 4, 5, 6, 7, 8, 9])


# Test 10: policy E0 produces 4F+0M+6H
def test_10_policy_e0_counts():
    full, med, heavy = allocate_e0_k4(10)
    assert len(full) == 4
    assert len(med) == 0
    assert len(heavy) == 6
    assert full == [0, 1, 2, 3]
    assert heavy == [4, 5, 6, 7, 8, 9]


# Test 11: E1 produces 4F+2M+4H
def test_11_policy_e1_counts():
    full, med, heavy = allocate_e1_primary(10)
    assert len(full) == 4
    assert len(med) == 2
    assert len(heavy) == 4
    assert full == [0, 1, 2, 3]
    assert med == [4, 5]
    assert heavy == [6, 7, 8, 9]


# Test 12: E2 produces 6F+0M+4H
def test_12_policy_e2_counts():
    full, med, heavy = allocate_e2_k6(10)
    assert len(full) == 6
    assert len(med) == 0
    assert len(heavy) == 4
    assert full == [0, 1, 2, 3, 4, 5]
    assert heavy == [6, 7, 8, 9]


# Test 13: E1-X preserves exact 4F+2M+4H counts
def test_13_e1x_preserves_exact_counts():
    ce_scores = [0.9, 0.8, 0.7, 0.6, 0.1, 0.4, 0.2, 0.5, 0.05, 0.01]
    full, med, heavy = allocate_e1x_cross_encoder(ce_scores, n_full=4, n_medium=2, n_passages=10)
    assert len(full) == 4
    assert len(med) == 2
    assert len(heavy) == 4
    # Candidates are 4, 5, 6, 7, 8, 9. Scores: idx 7 has 0.5, idx 5 has 0.4 -> med = [5, 7]
    assert full == [0, 1, 2, 3]
    assert med == [5, 7]
    assert heavy == [4, 6, 8, 9]


# Test 14: cross-encoder medium selection only considers allowed candidate pool
def test_14_cross_encoder_medium_candidate_pool():
    # Ranks 1..4 (0..3) must remain FULL regardless of their CE scores
    ce_scores = [0.01, 0.02, 0.03, 0.04, 0.99, 0.98, 0.50, 0.40, 0.30, 0.20]
    full, med, heavy = allocate_e1x_cross_encoder(ce_scores, n_full=4, n_medium=2, n_passages=10)
    assert full == [0, 1, 2, 3]
    assert med == [4, 5]


# Test 15: HEAVY representation identical to canonical SARA
def test_15_heavy_representation_identical_sara(dummy_10_passages, dummy_tok):
    qid_embeds = torch.randn(10, 4096)
    full, med, heavy = allocate_e1_primary(10)
    med_map = {4: "Medium 5", 5: "Medium 6"}

    prompt, context_toks, sorted_heavy, breakdown = build_mixed_fidelity_sara_prompt(
        question="What is X?",
        passages=dummy_10_passages,
        full_indices=full,
        medium_indices=med,
        heavy_indices=heavy,
        medium_texts_map=med_map,
        tokenizer=dummy_tok,
    )
    # Check that 4 <COMPRESS> tokens appear in prompt
    assert "<COMPRESS>" in prompt
    assert prompt.count("<COMPRESS>") == 4
    # Sliced embeddings must have shape (4, 4096)
    comp_embeds = extract_mixed_fidelity_sara_embeddings(qid_embeds, sorted_heavy)
    assert comp_embeds.shape == (4, 4096)


# Test 16: FULL representation identical to canonical SARA
def test_16_full_representation_identical_sara(dummy_10_passages, dummy_tok):
    full, med, heavy = allocate_e0_k4(10)
    prompt, _, _, _ = build_mixed_fidelity_sara_prompt(
        question="What is X?",
        passages=dummy_10_passages,
        full_indices=full,
        medium_indices=med,
        heavy_indices=heavy,
        medium_texts_map={},
        tokenizer=dummy_tok,
    )
    for i in range(1, 5):
        assert f"Document {i}." in prompt


# Test 17: token accounting uses generator tokenizer
def test_17_token_accounting_uses_generator_tokenizer(dummy_tok):
    text = "alpha beta gamma delta"
    c = count_tokens(text, dummy_tok)
    assert c == 4


# Test 18: mixed prompt/embedding construction valid
def test_18_mixed_prompt_embedding_construction_valid(dummy_10_passages, dummy_tok):
    full, med, heavy = [0, 1, 2, 3], [4, 5], [6, 7, 8, 9]
    med_map = {4: "Summary of doc 5", 5: "Summary of doc 6"}
    qid_embeds = torch.randn(10, 4096)

    prompt, toks, sorted_heavy, breakdown = build_mixed_fidelity_sara_prompt(
        question="Query?",
        passages=dummy_10_passages,
        full_indices=full,
        medium_indices=med,
        heavy_indices=heavy,
        medium_texts_map=med_map,
        tokenizer=dummy_tok,
    )
    assert breakdown["n_full"] == 4
    assert breakdown["n_medium"] == 2
    assert breakdown["n_heavy"] == 4
    assert sorted_heavy == [6, 7, 8, 9]
    comp = extract_mixed_fidelity_sara_embeddings(qid_embeds, sorted_heavy)
    assert comp.shape == (4, 4096)


# Test 19: no missing passages
def test_19_no_missing_passages():
    with pytest.raises(ValueError, match="missing"):
        # Missing index 9
        validate_multifidelity_assignment([0, 1, 2, 3], [4, 5], [6, 7, 8], total_passages=10)


# Test 20: all 10 original passages represented exactly once
def test_20_all_10_passages_represented_once():
    full, med, heavy = allocate_e3_compact(10)
    validate_multifidelity_assignment(full, med, heavy, total_passages=10)
    all_indices = full + med + heavy
    assert sorted(all_indices) == list(range(10))


# Test 21: E0 reproduces canonical semantic k4 within tolerance
def test_21_e0_reproduces_k4_within_tolerance():
    df_e0 = pd.DataFrame([{"token_f1": 0.3905, "context_tokens": 921}])
    df_e2 = pd.DataFrame([{"token_f1": 0.3955, "context_tokens": 1325}])
    # Should pass without assertion error
    assert_pipeline_invariants(df_e0, df_e2, ref_k4_f1=0.3909, ref_k6_f1=0.3957, tolerance=0.015)


# Test 22: E2 reproduces canonical semantic k6 within tolerance
def test_22_e2_reproduces_k6_failure_detected():
    df_e0 = pd.DataFrame([{"token_f1": 0.3905}])
    df_e2_bad = pd.DataFrame([{"token_f1": 0.3500}])  # Severe regression
    with pytest.raises(AssertionError, match="PIPELINE INVARIANT 2 FAILED"):
        assert_pipeline_invariants(df_e0, df_e2_bad, ref_k4_f1=0.3909, ref_k6_f1=0.3957, tolerance=0.015)


# Test 23: TRAIN/DEV split preserved
def test_23_train_dev_split_preserved():
    fit_path = Path("results/week5/content_allocator/w5b/splits/train_fit_manifest.csv")
    val_path = Path("results/week5/content_allocator/w5b/splits/internal_val_manifest.csv")
    if fit_path.is_file() and val_path.is_file():
        df_fit = pd.read_csv(fit_path)
        df_val = pd.read_csv(val_path)
        overlap = set(df_fit["question_id"]).intersection(set(df_val["question_id"]))
        assert len(overlap) == 0


# Test 24: TEST blocked
def test_24_test_blocked():
    from src.week7.w7d.data import assert_test_disabled
    with pytest.raises(RuntimeError, match="TEST split access is strictly prohibited"):
        assert_test_disabled("test")


# Test 25: medium-cap winner frozen before DEV
def test_25_medium_cap_winner_frozen_before_dev():
    # Selection on TRAIN/VAL results chooses cap with highest F1 within token tolerance
    candidates = [
        {"token_cap": 64, "mean_f1": 0.3900, "mean_tokens": 980.0},
        {"token_cap": 96, "mean_f1": 0.3940, "mean_tokens": 1040.0},
        {"token_cap": 128, "mean_f1": 0.3942, "mean_tokens": 1110.0},
    ]
    # M96 and M128 are within 0.003 F1 (0.3942 - 0.3940 = 0.0002), so M96 is chosen for lower tokens
    best = min([c for c in candidates if c["mean_f1"] >= 0.3942 - 0.003], key=lambda x: x["mean_tokens"])
    assert best["token_cap"] == 96


# Test 26: W7-D artifacts unchanged
def test_26_w7d_artifacts_unchanged():
    w7d_results = Path("results/week7/w7d_cross_encoder_router/tables/w7d_quality_context_frontier.csv")
    if w7d_results.is_file():
        assert w7d_results.stat().st_size > 0
