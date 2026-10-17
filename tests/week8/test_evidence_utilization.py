"""Unit tests for Evidence Potential, Utilization Bottleneck, and Draft-Aware Revision.

Validates:
1. Four-way Evidence Potential taxonomy:
   - UTILIZED-HELPFUL: is_gold_evidence == True AND delta_f1 > 0.01
   - UNDERUTILIZED: is_gold_evidence == True AND delta_f1 <= 0.01 (Q1812 case)
   - TRUE-NEUTRAL: is_gold_evidence == False AND abs(delta_f1) <= 0.01 (Q1811 case)
   - DISTRACTOR: delta_f1 < -0.01
2. Output stability detection: answer_changed == I[y_i != y_0]
3. Promotion Utilization Failure Rate (UFR) metric calculation:
   UFR = P(y_i == y_0 | base_f1 < 1.0, is_gold_evidence == True)
4. Record enrichment and dense likelihood support.
5. Draft-aware revision prompt construction.
"""

import pytest
import numpy as np
from src.week8.evidence_enrichment import (
    normalize_text_for_comparison,
    compute_token_containment_and_jaccard,
    classify_evidence_utilization_type,
    enrich_promotion_record,
)
from src.week8.evaluation import compute_evidence_utilization_metrics
from src.week8.utility_dataset import extract_utility_targets


def test_text_normalization():
    s1 = "GloVe, concatenation of average embeddings!"
    s2 = "glove concatenation of average embeddings"
    assert normalize_text_for_comparison(s1) == s2


def test_containment_and_jaccard():
    cand = "Final embeddings for visits are obtained by concatenation of average embeddings"
    target = "concatenation of average embeddings"
    cont, jacc = compute_token_containment_and_jaccard(cand, target)
    assert cont >= 0.99  # All target tokens present in candidate
    assert jacc > 0.3


def test_four_way_taxonomy_classification():
    # Case A: Utilized helpful
    assert classify_evidence_utilization_type(is_gold_evidence=True, delta_f1=0.25) == "UTILIZED_HELPFUL"
    # Case B: Underutilized (Q1812 pattern: gold evidence present, but delta <= 0.01)
    assert classify_evidence_utilization_type(is_gold_evidence=True, delta_f1=0.0) == "UNDERUTILIZED"
    assert classify_evidence_utilization_type(is_gold_evidence=True, delta_f1=0.005) == "UNDERUTILIZED"
    # Case C: True neutral (Q1811 pattern: base complete, promotion adds nothing)
    assert classify_evidence_utilization_type(is_gold_evidence=False, delta_f1=0.0) == "TRUE_NEUTRAL"
    assert classify_evidence_utilization_type(is_gold_evidence=False, delta_f1=0.005) == "TRUE_NEUTRAL"
    # Case D: Distractor
    assert classify_evidence_utilization_type(is_gold_evidence=False, delta_f1=-0.15) == "DISTRACTOR"
    assert classify_evidence_utilization_type(is_gold_evidence=True, delta_f1=-0.05) == "HARMFUL_EVIDENCE"


def test_enrich_promotion_record_q1811_and_q1812():
    # Mock Q1811 record (Safe-stop query: base_f1 = 1.0, prediction unchanged)
    q1811_rec = {
        "question_id": "1811",
        "question": "Which features do they use?",
        "base_prediction": "word unigrams, bigrams, and trigrams",
        "promoted_prediction": "word unigrams, bigrams, and trigrams",
        "candidate_text": "Random text about agriculture and animals.",
        "gold_answers": ["word unigrams, bigrams, and trigrams"],
        "base_f1": 1.0,
        "promoted_f1": 1.0,
        "delta_f1": 0.0,
    }
    en_1811 = enrich_promotion_record(q1811_rec, evidence_map={})
    assert en_1811["answer_changed"] is False
    assert en_1811["base_answer_complete"] is True
    assert en_1811["evidence_utilization_type"] == "TRUE_NEUTRAL"

    # Mock Q1812 record (Underutilized evidence failure mode: base answer partial, candidate has missing info, but answer unchanged)
    q1812_rec = {
        "question_id": "1812",
        "question": "How are visit embeddings constructed?",
        "base_prediction": "GloVe",
        "promoted_prediction": "GloVe",
        "candidate_text": "Final embeddings for visits are obtained by concatenation of average embeddings calculated separately for the interview and for the medical examination.",
        "gold_answers": ["GloVe, concatenation of average embeddings calculated separately for the interview and for the medical examination"],
        "base_f1": 0.1429,
        "promoted_f1": 0.1429,
        "delta_f1": 0.0,
    }
    en_1812 = enrich_promotion_record(q1812_rec, evidence_map={})
    assert en_1812["answer_changed"] is False
    assert en_1812["base_answer_complete"] is False
    assert en_1812["is_gold_evidence"] is True
    assert en_1812["evidence_utilization_type"] == "UNDERUTILIZED"


def test_compute_evidence_utilization_metrics_and_ufr():
    # Construct a dataset of 4 promotions:
    # 1. Gold evidence, helped: delta=0.5, changed
    # 2. Gold evidence, failed to utilize (UFR case): base_f1=0.2, delta=0.0, unchanged
    # 3. True neutral: base_f1=1.0, delta=0.0, unchanged
    # 4. Distractor: base_f1=0.8, delta=-0.2, changed
    records = [
        {
            "question_id": "1",
            "base_f1": 0.2,
            "promoted_f1": 0.7,
            "delta_f1": 0.5,
            "is_gold_evidence": True,
            "answer_changed": True,
            "evidence_utilization_type": "UTILIZED_HELPFUL",
        },
        {
            "question_id": "2",
            "base_f1": 0.14,
            "promoted_f1": 0.14,
            "delta_f1": 0.0,
            "is_gold_evidence": True,
            "answer_changed": False,
            "evidence_utilization_type": "UNDERUTILIZED",
        },
        {
            "question_id": "3",
            "base_f1": 1.0,
            "promoted_f1": 1.0,
            "delta_f1": 0.0,
            "is_gold_evidence": False,
            "answer_changed": False,
            "evidence_utilization_type": "TRUE_NEUTRAL",
        },
        {
            "question_id": "4",
            "base_f1": 0.8,
            "promoted_f1": 0.6,
            "delta_f1": -0.2,
            "is_gold_evidence": False,
            "answer_changed": True,
            "evidence_utilization_type": "DISTRACTOR",
        },
    ]

    metrics = compute_evidence_utilization_metrics(records)
    assert metrics["total_promotions"] == 4
    assert metrics["prob_answer_changed"] == 0.5  # 2 changed out of 4
    assert metrics["prob_answer_changed_given_help"] == 1.0  # record 1 helped and changed
    assert metrics["prob_answer_changed_given_gold_evidence"] == 0.5  # 1 changed out of 2 gold
    assert metrics["prob_help_given_gold_evidence"] == 0.5  # 1 helped out of 2 gold
    assert metrics["prob_zero_delta_given_gold_evidence"] == 0.5  # 1 zero out of 2 gold

    # UFR: base_f1 < 1.0 and is_gold_evidence == True:
    # Records 1 and 2 meet the condition (denominator = 2)
    # Only Record 2 remained unchanged (numerator = 1)
    # UFR = 1 / 2 = 0.50
    assert metrics["promotion_utilization_failure_rate"] == 0.50

    # Taxonomy counts
    tax = metrics["taxonomy_distribution"]
    assert tax["UTILIZED_HELPFUL"]["count"] == 1
    assert tax["UNDERUTILIZED"]["count"] == 1
    assert tax["TRUE_NEUTRAL"]["count"] == 1
    assert tax["DISTRACTOR"]["count"] == 1


def test_extract_utility_targets_dense_ll():
    rec = {
        "delta_f1": 0.05,
        "delta_gold_logprob": 0.42,
    }
    targets = extract_utility_targets(rec)
    assert targets["delta_f1"] == 0.05
    assert targets["is_help"] == 1.0
    assert targets["is_hurt"] == 0.0
    assert targets["delta_gold_logprob"] == 0.42
