"""Unit tests for output record schema validation."""

import pytest
from typing import Dict, Any


REQUIRED_FIELDS = {
    "question_id",
    "paper_id",
    "k",
    "method",
    "prediction",
    "gold_answers",
    "retrieved_passage_ids",
    "bm25_scores",
    "rho_1",
    "delta_12",
    "entropy",
    "entropy_normalized",
    "query_length",
    "n_high",
    "input_tokens",
    "context_tokens",
    "generated_tokens",
    "token_f1",
    "rouge_l",
    "exact_match",
    "generation_latency",
    "peak_vram",
    "checkpoint_id",
    "model_name",
    "seed",
}


def validate_record_schema(record: Dict[str, Any]) -> bool:
    missing = REQUIRED_FIELDS - set(record.keys())
    if missing:
        raise KeyError(f"Missing required fields: {missing}")
    return True


def test_valid_record_schema():
    valid_record = {
        "question_id": "1912",
        "paper_id": "638",
        "k": 5,
        "method": "sara",
        "prediction": "WN18 and FB15k",
        "gold_answers": ["WN18 and FB15k"],
        "retrieved_passage_ids": [f"638_chunk_{i}" for i in range(10)],
        "bm25_scores": [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.04, 0.02, 0.01],
        "rho_1": 0.31,
        "delta_12": 0.20,
        "entropy": 1.5,
        "entropy_normalized": 0.65,
        "query_length": 8,
        "n_high": 2,
        "input_tokens": 1250,
        "context_tokens": 1180,
        "generated_tokens": 6,
        "token_f1": 1.0,
        "rouge_l": 1.0,
        "exact_match": 1.0,
        "generation_latency": 0.85,
        "peak_vram": 13.55,
        "checkpoint_id": "sara_qasper_proj_lr5e4_seed42",
        "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
        "seed": 42,
    }
    
    assert validate_record_schema(valid_record) is True


def test_missing_fields():
    incomplete_record = {
        "question_id": "1912",
        "k": 5,
    }
    with pytest.raises(KeyError) as exc_info:
        validate_record_schema(incomplete_record)
    assert "Missing required fields" in str(exc_info.value)
