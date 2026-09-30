"""Tests for data alignment, question-id integrity, and joins."""

import json
from pathlib import Path
import pandas as pd
import pytest


def test_frozen_week2_alignment_and_integrity():
    raw_path = "results/week2/raw/dev_fixed_k_matrix.jsonl"
    feat_path = "results/week2/processed/dev_retrieval_features.jsonl"
    mat_path = "results/week2/processed/dev_per_query_k_matrix.csv"
    
    assert Path(raw_path).exists(), f"Missing {raw_path}"
    assert Path(feat_path).exists(), f"Missing {feat_path}"
    assert Path(mat_path).exists(), f"Missing {mat_path}"
    
    # 1. Raw matrix check
    records = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    df_raw = pd.DataFrame(records)
    
    assert len(df_raw) == 1617
    assert df_raw["question_id"].nunique() == 231
    assert df_raw["paper_id"].nunique() == 86
    assert not df_raw.duplicated(subset=["question_id", "k"]).any()
    
    # Check all 7 k values present for each question
    k_counts = df_raw.groupby("question_id")["k"].apply(set)
    expected_k = {0, 2, 4, 5, 6, 8, 10}
    for q_id, present_k in k_counts.items():
        assert present_k == expected_k, f"Question {q_id} has k={present_k} != {expected_k}"
        
    # 2. Features check
    feat_records = []
    with open(feat_path, "r", encoding="utf-8") as f:
        for line in f:
            feat_records.append(json.loads(line))
    df_feat = pd.DataFrame(feat_records)
    
    assert len(df_feat) == 231
    assert not df_feat["question_id"].duplicated().any()
    assert set(df_feat["question_id"].astype(str)) == set(df_raw["question_id"].astype(str))
    
    # 3. Per query matrix check
    df_mat = pd.read_csv(mat_path)
    assert len(df_mat) == 231
    assert set(df_mat["question_id"].astype(str)) == set(df_raw["question_id"].astype(str))
