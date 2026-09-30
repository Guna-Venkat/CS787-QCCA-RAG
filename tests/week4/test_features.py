"""Unit tests for Week 4 feature processing, schema, and leakage audits."""

import pandas as pd
import pytest

from src.week4.features import audit_for_leakage, load_retrieval_features


def test_feature_loading_and_alignment():
    """Verify features load cleanly with 231 questions and 86 papers."""
    df_feat = load_retrieval_features("results/week2/processed/dev_retrieval_features.jsonl")
    assert len(df_feat) == 231
    assert df_feat["question_id"].nunique() == 231
    assert df_feat["paper_id"].nunique() == 86
    assert not df_feat.isna().any().any()


def test_leakage_detection_rejects_forbidden_columns():
    """Artificially introduce forbidden post-generation columns and verify rejection."""
    df_clean = pd.DataFrame({
        "question_id": ["q1", "q2"],
        "paper_id": ["p1", "p2"],
        "rho_1": [0.4, 0.5],
        "delta_12": [0.1, 0.2]
    })
    # Clean df must pass
    audit_for_leakage(df_clean)

    # Test F1 leakage
    df_f1 = df_clean.copy()
    df_f1["F1_score"] = [0.5, 0.6]
    with pytest.raises(ValueError, match="Data Leakage Detected"):
        audit_for_leakage(df_f1)

    # Test gold answer leakage
    df_gold = df_clean.copy()
    df_gold["gold_answer"] = ["A", "B"]
    with pytest.raises(ValueError, match="Data Leakage Detected"):
        audit_for_leakage(df_gold)

    # Test oracle gain leakage
    df_gain = df_clean.copy()
    df_gain["oracle_gain"] = [0.1, 0.2]
    with pytest.raises(ValueError, match="Data Leakage Detected"):
        audit_for_leakage(df_gain)
