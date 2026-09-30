"""Unit tests for Week 4 target loading, validation, and action space."""

import pandas as pd
import pytest

from src.week4.features import load_retrieval_features
from src.week4.targets import load_epsilon_targets, prepare_training_dataset


def test_target_loading_and_properties():
    """Verify target dataset has 231 rows at epsilon*=0.01 within K_alloc={2,4,5,6,8}."""
    df_targets = load_epsilon_targets(
        targets_path="results/week3/processed/epsilon_oracle_all.csv",
        epsilon_star=0.01
    )
    assert len(df_targets) == 231
    assert set(df_targets["selected_k_epsilon"].unique()).issubset({2, 4, 5, 6, 8})
    assert 0 not in df_targets["selected_k_epsilon"].values
    assert 10 not in df_targets["selected_k_epsilon"].values


def test_feature_target_join_and_paper_alignment():
    """Verify one-to-one join on question_id preserves paper_id exactly."""
    df_feat = load_retrieval_features("results/week2/processed/dev_retrieval_features.jsonl")
    df_targets = load_epsilon_targets("results/week3/processed/epsilon_oracle_all.csv", epsilon_star=0.01)
    
    df_data = prepare_training_dataset(df_feat, df_targets)
    assert len(df_data) == 231
    assert "target_k" in df_data.columns
    assert "paper_id" in df_data.columns
    assert df_data["paper_id"].nunique() == 86
