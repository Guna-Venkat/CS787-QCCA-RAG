"""Unit tests for GroupKFold cross-validation and disjoint document partitions."""

import pandas as pd
import pytest

from src.week4.cross_validation import run_grouped_cross_validation, verify_oof_integrity
from src.week4.features import load_retrieval_features
from src.week4.targets import load_epsilon_targets, prepare_training_dataset


def test_groupkfold_paper_disjointness():
    """Verify for every fold that train papers and val papers are strictly disjoint."""
    df_feat = load_retrieval_features("results/week2/processed/dev_retrieval_features.jsonl")
    df_targets = load_epsilon_targets("results/week3/processed/epsilon_oracle_all.csv", epsilon_star=0.01)
    df_data = prepare_training_dataset(df_feat, df_targets)

    feature_cols = ["rho_1", "delta_12", "entropy_normalized", "n_high"]
    df_oof, cv_meta = run_grouped_cross_validation(
        df_data=df_data,
        feature_cols=feature_cols,
        n_splits=5,
        random_state=42
    )

    for fold_info in cv_meta["folds"]:
        assert fold_info["paper_overlap_count"] == 0, f"Overlap in fold {fold_info['fold']}"

    assert len(df_oof) == 231
    assert df_oof["question_id"].nunique() == 231
    verify_oof_integrity(df_oof, expected_count=231, k_alloc=[2, 4, 5, 6, 8])


def test_cv_reproducibility():
    """Verify that identical random seeds yield byte-identical OOF predictions."""
    df_feat = load_retrieval_features("results/week2/processed/dev_retrieval_features.jsonl")
    df_targets = load_epsilon_targets("results/week3/processed/epsilon_oracle_all.csv", epsilon_star=0.01)
    df_data = prepare_training_dataset(df_feat, df_targets)

    feature_cols = ["rho_1", "delta_12", "entropy_normalized", "n_high"]
    df_oof1, _ = run_grouped_cross_validation(df_data, feature_cols, n_splits=5, random_state=42)
    df_oof2, _ = run_grouped_cross_validation(df_data, feature_cols, n_splits=5, random_state=42)

    pd.testing.assert_frame_equal(df_oof1, df_oof2)

