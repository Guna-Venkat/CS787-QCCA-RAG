"""Tests for statistical methods, bootstrap reproducibility, tie consistency, and audit integrity."""

import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from src.week3.heterogeneity_analysis import compute_paper_clustered_bootstrap_cis


def test_bootstrap_reproducibility():
    records = []
    for paper in ["p1", "p2", "p3", "p4", "p5"]:
        for i in range(4):
            records.append({
                "question_id": f"{paper}_q{i}",
                "paper_id": paper,
                "best_f1": 0.5 + 0.1 * i,
                "static_f1": 0.4,
                "f1_range": 0.2,
                "gain_gt_00": 1 if i > 0 else 0,
                "gain_gt_01": 1 if i > 0 else 0,
                "gain_gt_02": 1 if i > 0 else 0,
                "gain_gt_05": 1 if i > 1 else 0
            })
    df_mock = pd.DataFrame(records)
    
    res1 = compute_paper_clustered_bootstrap_cis(df_mock, n_iterations=100, seed=42)
    res2 = compute_paper_clustered_bootstrap_cis(df_mock, n_iterations=100, seed=42)
    
    for metric in res1:
        assert res1[metric]["mean"] == res2[metric]["mean"]
        assert res1[metric]["ci_lower"] == res2[metric]["ci_lower"]
        assert res1[metric]["ci_upper"] == res2[metric]["ci_upper"]
        assert res1[metric]["ci_lower"] <= res1[metric]["mean"] <= res1[metric]["ci_upper"]


def test_tie_aware_partitioning():
    het_path = Path("results/week3/processed/per_query_heterogeneity.csv")
    assert het_path.exists()
    df_het = pd.read_csv(het_path)
    
    # Check that k8_in_best_set and k8_strictly_suboptimal are mutually exclusive and exhaustive
    for _, row in df_het.iterrows():
        assert bool(row["k8_in_best_set"]) != bool(row["k8_strictly_suboptimal"]), (
            f"Query {row['question_id']} invalid: in_best={row['k8_in_best_set']}, subopt={row['k8_strictly_suboptimal']}"
        )
    assert df_het["k8_strictly_suboptimal"].sum() == 68
    assert df_het["k8_in_best_set"].sum() == 163
    assert len(df_het) == 231


def test_gain_attribution_sum():
    attr_path = Path("results/week3/processed/oracle_gain_attribution.csv")
    het_path = Path("results/week3/processed/per_query_heterogeneity.csv")
    df_attr = pd.read_csv(attr_path)
    df_het = pd.read_csv(het_path)
    
    total_gain_het = df_het["strict_gain"].sum()
    total_gain_attr = df_attr["total_gain"].sum()
    assert pytest.approx(total_gain_het, 1e-4) == total_gain_attr


def test_human_audit_template_integrity():
    template_path = Path("results/week3/human_audit/human_scoring_template.csv")
    assert template_path.exists(), "human_scoring_template.csv missing!"
    df_template = pd.read_csv(template_path)
    
    assert len(df_template) == 150
    assert df_template["question_id"].nunique() == 50
    # Verify k is NOT in columns
    assert "k" not in df_template.columns
    # Verify scores are unpopulated (all NA/empty)
    assert df_template["score_0_1_2"].isna().all() or (df_template["score_0_1_2"] == "").all()
    # Verify anonymous system IDs are System A, B, C
    assert set(df_template["anonymous_system_id"].unique()) == {"System A", "System B", "System C"}


def test_no_learning_collapse_language():
    files_to_check = [
        "results/week3/processed/selected_epsilon.json",
        "results/week3/processed/week3_summary.json",
        "results/week3/processed/week3_gate.json",
        "results/week3/processed/week3_scientific_report.md"
    ]
    for fp in files_to_check:
        path = Path(fp)
        if path.exists():
            content = path.read_text().lower()
            assert "learning collapse" not in content, f"Forbidden phrase 'learning collapse' found in {fp}"
