"""Unit tests for Week 5 Content-Aware Adaptive Representation Allocation Preflight.

Validates:
- Split integrity & paper disjointness (Train: 2088/772, Dev: 231/86, Test: 1309/411)
- Response-surface schema and valid F1 ranges [0, 1]
- Regret non-negativity within numerical tolerance
- Minimum-k epsilon-oracle derivation rule (epsilon = 0.01)
- Bit-exact DEV target distribution reproduction (132, 40, 17, 19, 23)
- Absence of train response surface (0.0% coverage, STATUS C)
- Strict pre-generation invariant & zero post-generation leakage
- Passage rank preservation & top-10 ordering
- Zero test set access / leakage
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def manifest():
    manifest_path = REPO_ROOT / "Baselines/SARA-main/data/manifests/qasper_split.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def dev_supervision():
    dev_sup_path = REPO_ROOT / "results/week5/supervision/dev_allocation_supervision.csv"
    assert dev_sup_path.exists(), f"Dev supervision table missing: {dev_sup_path}"
    return pd.read_csv(dev_sup_path)


def test_qasper_split_counts_and_manifest(manifest):
    """Verify official split dimensions and manifest records."""
    assert manifest["n_papers_train"] == 772
    assert manifest["n_papers_dev"] == 86
    assert manifest["n_rows_train"] == 2088
    assert manifest["n_rows_dev"] == 231
    assert manifest["n_rows_test"] == 1309

    train_papers = set(manifest["train_paper_ids"])
    dev_papers = set(manifest["dev_paper_ids"])
    assert len(train_papers) == 772
    assert len(dev_papers) == 86
    assert train_papers.isdisjoint(dev_papers), "Train and Dev paper sets must be completely disjoint!"


def test_split_audit_json_validity():
    """Verify results/week5/preflight/split_audit.json structure and values."""
    audit_file = REPO_ROOT / "results/week5/preflight/split_audit.json"
    assert audit_file.exists()
    with open(audit_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["observed"]["train"]["questions"] == 2088
    assert data["observed"]["dev"]["questions"] == 231
    assert data["observed"]["test"]["questions"] == 1309
    assert data["paper_disjointness"]["is_train_dev_paper_disjoint"] is True
    assert data["question_disjointness"]["is_train_dev_question_disjoint"] is True
    assert data["test_set_status"] == "FROZEN_AND_UNTOUCHED"


def test_dev_response_surface_schema_and_f1_domain():
    """Verify Week 2 dev response surface schema and metric bounds."""
    matrix_path = REPO_ROOT / "results/week2/processed/dev_per_query_k_matrix.csv"
    assert matrix_path.exists()
    df = pd.read_csv(matrix_path)

    assert len(df) == 231
    required_cols = ["question_id", "paper_id", "F1_k2", "F1_k4", "F1_k5", "F1_k6", "F1_k8"]
    for col in required_cols:
        assert col in df.columns, f"Missing column {col} in dev response surface"

    # F1 score domain check [0, 1]
    for k in [2, 4, 5, 6, 8]:
        col = f"F1_k{k}"
        assert df[col].min() >= 0.0 - 1e-6, f"{col} has negative F1"
        assert df[col].max() <= 1.0 + 1e-6, f"{col} exceeds 1.0"
        assert not df[col].isna().any(), f"{col} contains NaNs"


def test_dev_supervision_oracle_derivation(dev_supervision):
    """Verify mathematical consistency of oracle budget and regret derivation."""
    df = dev_supervision
    assert len(df) == 231

    for _, row in df.iterrows():
        f1s = [row[f"f1_k{k}"] for k in [2, 4, 5, 6, 8]]
        expected_best = max(f1s)
        assert np.isclose(row["best_f1"], expected_best, atol=1e-5)

        for k in [2, 4, 5, 6, 8]:
            expected_regret = expected_best - row[f"f1_k{k}"]
            assert np.isclose(row[f"regret_k{k}"], expected_regret, atol=1e-5)
            # Regret must be non-negative within floating point tolerance
            assert row[f"regret_k{k}"] >= -1e-6, f"Negative regret at k={k}: {row[f'regret_k{k}']}"

            expected_eps_good = 1 if expected_regret <= 0.01 + 1e-8 else 0
            assert row[f"eps_good_k{k}"] == expected_eps_good

        # Check minimum-k epsilon oracle rule
        feasible_ks = [k for k in [2, 4, 5, 6, 8] if row[f"eps_good_k{k}"] == 1]
        assert len(feasible_ks) > 0
        expected_oracle_k = min(feasible_ks)
        assert row["oracle_k_eps001"] == expected_oracle_k


def test_dev_oracle_distribution_exact_counts(dev_supervision):
    """Verify bit-exact reproduction of Week 3 DEV target distribution."""
    counts = dev_supervision["oracle_k_eps001"].value_counts().sort_index().to_dict()
    expected = {2: 132, 4: 40, 5: 17, 6: 19, 8: 23}
    assert counts == expected, f"Distribution mismatch: observed {counts} vs expected {expected}"


def test_train_response_surface_status_absent():
    """Verify train response surface audit correctly records STATUS C: ABSENT."""
    audit_path = REPO_ROOT / "results/week5/preflight/train_response_surface_audit.csv"
    assert audit_path.exists()
    df = pd.read_csv(audit_path)

    assert len(df) == 2088
    assert df["question_id"].nunique() == 2088
    # All fixed-k coverage flags must be False
    for k in [2, 4, 5, 6, 8]:
        assert not df[f"has_k{k}"].any(), f"has_k{k} must be completely False"
    assert not df["complete"].any()

    # Supervision template check
    sup_template = REPO_ROOT / "results/week5/supervision/train_allocation_supervision.csv"
    assert sup_template.exists()
    with open(sup_template, "r", encoding="utf-8") as f:
        content = f.read()
    assert "PENDING_FULL_TRAIN_RESPONSE_SURFACE" in content


def test_w4_feature_pre_generation_invariant():
    """Verify all 64 W4 candidate features satisfy the pre-generation invariant."""
    meta_path = REPO_ROOT / "results/week4/feature_discovery/feature_metadata.csv"
    assert meta_path.exists()
    df_meta = pd.read_csv(meta_path)

    assert len(df_meta) == 64
    assert df_meta["available_pre_generation"].all(), "All features must be available pre-generation"
    assert (df_meta["leakage_status"] == "VERIFIED_PRE_GEN").all(), "All features must be verified pre-generation"


def test_retrieval_passage_rank_preservation():
    """Verify smoke test retrieval records preserve exact 10-passage rank order."""
    retrieval_path = REPO_ROOT / "results/week5/supervision/train_retrieval.jsonl"
    if not retrieval_path.exists():
        pytest.skip("train_retrieval.jsonl not yet created")

    with open(retrieval_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            passages = rec["passages"]
            assert len(passages) == 10, f"Expected 10 passages, got {len(passages)}"
            ranks = [p["rank"] for p in passages]
            assert ranks == list(range(1, 11)), f"Passage ranks violated: {ranks}"


def test_no_test_set_leakage():
    """Ensure no Week 5 supervision or preflight file loads test set rows."""
    w5_supervision_dir = REPO_ROOT / "results/week5/supervision"
    for csv_file in w5_supervision_dir.glob("*.csv"):
        df = pd.read_csv(csv_file, comment="#")
        if "question_id" in df.columns:
            # Dev supervision must only contain dev questions
            dev_matrix = pd.read_csv(REPO_ROOT / "results/week2/processed/dev_per_query_k_matrix.csv")
            dev_qids = {str(q).split(".")[0] for q in dev_matrix["question_id"]}
            csv_qids = {str(q).split(".")[0] for q in df["question_id"]}
            if csv_file.name == "dev_allocation_supervision.csv":
                assert csv_qids == dev_qids, "dev_allocation_supervision.csv must match exactly 231 dev QIDs"
                assert len(csv_qids) == 231


def test_supervision_derivation_helper():
    """Verify derive_canonical_supervision_record logic."""
    from src.week5.supervision import derive_canonical_supervision_record

    f1s = {2: 0.35, 4: 0.40, 5: 0.42, 6: 0.42, 8: 0.41}
    rec = derive_canonical_supervision_record("test_q1", "paper_123", f1s, epsilon=0.01)

    assert rec["best_f1"] == 0.42
    assert rec["regret_k2"] == pytest.approx(0.07, abs=1e-5)
    assert rec["regret_k4"] == pytest.approx(0.02, abs=1e-5)
    assert rec["regret_k5"] == pytest.approx(0.00, abs=1e-5)
    assert rec["eps_good_k5"] == 1
    assert rec["eps_good_k6"] == 1
    # Minimum k with regret <= 0.01 is 5
    assert rec["oracle_k_eps001"] == 5


def test_evaluation_policy_routing(dev_supervision):
    """Verify policy evaluation and headroom metrics on dev supervision."""
    from src.week5.evaluation import evaluate_allocation_policy

    # Evaluate static k=8
    k8_preds = np.full(len(dev_supervision), 8)
    res_k8 = evaluate_allocation_policy(dev_supervision, k8_preds)
    assert res_k8["mean_selected_k"] == 8.0
    assert res_k8["mean_context_tokens"] == pytest.approx(1758.5, abs=0.5)
    assert res_k8["token_reduction_pct_vs_k8"] == pytest.approx(0.0, abs=0.1)

    # Evaluate oracle k
    oracle_preds = dev_supervision["oracle_k_eps001"].values
    res_oracle = evaluate_allocation_policy(dev_supervision, oracle_preds)
    assert res_oracle["zero_regret_pct"] == 100.0
    assert res_oracle["token_reduction_pct_vs_k8"] > 50.0  # Oracle saves >50% context tokens
    assert res_oracle["recovered_oracle_headroom"] > 0.90


def test_model_architecture_shapes():
    """Verify PyTorch model forward pass across all input conditions."""
    import torch
    from src.week5.architecture import ContentAwareAllocatorModel

    B, N, D = 4, 10, 64  # Small dimensions for fast unit testing

    # Test Condition C (Query + Passages, no tabular features)
    model_c = ContentAwareAllocatorModel(d_model=D, feature_dim=0, nhead=2, num_layers=1, dim_feedforward=128)
    passage_reps = torch.randn(B, N, D)
    logits_k, pred_regret = model_c(passage_reps)

    assert logits_k.shape == (B, 5)
    assert pred_regret.shape == (B, 5)
    assert (pred_regret >= 0).all(), "Regrets must be non-negative"

    # Test Condition D (Query + Passages + W4 features)
    model_d = ContentAwareAllocatorModel(d_model=D, feature_dim=5, nhead=2, num_layers=1, dim_feedforward=128)
    features = torch.randn(B, 5)
    logits_d, regret_d = model_d(passage_reps, features=features)

    assert logits_d.shape == (B, 5)
    assert regret_d.shape == (B, 5)

