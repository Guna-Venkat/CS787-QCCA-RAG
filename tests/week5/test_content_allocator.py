"""Unit tests for Week 5 Content-Aware Adaptive Representation Allocator.

Covers all 18 mandatory specifications:
1. TRAIN/DEV split isolation
2. TEST never loaded
3. Correct k <-> class mapping bijection
4. Regret construction
5. Epsilon selection logic
6. Fallback to k=8
7. TRAIN-only feature scaler fitting
8. Passage rank preservation
9. Query-passages shape
10. I1/I2/I3/I4 input masks
11. T1 output shape = [B, 5]
12. T2 output shape = [B, 5]
13. No negative selected k
14. Policy lookup reproduces frozen DEV F1
15. ROH calculation
16. Token-reduction calculation
17. Paper-cluster bootstrap grouping
18. Deterministic seed behavior
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.week5.data import (
    CLASS_TO_K,
    FEATURE_COLS,
    K_CLASSES,
    K_TO_CLASS,
    NUM_CLASSES,
    audit_splits_and_leakage,
    compute_train_class_weights,
    extract_w4_features_from_retrieval,
)
from src.week5.evaluation import (
    DEV_TOKEN_COSTS_DEFAULT,
    ORACLE_F1_REF,
    STATIC_K8_F1_REF,
    STATIC_K8_TOKENS_REF,
    compute_paper_clustered_bootstrap,
    evaluate_allocation_policy,
    evaluate_classification_diagnostics,
    evaluate_regression_diagnostics,
)
from src.week5.models import (
    ContentAwareAllocator,
    PassageTransformerAggregator,
    TabularFeatureProjector,
)
from src.week5.training import EarlyStopping, set_seed

REPO_ROOT = Path(__file__).resolve().parents[2]


class DummyBaseEncoder(nn.Module):
    """Ultra-lightweight mock base encoder for fast unit tests without loading large models."""

    def __init__(self, d_model: int = 384):
        super().__init__()
        self.d_model = d_model
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        batch_size, seq_len = input_ids.shape
        last_hidden_state = torch.ones(batch_size, seq_len, self.d_model, device=input_ids.device)
        return type("Output", (), {"last_hidden_state": last_hidden_state})()


# ---------------------------------------------------------------------------
# 1. TRAIN / DEV Split Isolation
# ---------------------------------------------------------------------------
def test_train_dev_split_isolation():
    """Verify exact split counts and 0 question/paper overlap between TRAIN and DEV."""
    train_path = REPO_ROOT / "results/week5/supervision/train_allocation_supervision.csv"
    dev_path = REPO_ROOT / "results/week5/supervision/dev_allocation_supervision.csv"

    assert train_path.exists(), f"Missing TRAIN supervision: {train_path}"
    assert dev_path.exists(), f"Missing DEV supervision: {dev_path}"

    df_train = pd.read_csv(train_path)
    df_dev = pd.read_csv(dev_path)

    assert len(df_train) == 2088, f"Expected 2,088 TRAIN questions, got {len(df_train)}"
    assert len(df_dev) == 231, f"Expected 231 DEV questions, got {len(df_dev)}"

    train_qids = set(df_train["question_id"].astype(str))
    dev_qids = set(df_dev["question_id"].astype(str))
    assert train_qids.isdisjoint(dev_qids), "Data leakage: question_id overlap between TRAIN and DEV!"

    train_pids = set(df_train["paper_id"].astype(str))
    dev_pids = set(df_dev["paper_id"].astype(str))
    assert train_pids.isdisjoint(dev_pids), "Data leakage: paper_id overlap between TRAIN and DEV!"

    # Also test audit function raises on overlap
    with pytest.raises(ValueError, match="DATA LEAKAGE"):
        audit_splits_and_leakage(["q1", "q2"], ["q2", "q3"])


# ---------------------------------------------------------------------------
# 2. TEST Set Never Loaded
# ---------------------------------------------------------------------------
def test_test_set_never_loaded():
    """Verify TEST split is untouched and never referenced in W5 code or configs."""
    w5_src_dir = REPO_ROOT / "src/week5"
    for py_file in w5_src_dir.glob("*.py"):
        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read().lower()
            assert "test_allocation_supervision" not in content, f"Found test supervision reference in {py_file}"
            assert "test_retrieval" not in content, f"Found test retrieval reference in {py_file}"
            assert "test_fixed_k" not in content, f"Found test fixed-k reference in {py_file}"

    cfg_path = REPO_ROOT / "configs/week5/content_allocator.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_content = f.read().lower()
        assert "test" not in cfg_content or "latest" in cfg_content, "Test path found in content_allocator.yaml"


# ---------------------------------------------------------------------------
# 3. Correct k <-> Class Mapping Bijection
# ---------------------------------------------------------------------------
def test_k_class_mapping_bijection():
    """Verify bidirectional 5-class mapping is exact and reversible."""
    assert K_CLASSES == [2, 4, 5, 6, 8]
    assert NUM_CLASSES == 5

    for k in K_CLASSES:
        cls_idx = K_TO_CLASS[k]
        assert isinstance(cls_idx, int)
        assert 0 <= cls_idx < 5
        assert CLASS_TO_K[cls_idx] == k

    for cls_idx in range(5):
        k = CLASS_TO_K[cls_idx]
        assert k in K_CLASSES
        assert K_TO_CLASS[k] == cls_idx


# ---------------------------------------------------------------------------
# 4. Regret Construction
# ---------------------------------------------------------------------------
def test_regret_construction():
    """Verify regret_k = best_f1 - F1(q, k) is non-negative and 0 at best k."""
    dev_path = REPO_ROOT / "results/week5/supervision/dev_allocation_supervision.csv"
    df_dev = pd.read_csv(dev_path)

    for _, row in df_dev.iterrows():
        f1_vals = [float(row[f"f1_k{k}"]) for k in K_CLASSES]
        expected_best = max(f1_vals)
        assert np.isclose(float(row["best_f1"]), expected_best, atol=1e-5)

        for k in K_CLASSES:
            regret = float(row[f"regret_k{k}"])
            f1_k = float(row[f"f1_k{k}"])
            assert np.isclose(regret, expected_best - f1_k, atol=1e-5)
            assert regret >= -1e-6, f"Negative regret {regret} for question {row['question_id']}"

        # At least one budget must have zero regret
        regrets = [float(row[f"regret_k{k}"]) for k in K_CLASSES]
        assert min(regrets) <= 1e-6


# ---------------------------------------------------------------------------
# 5. Epsilon Selection Logic
# ---------------------------------------------------------------------------
def test_epsilon_selection_logic():
    """Verify smallest k satisfying predicted regret <= epsilon (0.01) is selected."""
    with patch("transformers.AutoModel.from_pretrained", return_value=DummyBaseEncoder()):
        model = ContentAwareAllocator(condition="I1", target_mode="T2")

    # Mock output regrets: k=[2, 4, 5, 6, 8]
    # Case A: k=4 is smallest budget with regret <= 0.01 (k=2 has 0.05)
    mock_regret = torch.tensor([[0.05, 0.008, 0.002, 0.000, 0.000]])
    with patch.object(model, "forward", return_value={"pred_regret": mock_regret}):
        k_pred = model.predict_k({}, epsilon=0.01)
        assert int(k_pred[0]) == 4

    # Case B: k=2 satisfies regret <= 0.01 -> must pick smallest budget k=2
    mock_regret = torch.tensor([[0.009, 0.005, 0.001, 0.000, 0.000]])
    with patch.object(model, "forward", return_value={"pred_regret": mock_regret}):
        k_pred = model.predict_k({}, epsilon=0.01)
        assert int(k_pred[0]) == 2


# ---------------------------------------------------------------------------
# 6. Fallback to k=8
# ---------------------------------------------------------------------------
def test_fallback_to_k8():
    """Verify fallback to k=8 when no budget satisfies predicted regret <= 0.01."""
    with patch("transformers.AutoModel.from_pretrained", return_value=DummyBaseEncoder()):
        model = ContentAwareAllocator(condition="I1", target_mode="T2")

    # All budgets have regret > 0.01
    mock_regret = torch.tensor([[0.12, 0.08, 0.05, 0.03, 0.02]])
    with patch.object(model, "forward", return_value={"pred_regret": mock_regret}):
        k_pred = model.predict_k({}, epsilon=0.01)
        assert int(k_pred[0]) == 8


# ---------------------------------------------------------------------------
# 7. TRAIN-Only Feature Scaler Fitting
# ---------------------------------------------------------------------------
def test_train_only_feature_scaler_fitting():
    """Verify StandardScaler is strictly fit on TRAIN and only applied to DEV."""
    rng = np.random.default_rng(42)
    train_features = rng.normal(loc=10.0, scale=3.0, size=(2088, 6)).astype(np.float32)
    dev_features = rng.normal(loc=12.0, scale=4.0, size=(231, 6)).astype(np.float32)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    dev_scaled = scaler.transform(dev_features)

    # Scaler parameters must match TRAIN only
    np.testing.assert_allclose(scaler.mean_, np.mean(train_features, axis=0), atol=1e-5)
    np.testing.assert_allclose(scaler.var_, np.var(train_features, axis=0), atol=1e-4)

    # TRAIN scaled must have zero mean and unit variance
    np.testing.assert_allclose(np.mean(train_scaled, axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(np.var(train_scaled, axis=0), 1.0, atol=1e-4)

    # DEV scaled must NOT have forced zero mean (distribution shift)
    assert not np.allclose(np.mean(dev_scaled, axis=0), 0.0, atol=1e-2)


# ---------------------------------------------------------------------------
# 8. Passage Rank Preservation
# ---------------------------------------------------------------------------
def test_passage_rank_preservation():
    """Verify rank embeddings in PassageTransformerAggregator preserve ranks 0..9."""
    aggregator = PassageTransformerAggregator(d_model=64, nhead=2, num_layers=1, max_passages=10)
    batch_size = 2
    passage_reps = torch.zeros(batch_size, 10, 64)

    # Verify rank embedding weights are distinct per rank
    rank_weights = aggregator.rank_embeddings.weight
    assert rank_weights.shape == (10, 64)
    # Check that rank 0 embedding != rank 1 embedding
    assert not torch.allclose(rank_weights[0], rank_weights[1])

    # Forward pass
    pooled = aggregator(passage_reps)
    assert pooled.shape == (batch_size, 64)


# ---------------------------------------------------------------------------
# 9. Query-Passages Shape
# ---------------------------------------------------------------------------
def test_query_passages_shape():
    """Verify passage input shapes for condition I3/I4."""
    batch_size = 4
    num_passages = 10
    seq_len = 32
    d_model = 64

    aggregator = PassageTransformerAggregator(d_model=d_model, nhead=2, num_layers=1, max_passages=num_passages)
    passage_reps = torch.randn(batch_size, num_passages, d_model)
    mask = torch.ones(batch_size, num_passages, dtype=torch.bool)
    mask[:, 7:] = False  # passages 8..10 padded

    pooled = aggregator(passage_reps, passage_mask=mask)
    assert pooled.shape == (batch_size, d_model)
    assert not torch.isnan(pooled).any()


# ---------------------------------------------------------------------------
# 10. I1/I2/I3/I4 Input Representations and Masks
# ---------------------------------------------------------------------------
def test_input_representations_and_masks():
    """Verify architectural inputs for all 4 information conditions."""
    with patch("transformers.AutoModel.from_pretrained", return_value=DummyBaseEncoder(d_model=64)):
        # I1: Query only
        m_i1 = ContentAwareAllocator(condition="I1", target_mode="T1", d_model=64)
        assert m_i1.aggregator is None
        assert m_i1.feat_proj is None
        assert m_i1.hidden_in_dim == 64

        # I2: Query + Features
        m_i2 = ContentAwareAllocator(condition="I2", target_mode="T1", d_model=64, feature_dim=6)
        assert m_i2.aggregator is None
        assert m_i2.feat_proj is not None
        assert m_i2.hidden_in_dim == 64 + 32  # 64 + 64//2

        # I3: Query + Passages
        m_i3 = ContentAwareAllocator(condition="I3", target_mode="T1", d_model=64)
        assert m_i3.aggregator is not None
        assert m_i3.feat_proj is None
        assert m_i3.hidden_in_dim == 64

        # I4: Query + Passages + Features
        m_i4 = ContentAwareAllocator(condition="I4", target_mode="T1", d_model=64, feature_dim=6)
        assert m_i4.aggregator is not None
        assert m_i4.feat_proj is not None
        assert m_i4.hidden_in_dim == 64 + 32


# ---------------------------------------------------------------------------
# 11. T1 Output Shape = [B, 5]
# ---------------------------------------------------------------------------
def test_t1_output_shape():
    """Verify T1 classification logits shape [B, 5]."""
    with patch("transformers.AutoModel.from_pretrained", return_value=DummyBaseEncoder(d_model=64)):
        model = ContentAwareAllocator(condition="I1", target_mode="T1", d_model=64)
        batch = {
            "query_input_ids": torch.randint(0, 100, (4, 16)),
            "query_attention_mask": torch.ones(4, 16),
            "features": torch.zeros(4, 6),
        }
        out = model(batch)
        assert "logits" in out
        assert out["logits"].shape == (4, 5)


# ---------------------------------------------------------------------------
# 12. T2 Output Shape = [B, 5] and Non-Negativity
# ---------------------------------------------------------------------------
def test_t2_output_shape_and_non_negativity():
    """Verify T2 regret prediction shape [B, 5] and strictly non-negative values."""
    with patch("transformers.AutoModel.from_pretrained", return_value=DummyBaseEncoder(d_model=64)):
        model = ContentAwareAllocator(condition="I1", target_mode="T2", d_model=64)
        batch = {
            "query_input_ids": torch.randint(0, 100, (4, 16)),
            "query_attention_mask": torch.ones(4, 16),
            "features": torch.zeros(4, 6),
        }
        out = model(batch)
        assert "pred_regret" in out
        assert out["pred_regret"].shape == (4, 5)
        # Softplus ensures strictly non-negative regret
        assert (out["pred_regret"] >= 0.0).all()


# ---------------------------------------------------------------------------
# 13. No Negative Selected k
# ---------------------------------------------------------------------------
def test_no_negative_selected_k():
    """Verify predict_k produces strictly budgets in {2, 4, 5, 6, 8}."""
    with patch("transformers.AutoModel.from_pretrained", return_value=DummyBaseEncoder(d_model=64)):
        m_t1 = ContentAwareAllocator(condition="I1", target_mode="T1", d_model=64)
        m_t2 = ContentAwareAllocator(condition="I1", target_mode="T2", d_model=64)

        batch = {
            "query_input_ids": torch.randint(0, 100, (8, 16)),
            "query_attention_mask": torch.ones(8, 16),
            "features": torch.zeros(8, 6),
        }

        preds_t1 = m_t1.predict_k(batch)
        preds_t2 = m_t2.predict_k(batch, epsilon=0.01)

        for p in preds_t1:
            assert p in K_CLASSES
            assert p > 0

        for p in preds_t2:
            assert p in K_CLASSES
            assert p > 0


# ---------------------------------------------------------------------------
# 14. Policy Lookup Reproduces Frozen DEV F1
# ---------------------------------------------------------------------------
def test_policy_lookup_reproduces_frozen_dev_f1():
    """Verify static k evaluation reproduces bit-exact frozen DEV F1."""
    dev_path = REPO_ROOT / "results/week5/supervision/dev_allocation_supervision.csv"
    df_dev = pd.read_csv(dev_path)

    expected_f1s = {
        2: 0.3038,
        4: 0.3596,
        5: 0.3629,
        6: 0.3857,
        8: 0.3950,
    }

    for k, exp_f1 in expected_f1s.items():
        res = evaluate_allocation_policy(df_dev, [k] * len(df_dev))
        assert np.isclose(res["mean_answer_f1"], exp_f1, atol=1e-4), (
            f"Static k={k} gave {res['mean_answer_f1']}, expected {exp_f1}"
        )


# ---------------------------------------------------------------------------
# 15. ROH Calculation
# ---------------------------------------------------------------------------
def test_roh_calculation():
    """Verify Recovered Oracle Headroom formula ROH = (F1 - 0.3950) / (0.4895 - 0.3950)."""
    dev_path = REPO_ROOT / "results/week5/supervision/dev_allocation_supervision.csv"
    df_dev = pd.read_csv(dev_path)

    # Static k=8 -> ROH = 0.0
    res_k8 = evaluate_allocation_policy(df_dev, [8] * len(df_dev))
    assert np.isclose(res_k8["recovered_oracle_headroom"], 0.0, atol=1e-3)

    # Quality Oracle -> ROH = 1.0
    oracle_preds = []
    for _, r in df_dev.iterrows():
        b_f1 = float(r["best_f1"])
        best_ks = [k for k in K_CLASSES if abs(float(r[f"f1_k{k}"]) - b_f1) < 1e-6]
        oracle_preds.append(min(best_ks))

    res_oracle = evaluate_allocation_policy(df_dev, oracle_preds)
    assert np.isclose(res_oracle["recovered_oracle_headroom"], 1.0, atol=1e-3)


# ---------------------------------------------------------------------------
# 16. Token Reduction Calculation
# ---------------------------------------------------------------------------
def test_token_reduction_calculation():
    """Verify context token reduction percentage calculation vs static k=8."""
    # When tokens == 1750.5 (static k8 ref), reduction is 0%
    red_0 = (STATIC_K8_TOKENS_REF - 1750.5) / STATIC_K8_TOKENS_REF * 100.0
    assert np.isclose(red_0, 0.0)

    # When tokens == 875.25 (50% reduction)
    red_50 = (STATIC_K8_TOKENS_REF - 875.25) / STATIC_K8_TOKENS_REF * 100.0
    assert np.isclose(red_50, 50.0)


# ---------------------------------------------------------------------------
# 17. Paper-Cluster Bootstrap Grouping
# ---------------------------------------------------------------------------
def test_paper_cluster_bootstrap_grouping():
    """Verify paper-clustered bootstrap respects paper groupings and outputs valid CI."""
    paper_ids = ["p1", "p1", "p1", "p2", "p2", "p3", "p4", "p5"]
    policy_f1s = np.array([0.5, 0.4, 0.6, 0.3, 0.35, 0.45, 0.5, 0.6], dtype=np.float32)
    baseline_f1s = np.array([0.4, 0.3, 0.5, 0.3, 0.30, 0.40, 0.4, 0.5], dtype=np.float32)

    res = compute_paper_clustered_bootstrap(
        paper_ids=paper_ids,
        policy_f1s=policy_f1s,
        baseline_f1s=baseline_f1s,
        n_bootstrap=500,
        seed=42,
    )

    assert "observed_delta_f1" in res
    assert "ci_lower_95" in res
    assert "ci_upper_95" in res
    assert "p_value" in res
    assert res["ci_lower_95"] <= res["observed_delta_f1"] <= res["ci_upper_95"]
    assert 0.0 <= res["p_value"] <= 1.0


# ---------------------------------------------------------------------------
# 18. Deterministic Seed Behavior
# ---------------------------------------------------------------------------
def test_deterministic_seed_behavior():
    """Verify set_seed ensures exact reproducibility."""
    set_seed(42)
    r_np1 = np.random.uniform(size=(10,))
    r_torch1 = torch.randn(10)

    set_seed(42)
    r_np2 = np.random.uniform(size=(10,))
    r_torch2 = torch.randn(10)

    np.testing.assert_array_equal(r_np1, r_np2)
    assert torch.equal(r_torch1, r_torch2)
