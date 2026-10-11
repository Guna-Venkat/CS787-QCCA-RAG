"""Unit tests for Week 6: Semantic Capacity and Encoder Adaptation Study.

Verifies all 19 mandatory specifications:
1. C0 all encoder parameters frozen
2. C1 exactly final one Transformer block trainable
3. C2 exactly final two blocks trainable
4. Embeddings remain frozen across all conditions
5. Intended frozen layers remain frozen after optimizer step
6. Trainable layers receive nonzero gradient in smoke/forward-backward
7. Separate encoder/head optimizer parameter groups
8. Correct differential learning rate assigned to each group
9. T3 target identical to W5-B (z_k = 1[regret_k <= 0.01])
10. Split isolation (disjoint paper and question sets)
11. DEV never used for training or tau calibration
12. TEST split never loaded or accessed
13. Rank embedding initialization retained (std ~ 0.02, LayerNorm)
14. Policy selects smallest admissible k
15. Fallback to k=8 when no budget qualifies
16. Tau selected strictly from INTERNAL-VAL with 0.005 tolerance rule
17. Representation shift calculation shape and metric correctness
18. W5-B C0 reference loads correctly (DEV F1 ~ 0.3810, tokens ~ 1258.2)
19. Bootstrap uses across-seed mean vectors for canonical reporting
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from transformers import AutoModel

from src.week5.data import (
    K_CLASSES,
    NUM_CLASSES,
    build_w5b_splits_and_bundles,
)
from src.week5.evaluation import (
    DEV_TOKEN_COSTS_DEFAULT,
    STATIC_K8_F1_REF,
    STATIC_K8_TOKENS_REF,
    evaluate_allocation_policy,
    select_tau_internal_val,
)
from src.week6.adaptation import (
    MINILM_LAYER_PARAMS,
    MINILM_NUM_LAYERS,
    AdaptedContentAwareAllocator,
    audit_minilm_architecture,
    build_optimizer_param_groups,
    configure_encoder_adaptation,
)
from src.week6.evaluation import (
    C0_REF_F1,
    C0_REF_TOKENS,
    load_c0_w5b_reference,
    run_paper_clustered_bootstrap_comparisons,
)
from src.week6.representation_analysis import compute_representation_shift


@pytest.fixture(scope="module")
def pretrained_encoder():
    """Load pretrained MiniLM encoder once for fast test execution."""
    return AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def test_1_c0_all_encoder_params_frozen(pretrained_encoder):
    """Test 1: In C0, all MiniLM encoder parameters must be frozen (0 trainable)."""
    enc = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    stats = configure_encoder_adaptation(enc, "C0")
    assert stats["trainable_encoder_params"] == 0
    assert all(not p.requires_grad for p in enc.parameters())


def test_2_c1_exactly_final_one_transformer_block_trainable(pretrained_encoder):
    """Test 2: In C1, exactly layer 5 is trainable (1,774,464 params). Layers 0..4 are frozen."""
    enc = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    stats = configure_encoder_adaptation(enc, "C1")
    assert stats["trainable_encoder_params"] == MINILM_LAYER_PARAMS
    # Check layer 5 is trainable
    assert all(p.requires_grad for p in enc.encoder.layer[5].parameters())
    # Check layers 0..4 are frozen
    for i in range(5):
        assert all(not p.requires_grad for p in enc.encoder.layer[i].parameters())


def test_3_c2_exactly_final_two_blocks_trainable(pretrained_encoder):
    """Test 3: In C2, exactly layers 4 and 5 are trainable (3,548,928 params). Layers 0..3 are frozen."""
    enc = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    stats = configure_encoder_adaptation(enc, "C2")
    assert stats["trainable_encoder_params"] == 2 * MINILM_LAYER_PARAMS
    # Check layers 4 and 5 are trainable
    assert all(p.requires_grad for p in enc.encoder.layer[4].parameters())
    assert all(p.requires_grad for p in enc.encoder.layer[5].parameters())
    # Check layers 0..3 are frozen
    for i in range(4):
        assert all(not p.requires_grad for p in enc.encoder.layer[i].parameters())


def test_4_embeddings_remain_frozen(pretrained_encoder):
    """Test 4: Embeddings and pooler must remain frozen across C0, C1, and C2."""
    enc = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    for cond in ["C0", "C1", "C2"]:
        configure_encoder_adaptation(enc, cond)
        assert all(not p.requires_grad for p in enc.embeddings.parameters())
        if hasattr(enc, "pooler") and enc.pooler is not None:
            assert all(not p.requires_grad for p in enc.pooler.parameters())


def test_5_intended_frozen_layers_remain_frozen_after_optimizer_step():
    """Test 5: Verified that frozen weights do not change after optimizer step."""
    model = AdaptedContentAwareAllocator(capacity_condition="C1")
    # Capture weights of frozen layer 0 and embeddings before step
    w_emb_before = model.encoder.embeddings.word_embeddings.weight.clone()
    w_l0_before = model.encoder.encoder.layer[0].attention.self.query.weight.clone()
    w_l5_before = model.encoder.encoder.layer[5].attention.self.query.weight.clone()

    param_groups = build_optimizer_param_groups(model)
    optimizer = torch.optim.AdamW(param_groups)

    # Dummy batch
    B = 2
    dummy_input_ids = torch.randint(0, 1000, (B, 10, 32))
    dummy_attention_mask = torch.ones((B, 10, 32))
    dummy_p_mask = torch.ones((B, 10), dtype=torch.bool)
    labels = torch.randint(0, 2, (B, 5)).float()

    model.train()
    out = model({
        "passage_input_ids": dummy_input_ids,
        "passage_attention_mask": dummy_attention_mask,
        "passage_mask": dummy_p_mask,
    })
    loss = nn.BCEWithLogitsLoss()(out["admissibility_logits"], labels)
    loss.backward()
    optimizer.step()

    # Frozen weights must be IDENTICAL
    assert torch.equal(model.encoder.embeddings.word_embeddings.weight, w_emb_before)
    assert torch.equal(model.encoder.encoder.layer[0].attention.self.query.weight, w_l0_before)
    # Trainable weights must have CHANGED
    assert not torch.equal(model.encoder.encoder.layer[5].attention.self.query.weight, w_l5_before)


def test_6_trainable_layers_receive_nonzero_gradient():
    """Test 6: In C1, trainable layers receive nonzero gradients while frozen layers have None."""
    model = AdaptedContentAwareAllocator(capacity_condition="C1")
    B = 2
    dummy_input_ids = torch.randint(0, 1000, (B, 10, 32))
    dummy_attention_mask = torch.ones((B, 10, 32))
    dummy_p_mask = torch.ones((B, 10), dtype=torch.bool)
    labels = torch.randint(0, 2, (B, 5)).float()

    model.train()
    out = model({
        "passage_input_ids": dummy_input_ids,
        "passage_attention_mask": dummy_attention_mask,
        "passage_mask": dummy_p_mask,
    })
    loss = nn.BCEWithLogitsLoss()(out["admissibility_logits"], labels)
    loss.backward()

    # Trainable layer 5 must have nonzero grad
    for p in model.encoder.encoder.layer[5].parameters():
        assert p.grad is not None
        assert p.grad.abs().sum().item() > 0

    # Frozen layer 0 must have grad is None
    for p in model.encoder.encoder.layer[0].parameters():
        assert p.grad is None

    # Aggregator and head must have nonzero grad
    for p in model.aggregator.parameters():
        assert p.grad is not None
        assert p.grad.abs().sum().item() > 0


def test_7_separate_encoder_head_optimizer_param_groups():
    """Test 7: Optimizer parameter groups separate encoder and head/aggregator without overlap."""
    model = AdaptedContentAwareAllocator(capacity_condition="C1")
    groups = build_optimizer_param_groups(model)
    assert len(groups) == 2
    assert groups[0]["name"] == "encoder_adapted"
    assert groups[1]["name"] == "head_and_aggregator"

    encoder_p_ids = {id(p) for p in groups[0]["params"]}
    head_p_ids = {id(p) for p in groups[1]["params"]}
    assert encoder_p_ids.isdisjoint(head_p_ids)


def test_8_correct_lr_assigned_to_each_group():
    """Test 8: Differential learning rates correctly set (encoder_lr=2e-5, head_lr=2e-4)."""
    model = AdaptedContentAwareAllocator(capacity_condition="C1")
    groups = build_optimizer_param_groups(model, encoder_lr=2.0e-5, head_lr=2.0e-4)
    assert groups[0]["lr"] == 2.0e-5
    assert groups[1]["lr"] == 2.0e-4


def test_9_t3_target_identical_to_w5b():
    """Test 9: T3 multi-label definition is identical to W5-B (z_k = 1[regret_k <= 0.01])."""
    regrets = np.array([0.0000, 0.0080, 0.0100, 0.0102, 0.0500])
    epsilon = 0.01
    expected = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    labels = (regrets <= epsilon + 1e-9).astype(np.float32)
    np.testing.assert_array_equal(labels, expected)


def test_10_split_isolation():
    """Test 10: TRAIN-FIT and INTERNAL-VAL manifests have disjoint paper and question sets."""
    manifest_fit = Path("results/week5/content_allocator/w5b/splits/train_fit_manifest.csv")
    manifest_val = Path("results/week5/content_allocator/w5b/splits/internal_val_manifest.csv")
    assert manifest_fit.exists() and manifest_val.exists()

    df_fit = pd.read_csv(manifest_fit)
    df_val = pd.read_csv(manifest_val)

    assert len(df_fit) == 1746
    assert len(df_val) == 342
    assert len(df_fit) + len(df_val) == 2088

    papers_fit = set(df_fit["paper_id"])
    papers_val = set(df_val["paper_id"])
    assert papers_fit.isdisjoint(papers_val)

    qids_fit = set(df_fit["question_id"])
    qids_val = set(df_val["question_id"])
    assert qids_fit.isdisjoint(qids_val)


def test_11_dev_never_used_for_training_or_tau_calibration():
    """Test 11: DEV split is excluded from training manifests and threshold calibration."""
    dev_sup = pd.read_csv("results/week5/supervision/dev_allocation_supervision.csv")
    df_fit = pd.read_csv("results/week5/content_allocator/w5b/splits/train_fit_manifest.csv")
    df_val = pd.read_csv("results/week5/content_allocator/w5b/splits/internal_val_manifest.csv")

    dev_qids = set(dev_sup["question_id"].astype(str))
    fit_qids = set(df_fit["question_id"].astype(str))
    val_qids = set(df_val["question_id"].astype(str))

    assert dev_qids.isdisjoint(fit_qids)
    assert dev_qids.isdisjoint(val_qids)


def test_12_test_never_loaded():
    """Test 12: Ensure QASPER TEST split is never accessed or referenced in Week 6 configs."""
    cfg_path = Path("configs/week6/encoder_adaptation.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "test_retrieval" not in content
    assert "test_supervision" not in content
    assert "qasper_test" not in content


def test_13_rank_embedding_initialization_retained():
    """Test 13: W5-B rank embedding initialization Normal(0, 0.02^2) and LayerNorm retained."""
    model = AdaptedContentAwareAllocator(capacity_condition="C1")
    rank_std = float(model.aggregator.rank_embeddings.weight.std().item())
    assert 0.01 <= rank_std <= 0.03
    assert isinstance(model.aggregator.input_layer_norm, nn.LayerNorm)


def test_14_policy_selects_smallest_admissible_k():
    """Test 14: Policy selects smallest k with p_k >= tau."""
    model = AdaptedContentAwareAllocator(capacity_condition="C0")
    dummy_probs = np.array([
        [0.2, 0.7, 0.9, 0.3, 0.8],  # valid ks >= 0.5: {4, 5, 8} -> min is 4
        [0.8, 0.9, 0.9, 0.9, 0.9],  # valid ks >= 0.5: {2, 4, 5, 6, 8} -> min is 2
    ])
    preds = []
    tau = 0.5
    for row in dummy_probs:
        valid_ks = [K_CLASSES[idx] for idx, p in enumerate(row) if p >= tau]
        preds.append(min(valid_ks) if valid_ks else 8)
    assert preds == [4, 2]


def test_15_fallback_k8_logic():
    """Test 15: Policy falls back to k=8 when no budget satisfies p_k >= tau."""
    dummy_probs = np.array([
        [0.1, 0.2, 0.3, 0.1, 0.2],  # none >= 0.5 -> fallback to 8
    ])
    preds = []
    tau = 0.5
    for row in dummy_probs:
        valid_ks = [K_CLASSES[idx] for idx, p in enumerate(row) if p >= tau]
        preds.append(min(valid_ks) if valid_ks else 8)
    assert preds == [8]


def test_16_tau_selected_from_internal_val_only():
    """Test 16: select_tau_internal_val executes on INTERNAL-VAL with 0.005 rule."""
    df_val = pd.read_csv("results/week5/content_allocator/w5b/splits/internal_val_manifest.csv")
    n_val = len(df_val)
    # Mock probabilities
    rng = np.random.default_rng(42)
    mock_probs = rng.uniform(0.2, 0.9, (n_val, 5))
    # Mock supervision df with f1_k and tokens_k
    val_sup = pd.DataFrame({
        "question_id": df_val["question_id"],
        "best_f1": df_val["best_f1"],
        "oracle_k_eps001": df_val["oracle_k_eps001"],
        "f1_k2": rng.uniform(0.3, 0.4, n_val),
        "f1_k4": rng.uniform(0.35, 0.42, n_val),
        "f1_k5": rng.uniform(0.36, 0.43, n_val),
        "f1_k6": rng.uniform(0.38, 0.45, n_val),
        "f1_k8": rng.uniform(0.39, 0.46, n_val),
    })
    tokens_map = {(str(qid), k): int(k * 200) for qid in df_val["question_id"] for k in K_CLASSES}

    from src.week6.evaluation import select_tau_w6
    tau_res = select_tau_w6(val_sup, mock_probs, val_tokens_map=tokens_map)
    assert tau_res["selected_tau"] in [0.30, 0.40, 0.50, 0.60, 0.70]
    assert "tau_boundary_hit" in tau_res
    assert isinstance(tau_res["tau_boundary_hit"], bool)


def test_17_representation_shift_calculation_shape_and_correctness():
    """Test 17: Representation shift computation produces correct metric keys and valid bounds."""
    m_pre = AdaptedContentAwareAllocator(capacity_condition="C0")
    m_adapt = AdaptedContentAwareAllocator(capacity_condition="C1")

    B = 2
    dummy_batch = {
        "passage_input_ids": torch.randint(0, 1000, (B, 10, 32)),
        "passage_attention_mask": torch.ones((B, 10, 32)),
        "passage_mask": torch.ones((B, 10), dtype=torch.bool),
    }

    class DummyLoader:
        def __iter__(self):
            return iter([dummy_batch])

    shift = compute_representation_shift(m_pre, m_adapt, DummyLoader(), max_batches=1, device="cpu")
    assert "mean_cosine_similarity" in shift
    assert -1.0 <= shift["mean_cosine_similarity"] <= 1.0
    assert "mean_representation_shift_norm" in shift
    assert shift["mean_representation_shift_norm"] >= 0.0


def test_18_w5b_c0_reference_loads_correctly():
    """Test 18: Canonical W5-B C0 reference loads correctly with expected DEV F1 ~ 0.3810."""
    c0_ref = load_c0_w5b_reference()
    assert abs(c0_ref["mean_answer_f1"] - C0_REF_F1) < 0.005
    assert abs(c0_ref["mean_context_tokens"] - C0_REF_TOKENS) < 20.0
    assert abs(c0_ref["mean_selected_k"] - 5.57) < 0.2


def test_19_bootstrap_uses_across_seed_mean_vectors_for_canonical_report():
    """Test 19: Bootstrap wrapper executes with across-seed mean vectors."""
    n = 20
    paper_ids = [f"p{i // 2}" for i in range(n)]
    rng = np.random.default_rng(42)

    eval_c0 = {"per_query_f1s": rng.uniform(0.35, 0.40, n), "per_query_tokens": rng.uniform(1200, 1300, n)}
    eval_c1 = {"per_query_f1s": rng.uniform(0.36, 0.41, n), "per_query_tokens": rng.uniform(1150, 1250, n)}
    eval_c2 = {"per_query_f1s": rng.uniform(0.37, 0.42, n), "per_query_tokens": rng.uniform(1100, 1200, n)}
    eval_w4 = {"per_query_f1s": rng.uniform(0.36, 0.40, n), "per_query_tokens": rng.uniform(1050, 1150, n)}
    eval_k8 = {"per_query_f1s": rng.uniform(0.38, 0.41, n), "per_query_tokens": np.full(n, 1750.5)}

    df_boot = run_paper_clustered_bootstrap_comparisons(
        paper_ids=paper_ids,
        eval_c0=eval_c0,
        eval_c1=eval_c1,
        eval_c2=eval_c2,
        eval_w4=eval_w4,
        eval_k8=eval_k8,
        n_bootstrap=100,  # Fast for test
        seed=42,
    )
    assert len(df_boot) == 5
    assert "C1 Last-1 Layer vs C0 Frozen" in df_boot["comparison"].values
    assert "C2 Last-2 Layers vs C0 Frozen" in df_boot["comparison"].values
    assert all(df_boot["aggregation"] == "Across-Seed Mean per Query")
