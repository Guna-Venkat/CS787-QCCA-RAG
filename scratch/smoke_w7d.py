"""Smoke test script for Week 7-D Generator-Aligned Cross-Encoder Fidelity Router.

Strictly follows Section 42 (Smoke Test Policy) & Section 44 (VRAM / Performance Report).
Performs tiny validation only:
- 6 TRAIN queries, 3 VAL queries.
- D0 score computation, feature extraction, tiny GBDT fit.
- D1 forward/backward pass, multi-task loss verification, peak VRAM measurement.
- D2 hardware feasibility check & input prompt verification.
- Saves diagnostic results marked SMOKE_ONLY_NOT_FOR_REPORTING.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from transformers import AutoTokenizer

from src.week7.w7d.cross_encoder import CrossEncoderScorer
from src.week7.w7d.cross_encoder_features import (
    build_d0_feature_dataframe,
    extract_query_cross_encoder_features,
)
from src.week7.w7d.data import (
    ALLOWED_BUDGETS,
    ANCHOR_K,
    assert_test_disabled,
    assert_zero_feature_leakage,
    audit_input_lengths,
    load_retrieval_passages,
    load_w7d_splits,
    truncate_document_deterministic,
)
from src.week7.w7d.hierarchical_model import HierarchicalCrossEncoderRouter
from src.week7.w7d.listwise_llm import format_listwise_causal_prompt
from src.week7.w7d.lora_utils import audit_d2_hardware_feasibility
from src.week7.w7d.multitask_heads import MultiTaskUtilityHeads
from src.week7.w7d.policy import (
    apply_w7d_anchor_deviate_policy,
    apply_w7d_multilabel_policy,
    evaluate_downstream_fidelity_policy,
)
from src.week7.w7d.targets import (
    BUDGETS,
    build_w7d_targets_dataframe,
    compute_sara_multitask_targets,
)
from src.week7.w7d.training import D0TabularRouter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("smoke_w7d")


def run_smoke_test():
    logger.info("Starting Week 7-D Smoke Validation (SMOKE_ONLY_NOT_FOR_REPORTING)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # 1. Load Split Manifests and verify paper-disjoint separation
    df_fit_man, df_val_man = load_w7d_splits()
    logger.info("Manifest loaded: %d TRAIN-FIT, %d INTERNAL-VAL queries.", len(df_fit_man), len(df_val_man))

    # Select smoke query IDs: 6 TRAIN-FIT, 3 INTERNAL-VAL
    smoke_train_qids = list(df_fit_man["question_id"].iloc[:6])
    smoke_val_qids = list(df_val_man["question_id"].iloc[:3])

    # 2. Load Retrieval Passages for smoke queries
    train_retrieval_path = "results/week5/train_response_surface/train_retrieval.jsonl"
    logger.info("Loading retrieval passages from %s...", train_retrieval_path)
    all_train_retrieval = load_retrieval_passages(train_retrieval_path, split_name="train")

    smoke_retrieval_train = {qid: all_train_retrieval[qid] for qid in smoke_train_qids if qid in all_train_retrieval}
    smoke_retrieval_val = {qid: all_train_retrieval[qid] for qid in smoke_val_qids if qid in all_train_retrieval}

    # Verify 10 passages per query
    for qid, qdata in smoke_retrieval_train.items():
        assert len(qdata["passages"]) == 10, f"Query {qid} does not have 10 passages!"
    for qid, qdata in smoke_retrieval_val.items():
        assert len(qdata["passages"]) == 10, f"Query {qid} does not have 10 passages!"

    # 3. Input Length Audit
    stats_dict, df_audit = audit_input_lengths(smoke_retrieval_train, smoke_retrieval_val)
    logger.info("Input Length Audit:\n%s", df_audit.to_string())

    # 4. Load Response Surfaces and compute targets
    df_resp_train = pd.read_csv("results/week7/w7b_semantic_budget/response_surface/train_semantic_fixed_k_matrix.csv")
    df_resp_train["question_id"] = df_resp_train["question_id"].astype(str).str.strip()
    df_targets_all = build_w7d_targets_dataframe(df_resp_train)

    df_smoke_targets_train = df_targets_all[df_targets_all["question_id"].isin(smoke_train_qids)].copy()
    df_smoke_targets_val = df_targets_all[df_targets_all["question_id"].isin(smoke_val_qids)].copy()

    # 5. D0 Smoke: Score caching + feature extraction + tabular GBDT fit
    smoke_dir = PROJECT_ROOT / "results/week7/w7d_cross_encoder_router/diagnostics/smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    smoke_cache_path = smoke_dir / "smoke_cross_encoder_scores.parquet"

    scorer = CrossEncoderScorer(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device=device, batch_size=16)
    df_smoke_scores_train = scorer.compute_and_cache_scores(
        smoke_retrieval_train, cache_path=smoke_cache_path, split_name="train"
    )
    df_smoke_scores_val = scorer.compute_and_cache_scores(
        smoke_retrieval_val, cache_path=smoke_cache_path, split_name="val"
    )

    df_d0_train_feats = build_d0_feature_dataframe(df_smoke_scores_train)
    df_d0_val_feats = build_d0_feature_dataframe(df_smoke_scores_val)

    feat_cols = [c for c in df_d0_train_feats.columns if c.startswith("feat_")]
    assert_zero_feature_leakage(feat_cols)

    # Train tiny D0 GBDT
    d0_router = D0TabularRouter(n_estimators=5, max_depth=2, min_child_samples=1)
    d0_router.fit(df_d0_train_feats[feat_cols], df_smoke_targets_train)
    d0_val_preds = d0_router.predict(df_d0_val_feats[feat_cols])
    logger.info("D0 Smoke Fit complete. Val p_down: %s", d0_val_preds["p_down"])

    # 6. D1 Smoke: Hierarchical Cross-Encoder forward/backward
    tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
    d1_model = HierarchicalCrossEncoderRouter(
        encoder_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        freeze_encoder=True,
        d_model=256,
        n_heads=4,
        n_layers=2,
    )
    d1_model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in d1_model.parameters())
    trainable_params = sum(p.numel() for p in d1_model.parameters() if p.requires_grad)
    logger.info("D1 Params: Total=%d, Trainable=%d (%.2f%% trainable)",
                total_params, trainable_params, (trainable_params / total_params) * 100.0)

    # Prepare batch of 2 queries
    batch_qids = smoke_train_qids[:2]
    batch_pairs = []
    batch_meta = []

    for qid in batch_qids:
        q_item = smoke_retrieval_train[qid]
        q_text = q_item["question"]
        for p in q_item["passages"]:
            batch_pairs.append((q_text, p["text"]))
            batch_meta.append([
                float(p["semantic_rank"]),
                float(p["bm25_rank"]),
                float(p["semantic_rank"]) ** -1,
                float(p["bm25_score"]),
                float(len(p["text"].split())),
            ])

    enc = tokenizer(
        [p[0] for p in batch_pairs],
        [p[1] for p in batch_pairs],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    B = len(batch_qids)
    input_ids = enc["input_ids"].view(B, 10, -1).to(device)
    attention_mask = enc["attention_mask"].view(B, 10, -1).to(device)
    meta_tensor = torch.tensor(batch_meta, dtype=torch.float32).view(B, 10, 5).to(device)

    # Forward pass
    d1_model.train()
    preds = d1_model(input_ids, attention_mask, meta_tensor)

    # Verify output keys and shapes
    assert preds["logit_down"].shape == (B, 1)
    assert preds["logit_up"].shape == (B, 1)
    assert preds["pred_deltas"].shape == (B, 3)
    assert preds["logit_admissibility"].shape == (B, 5)
    assert preds["r_q"].shape == (B, 256)

    # Backward pass
    dummy_targets = {
        "target_y_down": torch.tensor([0.0, 1.0], device=device),
        "target_y_up": torch.tensor([1.0, 0.0], device=device),
        "target_deltas": torch.randn(B, 3, device=device),
        "target_admissibility": torch.tensor([[0.0, 1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0]], device=device),
        "target_aux_k": torch.tensor([2, 0], device=device),
    }
    loss, loss_breakdown = d1_model.heads.compute_loss(preds, dummy_targets)
    loss.backward()

    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    logger.info("D1 Smoke Backward complete. Loss=%.4f, Peak VRAM=%.2f GB", loss.item(), peak_vram_gb)

    # 7. D2 Feasibility Audit
    d2_audit = audit_d2_hardware_feasibility(device_id=0, required_vram_gb=14.0)
    logger.info("D2 Hardware Feasibility: %s", d2_audit)

    # Verify Listwise causal prompt formatting on 1 example
    sample_prompt = format_listwise_causal_prompt(
        question=smoke_retrieval_train[smoke_train_qids[0]]["question"],
        passages=smoke_retrieval_train[smoke_train_qids[0]]["passages"],
        max_passages=10,
    )
    assert "<DECIDE>" in sample_prompt
    assert '<DOC rank="10"' in sample_prompt

    # 8. Save diagnostic summary
    summary_data = {
        "BANNER": "SMOKE_ONLY_NOT_FOR_REPORTING",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "total_vram_gb": float(torch.cuda.get_device_properties(0).total_memory / (1024**3)) if torch.cuda.is_available() else 0.0,
            "d1_peak_vram_gb": float(peak_vram_gb),
        },
        "d0_smoke": {
            "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "n_train_queries": len(smoke_train_qids),
            "n_val_queries": len(smoke_val_qids),
            "n_features": len(feat_cols),
            "val_p_down": [float(p) for p in d0_val_preds["p_down"]],
            "val_p_up": [float(p) for p in d0_val_preds["p_up"]],
        },
        "d1_smoke": {
            "encoder": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "freeze_encoder": True,
            "d_model": 256,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "batch_size": B,
            "train_loss": float(loss.item()),
            "loss_breakdown": loss_breakdown,
        },
        "d2_feasibility": d2_audit,
        "input_length_audit": df_audit.to_dict(orient="records"),
    }

    out_file = smoke_dir / "smoke_w7d_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    logger.info("Saved smoke summary to %s", out_file)
    print("SMOKE TEST COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    run_smoke_test()
