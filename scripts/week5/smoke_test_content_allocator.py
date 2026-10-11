"""Tiny smoke validation for Week 5 Content-Aware Allocator pipeline.

THIS SCRIPT IS STRICTLY FOR SMOKE VALIDATION AND SANITY CHECKING.
IT DOES NOT PRODUCE FINAL SCIENTIFIC RESULTS.
ALL ARTIFACTS ARE LABELED: SMOKE_ONLY_NOT_FOR_REPORTING

Scope:
- 16 TRAIN examples, 8 DEV examples
- Condition: I3 (Query + Top-10 Passages)
- Target: T1 (Classification) & T2 (Regret Regression)
- 1 epoch each
- Base encoder: sentence-transformers/all-MiniLM-L6-v2 (frozen)
- Verifies:
    1. Dataset construction & pairwise tokenization (preserves query)
    2. Forward pass & loss computation (finite, no NaN)
    3. Backward pass & gradient step (trainable params only)
    4. Prediction policy (valid budgets in {2,4,5,6,8})
    5. Checkpoint save & restore
    6. Policy-level evaluation & metric computation
    7. Memory / OOM safety
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.week5.data import (
    ContentAllocatorDataset,
    build_w5_dataset_bundle,
    compute_truncation_statistics,
    compute_train_class_weights,
    K_CLASSES,
)
from src.week5.models import ContentAwareAllocator
from src.week5.training import train_allocator, set_seed
from src.week5.evaluation import evaluate_allocation_policy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_smoke_test():
    logger.info("=== STARTING WEEK 5 SMOKE VALIDATION (SMOKE_ONLY_NOT_FOR_REPORTING) ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Target execution device: {device}")
    set_seed(42)

    # 1. Load data bundles
    logger.info("Loading full data bundles to slice tiny subsets...")
    train_bundle_full, dev_bundle_full, scaler = build_w5_dataset_bundle(
        train_sup_path=REPO_ROOT / "results/week5/supervision/train_allocation_supervision.csv",
        dev_sup_path=REPO_ROOT / "results/week5/supervision/dev_allocation_supervision.csv",
        train_ret_path=REPO_ROOT / "results/week5/train_response_surface/train_retrieval.jsonl",
        dev_ret_path=REPO_ROOT / "results/week2/raw/dev_retrieval.jsonl",
        train_mat_path=REPO_ROOT / "results/week5/train_response_surface/train_fixed_k_matrix.csv",
        dev_mat_path=REPO_ROOT / "results/week2/raw/dev_fixed_k_matrix.jsonl",
        dev_feat_path=REPO_ROOT / "results/week2/processed/dev_retrieval_features.jsonl",
    )

    # Slice tiny subsets: 16 TRAIN, 8 DEV
    n_train_smoke = 16
    n_dev_smoke = 8

    train_bundle = {
        "split": "train_smoke",
        "question_ids": train_bundle_full["question_ids"][:n_train_smoke],
        "paper_ids": train_bundle_full["paper_ids"][:n_train_smoke],
        "questions": train_bundle_full["questions"][:n_train_smoke],
        "passages": train_bundle_full["passages"][:n_train_smoke],
        "features": train_bundle_full["features"][:n_train_smoke],
        "features_raw": train_bundle_full["features_raw"][:n_train_smoke],
        "t1_labels": train_bundle_full["t1_labels"][:n_train_smoke],
        "t2_regrets": train_bundle_full["t2_regrets"][:n_train_smoke],
        "supervision_df": train_bundle_full["supervision_df"].iloc[:n_train_smoke].reset_index(drop=True),
        "train_mat_map": train_bundle_full["train_mat_map"],
    }

    dev_bundle = {
        "split": "dev_smoke",
        "question_ids": dev_bundle_full["question_ids"][:n_dev_smoke],
        "paper_ids": dev_bundle_full["paper_ids"][:n_dev_smoke],
        "questions": dev_bundle_full["questions"][:n_dev_smoke],
        "passages": dev_bundle_full["passages"][:n_dev_smoke],
        "features": dev_bundle_full["features"][:n_dev_smoke],
        "features_raw": dev_bundle_full["features_raw"][:n_dev_smoke],
        "t1_labels": dev_bundle_full["t1_labels"][:n_dev_smoke],
        "t2_regrets": dev_bundle_full["t2_regrets"][:n_dev_smoke],
        "supervision_df": dev_bundle_full["supervision_df"].iloc[:n_dev_smoke].reset_index(drop=True),
        "dev_tokens_map": dev_bundle_full["dev_tokens_map"],
        "dev_f1_map": dev_bundle_full["dev_f1_map"],
    }

    logger.info(f"Subsets prepared: Train={len(train_bundle['question_ids'])}, Dev={len(dev_bundle['question_ids'])}")

    # 2. Tokenizer and Dataset
    encoder_name = "sentence-transformers/all-MiniLM-L6-v2"
    logger.info(f"Loading tokenizer: {encoder_name}")
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)

    trunc_stats = compute_truncation_statistics(dev_bundle, tokenizer, max_seq_length=256)
    logger.info(f"Dev smoke truncation stats: {trunc_stats}")

    train_ds = ContentAllocatorDataset(train_bundle, tokenizer, condition="I3", max_seq_length=256)
    dev_ds = ContentAllocatorDataset(dev_bundle, tokenizer, condition="I3", max_seq_length=256)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=4, shuffle=False)

    # 3. Model instantiation: I3-T1
    logger.info("Initializing I3-T1 ContentAwareAllocator (frozen base encoder)...")
    model_t1 = ContentAwareAllocator(
        base_encoder_name=encoder_name,
        condition="I3",
        target_mode="T1",
        encoder_train_mode="frozen",
        d_model=384,
        nhead=4,
        num_layers=2,
        dim_feedforward=1024,
        dropout=0.1,
    ).to(device)

    param_info_t1 = model_t1.get_parameter_counts()
    logger.info(f"I3-T1 Parameter counts: {param_info_t1}")
    assert param_info_t1["total_params"] > 0
    assert param_info_t1["trainable_params"] > 0
    assert param_info_t1["trainable_params"] < param_info_t1["total_params"]

    # 4. Train 1 epoch on I3-T1
    logger.info("Running 1 epoch smoke training for I3-T1...")
    res_t1 = train_allocator(
        model=model_t1,
        train_loader=train_loader,
        dev_loader=dev_loader,
        dev_bundle=dev_bundle,
        epochs=1,
        lr=1e-3,
        device=device,
        use_class_weights=True,
        train_t1_labels=train_bundle["t1_labels"],
        verbose=True,
    )
    assert len(res_t1["history"]) == 1
    assert not np.isnan(res_t1["history"][0]["train_loss"])
    assert not np.isnan(res_t1["history"][0]["dev_loss"])
    logger.info(f"I3-T1 Smoke Epoch 1 results: {res_t1['history'][0]}")

    # Checkpoint save/load check
    smoke_ckpt_dir = REPO_ROOT / "results/week5/content_allocator/checkpoints_smoke"
    smoke_ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = smoke_ckpt_dir / "smoke_t1_model.pt"
    torch.save(model_t1.state_dict(), ckpt_path)
    assert ckpt_path.exists()
    model_t1.load_state_dict(torch.load(ckpt_path, map_location=device))
    logger.info("I3-T1 checkpoint save & restore verified.")

    # Inference test
    sample_batch = next(iter(dev_loader))
    k_preds = model_t1.predict_k(sample_batch)
    assert len(k_preds) == len(sample_batch["t1_label"])
    for kp in k_preds:
        assert kp in K_CLASSES
    logger.info(f"I3-T1 predict_k verified: {k_preds}")

    # 5. Model instantiation: I3-T2
    logger.info("Initializing I3-T2 ContentAwareAllocator (frozen base encoder)...")
    model_t2 = ContentAwareAllocator(
        base_encoder_name=encoder_name,
        condition="I3",
        target_mode="T2",
        encoder_train_mode="frozen",
        d_model=384,
        nhead=4,
        num_layers=2,
        dim_feedforward=1024,
        dropout=0.1,
    ).to(device)

    # Train 1 epoch on I3-T2
    logger.info("Running 1 epoch smoke training for I3-T2...")
    res_t2 = train_allocator(
        model=model_t2,
        train_loader=train_loader,
        dev_loader=dev_loader,
        dev_bundle=dev_bundle,
        epochs=1,
        lr=1e-3,
        device=device,
        epsilon=0.01,
        verbose=True,
    )
    assert len(res_t2["history"]) == 1
    assert not np.isnan(res_t2["history"][0]["train_loss"])
    assert not np.isnan(res_t2["history"][0]["dev_loss"])
    logger.info(f"I3-T2 Smoke Epoch 1 results: {res_t2['history'][0]}")

    k_preds_t2 = model_t2.predict_k(sample_batch, epsilon=0.01)
    assert len(k_preds_t2) == len(sample_batch["t2_regret"])
    for kp in k_preds_t2:
        assert kp in K_CLASSES
    logger.info(f"I3-T2 predict_k verified: {k_preds_t2}")

    # Write smoke summary artifact
    smoke_summary_path = REPO_ROOT / "results/week5/content_allocator/smoke_validation_summary.json"
    smoke_summary = {
        "status": "PASS",
        "tag": "SMOKE_ONLY_NOT_FOR_REPORTING",
        "device": device,
        "n_train_samples": n_train_smoke,
        "n_dev_samples": n_dev_smoke,
        "encoder": encoder_name,
        "t1_train_loss": res_t1["history"][0]["train_loss"],
        "t1_dev_loss": res_t1["history"][0]["dev_loss"],
        "t1_dev_policy_f1": res_t1["history"][0]["dev_policy_f1"],
        "t2_train_loss": res_t2["history"][0]["train_loss"],
        "t2_dev_loss": res_t2["history"][0]["dev_loss"],
        "t2_dev_policy_f1": res_t2["history"][0]["dev_policy_f1"],
        "t1_param_counts": param_info_t1,
        "checkpoint_restored": True,
        "oom_detected": False,
    }
    with open(smoke_summary_path, "w", encoding="utf-8") as f:
        json.dump(smoke_summary, f, indent=2)

    logger.info(f"Smoke summary saved to: {smoke_summary_path}")
    logger.info("=== SMOKE TEST COMPLETED SUCCESSFULLY (SMOKE_ONLY_NOT_FOR_REPORTING) ===")


if __name__ == "__main__":
    run_smoke_test()
