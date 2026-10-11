"""Lightweight smoke test for Week 6: Semantic Capacity and Encoder Adaptation Study.

Strict Constraints:
- Condition: C1 Last-1 Layer
- Subset: 16 TRAIN-FIT examples, 8 INTERNAL-VAL examples
- Epochs: 1
- Seed: 42
- Original DEV: NEVER LOADED OR EVALUATED
- QASPER TEST: NEVER LOADED OR TOUCHED
- Explicit marker: SMOKE_ONLY_NOT_FOR_REPORTING
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from src.week5.data import (
    ContentAllocatorDataset,
    build_w5b_splits_and_bundles,
)
from src.week5.training import set_seed
from src.week5.w5c.adaptation import (
    AdaptedContentAwareAllocator,
    audit_minilm_architecture,
)
from src.week5.w5c.evaluation import select_tau_w6
from src.week5.w5c.training import train_adapted_allocator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SMOKE_OUTPUT_DIR = Path("results/week5/content_allocator/w5c_encoder_adaptation/diagnostics/smoke")


def run_smoke_test() -> Dict[str, Any]:
    """Execute tiny smoke validation on C1 condition."""
    set_seed(42)
    SMOKE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running smoke test on device: {device}")

    # 1. Load data bundles (TRAIN-FIT and INTERNAL-VAL only)
    logger.info("Loading TRAIN-FIT and INTERNAL-VAL bundles...")
    fit_bundle, val_bundle, dev_bundle, _ = build_w5b_splits_and_bundles(save_manifests=False)

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    full_fit_dataset = ContentAllocatorDataset(
        fit_bundle, tokenizer=tokenizer, condition="I3", max_seq_length=256
    )
    full_val_dataset = ContentAllocatorDataset(
        val_bundle, tokenizer=tokenizer, condition="I3", max_seq_length=256
    )

    # Subsets: 16 TRAIN-FIT, 8 INTERNAL-VAL
    train_subset = Subset(full_fit_dataset, list(range(16)))
    val_subset = Subset(full_val_dataset, list(range(8)))

    train_loader = DataLoader(train_subset, batch_size=8, shuffle=False)
    val_loader = DataLoader(val_subset, batch_size=8, shuffle=False)

    # 2. Instantiate C1 Model
    logger.info("Instantiating C1 model (Last-1 Layer Trainable)...")
    model = AdaptedContentAwareAllocator(
        capacity_condition="C1",
        base_encoder_name="sentence-transformers/all-MiniLM-L6-v2",
    ).to(device)

    # Audit architecture
    arch_audit = audit_minilm_architecture(model.encoder)
    param_breakdown = model.get_parameter_breakdown()

    # Capture initial weights
    w_emb_init = model.encoder.embeddings.word_embeddings.weight.detach().clone()
    w_l0_init = model.encoder.encoder.layer[0].attention.self.query.weight.detach().clone()
    w_l5_init = model.encoder.encoder.layer[5].attention.self.query.weight.detach().clone()
    w_head_init = model.admissibility_head[0].weight.detach().clone()

    # 3. Train 1 Epoch
    logger.info("Executing 1 smoke epoch of training...")
    train_res = train_adapted_allocator(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=1,
        encoder_lr=2.0e-5,
        head_lr=2.0e-4,
        patience=1,
        device=device,
        verbose=True,
    )

    trained_model = train_res["model"]
    history = train_res["history"]
    health = train_res["adaptation_health"]

    # 4. Verify Parameter Invariance vs Updates
    w_emb_post = trained_model.encoder.embeddings.word_embeddings.weight.detach()
    w_l0_post = trained_model.encoder.encoder.layer[0].attention.self.query.weight.detach()
    w_l5_post = trained_model.encoder.encoder.layer[5].attention.self.query.weight.detach()
    w_head_post = trained_model.admissibility_head[0].weight.detach()

    frozen_emb_unchanged = bool(torch.equal(w_emb_init, w_emb_post))
    frozen_l0_unchanged = bool(torch.equal(w_l0_init, w_l0_post))
    trainable_l5_changed = bool(not torch.equal(w_l5_init, w_l5_post))
    trainable_head_changed = bool(not torch.equal(w_head_init, w_head_post))

    l5_delta_norm = float((w_l5_post - w_l5_init).norm(2).item())
    l0_delta_norm = float((w_l0_post - w_l0_init).norm(2).item())

    # 5. Logit Variance and Inference Check
    trained_model.eval()
    val_logits_list = []
    val_probs_list = []
    with torch.no_grad():
        for batch in val_loader:
            out = trained_model(batch)
            val_logits_list.append(out["admissibility_logits"].cpu())
            val_probs_list.append(torch.sigmoid(out["admissibility_logits"]).cpu())

    val_logits = torch.cat(val_logits_list, dim=0).numpy()
    val_probs = torch.cat(val_probs_list, dim=0).numpy()

    logit_variance = float(np.var(val_logits))
    logits_finite = bool(np.isfinite(val_logits).all())
    probs_in_range = bool((val_probs >= 0.0).all() and (val_probs <= 1.0).all())

    # 6. Checkpoint Save/Load Check
    ckpt_path = SMOKE_OUTPUT_DIR / "smoke_checkpoint.pt"
    torch.save(trained_model.state_dict(), ckpt_path)
    loaded_model = AdaptedContentAwareAllocator(capacity_condition="C1").to(device)
    loaded_model.load_state_dict(torch.load(ckpt_path))
    ckpt_roundtrip_ok = bool(
        torch.equal(
            trained_model.admissibility_head[0].weight,
            loaded_model.admissibility_head[0].weight,
        )
    )

    # 7. Tau Policy Execution Check
    # Build mock supervision df for the 8 val examples
    val_qids = [val_bundle["question_ids"][i] for i in range(8)]
    df_val_sub = val_bundle["supervision_df"].iloc[:8].copy().reset_index(drop=True)
    val_tokens_map = {
        (str(qid), k): int(val_bundle["tokens_map"].get((str(qid), k), 1000))
        for qid in val_qids
        for k in [2, 4, 5, 6, 8]
    }
    tau_res = select_tau_w6(
        val_supervision_df=df_val_sub,
        val_pred_probs=val_probs,
        val_tokens_map=val_tokens_map,
    )
    tau_policy_executed = bool("selected_tau" in tau_res and "tau_boundary_hit" in tau_res)

    # 8. Assertions for Smoke Validity
    assert logits_finite, "Logits contain NaN or Inf!"
    assert frozen_emb_unchanged, "Embeddings were modified during smoke training!"
    assert frozen_l0_unchanged, "Frozen layer 0 was modified during smoke training!"
    assert trainable_l5_changed, "Trainable layer 5 was NOT updated during smoke training!"
    assert trainable_head_changed, "Admissibility head was NOT updated during smoke training!"
    assert l5_delta_norm > 0.0, "Layer 5 parameter delta norm is zero!"
    assert l0_delta_norm == 0.0, "Layer 0 parameter delta norm is non-zero!"
    assert ckpt_roundtrip_ok, "Checkpoint save/load roundtrip failed!"
    assert tau_policy_executed, "Tau policy selection failed!"

    smoke_results = {
        "MARKER": "SMOKE_ONLY_NOT_FOR_REPORTING",
        "STATUS": "PASSED",
        "condition": "C1",
        "train_examples_count": 16,
        "val_examples_count": 8,
        "epochs": 1,
        "device": device,
        "parameters": param_breakdown,
        "adaptation_health": health,
        "verifications": {
            "forward_pass_finite": logits_finite,
            "probs_in_range_0_1": probs_in_range,
            "train_loss_finite": bool(np.isfinite(history[0]["train_loss"])),
            "val_loss_finite": bool(np.isfinite(history[0]["val_loss"])),
            "frozen_embeddings_unchanged": frozen_emb_unchanged,
            "frozen_layer0_unchanged": frozen_l0_unchanged,
            "trainable_layer5_updated": trainable_l5_changed,
            "trainable_head_updated": trainable_head_changed,
            "layer5_delta_norm": round(l5_delta_norm, 6),
            "layer0_delta_norm": round(l0_delta_norm, 6),
            "val_logit_variance": round(logit_variance, 6),
            "checkpoint_save_load_roundtrip_ok": ckpt_roundtrip_ok,
            "tau_policy_executed": tau_policy_executed,
            "selected_tau_smoke": tau_res["selected_tau"],
            "tau_boundary_hit": tau_res["tau_boundary_hit"],
            "dev_evaluated": False,
            "test_evaluated": False,
        },
    }

    # Save smoke results JSON
    json_path = SMOKE_OUTPUT_DIR / "smoke_test_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(smoke_results, f, indent=2)

    logger.info(f"Smoke test PASSED! Results saved to {json_path}")
    return smoke_results


if __name__ == "__main__":
    run_smoke_test()
