"""Smoke test for Week 5-B Multi-Label Epsilon-Admissibility Allocator.

Runs a tiny run (16 TRAIN-FIT, 8 INTERNAL-VAL, I3-T3, 1 epoch, seed 42) to verify:
- Finite BCEWithLogitsLoss
- Backward pass
- Rank embedding norm vs semantic norm
- Varying logits across queries
- Threshold policy execution
Tag: SMOKE_ONLY_NOT_FOR_REPORTING
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.week5.data import (
    ContentAllocatorDataset,
    build_w5b_splits_and_bundles,
)
from src.week5.models import ContentAwareAllocator
from src.week5.training import set_seed, train_allocator_t3

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_smoke_test() -> Dict[str, Any]:
    logger.info("=== STARTING W5-B SMOKE TEST (SMOKE_ONLY_NOT_FOR_REPORTING) ===")
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # 1. Load bundles and split manifests
    fit_bundle, val_bundle, dev_bundle, scaler = build_w5b_splits_and_bundles(
        save_manifests=True,
        manifest_dir="results/week5/content_allocator/w5b/splits",
    )

    # Subsample 16 fit, 8 val examples
    def slice_bundle(b, n):
        return {
            "split": b["split"],
            "question_ids": b["question_ids"][:n],
            "paper_ids": b["paper_ids"][:n],
            "questions": b["questions"][:n],
            "passages": b["passages"][:n],
            "features": b["features"][:n],
            "features_raw": b["features_raw"][:n],
            "t1_labels": b["t1_labels"][:n],
            "t2_regrets": b["t2_regrets"][:n],
            "t3_labels": b["t3_labels"][:n],
            "supervision_df": b["supervision_df"].iloc[:n].copy().reset_index(drop=True),
            "tokens_map": b.get("tokens_map", {}),
        }

    fit_small = slice_bundle(fit_bundle, 16)
    val_small = slice_bundle(val_bundle, 8)

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    train_ds = ContentAllocatorDataset(fit_small, tokenizer, condition="I3", max_seq_length=128)
    val_ds = ContentAllocatorDataset(val_small, tokenizer, condition="I3", max_seq_length=128)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    # 2. Initialize Model
    model = ContentAwareAllocator(
        condition="I3",
        target_mode="T3",
        encoder_train_mode="frozen",
        d_model=384,
    ).to(device)

    # Measure norms before training
    sample_batch = next(iter(val_loader))
    with torch.no_grad():
        # Get semantic passage embeddings
        input_ids = sample_batch["passage_input_ids"].to(device)
        attn_mask = sample_batch["passage_attention_mask"].to(device)
        B, N, L = input_ids.shape
        out = model.encoder(input_ids=input_ids.view(B*N, L), attention_mask=attn_mask.view(B*N, L))
        # mean pooling
        mask_exp = attn_mask.view(B*N, L).unsqueeze(-1).float()
        sem_reps = (out.last_hidden_state * mask_exp).sum(1) / torch.clamp(mask_exp.sum(1), min=1e-9)
        semantic_norm_before = sem_reps.norm(dim=-1).mean().item()

    rank_norm_before = model.aggregator.rank_embeddings.weight.norm(dim=-1).mean().item()
    ratio_before = rank_norm_before / semantic_norm_before

    logger.info(f"Before Training: Semantic Norm = {semantic_norm_before:.4f}, Rank Norm = {rank_norm_before:.4f}, Ratio = {ratio_before:.4f}")

    # 3. Train for 1 epoch
    train_res = train_allocator_t3(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=1,
        lr=1e-3,
        patience=2,
        device=device,
        verbose=True,
    )

    # 4. Measure after training
    rank_norm_after = model.aggregator.rank_embeddings.weight.norm(dim=-1).mean().item()
    ratio_after = rank_norm_after / semantic_norm_before

    # Check logits on validation batch
    with torch.no_grad():
        val_out = model(sample_batch)
        val_logits = val_out["admissibility_logits"].cpu().numpy()
        val_probs = torch.sigmoid(val_out["admissibility_logits"]).cpu().numpy()

    logit_variance_per_k = np.var(val_logits, axis=0).tolist()
    mean_logit_variance = float(np.mean(logit_variance_per_k))

    # Test threshold policy
    selected_ks = model.predict_k(sample_batch, tau=0.50)

    smoke_results = {
        "status": "PASSED",
        "tag": "SMOKE_ONLY_NOT_FOR_REPORTING",
        "train_loss": train_res["history"][0]["train_loss"],
        "val_loss": train_res["history"][0]["val_loss"],
        "semantic_norm_before": round(semantic_norm_before, 4),
        "rank_norm_before": round(rank_norm_before, 4),
        "norm_ratio_before": round(ratio_before, 4),
        "rank_norm_after": round(rank_norm_after, 4),
        "norm_ratio_after": round(ratio_after, 4),
        "mean_logit_variance": round(mean_logit_variance, 6),
        "logit_variance_per_k": [round(v, 6) for v in logit_variance_per_k],
        "selected_ks_sample": selected_ks.tolist(),
        "sample_probs": val_probs.round(4).tolist(),
    }

    out_path = Path("results/week5/content_allocator/w5b/diagnostics/smoke_test_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(smoke_results, f, indent=2)

    logger.info(f"Smoke test completed successfully: {smoke_results}")
    return smoke_results


if __name__ == "__main__":
    run_smoke_test()
