"""Outcome-Weighted LoRA (OW-LoRA) training and tuning for Week 8-G.

Implements sample-weighted generator adaptation where training loss is scaled by:
- Helpful cases: w = 1 + gamma * max_i(Delta_i)
- Distractor cases: w = 1 + gamma * max_i(-Delta_i)
- Neutral cases: w = 1.0

Tuning grid: gamma in [0.5, 1.0, 2.0, 4.0] selected on INTERNAL-VAL.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader

from src.week8.hard_case_dataset import (
    HardCasePromptDataset,
    build_hard_case_dataset,
)
from src.week8.hard_case_lora import (
    evaluate_generator_lora,
    setup_lora_generator,
    train_generator_lora_epoch,
)

logger = logging.getLogger(__name__)


def run_ow_lora_training(
    sara_model: Any,
    sara_tokenizer: Any,
    raw_train_queries: List[Dict[str, Any]],
    train_promotion_records: List[Dict[str, Any]],
    raw_val_queries: List[Dict[str, Any]],
    val_promotion_records: List[Dict[str, Any]],
    sfr_embeddings_by_qid: Dict[str, torch.Tensor],
    output_dir: Union[str, Path],
    gamma_grid: Sequence[float] = (0.5, 1.0, 2.0),
    num_epochs: int = 3,
    lr: float = 1e-4,
    device: str = "cuda",
    smoke: bool = False,
) -> Dict[str, Any]:
    """Train and select best OW-LoRA model on INTERNAL-VAL across gamma grid."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if smoke:
        gamma_grid = [1.0]
        num_epochs = 1
        max_train_samples = 3
        max_val_samples = 2
    else:
        max_train_samples = None
        max_val_samples = None

    best_gamma = gamma_grid[0]
    best_val_loss = float("inf")
    grid_results = {}

    for gamma in gamma_grid:
        logger.info("Training OW-LoRA with gamma=%.2f (smoke=%s)", gamma, smoke)

        # Build datasets
        train_examples, _ = build_hard_case_dataset(
            raw_queries=raw_train_queries,
            promotion_records=train_promotion_records,
            tokenizer=sara_tokenizer,
            mode="outcome_weighted",
            gamma=gamma,
        )
        val_examples, _ = build_hard_case_dataset(
            raw_queries=raw_val_queries,
            promotion_records=val_promotion_records,
            tokenizer=sara_tokenizer,
            mode="outcome_weighted",
            gamma=gamma,
        )

        if max_train_samples:
            train_examples = train_examples[:max_train_samples]
        if max_val_samples:
            val_examples = val_examples[:max_val_samples]

        train_ds = HardCasePromptDataset(train_examples, sara_tokenizer)
        val_ds = HardCasePromptDataset(val_examples, sara_tokenizer)

        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

        # Wrap LoRA
        lora_model = setup_lora_generator(sara_model, r=8, lora_alpha=16)
        trainable_params = [p for p in lora_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr)

        val_loss = float("inf")
        for epoch in range(num_epochs):
            train_loss = train_generator_lora_epoch(
                sara_model=lora_model,
                dataloader=train_loader,
                optimizer=optimizer,
                sfr_embeddings_by_qid=sfr_embeddings_by_qid,
                device=device,
                max_steps=1 if smoke else None,
            )
            val_loss = evaluate_generator_lora(
                sara_model=lora_model,
                dataloader=val_loader,
                sfr_embeddings_by_qid=sfr_embeddings_by_qid,
                device=device,
                max_steps=1 if smoke else None,
            )
            logger.info("Epoch %d: train_loss=%.4f, val_loss=%.4f", epoch + 1, train_loss, val_loss)

        grid_results[gamma] = val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_gamma = gamma
            # Save checkpoint
            ckpt_dir = out_path / f"ow_lora_best_gamma_{gamma}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            if hasattr(lora_model, "save_pretrained"):
                lora_model.save_pretrained(str(ckpt_dir))

    return {
        "best_gamma": best_gamma,
        "best_val_loss": best_val_loss,
        "grid_results": grid_results,
    }
