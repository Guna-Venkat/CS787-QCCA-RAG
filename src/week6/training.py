"""Training loop and early stopping for R2 Learned Passage Router in Week 6.

Key requirements:
- Trained strictly on TRAIN-FIT (1,746 questions).
- Validated and early-stopped strictly on INTERNAL-VAL (342 questions).
- DEV is strictly excluded from training and model selection.
- TEST is never accessed.
- Tracks ranking metrics (AUPRC / MAP, Recall@k) on validation.
- Selects best checkpoint based on INTERNAL-VAL ranking performance.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score

from src.week6.data import routing_collate_fn
from src.week6.models import LearnedPassageRouter
from src.week6.routing import route_r2_learned_topk

logger = logging.getLogger(__name__)


def compute_batch_ranking_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """Compute Average Precision and Recall@k for a batch of predictions."""
    l_np = logits.detach().cpu().numpy()
    y_np = labels.detach().cpu().numpy()

    aps = []
    recalls_k = {2: [], 4: [], 6: []}

    for i in range(len(l_np)):
        y_true = y_np[i]
        y_score = l_np[i]
        num_pos = int(np.sum(y_true))
        if num_pos > 0:
            ap = average_precision_score(y_true, y_score)
            aps.append(ap)
            for k in [2, 4, 6]:
                topk_idx = np.argsort(-y_score)[:k]
                rec = np.sum(y_true[topk_idx]) / float(num_pos)
                recalls_k[k].append(rec)

    mean_ap = float(np.mean(aps)) if aps else 0.0
    res = {"map": mean_ap}
    for k in [2, 4, 6]:
        res[f"recall_at_{k}"] = float(np.mean(recalls_k[k])) if recalls_k[k] else 0.0
    return res


def evaluate_r2_on_loader(
    model: LearnedPassageRouter,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Dict[str, float]:
    """Evaluate R2 router on a validation DataLoader."""
    model.eval()
    total_loss = 0.0
    total_examples = 0

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            rank_indices = batch["rank_indices"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask, rank_indices)
            loss = criterion(logits, labels)

            B = len(labels)
            total_loss += loss.item() * B
            total_examples += B

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    mean_loss = total_loss / max(total_examples, 1)
    cat_logits = torch.cat(all_logits, dim=0)
    cat_labels = torch.cat(all_labels, dim=0)

    ranking_metrics = compute_batch_ranking_metrics(cat_logits, cat_labels)
    ranking_metrics["val_loss"] = mean_loss
    return ranking_metrics


def train_single_r2_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: LearnedPassageRouter,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    max_epochs: int = 15,
    patience: int = 3,
    pos_weight: float = 1.5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Train single R2 model on TRAIN-FIT with early stopping on INTERNAL-VAL."""
    model.to(device)

    # Loss with modest pos_weight to balance positive evidence vs background passages
    pos_weight_tensor = torch.tensor([pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    best_metric = -1.0
    best_epoch = -1
    best_state = None
    epochs_no_improve = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        train_examples = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            rank_indices = batch["rank_indices"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, rank_indices)
            loss = criterion(logits, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            B = len(labels)
            train_loss += loss.item() * B
            train_examples += B

        avg_train_loss = train_loss / max(train_examples, 1)

        # Validate on INTERNAL-VAL
        val_metrics = evaluate_r2_on_loader(model, val_loader, criterion, device)
        val_map = val_metrics["map"]
        val_loss = val_metrics["val_loss"]

        record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            **val_metrics,
        }
        history.append(record)

        logger.info(
            f"Epoch {epoch:2d}/{max_epochs:2d} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val MAP: {val_map:.4f} | Val Rec@4: {val_metrics['recall_at_4']:.4f}"
        )

        # Check early stopping on validation MAP
        if val_map > best_metric:
            best_metric = val_map
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch} (patience={patience}).")
                break

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "best_epoch": best_epoch,
                "best_metric": best_metric,
                "history": history,
            },
            p,
        )
        logger.info(f"Saved best model checkpoint to {p} (epoch {best_epoch}, MAP={best_metric:.4f})")

    return {
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "history": history,
        "final_val_metrics": history[best_epoch - 1] if best_epoch > 0 else {},
    }
