"""Training loops, early stopping, checkpointing, and optimization for Week 5 Content-Aware Allocator.

Implements:
- Training for Target 1 (Cross-Entropy with TRAIN-only inverse class weighting)
- Training for Target 2 (SmoothL1 / Huber loss for multi-output regret prediction)
- AdamW optimizer with gradient clipping and linear/cosine decay
- Primary policy-driven early stopping on DEV (hierarchical: policy F1 -> lower tokens -> lower regret)
- Checkpointing and clean state restoration
- Deterministic seed setting across Python, NumPy, and PyTorch
"""

from __future__ import annotations

import copy
import logging
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.week5.data import (
    compute_train_class_weights,
    K_CLASSES,
)
from src.week5.evaluation import evaluate_allocation_policy
from src.week5.models import ContentAwareAllocator

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Ensure deterministic execution across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Hierarchical early stopping tracker based primarily on DEV policy utility."""

    def __init__(
        self,
        patience: int = 4,
        min_delta: float = 1e-4,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.best_f1 = -1.0
        self.best_tokens = float("inf")
        self.best_regret = float("inf")
        self.best_epoch = -1
        self.best_state_dict: Optional[Dict[str, Any]] = None
        self.counter = 0
        self.early_stop = False

    def is_better(self, f1: float, tokens: float, regret: float) -> bool:
        """Hierarchy:
        1. Higher DEV Policy F1 (by at least min_delta)
        2. If F1 nearly tied (|diff| < min_delta), lower context tokens
        3. If tokens also tied, lower regret
        """
        if f1 > self.best_f1 + self.min_delta:
            return True
        elif abs(f1 - self.best_f1) <= self.min_delta:
            if tokens < self.best_tokens - 1.0:
                return True
            elif abs(tokens - self.best_tokens) <= 1.0:
                return regret < self.best_regret - self.min_delta
        return False

    def step(
        self,
        f1: float,
        tokens: float,
        regret: float,
        model: nn.Module,
        epoch: int,
    ) -> bool:
        """Returns True if this epoch improved the best state."""
        if self.is_better(f1, tokens, regret):
            self.best_f1 = f1
            self.best_tokens = tokens
            self.best_regret = regret
            self.best_epoch = epoch
            self.best_state_dict = copy.deepcopy(model.state_dict())
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


def train_allocator(
    model: ContentAwareAllocator,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    dev_bundle: Dict[str, Any],
    epochs: int = 25,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    patience: int = 4,
    grad_clip: float = 1.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_class_weights: bool = True,
    train_t1_labels: Optional[np.ndarray] = None,
    epsilon: float = 0.01,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train ContentAwareAllocator for Target 1 or Target 2 with policy-driven early stopping."""
    model = model.to(device)
    target_mode = model.target_mode
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    # Loss criterion
    if target_mode == "T1":
        if use_class_weights and train_t1_labels is not None:
            weights = compute_train_class_weights(train_t1_labels).to(device)
            criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            criterion = nn.CrossEntropyLoss()
    elif target_mode == "T2":
        criterion = nn.SmoothL1Loss(beta=0.01)
    else:
        criterion = nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(patience=patience)
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        # 1. Training Phase
        model.train()
        train_loss_total = 0.0
        n_train_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch)

            if target_mode == "T1":
                labels = batch["t1_label"].to(device)
                loss = criterion(outputs["logits"], labels)
            elif target_mode == "T2":
                regrets = batch["t2_regret"].to(device)
                loss = criterion(outputs["pred_regret"], regrets)
            else:
                loss = criterion(outputs["logits"], batch["t1_label"].to(device))

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    grad_clip,
                )
            optimizer.step()

            train_loss_total += float(loss.item())
            n_train_batches += 1

        avg_train_loss = train_loss_total / max(1, n_train_batches)

        # 2. DEV Evaluation Phase
        model.eval()
        dev_loss_total = 0.0
        n_dev_batches = 0
        all_k_preds: List[int] = []

        with torch.no_grad():
            for batch in dev_loader:
                outputs = model(batch)
                if target_mode == "T1":
                    labels = batch["t1_label"].to(device)
                    d_loss = criterion(outputs["logits"], labels)
                elif target_mode == "T2":
                    regrets = batch["t2_regret"].to(device)
                    d_loss = criterion(outputs["pred_regret"], regrets)
                else:
                    d_loss = criterion(outputs["logits"], batch["t1_label"].to(device))
                dev_loss_total += float(d_loss.item())
                n_dev_batches += 1

                # Predict k
                k_batch = model.predict_k(batch, epsilon=epsilon)
                all_k_preds.extend(k_batch.tolist())

        avg_dev_loss = dev_loss_total / max(1, n_dev_batches)

        # Compute DEV Policy metrics
        policy_eval = evaluate_allocation_policy(
            df_supervision=dev_bundle["supervision_df"],
            predicted_k_series=all_k_preds,
            dev_tokens_map=dev_bundle.get("dev_tokens_map"),
        )

        dev_f1 = policy_eval["mean_answer_f1"]
        dev_tokens = policy_eval["mean_context_tokens"]
        dev_regret = policy_eval["mean_oracle_regret"]
        dev_k = policy_eval["mean_selected_k"]

        epoch_stats = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "dev_loss": round(avg_dev_loss, 4),
            "dev_policy_f1": dev_f1,
            "dev_mean_tokens": dev_tokens,
            "dev_mean_k": dev_k,
            "dev_mean_regret": dev_regret,
        }
        history.append(epoch_stats)

        improved = early_stopping.step(dev_f1, dev_tokens, dev_regret, model, epoch)
        if verbose:
            tag = " [BEST]" if improved else ""
            logger.info(
                f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Dev Loss: {avg_dev_loss:.4f} | "
                f"Dev F1: {dev_f1:.4f} | Dev Tokens: {dev_tokens:.1f} | Dev k: {dev_k:.2f}{tag}"
            )

        if early_stopping.early_stop:
            if verbose:
                logger.info(f"Early stopping triggered at epoch {epoch}. Best epoch was {early_stopping.best_epoch}.")
            break

    # Restore best checkpoint
    if early_stopping.best_state_dict is not None:
        model.load_state_dict(early_stopping.best_state_dict)

    return {
        "model": model,
        "history": history,
        "best_epoch": early_stopping.best_epoch,
        "best_f1": early_stopping.best_f1,
        "best_tokens": early_stopping.best_tokens,
        "best_regret": early_stopping.best_regret,
    }
