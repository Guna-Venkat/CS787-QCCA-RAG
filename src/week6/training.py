"""Training loops, differential learning rate optimization, and gradient diagnostics for Week 6.

Implements:
- Two-group AdamW optimization (encoder_lr for adapted layers, head_lr for aggregator/head)
- Unweighted BCEWithLogitsLoss for Target 3 (Multi-Label Epsilon-Admissibility)
- Gradient clipping (1.0 default)
- Checkpoint selection via INTERNAL-VAL multi-label macro AUPRC (tie-break: lower BCE)
- Early stopping with patience=3 (max epochs 10-15)
- Strict gradient & adaptation health diagnostics:
    - Verifies frozen layers receive zero gradient
    - Verifies trainable layers receive nonzero gradient
    - Computes gradient norms per parameter group
    - Computes parameter delta norms ||W_final - W_init||_2 for each adapted layer
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.week5.evaluation import evaluate_multilabel_diagnostics
from src.week5.training import EarlyStoppingMultiLabel, set_seed
from src.week6.adaptation import (
    AdaptedContentAwareAllocator,
    build_optimizer_param_groups,
)

logger = logging.getLogger(__name__)


def compute_gradient_norms(model: AdaptedContentAwareAllocator) -> Dict[str, Any]:
    """Inspect gradients across frozen and trainable modules."""
    encoder_trainable_sq = 0.0
    head_aggregator_sq = 0.0
    frozen_grad_detected = False
    zero_grad_on_trainable = False

    for name, param in model.named_parameters():
        if not param.requires_grad:
            if param.grad is not None and param.grad.abs().sum().item() > 0:
                frozen_grad_detected = True
        else:
            if param.grad is None or param.grad.abs().sum().item() == 0:
                zero_grad_on_trainable = True
            else:
                g_norm_sq = param.grad.detach().norm(2).item() ** 2
                if "encoder." in name:
                    encoder_trainable_sq += g_norm_sq
                else:
                    head_aggregator_sq += g_norm_sq

    return {
        "encoder_trainable_grad_norm": round(encoder_trainable_sq ** 0.5, 6),
        "head_aggregator_grad_norm": round(head_aggregator_sq ** 0.5, 6),
        "frozen_layers_zero_grad": not frozen_grad_detected,
        "trainable_layers_nonzero_grad": not zero_grad_on_trainable,
    }


def train_adapted_allocator(
    model: AdaptedContentAwareAllocator,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 15,
    encoder_lr: float = 2.0e-5,
    head_lr: float = 2.0e-4,
    weight_decay: float = 0.01,
    patience: int = 3,
    grad_clip: float = 1.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    pos_weight: Optional[torch.Tensor] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train AdaptedContentAwareAllocator with differential learning rates and AUPRC early stopping.

    Returns:
        Dict containing model, history, best_epoch, best_macro_auprc, and gradient health check.
    """
    model = model.to(device)

    # 1. Capture initial weights of trainable encoder parameters for delta tracking
    init_encoder_weights = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if p.requires_grad and "encoder." in name
    }

    # 2. Build Differential Parameter Groups
    param_groups = build_optimizer_param_groups(
        model,
        encoder_lr=encoder_lr,
        head_lr=head_lr,
        weight_decay=weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups)

    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    early_stopping = EarlyStoppingMultiLabel(patience=patience)
    history: List[Dict[str, Any]] = []

    latest_grad_health: Dict[str, Any] = {}

    for epoch in range(1, epochs + 1):
        # -----------------------------
        # 1. Training Phase
        # -----------------------------
        model.train()
        train_loss_total = 0.0
        n_train_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(batch)
            logits = outputs["admissibility_logits"]
            labels = batch["t3_labels"].to(device)

            loss = criterion(logits, labels)
            loss.backward()

            # Audit gradients on the first batch of every epoch
            if batch_idx == 0:
                latest_grad_health = compute_gradient_norms(model)

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    grad_clip,
                )

            optimizer.step()

            train_loss_total += float(loss.item())
            n_train_batches += 1

        avg_train_loss = train_loss_total / max(1, n_train_batches)

        # -----------------------------
        # 2. INTERNAL-VAL Phase
        # -----------------------------
        model.eval()
        val_loss_total = 0.0
        n_val_batches = 0
        all_val_probs: List[np.ndarray] = []
        all_val_targets: List[np.ndarray] = []

        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch)
                logits = outputs["admissibility_logits"]
                labels = batch["t3_labels"].to(device)

                v_loss = criterion(logits, labels)
                val_loss_total += float(v_loss.item())
                n_val_batches += 1

                probs = torch.sigmoid(logits).cpu().numpy()
                all_val_probs.append(probs)
                all_val_targets.append(labels.cpu().numpy())

        avg_val_loss = val_loss_total / max(1, n_val_batches)
        y_val_probs = np.concatenate(all_val_probs, axis=0)
        y_val_targets = np.concatenate(all_val_targets, axis=0)

        val_diag = evaluate_multilabel_diagnostics(y_val_targets, y_val_probs)
        macro_auprc = val_diag["macro_auprc"]
        macro_auroc = val_diag["macro_auroc"]

        epoch_stats = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_macro_auprc": macro_auprc,
            "val_macro_auroc": macro_auroc,
            "val_micro_auprc": val_diag["micro_auprc"],
            "val_brier": val_diag["overall_brier"],
            "val_per_budget_auprc": val_diag["per_budget_auprc"],
            "val_per_budget_auroc": val_diag["per_budget_auroc"],
            "encoder_trainable_grad_norm": latest_grad_health.get("encoder_trainable_grad_norm", 0.0),
            "head_aggregator_grad_norm": latest_grad_health.get("head_aggregator_grad_norm", 0.0),
        }
        history.append(epoch_stats)

        improved = early_stopping.step(macro_auprc, avg_val_loss, model, epoch)
        if verbose:
            tag = " [BEST]" if improved else ""
            logger.info(
                f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | Val AUPRC: {macro_auprc:.4f} | "
                f"Val AUROC: {macro_auroc:.4f}{tag}"
            )

        if early_stopping.early_stop:
            if verbose:
                logger.info(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best epoch was {early_stopping.best_epoch} (Val AUPRC: {early_stopping.best_macro_auprc:.4f})."
                )
            break

    # Restore best checkpoint
    if early_stopping.best_state_dict is not None:
        model.load_state_dict(early_stopping.best_state_dict)

    # 3. Compute parameter deltas from initialization to best checkpoint
    param_deltas: Dict[str, float] = {}
    for name, init_w in init_encoder_weights.items():
        curr_p = dict(model.named_parameters())[name]
        delta_norm = (curr_p.detach() - init_w).norm(2).item()
        param_deltas[name] = round(delta_norm, 6)

    # Aggregate deltas by layer
    layer_deltas: Dict[str, float] = {}
    for name, d_val in param_deltas.items():
        parts = name.split(".")
        layer_tag = ".".join(parts[:3])  # e.g., 'encoder.encoder.layer.5'
        layer_deltas[layer_tag] = round(layer_deltas.get(layer_tag, 0.0) + (d_val ** 2), 6)
    for k in layer_deltas:
        layer_deltas[k] = round(layer_deltas[k] ** 0.5, 6)

    # Compile comprehensive adaptation health report
    param_stats = model.get_parameter_breakdown()
    adaptation_health = {
        "capacity_condition": model.capacity_condition,
        "encoder_trainable_params": param_stats["encoder_trainable_params"],
        "encoder_frozen_params": param_stats["encoder_frozen_params"],
        "head_aggregator_params": param_stats["head_and_aggregator_trainable_params"],
        "latest_encoder_grad_norm": latest_grad_health.get("encoder_trainable_grad_norm", 0.0),
        "latest_head_aggregator_grad_norm": latest_grad_health.get("head_aggregator_grad_norm", 0.0),
        "frozen_layers_zero_grad": latest_grad_health.get("frozen_layers_zero_grad", True),
        "trainable_layers_nonzero_grad": latest_grad_health.get("trainable_layers_nonzero_grad", True),
        "layer_parameter_deltas": layer_deltas,
        "num_adapted_parameters_tracked": len(param_deltas),
    }

    return {
        "model": model,
        "history": history,
        "best_epoch": early_stopping.best_epoch,
        "best_macro_auprc": early_stopping.best_macro_auprc,
        "best_val_loss": early_stopping.best_val_loss,
        "adaptation_health": adaptation_health,
    }
