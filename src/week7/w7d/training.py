"""Training routines for D0 (GBDT) and D1 (Hierarchical Cross-Encoder) for Week 7-D.

Provides:
- D0 tabular GBDT training on enriched cross-encoder features.
- PyTorch HierarchicalFidelityDataset for D1.
- Training loop for D1 with multi-task loss, validation tracking, and early stopping.
- Representation caching utility for D1-A fast experimentation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.week7.w7d.hierarchical_model import HierarchicalCrossEncoderRouter
from src.week7.w7d.multitask_heads import MultiTaskUtilityHeads

logger = logging.getLogger(__name__)


class D0TabularRouter:
    """D0 Model: GBDT models trained on joint cross-encoder features."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ) -> None:
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "verbose": -1,
        }
        self.clf_down = lgb.LGBMClassifier(**self.params)
        self.clf_up = lgb.LGBMClassifier(**self.params)
        self.reg_delta_5 = lgb.LGBMRegressor(**self.params)
        self.reg_delta_6 = lgb.LGBMRegressor(**self.params)
        self.reg_delta_8 = lgb.LGBMRegressor(**self.params)
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, df_targets: pd.DataFrame) -> "D0TabularRouter":
        """Fit downshift, upshift, and delta models."""
        self.feature_names_ = list(X.columns)

        # Fit classifiers
        self.clf_down.fit(X, df_targets["target_y_down"].values.astype(int))
        self.clf_up.fit(X, df_targets["target_y_up"].values.astype(int))

        # Fit delta regressors
        self.reg_delta_5.fit(X, df_targets["target_delta_5"].values.astype(float))
        self.reg_delta_6.fit(X, df_targets["target_delta_6"].values.astype(float))
        self.reg_delta_8.fit(X, df_targets["target_delta_8"].values.astype(float))
        return self

    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predict probabilities and deltas."""
        p_down = self.clf_down.predict_proba(X)[:, 1]
        p_up = self.clf_up.predict_proba(X)[:, 1]
        d5 = self.reg_delta_5.predict(X)
        d6 = self.reg_delta_6.predict(X)
        d8 = self.reg_delta_8.predict(X)

        return {
            "p_down": p_down,
            "p_up": p_up,
            "pred_delta_5": d5,
            "pred_delta_6": d6,
            "pred_delta_8": d8,
        }


class CachedEmbeddingDataset(Dataset):
    """PyTorch Dataset loading precomputed (B, 10, d_enc) representations and targets."""

    def __init__(
        self,
        embeddings: torch.Tensor,       # (N, 10, d_enc)
        metadata: torch.Tensor,         # (N, 10, meta_dim)
        df_targets: pd.DataFrame,
    ) -> None:
        self.embeddings = embeddings
        self.metadata = metadata
        self.df_targets = df_targets.reset_index(drop=True)

        self.y_down = torch.tensor(self.df_targets["target_y_down"].values, dtype=torch.float32)
        self.y_up = torch.tensor(self.df_targets["target_y_up"].values, dtype=torch.float32)

        deltas = self.df_targets[["target_delta_5", "target_delta_6", "target_delta_8"]].values
        self.deltas = torch.tensor(deltas, dtype=torch.float32)

        adm = self.df_targets[["target_z_2", "target_z_4", "target_z_5", "target_z_6", "target_z_8"]].values
        self.admissibility = torch.tensor(adm, dtype=torch.float32)

        if "target_aux_k_index" in self.df_targets.columns:
            self.aux_k = torch.tensor(self.df_targets["target_aux_k_index"].values, dtype=torch.long)
        else:
            self.aux_k = torch.zeros(len(self.df_targets), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.df_targets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "embeddings": self.embeddings[idx],
            "metadata": self.metadata[idx],
            "target_y_down": self.y_down[idx],
            "target_y_up": self.y_up[idx],
            "target_deltas": self.deltas[idx],
            "target_admissibility": self.admissibility[idx],
            "target_aux_k": self.aux_k[idx],
        }


def train_d1_aggregator(
    model: HierarchicalCrossEncoderRouter,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 15,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    loss_weights: Optional[Dict[str, float]] = None,
    early_stopping_patience: int = 4,
    device: Optional[str] = None,
) -> Tuple[HierarchicalCrossEncoderRouter, List[Dict[str, float]]]:
    """Train D1 set-level aggregator and multi-task heads on cached representations."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss_total = 0.0
        n_train_batches = 0

        for batch in train_loader:
            emb = batch["embeddings"].to(dev)
            meta = batch["metadata"].to(dev)

            targets = {
                "target_y_down": batch["target_y_down"].to(dev),
                "target_y_up": batch["target_y_up"].to(dev),
                "target_deltas": batch["target_deltas"].to(dev),
                "target_admissibility": batch["target_admissibility"].to(dev),
                "target_aux_k": batch["target_aux_k"].to(dev),
            }

            optimizer.zero_grad()
            preds = model.forward_from_cached_embeddings(emb, meta)
            loss, _ = model.heads.compute_loss(preds, targets, loss_weights=loss_weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_total += float(loss.item())
            n_train_batches += 1

        avg_train_loss = train_loss_total / max(1, n_train_batches)

        # Validation phase
        model.eval()
        val_loss_total = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                emb = batch["embeddings"].to(dev)
                meta = batch["metadata"].to(dev)
                targets = {
                    "target_y_down": batch["target_y_down"].to(dev),
                    "target_y_up": batch["target_y_up"].to(dev),
                    "target_deltas": batch["target_deltas"].to(dev),
                    "target_admissibility": batch["target_admissibility"].to(dev),
                    "target_aux_k": batch["target_aux_k"].to(dev),
                }
                preds = model.forward_from_cached_embeddings(emb, meta)
                loss, _ = model.heads.compute_loss(preds, targets, loss_weights=loss_weights)
                val_loss_total += float(loss.item())
                n_val_batches += 1

        avg_val_loss = val_loss_total / max(1, n_val_batches)

        record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        }
        history.append(record)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info("Early stopping triggered at epoch %d (best val loss: %.4f).", epoch, best_val_loss)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


class HierarchicalTokenDataset(Dataset):
    """PyTorch Dataset loading raw token IDs, masks, and metadata for D1-B end-to-end training."""

    def __init__(
        self,
        input_ids: torch.Tensor,       # (N, 10, seq_len)
        attention_mask: torch.Tensor,  # (N, 10, seq_len)
        metadata: torch.Tensor,        # (N, 10, meta_dim)
        df_targets: pd.DataFrame,
    ) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.metadata = metadata
        self.df_targets = df_targets.reset_index(drop=True)

        self.y_down = torch.tensor(self.df_targets["target_y_down"].values, dtype=torch.float32)
        self.y_up = torch.tensor(self.df_targets["target_y_up"].values, dtype=torch.float32)
        deltas = self.df_targets[["target_delta_5", "target_delta_6", "target_delta_8"]].values
        self.deltas = torch.tensor(deltas, dtype=torch.float32)
        adm = self.df_targets[["target_z_2", "target_z_4", "target_z_5", "target_z_6", "target_z_8"]].values
        self.admissibility = torch.tensor(adm, dtype=torch.float32)
        if "target_aux_k_index" in self.df_targets.columns:
            self.aux_k = torch.tensor(self.df_targets["target_aux_k_index"].values, dtype=torch.long)
        else:
            self.aux_k = torch.zeros(len(self.df_targets), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.df_targets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "metadata": self.metadata[idx],
            "target_y_down": self.y_down[idx],
            "target_y_up": self.y_up[idx],
            "target_deltas": self.deltas[idx],
            "target_admissibility": self.admissibility[idx],
            "target_aux_k": self.aux_k[idx],
        }


def train_d1b_end_to_end(
    model: HierarchicalCrossEncoderRouter,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 4,
    encoder_lr: float = 2e-5,
    aggregator_lr: float = 1e-4,
    weight_decay: float = 1e-4,
    loss_weights: Optional[Dict[str, float]] = None,
    early_stopping_patience: int = 3,
    device: Optional[str] = None,
) -> Tuple[HierarchicalCrossEncoderRouter, List[Dict[str, float]]]:
    """Train D1-B with unfrozen top encoder layers and set aggregator end-to-end."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)

    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    other_params = [p for p in model.aggregator.parameters() if p.requires_grad] + [
        p for p in model.heads.parameters() if p.requires_grad
    ]

    param_groups = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": encoder_lr})
    if other_params:
        param_groups.append({"params": other_params, "lr": aggregator_lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    history: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        n_train = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(dev)
            attention_mask = batch["attention_mask"].to(dev)
            meta = batch["metadata"].to(dev)

            targets = {
                "target_y_down": batch["target_y_down"].to(dev),
                "target_y_up": batch["target_y_up"].to(dev),
                "target_deltas": batch["target_deltas"].to(dev),
                "target_admissibility": batch["target_admissibility"].to(dev),
                "target_aux_k": batch["target_aux_k"].to(dev),
            }

            optimizer.zero_grad()
            preds = model(input_ids, attention_mask, meta)
            loss, _ = model.heads.compute_loss(preds, targets, loss_weights=loss_weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_total += float(loss.item())
            n_train += 1

        avg_train_loss = train_loss_total / max(1, n_train)

        # Validation phase
        model.eval()
        val_loss_total = 0.0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(dev)
                attention_mask = batch["attention_mask"].to(dev)
                meta = batch["metadata"].to(dev)
                targets = {
                    "target_y_down": batch["target_y_down"].to(dev),
                    "target_y_up": batch["target_y_up"].to(dev),
                    "target_deltas": batch["target_deltas"].to(dev),
                    "target_admissibility": batch["target_admissibility"].to(dev),
                    "target_aux_k": batch["target_aux_k"].to(dev),
                }
                preds = model(input_ids, attention_mask, meta)
                loss, _ = model.heads.compute_loss(preds, targets, loss_weights=loss_weights)
                val_loss_total += float(loss.item())
                n_val += 1

        avg_val_loss = val_loss_total / max(1, n_val)
        history.append({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info("D1-B Early stopping at epoch %d (best val loss: %.4f).", epoch, best_val_loss)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


class ListwisePromptDataset(Dataset):
    """PyTorch Dataset loading tokenized listwise causal prompts and downstream targets for D2."""

    def __init__(
        self,
        input_ids: torch.Tensor,       # (N, seq_len)
        attention_mask: torch.Tensor,  # (N, seq_len)
        df_targets: pd.DataFrame,
    ) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.df_targets = df_targets.reset_index(drop=True)

        self.y_down = torch.tensor(self.df_targets["target_y_down"].values, dtype=torch.float32)
        self.y_up = torch.tensor(self.df_targets["target_y_up"].values, dtype=torch.float32)
        deltas = self.df_targets[["target_delta_5", "target_delta_6", "target_delta_8"]].values
        self.deltas = torch.tensor(deltas, dtype=torch.float32)
        adm = self.df_targets[["target_z_2", "target_z_4", "target_z_5", "target_z_6", "target_z_8"]].values
        self.admissibility = torch.tensor(adm, dtype=torch.float32)
        if "target_aux_k_index" in self.df_targets.columns:
            self.aux_k = torch.tensor(self.df_targets["target_aux_k_index"].values, dtype=torch.long)
        else:
            self.aux_k = torch.zeros(len(self.df_targets), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.df_targets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "target_y_down": self.y_down[idx],
            "target_y_up": self.y_up[idx],
            "target_deltas": self.deltas[idx],
            "target_admissibility": self.admissibility[idx],
            "target_aux_k": self.aux_k[idx],
        }


def train_d2_listwise_router(
    model: Any,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 2,
    gradient_accumulation_steps: int = 16,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    loss_weights: Optional[Dict[str, float]] = None,
    device: Optional[str] = None,
) -> Tuple[Any, List[Dict[str, float]]]:
    """Train D2 Listwise 7B LLM router with LoRA and gradient accumulation."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)

    history: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        n_train = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(dev)
            attention_mask = batch["attention_mask"].to(dev)

            targets = {
                "target_y_down": batch["target_y_down"].to(dev),
                "target_y_up": batch["target_y_up"].to(dev),
                "target_deltas": batch["target_deltas"].to(dev),
                "target_admissibility": batch["target_admissibility"].to(dev),
                "target_aux_k": batch["target_aux_k"].to(dev),
            }

            preds = model(input_ids, attention_mask)
            loss, _ = model.heads.compute_loss(preds, targets, loss_weights=loss_weights)
            loss_scaled = loss / gradient_accumulation_steps
            loss_scaled.backward()

            train_loss_total += float(loss.item())
            n_train += 1

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        avg_train_loss = train_loss_total / max(1, n_train)

        # Validation
        model.eval()
        val_loss_total = 0.0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(dev)
                attention_mask = batch["attention_mask"].to(dev)
                targets = {
                    "target_y_down": batch["target_y_down"].to(dev),
                    "target_y_up": batch["target_y_up"].to(dev),
                    "target_deltas": batch["target_deltas"].to(dev),
                    "target_admissibility": batch["target_admissibility"].to(dev),
                    "target_aux_k": batch["target_aux_k"].to(dev),
                }
                preds = model(input_ids, attention_mask)
                loss, _ = model.heads.compute_loss(preds, targets, loss_weights=loss_weights)
                val_loss_total += float(loss.item())
                n_val += 1

        avg_val_loss = val_loss_total / max(1, n_val)
        history.append({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.heads.state_dict().items()}

    if best_state is not None:
        model.heads.load_state_dict(best_state)

    return model, history
