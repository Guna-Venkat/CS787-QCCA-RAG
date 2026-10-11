"""Experiment orchestration for Week 5 Content-Aware Adaptive Representation Allocation.

Implements:
- Full evaluation of frozen reference baselines:
    - Static k in {2, 4, 5, 6, 8}
    - Quality Oracle (strictly max F1)
    - Cost-Aware Epsilon Oracle (eps=0.01)
    - Week 4 Best Allocator (Balanced LogReg reference)
- Orchestration of the 8 main experimental cells:
    - I1-T1, I1-T2 (Query Only)
    - I2-T1, I2-T2 (Query + W4 Features)
    - I3-T1, I3-T2 (Query + Top-10 Passages)
    - I4-T1, I4-T2 (Query + Top-10 Passages + W4 Features)
- Comprehensive comparative table formatting (Main Results, Information Ablation, Target Comparison)
- Paper-clustered bootstrap hypothesis testing against static k=8 and W4 baselines
- Export of predictions and structured tables to results/week5/content_allocator/
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.week5.data import (
    ContentAllocatorDataset,
    build_w5_dataset_bundle,
    compute_truncation_statistics,
    K_CLASSES,
)
from src.week5.evaluation import (
    compute_paper_clustered_bootstrap,
    evaluate_allocation_policy,
    evaluate_classification_diagnostics,
    evaluate_regression_diagnostics,
    STATIC_K8_F1_REF,
    ORACLE_F1_REF,
    STATIC_K8_TOKENS_REF,
)
from src.week5.models import ContentAwareAllocator
from src.week5.training import set_seed, train_allocator

logger = logging.getLogger(__name__)


def evaluate_static_baselines(dev_bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compute exact policy metrics for static k policies and oracles on DEV."""
    df_sup = dev_bundle["supervision_df"]
    dev_tokens_map = dev_bundle.get("dev_tokens_map")
    n_dev = len(df_sup)

    rows: List[Dict[str, Any]] = []

    # 1. Static k in {2, 4, 5, 6, 8}
    for k in K_CLASSES:
        preds = [k] * n_dev
        res = evaluate_allocation_policy(df_sup, preds, dev_tokens_map)
        rows.append({
            "model_name": f"Static k={k}",
            "condition": "Static Baseline",
            "target": "N/A",
            "mean_answer_f1": res["mean_answer_f1"],
            "mean_selected_k": res["mean_selected_k"],
            "mean_context_tokens": res["mean_context_tokens"],
            "token_reduction_pct_vs_k8": res["token_reduction_pct_vs_k8"],
            "mean_oracle_regret": res["mean_oracle_regret"],
            "recovered_oracle_headroom": res["recovered_oracle_headroom"],
            "zero_regret_pct": res["zero_regret_pct"],
        })

    # 2. Week 4 Best Allocator (Balanced LogReg) Reference Policy
    # Known frozen Week 4 DEV reference performance: F1=0.3842, tokens=1115.9, mean_k=4.82
    # In W4, balanced LogReg achieved F1=0.3842 on DEV
    w4_f1 = 0.3842
    w4_tokens = 1115.9
    w4_k = 4.82
    w4_regret = float(np.mean(df_sup["best_f1"] - w4_f1))
    w4_red = round((STATIC_K8_TOKENS_REF - w4_tokens) / STATIC_K8_TOKENS_REF * 100.0, 2)
    w4_roh = round((w4_f1 - STATIC_K8_F1_REF) / (ORACLE_F1_REF - STATIC_K8_F1_REF), 4)

    rows.append({
        "model_name": "W4 Best Allocator (Balanced LogReg)",
        "condition": "Handcrafted Features Only",
        "target": "Classification",
        "mean_answer_f1": w4_f1,
        "mean_selected_k": w4_k,
        "mean_context_tokens": w4_tokens,
        "token_reduction_pct_vs_k8": w4_red,
        "mean_oracle_regret": round(w4_regret, 4),
        "recovered_oracle_headroom": w4_roh,
        "zero_regret_pct": 52.38,
    })

    # 3. Quality Oracle (strictly best F1)
    quality_preds = []
    for _, r in df_sup.iterrows():
        b_f1 = float(r["best_f1"])
        # smallest k with exact max F1
        best_ks = [k for k in K_CLASSES if abs(float(r[f"f1_k{k}"]) - b_f1) < 1e-6]
        quality_preds.append(min(best_ks))
    q_res = evaluate_allocation_policy(df_sup, quality_preds, dev_tokens_map)
    rows.append({
        "model_name": "Quality Oracle (Upper Bound)",
        "condition": "Theoretical Upper Bound",
        "target": "Oracle max F1",
        "mean_answer_f1": q_res["mean_answer_f1"],
        "mean_selected_k": q_res["mean_selected_k"],
        "mean_context_tokens": q_res["mean_context_tokens"],
        "token_reduction_pct_vs_k8": q_res["token_reduction_pct_vs_k8"],
        "mean_oracle_regret": 0.0000,
        "recovered_oracle_headroom": 1.0000,
        "zero_regret_pct": 100.0,
    })

    # 4. Cost-Aware Epsilon Oracle (eps=0.01)
    eps_preds = [int(r["oracle_k_eps001"]) for _, r in df_sup.iterrows()]
    eps_res = evaluate_allocation_policy(df_sup, eps_preds, dev_tokens_map)
    rows.append({
        "model_name": "Cost-Aware Oracle (eps=0.01)",
        "condition": "Theoretical Cost-Aware Target",
        "target": "Oracle eps=0.01",
        "mean_answer_f1": eps_res["mean_answer_f1"],
        "mean_selected_k": eps_res["mean_selected_k"],
        "mean_context_tokens": eps_res["mean_context_tokens"],
        "token_reduction_pct_vs_k8": eps_res["token_reduction_pct_vs_k8"],
        "mean_oracle_regret": eps_res["mean_oracle_regret"],
        "recovered_oracle_headroom": eps_res["recovered_oracle_headroom"],
        "zero_regret_pct": 100.0,
    })

    return rows


def run_single_experimental_cell(
    condition: str,  # I1, I2, I3, I4
    target: str,  # T1, T2
    train_bundle: Dict[str, Any],
    dev_bundle: Dict[str, Any],
    tokenizer: Any,
    config: Dict[str, Any],
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train and evaluate a single (condition, target) cell."""
    set_seed(seed)
    max_seq_length = config.get("model", {}).get("max_seq_length", 256)
    base_encoder_name = config.get("model", {}).get("base_encoder_name", "sentence-transformers/all-MiniLM-L6-v2")
    encoder_train_mode = config.get("model", {}).get("encoder_train_mode", "frozen")
    d_model = config.get("model", {}).get("d_model", 384)
    nhead = config.get("model", {}).get("nhead", 4)
    num_layers = config.get("model", {}).get("num_layers", 2)
    dim_feedforward = config.get("model", {}).get("dim_feedforward", 1024)
    dropout = config.get("model", {}).get("dropout", 0.1)

    t_cfg = config.get("training", {})
    epochs = t_cfg.get("max_epochs", 25)
    lr = t_cfg.get("lr", 1e-3)
    weight_decay = t_cfg.get("weight_decay", 1e-2)
    patience = t_cfg.get("patience", 4)
    grad_clip = t_cfg.get("grad_clip", 1.0)
    batch_size = t_cfg.get("batch_size", 32)
    epsilon = config.get("experiment", {}).get("epsilon", 0.01)

    # 1. Datasets and Loaders
    train_ds = ContentAllocatorDataset(
        bundle=train_bundle,
        tokenizer=tokenizer,
        condition=condition,
        max_seq_length=max_seq_length,
    )
    dev_ds = ContentAllocatorDataset(
        bundle=dev_bundle,
        tokenizer=tokenizer,
        condition=condition,
        max_seq_length=max_seq_length,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # 2. Instantiate Model
    model = ContentAwareAllocator(
        base_encoder_name=base_encoder_name,
        condition=condition,
        target_mode=target,
        encoder_train_mode=encoder_train_mode,
        d_model=d_model,
        feature_dim=6,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )

    # 3. Train
    train_res = train_allocator(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        dev_bundle=dev_bundle,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        grad_clip=grad_clip,
        device=device,
        use_class_weights=True,
        train_t1_labels=train_bundle["t1_labels"],
        epsilon=epsilon,
        verbose=verbose,
    )

    # 4. Final Inference and Policy Evaluation on DEV
    model.eval()
    dev_k_preds: List[int] = []
    dev_raw_outputs: List[np.ndarray] = []

    with torch.no_grad():
        for batch in dev_loader:
            k_batch = model.predict_k(batch, epsilon=epsilon)
            dev_k_preds.extend(k_batch.tolist())
            out = model(batch)
            if target == "T1":
                dev_raw_outputs.append(out["logits"].cpu().numpy())
            else:
                dev_raw_outputs.append(out["pred_regret"].cpu().numpy())

    dev_raw_arr = np.concatenate(dev_raw_outputs, axis=0) if dev_raw_outputs else np.array([])

    policy_metrics = evaluate_allocation_policy(
        df_supervision=dev_bundle["supervision_df"],
        predicted_k_series=dev_k_preds,
        dev_tokens_map=dev_bundle.get("dev_tokens_map"),
    )

    # Secondary diagnostics
    diagnostics: Dict[str, Any] = {}
    if target == "T1":
        y_true = dev_bundle["t1_labels"]
        y_pred = np.array([K_CLASSES.index(k) for k in dev_k_preds])
        diagnostics = evaluate_classification_diagnostics(y_true, y_pred)
    elif target == "T2":
        y_true_regrets = dev_bundle["t2_regrets"]
        diagnostics = evaluate_regression_diagnostics(y_true_regrets, dev_raw_arr)

    param_counts = model.get_parameter_counts()

    return {
        "condition": condition,
        "target": target,
        "seed": seed,
        "model": model,
        "best_epoch": train_res["best_epoch"],
        "history": train_res["history"],
        "policy_metrics": policy_metrics,
        "diagnostics": diagnostics,
        "parameter_counts": param_counts,
        "dev_k_preds": np.array(dev_k_preds),
        "dev_raw_outputs": dev_raw_arr,
    }
