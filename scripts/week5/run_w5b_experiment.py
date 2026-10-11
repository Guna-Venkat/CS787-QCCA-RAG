"""Full execution script for Week 5-B: Corrected Content-Aware Allocation via Multi-Label Epsilon-Admissibility.

Executes:
- Paper-disjoint TRAIN-FIT / INTERNAL-VAL partitioning
- MiniLM frozen semantic representation pre-extraction and caching
- 4 Information Conditions (I1, I2, I3, I4) x 3 Seeds (42, 123, 2026) = 12 models
- T3 Multi-label BCEWithLogits training with macro AUPRC early stopping on INTERNAL-VAL
- Strict INTERNAL-VAL probability threshold tau selection (tau in {0.30, 0.40, 0.50, 0.60, 0.70})
- Single evaluation on frozen original DEV (231 questions)
- Full policy metrics, multi-label classification metrics, representation health checks
- Paper-clustered bootstrap (2000 replicates)
- Automated generation of all required tables and figures
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from src.week5.data import (
    FEATURE_COLS,
    K_CLASSES,
    NUM_CLASSES,
    build_w5b_splits_and_bundles,
)
from src.week5.evaluation import (
    DEV_TOKEN_COSTS_DEFAULT,
    ORACLE_F1_REF,
    STATIC_K8_F1_REF,
    STATIC_K8_TOKENS_REF,
    compute_paper_clustered_bootstrap,
    evaluate_allocation_policy,
    evaluate_multilabel_diagnostics,
    select_tau_internal_val,
)
from src.week5.models import (
    ContentAwareAllocator,
    PassageTransformerAggregator,
    mean_pooling,
)
from src.week5.training import set_seed, train_allocator_t3

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 123, 2026]
CONDITIONS = ["I1", "I2", "I3", "I4"]
TAU_GRID = [0.30, 0.40, 0.50, 0.60, 0.70]

import subprocess
import time
from datetime import datetime

BASE_DIR = Path("results/week5/content_allocator/w5b")
SPLITS_DIR = BASE_DIR / "splits"
MODELS_DIR = BASE_DIR / "models"
PREDS_DIR = BASE_DIR / "predictions"
TABLES_DIR = BASE_DIR / "tables"
FIGURES_DIR = BASE_DIR / "figures"
BOOTSTRAP_DIR = BASE_DIR / "bootstrap"
DIAG_DIR = BASE_DIR / "diagnostics"
CACHE_DIR = BASE_DIR / "cache"

for d in [SPLITS_DIR, MODELS_DIR, PREDS_DIR, TABLES_DIR, FIGURES_DIR, BOOTSTRAP_DIR, DIAG_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ARTIFACT_TRACKER_PATH = Path("/home/gunavenkat/.gemini/antigravity-ide/brain/229eda70-9d43-4379-99a1-29c9b26b9b5e/w5b_live_tracker.md")
LOCAL_TRACKER_PATH = BASE_DIR / "w5b_live_tracker.md"


def get_gpu_status() -> str:
    try:
        res = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw", "--format=csv,noheader"],
            universal_newlines=True,
        ).strip()
        parts = [p.strip() for p in res.split(",")]
        return f"{parts[0]} | Util: {parts[1]} | VRAM: {parts[2]} / {parts[3]} | Temp: {parts[4]}°C | Power: {parts[5]}"
    except Exception as e:
        return f"N/A ({e})"


def update_live_tracker(
    completed_runs: List[Dict[str, Any]],
    total_runs: int = 12,
    current_action: str = "",
    start_time: float = 0.0,
    status: str = "IN_PROGRESS",
):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
    n_done = len(completed_runs)
    elapsed_sec = time.time() - start_time if start_time > 0 else 0
    if n_done > 0 and n_done < total_runs:
        sec_per_run = elapsed_sec / n_done
        eta_sec = sec_per_run * (total_runs - n_done)
        eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60)}s"
    elif n_done >= total_runs:
        eta_str = "0s (Complete)"
    else:
        eta_str = "Calculating..."

    gpu_info = get_gpu_status()
    pct_done = (n_done / total_runs) * 100.0 if total_runs > 0 else 0.0

    lines = [
        "# Week 5-B Content-Aware Allocator Live Tracking\n",
        f"**Status:** {'🏁 **COMPLETE**' if status == 'COMPLETE' else '⏳ **IN PROGRESS**'} ({n_done} / {total_runs} Models Finished)  ",
        f"**Current Activity:** {current_action}  ",
        f"**Last Updated:** {now_str}  \n",
        "---\n",
        "## 1. Overall Progress\n",
        "| Metric | Target | Current Value | Status |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Models Completed** | {total_runs} | **{n_done} / {total_runs}** | **{pct_done:.1f}%** {'🎯' if n_done==total_runs else '⚡'} |",
        f"| **Elapsed Time** | — | {int(elapsed_sec // 60)}m {int(elapsed_sec % 60)}s | Active |",
        f"| **Estimated Remaining Time** | — | **{eta_str}** | — |",
        f"| **GPU Status** | NVIDIA TITAN RTX | {gpu_info} | Monitoring |",
        "\n---\n",
        "## 2. Model Execution Log\n",
        "| Condition | Seed | Selected $\\tau$ | DEV Policy $F_1$ | Context Tokens | Mean $k$ | Fallback Rate | Status |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    all_combos = [(c, s) for c in CONDITIONS for s in SEEDS]
    done_dict = {(r["condition"], r["seed"]): r for r in completed_runs}

    for c, s in all_combos:
        if (c, s) in done_dict:
            r = done_dict[(c, s)]
            lines.append(
                f"| ${c}$-T3 | {s} | {r['selected_tau']:.2f} | **{r['dev_answer_f1']:.4f}** | "
                f"{r['dev_context_tokens']:.1f} | {r['dev_mean_k']:.2f} | {r['fallback_rate']:.1f}% | ✅ Done |"
            )
        elif current_action.startswith(f"Training {c}-T3 (Seed {s})"):
            lines.append(f"| ${c}$-T3 | {s} | — | — | — | — | — | 🔄 Training... |")
        else:
            lines.append(f"| ${c}$-T3 | {s} | — | — | — | — | — | Queued |")

    lines.extend([
        "\n---\n",
        "## 3. Reference Baseline Anchors\n",
        "* **Static $k=8$:** $F_1 = 0.3950$ | 1,750.5 context tokens | Mean $k = 8.00$",
        "* **W4 Best Allocator (Balanced LogReg):** $F_1 = 0.3842$ | 1,115.9 context tokens | Mean $k = 4.82$",
        "* **Static $k=6$:** $F_1 = 0.3857$ | 1,351.1 context tokens",
        "* **Cost-Aware $\\epsilon=0.01$ Oracle:** $F_1 = 0.4894$ | 839.4 context tokens | Mean $k = 3.49$",
        "\n",
    ])

    md_content = "\n".join(lines)
    try:
        LOCAL_TRACKER_PATH.write_text(md_content)
        ARTIFACT_TRACKER_PATH.write_text(md_content)
    except Exception as e:
        logger.warning(f"Could not write live tracker: {e}")


class CachedEmbeddingDataset(Dataset):
    """Fast Dataset using pre-computed frozen encoder representations."""

    def __init__(
        self,
        bundle: Dict[str, Any],
        query_embs: torch.Tensor,  # (N, 384)
        passage_embs: torch.Tensor,  # (N, 10, 384)
        passage_masks: torch.Tensor,  # (N, 10)
        condition: str = "I3",
    ):
        self.bundle = bundle
        self.query_embs = query_embs
        self.passage_embs = passage_embs
        self.passage_masks = passage_masks
        self.condition = condition
        self.n_items = len(bundle["question_ids"])

    def __len__(self) -> int:
        return self.n_items

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            "question_id": self.bundle["question_ids"][idx],
            "paper_id": self.bundle["paper_ids"][idx],
            "t3_labels": torch.tensor(self.bundle["t3_labels"][idx], dtype=torch.float32),
            "condition": self.condition,
        }

        if self.condition in ["I1", "I2"]:
            item["query_embedding"] = self.query_embs[idx]
        if self.condition in ["I3", "I4"]:
            item["passage_embeddings"] = self.passage_embs[idx]
            item["passage_mask"] = self.passage_masks[idx]

        if self.condition in ["I2", "I4"]:
            item["features"] = torch.tensor(self.bundle["features"][idx], dtype=torch.float32)
        else:
            item["features"] = torch.zeros(len(FEATURE_COLS), dtype=torch.float32)

        return item


def extract_embeddings_for_bundle(
    bundle: Dict[str, Any],
    encoder: AutoModel,
    tokenizer: AutoTokenizer,
    cache_prefix: str,
    batch_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pre-extract query representations and (query, passage_i) representations."""
    q_cache = CACHE_DIR / f"{cache_prefix}_query_embs.pt"
    p_cache = CACHE_DIR / f"{cache_prefix}_passage_embs.pt"
    m_cache = CACHE_DIR / f"{cache_prefix}_passage_masks.pt"

    if q_cache.exists() and p_cache.exists() and m_cache.exists():
        logger.info(f"Loading cached embeddings from {CACHE_DIR} for {cache_prefix}")
        q_embs = torch.load(q_cache, map_location="cpu")
        p_embs = torch.load(p_cache, map_location="cpu")
        p_masks = torch.load(m_cache, map_location="cpu")
        return q_embs, p_embs, p_masks

    logger.info(f"Pre-extracting embeddings for {cache_prefix} ({len(bundle['question_ids'])} queries)...")
    encoder.eval()
    encoder.to(DEVICE)

    all_q_embs: List[torch.Tensor] = []
    all_p_embs: List[torch.Tensor] = []
    all_p_masks: List[torch.Tensor] = []

    questions = bundle["questions"]
    all_passages = bundle["passages"]

    with torch.no_grad():
        # 1. Query embeddings
        for i in range(0, len(questions), batch_size):
            q_batch = questions[i : i + batch_size]
            q_enc = tokenizer(
                q_batch,
                max_length=256,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(DEVICE)
            out = encoder(input_ids=q_enc["input_ids"], attention_mask=q_enc["attention_mask"])
            pooled = mean_pooling(out.last_hidden_state, q_enc["attention_mask"]).cpu()
            all_q_embs.append(pooled)

        q_embs = torch.cat(all_q_embs, dim=0)

        # 2. Passage pair embeddings
        for i in range(len(questions)):
            q_text = questions[i]
            passages = all_passages[i]
            p_texts = [p.get("text", "") for p in passages[:10]]
            while len(p_texts) < 10:
                p_texts.append("")
            p_texts = p_texts[:10]

            p_mask = torch.tensor([len(t.strip()) > 0 for t in p_texts], dtype=torch.bool)
            if not p_mask.any():
                p_mask[0] = True
            all_p_masks.append(p_mask.unsqueeze(0))

            q_rep = [q_text] * 10
            pairs_enc = tokenizer(
                q_rep,
                p_texts,
                max_length=256,
                truncation="only_second",
                padding=True,
                return_tensors="pt",
            ).to(DEVICE)

            out = encoder(input_ids=pairs_enc["input_ids"], attention_mask=pairs_enc["attention_mask"])
            pooled_passages = mean_pooling(out.last_hidden_state, pairs_enc["attention_mask"]).cpu()
            all_p_embs.append(pooled_passages.unsqueeze(0))  # (1, 10, 384)

        p_embs = torch.cat(all_p_embs, dim=0)  # (N, 10, 384)
        p_masks = torch.cat(all_p_masks, dim=0)  # (N, 10)

    torch.save(q_embs, q_cache)
    torch.save(p_embs, p_cache)
    torch.save(p_masks, m_cache)
    logger.info(f"Saved pre-extracted embeddings to {CACHE_DIR}")
    return q_embs, p_embs, p_masks


def run_experiment():
    logger.info("================================================================")
    logger.info("STARTING WEEK 5-B: MULTI-LABEL EPSILON-ADMISSIBILITY ALLOCATION")
    logger.info("================================================================")

    # 1. Load splits and fit scaler strictly on TRAIN-FIT
    fit_bundle, val_bundle, dev_bundle, scaler = build_w5b_splits_and_bundles(
        save_manifests=True,
        manifest_dir=SPLITS_DIR,
    )

    logger.info(f"TRAIN-FIT: {len(fit_bundle['question_ids'])} queries, {len(set(fit_bundle['paper_ids']))} papers")
    logger.info(f"INTERNAL-VAL: {len(val_bundle['question_ids'])} queries, {len(set(val_bundle['paper_ids']))} papers")
    logger.info(f"DEV: {len(dev_bundle['question_ids'])} queries, {len(set(dev_bundle['paper_ids']))} papers")

    # Positive rates audit
    for k_idx, k in enumerate(K_CLASSES):
        fit_rate = np.mean(fit_bundle["t3_labels"][:, k_idx]) * 100
        val_rate = np.mean(val_bundle["t3_labels"][:, k_idx]) * 100
        dev_rate = np.mean(dev_bundle["t3_labels"][:, k_idx]) * 100
        logger.info(f"Target T3 Positive Rate k={k}: FIT={fit_rate:.1f}%, VAL={val_rate:.1f}%, DEV={dev_rate:.1f}%")

    # 2. Extract embeddings
    encoder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    fit_q_emb, fit_p_emb, fit_p_mask = extract_embeddings_for_bundle(fit_bundle, encoder, tokenizer, "train_fit")
    val_q_emb, val_p_emb, val_p_mask = extract_embeddings_for_bundle(val_bundle, encoder, tokenizer, "internal_val")
    dev_q_emb, dev_p_emb, dev_p_mask = extract_embeddings_for_bundle(dev_bundle, encoder, tokenizer, "dev")

    # Compute baseline semantic norm
    mean_semantic_norm = float(fit_p_emb.norm(dim=-1).mean().item())
    logger.info(f"Measured Frozen MiniLM Semantic Vector Mean Norm: {mean_semantic_norm:.4f}")

    # Track health checks
    rep_health: Dict[str, Any] = {
        "mean_semantic_norm_base": round(mean_semantic_norm, 4),
        "conditions": {},
    }

    # Data structures for results
    seed_records: List[Dict[str, Any]] = []
    tau_records: List[Dict[str, Any]] = []
    multilabel_records: List[Dict[str, Any]] = []
    dev_predictions_dict: Dict[str, Any] = {
        "question_id": dev_bundle["question_ids"],
        "paper_id": dev_bundle["paper_ids"],
        "oracle_k": dev_bundle["supervision_df"]["oracle_k_eps001"].tolist(),
        "best_f1": dev_bundle["supervision_df"]["best_f1"].tolist(),
    }
    for k in K_CLASSES:
        dev_predictions_dict[f"f1_k{k}"] = dev_bundle["supervision_df"][f"f1_k{k}"].tolist()
        dev_predictions_dict[f"true_z_k{k}"] = dev_bundle["t3_labels"][:, K_CLASSES.index(k)].tolist()

    # Model training across conditions and seeds
    trained_models: Dict[Tuple[str, int], ContentAwareAllocator] = {}
    selected_taus: Dict[Tuple[str, int], float] = {}
    dev_policy_results: Dict[Tuple[str, int], Dict[str, Any]] = {}
    t0 = time.time()
    update_live_tracker(seed_records, total_runs=12, current_action="Starting model runs", start_time=t0)

    for cond in CONDITIONS:
        rep_health["conditions"][cond] = {}
        for seed in SEEDS:
            logger.info(f"\n--- Training Condition {cond}-T3 | Seed {seed} ---")
            update_live_tracker(seed_records, total_runs=12, current_action=f"Training {cond}-T3 (Seed {seed})", start_time=t0)
            set_seed(seed)

            # Build fast DataLoaders
            train_ds = CachedEmbeddingDataset(fit_bundle, fit_q_emb, fit_p_emb, fit_p_mask, condition=cond)
            val_ds = CachedEmbeddingDataset(val_bundle, val_q_emb, val_p_emb, val_p_mask, condition=cond)
            dev_ds = CachedEmbeddingDataset(dev_bundle, dev_q_emb, dev_p_emb, dev_p_mask, condition=cond)

            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
            dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=False)

            model = ContentAwareAllocator(
                condition=cond,
                target_mode="T3",
                encoder_train_mode="frozen",
                d_model=384,
                dropout=0.1,
            )

            # Record norms before training for I3, I4
            rank_norm_before = 0.0
            ratio_before = 0.0
            if cond in ["I3", "I4"]:
                rank_norm_before = float(model.aggregator.rank_embeddings.weight.norm(dim=-1).mean().item())
                ratio_before = rank_norm_before / mean_semantic_norm
                logger.info(f"[{cond} seed {seed}] Before: Rank Norm = {rank_norm_before:.4f}, Ratio = {ratio_before:.4f}")

            # Train
            train_res = train_allocator_t3(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=20,
                lr=1e-3,
                weight_decay=1e-2,
                patience=4,
                grad_clip=1.0,
                device=DEVICE,
                verbose=False,
            )

            best_model = train_res["model"]
            best_epoch = train_res["best_epoch"]
            best_val_auprc = train_res["best_macro_auprc"]

            # Save checkpoint
            model_save_path = MODELS_DIR / f"{cond}_seed_{seed}.pt"
            torch.save(best_model.state_dict(), model_save_path)
            trained_models[(cond, seed)] = best_model

            # Measure post-training norms
            rank_norm_after = 0.0
            ratio_after = 0.0
            if cond in ["I3", "I4"]:
                rank_norm_after = float(best_model.aggregator.rank_embeddings.weight.norm(dim=-1).mean().item())
                ratio_after = rank_norm_after / mean_semantic_norm
                logger.info(f"[{cond} seed {seed}] After: Rank Norm = {rank_norm_after:.4f}, Ratio = {ratio_after:.4f}")

            # Collect INTERNAL-VAL probabilities for tau selection
            best_model.eval()
            val_probs_list = []
            with torch.no_grad():
                for batch in val_loader:
                    p = best_model.predict_admissibility_probs(batch)
                    val_probs_list.append(p)
            val_probs = np.concatenate(val_probs_list, axis=0)

            # Frozen tau selection on INTERNAL-VAL ONLY!
            tau_res = select_tau_internal_val(
                val_supervision_df=val_bundle["supervision_df"],
                val_pred_probs=val_probs,
                val_tokens_map=val_bundle.get("tokens_map"),
                tau_grid=TAU_GRID,
                quality_tolerance=0.005,
            )
            chosen_tau = tau_res["selected_tau"]
            selected_taus[(cond, seed)] = chosen_tau
            logger.info(f"[{cond} seed {seed}] Selected Tau = {chosen_tau:.2f} (Constraint Met: {tau_res['constraint_met']})")

            for gr in tau_res["grid_results"]:
                tau_records.append({
                    "condition": cond,
                    "seed": seed,
                    "tau": gr["tau"],
                    "val_answer_f1": gr["mean_answer_f1"],
                    "val_tokens": gr["mean_context_tokens"],
                    "val_selected_k": gr["mean_selected_k"],
                    "val_oracle_regret": gr["mean_oracle_regret"],
                    "fallback_rate": gr["fallback_rate"],
                    "zero_regret_pct": gr["zero_regret_pct"],
                    "is_selected": gr["tau"] == chosen_tau,
                })

            # NOW: Evaluate ONCE on original DEV (231 questions)
            dev_probs_list = []
            dev_logits_list = []
            dev_preds_k = []
            fallbacks_dev = []

            with torch.no_grad():
                for batch in dev_loader:
                    out = best_model(batch)
                    logits = out["admissibility_logits"].cpu().numpy()
                    probs = torch.sigmoid(out["admissibility_logits"]).cpu().numpy()
                    dev_logits_list.append(logits)
                    dev_probs_list.append(probs)

                    # Predict k using chosen tau
                    for row in probs:
                        valid_ks = [K_CLASSES[idx] for idx, pr in enumerate(row) if pr >= chosen_tau]
                        if valid_ks:
                            dev_preds_k.append(min(valid_ks))
                            fallbacks_dev.append(0.0)
                        else:
                            dev_preds_k.append(8)
                            fallbacks_dev.append(1.0)

            dev_probs = np.concatenate(dev_probs_list, axis=0)
            dev_logits = np.concatenate(dev_logits_list, axis=0)
            dev_fallback_rate = float(np.mean(fallbacks_dev) * 100.0)

            # Logit variance
            dev_logit_var = float(np.mean(np.var(dev_logits, axis=0)))

            if cond in ["I3", "I4"]:
                rep_health["conditions"][cond][str(seed)] = {
                    "rank_norm_before": round(rank_norm_before, 4),
                    "rank_norm_after": round(rank_norm_after, 4),
                    "norm_ratio_before": round(ratio_before, 4),
                    "norm_ratio_after": round(ratio_after, 4),
                    "dev_logit_variance": round(dev_logit_var, 6),
                }

            # Evaluate DEV policy through frozen response surface
            policy_eval = evaluate_allocation_policy(
                df_supervision=dev_bundle["supervision_df"],
                predicted_k_series=dev_preds_k,
                dev_tokens_map=dev_bundle.get("dev_tokens_map"),
            )
            dev_policy_results[(cond, seed)] = policy_eval

            # Selected k distribution on DEV
            k_counts = {f"pct_k{k}": round(float(np.mean(np.array(dev_preds_k) == k) * 100.0), 2) for k in K_CLASSES}

            # Multi-label classification diagnostics on DEV
            ml_diag = evaluate_multilabel_diagnostics(dev_bundle["t3_labels"], dev_probs, tau=chosen_tau)

            multilabel_records.append({
                "condition": cond,
                "seed": seed,
                "macro_auroc": ml_diag["macro_auroc"],
                "macro_auprc": ml_diag["macro_auprc"],
                "micro_auprc": ml_diag["micro_auprc"],
                "brier_score": ml_diag["overall_brier"],
                **{f"auroc_k{k}": ml_diag["per_budget_auroc"][f"k{k}"] for k in K_CLASSES},
                **{f"auprc_k{k}": ml_diag["per_budget_auprc"][f"k{k}"] for k in K_CLASSES},
            })

            seed_records.append({
                "condition": cond,
                "seed": seed,
                "selected_tau": chosen_tau,
                "best_epoch": best_epoch,
                "val_macro_auprc": round(best_val_auprc, 4),
                "dev_answer_f1": policy_eval["mean_answer_f1"],
                "dev_mean_k": policy_eval["mean_selected_k"],
                "dev_context_tokens": policy_eval["mean_context_tokens"],
                "token_reduction_pct": policy_eval["token_reduction_pct_vs_k8"],
                "mean_oracle_regret": policy_eval["mean_oracle_regret"],
                "zero_regret_pct": policy_eval["zero_regret_pct"],
                "fallback_rate": round(dev_fallback_rate, 2),
                "recovered_oracle_headroom": policy_eval["recovered_oracle_headroom"],
                "oracle_accuracy": policy_eval["oracle_budget_accuracy"],
                "dev_macro_auprc": ml_diag["macro_auprc"],
                "dev_macro_auroc": ml_diag["macro_auroc"],
                "dev_brier": ml_diag["overall_brier"],
                "dev_logit_variance": round(dev_logit_var, 6),
                **k_counts,
            })

            # Store predictions
            dev_predictions_dict[f"pred_k_{cond}_seed{seed}"] = dev_preds_k
            for idx, k in enumerate(K_CLASSES):
                dev_predictions_dict[f"prob_k{k}_{cond}_seed{seed}"] = dev_probs[:, idx].tolist()

            logger.info(
                f"[{cond} seed {seed} DEV Results] F1: {policy_eval['mean_answer_f1']:.4f} | "
                f"Tokens: {policy_eval['mean_context_tokens']:.1f} | Mean k: {policy_eval['mean_selected_k']:.2f} | "
                f"Fallback: {dev_fallback_rate:.1f}% | ROH: {policy_eval['recovered_oracle_headroom']:.4f}"
            )
            update_live_tracker(seed_records, total_runs=12, current_action=f"Completed {cond}-T3 (Seed {seed})", start_time=t0)

    # 3. Save predictions
    df_preds = pd.DataFrame(dev_predictions_dict)
    df_preds.to_csv(PREDS_DIR / "dev_predictions_t3.csv", index=False)
    logger.info(f"Saved DEV predictions to {PREDS_DIR / 'dev_predictions_t3.csv'}")

    # 4. Save seed records and tau selection records
    df_seed_res = pd.DataFrame(seed_records)
    df_seed_res.to_csv(TABLES_DIR / "w5b_seed_results.csv", index=False)

    df_tau_res = pd.DataFrame(tau_records)
    df_tau_res.to_csv(TABLES_DIR / "w5b_tau_selection.csv", index=False)

    df_ml_res = pd.DataFrame(multilabel_records)
    df_ml_res.to_csv(TABLES_DIR / "w5b_multilabel_metrics.csv", index=False)

    # 5. Build Aggregated Main Results Table
    main_rows: List[Dict[str, Any]] = []

    # Baselines
    baselines_static = [
        ("Static k=2", 0.3038, 2.00, 526.5, 69.92, 0.1857, 100.0 * (1750.5 - 526.5) / 1750.5, (0.3038 - 0.3950) / 0.0945),
        ("Static k=4", 0.3596, 4.00, 938.1, 46.41, 0.1299, 100.0 * (1750.5 - 938.1) / 1750.5, (0.3596 - 0.3950) / 0.0945),
        ("Static k=5", 0.3629, 5.00, 1149.0, 34.36, 0.1266, 100.0 * (1750.5 - 1149.0) / 1750.5, (0.3629 - 0.3950) / 0.0945),
        ("Static k=6", 0.3857, 6.00, 1351.1, 22.82, 0.1038, 100.0 * (1750.5 - 1351.1) / 1750.5, (0.3857 - 0.3950) / 0.0945),
        ("Best Static SARA Baseline (k=8)", 0.3950, 8.00, 1750.5, 0.0, 0.0945, 0.0, 0.0),
        ("W4 Balanced Logistic Regression", 0.3842, 4.82, 1115.9, 36.25, 0.1053, 36.25, (0.3842 - 0.3950) / 0.0945),
        ("Cost-Aware Epsilon Oracle (eps=0.01)", 0.4894, 3.49, 839.4, 52.05, 0.0001, 52.05, 0.9989),
        ("Quality Oracle", 0.4895, 3.53, 1021.5, 41.65, 0.0, 41.65, 1.0000),
    ]

    for name, f1, mk, tok, red, reg, tok_red, roh in baselines_static:
        main_rows.append({
            "model_family": "Baseline",
            "condition": name,
            "mean_answer_f1": round(f1, 4),
            "std_answer_f1": 0.0,
            "mean_selected_k": round(mk, 2),
            "std_selected_k": 0.0,
            "mean_context_tokens": round(tok, 1),
            "std_context_tokens": 0.0,
            "token_reduction_pct": round(tok_red, 2),
            "mean_oracle_regret": round(reg, 4),
            "recovered_oracle_headroom": round(roh, 4),
            "fallback_rate_pct": 0.0,
        })

    # W5-A Reference Baselines
    w5a_refs = [
        ("W5-A I1-T1 (Query Direct)", 0.3709, 4.69, 1076.1, 38.5, 0.1186, -0.2550),
        ("W5-A I1-T2 (Query Regret)", 0.3950, 8.00, 1750.5, 0.0, 0.0945, 0.0),
        ("W5-A I2-T1 (Query+W4 Direct)", 0.3693, 5.13, 1173.0, 33.0, 0.1202, -0.2720),
        ("W5-A I2-T2 (Query+W4 Regret)", 0.3952, 7.85, 1720.4, 1.7, 0.0943, 0.0021),
        ("W5-A I3-T1 (Query+Passages Direct)", 0.3950, 8.00, 1750.5, 0.0, 0.0945, 0.0),
        ("W5-A I3-T2 (Query+Passages Regret)", 0.3950, 8.00, 1750.5, 0.0, 0.0945, 0.0),
        ("W5-A I4-T1 (Query+Passages+W4 Direct)", 0.3630, 5.05, 1154.1, 34.1, 0.1265, -0.3386),
        ("W5-A I4-T2 (Query+Passages+W4 Regret)", 0.3950, 8.00, 1750.5, 0.0, 0.0945, 0.0),
    ]

    for name, f1, mk, tok, red, reg, roh in w5a_refs:
        main_rows.append({
            "model_family": "W5-A (Historical)",
            "condition": name,
            "mean_answer_f1": round(f1, 4),
            "std_answer_f1": 0.0,
            "mean_selected_k": round(mk, 2),
            "std_selected_k": 0.0,
            "mean_context_tokens": round(tok, 1),
            "std_context_tokens": 0.0,
            "token_reduction_pct": round(red, 2),
            "mean_oracle_regret": round(reg, 4),
            "recovered_oracle_headroom": round(roh, 4),
            "fallback_rate_pct": 100.0 if "Regret" in name and mk >= 7.9 else 0.0,
        })

    # W5-B Conditions (Mean ± Std over 3 seeds)
    for cond in CONDITIONS:
        sub = df_seed_res[df_seed_res["condition"] == cond]
        main_rows.append({
            "model_family": "W5-B (T3 Multi-Label)",
            "condition": f"{cond}-T3",
            "mean_answer_f1": round(float(sub["dev_answer_f1"].mean()), 4),
            "std_answer_f1": round(float(sub["dev_answer_f1"].std()), 4),
            "mean_selected_k": round(float(sub["dev_mean_k"].mean()), 2),
            "std_selected_k": round(float(sub["dev_mean_k"].std()), 2),
            "mean_context_tokens": round(float(sub["dev_context_tokens"].mean()), 1),
            "std_context_tokens": round(float(sub["dev_context_tokens"].std()), 1),
            "token_reduction_pct": round(float(sub["token_reduction_pct"].mean()), 2),
            "mean_oracle_regret": round(float(sub["mean_oracle_regret"].mean()), 4),
            "recovered_oracle_headroom": round(float(sub["recovered_oracle_headroom"].mean()), 4),
            "fallback_rate_pct": round(float(sub["fallback_rate"].mean()), 2),
        })

    df_main_res = pd.DataFrame(main_rows)
    df_main_res.to_csv(TABLES_DIR / "w5b_main_results.csv", index=False)
    logger.info(f"Saved main results table to {TABLES_DIR / 'w5b_main_results.csv'}")

    # 6. W5-A vs W5-B Comparison Table
    w5a_vs_w5b_rows = []
    for cond in CONDITIONS:
        sub_b = df_seed_res[df_seed_res["condition"] == cond]
        f1_b = float(sub_b["dev_answer_f1"].mean())
        tok_b = float(sub_b["dev_context_tokens"].mean())
        k_b = float(sub_b["dev_mean_k"].mean())
        fb_b = float(sub_b["fallback_rate"].mean())

        # Match W5-A T1 and T2
        t1_ref = next(r for r in w5a_refs if f"{cond}-T1" in r[0])
        t2_ref = next(r for r in w5a_refs if f"{cond}-T2" in r[0])

        w5a_vs_w5b_rows.append({
            "condition": cond,
            "w5a_t1_f1": t1_ref[1],
            "w5a_t1_tokens": t1_ref[3],
            "w5a_t1_meank": t1_ref[2],
            "w5a_t2_f1": t2_ref[1],
            "w5a_t2_tokens": t2_ref[3],
            "w5a_t2_meank": t2_ref[2],
            "w5b_t3_f1": round(f1_b, 4),
            "w5b_t3_tokens": round(tok_b, 1),
            "w5b_t3_meank": round(k_b, 2),
            "w5b_t3_fallback_rate": round(fb_b, 2),
            "delta_f1_vs_w5a_t1": round(f1_b - t1_ref[1], 4),
            "delta_tokens_vs_w5a_t1": round(tok_b - t1_ref[3], 1),
            "delta_tokens_vs_w5a_t2": round(tok_b - t2_ref[3], 1),
        })

    df_comp = pd.DataFrame(w5a_vs_w5b_rows)
    df_comp.to_csv(TABLES_DIR / "w5a_vs_w5b_comparison.csv", index=False)
    logger.info(f"Saved W5-A vs W5-B comparison to {TABLES_DIR / 'w5a_vs_w5b_comparison.csv'}")

    # 7. Paper-Clustered Bootstrap Comparisons (2000 replicates)
    # Using Seed 42 predictions as canonical seed for statistical tests
    logger.info("Running paper-clustered bootstrap comparisons (2000 replicates)...")
    update_live_tracker(seed_records, total_runs=12, current_action="Running paper-clustered bootstrap (2000 replicates)", start_time=t0)
    paper_ids = dev_bundle["paper_ids"]

    # Baselines vectors
    w4_df = pd.read_csv("results/week5/supervision/dev_allocation_supervision.csv")
    # For static k=8
    k8_eval = evaluate_allocation_policy(w4_df, [8] * len(w4_df), dev_bundle.get("dev_tokens_map"))
    k8_f1s = k8_eval["per_query_f1s"]
    k8_toks = k8_eval["per_query_tokens"]

    # Canonical W4 Balanced Logistic Regression predictions
    w4_oof = pd.read_csv("results/week4/model_benchmark/oof_predictions.csv")
    w4_oof["question_id"] = w4_oof["question_id"].astype(str)
    w4_col = "pred_t05_Balanced Logistic Regression (5 feats, Baseline)"
    # Merge with dev_bundle to preserve order
    dev_qids_df = pd.DataFrame({"question_id": dev_bundle["question_ids"]})
    w4_merged = pd.merge(dev_qids_df, w4_oof[["question_id", w4_col]], on="question_id", how="left")
    w4_preds_k = np.where(w4_merged[w4_col].values == 1, 6, 4)
    w4_eval = evaluate_allocation_policy(w4_df, w4_preds_k, dev_bundle.get("dev_tokens_map"))
    w4_f1s = w4_eval["per_query_f1s"]
    w4_toks = w4_eval["per_query_tokens"]

    i3_eval = dev_policy_results[("I3", 42)]
    i3_f1s = i3_eval["per_query_f1s"]
    i3_toks = i3_eval["per_query_tokens"]

    i4_eval = dev_policy_results[("I4", 42)]
    i4_f1s = i4_eval["per_query_f1s"]
    i4_toks = i4_eval["per_query_tokens"]

    bootstrap_comparisons = [
        ("I3-T3 vs W4", i3_f1s, w4_f1s, i3_toks, w4_toks),
        ("I3-T3 vs Static k=8", i3_f1s, k8_f1s, i3_toks, k8_toks),
        ("I4-T3 vs I3-T3", i4_f1s, i3_f1s, i4_toks, i3_toks),
        ("I4-T3 vs W4", i4_f1s, w4_f1s, i4_toks, w4_toks),
    ]

    boot_rows = []
    for comp_name, pol_f1, base_f1, pol_tok, base_tok in bootstrap_comparisons:
        b_res = compute_paper_clustered_bootstrap(
            paper_ids=paper_ids,
            policy_f1s=pol_f1,
            baseline_f1s=base_f1,
            policy_tokens=pol_tok,
            baseline_tokens=base_tok,
            n_bootstrap=2000,
            seed=42,
        )
        boot_rows.append({
            "comparison": comp_name,
            "observed_delta_f1": b_res["observed_delta_f1"],
            "ci_lower_95_f1": b_res["ci_lower_95"],
            "ci_upper_95_f1": b_res["ci_upper_95"],
            "p_value_f1": b_res["p_value"],
            "observed_delta_tokens": b_res["observed_delta_tokens"],
            "ci_lower_95_tokens": b_res["tokens_ci_lower_95"],
            "ci_upper_95_tokens": b_res["tokens_ci_upper_95"],
            "significant_f1": b_res["ci_lower_95"] > 0 or b_res["ci_upper_95"] < 0,
        })
        logger.info(
            f"Bootstrap [{comp_name}]: Delta F1 = {b_res['observed_delta_f1']:+.4f} "
            f"[{b_res['ci_lower_95']:+.4f}, {b_res['ci_upper_95']:+.4f}] (p={b_res['p_value']:.4f}) | "
            f"Delta Tokens = {b_res['observed_delta_tokens']:+.1f} [{b_res['tokens_ci_lower_95']:+.1f}, {b_res['tokens_ci_upper_95']:+.1f}]"
        )

    df_boot = pd.DataFrame(boot_rows)
    df_boot.to_csv(BOOTSTRAP_DIR / "w5b_bootstrap_comparisons.csv", index=False)
    logger.info(f"Saved bootstrap comparisons to {BOOTSTRAP_DIR / 'w5b_bootstrap_comparisons.csv'}")

    # 8. Save Representation Health Check
    with open(DIAG_DIR / "representation_health_check.json", "w") as f:
        json.dump(rep_health, f, indent=2)
    logger.info(f"Saved representation health check to {DIAG_DIR / 'representation_health_check.json'}")

    # 9. Generate Figures
    generate_w5b_plots(df_main_res, df_seed_res, df_comp, df_preds)

    update_live_tracker(seed_records, total_runs=12, current_action="All runs, bootstrap, and figures generated", start_time=t0, status="COMPLETE")
    logger.info("================================================================")
    logger.info("WEEK 5-B EXPERIMENT COMPLETED SUCCESSFULLY!")
    logger.info("================================================================")


def generate_w5b_plots(
    df_main: pd.DataFrame,
    df_seeds: pd.DataFrame,
    df_comp: pd.DataFrame,
    df_preds: pd.DataFrame,
):
    """Generate all 4 required figures for Week 5-B."""
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Pareto Frontier Plot
    plt.figure(figsize=(10, 7))
    # Filter baselines
    df_b = df_main[df_main["model_family"] == "Baseline"]
    plt.scatter(
        df_b["mean_context_tokens"],
        df_b["mean_answer_f1"],
        color="gray",
        marker="o",
        s=90,
        alpha=0.8,
        label="Static & Oracle Baselines",
    )
    for _, row in df_b.iterrows():
        plt.annotate(
            row["condition"],
            (row["mean_context_tokens"], row["mean_answer_f1"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
            color="#444",
        )

    # W5-B conditions
    df_w5b = df_main[df_main["model_family"] == "W5-B (T3 Multi-Label)"]
    palette = {"I1-T3": "#1f77b4", "I2-T3": "#2ca02c", "I3-T3": "#d62728", "I4-T3": "#9467bd"}

    for _, row in df_w5b.iterrows():
        cond = row["condition"]
        c = palette.get(cond, "blue")
        plt.errorbar(
            row["mean_context_tokens"],
            row["mean_answer_f1"],
            xerr=row["std_context_tokens"],
            yerr=row["std_answer_f1"],
            fmt="D",
            color=c,
            capsize=5,
            markersize=10,
            label=f"W5-B {cond}",
        )
        plt.annotate(
            f"{cond} ({row['mean_answer_f1']:.4f})",
            (row["mean_context_tokens"], row["mean_answer_f1"]),
            textcoords="offset points",
            xytext=(8, -12),
            fontweight="bold",
            fontsize=10,
            color=c,
        )

    plt.axhline(STATIC_K8_F1_REF, color="black", linestyle="--", alpha=0.5, label="Static k=8 F1 (0.3950)")
    plt.axvline(1115.9, color="green", linestyle=":", alpha=0.4, label="W4 Context Tokens (1115.9)")
    plt.xlabel("Mean Context Tokens (Cost)")
    plt.ylabel("Mean Answer F1 (Quality)")
    plt.title("Week 5-B: Quality-Cost Pareto Frontier (T3 Multi-Label Admissibility)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "w5b_pareto.png", dpi=300)
    plt.close()
    logger.info(f"Saved {FIGURES_DIR / 'w5b_pareto.png'}")

    # 2. Selected k Distribution Plot
    plt.figure(figsize=(10, 6))
    k_cols = [f"pct_k{k}" for k in K_CLASSES]
    df_k_mean = df_seeds.groupby("condition")[k_cols].mean().reset_index()

    df_k_melted = df_k_mean.melt(id_vars=["condition"], value_vars=k_cols, var_name="budget", value_name="percentage")
    df_k_melted["budget"] = df_k_melted["budget"].str.replace("pct_", "")

    sns.barplot(data=df_k_melted, x="condition", y="percentage", hue="budget", palette="Blues_r")
    plt.ylabel("Selected Budget Percentage (%)")
    plt.xlabel("Information Condition")
    plt.title("Week 5-B: Selected-k Distribution by Condition (Averaged across 3 Seeds)")
    plt.legend(title="Allocated Budget")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "w5b_selected_k_distribution.png", dpi=300)
    plt.close()
    logger.info(f"Saved {FIGURES_DIR / 'w5b_selected_k_distribution.png'}")

    # 3. Multi-label Precision-Recall Curves (Seed 42)
    from sklearn.metrics import precision_recall_curve
    plt.figure(figsize=(9, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, k in enumerate(K_CLASSES):
        y_true = df_preds[f"true_z_k{k}"].values
        # Using I3-T3 seed 42 probabilities
        y_score = df_preds[f"prob_k{k}_I3_seed42"].values
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        plt.plot(rec, prec, label=f"Budget k={k}", color=colors[idx], lw=2)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Week 5-B: Multi-Label Precision-Recall Curves (I3-T3 Seed 42)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "w5b_multilabel_pr_curves.png", dpi=300)
    plt.close()
    logger.info(f"Saved {FIGURES_DIR / 'w5b_multilabel_pr_curves.png'}")

    # 4. W5-A vs W5-B Comparison Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: F1 Comparison
    df_plot_f1 = pd.DataFrame({
        "Condition": CONDITIONS,
        "W5-A T1 (Direct)": df_comp["w5a_t1_f1"],
        "W5-A T2 (Regret)": df_comp["w5a_t2_f1"],
        "W5-B T3 (Multi-Label)": df_comp["w5b_t3_f1"],
    }).melt(id_vars=["Condition"], var_name="Formulation", value_name="Answer F1")

    sns.barplot(data=df_plot_f1, x="Condition", y="Answer F1", hue="Formulation", ax=axes[0], palette="Set2")
    axes[0].axhline(STATIC_K8_F1_REF, color="red", linestyle="--", alpha=0.6, label="Static k=8 (0.3950)")
    axes[0].set_title("Answer F1: W5-A vs W5-B")
    axes[0].set_ylim(0.30, 0.42)
    axes[0].legend(loc="lower left")

    # Panel B: Tokens Comparison
    df_plot_tok = pd.DataFrame({
        "Condition": CONDITIONS,
        "W5-A T1 (Direct)": df_comp["w5a_t1_tokens"],
        "W5-A T2 (Regret)": df_comp["w5a_t2_tokens"],
        "W5-B T3 (Multi-Label)": df_comp["w5b_t3_tokens"],
    }).melt(id_vars=["Condition"], var_name="Formulation", value_name="Context Tokens")

    sns.barplot(data=df_plot_tok, x="Condition", y="Context Tokens", hue="Formulation", ax=axes[1], palette="Set2")
    axes[1].axhline(1750.5, color="red", linestyle="--", alpha=0.6, label="Static k=8 (1750.5)")
    axes[1].axhline(1115.9, color="green", linestyle=":", alpha=0.6, label="W4 Baseline (1115.9)")
    axes[1].set_title("Context Tokens: W5-A vs W5-B")
    axes[1].legend(loc="lower left")

    plt.suptitle("Week 5-A vs Week 5-B Formulation Comparison across Conditions", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "w5a_vs_w5b.png", dpi=300)
    plt.close()
    logger.info(f"Saved {FIGURES_DIR / 'w5a_vs_w5b.png'}")


if __name__ == "__main__":
    run_experiment()
