"""Generator script to produce the canonical Week 5-C experiment notebook:
notebooks/week5/week5c_encoder_adaptation.ipynb

Covers Sections 0 through 21 with complete code and markdown cells.
Configured with SMOKE_MODE = False as default.
DOES NOT EXECUTE THE NOTEBOOK.
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_DIR = Path("notebooks/week5")
NOTEBOOK_PATH = NOTEBOOK_DIR / "week5c_encoder_adaptation.ipynb"


def make_markdown_cell(source_text: str) -> dict:
    lines = [line + "\n" for line in source_text.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines,
    }


def make_code_cell(source_text: str) -> dict:
    lines = [line + "\n" for line in source_text.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def build_notebook():
    cells = []

    # -------------------------------------------------------------
    # Section 0 — Title / Research Question
    # -------------------------------------------------------------
    cells.append(make_markdown_cell("""# Week 5-C: Semantic Encoder Adaptation and Capacity Study

**Project:** Query-Conditioned Context Allocator (QCCA) for Efficient Retrieval-Augmented Generation  
**Parent Architecture:** SARA — Selective and Adaptive Retrieval-augmented Generation with Context Compression  
**Experiment:** Semantic Capacity and Encoder Adaptation Ladder (C0 Frozen, C1 Last-1 Layer, C2 Last-2 Layers)  
**Population:** QASPER TRAIN (2,088 queries) & DEV (231 queries)  
**Governance:** Strict Split Isolation. Held-out TEST set (1,309 queries) remains 100% frozen & untouched.  

---

### Research Context & Transition from Week 5:
Week 5 established that direct retrieved passage content contains allocation signal, but under a frozen sentence encoder (`all-MiniLM-L6-v2`) with a shallow transformer aggregator, the semantic allocator (`I3-T3`: DEV F1 ≈ 0.3810, tokens ≈ 1258.2) did not outperform the canonical handcrafted Week 4 allocator (Balanced Logistic Regression: DEV F1 ≈ 0.3842, tokens ≈ 1115.9).

### Core Research Question:
> **\"Does task-specific adaptation of the semantic encoder allow the allocator to exploit retrieved-passage content more effectively than a frozen pretrained sentence encoder?\"**

### Frozen Protocol Elements from Week 5-B:
1. **Target Formulation:** Multi-Label Epsilon-Admissibility ($T_3$): $z_k(q) = 1[\\text{regret}(q, k) \\le 0.01]$ over $k \\in \\{2, 4, 5, 6, 8\\}$ via `BCEWithLogitsLoss`.
2. **Retrieval Pool:** BM25 top-10 retrieved passages.
3. **Budget Set:** $k \\in \\{2, 4, 5, 6, 8\\}$.
4. **Information Condition:** Fixed to $I_3$ (Query + Top-10 Passages).
5. **Downstream Policy Rule:** Select smallest $k$ such that $p_k \\ge \\tau$, fallback to $k=8$.
6. **Aggregator Architecture:** Corrected rank embeddings initialized $\\mathcal{N}(0, 0.02^2)$ with `LayerNorm(h_i + r_i)`.
7. **Threshold Grid:** $\\tau \\in \\{0.30, 0.40, 0.50, 0.60, 0.70\\}$ calibrated on INTERNAL-VAL using frozen 0.005 tolerance rule."""))

    # -------------------------------------------------------------
    # Section 1 — Frozen References
    # -------------------------------------------------------------
    cells.append(make_markdown_cell("""## Section 1: Frozen Baselines and Historical References"""))

    cells.append(make_code_cell("""import os
import sys
import json
import logging
from pathlib import Path
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path.cwd().resolve()
while not (PROJECT_ROOT / "src").is_dir() and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from src.week5.data import (
    K_CLASSES,
    NUM_CLASSES,
    ContentAllocatorDataset,
    build_w5b_splits_and_bundles,
)
from src.week5.training import set_seed
from src.week5.w5c.adaptation import (
    AdaptedContentAwareAllocator,
    audit_minilm_architecture,
    build_optimizer_param_groups,
)
from src.week5.w5c.evaluation import (
    load_c0_w5b_reference,
    load_w4_baseline_dev,
    select_tau_w5c,
    evaluate_w5c_model_on_dev,
    evaluate_allocation_policy,
    run_paper_clustered_bootstrap_comparisons,
    plot_w5c_pareto_curve,
    W4_REF_F1,
    W4_REF_TOKENS,
    C0_REF_F1,
    C0_REF_TOKENS,
)
from src.week5.w5c.training import train_adapted_allocator
from src.week5.w5c.representation_analysis import compute_representation_shift

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Load canonical C0 reference from Week 5-B
c0_ref = load_c0_w5b_reference("results/week5/content_allocator/w5b")
print("=== CANONICAL REFERENCES ===")
print(f"C0 Frozen MiniLM (I3-T3): DEV F1 = {c0_ref['mean_answer_f1']:.4f}, Tokens = {c0_ref['mean_context_tokens']:.1f}, Mean k = {c0_ref['mean_selected_k']:.2f}")
print(f"W4 Balanced LogReg Baseline: DEV F1 = {W4_REF_F1:.4f}, Tokens = {W4_REF_TOKENS:.1f}, Mean k = 4.82")
print(f"Best Static SARA (k=8): DEV F1 = 0.3950, Tokens = 1750.5, Mean k = 8.00")
print(f"Quality Oracle: DEV F1 = 0.4895, Tokens = 1021.5, Mean k = 3.53")
print(f"Cost-Aware Epsilon Oracle: DEV F1 = 0.4894, Tokens = 839.4, Mean k = 3.49")"""))

    # -------------------------------------------------------------
    # Section 2 — Configuration
    # -------------------------------------------------------------
    cells.append(make_markdown_cell("""## Section 2: Experiment Configuration"""))

    cells.append(make_code_cell("""with open("configs/week5/w5c_encoder_adaptation.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

SEEDS = config["experiment"]["seeds"]
PRIMARY_SEED = config["experiment"]["primary_seed"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPSILON = config["experiment"]["epsilon"]
TAU_GRID = config["experiment"]["tau_grid"]
TAU_TOLERANCE = config["experiment"]["tau_selection_tolerance"]

ENCODER_LR = float(config["training"]["encoder_lr"])
HEAD_LR = float(config["training"]["head_lr"])
BATCH_SIZE = int(config["training"]["batch_size"])
MAX_EPOCHS = int(config["training"]["max_epochs"])
PATIENCE = int(config["training"]["patience"])
GRAD_CLIP = float(config["training"]["grad_clip"])

print(f"Device: {DEVICE}")
print(f"Seeds: {SEEDS}")
print(f"Differential LRs: Encoder={ENCODER_LR}, Head/Aggregator={HEAD_LR}")
print(f"Epochs: {MAX_EPOCHS}, Patience: {PATIENCE}, Batch Size: {BATCH_SIZE}")
print(f"Epsilon: {EPSILON}, Tau Grid: {TAU_GRID}")"""))

    # -------------------------------------------------------------
    # Section 3 — Split Integrity Audit
    # -------------------------------------------------------------
    cells.append(make_markdown_cell("""## Section 3: Split Integrity and Manifest Audit"""))

    cells.append(make_code_cell("""# Load manifests
manifest_fit_path = Path("results/week5/content_allocator/w5b/splits/train_fit_manifest.csv")
manifest_val_path = Path("results/week5/content_allocator/w5b/splits/internal_val_manifest.csv")
assert manifest_fit_path.exists() and manifest_val_path.exists(), "Missing split manifests!"

df_fit_man = pd.read_csv(manifest_fit_path)
df_val_man = pd.read_csv(manifest_val_path)

print(f"TRAIN-FIT Manifest: {len(df_fit_man)} questions across {df_fit_man['paper_id'].nunique()} papers")
print(f"INTERNAL-VAL Manifest: {len(df_val_man)} questions across {df_val_man['paper_id'].nunique()} papers")
assert len(df_fit_man) == 1746, f"Expected 1746 FIT queries, found {len(df_fit_man)}"
assert len(df_val_man) == 342, f"Expected 342 VAL queries, found {len(df_val_man)}"

# Verify paper and question disjointness
papers_fit = set(df_fit_man["paper_id"])
papers_val = set(df_val_man["paper_id"])
qids_fit = set(df_fit_man["question_id"])
qids_val = set(df_val_man["question_id"])

assert papers_fit.isdisjoint(papers_val), "DATA LEAKAGE: Overlapping papers between FIT and VAL!"
assert qids_fit.isdisjoint(qids_val), "DATA LEAKAGE: Overlapping questions between FIT and VAL!"
print("Split Integrity Audit: PASSED (Zero paper or question overlap between FIT and VAL).")"""))

    # -------------------------------------------------------------
    # Section 4 — T3 Target Verification
    # -------------------------------------------------------------
    cells.append(make_markdown_cell("""## Section 4: Target Formulation ($T_3$ Multi-Label Admissibility) Verification"""))

    cells.append(make_code_cell("""fit_bundle, val_bundle, dev_bundle, _ = build_w5b_splits_and_bundles(save_manifests=False)

# Admissibility rates across budgets k in {2, 4, 5, 6, 8}
fit_t3_rates = fit_bundle["t3_labels"].mean(axis=0)
val_t3_rates = val_bundle["t3_labels"].mean(axis=0)
dev_t3_rates = dev_bundle["t3_labels"].mean(axis=0)

df_target_rates = pd.DataFrame({
    "Budget (k)": K_CLASSES,
    "TRAIN-FIT Positive Rate": np.round(fit_t3_rates, 4),
    "INTERNAL-VAL Positive Rate": np.round(val_t3_rates, 4),
    "DEV Positive Rate": np.round(dev_t3_rates, 4),
})
print("=== T3 MULTI-LABEL ADMISSIBILITY RATES ===")
print(df_target_rates.to_string(index=False))"""))

    # -------------------------------------------------------------
    # Section 5 — MiniLM Architecture Audit
    # -------------------------------------------------------------
    cells.append(make_markdown_cell("""## Section 5: MiniLM Encoder Architecture Audit"""))

    cells.append(make_code_cell("""base_encoder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
audit_info = audit_minilm_architecture(base_encoder)

print("=== MINILM-L6-V2 ARCHITECTURE AUDIT ===")
print(f"Total Transformer Layers: {audit_info['num_transformer_layers']}")
print(f"Total Base Encoder Parameters: {audit_info['total_encoder_params']:,}")
print(f"Embeddings Parameters: {audit_info['embeddings_param_count']:,}")
print(f"Layer 4 Parameters: {audit_info['layer_4_param_count']:,}")
print(f"Layer 5 Parameters: {audit_info['layer_5_param_count']:,}")
print(f"Pooler exists: {audit_info['has_pooler']}")"""))

    # -------------------------------------------------------------
    # Section 6 — Model Definitions (C0, C1, C2)
    # -------------------------------------------------------------
    cells.append(make_markdown_cell("""## Section 6: Capacity Ladder Model Definitions and Parameter Breakdown"""))

    cells.append(make_code_cell("""model_c0 = AdaptedContentAwareAllocator(capacity_condition="C0")
model_c1 = AdaptedContentAwareAllocator(capacity_condition="C1")
model_c2 = AdaptedContentAwareAllocator(capacity_condition="C2")

breakdowns = [
    model_c0.get_parameter_breakdown(),
    model_c1.get_parameter_breakdown(),
    model_c2.get_parameter_breakdown(),
]

df_params = pd.DataFrame(breakdowns)
print("=== CAPACITY LADDER PARAMETER COUNTS ===")
print(df_params[[
    "capacity_condition",
    "total_params",
    "trainable_params",
    "trainable_pct",
    "encoder_trainable_params",
    "encoder_frozen_params",
    "head_and_aggregator_trainable_params",
]].to_string(index=False))"""))

    # -------------------------------------------------------------
    # Section 7 — Tokenization Audit
    # -------------------------------------------------------------
    cells.append(make_markdown_cell("""## Section 7: Pairwise Query-Passage Tokenization Audit"""))

    cells.append(make_code_cell("""from src.week5.data import compute_truncation_statistics

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
trunc_stats = compute_truncation_statistics(dev_bundle, tokenizer, max_seq_length=256)

print("=== TOKENIZATION TRUNCATION AUDIT (DEV) ===")
for k, v in trunc_stats.items():
    print(f"{k}: {v}")"""))

    # -------------------------------------------------------------
    # Section 8 — Smoke Configuration
    # -------------------------------------------------------------
    cells.append(make_markdown_cell("""## Section 8: Execution Mode Configuration"""))

    cells.append(make_code_cell("""# Master execution flag: Default False for full canonical experiment execution
SMOKE_MODE = False
print(f"SMOKE_MODE = {SMOKE_MODE}")
if SMOKE_MODE:
    print("WARNING: Running in FAST SMOKE MODE (1 epoch, subset of data, seed 42 only).")
else:
    print("Running in FULL CANONICAL EXPERIMENT MODE (All 3 seeds, full data, early stopping).")"""))

    # -------------------------------------------------------------
    # Section 9 — C1 Training (3 Seeds)
    # -------------------------------------------------------------
    cells.append(make_markdown_cell("""## Section 9: C1 Training (Last-1 Transformer Block Trainable) Across 3 Seeds"""))

    cells.append(make_code_cell("""from torch.utils.data import DataLoader, Subset

# Construct full Datasets
train_dataset = ContentAllocatorDataset(fit_bundle, tokenizer=tokenizer, condition="I3", max_seq_length=256)
val_dataset = ContentAllocatorDataset(val_bundle, tokenizer=tokenizer, condition="I3", max_seq_length=256)
dev_dataset = ContentAllocatorDataset(dev_bundle, tokenizer=tokenizer, condition="I3", max_seq_length=256)

c1_seeds_to_run = [PRIMARY_SEED] if SMOKE_MODE else SEEDS
c1_epochs = 1 if SMOKE_MODE else MAX_EPOCHS

models_c1 = {}
histories_c1 = {}
healths_c1 = {}

for seed in c1_seeds_to_run:
    print(f"\\n--- Training C1 (Last-1 Layer) | Seed {seed} ---")
    set_seed(seed)
    
    train_loader = DataLoader(
        Subset(train_dataset, range(16)) if SMOKE_MODE else train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        Subset(val_dataset, range(8)) if SMOKE_MODE else val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    
    model = AdaptedContentAwareAllocator(
        capacity_condition="C1",
        base_encoder_name="sentence-transformers/all-MiniLM-L6-v2",
    ).to(DEVICE)
    
    train_res = train_adapted_allocator(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=c1_epochs,
        encoder_lr=ENCODER_LR,
        head_lr=HEAD_LR,
        patience=PATIENCE,
        grad_clip=GRAD_CLIP,
        device=DEVICE,
        verbose=True,
    )
    
    models_c1[seed] = train_res["model"]
    histories_c1[seed] = train_res["history"]
    healths_c1[seed] = train_res["adaptation_health"]
    
    # Save checkpoint
    out_pt = Path(f"results/week5/content_allocator/w5c_encoder_adaptation/models/C1_seed_{seed}.pt")
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(train_res["model"].state_dict(), out_pt)
    print(f"Saved C1 Seed {seed} model to {out_pt}")"""))

    # -------------------------------------------------------------
    # Section 10 — C2 Training (3 Seeds)
    # -------------------------------------------------------------
    cells.append(make_markdown_cell("""## Section 10: C2 Training (Last-2 Transformer Blocks Trainable) Across 3 Seeds"""))

    cells.append(make_code_cell("""c2_seeds_to_run = [PRIMARY_SEED] if SMOKE_MODE else SEEDS
c2_epochs = 1 if SMOKE_MODE else MAX_EPOCHS

models_c2 = {}
histories_c2 = {}
healths_c2 = {}

for seed in c2_seeds_to_run:
    print(f"\\n--- Training C2 (Last-2 Layers) | Seed {seed} ---")
    set_seed(seed)
    
    train_loader = DataLoader(
        Subset(train_dataset, range(16)) if SMOKE_MODE else train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        Subset(val_dataset, range(8)) if SMOKE_MODE else val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    
    model = AdaptedContentAwareAllocator(
        capacity_condition="C2",
        base_encoder_name="sentence-transformers/all-MiniLM-L6-v2",
    ).to(DEVICE)
    
    train_res = train_adapted_allocator(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=c2_epochs,
        encoder_lr=ENCODER_LR,
        head_lr=HEAD_LR,
        patience=PATIENCE,
        grad_clip=GRAD_CLIP,
        device=DEVICE,
        verbose=True,
    )
    
    models_c2[seed] = train_res["model"]
    histories_c2[seed] = train_res["history"]
    healths_c2[seed] = train_res["adaptation_health"]
    
    # Save checkpoint
    out_pt = Path(f"results/week5/content_allocator/w5c_encoder_adaptation/models/C2_seed_{seed}.pt")
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(train_res["model"].state_dict(), out_pt)
    print(f"Saved C2 Seed {seed} model to {out_pt}")"""))

    # -------------------------------------------------------------
    # Section 11 — Internal Validation & Tau Selection
    # -------------------------------------------------------------
    cells.append(make_markdown_cell("""## Section 11: INTERNAL-VAL Performance and Threshold Calibration"""))

    cells.append(make_code_cell("""val_loader_full = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

tau_records = []
selected_taus = {}

# Evaluate on INTERNAL-VAL and select tau for each model
for cond, models_dict in [("C1", models_c1), ("C2", models_c2)]:
    for seed, m in models_dict.items():
        m.eval()
        all_probs = []
        with torch.no_grad():
            for batch in val_loader_full:
                logits = m(batch)["admissibility_logits"]
                all_probs.append(torch.sigmoid(logits).cpu().numpy())
        val_probs = np.concatenate(all_probs, axis=0)
        
        tau_res = select_tau_w5c(
            val_supervision_df=val_bundle["supervision_df"],
            val_pred_probs=val_probs,
            val_tokens_map=val_bundle["tokens_map"],
            tau_grid=TAU_GRID,
            quality_tolerance=TAU_TOLERANCE,
        )
        
        selected_taus[(cond, seed)] = tau_res["selected_tau"]
        tau_records.append({
            "capacity_condition": cond,
            "seed": seed,
            "selected_tau": tau_res["selected_tau"],
            "tau_boundary_hit": tau_res["tau_boundary_hit"],
            "quality_constraint_met": tau_res["constraint_met"],
            "static_k8_val_f1": round(tau_res["static_k8_f1"], 4),
            "selected_val_f1": round(tau_res["selected_result"]["mean_answer_f1"], 4),
            "selected_val_tokens": round(tau_res["selected_result"]["mean_context_tokens"], 1),
        })

df_tau_sel = pd.DataFrame(tau_records)
df_tau_sel.to_csv("results/week5/content_allocator/w5c_encoder_adaptation/tables/w5c_tau_selection.csv", index=False)
print("=== THRESHOLD CALIBRATION ON INTERNAL-VAL ===")
print(df_tau_sel.to_string(index=False))"""))

    # -------------------------------------------------------------
    # Section 12 — Adaptation Health & Representation Shift
    # -------------------------------------------------------------
    cells.append(make_markdown_cell("""## Section 12: Adaptation Health Check & Representation Shift Analysis"""))

    cells.append(make_code_cell("""# 1. Save Gradient & Adaptation Health Check
all_health = {
    "C1": {str(seed): healths_c1[seed] for seed in healths_c1},
    "C2": {str(seed): healths_c2[seed] for seed in healths_c2},
}
with open("results/week5/content_allocator/w5c_encoder_adaptation/diagnostics/representation_adaptation_health.json", "w", encoding="utf-8") as f:
    json.dump(all_health, f, indent=2)

# 2. Compute Representation Shift on INTERNAL-VAL
shift_records = []
for cond, models_dict in [("C1", models_c1), ("C2", models_c2)]:
    for seed, m in models_dict.items():
        shift_dict = compute_representation_shift(
            pretrained_model=model_c0,
            adapted_model=m,
            val_loader=val_loader_full,
            max_batches=5,
            device=DEVICE,
        )
        shift_dict["capacity_condition"] = cond
        shift_dict["seed"] = seed
        shift_records.append(shift_dict)

df_rep_shift = pd.DataFrame(shift_records)
df_rep_shift.to_csv("results/week5/content_allocator/w5c_encoder_adaptation/diagnostics/representation_shift.csv", index=False)

print("=== REPRESENTATION SHIFT DIAGNOSTICS ===")
print(df_rep_shift[[
    "capacity_condition",
    "seed",
    "mean_cosine_similarity",
    "mean_representation_shift_norm",
    "within_query_passage_var_adapted",
    "across_query_var_adapted",
]].to_string(index=False))"""))

    # -------------------------------------------------------------
    # Section 13 — Frozen DEV Evaluation
    # -------------------------------------------------------------
    cells.append(make_markdown_cell("""## Section 13: Final Evaluation on Frozen DEV Set"""))

    cells.append(make_code_cell("""dev_loader_full = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
dev_sup_df = dev_bundle["supervision_df"]

eval_results_dev = {}
predictions_dict = {
    "question_id": dev_bundle["question_ids"],
    "paper_id": dev_bundle["paper_ids"],
}

# Add W5-B C0 reference predictions if available
if c0_ref["predictions_df"] is not None:
    for seed in SEEDS:
        col = f"pred_k_I3_seed{seed}"
        if col in c0_ref["predictions_df"]:
            predictions_dict[f"pred_k_C0_seed{seed}"] = c0_ref["predictions_df"][col]

for cond, models_dict in [("C1", models_c1), ("C2", models_c2)]:
    for seed, m in models_dict.items():
        tau = selected_taus[(cond, seed)]
        eval_res = evaluate_w5c_model_on_dev(
            model=m,
            dev_loader=dev_loader_full,
            dev_sup_df=dev_sup_df,
            tau=tau,
            device=DEVICE,
        )
        eval_results_dev[(cond, seed)] = eval_res
        predictions_dict[f"pred_k_{cond}_seed{seed}"] = eval_res["preds_k"]

df_dev_preds = pd.DataFrame(predictions_dict)
df_dev_preds.to_csv("results/week5/content_allocator/w5c_encoder_adaptation/predictions/dev_predictions.csv", index=False)
print("Saved DEV predictions to results/week5/content_allocator/w5c_encoder_adaptation/predictions/dev_predictions.csv")"""))

    # -------------------------------------------------------------
    # Section 14 — Main Policy Table
    # -------------------------------------------------------------
    cells.append(make_markdown_cell("""## Section 14: Main Policy Table across Capacity Ladder and Baselines"""))

    cells.append(make_code_cell("""# Compile Main Results Table
main_rows = []

# Baselines
w4_eval = load_w4_baseline_dev(dev_sup_df=dev_sup_df)

main_rows.append({
    "model_name": "Quality Oracle",
    "mean_answer_f1": 0.4895,
    "std_answer_f1": 0.0,
    "mean_selected_k": 3.53,
    "mean_context_tokens": 1021.5,
    "token_reduction_pct": 41.65,
    "mean_oracle_regret": 0.0000,
    "fallback_rate_pct": 0.0,
    "recovered_oracle_headroom": 1.0,
})

main_rows.append({
    "model_name": "Cost-Aware Epsilon Oracle",
    "mean_answer_f1": 0.4894,
    "std_answer_f1": 0.0,
    "mean_selected_k": 3.49,
    "mean_context_tokens": 839.4,
    "token_reduction_pct": 52.05,
    "mean_oracle_regret": 0.0001,
    "fallback_rate_pct": 0.0,
    "recovered_oracle_headroom": 0.9989,
})

main_rows.append({
    "model_name": "Best Static SARA (k=8)",
    "mean_answer_f1": 0.3950,
    "std_answer_f1": 0.0,
    "mean_selected_k": 8.00,
    "mean_context_tokens": 1750.5,
    "token_reduction_pct": 0.0,
    "mean_oracle_regret": 0.0945,
    "fallback_rate_pct": 0.0,
    "recovered_oracle_headroom": 0.0,
})

main_rows.append({
    "model_name": "Static k=6",
    "mean_answer_f1": 0.3857,
    "std_answer_f1": 0.0,
    "mean_selected_k": 6.00,
    "mean_context_tokens": 1351.1,
    "token_reduction_pct": 22.82,
    "mean_oracle_regret": 0.1038,
    "fallback_rate_pct": 0.0,
    "recovered_oracle_headroom": -0.0984,
})

main_rows.append({
    "model_name": "W4 Balanced Logistic Regression",
    "mean_answer_f1": w4_eval["mean_answer_f1"],
    "std_answer_f1": 0.0,
    "mean_selected_k": w4_eval["mean_selected_k"],
    "mean_context_tokens": w4_eval["mean_context_tokens"],
    "token_reduction_pct": w4_eval["token_reduction_pct_vs_k8"],
    "mean_oracle_regret": w4_eval["mean_oracle_regret"],
    "fallback_rate_pct": 0.0,
    "recovered_oracle_headroom": w4_eval["recovered_oracle_headroom"],
})

# C0 Frozen MiniLM (W5-B Reference)
main_rows.append({
    "model_name": "C0 Frozen MiniLM",
    "mean_answer_f1": c0_ref["mean_answer_f1"],
    "std_answer_f1": c0_ref.get("std_answer_f1", 0.0090),
    "mean_selected_k": c0_ref["mean_selected_k"],
    "mean_context_tokens": c0_ref["mean_context_tokens"],
    "token_reduction_pct": c0_ref["token_reduction_pct"],
    "mean_oracle_regret": 0.1085,
    "fallback_rate_pct": c0_ref["fallback_rate_pct"],
    "recovered_oracle_headroom": c0_ref["recovered_oracle_headroom"],
})

# C1 and C2 aggregates
for cond in ["C1", "C2"]:
    cond_evals = [eval_results_dev[(cond, seed)]["policy_metrics"] for seed in SEEDS if (cond, seed) in eval_results_dev]
    f1s = [e["mean_answer_f1"] for e in cond_evals]
    ks = [e["mean_selected_k"] for e in cond_evals]
    toks = [e["mean_context_tokens"] for e in cond_evals]
    reducs = [e["token_reduction_pct_vs_k8"] for e in cond_evals]
    regs = [e["mean_oracle_regret"] for e in cond_evals]
    rohs = [e["recovered_oracle_headroom"] for e in cond_evals]
    fbs = [eval_results_dev[(cond, seed)]["fallback_rate_pct"] for seed in SEEDS if (cond, seed) in eval_results_dev]
    
    label = "C1 Last-1 Layer Trainable" if cond == "C1" else "C2 Last-2 Layers Trainable"
    main_rows.append({
        "model_name": label,
        "mean_answer_f1": round(float(np.mean(f1s)), 4),
        "std_answer_f1": round(float(np.std(f1s)), 4) if len(f1s) > 1 else 0.0,
        "mean_selected_k": round(float(np.mean(ks)), 2),
        "mean_context_tokens": round(float(np.mean(toks)), 1),
        "token_reduction_pct": round(float(np.mean(reducs)), 2),
        "mean_oracle_regret": round(float(np.mean(regs)), 4),
        "fallback_rate_pct": round(float(np.mean(fbs)), 2),
        "recovered_oracle_headroom": round(float(np.mean(rohs)), 4),
    })

df_main = pd.DataFrame(main_rows)
df_main.to_csv("results/week5/content_allocator/w5c_encoder_adaptation/tables/w5c_main_results.csv", index=False)
print("=== MAIN POLICY RESULTS TABLE ===")
print(df_main.to_string(index=False))"""))

    # -------------------------------------------------------------
    # Section 15 — Seed Stability
    # -------------------------------------------------------------
    cells.append(make_markdown_cell("""## Section 15: Seed Stability and Per-Seed Breakdown"""))

    cells.append(make_code_cell("""seed_rows = []
for (cond, seed), res in eval_results_dev.items():
    p = res["policy_metrics"]
    m = res["multilabel_metrics"]
    dist = res["k_distribution"]
    seed_rows.append({
        "capacity_condition": cond,
        "seed": seed,
        "selected_tau": res["selected_tau"],
        "dev_answer_f1": p["mean_answer_f1"],
        "dev_mean_k": p["mean_selected_k"],
        "dev_context_tokens": p["mean_context_tokens"],
        "token_reduction_pct": p["token_reduction_pct_vs_k8"],
        "mean_oracle_regret": p["mean_oracle_regret"],
        "zero_regret_pct": p["zero_regret_pct"],
        "fallback_rate_pct": res["fallback_rate_pct"],
        "recovered_oracle_headroom": p["recovered_oracle_headroom"],
        "macro_auprc": m["macro_auprc"],
        "macro_auroc": m["macro_auroc"],
        "logit_variance": res["logit_variance"],
        **dist,
    })

df_seeds = pd.DataFrame(seed_rows)
df_seeds.to_csv("results/week5/content_allocator/w5c_encoder_adaptation/tables/w5c_seed_results.csv", index=False)
print("=== PER-SEED POLICY RESULTS ===")
print(df_seeds[[
    "capacity_condition",
    "seed",
    "dev_answer_f1",
    "dev_mean_k",
    "dev_context_tokens",
    "token_reduction_pct",
    "fallback_rate_pct",
    "macro_auprc",
]].to_string(index=False))"""))

    # -------------------------------------------------------------
    # Section 16 — Selected-k Distributions
    # -------------------------------------------------------------
    cells.append(make_markdown_cell("""## Section 16: Selected Budget Distribution ($k \\in \\{2, 4, 5, 6, 8\\}$)"""))

    cells.append(make_code_cell("""fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

plot_data = []
for cond in ["C1", "C2"]:
    c_df = df_seeds[df_seeds["capacity_condition"] == cond]
    if len(c_df) > 0:
        row = {f"k={k}": c_df[f"pct_k{k}"].mean() for k in K_CLASSES}
        row["Condition"] = cond
        plot_data.append(row)

if plot_data:
    df_plot_k = pd.DataFrame(plot_data).set_index("Condition")
    df_plot_k.plot(kind="bar", stacked=True, ax=ax, colormap="viridis", width=0.5)
    plt.ylabel("Percentage of DEV Queries (%)", fontweight="bold")
    plt.title("Budget Allocation Distribution across Capacity Conditions", fontweight="bold")
    plt.legend(title="Budget", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("results/week5/content_allocator/w5c_encoder_adaptation/figures/w5c_selected_k_distribution.png")
    plt.close()
    print("Saved selected-k distribution plot to results/week5/content_allocator/w5c_encoder_adaptation/figures/w5c_selected_k_distribution.png")"""))

    # -------------------------------------------------------------
    # Section 17 — Multi-Label Diagnostics
    # -------------------------------------------------------------
    cells.append(make_markdown_cell("""## Section 17: Secondary Multi-Label Diagnostic Metrics"""))

    cells.append(make_code_cell("""multi_rows = []
for (cond, seed), res in eval_results_dev.items():
    m = res["multilabel_metrics"]
    multi_rows.append({
        "capacity_condition": cond,
        "seed": seed,
        "macro_auprc": m["macro_auprc"],
        "macro_auroc": m["macro_auroc"],
        "micro_auprc": m["micro_auprc"],
        "overall_brier": m["overall_brier"],
        "auprc_k2": m["per_budget_auprc"]["k2"],
        "auprc_k4": m["per_budget_auprc"]["k4"],
        "auprc_k5": m["per_budget_auprc"]["k5"],
        "auprc_k6": m["per_budget_auprc"]["k6"],
        "auprc_k8": m["per_budget_auprc"]["k8"],
    })

df_multi = pd.DataFrame(multi_rows)
df_multi.to_csv("results/week5/content_allocator/w5c_encoder_adaptation/tables/w5c_multilabel_metrics.csv", index=False)
print("=== MULTI-LABEL METRICS TABLE ===")
print(df_multi[[
    "capacity_condition",
    "seed",
    "macro_auprc",
    "macro_auroc",
    "micro_auprc",
    "overall_brier",
]].to_string(index=False))"""))

    # -------------------------------------------------------------
    # Section 18 — Pareto Frontier Plot
    # -------------------------------------------------------------
    cells.append(make_markdown_cell("""## Section 18: Quality–Cost Pareto Frontier Visualization"""))

    cells.append(make_code_cell("""plot_w5c_pareto_curve(
    results_df=df_main,
    output_path="results/week5/content_allocator/w5c_encoder_adaptation/figures/w5c_pareto.png",
)
print("Pareto frontier figure saved to results/week5/content_allocator/w5c_encoder_adaptation/figures/w5c_pareto.png")"""))

    # -------------------------------------------------------------
    # Section 19 — Paper-Clustered Bootstrap
    # -------------------------------------------------------------
    cells.append(make_markdown_cell("""## Section 19: Paper-Clustered Statistical Bootstrap Analysis"""))

    cells.append(make_code_cell("""from src.week5.evaluation import evaluate_allocation_policy

paper_ids = dev_bundle["paper_ids"]

# Compute across-seed mean per-query vectors for C1 and C2
c1_query_f1s = np.mean([eval_results_dev[("C1", s)]["policy_metrics"]["per_query_f1s"] for s in c1_seeds_to_run], axis=0)
c1_query_toks = np.mean([eval_results_dev[("C1", s)]["policy_metrics"]["per_query_tokens"] for s in c1_seeds_to_run], axis=0)

c2_query_f1s = np.mean([eval_results_dev[("C2", s)]["policy_metrics"]["per_query_f1s"] for s in c2_seeds_to_run], axis=0)
c2_query_toks = np.mean([eval_results_dev[("C2", s)]["policy_metrics"]["per_query_tokens"] for s in c2_seeds_to_run], axis=0)

# Load C0 per-query vectors from W5-B predictions
if c0_ref["predictions_df"] is not None:
    c0_query_f1s_list = []
    c0_query_toks_list = []
    for s in SEEDS:
        col = f"pred_k_I3_seed{s}"
        if col in c0_ref["predictions_df"]:
            e = evaluate_allocation_policy(dev_sup_df, c0_ref["predictions_df"][col].values)
            c0_query_f1s_list.append(e["per_query_f1s"])
            c0_query_toks_list.append(e["per_query_tokens"])
    c0_query_f1s = np.mean(c0_query_f1s_list, axis=0)
    c0_query_toks = np.mean(c0_query_toks_list, axis=0)
else:
    c0_query_f1s = np.full(len(dev_sup_df), C0_REF_F1)
    c0_query_toks = np.full(len(dev_sup_df), C0_REF_TOKENS)

eval_c0_dict = {"per_query_f1s": c0_query_f1s, "per_query_tokens": c0_query_toks}
eval_c1_dict = {"per_query_f1s": c1_query_f1s, "per_query_tokens": c1_query_toks}
eval_c2_dict = {"per_query_f1s": c2_query_f1s, "per_query_tokens": c2_query_toks}
eval_k8_dict = {"per_query_f1s": dev_sup_df["f1_k8"].values, "per_query_tokens": np.full(len(dev_sup_df), 1750.5)}

n_boot = 100 if SMOKE_MODE else 2000
df_boot = run_paper_clustered_bootstrap_comparisons(
    paper_ids=paper_ids,
    eval_c0=eval_c0_dict,
    eval_c1=eval_c1_dict,
    eval_c2=eval_c2_dict,
    eval_w4=w4_eval,
    eval_k8=eval_k8_dict,
    n_bootstrap=n_boot,
    seed=42,
)

df_boot.to_csv("results/week5/content_allocator/w5c_encoder_adaptation/bootstrap/w5c_bootstrap_comparisons.csv", index=False)
print("=== PAPER-CLUSTERED BOOTSTRAP RESULTS (ACROSS-SEED MEAN VECTORS) ===")
print(df_boot[[
    "comparison",
    "observed_delta_f1",
    "ci_lower_95_f1",
    "ci_upper_95_f1",
    "p_value_f1",
    "significant_f1",
    "observed_delta_tokens",
    "ci_lower_95_tokens",
    "ci_upper_95_tokens",
]].to_string(index=False))"""))

    # -------------------------------------------------------------
    # Section 20 — Capacity Interpretation
    # -------------------------------------------------------------
    cells.append(make_markdown_cell("""## Section 20: Scientific Capacity Interpretation

Here we synthesize empirical evidence across the capacity ladder to address the five pre-registered questions:
1. **Does task-specific adaptation of the semantic encoder improve allocation quality over frozen MiniLM?**
2. **Is adapting a single Transformer block ($C_1$) sufficient to achieve significant gains?**
3. **Does adapting two Transformer blocks ($C_2$) provide additional benefit over $C_1$?**
4. **Does adapted semantic allocation outperform the handcrafted Week 4 baseline ($W_4$)?**
5. **Does semantic capacity remain the primary bottleneck on the quality–context frontier?**"""))

    cells.append(make_code_cell("""# Compile capacity comparison table
cap_rows = [
    {
        "Condition": "C0 Frozen MiniLM",
        "Trainable Encoder Layers": 0,
        "Trainable Params": "2.82M (11.0%)",
        "DEV Answer F1": f"{c0_ref['mean_answer_f1']:.4f}",
        "Context Tokens": f"{c0_ref['mean_context_tokens']:.1f}",
        "Token Reduction": f"{c0_ref['token_reduction_pct']:.1f}%",
        "ROH": f"{c0_ref['recovered_oracle_headroom']:.4f}",
    },
]

for cond, label in [("C1", "C1 Last-1 Layer"), ("C2", "C2 Last-2 Layers")]:
    c_df = df_main[df_main["model_name"].str.startswith(cond)]
    if len(c_df) > 0:
        row = c_df.iloc[0]
        n_layers = 1 if cond == "C1" else 2
        pct = "18.0%" if cond == "C1" else "24.9%"
        cap_rows.append({
            "Condition": label,
            "Trainable Encoder Layers": n_layers,
            "Trainable Params": f"{row['mean_answer_f1']:.4f}" if False else f"{df_params[df_params['capacity_condition'] == cond]['trainable_params'].values[0] / 1e6:.2f}M ({pct})",
            "DEV Answer F1": f"{row['mean_answer_f1']:.4f}",
            "Context Tokens": f"{row['mean_context_tokens']:.1f}",
            "Token Reduction": f"{row['token_reduction_pct']:.1f}%",
            "ROH": f"{row['recovered_oracle_headroom']:.4f}",
        })

df_cap_comp = pd.DataFrame(cap_rows)
df_cap_comp.to_csv("results/week5/content_allocator/w5c_encoder_adaptation/tables/w5c_capacity_comparison.csv", index=False)
print("=== CAPACITY LADDER COMPARISON SUMMARY ===")
print(df_cap_comp.to_string(index=False))"""))

    # -------------------------------------------------------------
    # Section 21 — Decision for Next Week
    # -------------------------------------------------------------
    cells.append(make_markdown_cell("""## Section 21: Pre-Registered Decision Rules for Week 6 (Passage Routing)

Based on empirical performance against pre-registered criteria:
- **Case 1 (Strong Adaptation Signal):** If $C_1$ or $C_2$ achieves $F_1 > 0.3842$ (surpassing $W_4$) with comparable or lower context tokens:
  $\\rightarrow$ Conclude that encoder adaptation unlocks passage semantic allocation. Freeze architecture and optimize calibration/routing.
- **Case 2 (Plateau / Insufficient MiniLM Capacity):** If $C_1$ and $C_2$ show modest or null improvement over $C_0$ ($F_1 \\approx 0.3810$) without beating $W_4$:
  $\\rightarrow$ Conclude that small sentence-transformer capacity (MiniLM, 22.7M params) is the fundamental bottleneck. Advance to larger capacity models (e.g., DeBERTa-v3, BGE, or generative LLM routing).
- **Case 3 (Overfitting / Negative Transfer):** If $C_1$ or $C_2$ underperforms $C_0$:
  $\\rightarrow$ Diagnose representation shift and feature drift before expanding capacity."""))

    cells.append(make_code_cell("""print("Week 5-C Notebook execution pipeline complete.")
print("Generated Outputs:")
print(" - tables/w5c_main_results.csv")
print(" - tables/w5c_seed_results.csv")
print(" - tables/w5c_multilabel_metrics.csv")
print(" - tables/w5c_tau_selection.csv")
print(" - tables/w5c_capacity_comparison.csv")
print(" - predictions/dev_predictions.csv")
print(" - bootstrap/w5c_bootstrap_comparisons.csv")
print(" - diagnostics/representation_adaptation_health.json")
print(" - diagnostics/representation_shift.csv")
print(" - figures/w5c_pareto.png")
print(" - figures/w5c_selected_k_distribution.png")"""))

    # Assemble notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python (sara_env)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.16",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    print(f"Generated {NOTEBOOK_PATH} ({len(cells)} cells)")


if __name__ == "__main__":
    build_notebook()
