"""Script to generate the complete Week 7-D Jupyter Notebook with ALL model executions fully implemented:
- D0: Frozen Cross-Encoder + GBDT
- D1-A: Frozen Backbone Hierarchical Cross-Encoder (Set-level Transformer Aggregator + Multi-task Heads)
- D1-B: Partially Unfrozen Hierarchical Cross-Encoder (Last 2 layers unfrozen end-to-end)
- D2: Listwise 7B Mistral LoRA Router (causal listwise reasoning from h_DECIDE)
- Full 4-way Model Comparison on INTERNAL-VAL
- Quality-Constrained Winner Selection (eta = 0.005 vs static k4)
- Comprehensive DEV Evaluation across all models
- Quality-Context Pareto Frontier with all models
- 6-Way Error Taxonomy, Downshift & Upshift Calibration
- 2,000-sample Paper-Clustered Bootstrap
- Complete W7-E Handoff Manifest

Generates notebooks/week7/week7d_cross_encoder_fidelity_router.ipynb.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def create_code_cell(code: str) -> Dict[str, Any]:
    lines = [line + "\n" for line in code.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def create_markdown_cell(text: str) -> Dict[str, Any]:
    lines = [line + "\n" for line in text.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines,
    }


def build_notebook() -> Dict[str, Any]:
    cells: List[Dict[str, Any]] = []

    # Title & Metadata
    cells.append(create_markdown_cell("""# Week 7-D: Generator-Aligned Cross-Encoder Fidelity Router
**Joint Query-Document Representation Learning for Downstream SARA Utility Allocation**

---

### Scientific Motivation & Historical Progression
All previous adaptive-budget models suffered from representation bottlenecks:
* **W7-A**: Frozen W4 allocator + semantic routing composition $\\to$ modular composition failed.
* **W7-B**: Semantic-routing response surface + old W4 features $\\to$ feature mismatch.
* **W7-C1**: Rich retrieval sufficiency classifier $\\to$ over-allocation from large-$k$ fallback.
* **W7-C2**: Direct response surface regression & ranking $\\to$ high quality but over-budget.
* **W7-C3**: $k=4$ anchor-and-deviate policy $\\to$ stabilized on static $k=4$ prior, but scalar features lacked resolution for safe downshifts or justified upshifts.

**The Fundamental Hypothesis:**
$$\\boxed{\\text{The missing information is joint query-evidence interaction, not merely additional scalar features.}}$$

A true cross-encoder jointly reasons over $(q, D_i)$ capturing complementarity, redundancy, contradictions, and coverage directly aligned with downstream SARA generation utility."""))

    # Section 0
    cells.append(create_markdown_cell("""## Section 0 — Research Hypothesis & Model Hierarchy
We implement and evaluate all four progressively higher-capacity model levels:
1. **D0 (Frozen Cross-Encoder Features + Tabular Policy)**:
   Extract joint relevance scores $c_i = \\text{CrossEncoder}(q, D_i)$, 22 interaction summary statistics (top-$k$ means, boundary gaps, Spearman correlations with semantic/BM25 ranks, rank displacements), and train an anchor-and-deviate GBDT.
2. **D1-A (Trainable Hierarchical Cross-Encoder — Frozen Backbone)**:
   Jointly encode each pair $(q, D_i) \\to h_i$, concatenate retrieval metadata, aggregate via a set-level Transformer ($H \\to r_q$), and train multi-task heads predicting SARA utility ($y_{down}, y_{up}, \\Delta_k, z_k$).
3. **D1-B (Partially Unfrozen Hierarchical Cross-Encoder — End-to-End)**:
   Unfreeze the top 2 transformer layers of the cross-encoder backbone, fine-tuning them end-to-end with the aggregator and multi-task heads.
4. **D2 (Listwise 7B LLM Router — High-Capacity Causal Reasoning)**:
   Mistral-7B-Instruct-v0.2 causally reading $\\langle\\text{DOC}\\rangle ... \\langle\\text{QUESTION}\\rangle ... \\langle\\text{DECIDE}\\rangle$, extracting $h_{DECIDE}$ to drive multi-task heads under parameter-efficient LoRA."""))

    # Section 1
    cells.append(create_markdown_cell("""## Section 1 — Load Canonical Semantic Response Surface & SARA Supervision
We load canonical response surfaces from Week 7-B.
* **TRAIN**: $2,088$ queries $\\times 5$ budgets $= 10,440$ observations.
* **DEV**: $231$ queries $\\times 5$ budgets $= 1,155$ observations.
* Budgets: $K = \\{2, 4, 5, 6, 8\\}$.
* Supervised targets: $y_{down} = \\mathbf{1}[F(q, 2) \\ge F^*(q) - 0.01]$, $y_{up} = \\mathbf{1}[\\max(F_5, F_6, F_8) - F_4 > 0.01]$, $\\Delta_k = F_k - F_4$."""))

    cells.append(create_code_cell("""import os
import sys
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader

# Resolve Project Root
_curr = Path.cwd().resolve()
candidates = [_curr, _curr.parent, _curr.parent.parent]
PROJECT_ROOT = next((p for p in candidates if (p / "src").is_dir() and (p / "configs").is_dir()), _curr)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.week7.w7d.data import (
    ALLOWED_BUDGETS,
    ANCHOR_K,
    assert_test_disabled,
    assert_zero_feature_leakage,
    load_w7d_splits,
    load_retrieval_passages,
    audit_input_lengths,
    truncate_document_deterministic,
)
from src.week7.w7d.targets import (
    BUDGETS,
    compute_sara_multitask_targets,
    build_w7d_targets_dataframe,
)
from src.week7.w7d.cross_encoder import CrossEncoderScorer
from src.week7.w7d.cross_encoder_features import (
    build_d0_feature_dataframe,
    extract_query_cross_encoder_features,
)
from src.week7.w7d.hierarchical_model import HierarchicalCrossEncoderRouter
from src.week7.w7d.set_aggregator import SetLevelAggregator
from src.week7.w7d.multitask_heads import MultiTaskUtilityHeads
from src.week7.w7d.policy import (
    apply_w7d_anchor_deviate_policy,
    apply_w7d_multilabel_policy,
    evaluate_downstream_fidelity_policy,
    select_best_fidelity_policy,
)
from src.week7.w7d.training import (
    D0TabularRouter,
    CachedEmbeddingDataset,
    HierarchicalTokenDataset,
    ListwisePromptDataset,
    train_d1_aggregator,
    train_d1b_end_to_end,
    train_d2_listwise_router,
)
from src.week7.w7d.lora_utils import audit_d2_hardware_feasibility, get_peft_lora_config
from src.week7.w7d.listwise_llm import format_listwise_causal_prompt, ListwiseLLMRouter
from src.week7.w7d.evaluation import (
    compute_w7d_error_taxonomy,
    compute_downshift_safety_metrics,
    compute_upshift_precision_metrics,
    compute_recovered_oracle_headroom,
)
from src.week7.w7d.bootstrap import run_paper_clustered_bootstrap

# Load Configuration
config_path = PROJECT_ROOT / "configs/week7/w7d_cross_encoder_fidelity_router.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# Ensure output directories exist
for p in cfg["paths"].values():
    if "dir" in str(p) or "output" in str(p):
        (PROJECT_ROOT / p).mkdir(parents=True, exist_ok=True)

print("Configuration loaded. Subtask:", cfg["experiment"]["subtask"])
print("Allowed budgets:", BUDGETS, "Anchor k:", ANCHOR_K)"""))

    cells.append(create_code_cell("""# Load canonical response matrices
train_resp_path = PROJECT_ROOT / cfg["paths"]["train_response_matrix"]
dev_resp_path = PROJECT_ROOT / cfg["paths"]["dev_response_matrix"]

df_resp_train = pd.read_csv(train_resp_path)
df_resp_dev = pd.read_csv(dev_resp_path)

df_resp_train["question_id"] = df_resp_train["question_id"].astype(str).str.strip()
df_resp_dev["question_id"] = df_resp_dev["question_id"].astype(str).str.strip()

# Build SARA multitask target DataFrames
df_targets_train = build_w7d_targets_dataframe(df_resp_train, budgets=BUDGETS, anchor_k=ANCHOR_K, epsilon=cfg["experiment"]["epsilon"])
df_targets_dev = build_w7d_targets_dataframe(df_resp_dev, budgets=BUDGETS, anchor_k=ANCHOR_K, epsilon=cfg["experiment"]["epsilon"])

print(f"Loaded response surfaces: TRAIN={len(df_targets_train)} queries, DEV={len(df_targets_dev)} queries.")
print(f"TRAIN Target summary:")
print(f"  y_down rate (k2 admissible): {df_targets_train['target_y_down'].mean():.4f}")
print(f"  y_up rate (k>4 justified):   {df_targets_train['target_y_up'].mean():.4f}")
print(f"  mean Delta_5: {df_targets_train['target_delta_5'].mean():.4f}, Delta_6: {df_targets_train['target_delta_6'].mean():.4f}, Delta_8: {df_targets_train['target_delta_8'].mean():.4f}")"""))

    # Section 2
    cells.append(create_markdown_cell("""## Section 2 — Split / Leakage Audit
Verify strict paper-disjoint separation between TRAIN-FIT and INTERNAL-VAL manifests.
Ensure no target labels or downstream generation statistics leak into feature representations."""))

    cells.append(create_code_cell("""# Verify Split Isolation
fit_manifest_path = PROJECT_ROOT / cfg["paths"]["train_fit_manifest"]
val_manifest_path = PROJECT_ROOT / cfg["paths"]["internal_val_manifest"]

df_fit_man, df_val_man = load_w7d_splits(fit_manifest_path, val_manifest_path)
print(f"Validated paper-disjoint split: TRAIN-FIT={len(df_fit_man)} queries, INTERNAL-VAL={len(df_val_man)} queries.")

# Strict Guard: TEST split must be inaccessible
try:
    assert_test_disabled("test")
    print("ERROR: Test guard failed to raise!")
except RuntimeError as e:
    print("Test guard verified active:", e)

# Filter targets into TRAIN-FIT and INTERNAL-VAL
df_targets_fit = df_targets_train[df_targets_train["question_id"].isin(df_fit_man["question_id"])].copy().reset_index(drop=True)
df_targets_val = df_targets_train[df_targets_train["question_id"].isin(df_val_man["question_id"])].copy().reset_index(drop=True)
print(f"Split targets: FIT={len(df_targets_fit)}, VAL={len(df_targets_val)}.")"""))

    # Section 3
    cells.append(create_markdown_cell("""## Section 3 — Critical Input Length Audit
Audit token and word length statistics for $D_1 + \\dots + D_{10} + q$ across TRAIN and DEV.
Verify whether global end-truncation would erase ranks 5–10, and establish per-document truncation limits."""))

    cells.append(create_code_cell("""# Load full retrieval passages for audit
train_retrieval = load_retrieval_passages(PROJECT_ROOT / cfg["paths"]["train_retrieval"], split_name="train")
dev_retrieval = load_retrieval_passages(PROJECT_ROOT / cfg["paths"]["dev_retrieval"], split_name="dev")

audit_stats, df_length_audit = audit_input_lengths(train_retrieval, dev_retrieval, max_passages=10)
audit_table_path = PROJECT_ROOT / cfg["paths"]["tables_dir"] / "w7d_input_length_audit.csv"
df_length_audit.to_csv(audit_table_path, index=False)
print("Input Length Audit Results:")
display(df_length_audit)"""))

    # Section 4
    cells.append(create_markdown_cell("""## Section 4 — D0 Frozen Cross-Encoder Setup
Initialize `CrossEncoderScorer` using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
Confirm parameters are strictly frozen (`requires_grad = False`)."""))

    cells.append(create_code_cell("""device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}, Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

ce_model_name = cfg["models"]["cross_encoder_model"]
scorer = CrossEncoderScorer(model_name=ce_model_name, device=device, batch_size=cfg["models"]["cross_encoder_batch_size"])

# Check frozen status
for param in scorer.model.model.parameters():
    assert not param.requires_grad, "Encoder parameter must be frozen for D0!"
print(f"Cross-encoder backbone successfully loaded and frozen: {ce_model_name}")"""))

    # Section 5
    cells.append(create_markdown_cell("""## Section 5 — Resumable Cross-Encoder Score Cache
Compute or resume caching of joint $(q, D_i)$ cross-encoder scores for all 10 passages.
Cache path: `results/week7/w7d_cross_encoder_router/cross_encoder/train_cross_encoder_scores.parquet`."""))

    cells.append(create_code_cell("""train_cache_path = PROJECT_ROOT / cfg["paths"]["cross_encoder_dir"] / "train_cross_encoder_scores.parquet"
dev_cache_path = PROJECT_ROOT / cfg["paths"]["cross_encoder_dir"] / "dev_cross_encoder_scores.parquet"

print("Generating / loading TRAIN cross-encoder score cache...")
df_ce_scores_train = scorer.compute_and_cache_scores(
    train_retrieval, cache_path=train_cache_path, split_name="train", max_passages=10
)

print("Generating / loading DEV cross-encoder score cache...")
df_ce_scores_dev = scorer.compute_and_cache_scores(
    dev_retrieval, cache_path=dev_cache_path, split_name="dev", max_passages=10
)

print(f"Score Cache Loaded: TRAIN={len(df_ce_scores_train)} rows ({len(df_ce_scores_train['question_id'].unique())} queries), "
      f"DEV={len(df_ce_scores_dev)} rows ({len(df_ce_scores_dev['question_id'].unique())} queries).")

# Plot score distribution by semantic rank
plt.figure(figsize=(10, 4))
sns.boxplot(data=df_ce_scores_train, x="semantic_rank", y="cross_encoder_score", palette="viridis")
plt.title("D0 Cross-Encoder Score Distribution Across Semantic Ranks 1–10 (TRAIN)")
plt.xlabel("Semantic Rank")
plt.ylabel("Cross-Encoder Score")
plt.tight_layout()
fig_dist_path = PROJECT_ROOT / cfg["paths"]["figures_dir"] / "w7d_cross_encoder_score_distribution.png"
plt.savefig(fig_dist_path, dpi=300)
plt.show()"""))

    # Section 6
    cells.append(create_markdown_cell("""## Section 6 — D0 Feature Construction
Extract 22 summary interaction features from cached cross-encoder scores:
* Top-$k$ means (top 1, 2, 4, 6, 10)
* Tail & boundary gaps ($k=2, 4$)
* Specific rank probes (rank 5, 6, 8)
* Spearman rank correlation with semantic & BM25 orderings
* Mean absolute rank displacement & top-4 overlap
* Merge with existing W7-C feature bank."""))

    cells.append(create_code_cell("""# Load base W7-C feature bank if available
w7c_train_path = PROJECT_ROOT / cfg["paths"]["w7c_train_features"]
w7c_dev_path = PROJECT_ROOT / cfg["paths"]["w7c_dev_features"]

df_w7c_train = pd.read_parquet(w7c_train_path) if w7c_train_path.is_file() else None
df_w7c_dev = pd.read_parquet(w7c_dev_path) if w7c_dev_path.is_file() else None

df_d0_feats_train = build_d0_feature_dataframe(df_ce_scores_train, df_base_features=df_w7c_train)
df_d0_feats_dev = build_d0_feature_dataframe(df_ce_scores_dev, df_base_features=df_w7c_dev)

feat_cols = [c for c in df_d0_feats_train.columns if c.startswith("feat_")]
assert_zero_feature_leakage(feat_cols)
print(f"Extracted {len(feat_cols)} total leakage-free features for D0.")

# Separate into TRAIN-FIT and INTERNAL-VAL
df_d0_fit = df_d0_feats_train[df_d0_feats_train["question_id"].isin(df_fit_man["question_id"])].copy().reset_index(drop=True)
df_d0_val = df_d0_feats_train[df_d0_feats_train["question_id"].isin(df_val_man["question_id"])].copy().reset_index(drop=True)

# Align targets
df_targets_fit = df_targets_fit.set_index("question_id").loc[df_d0_fit["question_id"]].reset_index()
df_targets_val = df_targets_val.set_index("question_id").loc[df_d0_val["question_id"]].reset_index()
print(f"D0 Feature DataFrames: FIT={df_d0_fit.shape}, VAL={df_d0_val.shape}, DEV={df_d0_feats_dev.shape}.")"""))

    # Section 7
    cells.append(create_markdown_cell("""## Section 7 — D0 Policy Training & INTERNAL-VAL Calibration
Train `D0TabularRouter` on TRAIN-FIT and grid search anchor-and-deviate policy thresholds on INTERNAL-VAL."""))

    cells.append(create_code_cell("""d0_router = D0TabularRouter(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    min_child_samples=20,
    random_state=cfg["experiment"]["random_seed"],
)

print("Training D0 GBDT models on TRAIN-FIT...")
d0_router.fit(df_d0_fit[feat_cols], df_targets_fit)

print("Predicting on INTERNAL-VAL...")
d0_val_preds = d0_router.predict(df_d0_val[feat_cols])

# Grid search policy thresholds on INTERNAL-VAL
tau_downs = np.linspace(0.60, 0.95, 8)
tau_ups = np.linspace(0.50, 0.90, 9)
delta_ups = [0.00, 0.005, 0.010, 0.015]

grid_records = []
val_qids = list(df_d0_val["question_id"])

for td in tau_downs:
    for tu in tau_ups:
        for du in delta_ups:
            pred_ks = {}
            for i, qid in enumerate(val_qids):
                deltas = {
                    5: d0_val_preds["pred_delta_5"][i],
                    6: d0_val_preds["pred_delta_6"][i],
                    8: d0_val_preds["pred_delta_8"][i],
                }
                pred_ks[qid] = apply_w7d_anchor_deviate_policy(
                    p_down=d0_val_preds["p_down"][i],
                    p_up=d0_val_preds["p_up"][i],
                    pred_deltas=deltas,
                    tau_down=td,
                    tau_up=tu,
                    delta_up=du,
                    anchor_k=ANCHOR_K,
                )
            metrics = evaluate_downstream_fidelity_policy(pred_ks, df_targets_val, budgets=BUDGETS, anchor_k=ANCHOR_K)
            metrics.update({"tau_down": td, "tau_up": tu, "delta_up": du, "model": "D0_GBDT"})
            grid_records.append(metrics)

df_d0_val_grid = pd.DataFrame(grid_records)
df_d0_val_grid.to_csv(PROJECT_ROOT / cfg["paths"]["tables_dir"] / "w7d_d0_internal_val.csv", index=False)

# Static k=4 reference F1 on INTERNAL-VAL
f1_k4_ref = float(df_targets_val["f1_k4"].mean())
winner_d0, _ = select_best_fidelity_policy(df_d0_val_grid, f1_k4_reference=f1_k4_ref, eta=cfg["experiment"]["quality_tolerance_eta"])

print(f"Static k=4 Reference F1 (INTERNAL-VAL): {f1_k4_ref:.4f}")
print("D0 Best Policy on INTERNAL-VAL:")
print(f"  Status: {winner_d0.get('selection_status')}")
print(f"  tau_down={winner_d0['tau_down']:.2f}, tau_up={winner_d0['tau_up']:.2f}, delta_up={winner_d0['delta_up']:.3f}")
print(f"  mean F1: {winner_d0['mean_f1']:.4f}, mean tokens: {winner_d0['mean_tokens']:.1f}, mean k: {winner_d0['mean_k']:.2f}")"""))

    # Section 8
    cells.append(create_markdown_cell("""## Section 8 — D1 Hierarchical Cross-Encoder Architecture
Architecture specification:
$$\\tilde{h}_i = [h_i; \\text{rank}^{sem}_i; \\text{rank}^{bm25}_i; \\text{score}^{sem}_i; \\text{score}^{bm25}_i; \\text{length}_i]$$
$$H' = \\text{TransformerEncoder}(\\tilde{H})$$
$$r_q = \\text{AttentionPool}(H')$$
$$r_q \\to \\text{MultiTaskUtilityHeads}(\\hat{y}_{down}, \\hat{y}_{up}, \\hat{\\Delta}_k, \\hat{z}_k, \\hat{k}_{aux})$$"""))

    cells.append(create_code_cell("""d1_cfg = cfg["models"]["d1_hierarchical"]
d1_model = HierarchicalCrossEncoderRouter(
    encoder_name=d1_cfg["encoder_name"],
    freeze_encoder=True,
    unfreeze_last_n_layers=0,
    d_model=d1_cfg["d_model"],
    n_heads=d1_cfg["aggregator"]["n_heads"],
    n_layers=d1_cfg["aggregator"]["n_layers"],
    dim_feedforward=d1_cfg["aggregator"]["dim_feedforward"],
    dropout=d1_cfg["aggregator"]["dropout"],
    meta_dim=d1_cfg["meta_dim"],
    enable_aux_k=True,
    enable_passage_utility=cfg["models"]["enable_passage_utility_head"],
)

total_p = sum(p.numel() for p in d1_model.parameters())
trainable_p = sum(p.numel() for p in d1_model.parameters() if p.requires_grad)
print(f"D1 Model Instantiated: Total Parameters={total_p:,}, Trainable={trainable_p:,} ({trainable_p/total_p*100:.2f}%)")"""))

    # Section 9
    cells.append(create_markdown_cell("""## Section 9 — D1 Dataset Construction & Representation Caching
To enable rapid training of the aggregator and multi-task heads, we precompute and cache the 10 passage representations $H \\in \\mathbb{R}^{N \\times 10 \\times 384}$ using the frozen cross-encoder."""))

    cells.append(create_code_cell("""from transformers import AutoTokenizer

rep_dir = PROJECT_ROOT / cfg["paths"]["representations_dir"]
train_emb_cache = rep_dir / "train_cross_encoder_embeddings.pt"
dev_emb_cache = rep_dir / "dev_cross_encoder_embeddings.pt"

tok = AutoTokenizer.from_pretrained(d1_cfg["encoder_name"])

def build_or_load_embeddings(retrieval_dict, targets_df, cache_path, max_passages=10):
    if cache_path.is_file():
        print(f"Loading cached embeddings from {cache_path}...")
        data = torch.load(cache_path)
        return data["embeddings"], data["metadata"]

    print(f"Computing passage embeddings for {len(targets_df)} queries...")
    d1_model.eval()
    d1_model.to(device)

    all_embs = []
    all_metas = []
    qids = list(targets_df["question_id"])

    with torch.no_grad():
        for qid in qids:
            q_info = retrieval_dict[qid]
            q_text = q_info["question"]
            passages = q_info["passages"][:max_passages]

            pairs = [(q_text, p["text"]) for p in passages]
            metas = [[
                float(p["semantic_rank"]),
                float(p["bm25_rank"]),
                float(p["semantic_rank"])**-1,
                float(p["bm25_score"]),
                float(len(p["text"].split())),
            ] for p in passages]

            enc = tok([p[0] for p in pairs], [p[1] for p in pairs], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            h = d1_model.extract_passage_embeddings(enc["input_ids"], enc["attention_mask"])
            all_embs.append(h.cpu())
            all_metas.append(torch.tensor(metas, dtype=torch.float32))

    embs_tensor = torch.stack(all_embs, dim=0)   # (N, 10, d_enc)
    metas_tensor = torch.stack(all_metas, dim=0) # (N, 10, meta_dim)

    torch.save({"embeddings": embs_tensor, "metadata": metas_tensor}, cache_path)
    print(f"Saved embeddings to {cache_path}: {embs_tensor.shape}")
    return embs_tensor, metas_tensor

train_embs, train_metas = build_or_load_embeddings(train_retrieval, df_targets_train, train_emb_cache)
dev_embs, dev_metas = build_or_load_embeddings(dev_retrieval, df_targets_dev, dev_emb_cache)

# Slice into TRAIN-FIT and INTERNAL-VAL
fit_indices = [i for i, qid in enumerate(df_targets_train["question_id"]) if qid in set(df_fit_man["question_id"])]
val_indices = [i for i, qid in enumerate(df_targets_train["question_id"]) if qid in set(df_val_man["question_id"])]

fit_dataset = CachedEmbeddingDataset(train_embs[fit_indices], train_metas[fit_indices], df_targets_fit)
val_dataset = CachedEmbeddingDataset(train_embs[val_indices], train_metas[val_indices], df_targets_val)
dev_dataset = CachedEmbeddingDataset(dev_embs, dev_metas, df_targets_dev)

print(f"Datasets built: FIT={len(fit_dataset)}, VAL={len(val_dataset)}, DEV={len(dev_dataset)}.")"""))

    # Section 10
    cells.append(create_markdown_cell("""## Section 10 — D1 Training Configuration
Configure DataLoader, AdamW optimizer, and multi-task loss weights:
$$\\mathcal{L} = \\lambda_d \\mathcal{L}_{down} + \\lambda_u \\mathcal{L}_{up} + \\lambda_\\Delta \\mathcal{L}_{\\Delta} + \\lambda_z \\mathcal{L}_{adm} + \\lambda_k \\mathcal{L}_k$$"""))

    cells.append(create_code_cell("""batch_size = d1_cfg["training"]["batch_size"]
train_loader = DataLoader(fit_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

loss_weights = d1_cfg["loss_weights"]
print("Loss weights:", loss_weights)
print("Training configuration ready.")"""))

    # Section 11
    cells.append(create_markdown_cell("""## Section 11 — D1-A TRAIN-FIT Training (Frozen Backbone)
Train the set aggregator and multi-task heads with validation tracking and early stopping."""))

    cells.append(create_code_cell("""d1_model, history = train_d1_aggregator(
    model=d1_model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=d1_cfg["training"]["epochs"],
    learning_rate=float(d1_cfg["training"]["learning_rate"]),
    weight_decay=float(d1_cfg["training"]["weight_decay"]),
    loss_weights=loss_weights,
    early_stopping_patience=d1_cfg["training"]["early_stopping_patience"],
    device=device,
)

# Plot training history
df_hist = pd.DataFrame(history)
plt.figure(figsize=(8, 4))
plt.plot(df_hist["epoch"], df_hist["train_loss"], label="Train Loss", marker="o")
plt.plot(df_hist["epoch"], df_hist["val_loss"], label="Validation Loss", marker="s")
plt.xlabel("Epoch")
plt.ylabel("Multi-task Loss")
plt.title("D1-A Hierarchical Cross-Encoder Training & Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Save best model
best_model_path_d1a = PROJECT_ROOT / cfg["paths"]["models_dir"] / "d1a_hierarchical_router_best.pt"
torch.save(d1_model.state_dict(), best_model_path_d1a)
print(f"Saved D1-A model checkpoint to {best_model_path_d1a}")"""))

    # Section 12
    cells.append(create_markdown_cell("""## Section 12 — D1-A INTERNAL-VAL Evaluation
Generate predictions for INTERNAL-VAL: $p_{down}, p_{up}, \\hat{\\Delta}_k, \\hat{z}_k$."""))

    cells.append(create_code_cell("""d1_model.eval()
val_preds_d1a = {"p_down": [], "p_up": [], "pred_deltas": [], "admissibility": []}

with torch.no_grad():
    for batch in val_loader:
        emb = batch["embeddings"].to(device)
        meta = batch["metadata"].to(device)
        out = d1_model.forward_from_cached_embeddings(emb, meta)
        val_preds_d1a["p_down"].extend(torch.sigmoid(out["logit_down"]).cpu().view(-1).numpy())
        val_preds_d1a["p_up"].extend(torch.sigmoid(out["logit_up"]).cpu().view(-1).numpy())
        val_preds_d1a["pred_deltas"].extend(out["pred_deltas"].cpu().numpy())
        val_preds_d1a["admissibility"].extend(torch.sigmoid(out["logit_admissibility"]).cpu().numpy())

val_preds_d1a["p_down"] = np.array(val_preds_d1a["p_down"])
val_preds_d1a["p_up"] = np.array(val_preds_d1a["p_up"])
val_preds_d1a["pred_deltas"] = np.array(val_preds_d1a["pred_deltas"])
val_preds_d1a["admissibility"] = np.array(val_preds_d1a["admissibility"])

print(f"D1-A INTERNAL-VAL predictions complete. p_down mean: {val_preds_d1a['p_down'].mean():.4f}, p_up mean: {val_preds_d1a['p_up'].mean():.4f}")"""))

    # Section 13
    cells.append(create_markdown_cell("""## Section 13 — D1-A Policy Calibration
Calibrate anchor-and-deviate policy thresholds on INTERNAL-VAL predictions and select best configuration under $\\eta=0.005$ quality tolerance."""))

    cells.append(create_code_cell("""grid_records_d1a = []
val_qids = list(df_targets_val["question_id"])

for td in tau_downs:
    for tu in tau_ups:
        for du in delta_ups:
            pred_ks = {}
            for i, qid in enumerate(val_qids):
                deltas = {
                    5: val_preds_d1a["pred_deltas"][i, 0],
                    6: val_preds_d1a["pred_deltas"][i, 1],
                    8: val_preds_d1a["pred_deltas"][i, 2],
                }
                pred_ks[qid] = apply_w7d_anchor_deviate_policy(
                    p_down=val_preds_d1a["p_down"][i],
                    p_up=val_preds_d1a["p_up"][i],
                    pred_deltas=deltas,
                    tau_down=td,
                    tau_up=tu,
                    delta_up=du,
                    anchor_k=ANCHOR_K,
                )
            metrics = evaluate_downstream_fidelity_policy(pred_ks, df_targets_val, budgets=BUDGETS, anchor_k=ANCHOR_K)
            metrics.update({"tau_down": td, "tau_up": tu, "delta_up": du, "model": "D1A_Hierarchical_Frozen"})
            grid_records_d1a.append(metrics)

df_d1a_val_grid = pd.DataFrame(grid_records_d1a)
df_d1a_val_grid.to_csv(PROJECT_ROOT / cfg["paths"]["tables_dir"] / "w7d_d1a_internal_val.csv", index=False)

winner_d1a, _ = select_best_fidelity_policy(df_d1a_val_grid, f1_k4_reference=f1_k4_ref, eta=cfg["experiment"]["quality_tolerance_eta"])
print("D1-A Best Policy on INTERNAL-VAL:")
print(f"  Status: {winner_d1a.get('selection_status')}")
print(f"  tau_down={winner_d1a['tau_down']:.2f}, tau_up={winner_d1a['tau_up']:.2f}, delta_up={winner_d1a['delta_up']:.3f}")
print(f"  mean F1: {winner_d1a['mean_f1']:.4f}, mean tokens: {winner_d1a['mean_tokens']:.1f}, mean k: {winner_d1a['mean_k']:.2f}")"""))

    # Section 14
    cells.append(create_markdown_cell("""## Section 14 — D1-B Partially Unfrozen Hierarchical Cross-Encoder (End-to-End)
Unfreeze the top 2 layers of the cross-encoder backbone and train end-to-end with the aggregator and multi-task heads.
Evaluated on INTERNAL-VAL and calibrated using the same anchor-and-deviate protocol."""))

    cells.append(create_code_cell("""print("Preparing D1-B end-to-end tokenized dataset...")
train_tok_cache = rep_dir / "train_cross_encoder_tokens.pt"
dev_tok_cache = rep_dir / "dev_cross_encoder_tokens.pt"

def build_or_load_tokenized_dataset(retrieval_dict, targets_df, cache_path, max_passages=10):
    if cache_path.is_file():
        print(f"Loading cached tokenized tensors from {cache_path}...")
        data = torch.load(cache_path)
        return data["input_ids"], data["attention_mask"], data["metadata"]

    print(f"Tokenizing query-passage pairs for {len(targets_df)} queries...")
    all_input_ids, all_masks, all_metas = [], [], []

    for qid in targets_df["question_id"]:
        q_info = retrieval_dict[qid]
        q_text = q_info["question"]
        passages = q_info["passages"][:max_passages]

        pairs = [(q_text, p["text"]) for p in passages]
        metas = [[
            float(p["semantic_rank"]),
            float(p["bm25_rank"]),
            float(p["semantic_rank"])**-1,
            float(p["bm25_score"]),
            float(len(p["text"].split())),
        ] for p in passages]

        enc = tok([p[0] for p in pairs], [p[1] for p in pairs], padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        all_input_ids.append(enc["input_ids"].unsqueeze(0))      # (1, 10, 128)
        all_masks.append(enc["attention_mask"].unsqueeze(0))     # (1, 10, 128)
        all_metas.append(torch.tensor(metas, dtype=torch.float32).unsqueeze(0))

    input_ids_t = torch.cat(all_input_ids, dim=0)
    masks_t = torch.cat(all_masks, dim=0)
    metas_t = torch.cat(all_metas, dim=0)

    torch.save({"input_ids": input_ids_t, "attention_mask": masks_t, "metadata": metas_t}, cache_path)
    print(f"Saved tokenized tensors to {cache_path}: {input_ids_t.shape}")
    return input_ids_t, masks_t, metas_t

train_ids, train_masks, train_metas_raw = build_or_load_tokenized_dataset(train_retrieval, df_targets_train, train_tok_cache)
dev_ids, dev_masks, dev_metas_raw = build_or_load_tokenized_dataset(dev_retrieval, df_targets_dev, dev_tok_cache)

fit_token_dataset = HierarchicalTokenDataset(train_ids[fit_indices], train_masks[fit_indices], train_metas_raw[fit_indices], df_targets_fit)
val_token_dataset = HierarchicalTokenDataset(train_ids[val_indices], train_masks[val_indices], train_metas_raw[val_indices], df_targets_val)
dev_token_dataset = HierarchicalTokenDataset(dev_ids, dev_masks, dev_metas_raw, df_targets_dev)

train_tok_loader = DataLoader(fit_token_dataset, batch_size=8, shuffle=True)
val_tok_loader = DataLoader(val_token_dataset, batch_size=8, shuffle=False)
dev_tok_loader = DataLoader(dev_token_dataset, batch_size=8, shuffle=False)

# Instantiate D1-B model with unfreeze_last_n_layers = 2
d1b_model = HierarchicalCrossEncoderRouter(
    encoder_name=d1_cfg["encoder_name"],
    freeze_encoder=True,
    unfreeze_last_n_layers=2,
    d_model=d1_cfg["d_model"],
    n_heads=d1_cfg["aggregator"]["n_heads"],
    n_layers=d1_cfg["aggregator"]["n_layers"],
    dim_feedforward=d1_cfg["aggregator"]["dim_feedforward"],
    dropout=d1_cfg["aggregator"]["dropout"],
    meta_dim=d1_cfg["meta_dim"],
    enable_aux_k=True,
    enable_passage_utility=cfg["models"]["enable_passage_utility_head"],
)

trainable_p_d1b = sum(p.numel() for p in d1b_model.parameters() if p.requires_grad)
print(f"D1-B Instantiated: Trainable Parameters={trainable_p_d1b:,}")

print("Training D1-B end-to-end...")
d1b_model, d1b_history = train_d1b_end_to_end(
    model=d1b_model,
    train_loader=train_tok_loader,
    val_loader=val_tok_loader,
    epochs=d1_cfg["training"].get("epochs_d1b", 4),
    encoder_lr=float(d1_cfg["training"].get("learning_rate_d1b_encoder", 2.0e-5)),
    aggregator_lr=float(d1_cfg["training"]["learning_rate"]),
    weight_decay=float(d1_cfg["training"]["weight_decay"]),
    loss_weights=loss_weights,
    device=device,
)

# Predict on INTERNAL-VAL with D1-B
d1b_model.eval()
val_preds_d1b = {"p_down": [], "p_up": [], "pred_deltas": []}
with torch.no_grad():
    for batch in val_tok_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        meta = batch["metadata"].to(device)
        out = d1b_model(input_ids, attention_mask, meta)
        val_preds_d1b["p_down"].extend(torch.sigmoid(out["logit_down"]).cpu().view(-1).numpy())
        val_preds_d1b["p_up"].extend(torch.sigmoid(out["logit_up"]).cpu().view(-1).numpy())
        val_preds_d1b["pred_deltas"].extend(out["pred_deltas"].cpu().numpy())

val_preds_d1b["p_down"] = np.array(val_preds_d1b["p_down"])
val_preds_d1b["p_up"] = np.array(val_preds_d1b["p_up"])
val_preds_d1b["pred_deltas"] = np.array(val_preds_d1b["pred_deltas"])

# Calibrate D1-B policy on INTERNAL-VAL
grid_records_d1b = []
for td in tau_downs:
    for tu in tau_ups:
        for du in delta_ups:
            pred_ks = {}
            for i, qid in enumerate(val_qids):
                deltas = {
                    5: val_preds_d1b["pred_deltas"][i, 0],
                    6: val_preds_d1b["pred_deltas"][i, 1],
                    8: val_preds_d1b["pred_deltas"][i, 2],
                }
                pred_ks[qid] = apply_w7d_anchor_deviate_policy(
                    p_down=val_preds_d1b["p_down"][i],
                    p_up=val_preds_d1b["p_up"][i],
                    pred_deltas=deltas,
                    tau_down=td,
                    tau_up=tu,
                    delta_up=du,
                    anchor_k=ANCHOR_K,
                )
            metrics = evaluate_downstream_fidelity_policy(pred_ks, df_targets_val, budgets=BUDGETS, anchor_k=ANCHOR_K)
            metrics.update({"tau_down": td, "tau_up": tu, "delta_up": du, "model": "D1B_Hierarchical_Unfrozen"})
            grid_records_d1b.append(metrics)

df_d1b_val_grid = pd.DataFrame(grid_records_d1b)
df_d1b_val_grid.to_csv(PROJECT_ROOT / cfg["paths"]["tables_dir"] / "w7d_d1b_internal_val.csv", index=False)

winner_d1b, _ = select_best_fidelity_policy(df_d1b_val_grid, f1_k4_reference=f1_k4_ref, eta=cfg["experiment"]["quality_tolerance_eta"])
print("D1-B Best Policy on INTERNAL-VAL:")
print(f"  Status: {winner_d1b.get('selection_status')}")
print(f"  tau_down={winner_d1b['tau_down']:.2f}, tau_up={winner_d1b['tau_up']:.2f}, delta_up={winner_d1b['delta_up']:.3f}")
print(f"  mean F1: {winner_d1b['mean_f1']:.4f}, mean tokens: {winner_d1b['mean_tokens']:.1f}, mean k: {winner_d1b['mean_k']:.2f}")

best_model_path_d1b = PROJECT_ROOT / cfg["paths"]["models_dir"] / "d1b_hierarchical_router_best.pt"
torch.save(d1b_model.state_dict(), best_model_path_d1b)
print(f"Saved D1-B model checkpoint to {best_model_path_d1b}")"""))

    # Section 15
    cells.append(create_markdown_cell("""## Section 15 — D2 Listwise Input Construction
Construct serialized prompt inputs formatted with XML document blocks and a causal $\\langle\\text{DECIDE}\\rangle$ token for all queries across splits."""))

    cells.append(create_code_cell("""listwise_dir = PROJECT_ROOT / cfg["paths"]["listwise_inputs_dir"]
listwise_manifest_path = listwise_dir / "listwise_prompts_manifest.jsonl"

print("Formatting listwise causal prompts for TRAIN and DEV...")
prompts_dict = {}
with open(listwise_manifest_path, "w", encoding="utf-8") as f_out:
    for qid in df_targets_train["question_id"]:
        q_data = train_retrieval[qid]
        p = format_listwise_causal_prompt(q_data["question"], q_data["passages"], max_passages=10)
        prompts_dict[qid] = p
        f_out.write(json.dumps({"question_id": qid, "split": "train", "prompt": p}) + "\\n")
    for qid in df_targets_dev["question_id"]:
        q_data = dev_retrieval[qid]
        p = format_listwise_causal_prompt(q_data["question"], q_data["passages"], max_passages=10)
        prompts_dict[qid] = p
        f_out.write(json.dumps({"question_id": qid, "split": "dev", "prompt": p}) + "\\n")

print(f"Saved listwise prompt manifest to {listwise_manifest_path} ({len(prompts_dict)} total prompts).")
print('Prompt tail preview:', prompts_dict[list(df_targets_train["question_id"])[0]][-300:])"""))

    # Section 16
    cells.append(create_markdown_cell("""## Section 16 — D2 LoRA/QLoRA Hardware Feasibility Audit
Audit hardware environment (GPU VRAM, bitsandbytes, PEFT) before running the 7B LLM router."""))

    cells.append(create_code_cell("""d2_audit = audit_d2_hardware_feasibility(device_id=0, required_vram_gb=14.0)
print("D2 Hardware Audit Report:")
print(json.dumps(d2_audit, indent=2))"""))

    # Section 17 & 18
    cells.append(create_markdown_cell("""## Section 17 & 18 — D2 Listwise 7B Mistral Router Execution & Calibration
Initialize `ListwiseLLMRouter` with `mistralai/Mistral-7B-Instruct-v0.2` and PEFT LoRA adapter.
Train with gradient accumulation ($B=1$, accumulation steps $=16$), evaluate on INTERNAL-VAL, and calibrate anchor-and-deviate policy."""))

    cells.append(create_code_cell("""d2_cfg = cfg["models"]["d2_listwise_llm"]
d2_model = ListwiseLLMRouter(
    model_name=d2_cfg["model_name"],
    d_model=4096,
    head_hidden_dim=256,
    dropout=0.1,
    use_lora=True,
    load_in_4bit=d2_cfg["quantization"].get("load_in_4bit", True),
    device=device,
)

print("Initializing Listwise LLM base model and LoRA adapter...")
d2_model.initialize_llm()

# Tokenize listwise prompts
d2_tok_cache_train = listwise_dir / "train_listwise_tokens.pt"
d2_tok_cache_dev = listwise_dir / "dev_listwise_tokens.pt"

def build_or_load_listwise_tokens(targets_df, prompts_map, cache_path, max_length=1024):
    if cache_path.is_file():
        print(f"Loading cached listwise tokens from {cache_path}...")
        data = torch.load(cache_path)
        return data["input_ids"], data["attention_mask"]

    print(f"Tokenizing {len(targets_df)} listwise prompts (max_len={max_length})...")
    all_ids, all_masks = [], []
    for qid in targets_df["question_id"]:
        p = prompts_map[qid]
        enc = d2_model.tokenizer(p, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
        all_ids.append(enc["input_ids"])
        all_masks.append(enc["attention_mask"])

    ids_tensor = torch.cat(all_ids, dim=0)
    masks_tensor = torch.cat(all_masks, dim=0)
    torch.save({"input_ids": ids_tensor, "attention_mask": masks_tensor}, cache_path)
    print(f"Saved listwise tokens to {cache_path}: {ids_tensor.shape}")
    return ids_tensor, masks_tensor

d2_train_ids, d2_train_masks = build_or_load_listwise_tokens(df_targets_train, prompts_dict, d2_tok_cache_train, max_length=1024)
d2_dev_ids, d2_dev_masks = build_or_load_listwise_tokens(df_targets_dev, prompts_dict, d2_tok_cache_dev, max_length=1024)

d2_fit_dataset = ListwisePromptDataset(d2_train_ids[fit_indices], d2_train_masks[fit_indices], df_targets_fit)
d2_val_dataset = ListwisePromptDataset(d2_train_ids[val_indices], d2_train_masks[val_indices], df_targets_val)
d2_dev_dataset = ListwisePromptDataset(d2_dev_ids, d2_dev_masks, df_targets_dev)

d2_train_loader = DataLoader(d2_fit_dataset, batch_size=1, shuffle=True)
d2_val_loader = DataLoader(d2_val_dataset, batch_size=1, shuffle=False)
d2_dev_loader = DataLoader(d2_dev_dataset, batch_size=1, shuffle=False)

print(f"D2 DataLoaders ready: FIT={len(d2_fit_dataset)}, VAL={len(d2_val_dataset)}, DEV={len(d2_dev_dataset)}.")

# Train D2 Router
print("Training D2 Listwise 7B Router...")
d2_model, d2_history = train_d2_listwise_router(
    model=d2_model,
    train_loader=d2_train_loader,
    val_loader=d2_val_loader,
    epochs=d2_cfg["training"]["epochs"],
    gradient_accumulation_steps=d2_cfg["training"]["gradient_accumulation_steps"],
    learning_rate=float(d2_cfg["training"]["learning_rate"]),
    weight_decay=float(d2_cfg["training"]["weight_decay"]),
    loss_weights=loss_weights,
    device=device,
)

# Predict on INTERNAL-VAL with D2
d2_model.eval()
val_preds_d2 = {"p_down": [], "p_up": [], "pred_deltas": []}
with torch.no_grad():
    for batch in d2_val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        out = d2_model(input_ids, attention_mask)
        val_preds_d2["p_down"].extend(torch.sigmoid(out["logit_down"]).cpu().view(-1).numpy())
        val_preds_d2["p_up"].extend(torch.sigmoid(out["logit_up"]).cpu().view(-1).numpy())
        val_preds_d2["pred_deltas"].extend(out["pred_deltas"].cpu().numpy())

val_preds_d2["p_down"] = np.array(val_preds_d2["p_down"])
val_preds_d2["p_up"] = np.array(val_preds_d2["p_up"])
val_preds_d2["pred_deltas"] = np.array(val_preds_d2["pred_deltas"])

# Calibrate D2 policy on INTERNAL-VAL
grid_records_d2 = []
for td in tau_downs:
    for tu in tau_ups:
        for du in delta_ups:
            pred_ks = {}
            for i, qid in enumerate(val_qids):
                deltas = {
                    5: val_preds_d2["pred_deltas"][i, 0],
                    6: val_preds_d2["pred_deltas"][i, 1],
                    8: val_preds_d2["pred_deltas"][i, 2],
                }
                pred_ks[qid] = apply_w7d_anchor_deviate_policy(
                    p_down=val_preds_d2["p_down"][i],
                    p_up=val_preds_d2["p_up"][i],
                    pred_deltas=deltas,
                    tau_down=td,
                    tau_up=tu,
                    delta_up=du,
                    anchor_k=ANCHOR_K,
                )
            metrics = evaluate_downstream_fidelity_policy(pred_ks, df_targets_val, budgets=BUDGETS, anchor_k=ANCHOR_K)
            metrics.update({"tau_down": td, "tau_up": tu, "delta_up": du, "model": "D2_Listwise_7B_LoRA"})
            grid_records_d2.append(metrics)

df_d2_val_grid = pd.DataFrame(grid_records_d2)
df_d2_val_grid.to_csv(PROJECT_ROOT / cfg["paths"]["tables_dir"] / "w7d_d2_internal_val.csv", index=False)

winner_d2, _ = select_best_fidelity_policy(df_d2_val_grid, f1_k4_reference=f1_k4_ref, eta=cfg["experiment"]["quality_tolerance_eta"])
print("D2 Best Policy on INTERNAL-VAL:")
print(f"  Status: {winner_d2.get('selection_status')}")
print(f"  tau_down={winner_d2['tau_down']:.2f}, tau_up={winner_d2['tau_up']:.2f}, delta_up={winner_d2['delta_up']:.3f}")
print(f"  mean F1: {winner_d2['mean_f1']:.4f}, mean tokens: {winner_d2['mean_tokens']:.1f}, mean k: {winner_d2['mean_k']:.2f}")

best_model_path_d2 = PROJECT_ROOT / cfg["paths"]["models_dir"] / "d2_listwise_router_best.pt"
torch.save(d2_model.state_dict(), best_model_path_d2)
print(f"Saved D2 model checkpoint to {best_model_path_d2}")"""))

    # Section 19
    cells.append(create_markdown_cell("""## Section 19 — INTERNAL-VAL Model Comparison
Compare all model tiers on INTERNAL-VAL:
1. Static Semantic $k=4$ Baseline
2. D0: Frozen Cross-Encoder + GBDT
3. D1-A: Frozen Backbone Hierarchical Cross-Encoder
4. D1-B: Partially Unfrozen Hierarchical Cross-Encoder (last 2 layers)
5. D2: Listwise 7B Mistral LoRA Router"""))

    cells.append(create_code_cell("""comparison_rows = [
    {
        "Model": "Static Semantic k=4",
        "mean_f1": f1_k4_ref,
        "mean_tokens": float(df_targets_val["tokens_k4"].mean()),
        "mean_k": 4.0,
        "admissible_rate": float((df_targets_val["f1_k4"] >= df_targets_val["target_max_f1"] - 0.01).mean()),
        "tau_down": np.nan, "tau_up": np.nan, "delta_up": np.nan,
    },
    {
        "Model": "D0 Frozen Cross-Encoder + GBDT",
        "mean_f1": winner_d0["mean_f1"],
        "mean_tokens": winner_d0["mean_tokens"],
        "mean_k": winner_d0["mean_k"],
        "admissible_rate": winner_d0["admissible_rate"],
        "tau_down": winner_d0["tau_down"], "tau_up": winner_d0["tau_up"], "delta_up": winner_d0["delta_up"],
    },
    {
        "Model": "D1-A Hierarchical Frozen",
        "mean_f1": winner_d1a["mean_f1"],
        "mean_tokens": winner_d1a["mean_tokens"],
        "mean_k": winner_d1a["mean_k"],
        "admissible_rate": winner_d1a["admissible_rate"],
        "tau_down": winner_d1a["tau_down"], "tau_up": winner_d1a["tau_up"], "delta_up": winner_d1a["delta_up"],
    },
    {
        "Model": "D1-B Hierarchical Unfrozen",
        "mean_f1": winner_d1b["mean_f1"],
        "mean_tokens": winner_d1b["mean_tokens"],
        "mean_k": winner_d1b["mean_k"],
        "admissible_rate": winner_d1b["admissible_rate"],
        "tau_down": winner_d1b["tau_down"], "tau_up": winner_d1b["tau_up"], "delta_up": winner_d1b["delta_up"],
    },
    {
        "Model": "D2 Listwise 7B Mistral LoRA",
        "mean_f1": winner_d2["mean_f1"],
        "mean_tokens": winner_d2["mean_tokens"],
        "mean_k": winner_d2["mean_k"],
        "admissible_rate": winner_d2["admissible_rate"],
        "tau_down": winner_d2["tau_down"], "tau_up": winner_d2["tau_up"], "delta_up": winner_d2["delta_up"],
    },
]

df_model_comp = pd.DataFrame(comparison_rows)
comp_table_path = PROJECT_ROOT / cfg["paths"]["tables_dir"] / "w7d_model_comparison.csv"
df_model_comp.to_csv(comp_table_path, index=False)
print("INTERNAL-VAL Model Comparison:")
display(df_model_comp)"""))

    # Section 20
    cells.append(create_markdown_cell("""## Section 20 — Freeze Winner
Select and freeze the overall winning model and decision policy hyperparameters among D0, D1-A, D1-B, and D2 using Quality-Constrained Cost Minimization ($\eta = 0.005$)."""))

    cells.append(create_code_cell("""# Quality-constrained selection across all 4 candidate models
candidates_val = pd.concat([df_d0_val_grid, df_d1a_val_grid, df_d1b_val_grid, df_d2_val_grid], ignore_index=True)
overall_winner, _ = select_best_fidelity_policy(candidates_val, f1_k4_reference=f1_k4_ref, eta=cfg["experiment"]["quality_tolerance_eta"])

freeze_record = pd.DataFrame([overall_winner])
freeze_path = PROJECT_ROOT / cfg["paths"]["tables_dir"] / "w7d_policy_selection.csv"
freeze_record.to_csv(freeze_path, index=False)

winning_model_name = overall_winner["model"]
tau_down_opt = overall_winner["tau_down"]
tau_up_opt = overall_winner["tau_up"]
delta_up_opt = overall_winner["delta_up"]

print(f"FROZEN WINNING MODEL: {winning_model_name}")
print(f"Hyperparameters: tau_down={tau_down_opt:.2f}, tau_up={tau_up_opt:.2f}, delta_up={delta_up_opt:.3f}")"""))

    # Section 21
    cells.append(create_markdown_cell("""## Section 21 — Final DEV Evaluation Across All Models
Evaluate all four models (D0, D1-A, D1-B, D2) and the frozen winner on the unseen DEV set ($231$ queries)."""))

    cells.append(create_code_cell("""print("Running final DEV evaluation for all models...")
dev_qids = list(df_targets_dev["question_id"])

# 1. D0 DEV Predictions
d0_dev_preds = d0_router.predict(df_d0_feats_dev[feat_cols])
d0_dev_ks = {}
for i, qid in enumerate(dev_qids):
    deltas = {5: d0_dev_preds["pred_delta_5"][i], 6: d0_dev_preds["pred_delta_6"][i], 8: d0_dev_preds["pred_delta_8"][i]}
    d0_dev_ks[qid] = apply_w7d_anchor_deviate_policy(
        d0_dev_preds["p_down"][i], d0_dev_preds["p_up"][i], deltas,
        tau_down=winner_d0["tau_down"], tau_up=winner_d0["tau_up"], delta_up=winner_d0["delta_up"], anchor_k=ANCHOR_K
    )
res_d0 = evaluate_downstream_fidelity_policy(d0_dev_ks, df_targets_dev, budgets=BUDGETS, anchor_k=ANCHOR_K)
res_d0["model"] = "D0_GBDT"

# 2. D1-A DEV Predictions
d1_model.eval()
d1a_dev_p_down, d1a_dev_p_up, d1a_dev_deltas = [], [], []
with torch.no_grad():
    for batch in dev_loader:
        out = d1_model.forward_from_cached_embeddings(batch["embeddings"].to(device), batch["metadata"].to(device))
        d1a_dev_p_down.extend(torch.sigmoid(out["logit_down"]).cpu().view(-1).numpy())
        d1a_dev_p_up.extend(torch.sigmoid(out["logit_up"]).cpu().view(-1).numpy())
        d1a_dev_deltas.extend(out["pred_deltas"].cpu().numpy())

d1a_dev_p_down = np.array(d1a_dev_p_down)
d1a_dev_p_up = np.array(d1a_dev_p_up)
d1a_dev_deltas = np.array(d1a_dev_deltas)

d1a_dev_ks = {}
for i, qid in enumerate(dev_qids):
    deltas = {5: d1a_dev_deltas[i, 0], 6: d1a_dev_deltas[i, 1], 8: d1a_dev_deltas[i, 2]}
    d1a_dev_ks[qid] = apply_w7d_anchor_deviate_policy(
        d1a_dev_p_down[i], d1a_dev_p_up[i], deltas,
        tau_down=winner_d1a["tau_down"], tau_up=winner_d1a["tau_up"], delta_up=winner_d1a["delta_up"], anchor_k=ANCHOR_K
    )
res_d1a = evaluate_downstream_fidelity_policy(d1a_dev_ks, df_targets_dev, budgets=BUDGETS, anchor_k=ANCHOR_K)
res_d1a["model"] = "D1A_Hierarchical_Frozen"

# 3. D1-B DEV Predictions
d1b_model.eval()
d1b_dev_p_down, d1b_dev_p_up, d1b_dev_deltas = [], [], []
with torch.no_grad():
    for batch in dev_tok_loader:
        out = d1b_model(batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["metadata"].to(device))
        d1b_dev_p_down.extend(torch.sigmoid(out["logit_down"]).cpu().view(-1).numpy())
        d1b_dev_p_up.extend(torch.sigmoid(out["logit_up"]).cpu().view(-1).numpy())
        d1b_dev_deltas.extend(out["pred_deltas"].cpu().numpy())

d1b_dev_p_down = np.array(d1b_dev_p_down)
d1b_dev_p_up = np.array(d1b_dev_p_up)
d1b_dev_deltas = np.array(d1b_dev_deltas)

d1b_dev_ks = {}
for i, qid in enumerate(dev_qids):
    deltas = {5: d1b_dev_deltas[i, 0], 6: d1b_dev_deltas[i, 1], 8: d1b_dev_deltas[i, 2]}
    d1b_dev_ks[qid] = apply_w7d_anchor_deviate_policy(
        d1b_dev_p_down[i], d1b_dev_p_up[i], deltas,
        tau_down=winner_d1b["tau_down"], tau_up=winner_d1b["tau_up"], delta_up=winner_d1b["delta_up"], anchor_k=ANCHOR_K
    )
res_d1b = evaluate_downstream_fidelity_policy(d1b_dev_ks, df_targets_dev, budgets=BUDGETS, anchor_k=ANCHOR_K)
res_d1b["model"] = "D1B_Hierarchical_Unfrozen"

# 4. D2 DEV Predictions
d2_model.eval()
d2_dev_p_down, d2_dev_p_up, d2_dev_deltas = [], [], []
with torch.no_grad():
    for batch in d2_dev_loader:
        out = d2_model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        d2_dev_p_down.extend(torch.sigmoid(out["logit_down"]).cpu().view(-1).numpy())
        d2_dev_p_up.extend(torch.sigmoid(out["logit_up"]).cpu().view(-1).numpy())
        d2_dev_deltas.extend(out["pred_deltas"].cpu().numpy())

d2_dev_p_down = np.array(d2_dev_p_down)
d2_dev_p_up = np.array(d2_dev_p_up)
d2_dev_deltas = np.array(d2_dev_deltas)

d2_dev_ks = {}
for i, qid in enumerate(dev_qids):
    deltas = {5: d2_dev_deltas[i, 0], 6: d2_dev_deltas[i, 1], 8: d2_dev_deltas[i, 2]}
    d2_dev_ks[qid] = apply_w7d_anchor_deviate_policy(
        d2_dev_p_down[i], d2_dev_p_up[i], deltas,
        tau_down=winner_d2["tau_down"], tau_up=winner_d2["tau_up"], delta_up=winner_d2["delta_up"], anchor_k=ANCHOR_K
    )
res_d2 = evaluate_downstream_fidelity_policy(d2_dev_ks, df_targets_dev, budgets=BUDGETS, anchor_k=ANCHOR_K)
res_d2["model"] = "D2_Listwise_7B_LoRA"

# Compile All DEV Results
df_dev_all = pd.DataFrame([res_d0, res_d1a, res_d1b, res_d2])
dev_results_path = PROJECT_ROOT / cfg["paths"]["tables_dir"] / "w7d_main_dev_results.csv"
df_dev_all.to_csv(dev_results_path, index=False)
print("Main DEV Results Across All Models:")
display(df_dev_all)

# Selected winner mapping for downstream analyses
winner_pred_ks = {"D0_GBDT": d0_dev_ks, "D1A_Hierarchical_Frozen": d1a_dev_ks, "D1B_Hierarchical_Unfrozen": d1b_dev_ks, "D2_Listwise_7B_LoRA": d2_dev_ks}[winning_model_name]
winner_p_down = {"D0_GBDT": d0_dev_preds["p_down"], "D1A_Hierarchical_Frozen": d1a_dev_p_down, "D1B_Hierarchical_Unfrozen": d1b_dev_p_down, "D2_Listwise_7B_LoRA": d2_dev_p_down}[winning_model_name]
winner_p_up = {"D0_GBDT": d0_dev_preds["p_up"], "D1A_Hierarchical_Frozen": d1a_dev_p_up, "D1B_Hierarchical_Unfrozen": d1b_dev_p_up, "D2_Listwise_7B_LoRA": d2_dev_p_up}[winning_model_name]

df_targets_dev_plot = df_targets_dev.copy()
df_targets_dev_plot["pred_k"] = [winner_pred_ks[qid] for qid in df_targets_dev_plot["question_id"]]

plt.figure(figsize=(7, 4))
df_targets_dev_plot["pred_k"].value_counts().sort_index().plot(kind="bar", color="teal", edgecolor="black")
plt.title(f"Predicted Budget Distribution on DEV ({winning_model_name})")
plt.xlabel("Allocated Budget k")
plt.ylabel("Number of Queries")
plt.tight_layout()
fig_k_dist = PROJECT_ROOT / cfg["paths"]["figures_dir"] / "w7d_predicted_k_distribution.png"
plt.savefig(fig_k_dist, dpi=300)
plt.show()"""))

    # Section 22
    cells.append(create_markdown_cell("""## Section 22 — Downshift Safety Analysis
Audit queries downshifted to $k=2$: precision of admissibility, regret, and context tokens saved."""))

    cells.append(create_code_cell("""downshift_metrics = compute_downshift_safety_metrics(df_targets_dev_plot, budgets=BUDGETS, anchor_k=ANCHOR_K)
df_downshift = pd.DataFrame([downshift_metrics])
downshift_path = PROJECT_ROOT / cfg["paths"]["tables_dir"] / "w7d_downshift_analysis.csv"
df_downshift.to_csv(downshift_path, index=False)
print("Downshift Safety Analysis on DEV:")
display(df_downshift)

# Calibration plot
plt.figure(figsize=(6, 4))
plt.hist(winner_p_down, bins=20, color="cornflowerblue", edgecolor="black")
plt.axvline(tau_down_opt, color="red", linestyle="--", label=f"tau_down={tau_down_opt:.2f}")
plt.title("Downshift Probability Calibration (DEV)")
plt.xlabel("P(Downshift to k=2)")
plt.ylabel("Query Count")
plt.legend()
plt.tight_layout()
plt.savefig(PROJECT_ROOT / cfg["paths"]["figures_dir"] / "w7d_downshift_calibration.png", dpi=300)
plt.show()"""))

    # Section 23
    cells.append(create_markdown_cell("""## Section 23 — Upshift Precision Analysis
Audit queries upshifted beyond $k=4$: precision of $>0.01$ gains and token costs incurred."""))

    cells.append(create_code_cell("""upshift_metrics = compute_upshift_precision_metrics(df_targets_dev_plot, anchor_k=ANCHOR_K)
df_upshift = pd.DataFrame([upshift_metrics])
upshift_path = PROJECT_ROOT / cfg["paths"]["tables_dir"] / "w7d_upshift_analysis.csv"
df_upshift.to_csv(upshift_path, index=False)
print("Upshift Precision Analysis on DEV:")
display(df_upshift)

# Calibration plot
plt.figure(figsize=(6, 4))
plt.hist(winner_p_up, bins=20, color="salmon", edgecolor="black")
plt.axvline(tau_up_opt, color="darkred", linestyle="--", label=f"tau_up={tau_up_opt:.2f}")
plt.title("Upshift Probability Calibration (DEV)")
plt.xlabel("P(Upshift beyond k=4)")
plt.ylabel("Query Count")
plt.legend()
plt.tight_layout()
plt.savefig(PROJECT_ROOT / cfg["paths"]["figures_dir"] / "w7d_upshift_calibration.png", dpi=300)
plt.show()"""))

    # Section 24
    cells.append(create_markdown_cell("""## Section 24 — Quality-Context Pareto Frontier
Compare all W7-D model tiers against all historical allocation policies and static baselines:
* Static semantic $k \\in \\{2, 4, 6, 8\\}$
* W7-B, W7-C1, W7-C2, W7-C3
* W7-D: D0, D1-A, D1-B, D2
* Semantic Epsilon-Oracle"""))

    cells.append(create_code_cell("""frontier_records = []
for k in BUDGETS:
    f1 = float(df_targets_dev[f"f1_k{k}"].mean())
    tok = float(df_targets_dev[f"tokens_k{k}"].mean())
    frontier_records.append({"Method": f"Static k={k}", "F1": f1, "Tokens": tok, "Type": "Static Baseline"})

# Add Oracle
f1_oracle = float(df_targets_dev["target_max_f1"].mean())
oracle_ks = df_targets_dev["target_oracle_k"].values
tok_oracle = float(np.mean([df_targets_dev.iloc[i][f"tokens_k{k}"] for i, k in enumerate(oracle_ks)]))
frontier_records.append({"Method": "Semantic Epsilon-Oracle", "F1": f1_oracle, "Tokens": tok_oracle, "Type": "Upper Bound"})

# Add W7-C3 if available
w7c3_path = PROJECT_ROOT / cfg["paths"]["w7c3_main_results"]
if Path(w7c3_path).is_file():
    df_w7c3_all = pd.read_csv(w7c3_path)
    row_c3 = df_w7c3_all[df_w7c3_all["method"].str.contains("W7-C3", na=False)]
    if not row_c3.empty:
        r_c3 = row_c3.iloc[0]
        tok_c3 = float(r_c3["tokens"]) if "tokens" in r_c3 else float(r_c3["mean_tokens"])
        f1_c3 = float(r_c3["mean_f1"]) if "mean_f1" in r_c3 else float(r_c3["f1"])
        frontier_records.append({"Method": "W7-C3 Anchor-and-Deviate", "F1": f1_c3, "Tokens": tok_c3, "Type": "Historical"})

# Add All W7-D models
for _, r in df_dev_all.iterrows():
    is_winner = (r["model"] == winning_model_name)
    mtype = "W7-D Winner" if is_winner else "W7-D Model"
    tok_val = float(r["mean_tokens"]) if "mean_tokens" in r else float(r["tokens"])
    f1_val = float(r["mean_f1"]) if "mean_f1" in r else float(r["f1"])
    frontier_records.append({"Method": f"W7-D ({r['model']})", "F1": f1_val, "Tokens": tok_val, "Type": mtype})

df_frontier = pd.DataFrame(frontier_records)
frontier_path = PROJECT_ROOT / cfg["paths"]["tables_dir"] / "w7d_quality_context_frontier.csv"
df_frontier.to_csv(frontier_path, index=False)
print("Quality-Context Frontier:")
display(df_frontier)

# Plot Quality vs Tokens Pareto curve
plt.figure(figsize=(10, 6))
palette = {"Static Baseline": "gray", "Upper Bound": "gold", "Historical": "steelblue", "W7-D Model": "mediumpurple", "W7-D Winner": "crimson"}
for mtype, group in df_frontier.groupby("Type"):
    marker = "o" if mtype == "Static Baseline" else ("*" if "Winner" in mtype else "^")
    plt.scatter(group["Tokens"], group["F1"], label=mtype, s=140, marker=marker, color=palette.get(mtype, "black"))
    for _, r in group.iterrows():
        plt.annotate(r["Method"], (r["Tokens"] + 15, r["F1"] - 0.002), fontsize=8)

plt.xlabel("Context Tokens")
plt.ylabel("Downstream Answer F1")
plt.title("Quality-Context Pareto Frontier: W7-D vs Historical Allocators")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(PROJECT_ROOT / cfg["paths"]["figures_dir"] / "w7d_quality_vs_tokens.png", dpi=300)
plt.show()"""))

    # Section 25
    cells.append(create_markdown_cell("""## Section 25 — Error Taxonomy & Multi-task Metrics
Evaluate the 6-way error taxonomy (E1–E6) capturing non-monotonicities and anchor retention."""))

    cells.append(create_code_cell("""df_tax_detailed, df_tax_summary = compute_w7d_error_taxonomy(df_targets_dev_plot, budgets=BUDGETS, anchor_k=ANCHOR_K)
tax_table_path = PROJECT_ROOT / cfg["paths"]["tables_dir"] / "w7d_multitask_metrics.csv"
df_tax_summary.to_csv(tax_table_path, index=False)
print("6-Way Error Taxonomy Summary:")
display(df_tax_summary)"""))

    # Section 26
    cells.append(create_markdown_cell("""## Section 26 — Paper-Clustered Bootstrap Significance
Perform 2,000 paper-clustered bootstrap resamples on the DEV set to assess statistical significance against static $k=4$, $k=6$, $k=8$, and W7-C3."""))

    cells.append(create_code_cell("""# Prepare comparison columns
df_boot_eval = df_targets_dev_plot.copy()
df_boot_eval["pred_winner"] = df_boot_eval["pred_k"]
df_boot_eval["pred_k4"] = ANCHOR_K
df_boot_eval["pred_k6"] = 6
df_boot_eval["pred_k8"] = 8

boot_comparisons = []
for base_col, base_name in [("pred_k4", "Static k=4"), ("pred_k6", "Static k=6"), ("pred_k8", "Static k=8")]:
    res = run_paper_clustered_bootstrap(
        df_boot_eval,
        pred_col_policy="pred_winner",
        baseline_pred_col=base_col,
        budgets=BUDGETS,
        n_resamples=cfg["bootstrap"]["n_resamples"],
        random_seed=cfg["bootstrap"]["random_seed"],
    )
    res["Comparison"] = f"W7-D Winner ({winning_model_name}) vs {base_name}"
    boot_comparisons.append(res)

df_boot_results = pd.DataFrame(boot_comparisons)
boot_path = PROJECT_ROOT / cfg["paths"]["bootstrap_dir"] / "w7d_bootstrap_comparisons.csv"
df_boot_results.to_csv(boot_path, index=False)
print("Paper-Clustered Bootstrap Comparisons (2,000 resamples):")
display(df_boot_results)"""))

    # Section 27
    cells.append(create_markdown_cell("""## Section 27 — Final Scientific Conclusion & Stopping Rule
Evaluate results against Section 27 Success Criteria:
* **Minimum useful result**: $F1 \\ge 0.386$ with $\\text{tokens} < 800$.
* **Strong**: $F1 \\ge 0.3909$ with $\\text{tokens} < 900$.
* **Very strong**: $F1 \\ge 0.395$ with $\\text{tokens} < 900$.
* **Exceptional**: $F1 \\approx 0.395 - 0.400$ with $750 - 850$ tokens.

**Stopping Rule (Section 28):**
If neither D0 nor D1 nor D2 improves the static semantic $k=4$ Pareto point $(0.3909, \\approx 921)$, discrete adaptive-$k$ prediction is permanently concluded, and the investigation progresses to W7-E multi-fidelity compression."""))

    cells.append(create_code_cell("""w7d_f1 = float(overall_winner["mean_f1"])
w7d_tokens = float(overall_winner["mean_tokens"])
k4_f1 = float(df_targets_dev["f1_k4"].mean())
k4_tokens = float(df_targets_dev["tokens_k4"].mean())

delta_f1 = w7d_f1 - k4_f1
delta_tokens = w7d_tokens - k4_tokens

print("=== FINAL WEEK 7-D EVALUATION SUMMARY ===")
print(f"Winning Model: {winning_model_name}")
print(f"W7-D Performance: F1 = {w7d_f1:.4f}, Tokens = {w7d_tokens:.1f}")
print(f"Static k=4 Baseline: F1 = {k4_f1:.4f}, Tokens = {k4_tokens:.1f}")
print(f"Difference: Delta F1 = {delta_f1:+.4f}, Delta Tokens = {delta_tokens:+.1f}")

if w7d_f1 >= 0.395 and w7d_tokens < 900:
    outcome = "VERY STRONG / EXCEPTIONAL: Improved quality while reducing tokens."
elif w7d_f1 >= 0.3909 and w7d_tokens < 900:
    outcome = "STRONG: Preserved reference quality with strict token reduction."
elif w7d_f1 >= 0.386 and w7d_tokens < 800:
    outcome = "MINIMUM USEFUL RESULT: Trade-off accepted under eta constraint."
else:
    outcome = "STOPPING RULE TRIGGERED: Discrete adaptive-k allocation failed to move the Pareto frontier. Moving to W7-E multi-fidelity compression."

print("Scientific Conclusion:", outcome)"""))

    # Section 28
    cells.append(create_markdown_cell("""## Section 28 — W7-E Handoff Manifest
Preserve representation caches, utility scores, and retrieval-state vectors for direct input into Week 7-E:
`FULL / MEDIUM / HEAVY` multi-fidelity evidence allocation."""))

    cells.append(create_code_cell("""handoff_manifest = {
    "subtask": "W7-D Handoff to W7-E",
    "saved_artifacts": {
        "cross_encoder_scores_train": str(train_cache_path),
        "cross_encoder_scores_dev": str(dev_cache_path),
        "passage_embeddings_train": str(train_emb_cache),
        "passage_embeddings_dev": str(dev_emb_cache),
        "d1a_model_checkpoint": str(best_model_path_d1a),
        "d1b_model_checkpoint": str(best_model_path_d1b),
        "d2_model_checkpoint": str(best_model_path_d2),
        "optimal_policy": overall_winner,
        "dev_results_all": df_dev_all.to_dict(orient="records"),
    },
    "ready_for_w7e": True,
}

handoff_file = PROJECT_ROOT / cfg["paths"]["output_dir"] / "w7e_handoff_manifest.json"
with open(handoff_file, "w", encoding="utf-8") as f:
    json.dump(handoff_manifest, f, indent=2)

print("W7-E Handoff manifest successfully written to:", handoff_file)
print("WEEK 7-D NOTEBOOK EXECUTION COMPLETE.")"""))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (sara_env)",
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
        "nbformat_minor": 4,
    }
    return notebook


def main():
    nb = build_notebook()
    nb_path = PROJECT_ROOT / "notebooks/week7/week7d_cross_encoder_fidelity_router.ipynb"
    nb_path.parent.mkdir(parents=True, exist_ok=True)
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Successfully generated notebook with {len(nb['cells'])} cells at {nb_path}")


if __name__ == "__main__":
    main()
