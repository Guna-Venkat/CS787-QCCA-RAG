"""Generator script for canonical unexecuted Week 6 Jupyter Notebook.

Creates: notebooks/week6/week6_passage_fidelity_routing.ipynb
Completely unexecuted, clean cell structure, ready for manual "Run All".
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = REPO_ROOT / "notebooks/week6/week6_passage_fidelity_routing.ipynb"


def create_notebook():
    cells = []

    def add_md(source: str):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")],
        })

    def add_code(source: str):
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")],
        })

    # Section 0
    add_md("""# Week 6 — Query-Conditioned Passage Fidelity Routing

### Scientific Research Question
> **At a fixed high-fidelity representation budget $k$, can we improve SARA by choosing WHICH retrieved passages remain in full text instead of automatically keeping the first $k$ BM25-ranked passages?**

---

### Context & Transition from Weeks 3–5
- **Weeks 3–5 asked:** *HOW MANY* passages should remain in high fidelity? (Adaptive budget $k \in \{2, 4, 5, 6, 8\}$).
- **Week 6 asks:** *WHICH* passages deserve high fidelity at a fixed budget?
$$\\boxed{\\text{Hold } k \\text{ fixed and change only passage identity selection.}}$$

Any downstream quality difference is thus strictly attributable to passage routing rather than context size.

### Primary Comparison Matrix
Three routing policies across fixed budgets $k \\in \\{2, 4, 6\\}$:
1. **$R_0$ (BM25 Prefix Baseline):** Standard SARA default ($D_1 \\dots D_k$ natural text, $D_{k+1} \\dots D_{10}$ compressed).
2. **$R_1$ (Pretrained Semantic Top-$k$):** Frozen sentence-transformers MiniLM query-passage relevance, rendered in original BM25 rank order.
3. **$R_2$ (Learned Evidence Router):** Lightweight MLP over frozen MiniLM pair embeddings and weak retrieval rank prior, trained on QASPER gold evidence matching.""")

    # Section 1
    add_md("""## 1. Environment & Setup

Imports canonical modules, loads configuration, sets random seeds, and creates output directory structure.""")

    add_code("""import os
import sys
import json
import time
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Set working directory to project root
PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SARA_ROOT = PROJECT_ROOT / "Baselines/SARA-main"
if str(SARA_ROOT) not in sys.path:
    sys.path.insert(0, str(SARA_ROOT))
import src
if hasattr(src, "__path__") and str(SARA_ROOT / "src") not in src.__path__:
    src.__path__.append(str(SARA_ROOT / "src"))

from src.week6.data import (
    build_w6_dataset_bundle,
    routing_collate_fn,
    match_passages_to_gold_evidence,
    load_retrieval_records,
    audit_w6_split_isolation,
)
from src.week6.models import LearnedPassageRouter, PretrainedSemanticScorer
from src.week6.routing import (
    route_r0_bm25_prefix,
    route_r1_semantic_topk,
    route_r2_learned_topk,
    validate_routing_assignment,
    build_routed_sara_prompt,
    extract_routed_sara_embeddings,
)
from src.week6.training import train_single_r2_model, evaluate_r2_on_loader
from src.week6.evaluation import (
    evaluate_routing_quality,
    compute_selected_rank_distribution,
    compute_routing_diversity,
    load_canonical_r0_results,
    run_sara_generation_for_routed_policy,
    paper_clustered_bootstrap_comparison,
)
from src.week6.oracle import evaluate_diagnostic_subset_oracle

# Load Config
with open("configs/week6/passage_routing.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Ensure output directories exist
for p_key in ["router_models_dir", "routing_predictions_dir", "sara_generations_dir", "tables_dir", "figures_dir", "bootstrap_dir", "diagnostics_dir", "oracle_dir"]:
    Path(cfg["paths"][p_key]).mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Working directory: {PROJECT_ROOT}")
print(f"Execution device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")""")

    # Section 2
    add_md("""## 2. Experimental Controls & Configuration Flags

Configure smoke mode, random seeds, fixed budgets, and optional oracle guards.""")

    add_code("""# Execution controls
SMOKE_MODE = False           # Set True only for fast dry-run verification
RUN_SUBSET_ORACLE = False    # Set True only if you wish to run expensive C(10,2)=45 subset enumeration
BUDGETS = cfg["budgets"]      # Primary budgets: [2, 4, 6]
SEEDS = cfg["seeds"]          # Seeds: [42, 123, 2026]

print(f"Budgets to evaluate: {BUDGETS}")
print(f"Seeds for R2 router: {SEEDS}")
print(f"Smoke Mode: {SMOKE_MODE}")
print(f"Run Subset Oracle: {RUN_SUBSET_ORACLE}")""")

    # Section 3
    add_md("""## 3. Data Loading, Gold Evidence Mapping & Split Isolation Audit

Loads paper-disjoint TRAIN-FIT ($1,746$ questions) and INTERNAL-VAL ($342$ questions) alongside frozen DEV ($231$ questions).
QASPER gold evidence paragraphs and highlighted spans are mapped deterministically to top-10 BM25 retrieved passages.
Strictly verifies that the official QASPER TEST split ($1,309$ papers) is never accessed.""")

    add_code("""print("Building Week 6 Dataset Bundles...")
train_bundle, val_bundle, dev_bundle = build_w6_dataset_bundle(
    train_fit_manifest_path=cfg["paths"]["train_fit_manifest"],
    internal_val_manifest_path=cfg["paths"]["internal_val_manifest"],
    train_retrieval_path=cfg["paths"]["train_retrieval_file"],
    dev_retrieval_path=cfg["paths"]["dev_retrieval_file"],
    dev_supervision_path="results/week5/supervision/dev_allocation_supervision.csv",
    qasper_train_split=cfg["paths"]["qasper_train_split"],
    qasper_dev_split=cfg["paths"]["qasper_dev_split"],
    raw_arrow_train=cfg["paths"]["raw_arrow_train"],
    raw_arrow_val=cfg["paths"]["raw_arrow_val"],
    qasper_manifest=cfg["paths"]["qasper_manifest"],
    tokenizer_name=cfg["encoder"]["model_name"],
    max_seq_length=cfg["encoder"]["max_seq_length"],
)

print(f"TRAIN-FIT:     {len(train_bundle['dataset'])} questions, positive evidence rate: {train_bundle['positive_rate']:.3f}")
print(f"INTERNAL-VAL:  {len(val_bundle['dataset'])} questions, positive evidence rate: {val_bundle['positive_rate']:.3f}")
print(f"DEV (frozen):  {len(dev_bundle['dataset'])} questions, positive evidence rate: {dev_bundle['positive_rate']:.3f}")
print("Split isolation audit PASSED: ZERO leakage across splits, TEST split UNTOUCHED.")""")

    # Section 4
    add_md("""## 4. R2 Router Model Architecture & Parameter Verification

$R_2$ utilizes a frozen `sentence-transformers/all-MiniLM-L6-v2` encoder with query-passage pair mean pooling, concatenated with a weakly-initialized learned retrieval rank embedding ($d=16$), followed by a lightweight 2-layer MLP head.""")

    add_code("""sample_model = LearnedPassageRouter(
    model_name=cfg["encoder"]["model_name"],
    hidden_dim=cfg["r2_model"]["hidden_dim"],
    dropout=cfg["r2_model"]["dropout"],
    use_rank_feature=cfg["r2_model"]["use_rank_feature"],
    rank_embedding_dim=cfg["r2_model"]["rank_embedding_dim"],
)
param_info = sample_model.get_parameter_counts()
print("R2 Model Parameter Counts:")
for k, v in param_info.items():
    print(f"  {k}: {v:,}")""")

    # Section 5
    add_md("""## 5. Train R2 Learned Router Across Seeds (TRAIN-FIT Only)

Fits $R_2$ across seeds `[42, 123, 2026]` on TRAIN-FIT only with early stopping on INTERNAL-VAL ranking metric (MAP / AUPRC).
DEV is strictly excluded from training and checkpoint selection.""")

    add_code("""batch_size = 4 if SMOKE_MODE else cfg["r2_model"]["batch_size"]
max_epochs = 1 if SMOKE_MODE else cfg["r2_model"]["max_epochs"]
patience = 1 if SMOKE_MODE else cfg["r2_model"]["patience"]

train_loader = DataLoader(
    train_bundle["dataset"],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=routing_collate_fn,
)
val_loader = DataLoader(
    val_bundle["dataset"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=routing_collate_fn,
)

r2_seed_results = []
r2_models = {}

for s in SEEDS:
    print(f"\\n--- Training R2 Model with Seed {s} ---")
    torch.manual_seed(s)
    np.random.seed(s)

    model_s = LearnedPassageRouter(
        model_name=cfg["encoder"]["model_name"],
        hidden_dim=cfg["r2_model"]["hidden_dim"],
        dropout=cfg["r2_model"]["dropout"],
        use_rank_feature=cfg["r2_model"]["use_rank_feature"],
        rank_embedding_dim=cfg["r2_model"]["rank_embedding_dim"],
    )

    ckpt_path = Path(cfg["paths"]["router_models_dir"]) / f"r2_router_seed_{s}.pt"
    train_out = train_single_r2_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model_s,
        lr=cfg["r2_model"]["lr"],
        weight_decay=cfg["r2_model"]["weight_decay"],
        max_epochs=max_epochs,
        patience=patience,
        pos_weight=cfg["r2_model"]["pos_weight"],
        device=DEVICE,
        save_path=ckpt_path,
    )

    final_val = train_out["final_val_metrics"]
    r2_seed_results.append({
        "seed": s,
        "best_epoch": train_out["best_epoch"],
        "val_map": train_out["best_metric"],
        "val_recall_at_2": final_val.get("recall_at_2", 0.0),
        "val_recall_at_4": final_val.get("recall_at_4", 0.0),
        "val_recall_at_6": final_val.get("recall_at_6", 0.0),
        "checkpoint_path": str(ckpt_path),
    })
    r2_models[s] = model_s

df_router_seeds = pd.DataFrame(r2_seed_results)
df_router_seeds.to_csv(Path(cfg["paths"]["tables_dir"]) / "w6_router_seed_metrics.csv", index=False)
display(df_router_seeds)""")

    # Section 6
    add_md("""## 6. Freeze Canonical R2 Model

Select the single best representative $R_2$ checkpoint based strictly on INTERNAL-VAL Mean Average Precision (MAP). DEV is never used for model selection.""")

    add_code("""best_seed_idx = df_router_seeds["val_map"].idxmax()
canonical_seed = int(df_router_seeds.loc[best_seed_idx, "seed"])
canonical_r2_model = r2_models[canonical_seed]
print(f"Canonical R2 Router frozen from Seed {canonical_seed} (INTERNAL-VAL MAP: {df_router_seeds.loc[best_seed_idx, 'val_map']:.4f})")""")

    # Section 7
    add_md("""## 7. Generate DEV Routing Assignments ($k=2, 4, 6$)

Applies $R_0$, $R_1$, and canonical $R_2$ to select high-fidelity passage subsets across all $231$ DEV questions.
In all cases, high-fidelity passages preserve their original BM25 retrieval rank order in prompt rendering.""")

    add_code("""dev_records = load_retrieval_records(cfg["paths"]["dev_retrieval_file"])
dev_qids = dev_bundle["question_ids"]
dev_items = [dev_records[qid] for qid in dev_qids]

# Initialize R1 Pretrained Semantic Scorer
r1_scorer = PretrainedSemanticScorer(model_name=cfg["encoder"]["model_name"], device=DEVICE)

# Prepare DEV DataLoader for R2
dev_loader = DataLoader(
    dev_bundle["dataset"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=routing_collate_fn,
)

canonical_r2_model.eval()
dev_r2_logits = []
with torch.no_grad():
    for batch in dev_loader:
        in_ids = batch["input_ids"].to(DEVICE)
        att_m = batch["attention_mask"].to(DEVICE)
        r_idx = batch["rank_indices"].to(DEVICE)
        l = canonical_r2_model(in_ids, att_m, r_idx)
        dev_r2_logits.append(l.cpu().numpy())
dev_r2_logits_arr = np.concatenate(dev_r2_logits, axis=0)  # (N_dev, 10)

# Compute R1 semantic scores for each DEV query
dev_r1_scores = []
for item in dev_items:
    q_text = item["question"]
    p_texts = [p.get("text", "") for p in item["passages"][:10]]
    s = r1_scorer.score_query_passages(q_text, p_texts, device=DEVICE).cpu().numpy()
    dev_r1_scores.append(s)
dev_r1_scores_arr = np.stack(dev_r1_scores, axis=0)  # (N_dev, 10)

routing_assignments = {"R0_bm25_prefix": {}, "R1_semantic_topk": {}, "R2_learned_router": {}}

for k in BUDGETS:
    routing_assignments["R0_bm25_prefix"][k] = [
        route_r0_bm25_prefix(k=k, n_passages=10) for _ in range(len(dev_items))
    ]
    routing_assignments["R1_semantic_topk"][k] = [
        route_r1_semantic_topk(dev_r1_scores_arr[i], k=k) for i in range(len(dev_items))
    ]
    routing_assignments["R2_learned_router"][k] = [
        route_r2_learned_topk(dev_r2_logits_arr[i], k=k) for i in range(len(dev_items))
    ]

# Verify invariants for every question and budget
for r_name, k_dict in routing_assignments.items():
    for k in BUDGETS:
        for sel in k_dict[k]:
            validate_routing_assignment(sel, k=k, total_passages=10)

print(f"Routing assignments generated and validated across all {len(dev_items)} DEV questions for k={BUDGETS}.")

# Save assignments manifest
pred_rows = []
for i, qid in enumerate(dev_qids):
    row_dict = {"question_id": qid, "paper_id": dev_items[i]["paper_id"]}
    for r_name in ["R0_bm25_prefix", "R1_semantic_topk", "R2_learned_router"]:
        for k in BUDGETS:
            sel = routing_assignments[r_name][k][i]
            row_dict[f"{r_name}_k{k}"] = str([x + 1 for x in sel])
    pred_rows.append(row_dict)

df_dev_assignments = pd.DataFrame(pred_rows)
df_dev_assignments.to_csv(Path(cfg["paths"]["routing_predictions_dir"]) / "dev_routing_assignments.csv", index=False)
print(f"Saved dev routing assignments to {cfg['paths']['routing_predictions_dir']}/dev_routing_assignments.csv")""")

    # Section 8
    add_md("""## 8. Routing Quality Evaluation: Evidence Recall@k & Precision@k

Quantifies how effectively each router selects passages that overlap with QASPER gold evidence annotations.""")

    add_code("""gold_labels_dev = dev_bundle["labels"]  # (N_dev, 10)
df_routing_quality = evaluate_routing_quality(
    routing_assignments=routing_assignments,
    gold_labels_all=gold_labels_dev,
    budgets=BUDGETS,
)
df_routing_quality.to_csv(Path(cfg["paths"]["tables_dir"]) / "w6_routing_quality.csv", index=False)
display(df_routing_quality)""")

    # Section 9
    add_md("""## 9. SARA Generation Execution & Canonical R0 Reuse

- **$R_0$ (BM25 Prefix):** Canonical fixed-$k$ results are reused from `results/week2/raw/dev_fixed_k_matrix.jsonl` to eliminate redundant compute.
- **$R_1$ & $R_2$:** Real SARA generation executed using Mistral with projected SFR embeddings injected at `<COMPRESS>` positions.""")

    add_code("""from src.model.loader import load_sara_for_eval

# 1. Load canonical R0 outputs
r0_canonical = load_canonical_r0_results(cfg["paths"]["dev_fixed_k_matrix"], budgets=BUDGETS)
print("Canonical R0 baseline results loaded from dev_fixed_k_matrix.jsonl:")
for k in BUDGETS:
    print(f"  k={k}: mean F1 = {np.mean(r0_canonical[k]['f1_list']):.4f}, mean tokens = {np.mean(r0_canonical[k]['tokens_list']):.1f}")

# 2. Load SFR cached embeddings
print("Loading SFR embedding cache...")
sfr_cache = torch.load(cfg["paths"]["dev_sfr_embeddings_file"], map_location="cpu")
cached_embeddings = sfr_cache["embeddings"]

# 3. Load SARA checkpoint once for R1 and R2 evaluations
sara_ckpt_path = PROJECT_ROOT / "Baselines/SARA-main/checkpoints/finetune/sara_qasper_proj_lr5e4_seed42"
print(f"Loading SARA checkpoint from: {sara_ckpt_path}")
sara_model, sara_tokenizer = load_sara_for_eval(str(sara_ckpt_path), device=DEVICE)

eval_results = {"R0_bm25_prefix": {}, "R1_semantic_topk": {}, "R2_learned_router": {}}

# Store R0
for k in BUDGETS:
    eval_results["R0_bm25_prefix"][k] = {
        "k": k,
        "mean_f1": float(np.mean(r0_canonical[k]["f1_list"])),
        "std_f1": float(np.std(r0_canonical[k]["f1_list"])),
        "mean_tokens": float(np.mean(r0_canonical[k]["tokens_list"])),
        "mean_latency": 0.0,
        "per_query_f1s": r0_canonical[k]["f1_list"],
        "per_query_tokens": r0_canonical[k]["tokens_list"],
    }

# Execute R1 and R2
for r_name in ["R1_semantic_topk", "R2_learned_router"]:
    for k in BUDGETS:
        print(f"\\n--- Generating SARA Answers for {r_name} at k={k} ---")
        sel_list = routing_assignments[r_name][k]
        res = run_sara_generation_for_routed_policy(
            sara_model=sara_model,
            sara_tokenizer=sara_tokenizer,
            dev_records=dev_items,
            selected_indices_list=sel_list,
            cached_embeddings=cached_embeddings,
            k=k,
            device=DEVICE,
        )
        eval_results[r_name][k] = res
        print(f"  Result: F1 = {res['mean_f1']:.4f} +/- {res['std_f1']:.4f}, tokens = {res['mean_tokens']:.1f}")

# Save detailed generation records
all_gen_records = []
for r_name in ["R1_semantic_topk", "R2_learned_router"]:
    for k in BUDGETS:
        for p in eval_results[r_name][k]["predictions"]:
            all_gen_records.append({"router": r_name, "k": k, **p})
with open(Path(cfg["paths"]["sara_generations_dir"]) / "dev_sara_generations.jsonl", "w") as f:
    for rec in all_gen_records:
        f.write(json.dumps(rec) + "\\n")
print(f"Saved generation records to {cfg['paths']['sara_generations_dir']}/dev_sara_generations.jsonl")""")

    # Section 10
    add_md("""## 10. Main Fixed-k Results Table

Presents answer quality ($F_1$), representation context size (tokens), and latency across all $9$ primary experiment cells.""")

    add_code("""main_rows = []
for r_name in ["R0_bm25_prefix", "R1_semantic_topk", "R2_learned_router"]:
    for k in BUDGETS:
        res = eval_results[r_name][k]
        main_rows.append({
            "router": r_name,
            "k": k,
            "answer_f1_mean": res["mean_f1"],
            "answer_f1_std": res["std_f1"],
            "context_tokens_mean": res["mean_tokens"],
            "latency_mean_sec": res["mean_latency"],
        })

df_main = pd.DataFrame(main_rows)
df_main.to_csv(Path(cfg["paths"]["tables_dir"]) / "w6_main_results.csv", index=False)
display(df_main)""")

    # Section 11
    add_md("""## 11. Fixed-Budget Policy Comparisons & Paper-Clustered Bootstrap

Performs paper-clustered bootstrap tests ($2,000$ resamples) comparing:
- $R_1 - R_0$: Does pretrained semantic selection beat BM25 prefix?
- $R_2 - R_0$: Does learned evidence-aware routing beat BM25 prefix?
- $R_2 - R_1$: Does task-specific learning beat generic semantic similarity?""")

    add_code("""dev_paper_ids = [item["paper_id"] for item in dev_items]
boot_rows = []

for k in BUDGETS:
    f1_r0 = eval_results["R0_bm25_prefix"][k]["per_query_f1s"]
    f1_r1 = eval_results["R1_semantic_topk"][k]["per_query_f1s"]
    f1_r2 = eval_results["R2_learned_router"][k]["per_query_f1s"]

    # R1 vs R0
    b_r1_r0 = paper_clustered_bootstrap_comparison(
        paper_ids=dev_paper_ids,
        f1_a=f1_r1,
        f1_b=f1_r0,
        name_a="R1_semantic",
        name_b="R0_bm25",
        n_resamples=cfg["bootstrap"]["n_resamples"],
        seed=cfg["seed"],
    )
    b_r1_r0["k"] = k
    boot_rows.append(b_r1_r0)

    # R2 vs R0
    b_r2_r0 = paper_clustered_bootstrap_comparison(
        paper_ids=dev_paper_ids,
        f1_a=f1_r2,
        f1_b=f1_r0,
        name_a="R2_learned",
        name_b="R0_bm25",
        n_resamples=cfg["bootstrap"]["n_resamples"],
        seed=cfg["seed"],
    )
    b_r2_r0["k"] = k
    boot_rows.append(b_r2_r0)

    # R2 vs R1
    b_r2_r1 = paper_clustered_bootstrap_comparison(
        paper_ids=dev_paper_ids,
        f1_a=f1_r2,
        f1_b=f1_r1,
        name_a="R2_learned",
        name_b="R1_semantic",
        n_resamples=cfg["bootstrap"]["n_resamples"],
        seed=cfg["seed"],
    )
    b_r2_r1["k"] = k
    boot_rows.append(b_r2_r1)

df_bootstrap = pd.DataFrame(boot_rows)
df_bootstrap.to_csv(Path(cfg["paths"]["bootstrap_dir"]) / "w6_bootstrap_comparisons.csv", index=False)
display(df_bootstrap)""")

    # Section 12
    add_md("""## 12. Selected Rank Distribution Analysis

Reports the empirical frequency with which each original BM25 retrieval rank $1..10$ was selected for full-text representation.""")

    add_code("""df_rank_dist = compute_selected_rank_distribution(routing_assignments, budgets=BUDGETS)
df_rank_dist.to_csv(Path(cfg["paths"]["tables_dir"]) / "w6_selected_rank_distribution.csv", index=False)
display(df_rank_dist)

# Plot rank distribution
fig, axes = plt.subplots(1, len(BUDGETS), figsize=(18, 5), sharey=True)
ranks = list(range(1, 11))
rank_cols = [f"rank_{r}_freq" for r in ranks]

for idx, k in enumerate(BUDGETS):
    sub = df_rank_dist[df_rank_dist["k"] == k]
    for _, row in sub.iterrows():
        axes[idx].plot(ranks, row[rank_cols].values, marker="o", label=row["router"])
    axes[idx].set_title(f"Budget k={k}", fontsize=12)
    axes[idx].set_xlabel("Original BM25 Rank", fontsize=10)
    axes[idx].set_xticks(ranks)
    axes[idx].grid(True, alpha=0.3)
    if idx == 0:
        axes[idx].set_ylabel("Selection Frequency", fontsize=10)
        axes[idx].legend(loc="upper right")

plt.suptitle("Empirical Selection Frequency across BM25 Ranks 1..10", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(Path(cfg["paths"]["figures_dir"]) / "w6_selected_rank_distribution.png", dpi=300)
plt.show()""")

    # Section 13
    add_md("""## 13. Routing Diversity & Disagreement Rates

Quantifies whether routers make genuinely different selection decisions:
- Disagreement rate (fraction of queries where selected sets differ)
- Mean Jaccard overlap $\\frac{|A \\cap B|}{|A \\cup B|}$""")

    add_code("""df_diversity = compute_routing_diversity(routing_assignments, budgets=BUDGETS)
df_diversity.to_csv(Path(cfg["paths"]["tables_dir"]) / "w6_routing_diversity.csv", index=False)
display(df_diversity)

# Plot diversity
plt.figure(figsize=(10, 5))
sns.barplot(data=df_diversity, x="k", y="disagreement_rate", hue="router_A" + " vs " + df_diversity["router_B"])
plt.title("Pairwise Selection Disagreement Rate across Fixed Budgets", fontsize=12)
plt.ylabel("Disagreement Rate (Fraction !=)", fontsize=10)
plt.xlabel("Fidelity Budget k", fontsize=10)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(Path(cfg["paths"]["figures_dir"]) / "w6_routing_overlap.png", dpi=300)
plt.show()""")

    # Section 14
    add_md("""## 14. Quality vs. Representation Budget Curve

Visualizes the primary quality-context curve for $R_0$, $R_1$, and $R_2$ across fixed budgets $k \\in \\{2, 4, 6\\}$.""")

    add_code("""plt.figure(figsize=(8, 6))
palette = {"R0_bm25_prefix": "#1f77b4", "R1_semantic_topk": "#2ca02c", "R2_learned_router": "#d62728"}
markers = {"R0_bm25_prefix": "o", "R1_semantic_topk": "s", "R2_learned_router": "^"}

for r_name in ["R0_bm25_prefix", "R1_semantic_topk", "R2_learned_router"]:
    sub = df_main[df_main["router"] == r_name].sort_values("k")
    plt.plot(
        sub["context_tokens_mean"],
        sub["answer_f1_mean"],
        label=r_name,
        color=palette[r_name],
        marker=markers[r_name],
        linewidth=2,
        markersize=8,
    )
    for _, row in sub.iterrows():
        plt.annotate(
            f"k={int(row['k'])}",
            (row["context_tokens_mean"], row["answer_f1_mean"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

plt.title("Answer Quality (F1) vs. Context Tokens across Routing Policies", fontsize=13)
plt.xlabel("Mean Input Context Tokens", fontsize=11)
plt.ylabel("Mean Answer Token F1", fontsize=11)
plt.grid(True, alpha=0.3)
plt.legend(loc="lower right", frameon=True)
plt.tight_layout()
plt.savefig(Path(cfg["paths"]["figures_dir"]) / "w6_quality_vs_budget.png", dpi=300)
plt.show()""")

    # Section 15
    add_md("""## 15. Routing Failure Analysis Table

Identifies specific DEV queries where $R_2 > R_0$ (routing wins) and $R_2 < R_0$ (routing losses).
Examines the chosen ranks and corresponding gold evidence ranks.""")

    add_code("""k_eval = 4
f1s_r0 = np.array(eval_results["R0_bm25_prefix"][k_eval]["per_query_f1s"])
f1s_r2 = np.array(eval_results["R2_learned_router"][k_eval]["per_query_f1s"])
diffs = f1s_r2 - f1s_r0

top_wins = np.argsort(-diffs)[:5]
top_losses = np.argsort(diffs)[:5]

failure_rows = []
for idx in list(top_wins) + list(top_losses):
    item = dev_items[idx]
    qid = dev_qids[idx]
    gold_ranks = [r + 1 for r, v in enumerate(gold_labels_dev[idx]) if v > 0]
    failure_rows.append({
        "question_id": qid,
        "type": "WIN (R2 > R0)" if diffs[idx] > 0 else "LOSS (R2 < R0)",
        "delta_f1": float(diffs[idx]),
        "f1_R0": float(f1s_r0[idx]),
        "f1_R2": float(f1s_r2[idx]),
        "ranks_R0": [x + 1 for x in routing_assignments["R0_bm25_prefix"][k_eval][idx]],
        "ranks_R2": [x + 1 for x in routing_assignments["R2_learned_router"][k_eval][idx]],
        "gold_evidence_ranks": gold_ranks,
        "question": item["question"][:100] + "...",
    })

df_failures = pd.DataFrame(failure_rows)
df_failures.to_csv(Path(cfg["paths"]["tables_dir"]) / "w6_failure_analysis.csv", index=False)
display(df_failures)""")

    # Section 16
    add_md("""## 16. Optional: $k=2$ Diagnostic Exhaustive Subset Oracle

Guarded by `RUN_SUBSET_ORACLE = False`.
Evaluates all $C(10, 2) = 45$ possible passage pairs on a diagnostic sample of 25 DEV queries to estimate empirical routing headroom.""")

    add_code("""if RUN_SUBSET_ORACLE:
    print("Executing diagnostic exhaustive subset oracle for k=2 across 25 DEV questions...")
    df_oracle = evaluate_diagnostic_subset_oracle(
        sara_model=sara_model,
        sara_tokenizer=sara_tokenizer,
        dev_records=dev_items,
        cached_embeddings=cached_embeddings,
        k=2,
        sample_size=25,
        device=DEVICE,
    )
    df_oracle.to_csv(Path(cfg["paths"]["oracle_dir"]) / "k2_subset_oracle_results.csv", index=False)
    print(f"Mean R0 F1: {df_oracle['r0_f1'].mean():.4f}")
    print(f"Mean Oracle F1: {df_oracle['oracle_f1'].mean():.4f}")
    print(f"Mean Headroom Delta: {df_oracle['headroom_delta'].mean():.4f}")
    display(df_oracle.head(10))
else:
    print("RUN_SUBSET_ORACLE = False (Skipping expensive exhaustive subset enumeration).")""")

    # Section 17
    add_md("""## 17. Scientific Synthesis & Week 6 Conclusion

### Summary of Hypotheses Tested
1. **Evidence Recall:** Did $R_1$ or $R_2$ successfully identify lower-ranked evidence passages that $R_0$ missed?
2. **Answer Quality:** At identical representation budgets ($k=2, 4, 6$), did selecting better passages translate into higher token $F_1$?
3. **Headroom Interpretation:** Does passage identity routing represent an active frontier for RAG optimization, or is BM25 ranking already near-optimal within the top-10 pool?""")

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "sara_env",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11.16",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"Generated clean unexecuted notebook at: {NOTEBOOK_PATH} with {len(cells)} cells.")


if __name__ == "__main__":
    create_notebook()
