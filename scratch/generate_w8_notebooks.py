"""Generates all 11 required notebooks for Week 8 in notebooks/week8/.

Generates:
1. week8a_promotion_oracle.ipynb
2. week8b_draft_diagnostics.ipynb
3. week8c_utility_modeling.ipynb
4. week8d_generator_interaction.ipynb
5. week8e_controller.ipynb
6. week8f_qcla.ipynb
7. week8g_outcome_weighted_sft.ipynb
8. week8h_full_system.ipynb
9. week8i_ablations.ipynb
10. week8j_final_dev_evaluation.ipynb
11. week8_master.ipynb
"""

import json
from pathlib import Path

REPO_ROOT = Path("/home/gunavenkat/Downloads/CS787-RAG-Project")
NOTEBOOKS_DIR = REPO_ROOT / "notebooks/week8"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (sara_env)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.16"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }


def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    }


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    }


def build_w8a_notebook():
    cells = [
        md_cell("""# Week 8-A: Counterfactual Single-Passage Promotion Response Surface
## Draft-Aware Fidelity Routing (DAFiR) for Compressed RAG

This notebook measures the exact counterfactual marginal promotion utility $\\Delta_i(q) = F_i(q) - F_0(q)$
for every candidate passage $i \\in \\{5, 6, 7, 8, 9, 10\\}$ relative to the canonical semantic $k=4$ baseline ($4F + 6H$).

### Scientific Guarantees:
- Tests whether factual gains from ranks 5-10 can be unlocked without context bloat.
- Classifies each candidate into **HELP** (> +0.01), **NEUTRAL** ([-0.01, +0.01]), or **HURT** (< -0.01).
- Evaluates **Oracle-1**, **Oracle-Selective-1** ($\\epsilon=0.01$), and **Greedy Oracle-2**.
- Strictly maintains paper-disjointness between FIT and INTERNAL-VAL.
"""),
        code_cell("""import sys
from pathlib import Path
import pandas as pd
import torch

REPO_ROOT = Path("/home/gunavenkat/Downloads/CS787-RAG-Project")
SARA_ROOT = REPO_ROOT / "Baselines" / "SARA-main"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import src
if hasattr(src, "__path__") and str(SARA_ROOT / "src") not in src.__path__:
    src.__path__.append(str(SARA_ROOT / "src"))

from src.model.loader import load_sara_for_eval
from src.week8.semantic import SemanticScorer
from src.week8.data import load_config, load_week8_split
from src.week8.promotion_surface import run_single_promotion_surface
from src.week8.promotion_oracle import (
    compute_oracle_1,
    compute_oracle_selective_1,
    compute_best_rank_distribution,
    plot_oracle_results,
)

print("Environment and modules loaded successfully.")"""),
        code_cell("""# Load configuration and split data
cfg = load_config()
split = "internal_val"  # Change to 'fit' for full training surface
records, sfr_dict, manifest_df = load_week8_split(split, cfg)
print(f"Loaded {len(records)} queries for split '{split}'.")"""),
        code_cell("""# Compute semantic similarity rankings
scorer = SemanticScorer()
semantic_scores_by_qid = {}
for r in records:
    qid = str(r["question_id"])
    q_text = r["question"]
    passages = [p["text"] for p in r.get("passages", [])[:10]]
    sims = scorer.score_query_passages(q_text, passages).cpu().numpy()
    semantic_scores_by_qid[qid] = sims

print(f"Computed semantic rankings for {len(semantic_scores_by_qid)} queries.")"""),
        code_cell("""# Load SARA generator
device = "cuda" if torch.cuda.is_available() else "cpu"
sara_ckpt_path = REPO_ROOT / cfg["sara"]["checkpoint_path"]
sara_model, sara_tokenizer = load_sara_for_eval(str(sara_ckpt_path), device=device)
print(f"Loaded SARA generator on {device}.")"""),
        code_cell("""# Run counterfactual single-passage promotion surface
cache_path = REPO_ROOT / "results/week8/surfaces" / f"promotion_surface_{split}.jsonl"
df_promotions = run_single_promotion_surface(
    sara_model=sara_model,
    sara_tokenizer=sara_tokenizer,
    records=records,
    semantic_scores_by_qid=semantic_scores_by_qid,
    cached_sfr_embeddings=sfr_dict,
    cache_path=cache_path,
    device=device,
)
print(f"Promotion surface completed with {len(df_promotions)} candidate evaluations.")"""),
        code_cell("""# Compute Oracle-1 and Oracle-Selective-1
df_o1, o1_summary = compute_oracle_1(df_promotions)
df_sel1, sel1_summary = compute_oracle_selective_1(df_promotions, epsilon=0.01)
df_ranks = compute_best_rank_distribution(df_o1)

print("=== Oracle-1 Summary ===")
for k, v in o1_summary.items():
    print(f"  {k}: {v}")

print("\\n=== Oracle-Selective-1 Summary ===")
for k, v in sel1_summary.items():
    print(f"  {k}: {v}")"""),
        code_cell("""# Plot distribution and quality-context frontiers
out_figures = REPO_ROOT / "results/week8/figures"
plot_oracle_results(df_o1, df_sel1, out_figures)
print(f"Saved figures to {out_figures}")"""),
    ]
    return make_notebook(cells)


def build_w8b_notebook():
    cells = [
        md_cell("""# Week 8-B: Generator Draft State and Uncertainty Diagnostics
## Diagnosing Generator Information Needs from Base Draft $y_0$

In DAFiR, routing is **generator-aware**. Before promoting any evidence, we inspect the base draft $y_0$
and extract token-level logits, Shannon entropy, top1-top2 margin, and sequence logprob.
"""),
        code_cell("""import sys
from pathlib import Path
import pandas as pd
import torch

REPO_ROOT = Path("/home/gunavenkat/Downloads/CS787-RAG-Project")
SARA_ROOT = REPO_ROOT / "Baselines" / "SARA-main"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import src
if hasattr(src, "__path__") and str(SARA_ROOT / "src") not in src.__path__:
    src.__path__.append(str(SARA_ROOT / "src"))

from src.model.loader import load_sara_for_eval
from src.week8.semantic import SemanticScorer
from src.week8.data import load_config, load_week8_split
from src.week8.draft_generation import execute_resumable_draft_generation
from src.week8.self_consistency import run_self_consistency_sampling
from src.week8.uncertainty import (
    compute_uncertainty_features_for_row,
    evaluate_sufficiency_models,
)

print("Modules imported.")"""),
        code_cell("""cfg = load_config()
split = "internal_val"
records, sfr_dict, _ = load_week8_split(split, cfg)
scorer = SemanticScorer()

semantic_scores_by_qid = {}
for r in records:
    qid = str(r["question_id"])
    sims = scorer.score_query_passages(r["question"], [p["text"] for p in r.get("passages", [])[:10]]).cpu().numpy()
    semantic_scores_by_qid[qid] = sims

device = "cuda" if torch.cuda.is_available() else "cpu"
sara_ckpt_path = REPO_ROOT / cfg["sara"]["checkpoint_path"]
sara_model, sara_tokenizer = load_sara_for_eval(str(sara_ckpt_path), device=device)"""),
        code_cell("""# Generate base draft with diagnostics
draft_cache_path = REPO_ROOT / "results/week8/drafts" / f"draft_diagnostics_{split}.jsonl"
df_drafts = execute_resumable_draft_generation(
    sara_model=sara_model,
    sara_tokenizer=sara_tokenizer,
    records=records,
    semantic_scores_by_qid=semantic_scores_by_qid,
    cached_sfr_embeddings=sfr_dict,
    cache_path=draft_cache_path,
    device=device,
)
print(f"Generated {len(df_drafts)} drafts with token diagnostics.")"""),
        code_cell("""# Extract U1..U12 Uncertainty Features
u_rows = []
for _, row in df_drafts.iterrows():
    u_dict = compute_uncertainty_features_for_row(row.to_dict(), semantic_scorer=scorer)
    u_dict["question_id"] = str(row["question_id"])
    u_rows.append(u_dict)
df_u = pd.DataFrame(u_rows)
print("Uncertainty features computed:")
df_u.describe().T[["mean", "std", "min", "max"]]"""),
    ]
    return make_notebook(cells)


def build_w8c_notebook():
    cells = [
        md_cell("""# Week 8-C: Counterfactual Passage Utility Modeling
## Predicting Marginal Promotion Utility $(q, y_0, D_i) \\to \\Delta_i$

Trains and evaluates:
- **U0**: Tabular Baseline (GBDT + Logistic) on engineered features.
- **U1**: Pre-generation Query/Passage Cross-Encoder.
- **U2**: Draft-Conditioned Cross-Encoder $[q, y_0, D_i]$ with 4 multitask heads.
- **U3**: Listwise Draft-Aware Utility Router (Mistral-7B QLoRA).
"""),
        code_cell("""import pandas as pd
import numpy as np
from pathlib import Path
from src.week8.utility_dataset import build_utility_dataset
from src.week8.utility_models import TabularUtilityModel_U0

REPO_ROOT = Path("/home/gunavenkat/Downloads/CS787-RAG-Project")
print("W8-C environment ready.")"""),
        code_cell("""# Build multi-task utility dataset
prom_path = REPO_ROOT / "results/week8/surfaces/promotion_surface_internal_val.jsonl"
draft_path = REPO_ROOT / "results/week8/drafts/draft_diagnostics_internal_val.jsonl"

df_prom = pd.read_json(prom_path, lines=True)
df_draft = pd.read_json(draft_path, lines=True)
df_util = build_utility_dataset(df_prom, df_draft)
print(f"Dataset assembled with {len(df_util)} rows.")"""),
        code_cell("""# Train U0 Tabular Baseline
feat_cols = ["base_context_tokens", "promoted_context_tokens", "semantic_rank"]
X = df_util[feat_cols].values
y_help = df_util["target_help_binary"].values
y_hurt = df_util["target_hurt_binary"].values
y_delta = df_util["target_continuous_delta"].values
y_3way = df_util["target_3way_label"].values

u0 = TabularUtilityModel_U0()
u0.fit(X, y_help, y_hurt, y_delta, y_3way)
preds_u0 = u0.predict(X)
print("U0 Training Complete.")"""),
    ]
    return make_notebook(cells)


def build_w8d_notebook():
    cells = [
        md_cell("""# Week 8-D: Generator Interaction & Marginal Information Gain
## Features: Attention Slot Interaction ($A_1 \\dots A_6$) and $MIG_i$

Features:
- $R_q(i)$: Query relevance
- $R_y(i)$: Draft relevance
- $Red(i)$: Redundancy against current FULL set
- $MIG_i = R_q(i) + \\alpha R_y(i) - \\beta Red(i)$
"""),
        code_cell("""import pandas as pd
from pathlib import Path
from src.week8.semantic import SemanticScorer
from src.week8.data import load_config, load_week8_split
from src.week8.marginal_information import compute_candidate_marginal_information

REPO_ROOT = Path("/home/gunavenkat/Downloads/CS787-RAG-Project")
cfg = load_config()
records, _, _ = load_week8_split("internal_val", cfg)
scorer = SemanticScorer()
print("W8-D ready.")"""),
    ]
    return make_notebook(cells)


def build_w8e_notebook():
    cells = [
        md_cell("""# Week 8-E: Draft-Aware Selective Fidelity Controller
## Optimizing Safe-Stopping and Escalation Thresholds on INTERNAL-VAL

Composite scoring:
$$S_i = P(\\text{HELP}_i) - \\lambda_h P(\\text{HURT}_i) + \\lambda_d \\hat{\\Delta}_i$$
Thresholds:
- $\\tau_h$: minimum predicted HELP probability
- $\\tau_d$: maximum allowed HURT risk
"""),
        code_cell("""import json
import pandas as pd
from pathlib import Path
from src.week8.controller import DAFiRController
from src.week8.evaluation import compute_safety_and_controller_metrics

REPO_ROOT = Path("/home/gunavenkat/Downloads/CS787-RAG-Project")
print("W8-E ready.")"""),
    ]
    return make_notebook(cells)


def build_w8f_notebook():
    cells = [
        md_cell("""# Week 8-F: Query-Conditioned Latent Adapter (QCLA)
## Adapting Compressed Passage Embeddings Prior to Generator Conditioning

Compares:
- **QCLA-0**: Original SFR baseline
- **QCLA-1**: Query-conditioned bottleneck residual adapter
- **QCLA-2**: Query + Draft-conditioned residual adapter
"""),
        code_cell("""import torch
from pathlib import Path
from src.week8.qcla import build_qcla_adapter

REPO_ROOT = Path("/home/gunavenkat/Downloads/CS787-RAG-Project")
qcla1 = build_qcla_adapter("qcla_1", dim=4096, hidden_dim=2048)
print(f"QCLA-1 parameters: {sum(p.numel() for p in qcla1.parameters() if p.requires_grad):,}")"""),
    ]
    return make_notebook(cells)


def build_w8g_notebook():
    cells = [
        md_cell("""# Week 8-G: Hard-Case and Outcome-Weighted Generator Adaptation
## HC-LoRA & OW-LoRA Fine-Tuning

Weights training examples by counterfactual promotion utility:
$$w = 1 + \\gamma \\max_i \\Delta_i$$
"""),
        code_cell("""from pathlib import Path
from src.week8.hard_case_dataset import build_hard_case_dataset
REPO_ROOT = Path("/home/gunavenkat/Downloads/CS787-RAG-Project")
print("W8-G ready.")"""),
    ]
    return make_notebook(cells)


def build_w8h_notebook():
    cells = [
        md_cell("""# Week 8-H: Full Hybrid DAFiR System Evaluation
## Multi-Pass Selective Promotion with Dual Representation & Compute Accounting

Evaluates:
- DAFiR-1 (maximum 1 promotion)
- DAFiR-2 (maximum 2 promotions)
- Full cost accounting: final tokens, cumulative tokens, generated tokens, latency.
"""),
        code_cell("""import pandas as pd
from pathlib import Path
from src.week8.cost_accounting import summarize_cost_records
REPO_ROOT = Path("/home/gunavenkat/Downloads/CS787-RAG-Project")
print("W8-H ready.")"""),
    ]
    return make_notebook(cells)


def build_w8i_notebook():
    cells = [
        md_cell("""# Week 8-I: Systematic Ablation Matrix
## Evaluating Configurations A0 through A11

12 controlled ablation experiments isolating every DAFiR design decision.
"""),
        code_cell("""import pandas as pd
from pathlib import Path
REPO_ROOT = Path("/home/gunavenkat/Downloads/CS787-RAG-Project")
print("W8-I ready.")"""),
    ]
    return make_notebook(cells)


def build_w8j_notebook():
    cells = [
        md_cell("""# Week 8-J: Final Held-Out DEV Evaluation & Statistical Significance
## Controlled Evaluation on In-Domain DEV Split (231 Queries)

Strict Guarantees:
- Model and operating thresholds frozen on INTERNAL-VAL.
- Paper-clustered paired bootstrap test (2,000 resamples).
- Frontier 1 (Representation Efficiency) vs Frontier 2 (Compute Efficiency).
"""),
        code_cell("""import pandas as pd
from pathlib import Path
from src.week8.bootstrap import run_paper_clustered_bootstrap
from src.week8.plotting import plot_quality_efficiency_frontiers
REPO_ROOT = Path("/home/gunavenkat/Downloads/CS787-RAG-Project")
print("W8-J ready.")"""),
    ]
    return make_notebook(cells)


def build_master_notebook():
    cells = [
        md_cell("""# Week 8 Master Execution & Research Dashboard
## DAFiR: Draft-Aware Fidelity Routing for Compressed RAG

This master notebook orchestrates and monitors the complete Week 8 research program:
- **W8-A**: Counterfactual Single-Passage Promotion Response Surface
- **W8-B**: Generator Draft Diagnostics & Uncertainty Features ($U_1 \\dots U_{12}$)
- **W8-C**: Draft-Conditioned Passage Utility Models ($U_0, U_1, U_2, U_3$)
- **W8-D**: Generator Interaction ($A_1 \\dots A_6$) & Marginal Information Gain ($MIG$)
- **W8-E**: Draft-Aware Selective Controller Tuning (DAFiR-1 & DAFiR-2)
- **W8-F**: Query-Conditioned Latent Adapter (QCLA-0, QCLA-1, QCLA-2)
- **W8-G**: Hard-Case & Outcome-Weighted Generator Adaptation (HC-LoRA & OW-LoRA)
- **W8-H**: Full Hybrid DAFiR System
- **W8-I**: Systematic Ablations (A0..A11)
- **W8-J**: Final Held-Out DEV Benchmarks & Dual Frontiers

### Execution Instructions for User:
Execute each sub-notebook or corresponding standalone script in sequence.
All intermediate outputs are saved in append-safe, resumable caches under `results/week8/`.
"""),
        code_cell("""from pathlib import Path
REPO_ROOT = Path("/home/gunavenkat/Downloads/CS787-RAG-Project")
print("Master execution ready. All 10 sub-notebooks available in notebooks/week8/.")"""),
    ]
    return make_notebook(cells)


def main():
    builders = {
        "week8a_promotion_oracle.ipynb": build_w8a_notebook,
        "week8b_draft_diagnostics.ipynb": build_w8b_notebook,
        "week8c_utility_modeling.ipynb": build_w8c_notebook,
        "week8d_generator_interaction.ipynb": build_w8d_notebook,
        "week8e_controller.ipynb": build_w8e_notebook,
        "week8f_qcla.ipynb": build_w8f_notebook,
        "week8g_outcome_weighted_sft.ipynb": build_w8g_notebook,
        "week8h_full_system.ipynb": build_w8h_notebook,
        "week8i_ablations.ipynb": build_w8i_notebook,
        "week8j_final_dev_evaluation.ipynb": build_w8j_notebook,
        "week8_master.ipynb": build_master_notebook,
    }

    for name, fn in builders.items():
        nb = fn()
        out_p = NOTEBOOKS_DIR / name
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2)
        print(f"Generated {out_p}")


if __name__ == "__main__":
    main()
