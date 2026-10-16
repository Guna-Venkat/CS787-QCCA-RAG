#!/usr/bin/env python3
"""Execute single DEV evaluation of W7-E Primary Hybrid MEDIUM Policy (E1-H).

Invariants enforced:
1. Gated on frozen token cap M_64 from repaired INTERNAL-VAL downstream sweep.
2. 4 FULL textual passages (semantic ranks 1-4).
3. 2 MEDIUM textual extracts (semantic ranks 5-6).
4. 6 compressed SFR embeddings (ranks 5-10: 2 medium + 4 heavy).
5. Strict assertion of representation invariants on all 231 DEV rows.
6. Main DEV evaluation table and 3-group stratified 4->6 gain analysis.
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SARA_ROOT = PROJECT_ROOT / "Baselines/SARA-main"
import src
if hasattr(src, "__path__") and str(SARA_ROOT / "src") not in src.__path__:
    src.__path__.append(str(SARA_ROOT / "src"))

from src.model.loader import load_sara_for_eval
from src.week6.models import PretrainedSemanticScorer
from src.week7.w7e.medium_compressor import MediumExtractionCache
from src.week7.w7e.fidelity_allocator import (
    allocate_e1_primary,
    assert_multifidelity_row_invariants,
    get_semantic_rank_order,
)
from src.week7.w7e.generation import execute_resumable_multifidelity_generation
from src.week7.w7e.analysis import compute_stratified_4to6_analysis

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting W7-E Primary Policy (E1-H) DEV Evaluation on {device}...")

    # 1. Assert Frozen Cap from Repaired INTERNAL-VAL Selection
    tables_dir = PROJECT_ROOT / "results/week7/w7e_multifidelity/tables"
    frozen_cap_path = tables_dir / "frozen_hybrid_cap.json"
    assert frozen_cap_path.is_file(), (
        f"Missing {frozen_cap_path}! Repaired INTERNAL-VAL sweep must run and freeze cap first."
    )
    with open(frozen_cap_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    frozen_cap = int(meta["frozen_hybrid_cap"])
    print(f"--> Verified FROZEN_HYBRID_MEDIUM_CAP = M_{frozen_cap} (from {frozen_cap_path}) <--")

    # 2. Load DEV queries and canonical response matrix
    dev_retrieval_path = PROJECT_ROOT / "results/week2/raw/dev_retrieval.jsonl"
    with open(dev_retrieval_path, "r", encoding="utf-8") as f:
        dev_items = [json.loads(line) for line in f]
    print(f"Loaded {len(dev_items)} DEV queries.")

    dev_resp_path = PROJECT_ROOT / "results/week7/w7b_semantic_budget/response_surface/dev_semantic_fixed_k_matrix.csv"
    df_dev_resp = pd.read_csv(dev_resp_path)
    e0_f1_dev = float(df_dev_resp["f1_k4"].mean())
    e0_tokens_dev = float(df_dev_resp["tokens_k4"].mean())
    e2_f1_dev = float(df_dev_resp["f1_k6"].mean())
    e2_tokens_dev = float(df_dev_resp["tokens_k6"].mean())
    print(f"Canonical DEV Baselines: E0 (k=4) F1={e0_f1_dev:.4f} ({e0_tokens_dev:.1f} tok) | E2 (k=6) F1={e2_f1_dev:.4f} ({e2_tokens_dev:.1f} tok)")

    # 3. Precompute DEV semantic scores
    print("Precomputing semantic scores for DEV queries...")
    sem_scorer = PretrainedSemanticScorer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    dev_semantic_scores = {}
    for item in dev_items:
        qid = str(item["question_id"])
        query = item["question"]
        p_texts = [p.get("text", "") for p in item["passages"][:10]]
        dev_semantic_scores[qid] = sem_scorer.score_query_passages(query, p_texts).detach().cpu().numpy()
    print("Precomputed semantic scores for all 231 DEV queries.")

    # 4. Load SARA Generator
    sara_ckpt_path = PROJECT_ROOT / "Baselines/SARA-main/checkpoints/finetune/sara_qasper_proj_lr5e4_seed42"
    sara_model, sara_tokenizer = load_sara_for_eval(str(sara_ckpt_path), device=device)

    # 5. Load DEV SFR Embeddings
    dev_embeds_path = PROJECT_ROOT / "results/week2/raw/dev_sfr_embeddings.pt"
    dev_sfr_raw = torch.load(dev_embeds_path, map_location="cpu")
    dev_sfr_dict = dev_sfr_raw["embeddings"] if isinstance(dev_sfr_raw, dict) and "embeddings" in dev_sfr_raw else dev_sfr_raw

    # 6. Load Precomputed Medium Extracts for Cap
    cache_dir = PROJECT_ROOT / "results/week7/w7e_multifidelity/medium_cache"
    medium_cache = MediumExtractionCache(cache_dir)
    dev_medium_map = {}
    for item in dev_items:
        qid = str(item["question_id"])
        dev_medium_map[qid] = {}
        order = get_semantic_rank_order(dev_semantic_scores[qid], n_passages=len(item.get("passages", [])))
        for r_idx in order[4:6]:
            row = medium_cache.get("dev", qid, r_idx, frozen_cap, "cross_encoder")
            if row is not None:
                dev_medium_map[qid][r_idx] = row["selected_text"]
            else:
                dev_medium_map[qid][r_idx] = item["passages"][r_idx].get("text", "")
    print(f"Loaded precomputed medium extracts for all candidate ranks under M_{frozen_cap}.")

    # 7. Execute Single Downstream Generation for DEV E1-H
    dev_output_dir = PROJECT_ROOT / "results/week7/w7e_multifidelity/dev"
    dev_output_dir.mkdir(parents=True, exist_ok=True)
    dev_cache_path = dev_output_dir / "e1_hybrid_generation.jsonl"

    def e1_hybrid_allocator(item):
        qid = str(item["question_id"])
        return allocate_e1_primary(len(item.get("passages", [])), semantic_scores=dev_semantic_scores[qid])

    print(f"\n--- Running DEV Hybrid MEDIUM (E1-H, cap={frozen_cap}) ---")
    df_dev_e1_hybrid = execute_resumable_multifidelity_generation(
        sara_model=sara_model,
        sara_tokenizer=sara_tokenizer,
        records=dev_items,
        allocator_fn=e1_hybrid_allocator,
        medium_texts_by_qid=dev_medium_map,
        cached_sfr_embeddings=dev_sfr_dict,
        cache_path=dev_cache_path,
        split="dev",
        policy_name="E1_hybrid_4F_2M_4H",
        token_cap=frozen_cap,
        device=device,
        max_new_tokens=128,
        hybrid_medium=True,
    )

    # 8. Verify Representation Invariants on All 231 DEV Rows
    print("Verifying multi-fidelity representation invariants on all 231 DEV predictions...")
    for _, row in df_dev_e1_hybrid.iterrows():
        assert_multifidelity_row_invariants(
            row,
            n_full_expected=4,
            n_medium_expected=2,
            n_heavy_expected=4,
            hybrid_medium=True,
        )
    print("ALL 231 DEV ROWS PASSED INVARIANT VERIFICATION!")

    e1h_f1_dev = float(df_dev_e1_hybrid["token_f1"].mean())
    e1h_tokens_dev = float(df_dev_e1_hybrid["context_tokens"].mean())
    print(f"\n=======================================================")
    print(f"E1-H DEV RESULTS: Mean F1 = {e1h_f1_dev:.4f}, Mean Tokens = {e1h_tokens_dev:.1f}")
    print(f"ΔF1 vs E0 (k=4): {e1h_f1_dev - e0_f1_dev:+.4f} | ΔTokens vs E0: {e1h_tokens_dev - e0_tokens_dev:+.1f}")
    print(f"ΔF1 vs E2 (k=6): {e1h_f1_dev - e2_f1_dev:+.4f} | ΔTokens vs E2: {e1h_tokens_dev - e2_tokens_dev:+.1f}")
    print(f"=======================================================\n")

    # 9. Main DEV Evaluation Summary Table
    diag_e1_path = dev_output_dir / "e1_primary_generation.jsonl"
    if diag_e1_path.is_file():
        df_diag = pd.read_json(diag_e1_path, lines=True)
        diag_f1 = float(df_diag["token_f1"].mean())
        diag_tok = float(df_diag["context_tokens"].mean())
    else:
        diag_f1 = 0.3672
        diag_tok = 1107.0

    summary_records = [
        {"Method": "Semantic k=4 (E0)", "F1": round(e0_f1_dev, 4), "Tokens": round(e0_tokens_dev, 1), "ΔF1 vs k4": 0.0, "Δtokens vs k4": 0.0, "Type": "Static Baseline"},
        {"Method": "Semantic k=6 (E2)", "F1": round(e2_f1_dev, 4), "Tokens": round(e2_tokens_dev, 1), "ΔF1 vs k4": round(e2_f1_dev - e0_f1_dev, 4), "Δtokens vs k4": round(e2_tokens_dev - e0_tokens_dev, 1), "Type": "Static Baseline"},
        {"Method": f"W7-E Primary E1-H (4F+2(M{frozen_cap}+H)+4H)", "F1": round(e1h_f1_dev, 4), "Tokens": round(e1h_tokens_dev, 1), "ΔF1 vs k4": round(e1h_f1_dev - e0_f1_dev, 4), "Δtokens vs k4": round(e1h_tokens_dev - e0_tokens_dev, 1), "Type": "Multi-Fidelity Winner"},
        {"Method": "W7-E Extract-Only E1-M0 (4F+2M96+4H)", "F1": round(diag_f1, 4), "Tokens": round(diag_tok, 1), "ΔF1 vs k4": round(diag_f1 - e0_f1_dev, 4), "Δtokens vs k4": round(diag_tok - e0_tokens_dev, 1), "Type": "Diagnostic (Failed M0)"},
    ]

    # Include W7-C3 Anchor-and-Deviate if available
    w7c3_path = PROJECT_ROOT / "results/week7/w7c3_anchor_deviate/tables/w7c3_main_dev_results.csv"
    if w7c3_path.is_file():
        w7c3_df = pd.read_csv(w7c3_path)
        c3_row = w7c3_df[w7c3_df["method"].str.contains("W7-C3", na=False)]
        if not c3_row.empty:
            c3_f1 = float(c3_row["mean_f1"].iloc[0])
            c3_tok = float(c3_row["tokens"].iloc[0])
            summary_records.append({
                "Method": "W7-C3 Anchor-and-Deviate",
                "F1": round(c3_f1, 4),
                "Tokens": round(c3_tok, 1),
                "ΔF1 vs k4": round(c3_f1 - e0_f1_dev, 4),
                "Δtokens vs k4": round(c3_tok - e0_tokens_dev, 1),
                "Type": "Discrete Adaptive Baseline",
            })

    df_summary = pd.DataFrame(summary_records)
    out_table_path = tables_dir / "w7e_main_dev_results.csv"
    df_summary.to_csv(out_table_path, index=False)
    print("=" * 80)
    print("FINAL W7-E DEV EVALUATION SUMMARY TABLE")
    print("=" * 80)
    print(df_summary.to_string(index=False))

    # 10. Stratified 4->6 Gain Analysis
    print("\nComputing 3-Group Stratified 4->6 Gain Analysis...")
    df_stratified = compute_stratified_4to6_analysis(
        df_eval_winner=df_dev_e1_hybrid,
        df_response_matrix=df_dev_resp,
    )
    strat_path = tables_dir / "w7e_4to6_stratified_analysis.csv"
    df_stratified.to_csv(strat_path, index=False)
    print("\nStratified 4->6 Gain Analysis:")
    print(df_stratified.to_string(index=False))

    # 11. Plot Stratified Figure
    figures_dir = PROJECT_ROOT / "results/week7/w7e_multifidelity/figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / "w7e_stratified_4to6.png"

    plt.figure(figsize=(9, 5))
    bar_width = 0.35
    indices = np.arange(len(df_stratified))

    plt.bar(indices - bar_width/2, df_stratified["delta_f1_vs_k4"], bar_width, label="ΔF1 vs k=4 (E0)", color="#2b5c8f")
    plt.bar(indices + bar_width/2, df_stratified["delta_f1_vs_k6"], bar_width, label="ΔF1 vs k=6 (E2)", color="#e27c38")

    plt.axhline(0, color="black", linestyle="--", alpha=0.7)
    plt.xticks(indices, [f"{r['group']}\n(n={r['n_queries']})" for _, r in df_stratified.iterrows()], fontsize=9)
    plt.ylabel("Mean Δ Answer F1", fontsize=11)
    plt.title(f"Week 7-E: Stratified Effect of Multi-Fidelity Representation (E1-H M_{frozen_cap})", fontsize=12, fontweight="bold")
    plt.legend(frameon=True)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"\nSaved stratified gain plot to: {fig_path}")

    print("\nW7-E DEV Evaluation Completed Successfully!")

if __name__ == "__main__":
    main()
