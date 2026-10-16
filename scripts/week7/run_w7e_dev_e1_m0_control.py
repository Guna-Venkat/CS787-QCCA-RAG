#!/usr/bin/env python3
"""Execute single DEV evaluation of Corrected Extract-Only M64 Control Policy (E1-M0).

Methodological Rationale:
To cleanly isolate the causal effect of retaining the SFR embedding:
- Hybrid M64 (E1-H): 4F + 2(M64 + H) + 4H
- Extract-Only M64 (E1-M0): 4F + 2M64 + 4H

Holding constant:
- Exact same semantic ranking (sentence-transformers MiniLM-L6)
- Exact same M64 extractive text
- Fixed prompt builder
- Same SARA checkpoint & greedy generation settings

Then:
Delta_SFR = F1(Hybrid M64) - F1(Extract-Only M64)
unambiguously measures the value of preserving latent whole-passage embeddings.
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

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
    print(f"Starting Corrected Extract-Only M64 Control DEV Evaluation on {device}...")

    token_cap = 64

    # 1. Load DEV queries and canonical response matrix
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

    # 2. Precompute DEV semantic scores
    print("Precomputing semantic scores for DEV queries...")
    sem_scorer = PretrainedSemanticScorer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    dev_semantic_scores = {}
    for item in dev_items:
        qid = str(item["question_id"])
        query = item["question"]
        p_texts = [p.get("text", "") for p in item["passages"][:10]]
        dev_semantic_scores[qid] = sem_scorer.score_query_passages(query, p_texts).detach().cpu().numpy()

    # 3. Load SARA Generator
    sara_ckpt_path = PROJECT_ROOT / "Baselines/SARA-main/checkpoints/finetune/sara_qasper_proj_lr5e4_seed42"
    sara_model, sara_tokenizer = load_sara_for_eval(str(sara_ckpt_path), device=device)

    # 4. Load DEV SFR Embeddings
    dev_embeds_path = PROJECT_ROOT / "results/week2/raw/dev_sfr_embeddings.pt"
    dev_sfr_raw = torch.load(dev_embeds_path, map_location="cpu")
    dev_sfr_dict = dev_sfr_raw["embeddings"] if isinstance(dev_sfr_raw, dict) and "embeddings" in dev_sfr_raw else dev_sfr_raw

    # 5. Load Precomputed Medium Extracts for Cap 64
    cache_dir = PROJECT_ROOT / "results/week7/w7e_multifidelity/medium_cache"
    medium_cache = MediumExtractionCache(cache_dir)
    dev_medium_map = {}
    for item in dev_items:
        qid = str(item["question_id"])
        dev_medium_map[qid] = {}
        order = get_semantic_rank_order(dev_semantic_scores[qid], n_passages=len(item.get("passages", [])))
        for r_idx in order[4:6]:
            row = medium_cache.get("dev", qid, r_idx, token_cap, "cross_encoder")
            if row is not None:
                dev_medium_map[qid][r_idx] = row["selected_text"]
            else:
                dev_medium_map[qid][r_idx] = item["passages"][r_idx].get("text", "")

    # 6. Run Extract-Only M64 Generation (hybrid_medium=False)
    dev_output_dir = PROJECT_ROOT / "results/week7/w7e_multifidelity/dev"
    dev_output_dir.mkdir(parents=True, exist_ok=True)
    m0_cache_path = dev_output_dir / "e1_m0_corrected_generation.jsonl"

    def e1_allocator(item):
        qid = str(item["question_id"])
        return allocate_e1_primary(len(item.get("passages", [])), semantic_scores=dev_semantic_scores[qid])

    print(f"\n--- Running Corrected DEV Extract-Only MEDIUM (E1-M0, cap={token_cap}) ---")
    df_dev_m0 = execute_resumable_multifidelity_generation(
        sara_model=sara_model,
        sara_tokenizer=sara_tokenizer,
        records=dev_items,
        allocator_fn=e1_allocator,
        medium_texts_by_qid=dev_medium_map,
        cached_sfr_embeddings=dev_sfr_dict,
        cache_path=m0_cache_path,
        split="dev",
        policy_name=f"E1_extract_only_corrected_4F_2M_4H_M{token_cap}",
        token_cap=token_cap,
        device=device,
        max_new_tokens=128,
        hybrid_medium=False,
    )

    # 7. Invariant Verification: exactly 4 heavy, 0 medium SFR embeddings
    print("Verifying Extract-Only representation invariants on all 231 DEV rows...")
    for _, row in df_dev_m0.iterrows():
        assert_multifidelity_row_invariants(
            row,
            n_full_expected=4,
            n_medium_expected=2,
            n_heavy_expected=4,
            hybrid_medium=False,
        )
    print("ALL 231 DEV ROWS PASSED EXTRACT-ONLY INVARIANT VERIFICATION!")

    m0_f1_dev = float(df_dev_m0["token_f1"].mean())
    m0_tokens_dev = float(df_dev_m0["context_tokens"].mean())

    # 8. Load Hybrid M64 Results for Clean Head-to-Head Comparison
    h_cache_path = dev_output_dir / "e1_hybrid_generation.jsonl"
    df_dev_hybrid = pd.read_json(h_cache_path, lines=True)
    h_f1_dev = float(df_dev_hybrid["token_f1"].mean())
    h_tokens_dev = float(df_dev_hybrid["context_tokens"].mean())

    delta_sfr = h_f1_dev - m0_f1_dev
    delta_tok = h_tokens_dev - m0_tokens_dev

    print(f"\n=======================================================")
    print("CLEAN CONTROLLED COMPARISON: HYBRID M64 vs EXTRACT-ONLY M64")
    print(f"=======================================================")
    print(f"Extract-Only M64 (E1-M0) : F1 = {m0_f1_dev:.4f}, Tokens = {m0_tokens_dev:.1f}")
    print(f"Hybrid M64 (E1-H)        : F1 = {h_f1_dev:.4f}, Tokens = {h_tokens_dev:.1f}")
    print(f"-------------------------------------------------------")
    print(f"Δ (SFR Retention Effect) : F1 = {delta_sfr:+.4f}, Tokens = {delta_tok:+.1f}")
    print(f"=======================================================\n")

    # 9. Assemble Controlled Ablation Table
    tables_dir = PROJECT_ROOT / "results/week7/w7e_multifidelity/tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    ablation_records = [
        {
            "Policy": "Semantic k=4 (E0)",
            "Formulation": "4F + 0M + 6H",
            "F1": round(e0_f1_dev, 4),
            "Tokens": round(e0_tokens_dev, 1),
            "ΔF1 vs k4": 0.0,
            "ΔF1 vs Extract-Only M64": round(e0_f1_dev - m0_f1_dev, 4),
            "Latent Embeddings": 6,
        },
        {
            "Policy": "Corrected Extract-Only (E1-M0)",
            "Formulation": "4F + 2M64 + 4H",
            "F1": round(m0_f1_dev, 4),
            "Tokens": round(m0_tokens_dev, 1),
            "ΔF1 vs k4": round(m0_f1_dev - e0_f1_dev, 4),
            "ΔF1 vs Extract-Only M64": 0.0,
            "Latent Embeddings": 4,
        },
        {
            "Policy": "Repaired Hybrid (E1-H)",
            "Formulation": "4F + 2(M64+H) + 4H",
            "F1": round(h_f1_dev, 4),
            "Tokens": round(h_tokens_dev, 1),
            "ΔF1 vs k4": round(h_f1_dev - e0_f1_dev, 4),
            "ΔF1 vs Extract-Only M64": round(delta_sfr, 4),
            "Latent Embeddings": 6,
        },
        {
            "Policy": "Semantic k=6 (E2)",
            "Formulation": "6F + 0M + 4H",
            "F1": round(e2_f1_dev, 4),
            "Tokens": round(e2_tokens_dev, 1),
            "ΔF1 vs k4": round(e2_f1_dev - e0_f1_dev, 4),
            "ΔF1 vs Extract-Only M64": round(e2_f1_dev - m0_f1_dev, 4),
            "Latent Embeddings": 4,
        },
    ]

    df_ablation = pd.DataFrame(ablation_records)
    df_ablation.to_csv(tables_dir / "w7e_m0_vs_m1_ablation.csv", index=False)
    print("CONTROLLED ABLATION TABLE (Isolating SFR Retention):")
    print(df_ablation.to_string(index=False))

    # 10. Stratified 3-Group Analysis for Extract-Only M64
    print("\nComputing 3-Group Stratified Analysis for Corrected Extract-Only M64...")
    df_strat_m0 = compute_stratified_4to6_analysis(
        df_eval_winner=df_dev_m0,
        df_response_matrix=df_dev_resp,
    )
    df_strat_m0.to_csv(tables_dir / "w7e_m0_stratified_analysis.csv", index=False)
    print("\nExtract-Only M64 Stratified Analysis:")
    print(df_strat_m0.to_string(index=False))

    print("\nControlled E1-M0 Evaluation Complete!")

if __name__ == "__main__":
    main()
