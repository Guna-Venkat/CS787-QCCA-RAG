#!/usr/bin/env python3
"""Execute repaired INTERNAL-VAL downstream token-cap sweep for Hybrid MEDIUM.

Evaluates on the 342 INTERNAL-VAL queries:
- 4F + 2(M64 + H) + 4H
- 4F + 2(M96 + H) + 4H
- 4F + 2(M128 + H) + 4H

Outputs:
- Resumable generation caches in results/week7/w7e_multifidelity/train/
- Comparative selection table in results/week7/w7e_multifidelity/tables/w7e_hybrid_cap_selection.csv
- Freezes the winner using predeclared rule: highest F1, within 0.003 choose fewer tokens.
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SARA_ROOT = PROJECT_ROOT / "Baselines/SARA-main"
import src
if hasattr(src, "__path__") and str(SARA_ROOT / "src") not in src.__path__:
    src.__path__.append(str(SARA_ROOT / "src"))

from src.model.loader import load_sara_for_eval
from src.week6.models import PretrainedSemanticScorer
from src.week7.w7e.medium_compressor import MediumExtractionCache, extract_medium_passage
from src.week7.w7e.sentence_scorer import SentenceScorer
from src.week7.w7e.sentence_splitter import split_into_sentences
from src.week7.w7e.fidelity_allocator import (
    allocate_e1_primary,
    assert_multifidelity_row_invariants,
    get_semantic_rank_order,
)
from src.week7.w7e.generation import execute_resumable_multifidelity_generation

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting Repaired INTERNAL-VAL Hybrid Cap Selection on {device}...")

    # 1. Load manifest and items
    manifest_path = PROJECT_ROOT / "results/week5/content_allocator/w5b/splits/internal_val_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    val_qids = set(manifest["question_id"].astype(str))

    retrieval_path = PROJECT_ROOT / "results/week5/train_response_surface/train_retrieval.jsonl"
    with open(retrieval_path, "r", encoding="utf-8") as f:
        val_items = [json.loads(line) for line in f if str(json.loads(line)["question_id"]) in val_qids]

    print(f"Loaded {len(val_items)} internal validation queries.")

    # Canonical baseline reference metrics on INTERNAL-VAL
    resp_matrix_path = PROJECT_ROOT / "results/week7/w7b_semantic_budget/response_surface/train_semantic_fixed_k_matrix.csv"
    resp_df = pd.read_csv(resp_matrix_path)
    val_resp_df = resp_df[resp_df["question_id"].astype(str).isin(val_qids)]
    ref_k4_f1 = float(val_resp_df["f1_k4"].mean())
    ref_k4_tok = float(val_resp_df["tokens_k4"].mean())
    ref_k6_f1 = float(val_resp_df["f1_k6"].mean())
    ref_k6_tok = float(val_resp_df["tokens_k6"].mean())

    print(f"Canonical Internal-Val k=4: F1={ref_k4_f1:.4f}, Tokens={ref_k4_tok:.1f}")
    print(f"Canonical Internal-Val k=6: F1={ref_k6_f1:.4f}, Tokens={ref_k6_tok:.1f}")

    # 2. Precompute semantic relevance scores
    sem_scorer = PretrainedSemanticScorer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    val_semantic_scores = {}
    for item in val_items:
        qid = str(item["question_id"])
        query = item["question"]
        p_texts = [p.get("text", "") for p in item["passages"][:10]]
        val_semantic_scores[qid] = sem_scorer.score_query_passages(query, p_texts).detach().cpu().numpy()
    print("Precomputed semantic scores for all internal validation items.")

    # 3. Load SARA model
    sara_ckpt_path = PROJECT_ROOT / "Baselines/SARA-main/checkpoints/finetune/sara_qasper_proj_lr5e4_seed42"
    sara_model, sara_tokenizer = load_sara_for_eval(str(sara_ckpt_path), device=device)

    # 4. Load SFR embeddings
    embeds_path = PROJECT_ROOT / "results/week5/train_response_surface/train_sfr_embeddings.pt"
    sfr_raw = torch.load(embeds_path, map_location="cpu")
    train_sfr_dict = sfr_raw["embeddings"] if isinstance(sfr_raw, dict) and "embeddings" in sfr_raw else sfr_raw

    # 5. Populate medium extractions for caps 64, 96, 128
    cache_dir = PROJECT_ROOT / "results/week7/w7e_multifidelity/medium_cache"
    medium_cache = MediumExtractionCache(cache_dir)
    sentence_scorer = SentenceScorer(scorer_type="cross_encoder", device=device)

    val_medium_by_cap = {cap: {} for cap in [64, 96, 128]}
    print("Ensuring medium extracts cached for all internal validation items across caps [64, 96, 128]...")
    for item in tqdm(val_items, desc="Caching Medium Extracts"):
        qid = str(item["question_id"])
        pid = str(item.get("paper_id", ""))
        query = item["question"]
        passages = item.get("passages", [])[:10]

        for cap in [64, 96, 128]:
            val_medium_by_cap[cap][qid] = {}

        # Target candidate semantic ranks 5 and 6 (which require MEDIUM extraction in E1)
        order = get_semantic_rank_order(val_semantic_scores[qid], n_passages=len(passages))
        candidate_ranks = order[4:6]

        for r_idx in candidate_ranks:
            p_text = passages[r_idx].get("text", "").strip()
            if not p_text:
                continue
            p_id = passages[r_idx].get("passage_id", f"p_{r_idx}")

            # Check existing cache
            cached_rows = {
                cap: medium_cache.get("internal_val", qid, r_idx, cap, "cross_encoder")
                for cap in [64, 96, 128]
            }

            # If any cap is missing, compute sentence scores once and extract for all missing caps
            missing_caps = [cap for cap, row in cached_rows.items() if row is None]
            if missing_caps:
                sentences = split_into_sentences(p_text)
                if sentences:
                    scores = sentence_scorer.score_sentences(query, sentences)
                else:
                    sentences = [p_text]
                    scores = [0.0]

                for cap in missing_caps:
                    ext = extract_medium_passage(
                        query=query,
                        passage_text=p_text,
                        token_cap=cap,
                        scorer=sentence_scorer,
                        tokenizer=sara_tokenizer,
                        sentence_scores=scores,
                    )
                    med_text = ext["selected_text"]
                    medium_cache.put(
                        split="internal_val",
                        question_id=qid,
                        paper_id=pid,
                        passage_id=p_id,
                        semantic_rank=r_idx,
                        original_text=p_text,
                        selected_sentence_ids=ext["selected_sentence_ids"],
                        selected_text=med_text,
                        sentence_scores=ext["sentence_scores"],
                        original_tokens=ext["original_tokens"],
                        medium_tokens=ext["medium_tokens"],
                        token_cap=cap,
                        scorer_type="cross_encoder",
                    )
                    cached_rows[cap] = {"selected_text": med_text}

            for cap in [64, 96, 128]:
                val_medium_by_cap[cap][qid][r_idx] = cached_rows[cap]["selected_text"]

    medium_cache.flush()
    print("Medium extractions verified and flushed to disk.")

    # 6. Run downstream generation across caps 64, 96, 128
    output_dir = PROJECT_ROOT / "results/week7/w7e_multifidelity/train"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_by_cap = {}
    for cap in [64, 96, 128]:
        cache_path = output_dir / f"e1h_m{cap}_val_generation.jsonl"
        print(f"\n--- Running INTERNAL-VAL Hybrid MEDIUM (M{cap}) ---")

        def allocator_fn(item):
            qid = str(item["question_id"])
            return allocate_e1_primary(len(item.get("passages", [])), semantic_scores=val_semantic_scores[qid])

        df_cap = execute_resumable_multifidelity_generation(
            sara_model=sara_model,
            sara_tokenizer=sara_tokenizer,
            records=val_items,
            allocator_fn=allocator_fn,
            medium_texts_by_qid=val_medium_by_cap[cap],
            cached_sfr_embeddings=train_sfr_dict,
            cache_path=cache_path,
            split="internal_val",
            policy_name=f"E1_hybrid_4F_2M_4H_M{cap}",
            token_cap=cap,
            device=device,
            max_new_tokens=128,
            hybrid_medium=True,
        )

        for _, row in df_cap.iterrows():
            assert_multifidelity_row_invariants(row, n_full_expected=4, n_medium_expected=2, n_heavy_expected=4, hybrid_medium=True)

        results_by_cap[cap] = df_cap
        f1_m = float(df_cap["token_f1"].mean())
        tok_m = float(df_cap["context_tokens"].mean())
        print(f"M{cap} Completed: Mean F1 = {f1_m:.4f}, Mean Tokens = {tok_m:.1f}")

    # 7. Assemble Comparative Selection Table
    table_rows = []
    for cap in [64, 96, 128]:
        df_c = results_by_cap[cap]
        m_f1 = float(df_c["token_f1"].mean())
        m_tok = float(df_c["context_tokens"].mean())
        table_rows.append({
            "Hybrid cap": f"M{cap}",
            "F1": round(m_f1, 4),
            "Tokens": round(m_tok, 1),
            "ΔF1 vs k4": round(m_f1 - ref_k4_f1, 4),
            "Δtokens vs k4": round(m_tok - ref_k4_tok, 1),
            "ΔF1 vs k6": round(m_f1 - ref_k6_f1, 4),
        })

    df_table = pd.DataFrame(table_rows)
    tables_dir = PROJECT_ROOT / "results/week7/w7e_multifidelity/tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    table_out_path = tables_dir / "w7e_hybrid_cap_selection.csv"
    df_table.to_csv(table_out_path, index=False)

    print("\n" + "=" * 70)
    print("INTERNAL-VAL HYBRID MEDIUM TOKEN CAP SELECTION TABLE")
    print("=" * 70)
    print(df_table.to_string(index=False))

    # 8. Apply Predeclared Freezing Rule
    # Rule: highest INTERNAL-VAL F1; within .003 choose fewer tokens
    max_f1 = df_table["F1"].max()
    within_eps = df_table[df_table["F1"] >= max_f1 - 0.003]
    winner_row = within_eps.sort_values("Tokens", ascending=True).iloc[0]
    frozen_cap = int(winner_row["Hybrid cap"].replace("M", ""))

    print("\n" + "=" * 70)
    print(f"--> FROZEN HYBRID MEDIUM TOKEN CAP WINNER: M_{frozen_cap} <--")
    print(f"Selected via rule: highest F1 ({max_f1:.4f}); within 0.003 tolerance chose lowest tokens ({winner_row['Tokens']})")
    print("=" * 70)

    # Save selection metadata
    meta = {
        "frozen_hybrid_cap": frozen_cap,
        "selection_rule": "highest F1 on INTERNAL-VAL; within 0.003 choose fewer tokens",
        "timestamp": time.time(),
        "summary": df_table.to_dict(orient="records"),
    }
    with open(tables_dir / "frozen_hybrid_cap.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
