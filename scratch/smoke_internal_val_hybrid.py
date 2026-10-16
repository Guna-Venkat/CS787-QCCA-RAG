#!/usr/bin/env python3
"""Smoke test for Internal-Val Hybrid MEDIUM generation on 3 queries."""

import json
import time
import torch
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SARA_ROOT = PROJECT_ROOT / "Baselines/SARA-main"
import src
if hasattr(src, "__path__") and str(SARA_ROOT / "src") not in src.__path__:
    src.__path__.append(str(SARA_ROOT / "src"))

from transformers import AutoTokenizer
from src.model.loader import load_sara_for_eval
from src.week6.models import PretrainedSemanticScorer
from src.week7.w7e.medium_compressor import MediumExtractionCache
from src.week7.w7e.sentence_scorer import SentenceScorer
from src.week7.w7e.sentence_splitter import split_into_sentences
from src.week7.w7e.fidelity_allocator import allocate_e1_primary, assert_multifidelity_row_invariants
from src.week7.w7e.generation import execute_resumable_multifidelity_generation

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running smoke test on {device}...")

    # Load 3 internal val items
    manifest = pd.read_csv("results/week5/content_allocator/w5b/splits/internal_val_manifest.csv")
    val_qids = set(manifest["question_id"].astype(str))

    with open("results/week5/train_response_surface/train_retrieval.jsonl") as f:
        val_items = [json.loads(line) for line in f if str(json.loads(line)["question_id"]) in val_qids][:3]

    print(f"Loaded {len(val_items)} sample items: {[item['question_id'] for item in val_items]}")

    # Load SARA
    sara_ckpt_path = Path("Baselines/SARA-main/checkpoints/finetune/sara_qasper_proj_lr5e4_seed42")
    sara_model, sara_tokenizer = load_sara_for_eval(str(sara_ckpt_path), device=device)

    # Precompute semantic scores
    sem_scorer = PretrainedSemanticScorer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    val_semantic_scores = {}
    for item in val_items:
        qid = str(item["question_id"])
        query = item["question"]
        p_texts = [p.get("text", "") for p in item["passages"][:10]]
        val_semantic_scores[qid] = sem_scorer.score_query_passages(query, p_texts).detach().cpu().numpy()

    # Load train SFR embeddings
    embeds_path = Path("results/week5/train_response_surface/train_sfr_embeddings.pt")
    sfr_raw = torch.load(embeds_path, map_location="cpu")
    train_sfr_dict = sfr_raw["embeddings"] if isinstance(sfr_raw, dict) and "embeddings" in sfr_raw else sfr_raw

    # Prepare medium extracts for caps 64, 96, 128
    sentence_scorer = SentenceScorer(scorer_type="cross_encoder", device=device)
    val_medium_by_cap = {cap: {} for cap in [64, 96, 128]}
    for item in val_items:
        qid = str(item["question_id"])
        query = item["question"]
        for cap in [64, 96, 128]:
            val_medium_by_cap[cap][qid] = {}
            for r_idx in range(len(item["passages"][:10])):
                p_text = item["passages"][r_idx].get("text", "")
                if not p_text.strip():
                    continue
                sentences = split_into_sentences(p_text)
                if not sentences:
                    continue
                sc = sentence_scorer.score_sentences(query, sentences)
                # Take top sentence
                val_medium_by_cap[cap][qid][r_idx] = sentences[int(np.argmax(sc))]

    # Test execution for M64
    smoke_cache = Path("results/week7/w7e_multifidelity/diagnostics/smoke/val_smoke_m64.jsonl")
    if smoke_cache.is_file():
        smoke_cache.unlink()

    def allocator_fn(item):
        qid = str(item["question_id"])
        return allocate_e1_primary(len(item.get("passages", [])), semantic_scores=val_semantic_scores[qid])

    df_smoke = execute_resumable_multifidelity_generation(
        sara_model=sara_model,
        sara_tokenizer=sara_tokenizer,
        records=val_items,
        allocator_fn=allocator_fn,
        medium_texts_by_qid=val_medium_by_cap[64],
        cached_sfr_embeddings=train_sfr_dict,
        cache_path=smoke_cache,
        split="internal_val",
        policy_name="E1_hybrid_4F_2M_4H",
        token_cap=64,
        device=device,
        max_new_tokens=64,
        hybrid_medium=True,
    )

    print(f"\nSmoke execution complete! Shape: {df_smoke.shape}")
    for _, row in df_smoke.iterrows():
        assert_multifidelity_row_invariants(row, n_full_expected=4, n_medium_expected=2, n_heavy_expected=4, hybrid_medium=True)
        print(f"QID {row['question_id']}: F1={row['token_f1']:.4f}, Tokens={row['context_tokens']}, Pred={repr(row['prediction'][:50])}")

    print("\n--> Smoke Test Succeeded! All invariants and generation verified. <--")

if __name__ == "__main__":
    main()
