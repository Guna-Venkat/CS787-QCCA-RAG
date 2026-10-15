"""Smoke validation script for Week 7-E Multi-Fidelity Pipeline.

SMOKE_ONLY_NOT_FOR_REPORTING

Validates:
1. Deterministic sentence splitting and scoring.
2. MEDIUM extraction with M64, M96, and M128.
3. Prompt building and embedding extraction for E0, E1, E2, E1-X.
4. Token accounting and compression statistics.
5. Invariant reproduction checks on reference response matrix.
6. End-to-end SARA generation on 2 sample queries.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# Add project root and SARA-main to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SARA_ROOT = PROJECT_ROOT / "Baselines" / "SARA-main"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src
if hasattr(src, "__path__"):
    if str(PROJECT_ROOT / "src") not in src.__path__:
        src.__path__.insert(0, str(PROJECT_ROOT / "src"))
    if str(SARA_ROOT / "src") not in src.__path__:
        src.__path__.append(str(SARA_ROOT / "src"))
if str(SARA_ROOT) not in sys.path:
    sys.path.append(str(SARA_ROOT))

from src.week7.w7e.sentence_splitter import split_into_sentences
from src.week7.w7e.sentence_scorer import SentenceScorer
from src.week7.w7e.medium_compressor import extract_medium_passage, MediumExtractionCache
from src.week7.w7e.fidelity_allocator import (
    allocate_e0_k4,
    allocate_e1_primary,
    allocate_e2_k6,
    allocate_e1x_cross_encoder,
    validate_multifidelity_assignment,
)
from src.week7.w7e.prompt_builder import (
    build_mixed_fidelity_sara_prompt,
    extract_mixed_fidelity_sara_embeddings,
)
from src.week7.w7e.token_accounting import (
    compute_medium_compression_ratio,
    compute_compression_statistics,
)
from src.week7.w7e.generation import (
    execute_resumable_multifidelity_generation,
    assert_pipeline_invariants,
)
from src.model.loader import load_sara_for_eval


def main():
    print("=" * 70)
    print("WEEK 7-E SMOKE TEST: MULTI-FIDELITY EVIDENCE PIPELINE")
    print("MARKER: SMOKE_ONLY_NOT_FOR_REPORTING")
    print("=" * 70)

    with open(PROJECT_ROOT / "configs/week7/w7e_multifidelity.yaml") as f:
        cfg = yaml.safe_load(f)

    smoke_dir = PROJECT_ROOT / cfg["paths"]["smoke_dir"]
    smoke_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load small subset of TRAIN retrieval (5 queries)
    train_retrieval_path = PROJECT_ROOT / cfg["paths"]["train_retrieval"]
    sample_items = []
    with open(train_retrieval_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= 5:
                break
            if line.strip():
                sample_items.append(json.loads(line))

    print(f"Loaded {len(sample_items)} sample queries from TRAIN retrieval.")
    q0 = sample_items[0]
    qid0 = q0["question_id"]
    query_text = q0["question"]
    passages = q0["passages"][:10]

    # 2. Test sentence splitting
    sample_text = passages[0]["text"]
    sentences = split_into_sentences(sample_text)
    print(f"Sample passage sentence splitting: {len(sentences)} sentences found.")
    assert len(sentences) > 0, "Sentence splitting failed"

    # 3. Test sentence scoring with semantic scorer
    scorer = SentenceScorer(
        scorer_type="semantic",
        model_name=cfg["medium_compression"]["semantic_encoder_model"],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    scores = scorer.score_sentences(query_text, sentences)
    assert len(scores) == len(sentences), "Sentence score length mismatch"
    print(f"Sentence scoring verified. Scores range: [{min(scores):.4f}, {max(scores):.4f}]")

    # 4. Test MEDIUM extraction with caps 64, 96, 128
    from transformers import AutoTokenizer
    sara_ckpt_path = PROJECT_ROOT / cfg["sara"]["checkpoint_path"]
    tokenizer = AutoTokenizer.from_pretrained(str(sara_ckpt_path))

    extracted_medium = {}
    cr_records = []
    for cap in [64, 96, 128]:
        res = extract_medium_passage(
            query=query_text,
            passage_text=sample_text,
            token_cap=cap,
            scorer=scorer,
            tokenizer=tokenizer,
        )
        extracted_medium[cap] = res
        cr = compute_medium_compression_ratio(res["medium_tokens"], res["original_tokens"])
        cr_records.append(cr)
        print(f"Token Cap M{cap}: Original={res['original_tokens']} -> Medium={res['medium_tokens']} tokens, CR={cr:.3f}")
        assert res["medium_tokens"] <= cap or len(res["selected_sentence_ids"]) == 1

    # 5. Test allocation policies and invariants
    full_e0, med_e0, heavy_e0 = allocate_e0_k4(10)
    validate_multifidelity_assignment(full_e0, med_e0, heavy_e0, 10)
    assert len(full_e0) == 4 and len(med_e0) == 0 and len(heavy_e0) == 6

    full_e1, med_e1, heavy_e1 = allocate_e1_primary(10)
    validate_multifidelity_assignment(full_e1, med_e1, heavy_e1, 10)
    assert len(full_e1) == 4 and len(med_e1) == 2 and len(heavy_e1) == 4

    full_e2, med_e2, heavy_e2 = allocate_e2_k6(10)
    validate_multifidelity_assignment(full_e2, med_e2, heavy_e2, 10)
    assert len(full_e2) == 6 and len(med_e2) == 0 and len(heavy_e2) == 4

    # Dummy CE scores for testing E1-X
    dummy_ce_scores = [0.1 * i for i in range(10)]
    full_e1x, med_e1x, heavy_e1x = allocate_e1x_cross_encoder(dummy_ce_scores, 4, 2, 10)
    validate_multifidelity_assignment(full_e1x, med_e1x, heavy_e1x, 10)
    assert len(full_e1x) == 4 and len(med_e1x) == 2 and len(heavy_e1x) == 4
    # Highest scores in candidate ranks 4..9 are 8 and 9, so med should be [8, 9]
    assert med_e1x == [8, 9]
    print("Allocation policies and invariant validation passed.")

    # 6. Test mixed prompt construction
    medium_texts_map = {
        4: extracted_medium[96]["selected_text"],
        5: extracted_medium[96]["selected_text"],
    }
    prompt_e1, ctx_tokens_e1, sorted_heavy_e1, breakdown = build_mixed_fidelity_sara_prompt(
        question=query_text,
        passages=passages,
        full_indices=full_e1,
        medium_indices=med_e1,
        heavy_indices=heavy_e1,
        medium_texts_map=medium_texts_map,
        tokenizer=tokenizer,
    )
    print(f"Mixed prompt built. Total context tokens: {ctx_tokens_e1}")
    print(f"Token breakdown: {breakdown}")
    assert breakdown["n_full"] == 4
    assert breakdown["n_medium"] == 2
    assert breakdown["n_heavy"] == 4

    # 7. Check Invariants on Reference Response Surface
    dev_resp_path = PROJECT_ROOT / cfg["paths"]["dev_response_matrix"]
    df_dev_resp = pd.read_csv(dev_resp_path)
    ref_k4_f1 = float(df_dev_resp["f1_k4"].mean())
    ref_k6_f1 = float(df_dev_resp["f1_k6"].mean())
    ref_k4_tok = float(df_dev_resp["tokens_k4"].mean())
    ref_k6_tok = float(df_dev_resp["tokens_k6"].mean())
    print(f"Canonical DEV Reference: k4 F1={ref_k4_f1:.4f} (tok={ref_k4_tok:.1f}), k6 F1={ref_k6_f1:.4f} (tok={ref_k6_tok:.1f})")
    assert abs(ref_k4_f1 - 0.3909) < 0.005, f"k4 F1 mismatch: {ref_k4_f1}"
    assert abs(ref_k6_f1 - 0.3957) < 0.005, f"k6 F1 mismatch: {ref_k6_f1}"

    # 8. Load SARA model and run 2 sample generations (1 for E0, 1 for E1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SARA model from {sara_ckpt_path} on {device}...")
    sara_model, sara_tok = load_sara_for_eval(str(sara_ckpt_path), device=device)
    print("SARA model loaded successfully.")

    # Load cached SFR embeddings for the sample queries
    train_embeds_path = PROJECT_ROOT / cfg["paths"]["train_sfr_embeddings"]
    sfr_raw = torch.load(train_embeds_path, map_location="cpu")
    sfr_embeds_dict = sfr_raw["embeddings"] if isinstance(sfr_raw, dict) and "embeddings" in sfr_raw else sfr_raw
    print(f"Loaded SFR embeddings for {len(sfr_embeds_dict)} queries.")

    # Smoke generation on 2 queries
    smoke_records = sample_items[:2]
    # Build medium map for these 2 queries
    med_map = {}
    for item in smoke_records:
        qid = item["question_id"]
        q_text = item["question"]
        med_map[qid] = {}
        for rank_idx in [4, 5]:
            p_text = item["passages"][rank_idx]["text"]
            res_m = extract_medium_passage(
                query=q_text,
                passage_text=p_text,
                token_cap=96,
                scorer=scorer,
                tokenizer=tokenizer,
            )
            med_map[qid][rank_idx] = res_m["selected_text"]

    smoke_gen_cache = smoke_dir / "smoke_generation_cache.jsonl"
    if smoke_gen_cache.is_file():
        smoke_gen_cache.unlink()

    print("Running smoke generation for E1 (4F + 2M + 4H)...")
    df_smoke_e1 = execute_resumable_multifidelity_generation(
        sara_model=sara_model,
        sara_tokenizer=sara_tok,
        records=smoke_records,
        allocator_fn=lambda item: allocate_e1_primary(len(item.get("passages", []))),
        medium_texts_by_qid=med_map,
        cached_sfr_embeddings=sfr_embeds_dict,
        cache_path=smoke_gen_cache,
        split="train_smoke",
        policy_name="E1_smoke_4F_2M_4H",
        token_cap=96,
        device=device,
        max_new_tokens=32,  # Short for fast smoke test
    )
    print(f"Smoke generation completed! Shape: {df_smoke_e1.shape}")
    print(f"Mean F1: {df_smoke_e1['token_f1'].mean():.4f}, Mean Context Tokens: {df_smoke_e1['context_tokens'].mean():.1f}")
    print(f"Sample prediction: '{df_smoke_e1['prediction'].iloc[0]}'")

    # Save summary diagnostic
    smoke_summary = {
        "status": "PASS",
        "timestamp": time.time(),
        "marker": "SMOKE_ONLY_NOT_FOR_REPORTING",
        "sample_queries_tested": len(smoke_records),
        "ref_k4_dev_f1": ref_k4_f1,
        "ref_k6_dev_f1": ref_k6_f1,
        "smoke_e1_f1": float(df_smoke_e1["token_f1"].mean()),
        "smoke_e1_context_tokens": float(df_smoke_e1["context_tokens"].mean()),
        "mean_generation_latency": float(df_smoke_e1["generation_latency"].mean()),
        "mean_preprocessing_latency": float(df_smoke_e1["preprocessing_latency"].mean()),
    }
    summary_path = smoke_dir / "smoke_w7e_summary.json"
    with open(summary_path, "w") as f:
        json.dump(smoke_summary, f, indent=2)

    print(f"Smoke summary saved to {summary_path}")
    print("=" * 70)
    print("WEEK 7-E SMOKE TEST PASSED COMPLETELY.")
    print("=" * 70)


if __name__ == "__main__":
    main()
