#!/usr/bin/env python3
"""Execute Multi-Vector MEDIUM (M2) evaluation on the 342 INTERNAL-VAL queries.

Fidelity Architecture:
- Ranks 1-4: FULL natural-language text in ## Context (0 compressed vectors)
- Ranks 5-6: MEDIUM (2 vectors each: whole SFR vector + semantic-focused chunk SFR vector)
- Ranks 7-10: HEAVY (1 vector each: whole SFR vector)
Total: exactly 4 full text passages in context, exactly 8 <COMPRESS> tokens in prompt,
and exactly 8 4096-dim vectors in SARA's retrieval_embeds.

Methodological Principle:
Validate on INTERNAL-VAL first as an exploratory method before running DEV.
"""

import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SARA_ROOT = PROJECT_ROOT / "Baselines/SARA-main"
import src
if hasattr(src, "__path__") and str(SARA_ROOT / "src") not in src.__path__:
    src.__path__.append(str(SARA_ROOT / "src"))

from src.model.loader import load_sara_for_eval
from src.week6.evaluation import compute_qa_answer_metrics, extract_generated_text
from src.week6.models import PretrainedSemanticScorer
from src.week7.w7e.fidelity_allocator import allocate_e1_primary, get_semantic_rank_order
from src.week7.w7e.m2_multi_vector import (
    assert_m2_row_invariants,
    build_m2_sara_prompt,
    extract_m2_sara_embeddings,
    select_focused_chunk,
    split_passage_into_sentence_halves,
)

logger = logging.getLogger(__name__)


def precompute_focused_sfr_embeddings(
    items: List[Dict[str, Any]],
    semantic_scores_by_qid: Dict[str, np.ndarray],
    semantic_scorer: PretrainedSemanticScorer,
    cache_path: Path,
    device: str = "cuda",
) -> Dict[str, Dict[int, torch.Tensor]]:
    """Precompute and cache SFR embeddings for the focused half of candidate ranks 5-6."""
    if cache_path.is_file():
        print(f"Loading existing focused SFR embeddings from {cache_path}...")
        loaded = torch.load(cache_path, map_location="cpu")
        return loaded

    print(f"Precomputing focused chunk SFR embeddings for {len(items)} items...")
    tasks: List[Tuple[str, int, str]] = []  # (qid, rank_idx, selected_chunk_text)

    for item in items:
        qid = str(item["question_id"])
        query = item["question"]
        passages = item.get("passages", [])[:10]
        order = get_semantic_rank_order(semantic_scores_by_qid[qid], n_passages=len(passages))
        # Candidate ranks 5 and 6
        for r_idx in order[4:6]:
            p_text = passages[r_idx].get("text", "")
            c1, c2 = split_passage_into_sentence_halves(p_text)
            best_chunk = select_focused_chunk(query, c1, c2, semantic_scorer)
            tasks.append((qid, r_idx, best_chunk if best_chunk else p_text))

    print(f"Extracted {len(tasks)} focused chunks. Encoding with SFR-Embedding-Mistral...")
    target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    sfr_model = SentenceTransformer(
        "Salesforce/SFR-Embedding-Mistral",
        device=device,
        model_kwargs={"torch_dtype": target_dtype},
    )

    texts_to_encode = [t[2] for t in tasks]
    batch_size = 32
    all_embeds = sfr_model.encode(
        texts_to_encode,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        device=device,
    ).cpu()

    # Clean up SFR model from GPU
    del sfr_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("SFR model unloaded from GPU.")

    focused_dict: Dict[str, Dict[int, torch.Tensor]] = {}
    for (qid, r_idx, _), emb in zip(tasks, all_embeds):
        if qid not in focused_dict:
            focused_dict[qid] = {}
        focused_dict[qid][r_idx] = emb

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(focused_dict, cache_path)
    print(f"Saved focused SFR embeddings to {cache_path}.")
    return focused_dict


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Running Multi-Vector MEDIUM (M2) on INTERNAL-VAL ({device}) ===")

    # 1. Load manifest and items
    manifest_path = PROJECT_ROOT / "results/week5/content_allocator/w5b/splits/internal_val_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    val_qids = set(manifest["question_id"].astype(str))

    retrieval_path = PROJECT_ROOT / "results/week5/train_response_surface/train_retrieval.jsonl"
    with open(retrieval_path, "r", encoding="utf-8") as f:
        val_items = [json.loads(line) for line in f if str(json.loads(line)["question_id"]) in val_qids]

    print(f"Loaded {len(val_items)} internal validation queries.")

    # Canonical baselines
    resp_matrix_path = PROJECT_ROOT / "results/week7/w7b_semantic_budget/response_surface/train_semantic_fixed_k_matrix.csv"
    resp_df = pd.read_csv(resp_matrix_path)
    val_resp_df = resp_df[resp_df["question_id"].astype(str).isin(val_qids)]
    ref_k4_f1 = float(val_resp_df["f1_k4"].mean())
    ref_k4_tok = float(val_resp_df["tokens_k4"].mean())
    ref_k6_f1 = float(val_resp_df["f1_k6"].mean())
    ref_k6_tok = float(val_resp_df["tokens_k6"].mean())

    print(f"Reference INTERNAL-VAL k=4 (E0): F1 = {ref_k4_f1:.4f}, Tokens = {ref_k4_tok:.1f}")
    print(f"Reference INTERNAL-VAL k=6 (E2): F1 = {ref_k6_f1:.4f}, Tokens = {ref_k6_tok:.1f}")

    # 2. Precompute MiniLM semantic scores
    print("Precomputing semantic scores with MiniLM...")
    sem_scorer = PretrainedSemanticScorer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    val_semantic_scores = {}
    for item in val_items:
        qid = str(item["question_id"])
        query = item["question"]
        p_texts = [p.get("text", "") for p in item["passages"][:10]]
        val_semantic_scores[qid] = sem_scorer.score_query_passages(query, p_texts).detach().cpu().numpy()

    # 3. Precompute focused chunk SFR embeddings
    focused_cache_path = PROJECT_ROOT / "results/week7/w7e_multifidelity/train/m2_internal_val_focused_sfr.pt"
    focused_sfr_dict = precompute_focused_sfr_embeddings(
        items=val_items,
        semantic_scores_by_qid=val_semantic_scores,
        semantic_scorer=sem_scorer,
        cache_path=focused_cache_path,
        device=device,
    )

    # 4. Load whole SFR embeddings
    train_embeds_path = PROJECT_ROOT / "results/week5/train_response_surface/train_sfr_embeddings.pt"
    sfr_raw = torch.load(train_embeds_path, map_location="cpu")
    whole_sfr_dict = sfr_raw["embeddings"] if isinstance(sfr_raw, dict) and "embeddings" in sfr_raw else sfr_raw

    # 5. Load SARA Generator
    print("Loading SARA model for M2 generation...")
    sara_ckpt_path = PROJECT_ROOT / "Baselines/SARA-main/checkpoints/finetune/sara_qasper_proj_lr5e4_seed42"
    sara_model, sara_tokenizer = load_sara_for_eval(str(sara_ckpt_path), device=device)
    pad_id = sara_tokenizer.pad_token_id if sara_tokenizer.pad_token_id is not None else sara_tokenizer.eos_token_id

    # 6. Execute M2 Generation
    output_dir = PROJECT_ROOT / "results/week7/w7e_multifidelity/train"
    output_dir.mkdir(parents=True, exist_ok=True)
    m2_gen_cache = output_dir / "m2_internal_val_generation.jsonl"

    cached_records: Dict[str, Dict[str, Any]] = {}
    if m2_gen_cache.is_file():
        with open(m2_gen_cache, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    cached_records[str(rec["question_id"])] = rec
        print(f"Loaded {len(cached_records)} completed records from cache.")

    cache_file = open(m2_gen_cache, "a", encoding="utf-8")
    all_results: List[Dict[str, Any]] = []

    pbar = tqdm(val_items, desc="Generating M2 (2-vector latent)", unit="query")
    for item in pbar:
        qid = str(item["question_id"])
        if qid in cached_records:
            rec = cached_records[qid]
            all_results.append(rec)
            pbar.set_postfix({"f1": f"{rec.get('token_f1', 0.0):.3f}", "status": "cached"})
            continue

        query = str(item["question"]).strip()
        passages = item.get("passages", [])[:10]
        gold_answers = item.get("gold_answers", [])
        gold_reformatted = item.get("gold_answers_reformatted", gold_answers)

        # Allocate 4F + 2M + 4H
        full_idx, med_idx, heavy_idx = allocate_e1_primary(
            len(passages), semantic_scores=val_semantic_scores[qid]
        )

        t_pre_start = time.time()
        formatted_prompt, context_tokens, slot_mapping, token_breakdown = build_m2_sara_prompt(
            question=query,
            passages=passages,
            full_indices=full_idx,
            medium_indices=med_idx,
            heavy_indices=heavy_idx,
            tokenizer=sara_tokenizer,
        )
        t_pre = time.time() - t_pre_start

        # Tokenize and assemble embeddings
        inputs = sara_tokenizer(formatted_prompt, return_tensors="pt").to(device)
        q_whole_embeds = whole_sfr_dict[qid]
        q_focused_dict = focused_sfr_dict.get(qid, {})
        m2_embeds = extract_m2_sara_embeddings(q_whole_embeds, q_focused_dict, slot_mapping).to(
            device=device, dtype=sara_model.dtype
        )

        # Generate
        t_gen_start = time.time()
        with torch.no_grad():
            outputs = sara_model.generate(
                **inputs,
                retrieval_embeds=m2_embeds,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=sara_tokenizer.eos_token_id,
            )
        t_gen = time.time() - t_gen_start

        pred_texts = extract_generated_text(sara_tokenizer, outputs)
        prediction = pred_texts[0].strip() if pred_texts else ""

        del inputs, m2_embeds, outputs

        qa_metrics = compute_qa_answer_metrics(
            prediction=prediction,
            gold_answers=gold_answers,
            gold_reformatted=gold_reformatted,
        )

        record = {
            "question_id": qid,
            "policy_name": "E1_M2_4F_2M_4H_latent2v",
            "prediction": prediction,
            "token_f1": qa_metrics["token_f1"],
            "exact_match": qa_metrics["exact_match"],
            "rouge_l_f1": qa_metrics.get("rouge_l", qa_metrics.get("rouge_l_f1", 0.0)),
            "context_tokens": context_tokens,
            "latency_pre_sec": t_pre,
            "latency_gen_sec": t_gen,
            "latency_total_sec": t_pre + t_gen,
            "prompt": formatted_prompt,
            "token_breakdown": token_breakdown,
        }

        cache_file.write(json.dumps(record) + "\n")
        cache_file.flush()
        all_results.append(record)
        pbar.set_postfix({"f1": f"{qa_metrics['token_f1']:.3f}"})

    cache_file.close()
    df_m2 = pd.DataFrame(all_results)

    # 7. Invariant Verification
    print("\nVerifying M2 representation invariants on all 342 rows...")
    for _, row in df_m2.iterrows():
        assert_m2_row_invariants(
            row,
            n_full_expected=4,
            n_medium_expected=2,
            n_heavy_expected=4,
            n_compress_tokens_expected=8,
        )
    print("ALL 342 INTERNAL-VAL ROWS PASSED M2 INVARIANT VERIFICATION!")

    # 8. Compute Aggregate Comparison
    m2_f1 = float(df_m2["token_f1"].mean())
    m2_tokens = float(df_m2["context_tokens"].mean())

    delta_k4 = m2_f1 - ref_k4_f1
    delta_k6 = m2_f1 - ref_k6_f1
    delta_tok_k4 = m2_tokens - ref_k4_tok

    print("\n=======================================================")
    print("INTERNAL-VAL RESULTS: M2 (2-VECTOR LATENT) BENCHMARK")
    print("=======================================================")
    print(f"E0 (k=4 baseline)  : F1 = {ref_k4_f1:.4f}, Tokens = {ref_k4_tok:.1f}")
    print(f"M2 (2-vector latent): F1 = {m2_f1:.4f}, Tokens = {m2_tokens:.1f}")
    print(f"E2 (k=6 baseline)  : F1 = {ref_k6_f1:.4f}, Tokens = {ref_k6_tok:.1f}")
    print("-------------------------------------------------------")
    print(f"ΔF1 vs k=4         : {delta_k4:+.4f} (Context Tokens: {delta_tok_k4:+.1f})")
    print(f"ΔF1 vs k=6         : {delta_k6:+.4f}")
    print("=======================================================\n")

    tables_dir = PROJECT_ROOT / "results/week7/w7e_multifidelity/tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    results_table = [
        {
            "Method": "E0 k4 (4F+6H)",
            "F1": round(ref_k4_f1, 4),
            "Tokens / Placeholders": f"{ref_k4_tok:.1f} tok / 6 vectors",
            "ΔF1 vs k4": 0.0,
            "ΔF1 vs k6": round(ref_k4_f1 - ref_k6_f1, 4),
            "Type": "Baseline",
        },
        {
            "Method": "M2 2-vector (4F+2(2V)+4H)",
            "F1": round(m2_f1, 4),
            "Tokens / Placeholders": f"{m2_tokens:.1f} tok / 8 vectors",
            "ΔF1 vs k4": round(delta_k4, 4),
            "ΔF1 vs k6": round(delta_k6, 4),
            "Type": "Multi-Vector Latent",
        },
        {
            "Method": "E2 k6 (6F+4H)",
            "F1": round(ref_k6_f1, 4),
            "Tokens / Placeholders": f"{ref_k6_tok:.1f} tok / 4 vectors",
            "ΔF1 vs k4": round(ref_k6_f1 - ref_k4_f1, 4),
            "ΔF1 vs k6": 0.0,
            "Type": "Baseline",
        },
    ]

    df_summary = pd.DataFrame(results_table)
    df_summary.to_csv(tables_dir / "w7e_m2_internal_val_results.csv", index=False)
    print("Summary Table:")
    print(df_summary.to_string(index=False))

    if delta_k4 > 0.0:
        print(f"\n[DECISION]: POSITIVE SIGNAL! M2 beats k4 by {delta_k4:+.4f} F1 on INTERNAL-VAL.")
    else:
        print(f"\n[DECISION]: NO SIGNAL (delta={delta_k4:+.4f}). M2 does not improve over k4 on INTERNAL-VAL.")

if __name__ == "__main__":
    main()
