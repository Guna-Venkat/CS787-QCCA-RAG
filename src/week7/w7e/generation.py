"""Resumable multi-fidelity SARA generation and invariant validation for W7-E.

Implements:
- Resumable generation execution for any multi-fidelity policy (E0, E1, E2, E3, E4, E1-X).
- Strict append-safe JSONL caching with composite key (question_id, policy_name, token_cap).
- Invariant reproduction checks for E0 (k=4 reference) and E2 (k=6 reference).
- Accurate end-to-end latency measurement (preprocessing, generation, total).
"""

import gc
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.week6.evaluation import compute_qa_answer_metrics, extract_generated_text
from src.week7.w7e.prompt_builder import (
    build_mixed_fidelity_sara_prompt,
    extract_mixed_fidelity_sara_embeddings,
)

logger = logging.getLogger(__name__)


def load_existing_generation_cache(
    cache_path: Union[str, Path],
) -> Tuple[Dict[Tuple[str, str, int], Dict[str, Any]], Set[Tuple[str, str, int]]]:
    """Load existing JSONL generation cache keyed by (question_id, policy_name, token_cap)."""
    p = Path(cache_path)
    cached_map: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    completed_keys: Set[Tuple[str, str, int]] = set()

    if not p.is_file():
        return cached_map, completed_keys

    with open(p, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                rec = json.loads(line)
                qid = str(rec["question_id"]).strip()
                pol = str(rec["policy_name"]).strip()
                cap = int(rec.get("token_cap", 0))
                key = (qid, pol, cap)
                completed_keys.add(key)
                cached_map[key] = rec

    return cached_map, completed_keys


def execute_resumable_multifidelity_generation(
    sara_model: Any,
    sara_tokenizer: Any,
    records: List[Dict[str, Any]],
    allocator_fn: Callable[[int], Tuple[List[int], List[int], List[int]]],
    medium_texts_by_qid: Dict[str, Dict[int, str]],
    cached_sfr_embeddings: Dict[str, torch.Tensor],
    cache_path: Union[str, Path],
    split: str = "dev",
    policy_name: str = "E1_4F_2M_4H",
    token_cap: int = 96,
    device: str = "cuda",
    checkpoint_id: str = "sara_qasper_proj_lr5e4_seed42",
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 0.0,
    hybrid_medium: bool = False,
) -> pd.DataFrame:
    """Execute resumable generation for a multi-fidelity policy on SARA.

    Key guarantees:
    - Skips already completed (qid, policy_name, token_cap) tuples.
    - Appends newly generated records immediately to disk.
    - Injects exactly SFR embeddings for compressed passages into SARA's retrieval_embeds.
    - Records detailed latency and token accounting.
    """
    cache_file_path = Path(cache_path)
    cache_file_path.parent.mkdir(parents=True, exist_ok=True)

    cached_map, completed_keys = load_existing_generation_cache(cache_file_path)
    logger.info("Loaded %d completed generation records from %s", len(completed_keys), cache_file_path)

    cache_file = open(cache_file_path, "a", encoding="utf-8")
    pad_id = sara_tokenizer.pad_token_id if sara_tokenizer.pad_token_id is not None else sara_tokenizer.eos_token_id

    # Auto-unpack if raw checkpoint container with 'embeddings' key was passed
    if isinstance(cached_sfr_embeddings, dict) and "embeddings" in cached_sfr_embeddings and isinstance(cached_sfr_embeddings["embeddings"], dict):
        cached_sfr_embeddings = cached_sfr_embeddings["embeddings"]

    gen_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "hybrid_medium": hybrid_medium,
    }

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    all_records: List[Dict[str, Any]] = []
    pbar = tqdm(records, desc=f"Generating {policy_name} (M{token_cap}{'-H' if hybrid_medium else ''})", unit="query")

    for item in pbar:
        qid = str(item["question_id"]).strip()
        pid = str(item.get("paper_id", "")).strip()
        question_text = str(item.get("question", "")).strip()
        passages = item.get("passages", [])[:10]
        gold_answers = item.get("gold_answers", [])
        gold_reformatted = item.get("gold_answers_reformatted", gold_answers)

        key = (qid, policy_name, token_cap)

        if key in completed_keys:
            cached_rec = cached_map[key]
            all_records.append(cached_rec)
            pbar.set_postfix({"f1": f"{cached_rec.get('token_f1', 0.0):.3f}", "status": "cached"})
            continue

        # Allocate full, medium, heavy indices
        if callable(allocator_fn):
            try:
                full_idx, med_idx, heavy_idx = allocator_fn(item)
            except TypeError:
                try:
                    full_idx, med_idx, heavy_idx = allocator_fn(qid)
                except TypeError:
                    full_idx, med_idx, heavy_idx = allocator_fn()
        else:
            full_idx, med_idx, heavy_idx = [], [], []

        # Get medium texts for this query
        q_med_map = medium_texts_by_qid.get(qid, {})

        t_pre_start = time.time()
        formatted_prompt, context_tokens, compressed_indices, token_breakdown = build_mixed_fidelity_sara_prompt(
            question=question_text,
            passages=passages,
            full_indices=full_idx,
            medium_indices=med_idx,
            heavy_indices=heavy_idx,
            medium_texts_map=q_med_map,
            tokenizer=sara_tokenizer,
            hybrid_medium=hybrid_medium,
        )
        t_pre = time.time() - t_pre_start

        # Tokenize and slice embeddings
        inputs = sara_tokenizer(formatted_prompt, return_tensors="pt").to(device)
        qid_embeds = cached_sfr_embeddings[qid]
        comp_embeds = extract_mixed_fidelity_sara_embeddings(qid_embeds, compressed_indices).to(
            device=device, dtype=sara_model.dtype
        )

        # SARA Generation
        t_gen_start = time.time()
        with torch.no_grad():
            outputs = sara_model.generate(
                **inputs,
                retrieval_embeds=comp_embeds,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=pad_id,
                eos_token_id=sara_tokenizer.eos_token_id,
            )
        t_gen = time.time() - t_gen_start
        total_latency = t_pre + t_gen

        pred_texts = extract_generated_text(sara_tokenizer, outputs)
        prediction = pred_texts[0].strip() if pred_texts else ""

        # Free temporary GPU memory
        del inputs, comp_embeds, outputs

        qa_metrics = compute_qa_answer_metrics(
            prediction=prediction,
            gold_answers=gold_answers,
            gold_reformatted=gold_reformatted,
        )

        record_data = {
            "split": split,
            "question_id": qid,
            "paper_id": pid,
            "policy_name": policy_name,
            "token_cap": token_cap,
            "hybrid_medium": hybrid_medium,
            "full_indices": [int(x) for x in full_idx],
            "medium_indices": [int(x) for x in med_idx],
            "heavy_indices": [int(x) for x in heavy_idx],
            "compressed_indices": [int(x) for x in compressed_indices],
            "prediction": prediction,
            "gold_answers": gold_answers,
            "token_f1": float(qa_metrics["token_f1"]),
            "exact_match": float(qa_metrics["exact_match"]),
            "rouge_l": float(qa_metrics["rouge_l"]),
            "context_tokens": int(context_tokens),
            "preprocessing_latency": float(t_pre),
            "generation_latency": float(t_gen),
            "total_latency": float(total_latency),
            "checkpoint_id": checkpoint_id,
            "generation_config": gen_config,
        }

        cache_file.write(json.dumps(record_data) + "\n")
        cache_file.flush()

        completed_keys.add(key)
        cached_map[key] = record_data
        all_records.append(record_data)

        pbar.set_postfix({"f1": f"{record_data['token_f1']:.3f}", "tok": record_data['context_tokens']})
        if len(all_records) % 20 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    cache_file.close()
    return pd.DataFrame(all_records)


def assert_pipeline_invariants(
    df_e0_eval: pd.DataFrame,
    df_e2_eval: pd.DataFrame,
    ref_k4_f1: float = 0.3909,
    ref_k6_f1: float = 0.3957,
    tolerance: float = 0.015,
) -> None:
    """Verify Section 47 pipeline invariants:
    - Invariant 1: E0 (4F+0M+6H) must reproduce canonical semantic k4 F1 within tolerance.
    - Invariant 2: E2 (6F+0M+4H) must reproduce canonical semantic k6 F1 within tolerance.
    """
    mean_f1_e0 = float(df_e0_eval["token_f1"].mean())
    mean_f1_e2 = float(df_e2_eval["token_f1"].mean())

    diff_e0 = abs(mean_f1_e0 - ref_k4_f1)
    diff_e2 = abs(mean_f1_e2 - ref_k6_f1)

    logger.info("Invariant 1 Check: E0 F1 = %.4f (Reference k4 = %.4f, Diff = %.4f)", mean_f1_e0, ref_k4_f1, diff_e0)
    logger.info("Invariant 2 Check: E2 F1 = %.4f (Reference k6 = %.4f, Diff = %.4f)", mean_f1_e2, ref_k6_f1, diff_e2)

    if diff_e0 > tolerance:
        raise AssertionError(
            f"PIPELINE INVARIANT 1 FAILED: E0 F1={mean_f1_e0:.4f} deviates from reference k4={ref_k4_f1:.4f} by {diff_e0:.4f} > {tolerance}!"
        )
    if diff_e2 > tolerance:
        raise AssertionError(
            f"PIPELINE INVARIANT 2 FAILED: E2 F1={mean_f1_e2:.4f} deviates from reference k6={ref_k6_f1:.4f} by {diff_e2:.4f} > {tolerance}!"
        )

    logger.info("ALL PIPELINE INVARIANTS SUCCESSFULLY VERIFIED.")
