"""W8-A: Counterfactual Single-Passage Promotion Response Surface.

Implements:
- Construction of counterfactual single-promotion contexts: C_i = 4F + 1F_i + 5H
- Exactly one candidate passage i in {5,6,7,8,9,10} promoted to FULL text
- Resumable append-safe JSONL caching with composite key (question_id, candidate_passage_index)
- Marginal promotion utility calculation: Delta_i = F_i - F_0
- Categorization into HELP (> +0.01), NEUTRAL (|Delta| <= 0.01), HURT (< -0.01)
- Strict prevention of gold answer leakage into model input features
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.week6.evaluation import compute_qa_answer_metrics, extract_generated_text
from src.week7.w7e.fidelity_allocator import get_semantic_rank_order
from src.week7.w7e.prompt_builder import (
    build_mixed_fidelity_sara_prompt,
    extract_mixed_fidelity_sara_embeddings,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def construct_single_promotion_allocation(
    semantic_order: Sequence[int],
    candidate_rank: int,  # 1-indexed, e.g. 5, 6, 7, 8, 9, 10
) -> Tuple[List[int], List[int], List[int], int]:
    """Construct (full_indices, medium_indices, heavy_indices, promoted_passage_idx).

    Base: semantic ranks 1..4 are FULL (ranks 5..10 are HEAVY).
    Promotion: candidate_rank (5..10) is also promoted to FULL.
    Result: 5 FULL, 0 MEDIUM, 5 HEAVY.
    """
    assert 5 <= candidate_rank <= 10, f"Candidate rank must be in 5..10, got {candidate_rank}"
    base_full = list(semantic_order[:4])
    promoted_idx = semantic_order[candidate_rank - 1]

    full_indices = sorted(base_full + [promoted_idx])
    medium_indices: List[int] = []
    heavy_indices = [idx for idx in semantic_order if idx not in full_indices]

    assert len(full_indices) == 5, f"Expected 5 full passages, got {len(full_indices)}"
    assert len(heavy_indices) == 5, f"Expected 5 heavy passages, got {len(heavy_indices)}"
    return full_indices, medium_indices, heavy_indices, promoted_idx


def compute_utility_label(delta_f1: float, epsilon: float = 0.01) -> str:
    """Classify marginal promotion utility into HELP, NEUTRAL, or HURT."""
    if delta_f1 > epsilon:
        return "HELP"
    elif delta_f1 < -epsilon:
        return "HURT"
    else:
        return "NEUTRAL"


def load_existing_promotion_cache(
    cache_path: Path | str,
) -> Tuple[Dict[Tuple[str, int], Dict[str, Any]], Set[Tuple[str, int]]]:
    """Load existing promotion cache keyed by (question_id, candidate_passage_index)."""
    p = Path(cache_path)
    completed_keys: Set[Tuple[str, int]] = set()
    cached_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
    if p.is_file():
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    k = (str(rec["question_id"]), int(rec.get("candidate_passage_index", rec.get("candidate_index", 0))))
                    completed_keys.add(k)
                    cached_map[k] = rec
    return cached_map, completed_keys


# Aliases
generate_resumable_promotion_surface = lambda *args, **kwargs: run_single_promotion_surface(*args, **kwargs)


def run_single_promotion_surface(
    sara_model: Any,
    sara_tokenizer: Any,
    records: List[Dict[str, Any]],
    semantic_scores_by_qid: Dict[str, np.ndarray],
    cached_sfr_embeddings: Dict[str, torch.Tensor],
    cache_path: Path,
    candidate_ranks: Sequence[int] = (5, 6, 7, 8, 9, 10),
    device: str = "cuda",
    max_new_tokens: int = 128,
    smoke_limit: Optional[int] = None,
) -> pd.DataFrame:
    """Execute resumable single-passage promotion response surface.

    Key guarantees:
    - Skips already completed (question_id, candidate_passage_index) records.
    - Appends immediately to disk upon each evaluation.
    - Accurately tracks base F1 (k=4) and promoted F1 (C_i).
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    completed_keys: Set[Tuple[str, int]] = set()
    existing_rows: List[Dict[str, Any]] = []

    if cache_path.is_file():
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    k = (str(rec["question_id"]), int(rec["candidate_passage_index"]))
                    completed_keys.add(k)
                    existing_rows.append(rec)
        logger.info("Loaded %d completed promotion records from %s", len(completed_keys), cache_path)

    cache_file = open(cache_path, "a", encoding="utf-8")
    pad_id = sara_tokenizer.pad_token_id if sara_tokenizer.pad_token_id is not None else sara_tokenizer.eos_token_id

    items = records[:smoke_limit] if smoke_limit is not None else records
    pbar = tqdm(items, desc="W8-A Promotion Surface", unit="query")

    for item in pbar:
        qid = str(item["question_id"])
        pid = str(item.get("paper_id", ""))
        question_text = str(item["question"]).strip()
        passages = item.get("passages", [])[:10]
        gold_answers = item.get("gold_answers", [])
        gold_reformatted = item.get("gold_answers_reformatted", gold_answers)
        qid_embeds = cached_sfr_embeddings[qid]

        sem_scores = semantic_scores_by_qid[qid]
        semantic_order = get_semantic_rank_order(sem_scores, n_passages=len(passages))

        # 1. Base Generation (k=4: 4F + 6H)
        base_full = list(semantic_order[:4])
        base_heavy = list(semantic_order[4:])
        base_prompt, base_tokens, base_comp_idx, _ = build_mixed_fidelity_sara_prompt(
            question=question_text,
            passages=passages,
            full_indices=base_full,
            medium_indices=[],
            heavy_indices=base_heavy,
            medium_texts_map={},
            tokenizer=sara_tokenizer,
            hybrid_medium=False,
        )

        base_inputs = sara_tokenizer(base_prompt, return_tensors="pt").to(device)
        base_comp_embeds = extract_mixed_fidelity_sara_embeddings(qid_embeds, base_comp_idx).to(
            device=device, dtype=sara_model.dtype
        )

        with torch.no_grad():
            base_out = sara_model.generate(
                **base_inputs,
                retrieval_embeds=base_comp_embeds,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=sara_tokenizer.eos_token_id,
            )
        base_pred_texts = extract_generated_text(sara_tokenizer, base_out)
        base_prediction = base_pred_texts[0].strip() if base_pred_texts else ""
        del base_inputs, base_comp_embeds, base_out

        base_metrics = compute_qa_answer_metrics(
            prediction=base_prediction,
            gold_answers=gold_answers,
            gold_reformatted=gold_reformatted,
        )
        base_f1 = float(base_metrics["token_f1"])

        # 2. Iterate through each candidate rank in {5,6,7,8,9,10}
        for rank in candidate_ranks:
            if rank > len(semantic_order):
                continue

            full_idx, med_idx, heavy_idx, promoted_idx = construct_single_promotion_allocation(
                semantic_order=semantic_order,
                candidate_rank=rank,
            )

            key = (qid, promoted_idx)
            if key in completed_keys:
                continue

            t0 = time.time()
            prom_prompt, prom_tokens, prom_comp_idx, _ = build_mixed_fidelity_sara_prompt(
                question=question_text,
                passages=passages,
                full_indices=full_idx,
                medium_indices=med_idx,
                heavy_indices=heavy_idx,
                medium_texts_map={},
                tokenizer=sara_tokenizer,
                hybrid_medium=False,
            )

            prom_inputs = sara_tokenizer(prom_prompt, return_tensors="pt").to(device)
            prom_comp_embeds = extract_mixed_fidelity_sara_embeddings(qid_embeds, prom_comp_idx).to(
                device=device, dtype=sara_model.dtype
            )

            with torch.no_grad():
                prom_out = sara_model.generate(
                    **prom_inputs,
                    retrieval_embeds=prom_comp_embeds,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_id,
                    eos_token_id=sara_tokenizer.eos_token_id,
                )
            latency = time.time() - t0

            prom_pred_texts = extract_generated_text(sara_tokenizer, prom_out)
            prom_prediction = prom_pred_texts[0].strip() if prom_pred_texts else ""
            del prom_inputs, prom_comp_embeds, prom_out

            prom_metrics = compute_qa_answer_metrics(
                prediction=prom_prediction,
                gold_answers=gold_answers,
                gold_reformatted=gold_reformatted,
            )
            prom_f1 = float(prom_metrics["token_f1"])
            delta_f1 = prom_f1 - base_f1
            utility_label = compute_utility_label(delta_f1)

            cand_p = passages[promoted_idx]
            cand_text = cand_p.get("text", "").strip()

            record = {
                "question_id": qid,
                "paper_id": pid,
                "question": question_text,
                "candidate_passage_index": promoted_idx,
                "semantic_rank": rank,
                "bm25_rank": promoted_idx + 1,
                "candidate_text": cand_text,
                "base_prediction": base_prediction,
                "promoted_prediction": prom_prediction,
                "gold_answers": gold_answers,
                "base_f1": base_f1,
                "promoted_f1": prom_f1,
                "delta_f1": delta_f1,
                "utility_label": utility_label,
                "base_context_tokens": base_tokens,
                "promoted_context_tokens": prom_tokens,
                "semantic_score": float(sem_scores[promoted_idx]),
                "bm25_score": float(cand_p.get("score", 0.0)),
                "passage_length": len(cand_text.split()),
                "compressed_embedding_index": promoted_idx,
                "promotion_latency": latency,
            }

            cache_file.write(json.dumps(record) + "\n")
            cache_file.flush()
            completed_keys.add(key)
            existing_rows.append(record)

    cache_file.close()
    return pd.DataFrame(existing_rows)
