"""Semantic response-surface generation, resumable execution, and matrix construction for Week 7-B.

Enforces:
- Budgets: K = {2, 4, 5, 6, 8}.
- Strict R1 semantic routing: passages ranked by semantic score, top-k selected, rendered in BM25 order.
- Invariant assertions: exactly k FULL, exactly 10 - k COMPRESSED, 10 passages accounted for, no duplicates.
- Append-safe & resume-safe cache with primary key (question_id, k).
- Strict cache schema storing split, k, passage indices, metrics, context tokens, and checkpoint/config identifiers.
- Construction of wide fixed-k matrix (one row per query, 5 budgets).
- Robust handling of empty generations (F1=0.0, EM=0.0).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch

from src.week6.routing import (
    build_routed_sara_prompt,
    extract_routed_sara_embeddings,
    route_r1_semantic_topk,
    validate_routing_assignment,
)
from src.week6.evaluation import (
    compute_qa_answer_metrics,
    extract_generated_text,
)

logger = logging.getLogger(__name__)

ALLOWED_BUDGETS: List[int] = [2, 4, 5, 6, 8]


def build_semantic_routing_assignments(
    semantic_scores_list: Sequence[Sequence[float]],
    budgets: Sequence[int] = (2, 4, 5, 6, 8),
    n_passages: int = 10,
) -> Dict[int, List[List[int]]]:
    """Build semantic top-k routing assignments for each budget k in budgets.

    Args:
        semantic_scores_list: Length N list of length-10 float sequences.
        budgets: Sequence of integer budgets (default 2, 4, 5, 6, 8).
        n_passages: Total candidate passages per query (10).

    Returns:
        Dictionary mapping budget k -> list of N 0-indexed passage selection lists.
        Each selection list contains exactly k unique indices sorted in ascending order.
    """
    assignments_by_k: Dict[int, List[List[int]]] = {k: [] for k in budgets}

    for scores in semantic_scores_list:
        for k in budgets:
            assert k in ALLOWED_BUDGETS, f"Invalid budget k={k}; must be in {ALLOWED_BUDGETS}"
            sel = route_r1_semantic_topk(scores, k=k)
            validate_routing_assignment(sel, k=k, total_passages=n_passages)
            assignments_by_k[k].append(sel)

    return assignments_by_k


def load_existing_response_surface_cache(
    cache_path: Union[str, Path],
) -> Tuple[Dict[Tuple[str, int], Dict[str, Any]], Set[Tuple[str, int]]]:
    """Load existing JSONL cache and index completed pairs by (question_id, k).

    Raises ValueError if duplicate (question_id, k) entries are found.
    """
    p = Path(cache_path)
    cached_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
    completed_keys: Set[Tuple[str, int]] = set()

    if not p.exists():
        return cached_map, completed_keys

    with open(p, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                rec = json.loads(line)
                qid = str(rec["question_id"]).strip()
                k = int(rec["k"])
                key = (qid, k)

                if key in completed_keys:
                    raise ValueError(f"Integrity violation: duplicate cache entry for key (qid='{qid}', k={k}) at line {line_num}")

                completed_keys.add(key)
                cached_map[key] = rec

    return cached_map, completed_keys


def execute_resumable_semantic_response_surface(
    sara_model: Any,
    sara_tokenizer: Any,
    records: List[Dict[str, Any]],
    semantic_scores_list: List[List[float]],
    cached_embeddings: Dict[str, torch.Tensor],
    cache_path: Union[str, Path],
    split: str = "train",
    budgets: Sequence[int] = (2, 4, 5, 6, 8),
    device: str = "cuda",
    checkpoint_id: str = "sara_qasper_proj_lr5e4_seed42",
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Execute resumable generation for the (records x budgets) grid under semantic routing.

    Key features:
    - Primary cache key: (question_id, k).
    - Skips already completed (qid, k) pairs.
    - Appends newly generated records safely to disk immediately.
    - Preserves gold answers for accurate F1 metric computation.
    - Strictly records split, routing, passage indices, context tokens, metrics, latency, and config.
    """
    cache_file_path = Path(cache_path)
    cache_file_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load existing cached progress
    cached_map, completed_keys = load_existing_response_surface_cache(cache_file_path)
    logger.info("Found %d completed pairs in cache: %s", len(completed_keys), cache_file_path)

    # 2. Precompute routing assignments
    assignments_by_k = build_semantic_routing_assignments(
        semantic_scores_list=semantic_scores_list,
        budgets=budgets,
        n_passages=10,
    )

    pad_id = sara_tokenizer.pad_token_id if sara_tokenizer.pad_token_id is not None else sara_tokenizer.eos_token_id
    gen_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
    }

    # Open in append mode
    cache_file = open(cache_file_path, "a", encoding="utf-8")

    all_records: List[Dict[str, Any]] = []
    total_pairs = len(records) * len(budgets)
    n_generated = 0
    t_start = time.time()

    try:
        from tqdm import tqdm
        progress_bar = tqdm(total=total_pairs, desc=f"Response Surface [{split}]", initial=len(completed_keys))
    except ImportError:
        progress_bar = None

    try:
        for idx, item in enumerate(records):
            qid = str(item["question_id"]).strip()
            pid = str(item.get("paper_id", "")).strip()
            question_text = str(item.get("question", "")).strip()
            passages = item.get("passages", [])[:10]
            gold_answers = item.get("gold_answers", [])
            gold_reformatted = item.get("gold_answers_reformatted", gold_answers)

            for k in budgets:
                key = (qid, k)
                selected_indices = assignments_by_k[k][idx]
                validate_routing_assignment(selected_indices, k=k, total_passages=len(passages))

                if key in completed_keys:
                    cached_rec = cached_map[key]
                    all_records.append(cached_rec)
                    continue

                # Not cached: generate via SARA
                formatted_prompt, context_tokens, unselected_indices = build_routed_sara_prompt(
                    question=question_text,
                    passages=passages,
                    selected_indices=selected_indices,
                    tokenizer=sara_tokenizer,
                )

                inputs = sara_tokenizer(formatted_prompt, return_tensors="pt").to(device)
                qid_embeds = cached_embeddings[qid]
                comp_embeds = extract_routed_sara_embeddings(qid_embeds, selected_indices).to(
                    device=device, dtype=sara_model.dtype
                )

                t0 = time.time()
                with torch.no_grad():
                    outputs = sara_model.generate(
                        **inputs,
                        retrieval_embeds=comp_embeds,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        pad_token_id=pad_id,
                        eos_token_id=sara_tokenizer.eos_token_id,
                    )
                latency = time.time() - t0

                pred_texts = extract_generated_text(sara_tokenizer, outputs)
                prediction = pred_texts[0].strip() if pred_texts else ""

                # Evaluate metrics
                qa_metrics = compute_qa_answer_metrics(
                    prediction=prediction,
                    gold_answers=gold_answers,
                    gold_reformatted=gold_reformatted,
                )

                record_data = {
                    "split": split,
                    "question_id": qid,
                    "paper_id": pid,
                    "k": k,
                    "routing": "semantic_topk",
                    "selected_full_indices": [int(x) for x in selected_indices],
                    "compressed_indices": [int(x) for x in unselected_indices],
                    "prediction": prediction,
                    "gold_answers": gold_answers,
                    "gold_answers_reformatted": gold_reformatted,
                    "token_f1": float(qa_metrics["token_f1"]),
                    "exact_match": float(qa_metrics["exact_match"]),
                    "rouge_l": float(qa_metrics["rouge_l"]),
                    "context_tokens": int(context_tokens),
                    "latency": float(latency),
                    "checkpoint_id": checkpoint_id,
                    "generation_config": gen_config,
                }

                # Append to cache
                cache_file.write(json.dumps(record_data) + "\n")
                cache_file.flush()

                completed_keys.add(key)
                cached_map[key] = record_data
                all_records.append(record_data)
                n_generated += 1

                if progress_bar:
                    progress_bar.update(1)

    finally:
        cache_file.close()
        if progress_bar:
            progress_bar.close()

    elapsed = time.time() - t_start
    logger.info(
        "Response surface [%s]: %d completed in %.1fs (%d newly generated).",
        split, len(all_records), elapsed, n_generated,
    )
    return {
        "split": split,
        "n_total": len(all_records),
        "n_generated": n_generated,
        "cache_path": str(cache_file_path),
        "records": all_records,
    }


def build_semantic_fixed_k_matrix(
    results_jsonl_path: Union[str, Path],
    output_matrix_csv_path: Union[str, Path],
    budgets: Sequence[int] = (2, 4, 5, 6, 8),
) -> pd.DataFrame:
    """Transform flat JSONL results into wide per-query fixed-k matrix.

    Schema:
    question_id, paper_id, f1_k2, f1_k4, f1_k5, f1_k6, f1_k8, tokens_k2, tokens_k4, tokens_k5, tokens_k6, tokens_k8
    """
    p_in = Path(results_jsonl_path)
    p_out = Path(output_matrix_csv_path)
    p_out.parent.mkdir(parents=True, exist_ok=True)

    records_by_qid: Dict[str, Dict[str, Any]] = {}

    with open(p_in, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                qid = str(d["question_id"]).strip()
                k = int(d["k"])
                if qid not in records_by_qid:
                    records_by_qid[qid] = {
                        "question_id": qid,
                        "paper_id": str(d.get("paper_id", "")).strip(),
                        "budgets": {},
                    }
                records_by_qid[qid]["budgets"][k] = {
                    "f1": float(d.get("token_f1", 0.0)),
                    "tokens": int(d.get("context_tokens", 0)),
                }

    rows = []
    for qid, qdata in records_by_qid.items():
        # Assert query has all required budgets
        for k in budgets:
            if k not in qdata["budgets"]:
                raise ValueError(f"Incomplete matrix: query '{qid}' missing budget k={k}")

        row: Dict[str, Any] = {
            "question_id": qid,
            "paper_id": qdata["paper_id"],
        }
        for k in budgets:
            row[f"f1_k{k}"] = qdata["budgets"][k]["f1"]
            row[f"tokens_k{k}"] = qdata["budgets"][k]["tokens"]

        rows.append(row)

    df_matrix = pd.DataFrame(rows)
    df_matrix.to_csv(p_out, index=False)
    logger.info("Saved wide response matrix with %d queries to %s", len(df_matrix), p_out)
    return df_matrix
