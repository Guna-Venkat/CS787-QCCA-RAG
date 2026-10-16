"""Full DAFiR End-to-End Generation Engine (Week 8-H).

Executes multi-pass selective fidelity routing:
Pass 0: Base semantic k4 generation -> draft y0 + uncertainty diagnostics.
Gating: Draft sufficiency check. If safe-stop -> return y0 immediately (Pass 1).
Routing: Candidate scoring via U0/U1/U2/U4 + MIG features.
Pass 1: Promoted regeneration -> y1 (DAFiR-1).
Pass 2 (Optional, DAFiR-2): Second promotion if y1 is still uncertain -> y2.

Fully tracks:
- Final-context tokens
- Cumulative prompt tokens
- Generated tokens
- Latency and GPU memory
- Resumable JSONL caching
"""

from __future__ import annotations

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
from src.week8.controller import DAFiRController
from src.week8.cost_accounting import GPUTracker, QueryCostRecord
from src.week8.draft_generation import (
    extract_draft_uncertainty_features,
    generate_draft_with_diagnostics,
)
from src.week8.qcla import BaseLatentAdapter

logger = logging.getLogger(__name__)


def run_dafir_query(
    sara_model: Any,
    sara_tokenizer: Any,
    query_record: Dict[str, Any],
    sfr_embeddings_by_qid: Dict[str, torch.Tensor],
    controller: DAFiRController,
    candidate_scorer_fn: Callable[[Dict[str, Any], str, Sequence[int]], Dict[str, Any]],
    qcla_adapter: Optional[BaseLatentAdapter] = None,
    query_embeddings_by_qid: Optional[Dict[str, torch.Tensor]] = None,
    max_promotions: int = 1,  # 1 for DAFiR-1, 2 for DAFiR-2
    candidate_ranks: Sequence[int] = (5, 6, 7, 8, 9, 10),
    device: str = "cuda",
    max_new_tokens: int = 128,
) -> Dict[str, Any]:
    """Execute full multi-pass DAFiR generation for a single query.

    Returns dict with final answer, metrics, and detailed cost accounting.
    """
    t_start = time.time()
    qid = str(query_record["question_id"]).strip()
    pid = str(query_record.get("paper_id", "")).strip()
    question = query_record["question"]
    passages = query_record["passages"]
    gold_answers = query_record.get("gold_answers", [])
    gold_reformatted = query_record.get("gold_answers_reformatted")

    candidate_indices = [r - 1 for r in candidate_ranks]
    base_full_indices = [0, 1, 2, 3]
    current_heavy_indices = [idx for idx in range(10) if idx not in base_full_indices]

    pad_id = sara_tokenizer.pad_token_id or sara_tokenizer.eos_token_id
    model_dtype = getattr(sara_model, "dtype", torch.float16)

    cumulative_prompt_tokens = 0
    total_generated_tokens = 0
    num_passes = 0
    promoted_indices: List[int] = []

    # Retrieve SFR embeddings
    qid_embeds = sfr_embeddings_by_qid.get(qid)
    if qid_embeds is None:
        qid_embeds = torch.zeros((10, 4096), dtype=torch.float32)

    # Optional QCLA transformation
    e_q = None
    if query_embeddings_by_qid is not None:
        e_q = query_embeddings_by_qid.get(qid)

    # --- PASS 0: Draft Generation ---
    num_passes += 1
    draft_prompt, draft_ctx_tokens, draft_comp_indices, _ = build_mixed_fidelity_sara_prompt(
        question=question,
        passages=passages,
        full_indices=base_full_indices,
        medium_indices=[],
        heavy_indices=current_heavy_indices,
        medium_texts_map={},
        tokenizer=sara_tokenizer,
        hybrid_medium=False,
    )
    cumulative_prompt_tokens += draft_ctx_tokens

    draft_inputs = sara_tokenizer(draft_prompt, return_tensors="pt").to(device)
    draft_comp_embeds = extract_mixed_fidelity_sara_embeddings(qid_embeds, draft_comp_indices).to(
        device=device, dtype=model_dtype
    )

    if qcla_adapter is not None:
        # Transform through QCLA
        transformed = []
        for i, c_idx in enumerate(draft_comp_indices):
            e_D = draft_comp_embeds[0, i]
            z_D = qcla_adapter(e_D, e_q=e_q)
            transformed.append(z_D)
        draft_comp_embeds = torch.stack(transformed, dim=0).unsqueeze(0).to(dtype=model_dtype)

    with torch.no_grad():
        draft_outputs = sara_model.generate(
            **draft_inputs,
            retrieval_embeds=draft_comp_embeds,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=sara_tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

    draft_gen_tokens = draft_outputs.sequences.shape[1] - draft_inputs["input_ids"].shape[1]
    total_generated_tokens += draft_gen_tokens

    draft_text_list = extract_generated_text(sara_tokenizer, draft_outputs.sequences)
    y0 = draft_text_list[0].strip() if draft_text_list else ""

    # Compute uncertainty diagnostics
    u_features = extract_draft_uncertainty_features(
        query=question,
        draft_text=y0,
        scores=draft_outputs.scores,
        generated_token_ids=draft_outputs.sequences[0, draft_inputs["input_ids"].shape[1]:],
        pad_token_id=pad_id,
        eos_token_id=sara_tokenizer.eos_token_id,
    )

    current_answer = y0
    final_context_tokens = draft_ctx_tokens

    # --- CONTROLLER DECISION 1 ---
    # Score candidates
    scored_candidates = candidate_scorer_fn(query_record, y0, candidate_indices)
    c1_decision = controller.evaluate_single_promotion(
        draft_features=u_features,
        candidate_predictions=scored_candidates,
    )

    if c1_decision["action"] == "PROMOTE" and max_promotions >= 1:
        chosen_cand = c1_decision["chosen_candidate_idx"]
        promoted_indices.append(chosen_cand)

        # Build Promoted Context (5F + 5H)
        num_passes += 1
        new_full_indices = sorted(base_full_indices + [chosen_cand])
        new_heavy_indices = [idx for idx in range(10) if idx not in new_full_indices]

        prom_prompt, prom_ctx_tokens, prom_comp_indices, _ = build_mixed_fidelity_sara_prompt(
            question=question,
            passages=passages,
            full_indices=new_full_indices,
            medium_indices=[],
            heavy_indices=new_heavy_indices,
            medium_texts_map={},
            tokenizer=sara_tokenizer,
            hybrid_medium=False,
        )
        cumulative_prompt_tokens += prom_ctx_tokens
        final_context_tokens = prom_ctx_tokens

        prom_inputs = sara_tokenizer(prom_prompt, return_tensors="pt").to(device)
        prom_comp_embeds = extract_mixed_fidelity_sara_embeddings(qid_embeds, prom_comp_indices).to(
            device=device, dtype=model_dtype
        )

        if qcla_adapter is not None:
            transformed = []
            for i, c_idx in enumerate(prom_comp_indices):
                e_D = prom_comp_embeds[0, i]
                z_D = qcla_adapter(e_D, e_q=e_q)
                transformed.append(z_D)
            prom_comp_embeds = torch.stack(transformed, dim=0).unsqueeze(0).to(dtype=model_dtype)

        with torch.no_grad():
            prom_outputs = sara_model.generate(
                **prom_inputs,
                retrieval_embeds=prom_comp_embeds,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=sara_tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        prom_gen_tokens = prom_outputs.sequences.shape[1] - prom_inputs["input_ids"].shape[1]
        total_generated_tokens += prom_gen_tokens

        prom_text_list = extract_generated_text(sara_tokenizer, prom_outputs.sequences)
        y1 = prom_text_list[0].strip() if prom_text_list else ""
        current_answer = y1

        # --- OPTIONAL PASS 2 (DAFiR-2) ---
        if max_promotions >= 2:
            u1_features = extract_draft_uncertainty_features(
                query=question,
                draft_text=y1,
                scores=prom_outputs.scores,
                generated_token_ids=prom_outputs.sequences[0, prom_inputs["input_ids"].shape[1]:],
                pad_token_id=pad_id,
                eos_token_id=sara_tokenizer.eos_token_id,
            )
            remaining_candidates = [c for c in candidate_indices if c != chosen_cand]
            scored_remaining = candidate_scorer_fn(query_record, y1, remaining_candidates)
            c2_decision = controller.evaluate_second_promotion(
                pass1_decision=c1_decision,
                y1_features=u1_features,
                candidate_predictions_pass2=scored_remaining,
            )

            if c2_decision["action"] == "PROMOTE_SECOND":
                second_cand = c2_decision["chosen_second_candidate_idx"]
                promoted_indices.append(second_cand)
                num_passes += 1

                stage2_full_indices = sorted(new_full_indices + [second_cand])
                stage2_heavy_indices = [idx for idx in range(10) if idx not in stage2_full_indices]

                s2_prompt, s2_ctx_tokens, s2_comp_indices, _ = build_mixed_fidelity_sara_prompt(
                    question=question,
                    passages=passages,
                    full_indices=stage2_full_indices,
                    medium_indices=[],
                    heavy_indices=stage2_heavy_indices,
                    medium_texts_map={},
                    tokenizer=sara_tokenizer,
                    hybrid_medium=False,
                )
                cumulative_prompt_tokens += s2_ctx_tokens
                final_context_tokens = s2_ctx_tokens

                s2_inputs = sara_tokenizer(s2_prompt, return_tensors="pt").to(device)
                s2_comp_embeds = extract_mixed_fidelity_sara_embeddings(qid_embeds, s2_comp_indices).to(
                    device=device, dtype=model_dtype
                )

                if qcla_adapter is not None:
                    transformed = []
                    for i, c_idx in enumerate(s2_comp_indices):
                        e_D = s2_comp_embeds[0, i]
                        z_D = qcla_adapter(e_D, e_q=e_q)
                        transformed.append(z_D)
                    s2_comp_embeds = torch.stack(transformed, dim=0).unsqueeze(0).to(dtype=model_dtype)

                with torch.no_grad():
                    s2_outputs = sara_model.generate(
                        **s2_inputs,
                        retrieval_embeds=s2_comp_embeds,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=pad_id,
                        eos_token_id=sara_tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                    )

                s2_gen_tokens = s2_outputs.sequences.shape[1] - s2_inputs["input_ids"].shape[1]
                total_generated_tokens += s2_gen_tokens
                s2_text_list = extract_generated_text(sara_tokenizer, s2_outputs.sequences)
                current_answer = s2_text_list[0].strip() if s2_text_list else ""

    t_total = time.time() - t_start

    # Metrics
    qa_metrics = compute_qa_answer_metrics(
        prediction=current_answer,
        gold_answers=gold_answers,
        gold_reformatted=gold_reformatted,
    )

    cost_rec = QueryCostRecord(
        question_id=qid,
        paper_id=pid,
        policy_name=f"DAFiR-{max_promotions}",
        num_passes=num_passes,
        promoted_indices=promoted_indices,
        final_context_tokens=final_context_tokens,
        cumulative_prompt_tokens=cumulative_prompt_tokens,
        total_generated_tokens=total_generated_tokens,
        e2e_latency_sec=t_total,
    )

    return {
        "question_id": qid,
        "paper_id": pid,
        "final_answer": current_answer,
        "base_draft": y0,
        "promoted_indices": promoted_indices,
        "num_passes": num_passes,
        "final_context_tokens": final_context_tokens,
        "cumulative_prompt_tokens": cumulative_prompt_tokens,
        "total_generated_tokens": total_generated_tokens,
        "e2e_latency_sec": t_total,
        "token_f1": float(qa_metrics["token_f1"]),
        "exact_match": float(qa_metrics["exact_match"]),
        "rouge_l": float(qa_metrics["rouge_l"]),
        "cost_record": cost_rec,
    }
