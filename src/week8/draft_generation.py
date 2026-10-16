"""W8-B: Canonical Draft Generation with Detailed Token Diagnostics.

Extracts:
- generated answer y_0
- token IDs
- per-token logits and probabilities
- per-step Shannon entropy
- top1-top2 logit margins
- normalized sequence log probability
- EOS confidence
- accurate latency tracking
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.week6.evaluation import compute_qa_answer_metrics, extract_generated_text
from src.week7.w7e.fidelity_allocator import get_semantic_rank_order
from src.week7.w7e.prompt_builder import (
    build_mixed_fidelity_sara_prompt,
    extract_mixed_fidelity_sara_embeddings,
)

logger = logging.getLogger(__name__)


def extract_draft_uncertainty_features(
    query: str,
    draft_text: str,
    scores: Optional[Sequence[torch.Tensor]] = None,
    generated_token_ids: Optional[torch.Tensor] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    semantic_scorer: Optional[Any] = None,
) -> Dict[str, float]:
    """Extract U1..U12 uncertainty features from generation scores and token IDs."""
    entropies = []
    margins = []
    logprobs = []
    eos_conf = 0.0

    if scores:
        for step, step_logits in enumerate(scores):
            logits = step_logits[0] if step_logits.dim() > 1 else step_logits
            probs = F.softmax(logits, dim=-1)
            p_nz = probs[probs > 1e-12]
            ent = -torch.sum(p_nz * torch.log(p_nz)).item()
            entropies.append(ent)

            top2 = torch.topk(logits, k=2).values
            margins.append((top2[0] - top2[1]).item())

            log_p = F.log_softmax(logits, dim=-1)
            if generated_token_ids is not None and step < len(generated_token_ids):
                tok_val = generated_token_ids[step]
                tok_id = int(tok_val.item() if isinstance(tok_val, torch.Tensor) else tok_val)
                logprobs.append(float(log_p[tok_id].item()))
            else:
                logprobs.append(float(torch.max(log_p).item()))

        last_p = F.softmax(scores[-1][0] if scores[-1].dim() > 1 else scores[-1], dim=-1)
        if eos_token_id is not None and eos_token_id < last_p.shape[0]:
            eos_conf = float(last_p[eos_token_id].item())

    tok_ids_list = (
        generated_token_ids.tolist()
        if isinstance(generated_token_ids, torch.Tensor)
        else (list(generated_token_ids) if generated_token_ids is not None else [])
    )

    u1 = float(np.mean(entropies)) if entropies else 0.0
    u2 = float(np.max(entropies)) if entropies else 0.0
    u3 = float(np.mean(margins)) if margins else 0.0
    u4 = float(np.min(margins)) if margins else 0.0
    u5 = float(np.mean(logprobs)) if logprobs else 0.0
    u6 = float(np.sum(logprobs) / len(logprobs)) if logprobs else -100.0
    u7 = float(len(tok_ids_list))
    u8 = float(eos_conf)
    u9 = 1.0 if len(draft_text.strip()) == 0 else 0.0
    u10 = 1.0 - (len(set(tok_ids_list)) / float(len(tok_ids_list))) if len(tok_ids_list) > 0 else 0.0

    d_toks = set(draft_text.lower().split())
    q_toks = set(query.lower().split())
    u11 = float(len(d_toks & q_toks) / len(d_toks | q_toks)) if (d_toks and q_toks) else 0.0
    u12 = 0.0

    return {
        "mean_entropy": u1,
        "max_entropy": u2,
        "mean_margin": u3,
        "min_margin": u4,
        "mean_logprob": u5,
        "norm_seq_logprob": u6,
        "answer_length": u7,
        "eos_confidence": u8,
        "empty_answer": u9,
        "repetition_ratio": u10,
        "lexical_overlap": u11,
        "semantic_similarity": u12,
        "u1_entropy_mean": u1,
        "u2_entropy_max": u2,
        "u3_margin_mean": u3,
        "u4_margin_min": u4,
        "u5_logprob_mean": u5,
        "u6_norm_seq_logprob": u6,
        "u7_answer_len": u7,
        "u8_eos_conf": u8,
        "u9_empty_answer": u9,
        "u10_repetition_ratio": u10,
        "u11_lexical_overlap": u11,
        "u12_semantic_sim": u12,
    }


def generate_draft_with_diagnostics(
    model: Any,
    tokenizer: Any,
    formatted_prompt: str,
    retrieval_embeds: torch.Tensor,
    device: str = "cuda",
    max_new_tokens: int = 128,
) -> Dict[str, Any]:
    """Execute greedy generation while capturing token-level logits and entropy.

    Returns dict containing:
    - prediction: string text
    - token_ids: list of generated token IDs
    - token_entropies: list of float Shannon entropies
    - top1_top2_margins: list of float logit differences
    - token_logprobs: list of log P(token)
    - eos_confidence: float probability of EOS at end
    - latency_sec: float total generation latency
    """
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    comp_embeds = retrieval_embeds.to(device=device, dtype=model.dtype)
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            retrieval_embeds=comp_embeds,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
    latency = time.time() - t0

    pred_texts = extract_generated_text(tokenizer, outputs.sequences)
    prediction = pred_texts[0].strip() if pred_texts else ""

    # Extract per-token generation metrics from outputs.scores
    # outputs.scores is a tuple of length T, each tensor of shape (1, vocab_size)
    gen_token_ids: List[int] = []
    entropies: List[float] = []
    margins: List[float] = []
    logprobs: List[float] = []
    eos_conf: float = 0.0

    input_len = inputs["input_ids"].shape[1]
    gen_seq = outputs.sequences[0, input_len:].tolist()

    if outputs.scores:
        for step, step_logits in enumerate(outputs.scores):
            # step_logits: (1, vocab_size)
            logits = step_logits[0]  # (vocab_size,)
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)

            # 1. Entropy: H = -sum(p * log p)
            # Filter non-zero probs for numerical stability
            p_nz = probs[probs > 1e-12]
            ent = -torch.sum(p_nz * torch.log(p_nz)).item()
            entropies.append(ent)

            # 2. Top1 - Top2 margin
            top2_vals = torch.topk(logits, k=2).values
            margin = (top2_vals[0] - top2_vals[1]).item()
            margins.append(margin)

            # 3. Selected token logprob
            if step < len(gen_seq):
                tok_id = gen_seq[step]
                gen_token_ids.append(tok_id)
                logp = log_probs[tok_id].item()
                logprobs.append(logp)

        # EOS probability at last step
        last_probs = F.softmax(outputs.scores[-1][0], dim=-1)
        eos_conf = float(last_probs[eos_id].item()) if eos_id is not None else 0.0

    return {
        "prediction": prediction,
        "token_ids": gen_token_ids,
        "token_entropies": entropies,
        "top1_top2_margins": margins,
        "token_logprobs": logprobs,
        "eos_confidence": eos_conf,
        "latency_sec": latency,
    }


def execute_resumable_draft_generation(
    sara_model: Any,
    sara_tokenizer: Any,
    records: List[Dict[str, Any]],
    semantic_scores_by_qid: Dict[str, np.ndarray],
    cached_sfr_embeddings: Dict[str, torch.Tensor],
    cache_path: Path,
    device: str = "cuda",
    smoke_limit: Optional[int] = None,
) -> pd.DataFrame:
    """Run canonical semantic k=4 draft generation and cache diagnostics."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    completed_qids = set()
    existing_rows: List[Dict[str, Any]] = []

    if cache_path.is_file():
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    completed_qids.add(str(rec["question_id"]))
                    existing_rows.append(rec)
        logger.info("Loaded %d existing draft records from %s", len(completed_qids), cache_path)

    cache_file = open(cache_path, "a", encoding="utf-8")
    items = records[:smoke_limit] if smoke_limit is not None else records
    pbar = tqdm(items, desc="W8-B Draft Generation", unit="query")

    for item in pbar:
        qid = str(item["question_id"])
        if qid in completed_qids:
            continue

        query = str(item["question"]).strip()
        passages = item.get("passages", [])[:10]
        gold_answers = item.get("gold_answers", [])
        gold_reformatted = item.get("gold_answers_reformatted", gold_answers)
        qid_embeds = cached_sfr_embeddings[qid]

        sem_scores = semantic_scores_by_qid[qid]
        semantic_order = get_semantic_rank_order(sem_scores, n_passages=len(passages))

        # Base k=4: top-4 FULL, remaining 6 HEAVY
        full_idx = list(semantic_order[:4])
        heavy_idx = list(semantic_order[4:])

        prompt, ctx_tokens, comp_idx, token_breakdown = build_mixed_fidelity_sara_prompt(
            question=query,
            passages=passages,
            full_indices=full_idx,
            medium_indices=[],
            heavy_indices=heavy_idx,
            medium_texts_map={},
            tokenizer=sara_tokenizer,
            hybrid_medium=False,
        )

        comp_embeds = extract_mixed_fidelity_sara_embeddings(qid_embeds, comp_idx)
        diag = generate_draft_with_diagnostics(
            model=sara_model,
            tokenizer=sara_tokenizer,
            formatted_prompt=prompt,
            retrieval_embeds=comp_embeds,
            device=device,
        )

        qa_metrics = compute_qa_answer_metrics(
            prediction=diag["prediction"],
            gold_answers=gold_answers,
            gold_reformatted=gold_reformatted,
        )

        record = {
            "question_id": qid,
            "paper_id": str(item.get("paper_id", "")),
            "question": query,
            "draft_answer": diag["prediction"],
            "base_f1": float(qa_metrics["token_f1"]),
            "exact_match": float(qa_metrics["exact_match"]),
            "rouge_l": float(qa_metrics.get("rouge_l", qa_metrics.get("rouge_l_f1", 0.0))),
            "context_tokens": ctx_tokens,
            "token_ids": diag["token_ids"],
            "token_entropies": diag["token_entropies"],
            "top1_top2_margins": diag["top1_top2_margins"],
            "token_logprobs": diag["token_logprobs"],
            "eos_confidence": diag["eos_confidence"],
            "latency_sec": diag["latency_sec"],
            "gold_answers": gold_answers,
        }

        cache_file.write(json.dumps(record) + "\n")
        cache_file.flush()
        completed_qids.add(qid)
        existing_rows.append(record)

    cache_file.close()
    return pd.DataFrame(existing_rows)
