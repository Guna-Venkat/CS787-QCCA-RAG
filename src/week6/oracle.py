"""Small diagnostic subset oracle for Week 6 Passage Routing.

Enumerates all C(10, k) possible passage fidelity subsets on a small configurable
diagnostic DEV sample to estimate empirical routing headroom.

For k=2:
- C(10, 2) = 45 possible high-fidelity passage pairs.
- Identifies optimal pair H*_2(q) = argmax_H F1(q, H).
- Quantifies headroom above BM25 prefix H_0 = {1, 2}.
- Strictly guarded by RUN_SUBSET_ORACLE = False.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.week6.evaluation import compute_qa_answer_metrics
from src.week6.routing import (
    build_routed_sara_prompt,
    extract_routed_sara_embeddings,
    validate_routing_assignment,
)

logger = logging.getLogger(__name__)


def get_all_subsets(n: int = 10, k: int = 2) -> List[List[int]]:
    """Enumerate all C(n, k) combinations of 0-indexed passage ranks."""
    return [list(comb) for comb in itertools.combinations(range(n), k)]


def evaluate_diagnostic_subset_oracle(
    sara_model: Any,
    sara_tokenizer: Any,
    dev_records: List[Dict[str, Any]],
    cached_embeddings: Dict[str, torch.Tensor],
    k: int = 2,
    sample_size: int = 25,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_new_tokens: int = 64,
) -> pd.DataFrame:
    """Evaluate exhaustive subset oracle on a diagnostic subset of DEV questions.

    WARNING: Evaluates C(10, k) SARA generations per question (45 per question for k=2).
    Only run when explicitly enabled.
    """
    subsets = get_all_subsets(n=10, k=k)
    assert len(subsets) == 45, f"Expected 45 subsets for C(10, 2), got {len(subsets)}"

    eval_records = dev_records[:sample_size]
    sara_model.eval()
    pad_id = sara_tokenizer.pad_token_id if sara_tokenizer.pad_token_id is not None else sara_tokenizer.eos_token_id

    rows = []
    from src.model.xMistral import extract_generated_text

    for item in eval_records:
        qid = str(item["question_id"])
        question_text = item["question"]
        passages = item["passages"][:10]
        gold_answers = item.get("gold_answers", [])
        gold_reformatted = item.get("gold_answers_reformatted", gold_answers)
        qid_embeds = cached_embeddings[qid]

        subset_f1s = []
        for subset in subsets:
            validate_routing_assignment(subset, k=k, total_passages=len(passages))
            prompt, _, _ = build_routed_sara_prompt(question_text, passages, subset, sara_tokenizer)
            inputs = sara_tokenizer(prompt, return_tensors="pt").to(device)
            comp_embeds = extract_routed_sara_embeddings(qid_embeds, subset).to(
                device=device, dtype=sara_model.dtype
            )

            with torch.no_grad():
                outputs = sara_model.generate(
                    **inputs,
                    retrieval_embeds=comp_embeds,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_id,
                    eos_token_id=sara_tokenizer.eos_token_id,
                )
            pred = extract_generated_text(sara_tokenizer, outputs)[0].strip()
            metrics = compute_qa_answer_metrics(pred, gold_answers, gold_reformatted)
            subset_f1s.append(metrics["token_f1"])

        best_idx = int(np.argmax(subset_f1s))
        best_f1 = float(subset_f1s[best_idx])
        best_subset = [x + 1 for x in subsets[best_idx]]  # 1-indexed ranks

        # R0 is the first subset [0, 1]
        r0_f1 = float(subset_f1s[0])

        rows.append({
            "question_id": qid,
            "r0_f1": r0_f1,
            "oracle_f1": best_f1,
            "headroom_delta": best_f1 - r0_f1,
            "best_ranks": str(best_subset),
            "mean_subset_f1": float(np.mean(subset_f1s)),
            "std_subset_f1": float(np.std(subset_f1s)),
        })

    return pd.DataFrame(rows)
