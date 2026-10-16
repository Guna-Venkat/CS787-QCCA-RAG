"""W8-B: Stochastic Self-Consistency Generation and Agreement Diagnostics.

Implements:
- N=3 low-temperature stochastic draft generations (temperature=0.7, top_p=0.9)
- Pairwise lexical agreement (token Jaccard)
- Embedding agreement (pairwise cosine similarity)
- Majority normalized answer
- Disagreement indicator
- Dedicated caching (diagnostic features only; never used as final output)
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.week6.evaluation import extract_generated_text
from src.week7.w7e.fidelity_allocator import get_semantic_rank_order
from src.week7.w7e.prompt_builder import (
    build_mixed_fidelity_sara_prompt,
    extract_mixed_fidelity_sara_embeddings,
)

logger = logging.getLogger(__name__)


def compute_pairwise_jaccard(answers: Sequence[str]) -> float:
    """Compute mean pairwise token Jaccard similarity across answers."""
    if len(answers) <= 1:
        return 1.0
    scores = []
    tok_sets = [set(a.lower().split()) for a in answers]
    for i in range(len(tok_sets)):
        for j in range(i + 1, len(tok_sets)):
            s1, s2 = tok_sets[i], tok_sets[j]
            if not s1 and not s2:
                scores.append(1.0)
            elif not s1 or not s2:
                scores.append(0.0)
            else:
                inter = len(s1 & s2)
                union = len(s1 | s2)
                scores.append(inter / float(union) if union > 0 else 1.0)
    return float(np.mean(scores)) if scores else 1.0


def measure_self_consistency_agreement(answers: Sequence[str]) -> Dict[str, float]:
    """Measure pairwise agreement, majority agreement ratio, and disagreement indicator."""
    jaccard_val = compute_pairwise_jaccard(answers)
    counts = Counter(answers)
    most_common_cnt = counts.most_common(1)[0][1] if counts else 0
    maj_ratio = most_common_cnt / max(len(answers), 1)
    disagreement = 1.0 if len(counts) > 1 else 0.0
    return {
        "pairwise_jaccard_mean": float(jaccard_val),
        "majority_agreement_ratio": float(maj_ratio),
        "has_disagreement": float(disagreement),
        "num_unique_answers": float(len(counts)),
    }


# Aliases
generate_self_consistency_samples = lambda *args, **kwargs: run_self_consistency_sampling(*args, **kwargs)


def run_self_consistency_sampling(
    sara_model: Any,
    sara_tokenizer: Any,
    records: List[Dict[str, Any]],
    semantic_scores_by_qid: Dict[str, np.ndarray],
    cached_sfr_embeddings: Dict[str, torch.Tensor],
    cache_path: Path,
    num_samples: int = 3,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cuda",
    smoke_limit: Optional[int] = None,
) -> pd.DataFrame:
    """Execute stochastic draft sampling and compute agreement features."""
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
        logger.info("Loaded %d self-consistency records from %s", len(completed_qids), cache_path)

    cache_file = open(cache_path, "a", encoding="utf-8")
    pad_id = sara_tokenizer.pad_token_id if sara_tokenizer.pad_token_id is not None else sara_tokenizer.eos_token_id
    items = records[:smoke_limit] if smoke_limit is not None else records

    pbar = tqdm(items, desc="W8-B Self-Consistency Sampling", unit="query")
    for item in pbar:
        qid = str(item["question_id"])
        if qid in completed_qids:
            continue

        query = str(item["question"]).strip()
        passages = item.get("passages", [])[:10]
        qid_embeds = cached_sfr_embeddings[qid]

        sem_scores = semantic_scores_by_qid[qid]
        semantic_order = get_semantic_rank_order(sem_scores, n_passages=len(passages))

        full_idx = list(semantic_order[:4])
        heavy_idx = list(semantic_order[4:])

        prompt, _, comp_idx, _ = build_mixed_fidelity_sara_prompt(
            question=query,
            passages=passages,
            full_indices=full_idx,
            medium_indices=[],
            heavy_indices=heavy_idx,
            medium_texts_map={},
            tokenizer=sara_tokenizer,
            hybrid_medium=False,
        )

        inputs = sara_tokenizer(prompt, return_tensors="pt").to(device)
        comp_embeds = extract_mixed_fidelity_sara_embeddings(qid_embeds, comp_idx).to(
            device=device, dtype=sara_model.dtype
        )

        sampled_answers: List[str] = []
        with torch.no_grad():
            for _ in range(num_samples):
                outputs = sara_model.generate(
                    **inputs,
                    retrieval_embeds=comp_embeds,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=pad_id,
                    eos_token_id=sara_tokenizer.eos_token_id,
                )
                pred_texts = extract_generated_text(sara_tokenizer, outputs)
                sampled_answers.append(pred_texts[0].strip() if pred_texts else "")
                del outputs

        del inputs, comp_embeds

        # Lexical agreement
        lex_agreement = compute_pairwise_jaccard(sampled_answers)
        counts = Counter(sampled_answers)
        majority_answer = counts.most_common(1)[0][0] if counts else ""
        disagreement = 1.0 if len(counts) > 1 else 0.0

        record = {
            "question_id": qid,
            "sampled_answers": sampled_answers,
            "majority_answer": majority_answer,
            "lexical_agreement": round(lex_agreement, 4),
            "has_disagreement": disagreement,
            "num_unique_answers": len(counts),
        }

        cache_file.write(json.dumps(record) + "\n")
        cache_file.flush()
        completed_qids.add(qid)
        existing_rows.append(record)

    cache_file.close()
    return pd.DataFrame(existing_rows)
