"""Hard-Case Dataset Construction for Week 8-G.

Constructs supervised fine-tuning dataset categorized by counterfactual promotion utility:
- Type A: Helpful promotion (k4 F1 low, max_i Delta_i > 0.01) -> context has best promoted passage explicit (5F + 5H).
- Type B: Distractor-sensitive (exists i: Delta_i < -0.01) -> standard k4 context (4F + 6H) avoiding distractor.
- Type C: Neutral (|Delta_i| <= 0.01 for all i) -> standard k4 context (4F + 6H).

Supports sample weighting:
- Standard Hard-Case (HC): Type A and B upweighted relative to Type C.
- Outcome-Weighted (OW): Sample weight w = 1.0 + gamma * max_i Delta_i for Type A, and safe-distractor weighting for Type B.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def categorize_hard_cases(
    promotion_records_by_qid: Dict[str, List[Dict[str, Any]]],
    epsilon_help: float = 0.01,
    epsilon_hurt: float = -0.01,
) -> Dict[str, Dict[str, Any]]:
    """Categorize each query into Type A, Type B, or Type C based on counterfactual promotion surface.

    Args:
        promotion_records_by_qid: Mapping from question_id -> list of candidate promotion dicts.
        epsilon_help: Threshold for helpful promotion (> 0.01).
        epsilon_hurt: Threshold for harmful promotion (< -0.01).

    Returns:
        Mapping from question_id -> metadata dict with type, best_candidate_idx, max_delta, min_delta.
    """
    categorized: Dict[str, Dict[str, Any]] = {}

    for qid, candidates in promotion_records_by_qid.items():
        if not candidates:
            continue

        base_f1 = candidates[0].get("base_f1", 0.0)
        deltas = [c.get("delta_f1", 0.0) for c in candidates]
        cand_indices = [c.get("candidate_passage_index", c.get("candidate_index", 4)) for c in candidates]

        max_delta = float(max(deltas))
        min_delta = float(min(deltas))
        best_cand_idx = cand_indices[int(np.argmax(deltas))]

        has_help = max_delta > epsilon_help
        has_hurt = min_delta < epsilon_hurt

        if has_help:
            case_type = "TYPE_A_HELPFUL"
        elif has_hurt:
            case_type = "TYPE_B_DISTRACTOR_SAFE"
        else:
            case_type = "TYPE_C_NEUTRAL"

        categorized[qid] = {
            "question_id": qid,
            "case_type": case_type,
            "base_f1": base_f1,
            "max_delta": max_delta,
            "min_delta": min_delta,
            "best_candidate_idx": best_cand_idx,
            "num_candidates": len(candidates),
        }

    return categorized


class HardCasePromptDataset(Dataset):
    """PyTorch Dataset for Hard-Case and Outcome-Weighted LoRA fine-tuning."""

    def __init__(
        self,
        examples: List[Dict[str, Any]],
        tokenizer: Any,
        max_seq_len: int = 2048,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        prompt_text = ex["prompt_text"]
        gold_text = ex["gold_text"]
        weight = float(ex.get("weight", 1.0))
        qid = ex.get("question_id", "")
        case_type = ex.get("case_type", "TYPE_C_NEUTRAL")

        full_text = prompt_text + "\nAnswer: " + gold_text
        prompt_inputs = self.tokenizer(prompt_text + "\nAnswer: ", return_tensors="pt")
        full_inputs = self.tokenizer(
            full_text,
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        prompt_len = min(prompt_inputs["input_ids"].shape[1], full_inputs["input_ids"].shape[1])
        input_ids = full_inputs["input_ids"].squeeze(0)
        attention_mask = full_inputs["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        # Compressed indices for SARA embedding injection
        compressed_indices = ex.get("compressed_indices", [4, 5, 6, 7, 8, 9])

        return {
            "question_id": qid,
            "case_type": case_type,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "weight": torch.tensor(weight, dtype=torch.float32),
            "compressed_indices": compressed_indices,
        }


def build_hard_case_dataset(
    raw_queries: List[Dict[str, Any]],
    promotion_records: List[Dict[str, Any]],
    tokenizer: Optional[Any] = None,
    mode: str = "hard_case",  # "hard_case" or "outcome_weighted"
    gamma: float = 2.0,
    type_a_weight: float = 2.0,
    type_b_weight: float = 2.0,
    type_c_weight: float = 1.0,
    epsilon_help: float = 0.01,
    epsilon_hurt: float = -0.01,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Assemble formatted training examples with weights for HC-LoRA or OW-LoRA."""
    from src.week7.w7e.prompt_builder import build_mixed_fidelity_sara_prompt

    # Group promotion records by qid
    prom_by_qid: Dict[str, List[Dict[str, Any]]] = {}
    for r in promotion_records:
        qid = str(r["question_id"]).strip()
        prom_by_qid.setdefault(qid, []).append(r)

    categorized = categorize_hard_cases(
        prom_by_qid, epsilon_help=epsilon_help, epsilon_hurt=epsilon_hurt
    )

    query_map = {str(q["question_id"]).strip(): q for q in raw_queries}
    examples: List[Dict[str, Any]] = []
    type_counts = {"TYPE_A_HELPFUL": 0, "TYPE_B_DISTRACTOR_SAFE": 0, "TYPE_C_NEUTRAL": 0}

    for qid, cat_info in categorized.items():
        q_record = query_map.get(qid)
        if q_record is None:
            continue

        c_type = cat_info["case_type"]
        type_counts[c_type] = type_counts.get(c_type, 0) + 1

        question = q_record["question"]
        passages = q_record["passages"]
        gold_answers = q_record.get("gold_answers", [])
        gold_text = gold_answers[0] if gold_answers else ""

        # Determine context configuration:
        # Type A: Promote best candidate passage to FULL (5F + 5H)
        # Type B & C: Standard k4 (4F + 6H)
        if c_type == "TYPE_A_HELPFUL":
            best_idx = cat_info["best_candidate_idx"]
            full_indices = [0, 1, 2, 3, best_idx]
            heavy_indices = [idx for idx in range(10) if idx not in full_indices]
        else:
            full_indices = [0, 1, 2, 3]
            heavy_indices = [4, 5, 6, 7, 8, 9]

        formatted_prompt, _, comp_indices, _ = build_mixed_fidelity_sara_prompt(
            question=question,
            passages=passages,
            full_indices=full_indices,
            medium_indices=[],
            heavy_indices=heavy_indices,
            medium_texts_map={},
            tokenizer=tokenizer,
            hybrid_medium=False,
        )

        # Compute sample weight
        if mode == "outcome_weighted":
            if c_type == "TYPE_A_HELPFUL":
                sample_weight = 1.0 + gamma * max(0.0, cat_info["max_delta"])
            elif c_type == "TYPE_B_DISTRACTOR_SAFE":
                # Upweight safe avoidance of distractor
                sample_weight = 1.0 + gamma * max(0.0, -cat_info["min_delta"])
            else:
                sample_weight = 1.0
        else:  # hard_case
            if c_type == "TYPE_A_HELPFUL":
                sample_weight = type_a_weight
            elif c_type == "TYPE_B_DISTRACTOR_SAFE":
                sample_weight = type_b_weight
            else:
                sample_weight = type_c_weight

        examples.append({
            "question_id": qid,
            "case_type": c_type,
            "prompt_text": formatted_prompt,
            "gold_text": gold_text,
            "weight": float(sample_weight),
            "compressed_indices": comp_indices,
            "max_delta": cat_info["max_delta"],
            "min_delta": cat_info["min_delta"],
        })

    logger.info("Built %d hard-case examples. Distribution: %s", len(examples), type_counts)
    return examples, type_counts
