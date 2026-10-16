"""QCLA (Query-Conditioned Latent Adapter) training loop for Week 8-F.

Trains lightweight bottleneck residual adapters on top of frozen SFR passage embeddings:
- Freezes SARA generator (Mistral-7B) and projector.
- Transforms compressed passage embeddings via QCLA-1 or QCLA-2.
- Minimizes negative log-likelihood (NLL) of gold answer tokens.
- Supports batch size 1, gradient accumulation, and mixed precision.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.week8.qcla import (
    BaseLatentAdapter,
    QCLA0_Baseline,
    QCLA1_ResidualAdapter,
    QCLA2_DraftConditionedAdapter,
    build_qcla_adapter,
)

logger = logging.getLogger(__name__)


class QCLATrainingDataset(Dataset):
    """Dataset for training QCLA adapters with frozen SARA generator."""

    def __init__(
        self,
        records: List[Dict[str, Any]],
        sfr_embeddings_by_qid: Dict[str, torch.Tensor],
        query_embeddings_by_qid: Optional[Dict[str, torch.Tensor]] = None,
        draft_embeddings_by_qid: Optional[Dict[str, torch.Tensor]] = None,
        max_samples: Optional[int] = None,
    ):
        self.records = records[:max_samples] if max_samples is not None else records
        self.sfr_embeddings = sfr_embeddings_by_qid
        self.query_embeddings = query_embeddings_by_qid or {}
        self.draft_embeddings = draft_embeddings_by_qid or {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        qid = str(rec["question_id"]).strip()
        # Passage embeddings [10, 4096]
        e_passages = self.sfr_embeddings.get(qid)
        if e_passages is None:
            # Fallback zero tensor
            e_passages = torch.zeros((10, 4096), dtype=torch.float32)

        e_q = self.query_embeddings.get(qid)
        if e_q is None:
            e_q = torch.zeros(4096, dtype=torch.float32)

        e_y = self.draft_embeddings.get(qid)
        if e_y is None:
            e_y = torch.zeros(4096, dtype=torch.float32)

        return {
            "question_id": qid,
            "record": rec,
            "passage_embeddings": e_passages,
            "query_embedding": e_q,
            "draft_embedding": e_y,
        }


def train_qcla_epoch(
    adapter: BaseLatentAdapter,
    sara_model: Any,
    sara_tokenizer: Any,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    gradient_accumulation_steps: int = 4,
    device: str = "cuda",
    candidate_indices: Sequence[int] = (4, 5, 6, 7, 8, 9),
    use_amp: bool = True,
    max_steps: Optional[int] = None,
) -> float:
    """Run one epoch of QCLA adapter training.
    
    Freezes SARA model weights and optimizes only adapter parameters.
    """
    adapter.train()
    # Freeze generator
    if hasattr(sara_model, "eval"):
        sara_model.eval()
    for p in sara_model.parameters():
        p.requires_grad = False

    if hasattr(sara_model, "gradient_checkpointing_enable"):
        try:
            sara_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except Exception:
            try:
                sara_model.gradient_checkpointing_enable()
            except Exception:
                pass

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    total_loss = 0.0
    accumulated_loss = 0.0
    num_batches = 0
    step_count = 0

    optimizer.zero_grad()

    for batch in dataloader:
        step_count += 1
        rec = batch["record"]
        # If rec is batched as list of dicts, take first element
        if isinstance(rec, list):
            rec = rec[0]

        qid = str(rec["question_id"])
        question = rec.get("question", "")
        if isinstance(question, (list, tuple)):
            question = question[0]
        question = str(question)

        passages = rec.get("passages", [])
        if isinstance(passages, tuple):
            passages = list(passages)

        gold_answers = rec.get("gold_answers", [])
        if isinstance(gold_answers, tuple):
            gold_answers = list(gold_answers)
        raw_gold = gold_answers[0] if gold_answers else ""
        gold_text = raw_gold[0] if isinstance(raw_gold, (list, tuple)) else str(raw_gold)

        # Extract embeddings
        e_passages = batch["passage_embeddings"].to(device)  # [1, 10, 4096]
        if e_passages.dim() == 3:
            e_passages = e_passages.squeeze(0)
        e_q = batch["query_embedding"].to(device)
        if e_q.dim() == 2:
            e_q = e_q.squeeze(0)
        e_y = batch["draft_embedding"].to(device)
        if e_y.dim() == 2:
            e_y = e_y.squeeze(0)

        # 1. Transform candidate passages (indices 4..9) using adapter
        transformed_candidates = []
        for c_idx in candidate_indices:
            e_D = e_passages[c_idx]
            z_D = adapter(e_D, e_q=e_q, e_y=e_y)
            transformed_candidates.append(z_D)

        if transformed_candidates:
            comp_embeds = torch.stack(transformed_candidates, dim=0)  # [6, 4096]
        else:
            comp_embeds = None

        # 2. Build prompt with gold answer appended
        from src.week7.w7e.prompt_builder import build_mixed_fidelity_sara_prompt
        # Top-4 passages are FULL (0, 1, 2, 3), remaining 6 are HEAVY
        full_indices = [0, 1, 2, 3]
        heavy_indices = list(candidate_indices)

        formatted_prompt, _, _, _ = build_mixed_fidelity_sara_prompt(
            question=question,
            passages=passages,
            full_indices=full_indices,
            medium_indices=[],
            heavy_indices=heavy_indices,
            medium_texts_map={},
            tokenizer=sara_tokenizer,
            hybrid_medium=False,
        )

        full_prompt_text = formatted_prompt + "\nAnswer: " + gold_text
        prompt_inputs = sara_tokenizer(formatted_prompt + "\nAnswer: ", return_tensors="pt")
        full_inputs = sara_tokenizer(full_prompt_text, return_tensors="pt").to(device)

        prompt_len = prompt_inputs["input_ids"].shape[1]
        input_ids = full_inputs["input_ids"]
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100
        if (labels != -100).sum() == 0:
            continue

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Cast comp_embeds to model dtype
        model_dtype = getattr(sara_model, "dtype", torch.bfloat16)
        if comp_embeds is not None:
            comp_embeds = comp_embeds.to(dtype=model_dtype)

        # Forward pass through frozen SARA model
        with torch.cuda.amp.autocast(enabled=use_amp and "cuda" in str(device), dtype=model_dtype):
            outputs = sara_model(
                input_ids=input_ids,
                retrieval_embeds=comp_embeds,
                labels=labels,
            )
            loss = outputs.loss / gradient_accumulation_steps

        # Backward
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accumulated_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1

        if step_count % gradient_accumulation_steps == 0 or (max_steps and step_count >= max_steps):
            if scaler is not None and use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            total_loss += accumulated_loss
            accumulated_loss = 0.0

        if max_steps and step_count >= max_steps:
            break

    # Final step if remainder
    if accumulated_loss > 0.0:
        if scaler is not None and use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
        total_loss += accumulated_loss

    avg_loss = total_loss / max(num_batches / gradient_accumulation_steps, 1)
    return avg_loss


def evaluate_qcla(
    adapter: BaseLatentAdapter,
    sara_model: Any,
    sara_tokenizer: Any,
    dataloader: DataLoader,
    device: str = "cuda",
    candidate_indices: Sequence[int] = (4, 5, 6, 7, 8, 9),
    use_amp: bool = True,
    max_steps: Optional[int] = None,
) -> float:
    """Evaluate NLL loss of adapter on validation split."""
    adapter.eval()
    if hasattr(sara_model, "eval"):
        sara_model.eval()

    total_loss = 0.0
    num_batches = 0
    step_count = 0

    with torch.no_grad():
        for batch in dataloader:
            step_count += 1
            rec = batch["record"]
            if isinstance(rec, list):
                rec = rec[0]

            question = rec.get("question", "")
            if isinstance(question, (list, tuple)):
                question = question[0]
            question = str(question)

            passages = rec.get("passages", [])
            if isinstance(passages, tuple):
                passages = list(passages)

            gold_answers = rec.get("gold_answers", [])
            if isinstance(gold_answers, tuple):
                gold_answers = list(gold_answers)
            raw_gold = gold_answers[0] if gold_answers else ""
            gold_text = raw_gold[0] if isinstance(raw_gold, (list, tuple)) else str(raw_gold)

            e_passages = batch["passage_embeddings"].to(device)
            if e_passages.dim() == 3:
                e_passages = e_passages.squeeze(0)
            e_q = batch["query_embedding"].to(device)
            if e_q.dim() == 2:
                e_q = e_q.squeeze(0)
            e_y = batch["draft_embedding"].to(device)
            if e_y.dim() == 2:
                e_y = e_y.squeeze(0)

            transformed_candidates = []
            for c_idx in candidate_indices:
                e_D = e_passages[c_idx]
                z_D = adapter(e_D, e_q=e_q, e_y=e_y)
                transformed_candidates.append(z_D)

            if transformed_candidates:
                comp_embeds = torch.stack(transformed_candidates, dim=0)  # [6, 4096]
            else:
                comp_embeds = None

            from src.week7.w7e.prompt_builder import build_mixed_fidelity_sara_prompt
            full_indices = [0, 1, 2, 3]
            heavy_indices = list(candidate_indices)

            formatted_prompt, _, _, _ = build_mixed_fidelity_sara_prompt(
                question=question,
                passages=passages,
                full_indices=full_indices,
                medium_indices=[],
                heavy_indices=heavy_indices,
                medium_texts_map={},
                tokenizer=sara_tokenizer,
                hybrid_medium=False,
            )

            full_prompt_text = formatted_prompt + "\nAnswer: " + gold_text
            prompt_inputs = sara_tokenizer(formatted_prompt + "\nAnswer: ", return_tensors="pt")
            full_inputs = sara_tokenizer(full_prompt_text, return_tensors="pt").to(device)

            prompt_len = prompt_inputs["input_ids"].shape[1]
            input_ids = full_inputs["input_ids"]
            labels = input_ids.clone()
            labels[:, :prompt_len] = -100
            if (labels != -100).sum() == 0:
                continue

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            model_dtype = getattr(sara_model, "dtype", torch.bfloat16)
            if comp_embeds is not None:
                comp_embeds = comp_embeds.to(dtype=model_dtype)

            with torch.cuda.amp.autocast(enabled=use_amp and "cuda" in str(device), dtype=model_dtype):
                outputs = sara_model(
                    input_ids=input_ids,
                    retrieval_embeds=comp_embeds,
                    labels=labels,
                )
                loss = outputs.loss

            total_loss += loss.item()
            num_batches += 1

            if max_steps and step_count >= max_steps:
                break

    return total_loss / max(num_batches, 1)
