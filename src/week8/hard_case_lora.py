"""Hard-Case LoRA (HC-LoRA) fine-tuning for SARA generator (Week 8-G).

Trains PEFT QLoRA adapter on Mistral-7B targeting:
- q_proj, k_proj, v_proj, o_proj
- r = 8, alpha = 16, dropout = 0.05
- Train on FIT split; select checkpoint on INTERNAL-VAL.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def setup_lora_generator(
    base_model: Any,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[Sequence[str]] = None,
) -> Any:
    """Wrap base Mistral model with PEFT LoRA adapter."""
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as e:
        logger.error("PEFT library is required for LoRA fine-tuning: %s", e)
        raise

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=list(target_modules),
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # In SARA, base_model is a wrapper (e.g. SARAForCausalLM), wrap its inner llm or generator
    if hasattr(base_model, "llm"):
        base_model.llm = get_peft_model(base_model.llm, lora_config)
        base_model.llm.print_trainable_parameters()
        return base_model
    elif hasattr(base_model, "model"):
        peft_model = get_peft_model(base_model, lora_config)
        peft_model.print_trainable_parameters()
        return peft_model
    else:
        peft_model = get_peft_model(base_model, lora_config)
        peft_model.print_trainable_parameters()
        return peft_model


def train_generator_lora_epoch(
    sara_model: Any,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    sfr_embeddings_by_qid: Dict[str, torch.Tensor],
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    gradient_accumulation_steps: int = 4,
    device: str = "cuda",
    use_amp: bool = True,
    max_steps: Optional[int] = None,
) -> float:
    """Train one epoch of LoRA adapter with sample weights and SARA embedding injection."""
    sara_model.train()
    total_loss = 0.0
    accumulated_loss = 0.0
    step_count = 0
    num_updates = 0

    optimizer.zero_grad()

    for batch in dataloader:
        step_count += 1
        qid = batch["question_id"][0] if isinstance(batch["question_id"], (list, tuple)) else batch["question_id"]
        input_ids = batch["input_ids"].to(device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        attention_mask = batch["attention_mask"].to(device)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        labels = batch["labels"].to(device)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)

        sample_weight = batch["weight"].to(device)
        if sample_weight.dim() > 0:
            sample_weight = sample_weight.item()
        else:
            sample_weight = float(sample_weight)

        compressed_indices = batch.get("compressed_indices", [])
        if isinstance(compressed_indices, torch.Tensor):
            compressed_indices = compressed_indices.squeeze(0).tolist()
        elif isinstance(compressed_indices, list) and len(compressed_indices) > 0 and isinstance(compressed_indices[0], list):
            compressed_indices = compressed_indices[0]

        # Extract SFR compressed embeddings
        qid_embeds = sfr_embeddings_by_qid.get(str(qid))
        if qid_embeds is not None and len(compressed_indices) > 0:
            comp_embeds = qid_embeds[compressed_indices].unsqueeze(0).to(
                device=device, dtype=getattr(sara_model, "dtype", torch.float16)
            )
        else:
            comp_embeds = None

        with torch.cuda.amp.autocast(enabled=use_amp and device == "cuda"):
            outputs = sara_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                retrieval_embeds=comp_embeds,
                labels=labels,
            )
            # Apply outcome/hard-case sample weight
            loss = (outputs.loss * sample_weight) / gradient_accumulation_steps

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accumulated_loss += loss.item() * gradient_accumulation_steps

        if step_count % gradient_accumulation_steps == 0 or (max_steps and step_count >= max_steps):
            if scaler is not None and use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            total_loss += accumulated_loss
            accumulated_loss = 0.0
            num_updates += 1

        if max_steps and step_count >= max_steps:
            break

    if accumulated_loss > 0.0:
        if scaler is not None and use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
        total_loss += accumulated_loss
        num_updates += 1

    return total_loss / max(num_updates, 1)


def evaluate_generator_lora(
    sara_model: Any,
    dataloader: DataLoader,
    sfr_embeddings_by_qid: Dict[str, torch.Tensor],
    device: str = "cuda",
    use_amp: bool = True,
    max_steps: Optional[int] = None,
) -> float:
    """Evaluate validation loss on held-out split."""
    sara_model.eval()
    total_loss = 0.0
    num_batches = 0
    step_count = 0

    with torch.no_grad():
        for batch in dataloader:
            step_count += 1
            qid = batch["question_id"][0] if isinstance(batch["question_id"], (list, tuple)) else batch["question_id"]
            input_ids = batch["input_ids"].to(device)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)

            attention_mask = batch["attention_mask"].to(device)
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)

            labels = batch["labels"].to(device)
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)

            compressed_indices = batch.get("compressed_indices", [])
            if isinstance(compressed_indices, torch.Tensor):
                compressed_indices = compressed_indices.squeeze(0).tolist()
            elif isinstance(compressed_indices, list) and len(compressed_indices) > 0 and isinstance(compressed_indices[0], list):
                compressed_indices = compressed_indices[0]

            qid_embeds = sfr_embeddings_by_qid.get(str(qid))
            if qid_embeds is not None and len(compressed_indices) > 0:
                comp_embeds = qid_embeds[compressed_indices].unsqueeze(0).to(
                    device=device, dtype=getattr(sara_model, "dtype", torch.float16)
                )
            else:
                comp_embeds = None

            with torch.cuda.amp.autocast(enabled=use_amp and device == "cuda"):
                outputs = sara_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    retrieval_embeds=comp_embeds,
                    labels=labels,
                )
                loss = outputs.loss

            total_loss += loss.item()
            num_batches += 1

            if max_steps and step_count >= max_steps:
                break

    return total_loss / max(num_batches, 1)
