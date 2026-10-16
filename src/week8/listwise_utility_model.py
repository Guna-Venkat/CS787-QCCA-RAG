"""W8-C: Listwise Draft-Aware Utility Model (U3).

Uses Mistral-7B-Instruct-v0.2 with QLoRA:
Inputs all six compressed candidates jointly alongside question and draft answer:
    <QUESTION> q
    <DRAFT> y_0
    <CANDIDATE rank=5> ...
    ...
    <CANDIDATE rank=10> ...
    <DECIDE>

Heads:
- Action head: 7 classes [NO-PROMOTION, rank5, rank6, rank7, rank8, rank9, rank10]
- Per-candidate HELP heads: 6 binary sigmoid outputs P(HELP_i)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

logger = logging.getLogger(__name__)


def format_u3_listwise_prompt(
    question: str,
    draft_answer: str,
    candidates_by_rank: Dict[int, str],  # rank (5..10) -> text
) -> str:
    """Format prompt presenting all six candidates jointly."""
    lines = [
        "<QUESTION>",
        question.strip(),
        "",
        "<DRAFT>",
        draft_answer.strip() if draft_answer else "None",
        "",
    ]
    for r in sorted(candidates_by_rank.keys()):
        text = candidates_by_rank[r].strip()
        lines.append(f"<CANDIDATE rank={r}>")
        lines.append(text[:300])  # Truncate slightly to fit context cleanly
        lines.append("")

    lines.append("<DECIDE>")
    return "\n".join(lines)


# Aliases
format_listwise_input_prompt = format_u3_listwise_prompt


class ListwiseDraftUtilityModelU3(nn.Module):
    """Mistral-7B QLoRA listwise draft-aware utility router (U3)."""

    def __init__(
        self,
        base_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        lora_r: int = 8,
        lora_alpha: int = 16,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        load_in_4bit: bool = True,
    ):
        super().__init__()
        # Load base model with optional 4-bit quantization
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
            ) if load_in_4bit else None
        except ImportError:
            bnb_config = None

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
        }
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config

        raw_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
        if load_in_4bit:
            raw_model = prepare_model_for_kbit_training(raw_model)

        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.peft_model = get_peft_model(raw_model, peft_config)

        hidden_dim = self.peft_model.config.hidden_size
        # Action head: 7 classes [0: NO-PROMOTION, 1: rank 5, ..., 6: rank 10]
        self.action_head = nn.Linear(hidden_dim, 7)
        # Per-candidate HELP heads: 6 binary predictions
        self.help_heads = nn.Linear(hidden_dim, 6)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.peft_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Extract last hidden state at the last token position
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        action_logits = self.action_head(last_hidden)
        help_logits = self.help_heads(last_hidden)

        return {
            "action_logits": action_logits,
            "help_logits": help_logits,
            "prob_help": torch.sigmoid(help_logits),
            "prob_action": torch.softmax(action_logits, dim=-1),
        }


# Aliases
MistralListwiseUtilityRouter = ListwiseDraftUtilityModelU3
