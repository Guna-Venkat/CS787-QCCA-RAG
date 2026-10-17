"""W8-C: Listwise Draft-Aware Utility Model (U3).

Observes all six candidate passages jointly alongside question and draft answer:
    <QUESTION> q
    <DRAFT> y_0
    <CANDIDATE rank=5> ...
    ...
    <CANDIDATE rank=10> ...
    <DECIDE>

Action space:
- 7 classes: [0: STOP/NO-PROMOTION, 1: rank 5, ..., 6: rank 10]
- Per-candidate utility / HELP scores: 6 continuous/probabilistic outputs.

Backbones:
- Causal: Mistral-7B-Instruct-v0.2 with QLoRA (full generative capability).
- Encoder: cross-encoder / transformer encoder with multi-candidate joint pooling (lightweight, memory-efficient).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.week8.utility_models import BasePassageSelector

logger = logging.getLogger(__name__)


def format_u3_listwise_prompt(
    question: str,
    draft_answer: str,
    candidates_by_rank: Dict[int, str],  # rank (5..10) -> text
    max_cand_chars: int = 300,
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
        text = str(candidates_by_rank[r]).strip()
        lines.append(f"<CANDIDATE rank={r}>")
        lines.append(text[:max_cand_chars])
        lines.append("")

    lines.append("<DECIDE>")
    return "\n".join(lines)


# Aliases
format_listwise_input_prompt = format_u3_listwise_prompt


class ListwiseDraftUtilityModelU3(BasePassageSelector, nn.Module):
    """Listwise draft-aware utility router (U3) with joint candidate reasoning."""

    def __init__(
        self,
        base_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        backbone_type: str = "encoder",  # 'encoder' or 'causal'
        lora_r: int = 8,
        lora_alpha: int = 16,
        torch_dtype: torch.dtype = torch.float32,
        load_in_4bit: bool = False,
        device_map: Optional[str] = None,
    ):
        super().__init__()
        self.backbone_type = backbone_type
        self.base_model_name = base_model_name

        if backbone_type == "causal":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            bnb_config = None
            if load_in_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                except ImportError:
                    pass

            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            model_kwargs = {"torch_dtype": torch_dtype}
            if device_map:
                model_kwargs["device_map"] = device_map
            if bnb_config:
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
            self.model = get_peft_model(raw_model, peft_config)
            hidden_dim = self.model.config.hidden_size
        else:
            # Lightweight transformer encoder (MiniLM / RoBERTa)
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            self.model = AutoModel.from_pretrained(base_model_name)
            hidden_dim = self.model.config.hidden_size

        # Action head: 7 classes [0: STOP/NO-PROMOTION, 1: rank 5, ..., 6: rank 10]
        self.action_head = nn.Linear(hidden_dim, 7)
        # Per-candidate utility score head: 6 outputs for candidates 5..10
        self.utility_head = nn.Linear(hidden_dim, 6)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.backbone_type == "causal":
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            last_hidden = outputs.hidden_states[-1][:, -1, :]
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state[:, 0, :]  # [CLS] representation

        action_logits = self.action_head(last_hidden)
        utility_scores = self.utility_head(last_hidden)

        return {
            "action_logits": action_logits,
            "utility_scores": utility_scores,
            "prob_action": F.softmax(action_logits, dim=-1),
        }

    def predict_scores(self, df_query: pd.DataFrame) -> np.ndarray:
        """Score all 6 candidates jointly given a single query's candidate rows."""
        # Reuse the training tokenizer/budget and map scores by semantic rank,
        # not by the original BM25 passage index or caller row order.
        from src.week8.utility_training import TrainingConfig, predict_neural
        cfg = getattr(self, 'training_config',
                      TrainingConfig(device=str(next(self.parameters()).device)))
        return predict_neural(self, df_query, cfg)['utility_score']


# Aliases
MistralListwiseUtilityRouter = ListwiseDraftUtilityModelU3
