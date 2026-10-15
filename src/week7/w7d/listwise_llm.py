"""Listwise 7B LLM Fidelity Router (D2) for Week 7-D.

Provides:
- Listwise causal prompt formatting:
    <DOC rank="1"> ... </DOC>
    ...
    <DOC rank="10"> ... </DOC>
    <QUESTION> ... </QUESTION>
    <DECIDE>
- Extraction of hidden state at <DECIDE> token (h_DECIDE in R^4096).
- Multi-task output heads attached to h_DECIDE.
- Config-guarded instantiation (ENABLE_LISTWISE_LLM_ROUTER: false by default).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.week7.w7d.multitask_heads import MultiTaskUtilityHeads

logger = logging.getLogger(__name__)

DECIDE_TOKEN = "<DECIDE>"


def format_listwise_causal_prompt(
    question: str,
    passages: Sequence[Dict[str, Any]],
    max_passages: int = 10,
) -> str:
    """Format full listwise prompt in causal order."""
    doc_blocks: List[str] = []
    for p in passages[:max_passages]:
        sem_rank = p.get("semantic_rank", 1)
        bm25_rank = p.get("bm25_rank", sem_rank)
        text = p.get("text", "").strip()
        doc_blocks.append(f'<DOC rank="{sem_rank}" bm25_rank="{bm25_rank}">\n{text}\n</DOC>')

    all_docs = "\n\n".join(doc_blocks)
    prompt = f"{all_docs}\n\n<QUESTION>\n{question.strip()}\n</QUESTION>\n\n{DECIDE_TOKEN}"
    return prompt


class ListwiseLLMRouter(nn.Module):
    """Listwise 7B LLM router predicting fidelity requirements from h_DECIDE."""

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        d_model: int = 4096,
        head_hidden_dim: int = 256,
        dropout: float = 0.1,
        use_lora: bool = True,
        load_in_4bit: bool = True,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.d_model = d_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Heads attached to h_DECIDE
        self.heads = MultiTaskUtilityHeads(
            d_in=d_model,
            hidden_dim=head_hidden_dim,
            dropout=dropout,
            enable_aux_k=True,
            enable_passage_utility=False,
        )

        self.use_lora = use_lora
        self.load_in_4bit = load_in_4bit
        self.tokenizer = None
        self.base_model = None
        self.decide_token_id = None

    def initialize_llm(self) -> None:
        """Initialize the tokenizer and base causal LM with optional LoRA/QLoRA."""
        logger.info("Initializing Listwise LLM: %s...", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add <DECIDE> special token if not present
        if DECIDE_TOKEN not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [DECIDE_TOKEN]})

        self.decide_token_id = self.tokenizer.convert_tokens_to_ids(DECIDE_TOKEN)

        bnb_available = False
        try:
            import bitsandbytes as bnb
            bnb_available = True
        except ImportError:
            bnb_available = False

        if self.load_in_4bit and bnb_available:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Loading %s in 4-bit NF4 quantization...", self.model_name)
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quant_config,
                device_map="auto" if torch.cuda.is_available() else None,
            )
        else:
            logger.info("Loading %s in float16...", self.model_name)
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )

        self.base_model.resize_token_embeddings(len(self.tokenizer))

        if self.use_lora:
            try:
                from peft import get_peft_model
                from src.week7.w7d.lora_utils import get_peft_lora_config
                peft_config = get_peft_lora_config()
                self.base_model = get_peft_model(self.base_model, peft_config)
                logger.info("Applied PEFT LoRA adapter to base LLM.")
            except Exception as e:
                logger.warning("Could not apply PEFT LoRA: %s. Freezing base parameters directly.", e)
                for p in self.base_model.parameters():
                    p.requires_grad = False

        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            self.base_model.gradient_checkpointing_enable()

        self.heads.to(self.device)

    def extract_decide_representation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract hidden state at <DECIDE> token position."""
        if self.base_model is None:
            raise RuntimeError("LLM base model has not been initialized!")

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1]  # (B, seq_len, d_model)

        # Locate <DECIDE> token or use last token
        B = input_ids.shape[0]
        h_decide_list = []
        for i in range(B):
            positions = (input_ids[i] == self.decide_token_id).nonzero(as_tuple=True)[0]
            if len(positions) > 0:
                pos = positions[-1].item()
            else:
                # Fallback to last non-pad token
                non_pad = (attention_mask[i] == 1).nonzero(as_tuple=True)[0]
                pos = non_pad[-1].item() if len(non_pad) > 0 else -1
            h_decide_list.append(last_hidden[i, pos, :])

        return torch.stack(h_decide_list, dim=0)  # (B, d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass from tokenized input prompt."""
        h_decide = self.extract_decide_representation(input_ids, attention_mask)
        # Cast to head dtype/device
        h_decide_cast = h_decide.to(dtype=self.heads.trunk[0].weight.dtype, device=self.device)
        outputs = self.heads(h_decide_cast)
        outputs["h_decide"] = h_decide
        return outputs

