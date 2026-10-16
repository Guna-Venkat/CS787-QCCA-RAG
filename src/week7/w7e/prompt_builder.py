"""Mixed-fidelity SARA prompt formatting and token accounting for W7-E.

Constructs unified prompt supporting:
- FULL natural-language passages
- MEDIUM query-focused extractive textual passages
- HEAVY <COMPRESS> tokens for SFR-compressed embeddings

Strictly guarantees:
- Every passage has exactly ONE representation.
- Deterministic presentation order.
- Exact slice mapping for SFR embeddings corresponding to HEAVY passages.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

try:
    from src.const import COMPRESS
except ImportError:
    COMPRESS = "<COMPRESS>"

from src.week7.w7e.fidelity_allocator import validate_multifidelity_assignment

logger = logging.getLogger(__name__)


def build_mixed_fidelity_sara_prompt(
    question: str,
    passages: List[Dict[str, Any]],
    full_indices: Sequence[int],
    medium_indices: Sequence[int],
    heavy_indices: Sequence[int],
    medium_texts_map: Dict[int, str],
    tokenizer: Any,
    hybrid_medium: bool = False,
) -> Tuple[str, int, List[int], Dict[str, int]]:
    """Format unified SARA prompt combining FULL, MEDIUM, and HEAVY representations.

    Args:
        question: Query text.
        passages: List of 10 retrieved passage dicts.
        full_indices: Passage indices assigned to FULL text.
        medium_indices: Passage indices assigned to MEDIUM extract text.
        heavy_indices: Passage indices assigned to HEAVY compression embeddings.
        medium_texts_map: Mapping from passage index -> extractive MEDIUM string.
        tokenizer: SARA generator tokenizer.
        hybrid_medium: If True, MEDIUM passages retain their <COMPRESS> tokens and
            SFR embeddings in addition to having extractive text in the context.

    Returns:
        (formatted_prompt, context_tokens, compressed_indices, token_breakdown)
    """
    total_passages = len(passages)
    validate_multifidelity_assignment(
        full_indices=full_indices,
        medium_indices=medium_indices,
        heavy_indices=heavy_indices,
        total_passages=total_passages,
    )

    full_set = set(full_indices)
    medium_set = set(medium_indices)
    heavy_set = set(heavy_indices)

    # 1. Text Context: All textual passages (FULL and MEDIUM) in original ascending BM25 index order
    textual_indices = sorted(full_set | medium_set)
    text_blocks: List[str] = []
    full_char_len = 0
    medium_char_len = 0

    if textual_indices:
        for idx in textual_indices:
            orig_rank = idx + 1
            if idx in full_set:
                p_raw = passages[idx].get("text", "")
                p_text = " ".join(p_raw).strip() if isinstance(p_raw, list) else str(p_raw).strip()
                if p_text:
                    full_char_len += len(p_text)
                    text_blocks.append(f"Document {orig_rank}. {p_text}")
            elif idx in medium_set:
                m_raw = medium_texts_map.get(idx, "")
                m_text = " ".join(m_raw).strip() if isinstance(m_raw, list) else str(m_raw).strip()
                if not m_text:
                    # Fallback to full text if medium is empty
                    p_raw = passages[idx].get("text", "")
                    m_text = " ".join(p_raw).strip() if isinstance(p_raw, list) else str(p_raw).strip()
                if m_text:
                    medium_char_len += len(m_text)
                    text_blocks.append(f"Document {orig_rank}. {m_text}")

        text_context = "\n\n".join(text_blocks) if text_blocks else "No direct text context provided."
    else:
        text_context = "No direct text context provided."

    # 2. Compressed Context: Unselected (HEAVY) passages mapped to <COMPRESS> tokens.
    # In Hybrid MEDIUM, MEDIUM passages also retain their <COMPRESS> token representation.
    if hybrid_medium:
        compressed_indices = sorted(heavy_set | medium_set)
    else:
        compressed_indices = sorted(heavy_set)

    num_compressed = len(compressed_indices)
    if num_compressed > 0:
        compress_tokens_str = "\n".join([f"{i + 1}. {COMPRESS}" for i in range(num_compressed)])
        additional_context_str = f"## Additional Context (compression tokens)\n{compress_tokens_str}\n---"
    else:
        additional_context_str = ""

    # 3. Assemble User Prompt
    if num_compressed == 0:
        user_prompt = f"""Answer my questions based on the given context.
---
## Context
{text_context}
---
## Your Task
Answer the following question in a succinct manner. Use a single phrase or a short sentence if possible.
Question: {question}
Your Answer:"""
    else:
        user_prompt = f"""Answer my questions based on the given context.
---
## Context
{text_context}
---
{additional_context_str}
## Your Task
Answer the following question in a succinct manner. Use a single phrase or a short sentence if possible.
Question: {question}
Your Answer:"""

    # 4. Tokenize prompt
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": user_prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted_prompt = f"<s>[INST] {user_prompt} [/INST]"

    context_tokens = len(tokenizer.tokenize(user_prompt)) if hasattr(tokenizer, "tokenize") else len(user_prompt.split())

    token_breakdown = {
        "n_full": len(full_indices),
        "n_medium": len(medium_indices),
        "n_heavy": len(heavy_indices),
        "n_compressed": len(compressed_indices),
        "context_tokens": context_tokens,
    }

    return formatted_prompt, context_tokens, compressed_indices, token_breakdown


def extract_mixed_fidelity_sara_embeddings(
    qid_embeddings: torch.Tensor,
    compressed_indices: Sequence[int],
) -> torch.Tensor:
    """Extract cached SFR embeddings corresponding to compressed passages.

    Args:
        qid_embeddings: Tensor of shape (10, 4096) for this question.
        compressed_indices: 0-indexed passage ranks assigned to compression tokens.

    Returns:
        Tensor of shape (len(compressed_indices), 4096) in ascending rank order.
    """
    sorted_comp = sorted(compressed_indices)
    if not sorted_comp:
        return torch.empty((0, qid_embeddings.shape[1]), dtype=qid_embeddings.dtype, device=qid_embeddings.device)
    return qid_embeddings[sorted_comp]
