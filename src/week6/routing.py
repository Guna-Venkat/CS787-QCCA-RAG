"""Passage fidelity routing policies, invariants validation, and SARA prompt/embedding construction for Week 6.

Key Policies:
- R0: BM25 prefix baseline (ranks 1..k full text, ranks k+1..10 compressed).
- R1: Pretrained semantic top-k (MiniLM query-passage relevance, sorted into original BM25 rank order).
- R2: Learned importance top-k (R2 model logits, sorted into original BM25 rank order).

Guarantees:
- Exactly k passages are rendered in high-fidelity text.
- Exactly 10 - k passages are compressed into <COMPRESS> tokens with cached SFR embeddings.
- High-fidelity passages preserve deterministic original BM25 order in the prompt.
- Total pool of 10 retrieved passages is strictly preserved; no passages dropped or reordered externally.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch

try:
    from src.const import COMPRESS
except ImportError:
    COMPRESS = "<COMPRESS>"

logger = logging.getLogger(__name__)


def route_r0_bm25_prefix(k: int, n_passages: int = 10) -> List[int]:
    """R0 Baseline: Automatically retain the first k BM25 retrieved passages."""
    assert 0 <= k <= n_passages, f"Budget k={k} must be in [0, {n_passages}]"
    return list(range(k))


def route_r1_semantic_topk(scores: Union[np.ndarray, torch.Tensor, Sequence[float]], k: int) -> List[int]:
    """R1: Select top-k passages by pretrained semantic relevance scores.

    Presentation order preserves original BM25 retrieval rank order!
    """
    if isinstance(scores, torch.Tensor):
        s_arr = scores.detach().cpu().numpy()
    else:
        s_arr = np.asarray(scores, dtype=np.float32)

    n_passages = len(s_arr)
    assert 0 <= k <= n_passages, f"Budget k={k} must be in [0, {n_passages}]"
    if k == 0:
        return []
    if k == n_passages:
        return list(range(n_passages))

    # Top-k selection by descending score
    topk_desc = np.argsort(-s_arr)[:k]
    # Presentation order preserves original BM25 rank order
    selected_indices = sorted(int(idx) for idx in topk_desc)
    return selected_indices


def route_r2_learned_topk(logits: Union[np.ndarray, torch.Tensor, Sequence[float]], k: int) -> List[int]:
    """R2: Select top-k passages by learned router logits.

    Presentation order preserves original BM25 retrieval rank order!
    """
    if isinstance(logits, torch.Tensor):
        l_arr = logits.detach().cpu().numpy()
    else:
        l_arr = np.asarray(logits, dtype=np.float32)

    n_passages = len(l_arr)
    assert 0 <= k <= n_passages, f"Budget k={k} must be in [0, {n_passages}]"
    if k == 0:
        return []
    if k == n_passages:
        return list(range(n_passages))

    # Top-k selection by descending logit
    topk_desc = np.argsort(-l_arr)[:k]
    # Presentation order preserves original BM25 rank order
    selected_indices = sorted(int(idx) for idx in topk_desc)
    return selected_indices


def validate_routing_assignment(
    selected_indices: List[int],
    k: int,
    total_passages: int = 10,
) -> None:
    """Strictly assert Week 6 routing invariants:
    - exactly k passages selected
    - all selected indices distinct and in [0, total_passages - 1]
    - exactly total_passages - k unselected passages
    - union of selected and unselected is exactly 0..total_passages - 1
    """
    if len(selected_indices) != k:
        raise ValueError(f"Invariant violation: expected {k} selected passages, got {len(selected_indices)}")

    if len(set(selected_indices)) != k:
        raise ValueError(f"Invariant violation: duplicate passage indices detected in {selected_indices}")

    for idx in selected_indices:
        if not (0 <= idx < total_passages):
            raise ValueError(f"Invariant violation: index {idx} out of range [0, {total_passages - 1}]")

    unselected_indices = [i for i in range(total_passages) if i not in set(selected_indices)]
    if len(unselected_indices) != (total_passages - k):
        raise ValueError(
            f"Invariant violation: expected {total_passages - k} unselected passages, got {len(unselected_indices)}"
        )

    full_union = sorted(set(selected_indices) | set(unselected_indices))
    if full_union != list(range(total_passages)):
        raise ValueError(f"Invariant violation: union of selected and unselected does not span 0..{total_passages - 1}")


def build_routed_sara_prompt(
    question: str,
    passages: List[Dict[str, Any]],
    selected_indices: List[int],
    tokenizer: Any,
) -> Tuple[str, int, List[int]]:
    """Format SARA prompt injecting selected passages as full text and unselected as <COMPRESS>.

    Args:
        question: Query text.
        passages: Top-10 retrieved passage dicts.
        selected_indices: 0-indexed passage ranks chosen for high fidelity.
        tokenizer: SARA Mistral tokenizer.

    Returns:
        (formatted_prompt, context_tokens, unselected_indices)
    """
    validate_routing_assignment(selected_indices, k=len(selected_indices), total_passages=len(passages))

    # Ensure deterministic ascending BM25 rank order
    sorted_selected = sorted(selected_indices)
    unselected_indices = [i for i in range(len(passages)) if i not in set(sorted_selected)]

    # 1. Text Context: Only selected passages, labeled by their original BM25 rank
    if sorted_selected:
        text_blocks = []
        for idx in sorted_selected:
            orig_rank = idx + 1
            p_text = passages[idx].get("text", "").strip()
            if p_text:
                text_blocks.append(f"Document {orig_rank}. {p_text}")
        text_context = "\n\n".join(text_blocks) if text_blocks else "No direct text context provided."
    else:
        text_context = "No direct text context provided."

    # 2. Compressed Context: Unselected passages mapped to <COMPRESS> tokens
    num_compressed = len(unselected_indices)
    if num_compressed > 0:
        compress_tokens_str = "\n".join([f"{idx + 1}. {COMPRESS}" for idx in range(num_compressed)])
        additional_context_str = f"## Additional Context (compression tokens)\n{compress_tokens_str}\n---"
    else:
        additional_context_str = ""

    if len(sorted_selected) == 10:
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

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": user_prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted_prompt = f"<s>[INST] {user_prompt} [/INST]"

    context_tokens = len(tokenizer.tokenize(user_prompt))
    return formatted_prompt, context_tokens, unselected_indices


def extract_routed_sara_embeddings(
    qid_embeddings: torch.Tensor,
    selected_indices: List[int],
) -> torch.Tensor:
    """Slice cached SFR embeddings tensor for the unselected passages.

    Args:
        qid_embeddings: Tensor of shape (10, 4096) for this question.
        selected_indices: 0-indexed passage ranks selected for full text.

    Returns:
        Tensor of shape (10 - k, 4096) in ascending rank order.
    """
    unselected_indices = [i for i in range(qid_embeddings.shape[0]) if i not in set(selected_indices)]
    if not unselected_indices:
        return torch.empty((0, qid_embeddings.shape[1]), dtype=qid_embeddings.dtype, device=qid_embeddings.device)
    return qid_embeddings[unselected_indices]
