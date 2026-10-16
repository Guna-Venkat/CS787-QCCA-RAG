"""Multi-Vector MEDIUM (M2) representation and prompt utilities for Week 7-E.

Implements the clean fidelity ladder:
- HEAVY: 1 whole-passage latent vector (e_whole)
- MEDIUM: 1 whole-passage latent vector (e_whole) + 1 focused chunk latent vector (e_focused)
- FULL: natural-language text

Guarantees:
- HEAVY is strictly nested in MEDIUM (HEAVY subset of MEDIUM).
- Zero natural-language surface text is exposed in ## Context for ranks 5-6 (no text distraction).
- For E1-M2 (4F + 2M2 + 4H): exactly 4 full passages in Context, exactly 8 <COMPRESS> tokens in prompt,
  and exactly 8 4096-dim vectors injected into SARA's retrieval_embeds.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

try:
    from src.const import COMPRESS
except ImportError:
    COMPRESS = "<COMPRESS>"

from src.week7.w7e.sentence_splitter import split_into_sentences

logger = logging.getLogger(__name__)


def split_passage_into_sentence_halves(text: str) -> Tuple[str, str]:
    """Split text into two contiguous sentence-aligned chunks near the midpoint.

    Guarantees:
    - Never splits within a sentence, formula, citation, or abbreviation.
    - If passage has 0 sentences, returns ('', '').
    - If passage has 1 sentence, returns (s, s).
    - If passage has >= 2 sentences, splits at sentence boundary minimizing
      difference from character midpoint.
    """
    sents = split_into_sentences(text)
    if not sents:
        return ("", "")
    if len(sents) == 1:
        return (sents[0], sents[0])

    lens = [len(s) for s in sents]
    tot_len = sum(lens)
    mid_target = tot_len / 2.0

    cum = 0
    best_k = 1
    best_diff = float("inf")
    for k in range(1, len(sents)):
        cum += lens[k - 1]
        diff = abs(cum - mid_target)
        if diff < best_diff:
            best_diff = diff
            best_k = k

    chunk1 = " ".join(sents[:best_k]).strip()
    chunk2 = " ".join(sents[best_k:]).strip()
    return (chunk1, chunk2)


def select_focused_chunk(
    query: str,
    chunk1: str,
    chunk2: str,
    semantic_scorer: Any,
) -> str:
    """Select the chunk more relevant to the query using MiniLM semantic scorer."""
    if not chunk1 and not chunk2:
        return ""
    if not chunk1:
        return chunk2
    if not chunk2:
        return chunk1
    if chunk1 == chunk2:
        return chunk1

    scores = semantic_scorer.score_query_passages(query, [chunk1, chunk2]).detach().cpu().numpy()
    if scores[0] >= scores[1]:
        return chunk1
    else:
        return chunk2


def build_m2_sara_prompt(
    question: str,
    passages: List[Dict[str, Any]],
    full_indices: Sequence[int],
    medium_indices: Sequence[int],
    heavy_indices: Sequence[int],
    tokenizer: Any,
) -> Tuple[str, int, List[Tuple[int, str]], Dict[str, int]]:
    """Build SARA prompt for Multi-Vector M2 policy.

    Args:
        question: Query text.
        passages: List of 10 retrieved passage dicts.
        full_indices: Indices for FULL text (e.g. ranks 1-4).
        medium_indices: Indices for MEDIUM multi-vector passages (e.g. ranks 5-6).
        heavy_indices: Indices for HEAVY 1-vector passages (e.g. ranks 7-10).
        tokenizer: SARA generator tokenizer.

    Returns:
        (formatted_prompt, context_tokens, compression_slot_mapping, token_breakdown)
    """
    full_set = set(full_indices)
    medium_set = set(medium_indices)
    heavy_set = set(heavy_indices)

    # 1. Text Context: ONLY FULL passages. Zero text for MEDIUM or HEAVY!
    text_blocks: List[str] = []
    for idx in sorted(full_set):
        orig_rank = idx + 1
        p_text = passages[idx].get("text", "").strip()
        if p_text:
            text_blocks.append(f"Document {orig_rank}. {p_text}")

    text_context = "\n\n".join(text_blocks) if text_blocks else "No direct text context provided."

    # 2. Compressed Context:
    # MEDIUM passages get TWO <COMPRESS> tokens.
    # HEAVY passages get ONE <COMPRESS> token.
    comp_docs = sorted(medium_set | heavy_set)
    compress_lines: List[str] = []
    slot_mapping: List[Tuple[int, str]] = []  # List of (doc_idx, 'whole'|'focused')

    for i, doc_idx in enumerate(comp_docs):
        if doc_idx in medium_set:
            # 2 consecutive <COMPRESS> tokens for this MEDIUM document
            compress_lines.append(f"{i + 1}. {COMPRESS} {COMPRESS}")
            slot_mapping.append((doc_idx, "whole"))
            slot_mapping.append((doc_idx, "focused"))
        else:
            # 1 <COMPRESS> token for this HEAVY document
            compress_lines.append(f"{i + 1}. {COMPRESS}")
            slot_mapping.append((doc_idx, "whole"))

    if compress_lines:
        compress_tokens_str = "\n".join(compress_lines)
        additional_context_str = f"## Additional Context (compression tokens)\n{compress_tokens_str}\n---"
    else:
        additional_context_str = ""

    # 3. Assemble User Prompt
    if not additional_context_str:
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

    context_tokens = len(tokenizer.tokenize(user_prompt)) if hasattr(tokenizer, "tokenize") else len(user_prompt.split())

    token_breakdown = {
        "n_full": len(full_indices),
        "n_medium": len(medium_indices),
        "n_heavy": len(heavy_indices),
        "n_compress_tokens": len(slot_mapping),
        "context_tokens": context_tokens,
    }

    return formatted_prompt, context_tokens, slot_mapping, token_breakdown


def extract_m2_sara_embeddings(
    whole_sfr_embeddings: torch.Tensor,
    focused_sfr_dict: Dict[int, torch.Tensor],
    slot_mapping: List[Tuple[int, str]],
) -> torch.Tensor:
    """Extract and assemble retrieval embeddings matching slot_mapping.

    Args:
        whole_sfr_embeddings: Tensor of shape (10, 4096) for this question.
        focused_sfr_dict: Dict mapping passage_idx -> focused chunk embedding (4096,).
        slot_mapping: List of (doc_idx, 'whole' | 'focused') specifying the sequential order.

    Returns:
        Tensor of shape (len(slot_mapping), 4096).
    """
    embed_list: List[torch.Tensor] = []
    device = whole_sfr_embeddings.device
    dtype = whole_sfr_embeddings.dtype

    for doc_idx, vector_type in slot_mapping:
        if vector_type == "whole":
            vec = whole_sfr_embeddings[doc_idx]
            embed_list.append(vec.unsqueeze(0))
        elif vector_type == "focused":
            f_vec = focused_sfr_dict.get(doc_idx)
            if f_vec is not None:
                if isinstance(f_vec, np.ndarray):
                    f_vec = torch.from_numpy(f_vec)
                f_vec = f_vec.to(device=device, dtype=dtype)
                if f_vec.dim() == 1:
                    f_vec = f_vec.unsqueeze(0)
                embed_list.append(f_vec)
            else:
                # Fallback to whole vector if focused is missing
                vec = whole_sfr_embeddings[doc_idx]
                embed_list.append(vec.unsqueeze(0))

    if not embed_list:
        return torch.empty((0, whole_sfr_embeddings.shape[1]), device=device, dtype=dtype)

    return torch.cat(embed_list, dim=0)


def assert_m2_row_invariants(
    row: Dict[str, Any],
    n_full_expected: int = 4,
    n_medium_expected: int = 2,
    n_heavy_expected: int = 4,
    n_compress_tokens_expected: int = 8,
) -> None:
    """Strictly verify representation invariants for M2 rows."""
    prompt = row.get("prompt", "")
    assert COMPRESS in prompt, "Missing <COMPRESS> tokens in prompt"
    count_compress = prompt.count(COMPRESS)
    assert count_compress == n_compress_tokens_expected, (
        f"Expected {n_compress_tokens_expected} <COMPRESS> tokens in prompt, got {count_compress}"
    )

    tb = row.get("token_breakdown", {})
    assert tb.get("n_full") == n_full_expected, f"Expected {n_full_expected} full passages, got {tb.get('n_full')}"
    assert tb.get("n_medium") == n_medium_expected, f"Expected {n_medium_expected} medium passages, got {tb.get('n_medium')}"
    assert tb.get("n_heavy") == n_heavy_expected, f"Expected {n_heavy_expected} heavy passages, got {tb.get('n_heavy')}"
    assert tb.get("n_compress_tokens") == n_compress_tokens_expected, (
        f"Expected {n_compress_tokens_expected} compression slots, got {tb.get('n_compress_tokens')}"
    )
