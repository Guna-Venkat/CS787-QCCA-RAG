"""W8-D: Generator Interaction Features (A1..A6) and Masking Attribution Fallback.

Implements:
- A1: answer-token -> compressed-slot attention mean
- A2: answer-token -> compressed-slot attention max
- A3: last-four-layer mean attention
- A4: layer-wise attention slope across layers
- A5: query-token -> compressed-slot attention
- A6: draft semantic similarity to candidate
- Masking Attribution Fallback: Delta log P(y_0) when compressed slot i is zeroed out during teacher-forcing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _masking_scores(model, input_ids, embeds, candidates, labels, device):
    """Mean draft log-probability change, with explicit causal target masking."""
    if labels is None:
        raise ValueError('Explicit draft-only labels are required for masking attribution')
    if getattr(model, 'training', False):
        raise ValueError('Masking attribution requires model.eval()')
    ids, labels = input_ids.to(device), labels.to(device)
    if ids.shape != labels.shape or ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError('Expected one input sequence and matching draft-only labels')
    if not int((labels[:, 1:] != -100).sum()):
        raise ValueError('No draft target tokens to score')
    base = embeds.to(device=device, dtype=getattr(model, 'dtype', embeds.dtype))
    if base.ndim == 3 and base.shape[0] == 1:
        base = base[0]
    if base.ndim != 2:
        raise ValueError('Expected flat [compressed_slots, embedding_dim] SARA embeddings')
    candidates = list(candidates)
    if len(set(candidates)) != len(candidates) or any(i < 0 or i >= len(base) for i in candidates):
        raise ValueError('Candidate compressed-slot indices are invalid')
    kwargs = dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False)
    with torch.inference_mode():
        output = model(**kwargs, retrieval_embeds=base)
        base_loss = float(output.loss.item())
        del output
        result = {}
        for i in candidates:
            masked = base.clone()
            masked[i, :] = 0
            output = model(**kwargs, retrieval_embeds=masked)
            result[i] = float(output.loss.item()) - base_loss
            del output
    if not np.isfinite([base_loss, *result.values()]).all():
        raise ValueError('Non-finite draft likelihood during masking attribution')
    return result


def compute_masking_attribution_fallback(
    sara_model, input_ids, comp_embeds, candidate_indices, device='cuda', *, labels=None,
):
    """Return mean log-probability changes for requested compressed-slot positions.

    Explicit draft-only labels are mandatory; passage indices must first be
    mapped to positions in the compressed embedding tensor.
    """
    scores = _masking_scores(sara_model, input_ids, comp_embeds,
                             candidate_indices, labels, device)
    return [scores[i] for i in candidate_indices]


def build_draft_teacher_forcing_inputs(tokenizer, prompt, draft_answer, device='cpu'):
    """Keep the generation prompt IDs exact and append draft targets separately.

    This avoids moving the prompt/draft boundary through whole-string
    retokenization and introduces no second BOS at the answer boundary.
    """
    if not str(draft_answer).strip():
        raise ValueError('An empty draft has no attribution target tokens')
    prompt_ids = tokenizer(prompt, return_tensors='pt')['input_ids']
    draft_ids = tokenizer(' ' + str(draft_answer), add_special_tokens=False,
                          return_tensors='pt')['input_ids']
    if not prompt_ids.shape[1] or not draft_ids.shape[1]:
        raise ValueError('Missing prompt or draft tokens')
    ids = torch.cat([prompt_ids, draft_ids], dim=1).to(device)
    labels = ids.clone()
    labels[:, :prompt_ids.shape[1]] = -100
    return ids, labels, int(draft_ids.shape[1])


def extract_compressed_slot_attention_features(
    attention_tensors: torch.Tensor,
    answer_token_indices: Sequence[int],
    compressed_slot_indices: Sequence[int],
    query_token_indices: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Extract A1..A5 attention measurements; column A6 remains zero (not measured).

    Args:
        attention_tensors: [num_layers, num_heads, seq_len, seq_len]
        answer_token_indices: Token positions of generated answer.
        compressed_slot_indices: Token positions of <COMPRESS> slots.
        query_token_indices: Token positions of question tokens.

    Returns:
        Tensor of shape [len(compressed_slot_indices), 6]
    """
    num_slots = len(compressed_slot_indices)
    feats = torch.zeros((num_slots, 6), dtype=torch.float32)

    if attention_tensors is None or attention_tensors.numel() == 0:
        return feats

    # attention_tensors: [L, H, Q, K]
    L = attention_tensors.shape[0]
    for s_idx, slot_pos in enumerate(compressed_slot_indices):
        if slot_pos >= attention_tensors.shape[-1]:
            continue
        # Answer token attention to this slot
        if answer_token_indices and max(answer_token_indices) < attention_tensors.shape[-2]:
            ans_attn = attention_tensors[:, :, answer_token_indices, slot_pos]  # [L, H, num_ans]
            a1_mean = float(ans_attn.mean().item())
            a2_max = float(ans_attn.max().item())
            a3_last4 = float(ans_attn[-min(4, L):].mean().item())
            if L >= 2:
                first_half = ans_attn[: L // 2].mean().item()
                second_half = ans_attn[L // 2 :].mean().item()
                a4_slope = float(second_half - first_half)
            else:
                a4_slope = 0.0
        else:
            a1_mean, a2_max, a3_last4, a4_slope = 0.0, 0.0, 0.0, 0.0

        if query_token_indices and max(query_token_indices) < attention_tensors.shape[-2]:
            q_attn = attention_tensors[:, :, query_token_indices, slot_pos]
            a5_query = float(q_attn.mean().item())
        else:
            a5_query = 0.0

        a6_sim = 0.0
        feats[s_idx] = torch.tensor([a1_mean, a2_max, a3_last4, a4_slope, a5_query, a6_sim])

    return feats


def compute_masking_attribution(
    model, tokenizer, prompt, draft_answer, retrieval_embeds,
    candidate_comp_indices, device='cuda',
):
    """Sum log P(draft|base) minus log P(draft|base with slot zeroed).

    Positive values indicate support for the existing draft, not promotion
    utility or correctness. No gold answer is used.
    """
    ids, labels, count = build_draft_teacher_forcing_inputs(
        tokenizer, prompt, draft_answer, device)
    scores = _masking_scores(model, ids, retrieval_embeds,
                             candidate_comp_indices, labels, device)
    return {i: value * count for i, value in scores.items()}


def extract_generator_interaction_features(
    model, tokenizer, prompt, draft_answer, retrieval_embeds,
    candidate_passages_texts, candidate_comp_indices, semantic_scorer, device='cuda',
):
    """Measured masking fallback plus draft cosine similarity.

    A1..A5 attention values are NOT measured by this fallback and are omitted.
    Empty drafts are explicitly marked unavailable, rather than scored as zeros.
    """
    if set(candidate_passages_texts) != set(candidate_comp_indices):
        raise ValueError('Candidate text and compressed-slot mappings disagree')
    ranks = sorted(candidate_passages_texts)
    if not ranks:
        return {}
    available = bool(str(draft_answer).strip())
    scores, count = {}, 0
    if available:
        ids, labels, count = build_draft_teacher_forcing_inputs(
            tokenizer, prompt, draft_answer, device)
        scores = _masking_scores(model, ids, retrieval_embeds,
                                 [candidate_comp_indices[r] for r in ranks], labels, device)
    from src.week8.marginal_information import compute_cosine_similarity_matrix
    dv = semantic_scorer.encoder.encode([draft_answer] if available else ['empty'], convert_to_numpy=True)
    cv = semantic_scorer.encoder.encode([candidate_passages_texts[r] for r in ranks], convert_to_numpy=True)
    similarities = compute_cosine_similarity_matrix(cv, dv).flatten()
    result = {}
    for j, rank in enumerate(ranks):
        mean = scores[candidate_comp_indices[rank]] if available else float('nan')
        result[rank] = {
            'masking_delta_logp': mean * count if available else float('nan'),
            'masking_delta_logp_per_token': mean,
            'masking_target_tokens': count,
            'masking_available': int(available),
            'a6_draft_sim': float(similarities[j]),
        }
    return result
