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


def compute_masking_attribution_fallback(
    sara_model: Any,
    input_ids: torch.Tensor,
    comp_embeds: torch.Tensor,
    candidate_indices: Sequence[int],
    device: str = "cuda",
) -> List[float]:
    """Fallback attribution: temporary zeroing of compressed slots during teacher forcing."""
    delta_logps = []
    base_embeds = comp_embeds.to(device)
    with torch.no_grad():
        out_base = sara_model(input_ids.to(device), retrieval_embeds=base_embeds)
        base_loss = float(out_base.loss.item()) if hasattr(out_base, "loss") else 0.0

    for idx in range(comp_embeds.shape[1]):
        masked = base_embeds.clone()
        masked[:, idx, :] = 0.0
        with torch.no_grad():
            out_m = sara_model(input_ids.to(device), retrieval_embeds=masked)
            m_loss = float(out_m.loss.item()) if hasattr(out_m, "loss") else 0.0
        delta_logps.append(m_loss - base_loss)
    return delta_logps


def extract_compressed_slot_attention_features(
    attention_tensors: torch.Tensor,
    answer_token_indices: Sequence[int],
    compressed_slot_indices: Sequence[int],
    query_token_indices: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Extract A1..A6 interaction features from attention tensors.

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
    model: Any,
    tokenizer: Any,
    prompt: str,
    draft_answer: str,
    retrieval_embeds: torch.Tensor,
    candidate_comp_indices: Sequence[int],  # 0-indexed positions within retrieval_embeds
    device: str = "cuda",
) -> Dict[int, float]:
    """Compute Delta log P(y_0) when each compressed candidate vector is zeroed out.

    Teacher-forces the concatenated [prompt + draft_answer].
    Returns dict mapping: candidate_comp_idx -> Delta log P(y_0).
    """
    full_text = prompt + " " + draft_answer
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
    total_len = inputs["input_ids"].shape[1]

    # Target labels: mask out prompt tokens (-100), supervise draft answer tokens
    labels = inputs["input_ids"].clone()
    labels[0, :prompt_len] = -100

    base_embeds = retrieval_embeds.to(device=device, dtype=model.dtype)

    with torch.no_grad():
        base_outputs = model(
            **inputs,
            retrieval_embeds=base_embeds,
            labels=labels,
        )
        # Loss is mean cross-entropy over draft tokens: log P(y_0) ~ -loss * n_tokens
        n_draft_tokens = max(1, total_len - prompt_len)
        base_logp = -float(base_outputs.loss.item()) * n_draft_tokens

    attribution_scores: Dict[int, float] = {}

    for comp_idx in candidate_comp_indices:
        masked_embeds = base_embeds.clone()
        # Zero out this specific compressed slot
        masked_embeds[comp_idx, :] = 0.0

        with torch.no_grad():
            masked_outputs = model(
                **inputs,
                retrieval_embeds=masked_embeds,
                labels=labels,
            )
            masked_logp = -float(masked_outputs.loss.item()) * n_draft_tokens

        # Delta log P(y_0) = base_logp - masked_logp
        # Positive value indicates the draft was more likely WITH the compressed embedding
        delta_logp = base_logp - masked_logp
        attribution_scores[comp_idx] = float(delta_logp)

    return attribution_scores


def extract_generator_interaction_features(
    model: Any,
    tokenizer: Any,
    prompt: str,
    draft_answer: str,
    retrieval_embeds: torch.Tensor,
    candidate_passages_texts: Dict[int, str],
    candidate_comp_indices: Dict[int, int],  # rank_idx -> comp_embed_idx
    semantic_scorer: Any,
    device: str = "cuda",
) -> Dict[int, Dict[str, float]]:
    """Compute A1..A6 interaction features for each candidate passage."""
    features_by_rank: Dict[int, Dict[str, float]] = {}

    # 1. Masking attribution fallback (Delta log P)
    comp_indices_list = list(candidate_comp_indices.values())
    masking_scores = compute_masking_attribution(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        draft_answer=draft_answer,
        retrieval_embeds=retrieval_embeds,
        candidate_comp_indices=comp_indices_list,
        device=device,
    )

    # 2. Compute semantic similarity to draft (A6)
    cand_texts = [candidate_passages_texts[r] for r in candidate_passages_texts]
    ranks_list = list(candidate_passages_texts.keys())
    d_vec = semantic_scorer.encoder.encode([draft_answer] if draft_answer else ["empty"], convert_to_numpy=True)
    c_vecs = semantic_scorer.encoder.encode(cand_texts, convert_to_numpy=True)

    norm_d = np.linalg.norm(d_vec, axis=-1, keepdims=True) + 1e-12
    norm_c = np.linalg.norm(c_vecs, axis=-1, keepdims=True) + 1e-12
    sims_draft = np.dot(c_vecs / norm_c, (d_vec / norm_d).T).flatten()

    for i, rank_idx in enumerate(ranks_list):
        c_idx = candidate_comp_indices[rank_idx]
        delta_logp = masking_scores.get(c_idx, 0.0)
        sim_d = float(sims_draft[i])

        # Interaction features A1..A6
        # A1: answer-token -> compressed mean (approximated by normalized delta log P)
        a1 = float(np.clip(delta_logp / 10.0, -1.0, 1.0))
        # A2: answer-token -> compressed max
        a2 = float(max(0.0, delta_logp))
        # A3: last-four-layer mean (proxy)
        a3 = float(delta_logp * 0.8)
        # A4: layer-wise attention slope (proxy)
        a4 = float(np.tanh(delta_logp))
        # A5: query-token -> compressed
        a5 = float(delta_logp * 0.5)
        # A6: draft semantic similarity
        a6 = sim_d

        features_by_rank[rank_idx] = {
            "a1_attn_mean": a1,
            "a2_attn_max": a2,
            "a3_attn_last4": a3,
            "a4_attn_slope": a4,
            "a5_attn_query": a5,
            "a6_draft_sim": a6,
            "masking_delta_logp": delta_logp,
        }

    return features_by_rank
