"""Mistral Layer and Representation Evolution Analyzer.

Extracts hidden states across transformer layers (embedding, early, middle, late, final),
tracks query <-> passage cosine similarity, computes passage separation geometry,
and compares evidence vs. non-evidence representations during RAG prompt processing.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch

from src.week0.dataset_adapter import DatasetAdapter
from src.week0.dataset_analyzer import check_passage_matches_evidence

logger = logging.getLogger(__name__)


def select_controlled_sample(
    records: List[Dict[str, Any]],
    adapter: DatasetAdapter,
    sample_size: int = 40,
    seed: int = 42,
    stratify_col: str = "question_category",
) -> List[Dict[str, Any]]:
    """Deterministically sample a controlled subset of dev questions stratified by category."""
    rng = np.random.RandomState(seed)
    
    # Categorize records
    categorized: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        meta = adapter.get_metadata(r)
        cat = str(meta.get(stratify_col, meta.get("question_type", "other"))).lower()
        if cat not in categorized:
            categorized[cat] = []
        categorized[cat].append(r)

    total_records = len(records)
    selected: List[Dict[str, Any]] = []

    # Proportional allocation
    for cat, cat_records in sorted(categorized.items()):
        cat_count = len(cat_records)
        target_k = max(1, int(round((cat_count / total_records) * sample_size)))
        indices = rng.choice(cat_count, size=min(target_k, cat_count), replace=False)
        for idx in indices:
            selected.append(cat_records[idx])

    # Trim or supplement to exact sample_size
    if len(selected) > sample_size:
        selected = selected[:sample_size]
    elif len(selected) < sample_size:
        remaining = [r for r in records if r not in selected]
        needed = sample_size - len(selected)
        if remaining and needed > 0:
            extra = rng.choice(len(remaining), size=min(needed, len(remaining)), replace=False)
            for idx in extra:
                selected.append(remaining[idx])

    logger.info(f"Selected controlled sample of {len(selected)} records across categories: {list(categorized.keys())}")
    return selected


def locate_token_spans(
    prompt: str,
    question: str,
    passages: List[Dict[str, Any]],
    tokenizer: Any,
) -> Dict[str, Tuple[int, int]]:
    """Map token index boundaries for the query and each retrieved passage in the prompt."""
    encoding = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    offset_mapping = encoding.get("offset_mapping", None)
    
    spans: Dict[str, Tuple[int, int]] = {}
    total_tokens = len(encoding["input_ids"])

    if offset_mapping is not None:
        # Character-level matching
        # 1. Query span
        q_pos = prompt.find(question)
        if q_pos >= 0:
            q_end = q_pos + len(question)
            token_start = None
            token_end = None
            for idx, (s, e) in enumerate(offset_mapping):
                if s < q_end and e > q_pos:
                    if token_start is None:
                        token_start = idx
                    token_end = idx + 1
            if token_start is not None and token_end is not None:
                spans["query"] = (token_start, token_end)

        # 2. Passages spans
        for p_idx, p in enumerate(passages):
            p_text = p.get("text", "").strip()
            # Match passage prefix
            sub_text = p_text[:60] if len(p_text) >= 60 else p_text
            p_pos = prompt.find(sub_text)
            if p_pos >= 0:
                p_end = p_pos + len(p_text)
                token_start = None
                token_end = None
                for idx, (s, e) in enumerate(offset_mapping):
                    if s < p_end and e > p_pos:
                        if token_start is None:
                            token_start = idx
                        token_end = idx + 1
                if token_start is not None and token_end is not None:
                    spans[f"passage_{p_idx}"] = (token_start, token_end)

    # Fallback heuristic if offset mapping is not available or partial
    if "query" not in spans:
        # Approximate: query is at the end of the prompt
        q_len = len(tokenizer.tokenize(question))
        spans["query"] = (max(0, total_tokens - q_len - 10), max(1, total_tokens - 10))

    for p_idx in range(len(passages)):
        if f"passage_{p_idx}" not in spans:
            # Fallback partition
            approx_span_len = max(5, int(total_tokens * 0.75 / max(len(passages), 1)))
            start_tok = 20 + p_idx * approx_span_len
            spans[f"passage_{p_idx}"] = (min(start_tok, total_tokens - 2), min(start_tok + approx_span_len, total_tokens - 1))

    return spans


def extract_layer_representations(
    model: Any,
    tokenizer: Any,
    prompt: str,
    spans: Dict[str, Tuple[int, int]],
    selected_layers: Sequence[int] = (0, 4, 8, 16, 24, 31),
    device: str = "cuda",
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Extract mean-pooled, L2-normalized representations for query and passages at each layer."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # outputs.hidden_states is a tuple of length (num_layers + 1): index 0 is embedding
        hidden_states = outputs.hidden_states

    layer_reps: Dict[int, Dict[str, torch.Tensor]] = {}

    for layer_idx in selected_layers:
        if layer_idx >= len(hidden_states):
            continue
        hs = hidden_states[layer_idx][0]  # shape: [seq_len, hidden_dim]

        layer_reps[layer_idx] = {}
        for span_name, (st, en) in spans.items():
            st_clamped = max(0, min(st, hs.shape[0] - 1))
            en_clamped = max(st_clamped + 1, min(en, hs.shape[0]))
            span_hs = hs[st_clamped:en_clamped]  # [span_len, hidden_dim]

            # Mean-pool along sequence dimension
            mean_vec = torch.mean(span_hs, dim=0)  # [hidden_dim]
            # L2-normalize
            norm_vec = mean_vec / (torch.norm(mean_vec, p=2) + 1e-8)
            layer_reps[layer_idx][span_name] = norm_vec.cpu()

    return layer_reps


def compute_layer_similarity_metrics(
    layer_reps: Dict[int, Dict[str, torch.Tensor]],
    passages: List[Dict[str, Any]],
    gold_evidence: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Compute cosine similarities between query, passages, and evaluate evidence separation."""
    # Extract flat gold evidence for matching
    flat_ev_paras: List[str] = []
    flat_highlighted: List[str] = []
    for entry in gold_evidence:
        flat_ev_paras.extend(entry.get("evidence_paragraphs", []))
        flat_highlighted.extend(entry.get("highlighted_spans", []))

    long_rows: List[Dict[str, Any]] = []
    summary_by_layer: Dict[str, Any] = {}

    # Initial layer representation for drift computation
    layers = sorted(layer_reps.keys())
    l0 = layers[0] if layers else 0
    q_vec_0 = layer_reps.get(l0, {}).get("query", None)

    for layer_idx in layers:
        reps = layer_reps[layer_idx]
        q_vec = reps.get("query", None)
        if q_vec is None:
            continue

        passage_sims: List[float] = []
        evidence_sims: List[float] = []
        nonevidence_sims: List[float] = []
        passage_vectors: List[torch.Tensor] = []

        # Query drift from initial layer
        q_drift = float(torch.dot(q_vec, q_vec_0)) if q_vec_0 is not None else 1.0

        for p_idx, p in enumerate(passages):
            p_name = f"passage_{p_idx}"
            p_vec = reps.get(p_name, None)
            if p_vec is None:
                continue

            sim_qp = float(torch.dot(q_vec, p_vec))
            passage_sims.append(sim_qp)
            passage_vectors.append(p_vec)

            is_evidence = check_passage_matches_evidence(
                p.get("text", ""), flat_ev_paras, flat_highlighted
            )
            if is_evidence:
                evidence_sims.append(sim_qp)
            else:
                nonevidence_sims.append(sim_qp)

            # Passage drift from initial layer
            p_vec_0 = layer_reps.get(l0, {}).get(p_name, None)
            p_drift = float(torch.dot(p_vec, p_vec_0)) if p_vec_0 is not None else 1.0

            long_rows.append({
                "layer": layer_idx,
                "passage_rank": p_idx + 1,
                "passage_id": p.get("passage_id", f"p_{p_idx}"),
                "bm25_score": round(float(p.get("score", 0.0)), 4),
                "is_gold_evidence": is_evidence,
                "cosine_sim_with_query": round(sim_qp, 4),
                "query_drift": round(q_drift, 4),
                "passage_drift": round(p_drift, 4),
            })

        # Inter-passage separation (pairwise cosine similarity among passages)
        inter_sims: List[float] = []
        for i in range(len(passage_vectors)):
            for j in range(i + 1, len(passage_vectors)):
                sim_ij = float(torch.dot(passage_vectors[i], passage_vectors[j]))
                inter_sims.append(sim_ij)

        mean_inter_sim = float(np.mean(inter_sims)) if inter_sims else 1.0
        mean_ev_sim = float(np.mean(evidence_sims)) if evidence_sims else np.nan
        mean_nonev_sim = float(np.mean(nonevidence_sims)) if nonevidence_sims else np.nan
        ev_gap = mean_ev_sim - mean_nonev_sim if not (np.isnan(mean_ev_sim) or np.isnan(mean_nonev_sim)) else np.nan

        summary_by_layer[f"layer_{layer_idx}_mean_sim"] = round(float(np.mean(passage_sims)), 4) if passage_sims else 0.0
        summary_by_layer[f"layer_{layer_idx}_inter_passage_sim"] = round(mean_inter_sim, 4)
        summary_by_layer[f"layer_{layer_idx}_evidence_sim"] = round(mean_ev_sim, 4) if not np.isnan(mean_ev_sim) else None
        summary_by_layer[f"layer_{layer_idx}_nonevidence_sim"] = round(mean_nonev_sim, 4) if not np.isnan(mean_nonev_sim) else None
        summary_by_layer[f"layer_{layer_idx}_evidence_gap"] = round(ev_gap, 4) if not np.isnan(ev_gap) else None

    return long_rows, summary_by_layer


def run_mock_layer_analysis(
    sample_records: List[Dict[str, Any]],
    adapter: DatasetAdapter,
    selected_layers: Sequence[int] = (0, 4, 8, 16, 24, 31),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Lightweight deterministic simulation for tests and CPU verification without loading 7B LM."""
    rng = np.random.RandomState(42)
    long_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for r in sample_records:
        meta = adapter.get_metadata(r)
        qid = meta["question_id"]
        pid = meta["paper_id"]
        passages = adapter.get_retrieved_passages(r)
        gold_ev = adapter.get_gold_evidence(r)
        if not passages:
            continue

        flat_ev = [p for e in gold_ev for p in e.get("evidence_paragraphs", [])]
        flat_hl = [s for e in gold_ev for s in e.get("highlighted_spans", [])]

        summary_entry: Dict[str, Any] = {"question_id": qid, "paper_id": pid}

        for layer in selected_layers:
            # Synthetic layer evolution: similarity increases slightly in middle layers, gap widens
            layer_frac = layer / 31.0
            p_sims = []
            ev_sims = []
            nonev_sims = []

            for p_idx, p in enumerate(passages):
                is_ev = check_passage_matches_evidence(p.get("text", ""), flat_ev, flat_hl)
                base_sim = 0.35 + 0.1 * math.sin(layer_frac * math.pi)
                ev_boost = 0.12 * layer_frac if is_ev else 0.0
                noise = rng.normal(0, 0.02)
                sim = float(np.clip(base_sim + ev_boost + noise, -1.0, 1.0))
                p_sims.append(sim)
                if is_ev:
                    ev_sims.append(sim)
                else:
                    nonev_sims.append(sim)

                long_rows.append({
                    "question_id": qid,
                    "paper_id": pid,
                    "layer": layer,
                    "passage_rank": p_idx + 1,
                    "passage_id": p.get("passage_id", f"p_{p_idx}"),
                    "bm25_score": round(float(p.get("score", 0.0)), 4),
                    "is_gold_evidence": is_ev,
                    "cosine_sim_with_query": round(sim, 4),
                    "query_drift": round(1.0 - 0.2 * layer_frac, 4),
                    "passage_drift": round(1.0 - 0.25 * layer_frac, 4),
                })

            summary_entry[f"layer_{layer}_mean_sim"] = round(float(np.mean(p_sims)), 4)
            summary_entry[f"layer_{layer}_evidence_gap"] = round(float(np.mean(ev_sims) - np.mean(nonev_sims)), 4) if ev_sims and nonev_sims else 0.0

        summary_rows.append(summary_entry)

    return pd.DataFrame(long_rows), pd.DataFrame(summary_rows)
