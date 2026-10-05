"""Retrieved Passage Utilization Diagnostics for RAG / SARA.

Implements causal attention-mass extraction, teacher-forced leave-one-out passage ablation,
utilization concentration metrics (entropy, Gini, N_eff, top-k shares), and multi-signal correlations.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import torch

from src.week0.dataset_adapter import DatasetAdapter
from src.week0.dataset_analyzer import check_passage_matches_evidence

logger = logging.getLogger(__name__)


def compute_concentration_metrics(
    shares: Sequence[float],
    mode: str = "positive",
) -> Dict[str, float]:
    """Calculate normalized entropy, Gini coefficient, effective count, and top-k shares.

    Supports signed input scores (e.g. leave-one-out likelihood differences Δ_i):
      - 'positive': c_i = max(Δ_i, 0). Quantifies concentration of beneficial evidence.
                    If all Δ_i <= 0, returns zeros (no beneficial evidence contributed).
      - 'absolute': c_i = |Δ_i|. Quantifies model sensitivity/responsiveness regardless of sign.
      - 'direct': assumes non-negative values (e.g. attention mass).

    Args:
        shares: Sequence of passage scores (attention mass, or signed ablation Δ_i).
        mode: Conversion mode ('positive', 'absolute', or 'direct').
    """
    raw_arr = np.array(shares, dtype=np.float64)
    n = len(raw_arr)
    if n == 0:
        return {
            "entropy_norm": 0.0,
            "effective_n_passages": 0.0,
            "gini_coefficient": 0.0,
            "top1_share": 0.0,
            "top3_share": 0.0,
        }

    if mode == "positive":
        c = np.maximum(raw_arr, 0.0)
    elif mode == "absolute":
        c = np.abs(raw_arr)
    elif mode == "direct":
        c = np.maximum(raw_arr, 0.0)
    else:
        raise ValueError(f"Unknown concentration mode: {mode}. Choose 'positive', 'absolute', or 'direct'.")

    total = float(np.sum(c))
    if total <= 1e-12:
        # Edge case: No positive evidence contributed (or zero sensitivity across all passages)
        return {
            "entropy_norm": 0.0,
            "effective_n_passages": 0.0,
            "gini_coefficient": 0.0,
            "top1_share": 0.0,
            "top3_share": 0.0,
        }

    p = c / total

    # 1. Entropy and Normalized Entropy
    log_n = np.log(max(n, 2))
    p_nz = p[p > 0]
    h = -float(np.sum(p_nz * np.log(p_nz)))
    h_norm = float(h / log_n)
    h_norm = float(np.clip(h_norm, 0.0, 1.0))

    # 2. Effective Number of Passages
    n_eff = float(np.exp(h))

    # 3. Gini Coefficient
    diff_sum = 0.0
    for i in range(n):
        for j in range(n):
            diff_sum += abs(p[i] - p[j])
    gini = float(diff_sum / (2.0 * n * np.sum(p) + 1e-12))
    gini = float(np.clip(gini, 0.0, 1.0))

    # 4. Top-1 and Top-3 Shares
    sorted_p = np.sort(p)[::-1]
    top1 = float(sorted_p[0])
    top3 = float(np.sum(sorted_p[:min(3, n)]))

    return {
        "entropy_norm": round(h_norm, 4),
        "effective_n_passages": round(n_eff, 4),
        "gini_coefficient": round(gini, 4),
        "top1_share": round(top1, 4),
        "top3_share": round(top3, 4),
    }


def extract_causal_passage_attention(
    attention_tensors: Tuple[torch.Tensor, ...],
    spans: Dict[str, Tuple[int, int]],
    num_passages: int = 10,
    selected_layers: Sequence[int] = (0, 4, 8, 16, 24, 31),
) -> Dict[int, Dict[int, float]]:
    """Extract causal attention mass from question/answer tokens back to each passage.

    Attention direction:
      In decoder-only autoregressive causal LM, question tokens t in I(Q)
      attend back to prior passage tokens k in I(D_i).
      Raw mass = sum_{t in I(Q)} sum_{k in I(D_i)} A_{t, k}
      Length-normalized mass = Raw mass / (|I(Q)| * |I(D_i)|)

    Returns:
      Dict mapping layer_idx -> {passage_idx -> normalized_attention_mass}
    """
    q_span = spans.get("query", None)
    if q_span is None:
        return {}

    q_st, q_en = q_span
    q_len = max(1, q_en - q_st)

    layer_passage_mass: Dict[int, Dict[int, float]] = {}

    for layer_idx in selected_layers:
        if layer_idx >= len(attention_tensors):
            continue
        # attn shape: [batch=1, num_heads, seq_len, seq_len]
        attn = attention_tensors[layer_idx][0]  # [num_heads, seq_len, seq_len]
        # Average across attention heads
        mean_attn = torch.mean(attn, dim=0)  # [seq_len, seq_len]

        seq_len = mean_attn.shape[0]
        q_st_clamped = min(max(0, q_st), seq_len - 1)
        q_en_clamped = min(max(q_st_clamped + 1, q_en), seq_len)

        p_masses: Dict[int, float] = {}
        for p_idx in range(num_passages):
            p_span = spans.get(f"passage_{p_idx}", None)
            if p_span is None:
                p_masses[p_idx] = 0.0
                continue
            p_st, p_en = p_span
            p_st_clamped = min(max(0, p_st), seq_len - 1)
            p_en_clamped = min(max(p_st_clamped + 1, p_en), seq_len)
            p_len = max(1, p_en_clamped - p_st_clamped)

            # Block of attention from query positions to passage positions
            # Shape: [q_len, p_len]
            attn_block = mean_attn[q_st_clamped:q_en_clamped, p_st_clamped:p_en_clamped]
            raw_mass = float(torch.sum(attn_block).item())
            norm_mass = raw_mass / (q_len * p_len)
            p_masses[p_idx] = norm_mass

        layer_passage_mass[layer_idx] = p_masses

    return layer_passage_mass


def compute_teacher_forced_gold_log_prob(
    model: Any,
    tokenizer: Any,
    prompt: str,
    gold_answer: str,
    device: str = "cuda",
) -> float:
    """Compute log P(gold_answer | prompt) using teacher forcing under causal language modeling."""
    full_text = f"{prompt.rstrip()} {gold_answer.strip()}"
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)

    prompt_len = prompt_ids.shape[1]
    seq_len = full_ids.shape[1]

    if seq_len <= prompt_len:
        return 0.0

    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    # Target tokens are at positions prompt_len .. seq_len - 1
    # Logits predicting target position t are at position t - 1
    target_logits = logits[0, prompt_len - 1 : seq_len - 1, :]  # [ans_len, vocab_size]
    target_labels = full_ids[0, prompt_len:seq_len]  # [ans_len]

    log_probs = torch.log_softmax(target_logits, dim=-1)
    # Gather log prob of actual gold tokens
    gold_log_probs = log_probs.gather(dim=-1, index=target_labels.unsqueeze(-1)).squeeze(-1)
    total_log_prob = float(torch.sum(gold_log_probs).item())

    return total_log_prob


def compute_signal_correlations(df_passages: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson and Spearman cross-correlations among utilization and relevance signals.

    Note on statistical validity:
      Standard p-values assume independent samples. If passages are pooled across queries,
      passages within the same prompt share context (pseudo-replication). We explicitly record
      both the passage count (n_passages) and distinct query count (n_queries).
    """
    signal_cols = [
        col for col in [
            "bm25_score",
            "cosine_sim_with_query",
            "attention_share",
            "ablation_importance",
            "is_gold_evidence",
        ] if col in df_passages.columns
    ]

    n_queries = int(df_passages["question_id"].nunique()) if "question_id" in df_passages.columns else None

    corr_rows: List[Dict[str, Any]] = []
    for i in range(len(signal_cols)):
        for j in range(i + 1, len(signal_cols)):
            c1 = signal_cols[i]
            c2 = signal_cols[j]
            valid_df = df_passages[[c1, c2]].dropna()
            if len(valid_df) > 5:
                v1 = valid_df[c1].values.astype(float)
                v2 = valid_df[c2].values.astype(float)
                # Check variance
                if np.std(v1) > 1e-8 and np.std(v2) > 1e-8:
                    p_corr, p_pval = stats.pearsonr(v1, v2)
                    s_corr, s_pval = stats.spearmanr(v1, v2)
                else:
                    p_corr, p_pval = 0.0, 1.0
                    s_corr, s_pval = 0.0, 1.0
            else:
                p_corr, p_pval = 0.0, 1.0
                s_corr, s_pval = 0.0, 1.0

            corr_rows.append({
                "signal_1": c1,
                "signal_2": c2,
                "n_passages": len(valid_df),
                "n_queries": n_queries,
                "sample_unit": "passage (clustered within queries; p-values exploratory)",
                "pearson_r": round(float(p_corr), 4),
                "pearson_pvalue_uncorrected": round(float(p_pval), 6),
                "spearman_rho": round(float(s_corr), 4),
                "spearman_pvalue_uncorrected": round(float(s_pval), 6),
            })

    return pd.DataFrame(corr_rows)


def run_mock_utilization_analysis(
    sample_records: List[Dict[str, Any]],
    adapter: DatasetAdapter,
    selected_layers: Sequence[int] = (0, 4, 8, 16, 24, 31),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Lightweight deterministic simulation for tests and CPU verification without loading 7B LM."""
    rng = np.random.RandomState(42)

    attn_rows: List[Dict[str, Any]] = []
    abl_rows: List[Dict[str, Any]] = []
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

        # Generate simulated raw attention and ablation scores
        p_attn_by_layer: Dict[int, List[float]] = {}
        for layer in selected_layers:
            layer_frac = layer / 31.0
            raw_attn = []
            for p_idx, p in enumerate(passages):
                is_ev = check_passage_matches_evidence(p.get("text", ""), flat_ev, flat_hl)
                # BM25 rank decay + evidence boost in later layers
                rank_decay = math.exp(-0.25 * p_idx)
                ev_boost = (0.35 * layer_frac) if is_ev else 0.0
                mass = max(0.01, rank_decay + ev_boost + rng.normal(0, 0.05))
                raw_attn.append(mass)

            # Normalize across passages
            attn_tot = sum(raw_attn)
            attn_shares = [m / attn_tot for m in raw_attn]
            p_attn_by_layer[layer] = attn_shares

            for p_idx, p in enumerate(passages):
                attn_rows.append({
                    "question_id": qid,
                    "paper_id": pid,
                    "layer": layer,
                    "passage_rank": p_idx + 1,
                    "passage_id": p.get("passage_id", f"p_{p_idx}"),
                    "attention_mass": round(float(raw_attn[p_idx]), 4),
                    "attention_share": round(float(attn_shares[p_idx]), 4),
                })

        # Ablation importance (leave-one-out log prob drop)
        abl_importances = []
        for p_idx, p in enumerate(passages):
            is_ev = check_passage_matches_evidence(p.get("text", ""), flat_ev, flat_hl)
            base_drop = 0.05 * math.exp(-0.3 * p_idx)
            ev_drop = 0.45 if is_ev else 0.0
            noise = rng.normal(0, 0.02)
            drop = float(max(0.0, base_drop + ev_drop + noise))
            abl_importances.append(drop)

            abl_rows.append({
                "question_id": qid,
                "paper_id": pid,
                "passage_rank": p_idx + 1,
                "passage_id": p.get("passage_id", f"p_{p_idx}"),
                "bm25_score": round(float(p.get("score", 0.0)), 4),
                "is_gold_evidence": is_ev,
                "ablation_importance": round(drop, 4),
            })

        # Concentration metrics on final layer attention and ablation
        final_layer = selected_layers[-1] if selected_layers else 31
        attn_metrics = compute_concentration_metrics(p_attn_by_layer.get(final_layer, []))
        abl_metrics = compute_concentration_metrics(abl_importances)

        summary_rows.append({
            "question_id": qid,
            "paper_id": pid,
            "attn_entropy_norm": attn_metrics["entropy_norm"],
            "attn_effective_n": attn_metrics["effective_n_passages"],
            "attn_gini": attn_metrics["gini_coefficient"],
            "attn_top1_share": attn_metrics["top1_share"],
            "attn_top3_share": attn_metrics["top3_share"],
            "abl_entropy_norm": abl_metrics["entropy_norm"],
            "abl_effective_n": abl_metrics["effective_n_passages"],
            "abl_gini": abl_metrics["gini_coefficient"],
            "abl_top1_share": abl_metrics["top1_share"],
            "abl_top3_share": abl_metrics["top3_share"],
        })

    df_attn = pd.DataFrame(attn_rows)
    df_abl = pd.DataFrame(abl_rows)
    df_summary = pd.DataFrame(summary_rows)

    # Merge for correlation
    df_merged = df_abl.copy()
    if not df_attn.empty:
        final_l = selected_layers[-1] if selected_layers else 31
        final_attn = df_attn[df_attn["layer"] == final_l][["question_id", "passage_rank", "attention_share"]]
        df_merged = df_merged.merge(final_attn, on=["question_id", "passage_rank"], how="left")

    df_corr = compute_signal_correlations(df_merged)

    return df_attn, df_abl, df_summary, df_corr
