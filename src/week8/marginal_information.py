"""W8-D: Marginal Information Gain (MIG) and Redundancy Feature Engineering.

Features for candidate passage D_i:
- R_q(i): Query relevance = sim(q, D_i)
- R_y(i): Draft relevance = sim(y_0, D_i)
- Red(i): Redundancy against current FULL set = max_{j in FULL} sim(D_i, D_j)
- Diversity(i): Mean distance from other compressed candidate passages
- MIG_i = R_q(i) + alpha * R_y(i) - beta * Red(i)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_cosine_similarity_matrix(embeds1: np.ndarray, embeds2: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarities between two sets of vectors (N, D) and (M, D)."""
    norm1 = np.linalg.norm(embeds1, axis=-1, keepdims=True) + 1e-12
    norm2 = np.linalg.norm(embeds2, axis=-1, keepdims=True) + 1e-12
    e1_norm = embeds1 / norm1
    e2_norm = embeds2 / norm2
    return np.dot(e1_norm, e2_norm.T)


def compute_mig_features_for_query(
    query_text: str,
    draft_text: str,
    full_passages_texts: List[str],
    candidate_passages_texts: Dict[int, str],  # map from rank_idx -> text
    semantic_scorer: Any,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> Dict[int, Dict[str, float]]:
    """Compute R_q, R_y, Red, Diversity, and MIG for each candidate passage."""
    all_cand_indices = sorted(candidate_passages_texts.keys())
    cand_texts = [candidate_passages_texts[idx] for idx in all_cand_indices]

    if not cand_texts:
        return {}

    # Encode query, draft, full set, and candidates using the semantic scorer
    q_vec = semantic_scorer.encoder.encode([query_text], convert_to_numpy=True)
    d_vec = semantic_scorer.encoder.encode([draft_text] if draft_text else ["empty"], convert_to_numpy=True)
    f_vecs = semantic_scorer.encoder.encode(full_passages_texts, convert_to_numpy=True) if full_passages_texts else np.zeros((1, q_vec.shape[1]))
    c_vecs = semantic_scorer.encoder.encode(cand_texts, convert_to_numpy=True)

    # 1. R_q: sim(q, D_i)
    sim_q = compute_cosine_similarity_matrix(c_vecs, q_vec).flatten()

    # 2. R_y: sim(y_0, D_i)
    sim_y = compute_cosine_similarity_matrix(c_vecs, d_vec).flatten()

    # 3. Red: max_{j in FULL} sim(D_i, D_j)
    sim_full = compute_cosine_similarity_matrix(c_vecs, f_vecs)
    red = np.max(sim_full, axis=1)

    # 4. Diversity: mean distance to other candidates
    sim_cands = compute_cosine_similarity_matrix(c_vecs, c_vecs)
    n_cands = len(c_vecs)
    diversity = np.zeros(n_cands)
    for i in range(n_cands):
        other_sims = [sim_cands[i, j] for j in range(n_cands) if j != i]
        diversity[i] = 1.0 - (float(np.mean(other_sims)) if other_sims else 1.0)

    # 5. MIG_i = R_q + alpha * R_y - beta * Red
    mig = sim_q + alpha * sim_y - beta * red

    features_by_idx: Dict[int, Dict[str, float]] = {}
    for i, rank_idx in enumerate(all_cand_indices):
        features_by_idx[rank_idx] = {
            "mig_score": float(mig[i]),
            "query_relevance_rq": float(sim_q[i]),
            "draft_relevance_ry": float(sim_y[i]),
            "redundancy_red": float(red[i]),
            "candidate_diversity": float(diversity[i]),
        }

    return features_by_idx


def tune_mig_alpha_beta(
    val_dataset: List[Dict[str, Any]],
    semantic_scorer: Any,
    alpha_candidates: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 1.0),
    beta_candidates: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 1.0),
) -> Tuple[float, float, float]:
    """Tune alpha and beta on INTERNAL-VAL by maximizing correlation with delta_F1."""
    best_corr = -1.0
    best_alpha, best_beta = 0.5, 0.5

    # Compute correlation over small grid
    for a in alpha_candidates:
        for b in beta_candidates:
            mig_vals = []
            delta_vals = []
            for item in val_dataset:
                rq = item.get("query_relevance_rq", 0.0)
                ry = item.get("draft_relevance_ry", 0.0)
                red = item.get("redundancy_red", 0.0)
                m = rq + a * ry - b * red
                mig_vals.append(m)
                delta_vals.append(item.get("delta_f1", 0.0))

            if len(mig_vals) > 1:
                corr = np.corrcoef(mig_vals, delta_vals)[0, 1]
                if not np.isnan(corr) and corr > best_corr:
                    best_corr = corr
                    best_alpha, best_beta = a, b

    logger.info("Tuned MIG: alpha=%.2f, beta=%.2f with correlation=%.4f", best_alpha, best_beta, best_corr)
    return best_alpha, best_beta, best_corr


# Aliases
tune_mig_hyperparameters = tune_mig_alpha_beta


def compute_candidate_marginal_information(
    query_text: str,
    draft_text: str,
    passages: List[Dict[str, Any]],
    full_indices: Sequence[int],
    candidate_indices: Sequence[int],
    alpha: float = 0.5,
    beta: float = 0.5,
    semantic_scorer: Optional[Any] = None,
) -> Dict[int, Dict[str, float]]:
    """Compute MIG features for candidate passages with or without pre-loaded semantic_scorer."""
    full_texts = [
        passages[idx].get("text", "") for idx in full_indices if idx < len(passages)
    ]
    cand_map = {
        idx: passages[idx].get("text", "")
        for idx in candidate_indices
        if idx < len(passages)
    }

    if semantic_scorer is not None:
        return compute_mig_features_for_query(
            query_text=query_text,
            draft_text=draft_text,
            full_passages_texts=full_texts,
            candidate_passages_texts=cand_map,
            semantic_scorer=semantic_scorer,
            alpha=alpha,
            beta=beta,
        )

    # Deterministic fallback using token Jaccard similarities
    def jaccard(s1: str, s2: str) -> float:
        w1 = set(s1.lower().split())
        w2 = set(s2.lower().split())
        if not w1 or not w2:
            return 0.0
        return float(len(w1 & w2) / len(w1 | w2))

    res: Dict[int, Dict[str, float]] = {}
    for c_idx, c_text in cand_map.items():
        rq = jaccard(query_text, c_text)
        ry = jaccard(draft_text, c_text)
        red = max([jaccard(c_text, f_text) for f_text in full_texts], default=0.0)

        # diversity: mean distance to other candidate texts
        other_texts = [v for k, v in cand_map.items() if k != c_idx]
        div = 1.0 - (float(np.mean([jaccard(c_text, ot) for ot in other_texts])) if other_texts else 0.0)

        mig_val = rq + alpha * ry - beta * red
        res[c_idx] = {
            "mig_score": float(mig_val),
            "query_relevance_rq": float(rq),
            "draft_relevance_ry": float(ry),
            "redundancy_red": float(red),
            "candidate_diversity": float(div),
        }

    return res

