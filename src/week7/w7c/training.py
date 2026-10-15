"""Feature assembly, ablation definitions, preprocessing, and model training for Week 7-C.

Coordinates extraction across feature families A-H, maps ablation subsets F0-F5,
fits preprocessing strictly on TRAIN-FIT, and manages feature caching.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.week7.w7c.features_query import extract_query_complexity_features
from src.week7.w7c.features_retrieval import (
    compute_bm25_qpp_features,
    compute_cross_retriever_features,
    compute_semantic_budget_geometry,
    compute_semantic_pool_geometry,
)
from src.week7.w7c.features_set import (
    compute_set_diversity_features,
    compute_tail_threat_features,
)
from src.week7.w7c.features_coverage import (
    compute_cost_features,
    compute_query_coverage_features,
)
from src.week7.w7c.features_crossencoder import (
    compute_cross_encoder_budget_features,
)
from src.week7.w7c.leakage import assert_zero_feature_leakage, sanitize_feature_matrix

logger = logging.getLogger(__name__)

# Feature ablation mappings
ABLATION_FAMILIES: Dict[str, str] = {
    "F0": "old_w4_baseline",
    "F1": "query_plus_retrieval_geometry",
    "F2": "retrieval_aware",
    "F3": "set_aware",
    "F4": "sufficiency_focused",
    "F5": "sufficiency_plus_cross_encoder",
}


def build_candidate_feature_bank(
    question_ids: Sequence[str],
    paper_ids: Sequence[str],
    questions: Sequence[str],
    passages_list: Sequence[Sequence[Dict[str, Any]]],
    bm25_scores_list: Sequence[Sequence[float]],
    semantic_scores_list: Sequence[Sequence[float]],
    cached_embeddings: Optional[Dict[str, Any]] = None,
    cross_encoder_scores_list: Optional[Sequence[Sequence[float]]] = None,
    budgets: Sequence[int] = (2, 4, 5, 6, 8),
    enable_cross_encoder: bool = False,
    split_name: str = "train",
) -> pd.DataFrame:
    """Build full candidate-budget feature bank (N * len(budgets) rows).

    Extracts Families A through H (and optional I) for every (q, k) candidate.
    """
    n_queries = len(question_ids)
    rows: List[Dict[str, Any]] = []

    # Pre-extract Family A query complexity features (query-level)
    df_query_feats = extract_query_complexity_features(question_ids, questions)
    query_feats_by_qid = {
        row["question_id"]: row.drop("question_id").to_dict()
        for _, row in df_query_feats.iterrows()
    }

    for idx in range(n_queries):
        qid = str(question_ids[idx]).strip()
        pid = str(paper_ids[idx]).strip()
        q_text = str(questions[idx]).strip()
        passages = passages_list[idx]
        b_scores = bm25_scores_list[idx]
        s_scores = semantic_scores_list[idx]

        # Semantic ranking of 10 passages
        sem_ranks = np.argsort(s_scores)[::-1].tolist()

        # Query features (Family A)
        q_feats = query_feats_by_qid[qid]

        # Pool semantic geometry (Family B pool)
        sem_pool_feats = compute_semantic_pool_geometry(s_scores)
        sorted_s = sorted(s_scores, reverse=True)

        # Passage embeddings for Family E and F
        if cached_embeddings and qid in cached_embeddings:
            p_embeds = cached_embeddings[qid]
            if isinstance(p_embeds, dict):
                p_embeds = p_embeds.get("retrieval_embeds", p_embeds.get("embeddings"))
            if hasattr(p_embeds, "cpu"):
                p_embeds = p_embeds.cpu().numpy()
            p_embeds = np.array(p_embeds, dtype=np.float32)
            if p_embeds.ndim == 3:
                p_embeds = p_embeds[0]  # Take first batch dim if (1, 10, D)
        else:
            # Fallback zero/identity embedding
            p_embeds = np.eye(10, dtype=np.float32)

        ce_scores = cross_encoder_scores_list[idx] if cross_encoder_scores_list else None

        for k in budgets:
            # Budget-specific Family B
            sem_k_feats = compute_semantic_budget_geometry(sorted_s, k=k)

            # Family C (BM25 & QPP)
            bm25_feats = compute_bm25_qpp_features(b_scores, sem_ranks, k=k)

            # Family D (Cross-Retriever)
            cross_feats = compute_cross_retriever_features(b_scores, s_scores, sem_ranks, k=k)

            # Family E (Set Diversity)
            div_feats = compute_set_diversity_features(p_embeds, sem_ranks, k=k)

            # Family F (Tail Threat - HIGH PRIORITY)
            tail_feats = compute_tail_threat_features(s_scores, p_embeds, sem_ranks, k=k)

            # Family G (Query Coverage)
            cov_feats = compute_query_coverage_features(q_text, passages, sem_ranks, k=k, budgets=budgets)

            # Family H (Length / Cost)
            cost_feats = compute_cost_features(passages, sem_ranks, k=k, budgets=budgets)

            # Family I (Optional Cross-Encoder)
            ce_feats = compute_cross_encoder_budget_features(ce_scores, sem_ranks, k=k, enabled=enable_cross_encoder)

            # W4 Baseline features proxy
            w4_baseline_feats = {
                "w4_bm25_mean": float(np.mean(b_scores)),
                "w4_lex_coverage_top2": float((b_scores[0] + b_scores[1]) / (np.sum(b_scores) + 1e-6)) if len(b_scores) >= 2 else 0.0,
                "w4_sem_sim_top6": float(np.mean(sorted_s[:6])) if len(sorted_s) >= 6 else float(np.mean(sorted_s)),
                "w4_cluster_span": float(np.std(sorted_s)),
                "w4_lex_overlap_mean": float(np.mean(b_scores[:min(k, len(b_scores))])),
            }

            row = {
                "split": split_name,
                "question_id": qid,
                "paper_id": pid,
                "k": int(k),
                **q_feats,
                **sem_pool_feats,
                **sem_k_feats,
                **bm25_feats,
                **cross_feats,
                **div_feats,
                **tail_feats,
                **cov_feats,
                **cost_feats,
                **ce_feats,
                **w4_baseline_feats,
            }
            rows.append(row)

    df_candidate = pd.DataFrame(rows)
    return df_candidate


def get_ablation_feature_names(
    ablation_name: str,
    all_columns: Sequence[str],
) -> List[str]:
    """Return feature column names corresponding to an ablation level (F0-F5)."""
    cols = [
        c for c in all_columns
        if not c.startswith("target_") and c not in {"split", "question_id", "paper_id", "k"}
    ]

    if ablation_name == "F0":
        # W4 baseline only
        return [c for c in cols if c.startswith("w4_")] + ["cost_k"]

    elif ablation_name == "F1":
        # A (q_) + B (sem_) + C (bm25_) + cost_k
        return [c for c in cols if c.startswith("q_") or c.startswith("sem_") or c.startswith("bm25_")] + ["cost_k"]

    elif ablation_name == "F2":
        # F1 + D (cross_)
        f1_cols = [c for c in cols if c.startswith("q_") or c.startswith("sem_") or c.startswith("bm25_")]
        return f1_cols + [c for c in cols if c.startswith("cross_")] + ["cost_k"]

    elif ablation_name == "F3":
        # F2 + E (div_)
        f2_cols = [c for c in cols if c.startswith("q_") or c.startswith("sem_") or c.startswith("bm25_") or c.startswith("cross_")]
        return f2_cols + [c for c in cols if c.startswith("div_")] + ["cost_k"]

    elif ablation_name == "F4":
        # Full sufficiency-focused: A + B + C + D + E + F (tail_) + G (cov_) + H (cost_)
        return [c for c in cols if not c.startswith("w4_") and not c.startswith("ce_")]

    elif ablation_name == "F5":
        # F4 + I (ce_)
        return [c for c in cols if not c.startswith("w4_")]

    else:
        raise ValueError(f"Unknown ablation: {ablation_name}. Choose from F0, F1, F2, F3, F4, F5.")


class PreprocessingPipeline:
    """Standardizes and imputes feature matrices, fitted strictly on TRAIN-FIT."""

    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame) -> "PreprocessingPipeline":
        self.feature_names = list(X.columns)
        assert_zero_feature_leakage(self.feature_names)
        imputed = self.imputer.fit_transform(X)
        self.scaler.fit(imputed)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("PreprocessingPipeline is not fitted.")
        X_aligned = X[self.feature_names].copy()
        imputed = self.imputer.transform(X_aligned)
        scaled = self.scaler.transform(imputed)
        return pd.DataFrame(scaled, columns=self.feature_names, index=X.index)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)
