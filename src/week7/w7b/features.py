"""Feature extraction, standardization, and loading for Week 7-B.

Enforces:
- Exact 5 canonical locked W4 Stage-2 features:
  1. bm25_mean_score (Family A: Retrieval Strength)
  2. lex_coverage_top2 (Family E: Cumulative Lexical Coverage)
  3. sem_sim_mean_top6 (Family C: Semantic Coherence of Tail)
  4. evidence_query_cluster_span (Family F: Multi-Aspect Dispersion)
  5. evidence_lexical_overlap_mean (Family F: Chunk Redundancy)
- Strict pre-generation invariant (no gold labels, answers, or downstream metrics).
- Scaler is fitted ONLY on TRAIN-FIT and applied to INTERNAL-VAL and DEV without data leakage.
"""

from __future__ import annotations

import logging
import re
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

STAGE_2_CORE_FEATURES: List[str] = [
    "bm25_mean_score",
    "lex_coverage_top2",
    "sem_sim_mean_top6",
    "evidence_query_cluster_span",
    "evidence_lexical_overlap_mean",
]

STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
    "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into",
    "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's",
    "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
    "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
    "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves",
}


def extract_content_words(text: str) -> List[str]:
    """Extract lowercased query content words excluding punctuation and stopwords."""
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    words = clean.split()
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def tokenize_clean(text: str) -> List[str]:
    """Tokenize passage text into lowercased alphanumeric words."""
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    return [w for w in clean.split() if len(w) > 1]


def extract_single_query_w4_features(
    record: Dict[str, Any],
    semantic_scores: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Extract the exact 5 locked W4 Stage-2 features for a single query record.

    Args:
        record: Dict containing 'question' and 'passages' list.
        semantic_scores: Optional length-10 sequence of query-passage semantic similarity scores.
                         If omitted, uses passage 'score' fields as fallback.

    Returns:
        Dict with keys: bm25_mean_score, lex_coverage_top2, sem_sim_mean_top6,
        evidence_query_cluster_span, evidence_lexical_overlap_mean.
    """
    query_text = str(record.get("question", "")).strip()
    passages = record.get("passages", [])[:10]

    # 1. BM25 Mean Score
    scores = [float(p.get("score", 0.0)) for p in passages]
    if len(scores) < 10:
        scores = scores + [0.0] * (10 - len(scores))
    bm25_mean = float(np.mean(scores))

    # 2. Lexical Coverage Top-2
    q_content = set(extract_content_words(query_text))
    passage_texts = [p.get("text", "") for p in passages]
    if len(passage_texts) < 10:
        passage_texts = passage_texts + [""] * (10 - len(passage_texts))
    passage_words = [set(tokenize_clean(pt)) for pt in passage_texts]

    if len(q_content) == 0:
        lex_cov_top2 = 1.0
    else:
        top2_words = set().union(*passage_words[:2]) if len(passage_words) >= 2 else set()
        matched = q_content.intersection(top2_words)
        lex_cov_top2 = float(len(matched) / len(q_content))

    # 3. Semantic Similarity Mean Top-6
    if semantic_scores is not None and len(semantic_scores) > 0:
        s_arr = np.array(semantic_scores[:10], dtype=float)
        if len(s_arr) < 10:
            s_arr = np.pad(s_arr, (0, 10 - len(s_arr)), mode="constant", constant_values=0.0)
        sem_mean_top6 = float(np.mean(s_arr[:6]))
    else:
        # Fallback to normalized BM25 score top-6
        s_arr = np.array(scores, dtype=float)
        sem_mean_top6 = float(np.mean(s_arr[:6]))

    # 4. Evidence Lexical Overlap Mean (Pairwise Jaccard)
    jaccards = []
    for i in range(len(passage_words)):
        for j in range(i + 1, len(passage_words)):
            w1, w2 = passage_words[i], passage_words[j]
            union_len = len(w1.union(w2))
            if union_len > 0:
                jaccards.append(len(w1.intersection(w2)) / union_len)
            else:
                jaccards.append(0.0)
    ev_lex_overlap = float(np.mean(jaccards)) if jaccards else 0.0

    # 5. Evidence Query Cluster Span
    # Compute passage similarity via token Jaccard for lightweight clustering
    n_p = len(passage_words)
    dist_mat = np.zeros((n_p, n_p), dtype=float)
    for i in range(n_p):
        for j in range(i + 1, n_p):
            w1, w2 = passage_words[i], passage_words[j]
            union_len = len(w1.union(w2))
            sim = (len(w1.intersection(w2)) / union_len) if union_len > 0 else 0.0
            dist_mat[i, j] = 1.0 - sim
            dist_mat[j, i] = 1.0 - sim

    try:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.85,
            metric="precomputed",
            linkage="average",
        )
        clusters = clustering.fit_predict(dist_mat).tolist()
    except Exception:
        clusters = list(range(n_p))

    cluster_has_term = set()
    for idx, c_id in enumerate(clusters[:n_p]):
        if any(w in passage_words[idx] for w in q_content):
            cluster_has_term.add(c_id)
    ev_cluster_span = float(len(cluster_has_term)) if len(q_content) > 0 else 1.0

    return {
        "bm25_mean_score": bm25_mean,
        "lex_coverage_top2": lex_cov_top2,
        "sem_sim_mean_top6": sem_mean_top6,
        "evidence_query_cluster_span": ev_cluster_span,
        "evidence_lexical_overlap_mean": ev_lex_overlap,
    }


def extract_features_for_bundle(
    bundle: Dict[str, Any],
    semantic_scores_arr: Optional[np.ndarray] = None,
    cache_csv_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Extract or load the 5 locked features for an entire dataset bundle.

    If cache_csv_path exists, loads directly from disk.
    """
    if cache_csv_path and Path(cache_csv_path).exists():
        df_cached = pd.read_csv(cache_csv_path)
        df_cached["question_id"] = df_cached["question_id"].astype(str).str.strip()
        # Verify required features present
        for feat in STAGE_2_CORE_FEATURES:
            if feat not in df_cached.columns:
                raise KeyError(f"Missing required feature column '{feat}' in {cache_csv_path}")
        return df_cached

    qids = bundle["question_ids"]
    raw_records = bundle["raw_records"]
    rows = []

    for idx, qid in enumerate(qids):
        sem_scores = semantic_scores_arr[idx] if semantic_scores_arr is not None else None
        feat_dict = extract_single_query_w4_features(raw_records[idx], semantic_scores=sem_scores)
        feat_dict["question_id"] = str(qid)
        feat_dict["paper_id"] = str(bundle["paper_ids"][idx])
        rows.append(feat_dict)

    df_feats = pd.DataFrame(rows)
    # Ensure correct column ordering
    cols = ["question_id", "paper_id"] + STAGE_2_CORE_FEATURES
    df_feats = df_feats[cols]

    if cache_csv_path:
        p = Path(cache_csv_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df_feats.to_csv(p, index=False)
        logger.info("Saved extracted W4 features to %s", p)

    return df_feats


def standardize_features_train_fit_only(
    df_fit_feats: pd.DataFrame,
    df_val_feats: pd.DataFrame,
    df_dev_feats: pd.DataFrame,
    feature_cols: Sequence[str] = STAGE_2_CORE_FEATURES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Fit StandardScaler ONLY on TRAIN-FIT and transform INTERNAL-VAL and DEV.

    Returns:
        (X_fit_scaled, X_val_scaled, X_dev_scaled, fitted_scaler)
    """
    scaler = StandardScaler()
    X_fit = df_fit_feats[list(feature_cols)].values.astype(np.float32)
    X_val = df_val_feats[list(feature_cols)].values.astype(np.float32)
    X_dev = df_dev_feats[list(feature_cols)].values.astype(np.float32)

    X_fit_scaled = scaler.fit_transform(X_fit)
    X_val_scaled = scaler.transform(X_val)
    X_dev_scaled = scaler.transform(X_dev)

    logger.info("StandardScaler fitted strictly on TRAIN-FIT (%d rows). Applied to VAL (%d) and DEV (%d).", len(X_fit), len(X_val), len(X_dev))
    return X_fit_scaled, X_val_scaled, X_dev_scaled, scaler
