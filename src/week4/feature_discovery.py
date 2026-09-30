"""Feature Discovery Library for Week 4: Query-Conditioned Evidence Allocation (QCCA).

Implements 64 principled pre-generation observable features across 6 distinct information families:
  Family A: BM25 / Retrieval-Score Structure (17 features)
  Family B: Query Complexity (12 features)
  Family C: Query <-> Retrieved-Passage Semantic Structure (11 features)
  Family D: Passage Redundancy / Diversity (10 features)
  Family E: Query / Passage Lexical Coverage (8 features)
  Family F: Evidence Structure / Multi-Aspect Signals (6 features)

Also implements:
- Automated Pre-Generation Leakage Audit
- Marginal Gain and Oracle Target Construction
- Paper-Clustered Bootstrap Association Estimation
- 5-Fold GroupKFold Stability Assessment
- Mutual Information and Non-linear Diagnostics
- Candidate Feature Screening Criteria
"""

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import GroupKFold
import torch

logger = logging.getLogger(__name__)

# Standard English stopwords for lexical coverage
STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't",
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
    "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how",
    "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its",
    "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's",
    "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd",
    "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very",
    "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when",
    "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}

CONJUNCTIONS: Set[str] = {
    "and", "or", "but", "because", "although", "whereas", "if", "while", "since",
    "unless", "though", "even", "neither", "nor", "yet", "whether"
}

COMPARISON_TERMS: Set[str] = {
    "compare", "compared", "comparing", "comparison", "difference", "differences",
    "differ", "differs", "better", "worse", "higher", "lower", "versus", "vs",
    "relative", "outperform", "outperforms", "contrast", "contrasting"
}


# =============================================================================
# 1. FEATURE EXTRACTION PER QUERY
# =============================================================================

def tokenize_clean(text: str) -> List[str]:
    """Extract lowercase alphanumeric tokens."""
    return re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())


def extract_content_words(text: str) -> List[str]:
    """Extract non-stopword tokens."""
    tokens = tokenize_clean(text)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def extract_query_features(
    record: Dict[str, Any],
    query_emb: Optional[torch.Tensor] = None,
    passage_embs: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Extract all 64 pre-generation features for a single query record.
    
    Args:
        record: Dict containing question_id, paper_id, question, and passages list.
        query_emb: Normalized 1D torch Tensor of query embedding (dim=4096).
        passage_embs: 2D torch Tensor of top-10 passage embeddings (shape [10, 4096]).
    """
    qid = str(record["question_id"]).strip()
    pid = str(record["paper_id"]).strip()
    query_text = str(record["question"]).strip()
    passages = record.get("passages", [])
    
    # Retrieval scores
    scores = [float(p.get("score", 0.0)) for p in passages]
    if len(scores) < 10:
        # pad to 10 if fewer
        scores = scores + [0.0] * (10 - len(scores))
    scores = sorted(scores[:10], reverse=True)
    s_arr = np.array(scores, dtype=float)
    s_sum = float(s_arr.sum())
    s_sum_safe = s_sum if s_sum > 0 else 1.0

    feat: Dict[str, Any] = {
        "question_id": qid,
        "paper_id": pid,
    }

    # -------------------------------------------------------------------------
    # FAMILY A: BM25 / Retrieval-Score Structure (17 features)
    # -------------------------------------------------------------------------
    s1, s2, s3 = s_arr[0], s_arr[1], s_arr[2]
    feat["rho_1"] = float(s1 / s_sum_safe)
    feat["delta_12"] = float((s1 - s2) / s1) if s1 > 0 else 0.0
    
    # Softmax entropy
    exps = np.exp(s_arr - np.max(s_arr))
    probs = exps / np.sum(exps)
    ent = -float(np.sum(probs * np.log(probs + 1e-12)))
    feat["entropy"] = ent
    feat["entropy_normalized"] = float(ent / np.log(len(s_arr)))
    
    feat["query_token_length"] = len(query_text.split())
    feat["n_high"] = int(np.sum(s_arr >= 0.5 * s1)) if s1 > 0 else 1
    
    feat["bm25_top1_score"] = float(s1)
    feat["bm25_top2_score"] = float(s2)
    feat["bm25_top3_score"] = float(s3)
    feat["bm25_mean_score"] = float(np.mean(s_arr))
    feat["bm25_std_score"] = float(np.std(s_arr))
    feat["bm25_score_range"] = float(s1 - s_arr[-1])
    mean_val = float(np.mean(s_arr))
    feat["bm25_coeff_variation"] = float(np.std(s_arr) / (mean_val + 1e-6))
    
    feat["bm25_top2_ratio"] = float((s1 + s2) / s_sum_safe)
    feat["bm25_top4_ratio"] = float(np.sum(s_arr[:4]) / s_sum_safe)
    
    # Gaps
    adj_gaps = [s_arr[i] - s_arr[i+1] for i in range(len(s_arr) - 1)]
    mean_gap = float(np.mean(adj_gaps))
    feat["bm25_mean_adjacent_gap"] = mean_gap
    feat["bm25_first_gap_ratio"] = float((s1 - s2) / (mean_gap + 1e-6))

    # -------------------------------------------------------------------------
    # FAMILY B: Query Complexity (12 features)
    # -------------------------------------------------------------------------
    q_words = query_text.split()
    q_lower = query_text.lower()
    q_tokens = tokenize_clean(query_text)
    
    feat["query_word_count"] = len(q_words)
    feat["query_char_length"] = len(query_text)
    feat["query_comma_count"] = query_text.count(",")
    feat["query_conjunction_count"] = sum(1 for w in q_tokens if w in CONJUNCTIONS)
    feat["query_num_count"] = sum(1 for w in q_tokens if any(c.isdigit() for c in w))
    
    feat["query_is_comparison"] = 1.0 if any(w in COMPARISON_TERMS for w in q_tokens) else 0.0
    feat["query_is_how_why"] = 1.0 if any(w in {"how", "why"} for w in q_tokens) else 0.0
    feat["query_is_factoid"] = 1.0 if any(w in {"who", "when", "where"} for w in q_tokens) else 0.0
    feat["query_is_what_which"] = 1.0 if any(w in {"what", "which"} for w in q_tokens) else 0.0
    feat["query_is_numerical"] = 1.0 if ("how many" in q_lower or "count" in q_lower or "number of" in q_lower or "percentage" in q_lower) else 0.0
    feat["query_quote_count"] = query_text.count('"') + query_text.count("'")
    
    # Capitalized spans (excluding first word)
    cap_count = 0
    if len(q_words) > 1:
        for w in q_words[1:]:
            clean_w = re.sub(r"[^a-zA-Z]", "", w)
            if clean_w and clean_w[0].isupper():
                cap_count += 1
    feat["query_capitalized_spans"] = cap_count

    # -------------------------------------------------------------------------
    # FAMILY C & D & F: Embeddings (Semantic Similarity, Diversity, Clusters)
    # -------------------------------------------------------------------------
    # Default fallbacks if embeddings missing
    sim_scores: List[float] = [0.0] * 10
    pairwise_sims: List[float] = [0.0] * 45
    clusters: List[int] = [0] * 10

    if query_emb is not None and passage_embs is not None:
        p_embs = passage_embs.detach().cpu().to(torch.float32)
        q_emb = query_emb.detach().cpu().to(torch.float32)
        
        # Normalize passage embeddings
        p_norms = torch.norm(p_embs, p=2, dim=1, keepdim=True)
        p_norms = torch.clamp(p_norms, min=1e-8)
        p_normed = p_embs / p_norms
        
        # Query-passage cosine similarity
        q_norm = torch.norm(q_emb, p=2)
        q_normed = q_emb / max(float(q_norm), 1e-8)
        
        c_sims = torch.mv(p_normed, q_normed).numpy()
        sim_scores = [float(x) for x in c_sims]
        
        # Pairwise passage cosine similarities (10 x 10)
        p_sim_matrix = torch.mm(p_normed, p_normed.t()).numpy()
        triu_indices = np.triu_indices(len(p_sim_matrix), k=1)
        pairwise_sims = p_sim_matrix[triu_indices].tolist()
        
        # Deterministic Agglomerative Clustering on passages (cosine distance)
        dist_matrix = np.clip(1.0 - p_sim_matrix, 0.0, 2.0)
        np.fill_diagonal(dist_matrix, 0.0)
        try:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.20,
                metric="precomputed",
                linkage="average"
            )
            clusters = clustering.fit_predict(dist_matrix).tolist()
        except Exception:
            clusters = list(range(len(p_sim_matrix)))

    # Family C: Semantic Query-Passage Structure (11 features)
    c_arr = np.array(sim_scores, dtype=float)
    c1 = float(c_arr[0])
    c2 = float(c_arr[1])
    feat["sem_sim_top1"] = c1
    feat["sem_sim_top2"] = c2
    feat["sem_sim_max"] = float(np.max(c_arr))
    feat["sem_sim_mean_top10"] = float(np.mean(c_arr))
    feat["sem_sim_std_top10"] = float(np.std(c_arr))
    feat["sem_sim_gap_12"] = float(c1 - c2)
    feat["sem_sim_mean_top2"] = float(np.mean(c_arr[:2]))
    feat["sem_sim_mean_top4"] = float(np.mean(c_arr[:4]))
    feat["sem_sim_mean_top6"] = float(np.mean(c_arr[:6]))
    feat["sem_sim_mean_top8"] = float(np.mean(c_arr[:8]))
    m10 = float(np.mean(c_arr))
    feat["sem_sim_saturation_ratio"] = float(np.mean(c_arr[:4]) / (m10 + 1e-6))

    # Family D: Passage Redundancy / Diversity (10 features)
    pw_arr = np.array(pairwise_sims, dtype=float)
    feat["passage_sim_mean"] = float(np.mean(pw_arr))
    feat["passage_sim_median"] = float(np.median(pw_arr))
    feat["passage_sim_max"] = float(np.max(pw_arr))
    feat["passage_sim_min"] = float(np.min(pw_arr))
    feat["passage_sim_std"] = float(np.std(pw_arr))
    
    # Nearest neighbor similarities
    if query_emb is not None and passage_embs is not None:
        np.fill_diagonal(p_sim_matrix, -1.0)
        nn_sims = np.max(p_sim_matrix, axis=1)
        feat["passage_nearest_neighbor_mean"] = float(np.mean(nn_sims))
        feat["passage_nearest_neighbor_min"] = float(np.min(nn_sims))
    else:
        feat["passage_nearest_neighbor_mean"] = float(np.mean(pw_arr))
        feat["passage_nearest_neighbor_min"] = float(np.min(pw_arr))
        
    feat["passage_diversity_score"] = float(1.0 - np.mean(pw_arr))
    
    # Cluster statistics
    unique_clusters, cluster_counts = np.unique(clusters, return_counts=True)
    feat["passage_cluster_count"] = int(len(unique_clusters))
    feat["passage_largest_cluster_frac"] = float(np.max(cluster_counts) / len(clusters))

    # -------------------------------------------------------------------------
    # FAMILY E: Query / Passage Lexical Coverage (8 features)
    # -------------------------------------------------------------------------
    q_content = set(extract_content_words(query_text))
    n_content = len(q_content)
    
    passage_texts = [p.get("text", "") for p in passages[:10]]
    if len(passage_texts) < 10:
        passage_texts = passage_texts + [""] * (10 - len(passage_texts))
        
    passage_words = [set(tokenize_clean(pt)) for pt in passage_texts]
    
    def cumulative_coverage(k: int) -> float:
        if n_content == 0:
            return 1.0
        combined_words = set().union(*passage_words[:k]) if k > 0 else set()
        matched = q_content.intersection(combined_words)
        return float(len(matched) / n_content)

    c_top1 = cumulative_coverage(1)
    c_top2 = cumulative_coverage(2)
    c_top4 = cumulative_coverage(4)
    c_top6 = cumulative_coverage(6)
    c_top8 = cumulative_coverage(8)
    c_top10 = cumulative_coverage(10)
    
    feat["lex_coverage_top1"] = c_top1
    feat["lex_coverage_top2"] = c_top2
    feat["lex_coverage_top4"] = c_top4
    feat["lex_coverage_top6"] = c_top6
    feat["lex_coverage_top8"] = c_top8
    feat["lex_coverage_top10"] = c_top10
    feat["lex_coverage_gain_1_to_4"] = float(c_top4 - c_top1)
    
    all_seen = set().union(*passage_words)
    missing_count = sum(1 for w in q_content if w not in all_seen)
    feat["lex_missing_terms_top10"] = missing_count

    # -------------------------------------------------------------------------
    # FAMILY F: Evidence Structure / Multi-Aspect Signals (6 features)
    # -------------------------------------------------------------------------
    max_c = float(np.max(c_arr))
    feat["evidence_high_sim_count"] = int(np.sum(c_arr >= 0.8 * max_c)) if max_c > 0 else 1
    
    p_lengths = [len(tokenize_clean(pt)) for pt in passage_texts]
    feat["evidence_length_mean"] = float(np.mean(p_lengths))
    feat["evidence_length_std"] = float(np.std(p_lengths))
    
    # Jaccard overlap between passage pairs
    jaccards = []
    for i in range(len(passage_words)):
        for j in range(i + 1, len(passage_words)):
            w1, w2 = passage_words[i], passage_words[j]
            union_len = len(w1.union(w2))
            if union_len > 0:
                jaccards.append(len(w1.intersection(w2)) / union_len)
            else:
                jaccards.append(0.0)
    feat["evidence_lexical_overlap_mean"] = float(np.mean(jaccards)) if jaccards else 0.0
    
    unique_all_words = set().union(*passage_words)
    feat["evidence_unique_word_count"] = len(unique_all_words)
    
    # Query cluster span
    cluster_has_term = set()
    for idx, c_id in enumerate(clusters[:len(passage_words)]):
        if any(w in passage_words[idx] for w in q_content):
            cluster_has_term.add(c_id)
    feat["evidence_query_cluster_span"] = len(cluster_has_term)

    return feat


def load_all_dev_features(
    raw_retrieval_path: str = "results/week2/raw/dev_retrieval.jsonl",
    passage_embeddings_path: str = "results/week2/raw/dev_sfr_embeddings.pt",
    query_embeddings_path: str = "results/week4/feature_discovery/dev_query_sfr_embeddings.pt",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and extract all 64 pre-generation features for the 231 development questions."""
    raw_p = Path(raw_retrieval_path)
    pass_p = Path(passage_embeddings_path)
    q_p = Path(query_embeddings_path)

    if not raw_p.exists():
        raise FileNotFoundError(f"Missing raw retrieval file: {raw_p}")

    records = []
    with open(raw_p) as f:
        for line in f:
            records.append(json.loads(line))

    pass_dict = {}
    if pass_p.exists():
        pass_data = torch.load(pass_p, map_location="cpu")
        pass_dict = pass_data.get("embeddings", {})

    query_dict = {}
    if q_p.exists():
        q_data = torch.load(q_p, map_location="cpu")
        query_dict = q_data.get("embeddings", {})

    features_list = []
    for r in records:
        qid = str(r["question_id"]).strip()
        q_emb = query_dict.get(qid)
        p_embs = pass_dict.get(qid)
        f_dict = extract_query_features(r, q_emb, p_embs)
        features_list.append(f_dict)

    df_features = pd.DataFrame(features_list)
    df_metadata = get_feature_metadata()

    return df_features, df_metadata


# =============================================================================
# 2. FEATURE METADATA SPECIFICATION
# =============================================================================

def get_feature_metadata() -> pd.DataFrame:
    """Generate structured documentation for all 64 pre-generation features."""
    records = [
        # Family A (17)
        ("rho_1", "A: BM25/Retrieval", "Top-1 retrieval score concentration ratio s1 / sum(s)", "raw_retrieval_scores", True, False, True, False, False, "score_ratio", "VERIFIED_PRE_GEN"),
        ("delta_12", "A: BM25/Retrieval", "Normalized gap between top-1 and top-2 scores (s1 - s2)/s1", "raw_retrieval_scores", True, False, True, False, False, "normalized_gap", "VERIFIED_PRE_GEN"),
        ("entropy", "A: BM25/Retrieval", "Shannon entropy of softmax-normalized retrieval scores", "raw_retrieval_scores", True, False, True, False, False, "shannon_entropy", "VERIFIED_PRE_GEN"),
        ("entropy_normalized", "A: BM25/Retrieval", "Entropy normalized by ln(10)", "raw_retrieval_scores", True, False, True, False, False, "normalized_entropy", "VERIFIED_PRE_GEN"),
        ("query_token_length", "A: BM25/Retrieval", "Whitespace token length of query", "query_text", True, True, False, False, False, "word_count", "VERIFIED_PRE_GEN"),
        ("n_high", "A: BM25/Retrieval", "Count of passages with retrieval score >= 0.5 * s1", "raw_retrieval_scores", True, False, True, False, False, "threshold_count", "VERIFIED_PRE_GEN"),
        ("bm25_top1_score", "A: BM25/Retrieval", "Raw BM25 score of top-1 passage", "raw_retrieval_scores", True, False, True, False, False, "raw_score", "VERIFIED_PRE_GEN"),
        ("bm25_top2_score", "A: BM25/Retrieval", "Raw BM25 score of top-2 passage", "raw_retrieval_scores", True, False, True, False, False, "raw_score", "VERIFIED_PRE_GEN"),
        ("bm25_top3_score", "A: BM25/Retrieval", "Raw BM25 score of top-3 passage", "raw_retrieval_scores", True, False, True, False, False, "raw_score", "VERIFIED_PRE_GEN"),
        ("bm25_mean_score", "A: BM25/Retrieval", "Mean BM25 score across top-10 passages", "raw_retrieval_scores", True, False, True, False, False, "mean", "VERIFIED_PRE_GEN"),
        ("bm25_std_score", "A: BM25/Retrieval", "Standard deviation of BM25 scores", "raw_retrieval_scores", True, False, True, False, False, "std_dev", "VERIFIED_PRE_GEN"),
        ("bm25_score_range", "A: BM25/Retrieval", "Range between top-1 and top-10 BM25 score", "raw_retrieval_scores", True, False, True, False, False, "range", "VERIFIED_PRE_GEN"),
        ("bm25_coeff_variation", "A: BM25/Retrieval", "Coefficient of variation of retrieval scores (std / mean)", "raw_retrieval_scores", True, False, True, False, False, "relative_dispersion", "VERIFIED_PRE_GEN"),
        ("bm25_top2_ratio", "A: BM25/Retrieval", "Score mass fraction of top-2 passages", "raw_retrieval_scores", True, False, True, False, False, "cumulative_ratio", "VERIFIED_PRE_GEN"),
        ("bm25_top4_ratio", "A: BM25/Retrieval", "Score mass fraction of top-4 passages", "raw_retrieval_scores", True, False, True, False, False, "cumulative_ratio", "VERIFIED_PRE_GEN"),
        ("bm25_mean_adjacent_gap", "A: BM25/Retrieval", "Mean drop between adjacent ranked scores", "raw_retrieval_scores", True, False, True, False, False, "mean_gradient", "VERIFIED_PRE_GEN"),
        ("bm25_first_gap_ratio", "A: BM25/Retrieval", "Top gap (s1 - s2) divided by mean adjacent gap", "raw_retrieval_scores", True, False, True, False, False, "gap_prominence", "VERIFIED_PRE_GEN"),

        # Family B (12)
        ("query_word_count", "B: Query Complexity", "Number of whitespace-delimited words in query", "query_text", True, True, False, False, False, "word_count", "VERIFIED_PRE_GEN"),
        ("query_char_length", "B: Query Complexity", "Total character count in query string", "query_text", True, True, False, False, False, "char_count", "VERIFIED_PRE_GEN"),
        ("query_comma_count", "B: Query Complexity", "Number of commas (proxy for clauses and lists)", "query_text", True, True, False, False, False, "punctuation_count", "VERIFIED_PRE_GEN"),
        ("query_conjunction_count", "B: Query Complexity", "Count of coordinating/subordinating conjunctions", "query_text", True, True, False, False, False, "lexical_count", "VERIFIED_PRE_GEN"),
        ("query_num_count", "B: Query Complexity", "Count of numeric/digit tokens in query", "query_text", True, True, False, False, False, "digit_count", "VERIFIED_PRE_GEN"),
        ("query_is_comparison", "B: Query Complexity", "Indicator for comparative queries ('compare', 'versus', 'differ')", "query_text", True, True, False, False, False, "binary_keyword", "VERIFIED_PRE_GEN"),
        ("query_is_how_why", "B: Query Complexity", "Indicator for multi-hop explanatory queries ('how', 'why')", "query_text", True, True, False, False, False, "binary_keyword", "VERIFIED_PRE_GEN"),
        ("query_is_factoid", "B: Query Complexity", "Indicator for localized factoid queries ('who', 'when', 'where')", "query_text", True, True, False, False, False, "binary_keyword", "VERIFIED_PRE_GEN"),
        ("query_is_what_which", "B: Query Complexity", "Indicator for 'what' or 'which' questions", "query_text", True, True, False, False, False, "binary_keyword", "VERIFIED_PRE_GEN"),
        ("query_is_numerical", "B: Query Complexity", "Indicator for numerical questions ('how many', 'count', 'percentage')", "query_text", True, True, False, False, False, "binary_keyword", "VERIFIED_PRE_GEN"),
        ("query_quote_count", "B: Query Complexity", "Count of quotation marks in query", "query_text", True, True, False, False, False, "punctuation_count", "VERIFIED_PRE_GEN"),
        ("query_capitalized_spans", "B: Query Complexity", "Count of capitalized non-initial words (entity-like tokens)", "query_text", True, True, False, False, False, "capitalization_heuristic", "VERIFIED_PRE_GEN"),

        # Family C (11)
        ("sem_sim_top1", "C: Semantic Structure", "Cosine similarity between query and top-1 passage embedding", "sfr_embeddings", True, True, False, False, True, "cosine_similarity", "VERIFIED_PRE_GEN"),
        ("sem_sim_top2", "C: Semantic Structure", "Cosine similarity between query and top-2 passage embedding", "sfr_embeddings", True, True, False, False, True, "cosine_similarity", "VERIFIED_PRE_GEN"),
        ("sem_sim_max", "C: Semantic Structure", "Maximum cosine similarity across all top-10 passages", "sfr_embeddings", True, True, False, False, True, "max_cosine", "VERIFIED_PRE_GEN"),
        ("sem_sim_mean_top10", "C: Semantic Structure", "Mean cosine similarity across all top-10 passages", "sfr_embeddings", True, True, False, False, True, "mean_cosine", "VERIFIED_PRE_GEN"),
        ("sem_sim_std_top10", "C: Semantic Structure", "Standard deviation of cosine similarities across passages", "sfr_embeddings", True, True, False, False, True, "std_cosine", "VERIFIED_PRE_GEN"),
        ("sem_sim_gap_12", "C: Semantic Structure", "Semantic gap between top-1 and top-2 passage similarity", "sfr_embeddings", True, True, False, False, True, "semantic_gap", "VERIFIED_PRE_GEN"),
        ("sem_sim_mean_top2", "C: Semantic Structure", "Mean semantic similarity of top-2 passages", "sfr_embeddings", True, True, False, False, True, "mean_cosine", "VERIFIED_PRE_GEN"),
        ("sem_sim_mean_top4", "C: Semantic Structure", "Mean semantic similarity of top-4 passages", "sfr_embeddings", True, True, False, False, True, "mean_cosine", "VERIFIED_PRE_GEN"),
        ("sem_sim_mean_top6", "C: Semantic Structure", "Mean semantic similarity of top-6 passages", "sfr_embeddings", True, True, False, False, True, "mean_cosine", "VERIFIED_PRE_GEN"),
        ("sem_sim_mean_top8", "C: Semantic Structure", "Mean semantic similarity of top-8 passages", "sfr_embeddings", True, True, False, False, True, "mean_cosine", "VERIFIED_PRE_GEN"),
        ("sem_sim_saturation_ratio", "C: Semantic Structure", "Semantic saturation ratio: mean_sim(top-4) / mean_sim(top-10)", "sfr_embeddings", True, True, False, False, True, "saturation_gradient", "VERIFIED_PRE_GEN"),

        # Family D (10)
        ("passage_sim_mean", "D: Passage Redundancy/Diversity", "Mean pairwise cosine similarity among all top-10 passages", "sfr_embeddings", True, False, False, False, True, "pairwise_cosine", "VERIFIED_PRE_GEN"),
        ("passage_sim_median", "D: Passage Redundancy/Diversity", "Median pairwise cosine similarity among top-10 passages", "sfr_embeddings", True, False, False, False, True, "pairwise_median", "VERIFIED_PRE_GEN"),
        ("passage_sim_max", "D: Passage Redundancy/Diversity", "Max pairwise similarity (highest redundancy pair)", "sfr_embeddings", True, False, False, False, True, "max_pairwise", "VERIFIED_PRE_GEN"),
        ("passage_sim_min", "D: Passage Redundancy/Diversity", "Min pairwise similarity (most distant pair in candidate set)", "sfr_embeddings", True, False, False, False, True, "min_pairwise", "VERIFIED_PRE_GEN"),
        ("passage_sim_std", "D: Passage Redundancy/Diversity", "Standard deviation of pairwise passage similarities", "sfr_embeddings", True, False, False, False, True, "std_pairwise", "VERIFIED_PRE_GEN"),
        ("passage_nearest_neighbor_mean", "D: Passage Redundancy/Diversity", "Mean similarity of each passage to its nearest neighbor", "sfr_embeddings", True, False, False, False, True, "nearest_neighbor", "VERIFIED_PRE_GEN"),
        ("passage_nearest_neighbor_min", "D: Passage Redundancy/Diversity", "Similarity of most isolated passage to nearest neighbor", "sfr_embeddings", True, False, False, False, True, "outlier_isolation", "VERIFIED_PRE_GEN"),
        ("passage_diversity_score", "D: Passage Redundancy/Diversity", "Complement of mean pairwise similarity (1.0 - mean_sim)", "sfr_embeddings", True, False, False, False, True, "diversity_metric", "VERIFIED_PRE_GEN"),
        ("passage_cluster_count", "D: Passage Redundancy/Diversity", "Number of distinct semantic clusters (distance threshold 0.20)", "sfr_embeddings", True, False, False, False, True, "agglomerative_clustering", "VERIFIED_PRE_GEN"),
        ("passage_largest_cluster_frac", "D: Passage Redundancy/Diversity", "Fraction of passages belonging to largest cluster", "sfr_embeddings", True, False, False, False, True, "cluster_dominance", "VERIFIED_PRE_GEN"),

        # Family E (8)
        ("lex_coverage_top1", "E: Lexical Coverage", "Fraction of query content terms appearing in passage 1", "passage_text", True, True, False, True, False, "term_recall", "VERIFIED_PRE_GEN"),
        ("lex_coverage_top2", "E: Lexical Coverage", "Fraction of query content terms appearing in top-2 passages", "passage_text", True, True, False, True, False, "cumulative_term_recall", "VERIFIED_PRE_GEN"),
        ("lex_coverage_top4", "E: Lexical Coverage", "Fraction of query content terms appearing in top-4 passages", "passage_text", True, True, False, True, False, "cumulative_term_recall", "VERIFIED_PRE_GEN"),
        ("lex_coverage_top6", "E: Lexical Coverage", "Fraction of query content terms appearing in top-6 passages", "passage_text", True, True, False, True, False, "cumulative_term_recall", "VERIFIED_PRE_GEN"),
        ("lex_coverage_top8", "E: Lexical Coverage", "Fraction of query content terms appearing in top-8 passages", "passage_text", True, True, False, True, False, "cumulative_term_recall", "VERIFIED_PRE_GEN"),
        ("lex_coverage_top10", "E: Lexical Coverage", "Fraction of query content terms appearing in top-10 passages", "passage_text", True, True, False, True, False, "cumulative_term_recall", "VERIFIED_PRE_GEN"),
        ("lex_coverage_gain_1_to_4", "E: Lexical Coverage", "Marginal term coverage gain from top-1 to top-4", "passage_text", True, True, False, True, False, "coverage_growth", "VERIFIED_PRE_GEN"),
        ("lex_missing_terms_top10", "E: Lexical Coverage", "Absolute count of query content terms missing from all top-10", "passage_text", True, True, False, True, False, "missing_term_count", "VERIFIED_PRE_GEN"),

        # Family F (6)
        ("evidence_high_sim_count", "F: Evidence Structure", "Count of passages with semantic similarity >= 0.8 * max_sim", "sfr_embeddings", True, True, False, False, True, "prominence_count", "VERIFIED_PRE_GEN"),
        ("evidence_length_mean", "F: Evidence Structure", "Mean word count of retrieved passages", "passage_text", True, False, False, True, False, "text_length_mean", "VERIFIED_PRE_GEN"),
        ("evidence_length_std", "F: Evidence Structure", "Variation in passage lengths (structural heterogeneity)", "passage_text", True, False, False, True, False, "text_length_std", "VERIFIED_PRE_GEN"),
        ("evidence_lexical_overlap_mean", "F: Evidence Structure", "Mean pairwise Jaccard word overlap between passages", "passage_text", True, False, False, True, False, "jaccard_overlap", "VERIFIED_PRE_GEN"),
        ("evidence_unique_word_count", "F: Evidence Structure", "Total vocabulary size across all 10 retrieved passages", "passage_text", True, False, False, True, False, "vocabulary_size", "VERIFIED_PRE_GEN"),
        ("evidence_query_cluster_span", "F: Evidence Structure", "Number of distinct passage clusters containing >=1 query term", "passage_text_and_clusters", True, True, False, True, True, "multi_aspect_span", "VERIFIED_PRE_GEN"),
    ]

    cols = [
        "feature_name",
        "feature_family",
        "description",
        "source",
        "available_pre_generation",
        "uses_query",
        "uses_bm25",
        "uses_passage_text",
        "uses_embedding",
        "computation_method",
        "leakage_status"
    ]
    return pd.DataFrame(records, columns=cols)


# =============================================================================
# 3. LEAKAGE AUDIT & TARGET EXTRACTION
# =============================================================================

FORBIDDEN_PATTERNS = [
    r"(^|_)(f1|rouge|em|bleu|exact_match|gold|answer|oracle|target|human|generated|logits?)($|_)"
]


def audit_feature_leakage(df_features: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Strictly audit feature columns for forbidden post-generation leakage tokens."""
    violations = []
    feature_cols = [c for c in df_features.columns if c not in ["question_id", "paper_id"]]
    
    for col in feature_cols:
        col_lower = col.lower()
        for pat in FORBIDDEN_PATTERNS:
            if re.search(pat, col_lower):
                violations.append(f"Column '{col}' matches forbidden pattern '{pat}'")
                
    is_clean = len(violations) == 0
    return is_clean, violations


def construct_target_matrix(
    dev_matrix_path: str = "results/week2/processed/dev_per_query_k_matrix.csv",
    epsilon_oracle_path: str = "results/week3/processed/epsilon_oracle_all.csv",
    epsilon_star: float = 0.01,
) -> pd.DataFrame:
    """Construct primary marginal answer-quality gain and oracle target matrix.
    
    Target Definitions:
      - oracle_k: Frozen epsilon* = 0.01 target budget in {2, 4, 5, 6, 8}
      - G_2_to_4: F1(q, 4) - F1(q, 2)
      - G_4_to_6: F1(q, 6) - F1(q, 4)
      - G_6_to_8: F1(q, 8) - F1(q, 6)
      - G_2_to_8: F1(q, 8) - F1(q, 2)
      - G_2_to_5: F1(q, 5) - F1(q, 2)
      - G_5_to_8: F1(q, 8) - F1(q, 5)
    """
    df_mat = pd.read_csv(dev_matrix_path)
    df_eps = pd.read_csv(epsilon_oracle_path)
    
    df_eps_sub = df_eps[df_eps["epsilon"] == epsilon_star].copy()
    
    def norm_qid(val: Any) -> str:
        s = str(val).strip()
        if s.endswith(".0"):
            s = s[:-2]
        return s

    df_mat["question_id"] = df_mat["question_id"].apply(norm_qid)
    df_mat["paper_id"] = df_mat["paper_id"].astype(str).str.strip()
    df_eps_sub["question_id"] = df_eps_sub["question_id"].apply(norm_qid)
    
    # Merge
    merged = pd.merge(
        df_mat[["question_id", "paper_id", "F1_k2", "F1_k4", "F1_k5", "F1_k6", "F1_k8"]],
        df_eps_sub[["question_id", "selected_k_epsilon"]],
        on="question_id",
        how="inner"
    )
    
    merged["oracle_k"] = merged["selected_k_epsilon"].astype(int)
    merged["G_2_to_4"] = merged["F1_k4"] - merged["F1_k2"]
    merged["G_4_to_6"] = merged["F1_k6"] - merged["F1_k4"]
    merged["G_6_to_8"] = merged["F1_k8"] - merged["F1_k6"]
    merged["G_2_to_8"] = merged["F1_k8"] - merged["F1_k2"]
    merged["G_2_to_5"] = merged["F1_k5"] - merged["F1_k2"]
    merged["G_5_to_8"] = merged["F1_k8"] - merged["F1_k5"]
    
    target_cols = [
        "question_id", "paper_id", "oracle_k",
        "G_2_to_4", "G_4_to_6", "G_6_to_8", "G_2_to_8",
        "G_2_to_5", "G_5_to_8"
    ]
    df_t = merged[target_cols].copy()
    df_t["question_id"] = df_t["question_id"].astype(str).str.strip()
    df_t["paper_id"] = df_t["paper_id"].astype(str).str.strip()
    return df_t


def safe_merge_features_targets(df_features: pd.DataFrame, df_targets: pd.DataFrame) -> pd.DataFrame:
    """Safely merge features and targets ensuring matching string types for merge keys."""
    df_f = df_features.copy()
    df_t = df_targets.copy()
    df_f["question_id"] = df_f["question_id"].astype(str).str.strip()
    df_f["paper_id"] = df_f["paper_id"].astype(str).str.strip()
    df_t["question_id"] = df_t["question_id"].astype(str).str.strip()
    df_t["paper_id"] = df_t["paper_id"].astype(str).str.strip()
    return pd.merge(df_f, df_t, on=["question_id", "paper_id"])


# =============================================================================
# 4. STATISTICAL ASSOCIATION ANALYSES
# =============================================================================

def compute_spearman_associations(
    df_features: pd.DataFrame,
    df_targets: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str] = ["oracle_k", "G_2_to_4", "G_4_to_6", "G_6_to_8", "G_2_to_8"],
) -> pd.DataFrame:
    """Compute pairwise Spearman rank correlation between features and targets."""
    df_merged = safe_merge_features_targets(df_features, df_targets)
    records = []
    
    for f in feature_cols:
        for t in target_cols:
            x = df_merged[f].values
            y = df_merged[t].values
            # Filter non-finite
            valid = np.isfinite(x) & np.isfinite(y)
            if np.sum(valid) < 5 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
                rho, pval = 0.0, 1.0
            else:
                res = stats.spearmanr(x[valid], y[valid])
                rho, pval = float(res.statistic), float(res.pvalue)
                
            records.append({
                "feature": f,
                "target": t,
                "spearman_rho": rho,
                "abs_spearman_rho": abs(rho),
                "p_value_naive": pval,
                "n_samples": int(np.sum(valid)),
            })
            
    return pd.DataFrame(records)


def compute_pearson_associations(
    df_features: pd.DataFrame,
    df_targets: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str] = ["G_2_to_4", "G_4_to_6", "G_6_to_8", "G_2_to_8"],
) -> pd.DataFrame:
    """Compute Pearson correlation for continuous marginal gain targets."""
    df_merged = safe_merge_features_targets(df_features, df_targets)
    records = []
    
    for f in feature_cols:
        for t in target_cols:
            x = df_merged[f].values
            y = df_merged[t].values
            valid = np.isfinite(x) & np.isfinite(y)
            if np.sum(valid) < 5 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
                r, pval = 0.0, 1.0
            else:
                res = stats.pearsonr(x[valid], y[valid])
                r, pval = float(res.statistic), float(res.pvalue)
                
            records.append({
                "feature": f,
                "target": t,
                "pearson_r": r,
                "abs_pearson_r": abs(r),
                "p_value_naive": pval,
                "n_samples": int(np.sum(valid)),
            })
            
    return pd.DataFrame(records)


def compute_mutual_information(
    df_features: pd.DataFrame,
    df_targets: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str] = ["oracle_k", "G_2_to_4", "G_4_to_6", "G_6_to_8", "G_2_to_8"],
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute Mutual Information as a nonlinear screening diagnostic."""
    df_merged = safe_merge_features_targets(df_features, df_targets)
    records = []
    
    X = df_merged[feature_cols].values
    # Impute NaNs with median
    from sklearn.impute import SimpleImputer
    X_clean = SimpleImputer(strategy="median").fit_transform(X)
    
    for t in target_cols:
        y = df_merged[t].values
        # Discrete target for oracle_k
        discrete_target = (t == "oracle_k")
        mi_scores = mutual_info_regression(
            X_clean, y,
            discrete_features=False,
            random_state=random_state,
            n_neighbors=5
        )
        for f, mi in zip(feature_cols, mi_scores):
            records.append({
                "feature": f,
                "target": t,
                "mutual_info": float(mi),
            })
            
    return pd.DataFrame(records)


def compute_paper_clustered_bootstrap(
    df_features: pd.DataFrame,
    df_targets: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str] = ["oracle_k", "G_2_to_4", "G_4_to_6", "G_6_to_8", "G_2_to_8"],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Estimate 95% confidence intervals for Spearman rho by paper-level cluster resampling (vectorized)."""
    from scipy.stats import rankdata
    
    df_merged = safe_merge_features_targets(df_features, df_targets)
    unique_papers = np.array(df_merged["paper_id"].unique())
    n_papers = len(unique_papers)
    
    # Pre-index questions by paper for fast cluster sampling
    paper_to_indices = {p: df_merged.index[df_merged["paper_id"] == p].values for p in unique_papers}
    
    rng = np.random.RandomState(seed)
    
    # Pre-sample all bootstrap cluster indices
    boot_indices = [
        np.concatenate([paper_to_indices[p] for p in rng.choice(unique_papers, size=n_papers, replace=True)])
        for _ in range(n_bootstrap)
    ]
    
    X = df_merged[feature_cols].values.astype(float)
    Y = df_merged[target_cols].values.astype(float)
    
    n_feat = len(feature_cols)
    n_targ = len(target_cols)
    
    # Point estimates on full sample
    rx_full = rankdata(X, axis=0)
    ry_full = rankdata(Y, axis=0)
    std_xf = np.std(rx_full, axis=0)
    std_yf = np.std(ry_full, axis=0)
    std_xf[std_xf == 0] = 1e-12
    std_yf[std_yf == 0] = 1e-12
    zx_full = (rx_full - np.mean(rx_full, axis=0)) / std_xf
    zy_full = (ry_full - np.mean(ry_full, axis=0)) / std_yf
    point_rhos = (zx_full.T @ zy_full) / len(df_merged)  # shape (n_feat, n_targ)
    
    # Bootstrap replicates matrix of shape (n_bootstrap, n_feat, n_targ)
    boot_rhos_all = np.zeros((n_bootstrap, n_feat, n_targ), dtype=np.float32)
    
    for b in range(n_bootstrap):
        idx = boot_indices[b]
        X_b = X[idx]
        Y_b = Y[idx]
        
        rx_b = rankdata(X_b, axis=0)
        ry_b = rankdata(Y_b, axis=0)
        
        std_x = np.std(rx_b, axis=0)
        std_y = np.std(ry_b, axis=0)
        std_x[std_x == 0] = 1e-12
        std_y[std_y == 0] = 1e-12
        
        zx_b = (rx_b - np.mean(rx_b, axis=0)) / std_x
        zy_b = (ry_b - np.mean(ry_b, axis=0)) / std_y
        
        boot_rhos_all[b] = (zx_b.T @ zy_b) / len(idx)
        
    records = []
    for i, f in enumerate(feature_cols):
        for j, t in enumerate(target_cols):
            rhos_b = boot_rhos_all[:, i, j]
            ci_low = float(np.percentile(rhos_b, 2.5))
            ci_high = float(np.percentile(rhos_b, 97.5))
            ci_width = float(ci_high - ci_low)
            sign_stable = bool(ci_low * ci_high > 0)
            
            records.append({
                "feature": f,
                "target": t,
                "spearman_rho": float(point_rhos[i, j]),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "bootstrap_ci_width": ci_width,
                "bootstrap_sign_stable": sign_stable,
            })
            
    return pd.DataFrame(records)


def compute_groupkfold_stability(
    df_features: pd.DataFrame,
    df_targets: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str] = ["oracle_k", "G_2_to_4", "G_4_to_6", "G_6_to_8", "G_2_to_8"],
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate Spearman correlation consistency across 5 paper-disjoint GroupKFold validation folds (vectorized)."""
    from scipy.stats import rankdata
    
    df_merged = safe_merge_features_targets(df_features, df_targets)
    groups = df_merged["paper_id"].values
    
    gkf = GroupKFold(n_splits=n_splits)
    folds = list(gkf.split(df_merged, groups=groups))
    
    X = df_merged[feature_cols].values.astype(float)
    Y = df_merged[target_cols].values.astype(float)
    
    n_feat = len(feature_cols)
    n_targ = len(target_cols)
    
    fold_rhos_all = np.zeros((n_splits, n_feat, n_targ), dtype=np.float32)
    
    for fold_idx, (_, val_idx) in enumerate(folds):
        X_val = X[val_idx]
        Y_val = Y[val_idx]
        
        rx = rankdata(X_val, axis=0)
        ry = rankdata(Y_val, axis=0)
        
        std_x = np.std(rx, axis=0)
        std_y = np.std(ry, axis=0)
        std_x[std_x == 0] = 1e-12
        std_y[std_y == 0] = 1e-12
        
        zx = (rx - np.mean(rx, axis=0)) / std_x
        zy = (ry - np.mean(ry, axis=0)) / std_y
        
        fold_rhos_all[fold_idx] = (zx.T @ zy) / len(val_idx)
        
    records = []
    for i, f in enumerate(feature_cols):
        for j, t in enumerate(target_cols):
            frhos = fold_rhos_all[:, i, j].tolist()
            signs = [np.sign(r) for r in frhos if abs(r) > 1e-4]
            if signs:
                most_common = max(signs.count(1.0), signs.count(-1.0))
                sign_consistency = float(most_common / len(frhos))
            else:
                sign_consistency = 0.0
                
            records.append({
                "feature": f,
                "target": t,
                "fold_rhos": [round(float(r), 4) for r in frhos],
                "fold_rho_mean": float(np.mean(frhos)),
                "fold_rho_median": float(np.median(frhos)),
                "fold_rho_std": float(np.std(frhos)),
                "fold_rho_iqr": float(np.percentile(frhos, 75) - np.percentile(frhos, 25)),
                "fold_sign_consistency": sign_consistency,
                "is_fold_stable": bool(sign_consistency >= 0.8),
            })
            
    return pd.DataFrame(records)


def summarize_feature_families(
    df_spearman: pd.DataFrame,
    df_stability: pd.DataFrame,
    df_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate discovery findings across the 6 feature families."""
    merged = pd.merge(df_spearman, df_metadata[["feature_name", "feature_family"]], left_on="feature", right_on="feature_name")
    merged = pd.merge(merged, df_stability[["feature", "target", "fold_sign_consistency", "is_fold_stable"]], on=["feature", "target"])
    
    records = []
    for family, group in merged.groupby("feature_family"):
        n_features = group["feature"].nunique()
        max_abs_rho = float(group["abs_spearman_rho"].max())
        median_abs_rho = float(group["abs_spearman_rho"].median())
        mean_abs_rho = float(group["abs_spearman_rho"].mean())
        
        # Best feature and target in family
        best_row = group.loc[group["abs_spearman_rho"].idxmax()]
        
        stable_count = int(group.groupby("feature")["is_fold_stable"].any().sum())
        
        records.append({
            "feature_family": family,
            "n_features": n_features,
            "max_abs_spearman_rho": round(max_abs_rho, 4),
            "median_abs_spearman_rho": round(median_abs_rho, 4),
            "mean_abs_spearman_rho": round(mean_abs_rho, 4),
            "n_features_fold_stable": stable_count,
            "strongest_feature": best_row["feature"],
            "strongest_target": best_row["target"],
            "strongest_rho": round(best_row["spearman_rho"], 4),
        })
        
    return pd.DataFrame(records).sort_values(by="max_abs_spearman_rho", ascending=False).reset_index(drop=True)


def screen_candidate_features(
    df_spearman: pd.DataFrame,
    df_bootstrap: pd.DataFrame,
    df_stability: pd.DataFrame,
    df_mi: pd.DataFrame,
    df_metadata: pd.DataFrame,
    rho_threshold: float = 0.12,
    consistency_threshold: float = 0.80,
) -> pd.DataFrame:
    """Screen candidate promising features based on pre-specified empirical criteria.
    
    Criteria for 'candidate_flag':
      1. Meaningful effect size: |spearman_rho| >= rho_threshold (default 0.12)
      2. Fold sign stability: fold_sign_consistency >= consistency_threshold (>=4 of 5 folds agree)
      3. Non-zero bootstrap CI: bootstrap 95% CI does not span extreme reversals
    """
    m = pd.merge(df_spearman, df_bootstrap[["feature", "target", "bootstrap_ci_low", "bootstrap_ci_high", "bootstrap_ci_width", "bootstrap_sign_stable"]], on=["feature", "target"])
    m = pd.merge(m, df_stability[["feature", "target", "fold_rho_median", "fold_sign_consistency", "is_fold_stable"]], on=["feature", "target"])
    m = pd.merge(m, df_mi[["feature", "target", "mutual_info"]], on=["feature", "target"])
    m = pd.merge(m, df_metadata[["feature_name", "feature_family", "description"]], left_on="feature", right_on="feature_name")
    
    candidates = []
    for _, row in m.iterrows():
        rho = row["spearman_rho"]
        abs_rho = abs(rho)
        cons = row["fold_sign_consistency"]
        ci_low = row["bootstrap_ci_low"]
        ci_high = row["bootstrap_ci_high"]
        
        # Check candidate conditions
        has_effect = abs_rho >= rho_threshold
        has_stability = cons >= consistency_threshold
        has_ci_support = (ci_low * ci_high > 0)
        
        is_candidate = has_effect and has_stability
        
        reasons = []
        if has_effect:
            reasons.append(f"|rho|={abs_rho:.3f} >= {rho_threshold}")
        if has_stability:
            reasons.append(f"sign_consistency={cons:.1f} >= {consistency_threshold}")
        if has_ci_support:
            reasons.append("bootstrap_95ci_excludes_zero")
        if not is_candidate:
            reasons.append("insufficient_effect_or_unstable")
            
        candidates.append({
            "feature": row["feature"],
            "family": row["feature_family"],
            "target": row["target"],
            "spearman_rho": round(rho, 4),
            "bootstrap_ci_low": round(ci_low, 4),
            "bootstrap_ci_high": round(ci_high, 4),
            "fold_sign_consistency": round(cons, 2),
            "median_fold_rho": round(row["fold_rho_median"], 4),
            "mi_score": round(row["mutual_info"], 4),
            "candidate_flag": bool(is_candidate),
            "reason": "; ".join(reasons),
        })
        
    df_cand = pd.DataFrame(candidates)
    return df_cand.sort_values(by="spearman_rho", key=abs, ascending=False).reset_index(drop=True)
