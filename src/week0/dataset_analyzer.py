"""Flexible Dataset Analyzer for QA/RAG Datasets.

Extracts dataset structure, observable query characteristics, gold evidence
properties, and retrieval dynamics across train and dev splits.
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.week0.dataset_adapter import DatasetAdapter

logger = logging.getLogger(__name__)

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
    "relative", "outperform", "outperforms", "contrast", "contrasting", "baseline"
}

NUMBER_REGEX = re.compile(r"\b\d+(\.\d+)?\b|\b(one|two|three|four|five|six|seven|eight|nine|ten|first|second|third)\b", re.IGNORECASE)


def tokenize_clean(text: str) -> List[str]:
    """Extract lowercase alphanumeric tokens."""
    return re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())


def extract_content_words(text: str) -> List[str]:
    """Extract non-stopword tokens of length > 1."""
    return [t for t in tokenize_clean(text) if t not in STOPWORDS and len(t) > 1]


def classify_wh_type(query: str) -> str:
    """Classify question by grammatical/interrogative prefix."""
    q_lower = query.lower().strip()
    words = tokenize_clean(q_lower)
    if not words:
        return "other"

    first_word = words[0]
    if first_word in ("what", "what's"):
        return "what"
    elif first_word in ("how", "how's"):
        if len(words) > 1 and words[1] in ("many", "much"):
            return "how_many"
        return "how"
    elif first_word in ("which",):
        return "which"
    elif first_word in ("why", "why's"):
        return "why"
    elif first_word in ("where", "where's"):
        return "where"
    elif first_word in ("who", "whom", "whose"):
        return "who"
    elif first_word in ("when", "when's"):
        return "when"
    elif first_word in ("is", "are", "was", "were", "do", "does", "did", "can", "could", "has", "have", "had", "will", "would", "should"):
        return "boolean"
    
    # Check if a WH-word appears in the first 3 tokens
    for w in words[:3]:
        if w in ("what", "how", "which", "why", "where", "who", "when"):
            return w

    return "other"


def extract_query_characteristics(query: str) -> Dict[str, Any]:
    """Compute observable query characteristics without claiming to measure true complexity."""
    tokens = tokenize_clean(query)
    content_words = extract_content_words(query)
    word_count = len(tokens)
    unique_words = len(set(tokens))
    ttr = float(unique_words / word_count) if word_count > 0 else 0.0

    wh_type = classify_wh_type(query)
    has_number = bool(NUMBER_REGEX.search(query))
    has_comparison = any(term in tokens for term in COMPARISON_TERMS)
    has_conjunction = any(term in tokens for term in CONJUNCTIONS)

    return {
        "query_word_count": word_count,
        "query_char_count": len(query),
        "query_content_words": len(content_words),
        "query_ttr": round(ttr, 4),
        "wh_type": wh_type,
        "is_numerical": has_number,
        "is_comparison": has_comparison,
        "is_multipart": has_conjunction,
    }


def compute_lexical_overlap(text1: str, text2: str) -> Tuple[float, float]:
    """Compute (Jaccard similarity, coverage of text1 in text2) using content words."""
    set1 = set(extract_content_words(text1))
    set2 = set(extract_content_words(text2))
    if not set1 or not set2:
        return 0.0, 0.0

    intersection = set1 & set2
    union = set1 | set2
    jaccard = float(len(intersection) / len(union))
    coverage = float(len(intersection) / len(set1))
    return round(jaccard, 4), round(coverage, 4)


def compute_evidence_redundancy(evidence_texts: List[str]) -> float:
    """Compute mean pairwise Jaccard similarity across multiple evidence passages."""
    if len(evidence_texts) <= 1:
        return 0.0

    token_sets = [set(extract_content_words(t)) for t in evidence_texts if t.strip()]
    token_sets = [s for s in token_sets if len(s) > 0]
    if len(token_sets) <= 1:
        return 0.0

    jaccards = []
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            union = token_sets[i] | token_sets[j]
            if union:
                jaccards.append(len(token_sets[i] & token_sets[j]) / len(union))
    return float(np.mean(jaccards)) if jaccards else 0.0


def calculate_evidence_depth(
    evidence_paras: List[str],
    all_doc_paras: List[str],
) -> List[float]:
    """Find the relative depth (0.0 to 1.0) of each evidence paragraph within the paper."""
    if not all_doc_paras or not evidence_paras:
        return []

    depths = []
    # Normalize doc paragraphs for matching
    cleaned_doc = [" ".join(p.split()[:30]).lower() for p in all_doc_paras]
    n_doc = len(all_doc_paras)

    for ev in evidence_paras:
        ev_prefix = " ".join(ev.split()[:30]).lower()
        matched_idx = -1
        # Exact prefix match or fuzzy substring
        for idx, doc_p in enumerate(cleaned_doc):
            if ev_prefix and (ev_prefix in doc_p or doc_p in ev_prefix):
                matched_idx = idx
                break
        if matched_idx >= 0:
            depths.append(round(matched_idx / max(n_doc - 1, 1), 4))
        else:
            # Fallback to Jaccard match
            ev_set = set(extract_content_words(ev))
            best_idx = 0
            best_j = 0.0
            for idx, doc_p in enumerate(all_doc_paras):
                p_set = set(extract_content_words(doc_p))
                if p_set:
                    j = len(ev_set & p_set) / len(ev_set | p_set)
                    if j > best_j:
                        best_j = j
                        best_idx = idx
            if best_j > 0.2:
                depths.append(round(best_idx / max(n_doc - 1, 1), 4))
            else:
                depths.append(0.5)  # neutral default

    return depths


def check_passage_matches_evidence(
    passage_text: str,
    evidence_texts: List[str],
    highlighted_spans: List[str],
    threshold: float = 0.3,
) -> bool:
    """Determine whether a retrieved passage contains or substantially overlaps with gold evidence."""
    p_lower = passage_text.lower()
    
    # 1. Exact substring match of highlighted spans (length > 15)
    for span in highlighted_spans:
        span_clean = span.strip().lower()
        if len(span_clean) > 15 and span_clean in p_lower:
            return True

    # 2. Content word overlap with full evidence paragraphs
    p_words = set(extract_content_words(passage_text))
    if not p_words:
        return False

    for ev in evidence_texts:
        ev_words = set(extract_content_words(ev))
        if not ev_words:
            continue
        # Overlap fraction: what fraction of the gold evidence words appear in the passage?
        overlap = len(p_words & ev_words) / len(ev_words)
        if overlap >= threshold:
            return True
        # Alternatively, what fraction of passage words match evidence?
        p_overlap = len(p_words & ev_words) / len(p_words)
        if p_overlap >= 0.5 and len(p_words & ev_words) >= 10:
            return True

    return False


def analyze_dataset(
    adapter: DatasetAdapter,
    split_name: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, pd.DataFrame]:
    """Execute comprehensive characterization of dataset split.

    Returns a dictionary of DataFrames:
      - 'summary': High-level dataset summary metrics.
      - 'questions': Question-level observable features.
      - 'papers': Paper-level structure distributions.
      - 'evidence': Evidence property metrics.
      - 'retrieval': Retrieval evaluation metrics (if available).
    """
    records = adapter.load_split(split_name)
    logger.info(f"Analyzing split '{split_name}' with {len(records)} records.")

    q_rows: List[Dict[str, Any]] = []
    ev_rows: List[Dict[str, Any]] = []
    ret_rows: List[Dict[str, Any]] = []
    paper_dict: Dict[str, Dict[str, Any]] = {}

    for record in records:
        meta = adapter.get_metadata(record)
        qid = meta["question_id"]
        pid = meta["paper_id"]
        query = adapter.get_query(record)
        answers = adapter.get_gold_answer(record)
        doc = adapter.get_document(record)
        evidence_entries = adapter.get_gold_evidence(record)
        retrieved_passages = adapter.get_retrieved_passages(record)

        # 1. Paper-level accounting
        if pid not in paper_dict:
            all_paras = doc.get("all_paragraphs", [])
            total_doc_words = sum(len(tokenize_clean(p)) for p in all_paras)
            paper_dict[pid] = {
                "paper_id": pid,
                "title": doc.get("title", ""),
                "section_count": len(doc.get("section_names", [])),
                "paragraph_count": len(all_paras),
                "total_doc_words": total_doc_words,
                "estimated_doc_tokens": int(total_doc_words * 1.3),
                "question_count": 0,
            }
        paper_dict[pid]["question_count"] += 1

        # 2. Query characterization
        q_feats = extract_query_characteristics(query)
        primary_ans = answers[0] if answers else ""
        ans_tokens = tokenize_clean(primary_ans)
        
        # Determine question category
        raw_type = meta.get("question_type", "unknown").lower()
        if raw_type in ("extractive", "abstractive", "yes_no", "unanswerable", "free_form"):
            q_category = raw_type
        elif any(e.get("unanswerable", False) for e in evidence_entries):
            q_category = "unanswerable"
        elif any(e.get("yes_no") is not None for e in evidence_entries):
            q_category = "yes_no"
        elif any(e.get("extractive_spans") for e in evidence_entries):
            q_category = "extractive"
        else:
            q_category = "abstractive"

        q_row = {
            "split": split_name,
            "question_id": qid,
            "paper_id": pid,
            "question": query,
            "question_category": q_category,
            "answer_count": len(answers),
            "primary_answer": primary_ans,
            "answer_word_count": len(ans_tokens),
            "answer_char_count": len(primary_ans),
            **q_feats,
        }
        q_rows.append(q_row)

        # 3. Evidence characterization
        flat_ev_paras: List[str] = []
        flat_highlighted: List[str] = []
        is_unanswerable = False

        for entry in evidence_entries:
            flat_ev_paras.extend(entry.get("evidence_paragraphs", []))
            flat_highlighted.extend(entry.get("highlighted_spans", []))
            if entry.get("unanswerable"):
                is_unanswerable = True

        # Remove duplicate evidence paragraphs
        unique_ev_paras = list(dict.fromkeys(flat_ev_paras))
        num_ev_paras = len(unique_ev_paras)
        ev_depths = calculate_evidence_depth(unique_ev_paras, doc.get("all_paragraphs", []))
        mean_depth = float(np.mean(ev_depths)) if ev_depths else np.nan
        ev_redundancy = compute_evidence_redundancy(unique_ev_paras)

        # Evidence text combined
        ev_combined = " ".join(unique_ev_paras)
        jaccard_qe, cov_qe = compute_lexical_overlap(query, ev_combined)

        ev_row = {
            "split": split_name,
            "question_id": qid,
            "paper_id": pid,
            "is_unanswerable": is_unanswerable,
            "num_evidence_paragraphs": num_ev_paras,
            "num_highlighted_spans": len(flat_highlighted),
            "total_evidence_words": sum(len(tokenize_clean(p)) for p in unique_ev_paras),
            "mean_evidence_depth": mean_depth,
            "evidence_redundancy": round(ev_redundancy, 4),
            "query_evidence_jaccard": jaccard_qe,
            "query_evidence_coverage": cov_qe,
        }
        ev_rows.append(ev_row)

        # 4. Retrieval analysis (if retrieved passages exist)
        if retrieved_passages:
            scores = [float(p.get("score", 0.0)) for p in retrieved_passages]
            n_p = len(scores)
            s_sum = float(np.sum(np.maximum(scores, 0.0)))
            s1 = scores[0] if scores else 0.0
            s2 = scores[1] if len(scores) > 1 else 0.0
            delta_12 = float((s1 - s2) / (s1 + 1e-8)) if s1 > 0 else 0.0
            rho_1 = float(s1 / (s_sum + 1e-8)) if s_sum > 0 else 0.0

            # Normalized entropy
            p_probs = np.maximum(scores, 0.0) / (s_sum + 1e-8) if s_sum > 0 else np.full(n_p, 1.0 / n_p)
            p_nz = p_probs[p_probs > 0]
            ent = -float(np.sum(p_nz * np.log(p_nz)))
            ent_norm = float(ent / np.log(max(n_p, 2)))

            # Matching retrieved passages to gold evidence
            hit_ranks = []
            for rank_idx, p in enumerate(retrieved_passages):
                p_text = p.get("text", "")
                if check_passage_matches_evidence(p_text, unique_ev_paras, flat_highlighted):
                    hit_ranks.append(rank_idx + 1)

            first_gold_rank = hit_ranks[0] if hit_ranks else -1
            n_gold = len(unique_ev_paras)

            # Compute Recall_any@k (at least one evidence paragraph retrieved)
            recall_any_1 = 1 if any(r <= 1 for r in hit_ranks) else 0
            recall_any_2 = 1 if any(r <= 2 for r in hit_ranks) else 0
            recall_any_4 = 1 if any(r <= 4 for r in hit_ranks) else 0
            recall_any_6 = 1 if any(r <= 6 for r in hit_ranks) else 0
            recall_any_8 = 1 if any(r <= 8 for r in hit_ranks) else 0
            recall_any_10 = 1 if any(r <= 10 for r in hit_ranks) else 0

            # Compute Recall_all@k (all annotated evidence paragraphs retrieved)
            def _check_all_retrieved(k_lim: int) -> int:
                if n_gold == 0:
                    return 0
                matched_indices = set()
                for p in retrieved_passages[:k_lim]:
                    p_txt = p.get("text", "")
                    for g_idx, g_para in enumerate(unique_ev_paras):
                        if check_passage_matches_evidence(p_txt, [g_para], flat_highlighted):
                            matched_indices.add(g_idx)
                return 1 if len(matched_indices) == n_gold else 0

            recall_all_1 = _check_all_retrieved(1)
            recall_all_2 = _check_all_retrieved(2)
            recall_all_4 = _check_all_retrieved(4)
            recall_all_6 = _check_all_retrieved(6)
            recall_all_8 = _check_all_retrieved(8)
            recall_all_10 = _check_all_retrieved(10)

            ret_row = {
                "question_id": qid,
                "paper_id": pid,
                "num_retrieved": n_p,
                "top1_score": round(s1, 4),
                "top2_score": round(s2, 4),
                "delta_12": round(delta_12, 4),
                "rho_1": round(rho_1, 4),
                "entropy_norm": round(ent_norm, 4),
                "first_gold_rank": first_gold_rank,
                # Explicit naming to prevent conflation between any vs all
                "recall_any_at_1": recall_any_1,
                "recall_any_at_2": recall_any_2,
                "recall_any_at_4": recall_any_4,
                "recall_any_at_6": recall_any_6,
                "recall_any_at_8": recall_any_8,
                "recall_any_at_10": recall_any_10,
                "recall_all_at_1": recall_all_1,
                "recall_all_at_2": recall_all_2,
                "recall_all_at_4": recall_all_4,
                "recall_all_at_6": recall_all_6,
                "recall_all_at_8": recall_all_8,
                "recall_all_at_10": recall_all_10,
                "has_gold_in_top10": int(len(hit_ranks) > 0),
            }
            ret_rows.append(ret_row)

    df_questions = pd.DataFrame(q_rows)
    df_evidence = pd.DataFrame(ev_rows)
    df_papers = pd.DataFrame(list(paper_dict.values()))
    df_retrieval = pd.DataFrame(ret_rows) if ret_rows else pd.DataFrame()

    # 5. Build high-level summary DataFrame
    summary_metrics = {
        "split": split_name,
        "num_papers": len(df_papers),
        "num_questions": len(df_questions),
        "questions_per_paper_mean": round(df_papers["question_count"].mean(), 2),
        "questions_per_paper_median": round(df_papers["question_count"].median(), 2),
        "doc_length_words_mean": round(df_papers["total_doc_words"].mean(), 1),
        "doc_length_words_median": round(df_papers["total_doc_words"].median(), 1),
        "question_length_words_mean": round(df_questions["query_word_count"].mean(), 2),
        "answer_length_words_mean": round(df_questions["answer_word_count"].mean(), 2),
        "pct_unanswerable": round(float((df_evidence["is_unanswerable"].sum() / len(df_evidence)) * 100), 2),
        "pct_extractive": round(float((df_questions["question_category"] == "extractive").sum() / len(df_questions) * 100), 2),
        "pct_abstractive": round(float((df_questions["question_category"].isin(["abstractive", "free_form"])).sum() / len(df_questions) * 100), 2),
        "pct_yes_no": round(float((df_questions["question_category"] == "yes_no").sum() / len(df_questions) * 100), 2),
        "evidence_paras_per_question_mean": round(df_evidence["num_evidence_paragraphs"].mean(), 2),
        "evidence_redundancy_mean": round(df_evidence["evidence_redundancy"].mean(), 4),
        "query_evidence_coverage_mean": round(df_evidence["query_evidence_coverage"].mean(), 4),
    }

    if not df_retrieval.empty:
        summary_metrics["recall_any_at_2"] = round(float(df_retrieval["recall_any_at_2"].mean() * 100), 2)
        summary_metrics["recall_any_at_4"] = round(float(df_retrieval["recall_any_at_4"].mean() * 100), 2)
        summary_metrics["recall_any_at_6"] = round(float(df_retrieval["recall_any_at_6"].mean() * 100), 2)
        summary_metrics["recall_any_at_10"] = round(float(df_retrieval["recall_any_at_10"].mean() * 100), 2)
        summary_metrics["recall_all_at_2"] = round(float(df_retrieval["recall_all_at_2"].mean() * 100), 2)
        summary_metrics["recall_all_at_4"] = round(float(df_retrieval["recall_all_at_4"].mean() * 100), 2)
        summary_metrics["recall_all_at_6"] = round(float(df_retrieval["recall_all_at_6"].mean() * 100), 2)
        summary_metrics["recall_all_at_10"] = round(float(df_retrieval["recall_all_at_10"].mean() * 100), 2)
        summary_metrics["mean_delta_12"] = round(float(df_retrieval["delta_12"].mean()), 4)
        summary_metrics["mean_entropy_norm"] = round(float(df_retrieval["entropy_norm"].mean()), 4)

    df_summary = pd.DataFrame([summary_metrics])

    return {
        "summary": df_summary,
        "questions": df_questions,
        "papers": df_papers,
        "evidence": df_evidence,
        "retrieval": df_retrieval,
    }
