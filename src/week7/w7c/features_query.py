"""Feature Family A: Pre-generation Query Complexity Features for Week 7-C.

Extracts deterministic linguistic, structural, and semantic complexity features
directly from query text without relying on an external LLM.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Union

import numpy as np
import pandas as pd

# Lexicons for query classification
WH_WORDS = {"what", "which", "who", "whom", "whose", "where", "when", "why", "how"}
YES_NO_STARTERS = {
    "is", "are", "was", "were", "do", "does", "did", "can", "could",
    "will", "would", "shall", "should", "has", "have", "had", "may", "might", "must"
}
CONJUNCTIONS = {
    "and", "or", "but", "nor", "so", "for", "yet", "although", "because",
    "since", "unless", "while", "whereas", "whether", "after", "before", "though"
}
COMPARATIVE_TERMS = {
    "compare", "difference", "different", "differ", "versus", "vs", "better",
    "worse", "higher", "lower", "greater", "less", "more", "faster", "slower"
}
NUMERICAL_TERMS = {
    "how many", "how much", "number", "count", "percentage", "proportion",
    "ratio", "total", "sum", "average", "frequency", "score"
}
CAUSAL_TERMS = {
    "why", "how", "cause", "caused", "leads to", "lead to", "resulting", "result in",
    "because", "mechanism", "effect", "impact", "influence", "affect"
}


def compute_single_query_features(question: str) -> Dict[str, float]:
    """Compute Family A features for a single query string."""
    q_str = str(question).strip()
    words = re.findall(r"\b\w+\b", q_str)
    lower_words = [w.lower() for w in words]
    n_tokens = len(words)
    n_chars = len(q_str)

    # Sentences and clauses
    sentences = [s for s in re.split(r"[.!?]+", q_str) if s.strip()]
    n_sentences = max(1, len(sentences))
    n_commas_clauses = len(re.findall(r"[,;:]", q_str))

    # Conjunctions
    n_conjunctions = sum(1 for w in lower_words if w in CONJUNCTIONS)

    # Noun chunks / capitalized entities / acronyms (lightweight regex)
    # Consecutive capitalized words or acronyms (e.g. BERT, F1, SARA, QASPER)
    acronyms = re.findall(r"\b[A-Z0-9]{2,}\b", q_str)
    n_acronyms = len(acronyms)
    acronym_density = n_acronyms / max(1, n_tokens)

    # Capitalized non-sentence-starter words as lightweight named entity proxy
    cap_words = [w for idx, w in enumerate(words) if idx > 0 and w[0].isupper() and w.isalpha()]
    n_named_entities = len(cap_words) + n_acronyms

    # Noun chunks proxy: sequence of capitalized words or sequences of content words
    noun_phrases = re.findall(r"(?:[A-Z][a-z]+\s*){2,}", q_str)
    n_noun_chunks = len(noun_phrases) + len(re.findall(r"\b(?:the|a|an)\s+\w+(?:\s+\w+)?\b", q_str, re.IGNORECASE))

    # Indicators
    first_word = lower_words[0] if lower_words else ""
    is_wh = 1.0 if (first_word in WH_WORDS or any(w in WH_WORDS for w in lower_words[:3])) else 0.0
    is_yes_no = 1.0 if (first_word in YES_NO_STARTERS) else 0.0

    lower_text = q_str.lower()
    is_comparative = 1.0 if (any(term in lower_text for term in COMPARATIVE_TERMS) or bool(re.search(r"\b\w+er\b", lower_text))) else 0.0
    is_numerical = 1.0 if (any(term in lower_text for term in NUMERICAL_TERMS) or bool(re.search(r"\b\d+\b", q_str))) else 0.0
    is_causal_why = 1.0 if any(term in lower_text for term in CAUSAL_TERMS) else 0.0
    is_multi_part = 1.0 if (q_str.count("?") > 1 or " and " in lower_text or "both" in lower_text or "as well as" in lower_text) else 0.0

    # Lexical diversity
    unique_words = set(lower_words)
    lexical_diversity = len(unique_words) / max(1, n_tokens)
    avg_word_length = float(np.mean([len(w) for w in words])) if words else 0.0

    return {
        "q_len_tokens": float(n_tokens),
        "q_len_chars": float(n_chars),
        "q_num_sentences": float(n_sentences),
        "q_num_noun_chunks": float(n_noun_chunks),
        "q_num_named_entities": float(n_named_entities),
        "q_num_conjunctions": float(n_conjunctions),
        "q_num_commas_clauses": float(n_commas_clauses),
        "q_is_wh_question": float(is_wh),
        "q_is_yes_no": float(is_yes_no),
        "q_is_comparative": float(is_comparative),
        "q_is_numerical": float(is_numerical),
        "q_is_causal_why": float(is_causal_why),
        "q_is_multi_part": float(is_multi_part),
        "q_acronym_density": float(acronym_density),
        "q_lexical_diversity": float(lexical_diversity),
        "q_avg_word_length": float(avg_word_length),
    }


def extract_query_complexity_features(
    question_ids: Sequence[str],
    questions: Sequence[str],
) -> pd.DataFrame:
    """Extract Family A query complexity features for a collection of questions.

    Args:
        question_ids: Sequence of question identifiers.
        questions: Sequence of question texts.

    Returns:
        DataFrame with columns ['question_id', 'q_len_tokens', ..., 'q_avg_word_length'].
    """
    rows: List[Dict[str, Any]] = []
    for qid, q_text in zip(question_ids, questions):
        feat = compute_single_query_features(q_text)
        feat["question_id"] = str(qid).strip()
        rows.append(feat)

    df_feats = pd.DataFrame(rows)
    # Ensure question_id is first column
    cols = ["question_id"] + [c for c in df_feats.columns if c != "question_id"]
    return df_feats[cols]
