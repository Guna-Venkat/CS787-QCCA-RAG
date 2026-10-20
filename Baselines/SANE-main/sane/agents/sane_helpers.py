"""Helper functions for the SANE (Select-And-Extract) method.

Pure functions for evidence extraction, synopsis generation, document selection,
and keyword extraction.  Prompt templates live in ``sane_prompts.py``.
Stateful orchestration (retrieval, recall tracking, config wiring) lives in
``sane_agent.py`` as methods on ``SANEAgent``.
"""

from __future__ import annotations

import json
import re
from collections import Counter

from sane.llm.openai_client import OpenAIClient
from sane.agents.sane_prompts import (
    EVIDENCE_EXTRACTION_PROMPT,
    DOC_SELECTION_PROMPT,
)

_STOPWORDS: set[str] | None = None


def _get_stopwords() -> set[str]:
    global _STOPWORDS
    if _STOPWORDS is None:
        _STOPWORDS = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "i", "you", "he", "she", "it", "we", "they", "me", "him",
            "her", "us", "them", "my", "your", "his", "its", "our",
            "their", "mine", "yours", "hers", "ours", "theirs",
            "this", "that", "these", "those", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "about",
            "like", "not", "no", "but", "or", "and", "if", "so",
            "than", "too", "very", "just", "then", "now", "also",
            "up", "out", "there", "here", "all", "some", "any",
            "more", "most", "other", "only", "over", "such", "what",
            "when", "where", "who", "how", "which", "really", "get",
            "got", "know", "think", "said", "say", "well", "yeah",
            "ok", "okay", "oh", "um", "uh", "hi",
            "hello", "hey", "going", "went", "need", "want", "one",
        }
    return _STOPWORDS


def extract_evidence(
    llm_client: OpenAIClient,
    query: str,
    formatted_history: str,
    prompt_template: str | None = None,
    max_tokens: int = 4096,
) -> str:
    """Extract minimum necessary evidence from retrieved history as a markdown table.

    Args:
        llm_client: LLM client for the extraction call.
        query: The user's question.
        formatted_history: Pre-formatted retrieved sessions string.
        prompt_template: Custom system prompt (uses EVIDENCE_EXTRACTION_PROMPT if None).
        max_tokens: Max output tokens.

    Returns:
        The extracted evidence as a markdown table string.
    """
    prompt = prompt_template or EVIDENCE_EXTRACTION_PROMPT
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"## Question\n{query}\n\n## Retrieved History\n{formatted_history}\n\nExtract the minimum evidence needed to answer this question as a markdown table."},
    ]
    response = llm_client.chat(messages=messages, max_tokens=max_tokens)
    return response.get("content", "").strip()



def _extract_keywords_tf(
    full_text: str,
    entities: list[str],
    stopwords: set[str],
    keyword_n: int,
    idf: dict[str, float] | None = None,
) -> list[str]:
    """Extract top-N TF or TF-IDF keywords from text, excluding entities and stopwords.

    When *idf* is provided, keywords are ranked by tf × idf instead of pure tf.
    """
    if keyword_n <= 0:
        return []
    tokens = re.findall(r"[^\W\d_]{3,}", full_text.lower())
    entity_set_kw = set(entities)
    filtered = [t for t in tokens if t not in stopwords and t not in entity_set_kw]
    tf = Counter(filtered)
    if idf:
        scored = [(w, c * idf.get(w, 0.0)) for w, c in tf.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [w for w, _ in scored[:keyword_n]]
    return [w for w, _ in tf.most_common(keyword_n)]



def _compute_idf(texts: list[str], stopwords: set[str]) -> dict[str, float]:
    """Compute IDF scores across a corpus of *texts* (typically the wide candidate set)."""
    import math
    N = len(texts)
    if N <= 1:
        return {}
    df: dict[str, int] = {}
    for text in texts:
        tokens = set(re.findall(r"[^\W\d_]{3,}", text.lower()))
        for t in tokens:
            if t not in stopwords:
                df[t] = df.get(t, 0) + 1
    return {t: math.log(N / c) for t, c in df.items()}



def _extract_first_sentence(docs: list[dict]) -> str:
    """Return the first user sentence from *doc* as a brief excerpt."""
    for turn in docs:
        if turn.get("role") != "user":
            continue
        content = turn.get("content", "").strip()
        if not content:
            continue
        # Split on sentence boundaries, take first non-empty
        sents = re.split(r"(?<=[.!?])\s+", content)
        for sent in sents:
            sent = sent.strip()
            if sent and len(sent) >= 10:
                return sent
    return ""



def _extract_best_sentence(
    docs: list[dict],
    query_words: set[str],
    stopwords: set[str],
    top_k: int = 1,
) -> str:
    """Return the top-*k* user sentences ranked by lexical overlap with *query_words*.

    For each sentence in the doc's user turns, compute the overlap between
    its content words and the query's content words.  The sentence with the
    highest overlap (Jaccard-like, normalized by query length) is selected.
    Falls back to the first user sentence when no sentence has overlap > 0.
    """
    if not query_words:
        return _extract_first_sentence(docs)

    # Collect all sentences from user turns with their text
    all_sents: list[tuple[str, float]] = []
    for turn in docs:
        if turn.get("role") != "user":
            continue
        content = turn.get("content", "").strip()
        if not content:
            continue
        for raw in re.split(r"(?<=[.!?])\s+", content):
            sent = raw.strip()
            if len(sent) < 10:
                continue
            sent_words = set(re.findall(r"[^\W\d_]{3,}", sent.lower())) - stopwords
            if not sent_words:
                continue
            # Overlap score: |sent ∩ query| / |query|  (Jaccard numerator, query-normalized)
            overlap = len(sent_words & query_words)
            if overlap > 0:
                all_sents.append((sent, overlap))

    if all_sents:
        all_sents.sort(key=lambda x: x[1], reverse=True)
        return " | ".join(s for s, _ in all_sents[:top_k])

    # Fallback: no lexical overlap — return first sentence
    return _extract_first_sentence(docs)



def generate_doc_synopses(
    docs: list[list[dict]],
    dates: list[str],
    indices: list[int],
    keyword_n: int = 10,
    show_keywords: bool = True,
    show_labels: bool = True,
    labels: list[str] | None = None,
    query: str | None = None,
    include_excerpt: bool = False,
) -> tuple[str, list[int]]:
    """Generate compact one-line synopses for the given session indices.

    Keywords are always TF-IDF weighted over the wide candidate set.

    Args:
        docs: All session turn lists.
        dates: All session date strings.
        indices: Which sessions (by index into docs) to synopsise.
        keyword_n: Top-N TF keywords per session.
        show_keywords: Whether to include keywords=[...] in the output lines.
        show_labels: Whether to use doc labels (titles) in the synopsis header.
        labels: Optional per-session labels (e.g. paragraph titles).
        query: Optional query string for excerpt scoring.
        include_excerpt: Whether to prepend an excerpt.

    Returns:
        (synopses_text, idx_map) where idx_map[i] = the docs index
        for the synopsis on line i (0-based i).
    """
    stopwords = _get_stopwords()
    idx_map: list[int] = list(indices)

    # Build full-text for each session
    full_texts = []
    for idx in indices:
        text = " ".join(t.get("content", "") for t in docs[idx])
        if labels and idx < len(labels):
            label_text = labels[idx].strip("[]")
            text = f"{label_text}. {text}"
        full_texts.append(text)

    # Query content words for lexical excerpt scoring
    query_words: set[str] | None = None
    if query:
        query_words = set(
            w for w in re.findall(r"[^\W\d_]{3,}", query.lower())
            if w not in stopwords
        )

    # Compute IDF across the candidate set (always-on TF-IDF)
    idf = _compute_idf(full_texts, stopwords) if keyword_n > 0 else None

    # Build synopsis lines
    lines: list[str] = []
    for i, (idx, full_text) in enumerate(zip(indices, full_texts)):
        date = dates[idx]

        keywords = _extract_keywords_tf(
            full_text, set(), stopwords, keyword_n, idf=idf)
        kw_field = f"keywords={keywords}" if keywords else ""

        # ── Excerpt ─────────────────────────────────────────────
        excerpt_prefix = ""
        if include_excerpt:
            if query_words:
                exc = _extract_best_sentence(docs[idx], query_words, stopwords)
            else:
                exc = _extract_first_sentence(docs[idx])
            if exc:
                excerpt_prefix = f'"{exc}"  '

        # Build line
        if show_labels and labels and idx < len(labels) and labels[idx]:
            header = f"{i + 1}. {labels[idx]}"
        elif show_labels and date:
            header = f"{i + 1}. Record {i + 1} ({date})"
        else:
            header = f"{i + 1}. Record {i + 1}"
        parts = [header, ": "]
        if excerpt_prefix:
            parts.append(excerpt_prefix)
        if show_keywords:
            parts.append(kw_field)
        lines.append("".join(parts))

    return "\n".join(lines), idx_map



def select_docs(
    llm_client: OpenAIClient,
    query: str,
    synopses: str,
    idx_map: list[int],
    max_select: int = 5,
    prompt_template: str | None = None,
    max_tokens: int = 256,
) -> list[int]:
    """Select sessions to expand based on synopses.

    Calls LM with query + synopses, parses JSON list of 1-based session numbers.

    Args:
        llm_client: LLM client for the selection call.
        query: The user's question.
        synopses: Formatted synopsis lines.
        idx_map: List mapping 1-based position to actual docs index.
        max_select: Maximum number of sessions to select.
        prompt_template: Custom prompt (uses DOC_SELECTION_PROMPT if None).
        max_tokens: Max output tokens.

    Returns:
        List of actual docs indices for selected sessions.
    """
    n_candidates = len(idx_map)
    prompt = (prompt_template or DOC_SELECTION_PROMPT).format(max_select=max_select)
    user_msg = (
        f"## Question\n{query}\n\n"
        f"## Candidate Sessions (sorted by embedding similarity, [sim=...] shown)\n{synopses}\n\n"
        f"Output a JSON list of up to {max_select} session numbers "
        f"that are most relevant to the question."
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_msg},
    ]
    response = llm_client.chat(messages=messages, max_tokens=max_tokens)
    content = response.get("content", "").strip()

    # Parse JSON list
    parsed: list[int] = []
    try:
        raw = json.loads(content)
        if isinstance(raw, list):
            parsed = [int(x) for x in raw if isinstance(x, (int, float))]
    except (json.JSONDecodeError, ValueError):
        match = re.search(r"\[.*?\]", content, re.DOTALL)
        if match:
            try:
                raw = json.loads(match.group(0))
                if isinstance(raw, list):
                    parsed = [int(x) for x in raw if isinstance(x, (int, float))]
            except (json.JSONDecodeError, ValueError):
                pass

    # Validate, deduplicate, clamp
    seen: set[int] = set()
    result: list[int] = []
    for num in parsed:
        if num in seen:
            continue
        if 1 <= num <= n_candidates:
            seen.add(num)
            result.append(idx_map[num - 1])
            if len(result) >= max_select:
                break

    return result
